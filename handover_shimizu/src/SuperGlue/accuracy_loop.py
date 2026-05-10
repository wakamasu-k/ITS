import argparse
import os
from pathlib import Path
import cv2
import numpy as np
import torch

from models.matching import Matching
from models.utils import make_matching_plot_fast, frame2tensor

torch.set_grad_enabled(False)

def main():
    parser = argparse.ArgumentParser(description="Multi-folder matching (161..201) with average stats output")
    parser.add_argument('--camera_folder', type=str, required=True,
                        help="Path to camera images folder (161.jpg..201.jpg).")
    parser.add_argument('--intensity_folder', type=str, required=True,
                        help="Path to intensity base folder, containing subfolders 1..6.")
    parser.add_argument('--output_folder', type=str, required=True,
                        help="Folder where to save match results & subfolder_accuracy.txt")
    parser.add_argument('--distance_threshold', type=float, default=15.0,
                        help="Pixel distance threshold for correctness (default=15)")
    parser.add_argument('--force_cpu', action='store_true',
                        help="Force CPU usage even if GPU is available.")

    # SuperGlue パラメータ（デフォルトで keypoint_threshold=0.005, match_threshold=0.2）
    parser.add_argument('--keypoint_threshold', type=float, default=0.005,
                        help="SuperPoint keypoint detector threshold (default=0.005)")
    parser.add_argument('--match_threshold', type=float, default=0.2,
                        help="SuperGlue match threshold (default=0.2)")

    args = parser.parse_args()

    # -------------------------------------------------------------
    # 固定設定: 画像番号は 161..201 を処理
    # -------------------------------------------------------------
    start_num = 364
    end_num = 403

    # -------------------------------------------------------------
    # デバイス選択
    # -------------------------------------------------------------
    device = 'cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu'
    print(f"Using device: {device}")

    # -------------------------------------------------------------
    # SuperGlue モデルの設定
    # -------------------------------------------------------------
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': args.keypoint_threshold,
            'max_keypoints': -1,
        },
        'superglue': {
            'weights': 'outdoor',  # 必要に応じて 'indoor' へ変更
            'sinkhorn_iterations': 20,
            'match_threshold': args.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)

    camera_folder = Path(args.camera_folder)
    intensity_base = Path(args.intensity_folder)
    output_base = Path(args.output_folder)
    output_base.mkdir(parents=True, exist_ok=True)

    if not camera_folder.is_dir():
        raise NotADirectoryError(f"Camera folder not found: {camera_folder}")
    if not intensity_base.is_dir():
        raise NotADirectoryError(f"Intensity folder not found: {intensity_base}")

    # -------------------------------------------------------------
    # サブフォルダごとに "平均対応点数" "平均正しい対応数" "平均正解率" を出力
    # -------------------------------------------------------------
    accuracy_txt = output_base / "subfolder_accuracy.txt"
    with open(accuracy_txt, 'w', encoding='utf-8') as f:
        f.write("=== Subfolder Averages (Matches, Correct, Accuracy) ===\n")

    # リサイズ関数 (例: 幅800×高さ450 に固定)
    def load_and_resize_800x450(path: Path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return cv2.resize(img, (800, 450))

    # -------------------------------------------------------------
    # サブフォルダ (1..6) をループ
    # -------------------------------------------------------------
    for sub_id in range(1, 7):
        sub_folder = intensity_base / str(sub_id)
        if not sub_folder.is_dir():
            print(f"[Warning] Subfolder not found: {sub_folder}")
            continue

        out_sub = output_base / str(sub_id)
        out_sub.mkdir(parents=True, exist_ok=True)

        # 以下の 4 つを足して、最後に平均を取る
        sub_sum_matches = 0
        sub_sum_correct = 0
        sub_sum_accuracy = 0.0
        sub_count_images = 0  # 画像の枚数(0マッチ含む)

        print("------------------------------------------------------------")
        print(f"Processing subfolder: {sub_id}")

        for i in range(start_num, end_num + 1):
            cam_path = camera_folder / f"{i}.jpg"
            ref_path = sub_folder / f"Intensity_Image_{i}.png"
            if not cam_path.is_file() or not ref_path.is_file():
                continue

            img1 = load_and_resize_800x450(cam_path)
            img2 = load_and_resize_800x450(ref_path)

            t1 = frame2tensor(img1, device)
            t2 = frame2tensor(img2, device)

            pred = matching({'image0': t1, 'image1': t2})
            kpts0 = pred['keypoints0'][0].cpu().numpy()
            kpts1 = pred['keypoints1'][0].cpu().numpy()
            matches = pred['matches0'][0].cpu().numpy()

            valid = (matches > -1)
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]

            num_matches = len(mkpts0)
            # キーポイント数
            k0, k1 = len(kpts0), len(kpts1)

            # -----------------------------------------------------
            # ピクセル距離しきい値
            # -----------------------------------------------------
            dist_thresh = args.distance_threshold
            if num_matches == 0:
                # マッチ0 でも同じ形式のテキスト
                text_lines = [
                    "SuperGlue (dist-based color)",
                    f"Keypoints: {k0}:{k1}",
                    "Matches: 0",
                    f"<{dist_thresh}px: 0  (0.0%)"
                ]
                # 黒画像にテキストを表示
                black_img = np.zeros((450, 800, 3), dtype=np.uint8)
                out_vis = put_text_overlay(black_img, text_lines)
                cv2.imwrite(str(out_sub / f"matches_{i}.png"), out_vis)

                # sub_sum_matches += 0
                # sub_sum_correct += 0
                # sub_sum_accuracy += 0
                sub_count_images += 1  # "1枚" は処理した
                continue

            # 距離計算
            distances = np.sqrt(np.sum((mkpts0 - mkpts1)**2, axis=1))
            within = (distances < dist_thresh)
            correct = np.sum(within)
            accuracy_i = (correct / num_matches) * 100

            # サブフォルダ合計
            sub_sum_matches += num_matches
            sub_sum_correct += correct
            sub_sum_accuracy += accuracy_i
            sub_count_images += 1

            # 緑/赤カラー
            colors = []
            for d in distances:
                if d < dist_thresh:
                    colors.append([0.0, 1.0, 0.0, 1.0])  # green
                else:
                    colors.append([1.0, 0.0, 0.0, 1.0])  # red
            colors = np.array(colors)

            text_lines = [
                "SuperGlue (dist-based color)",
                f"Keypoints: {k0}:{k1}",
                f"Matches: {num_matches}",
                f"<{dist_thresh}px: {correct}  ({accuracy_i:.1f}%)"
            ]

            # 可視化
            out_img = make_matching_plot_fast(
                img1, img2,
                kpts0, kpts1,
                mkpts0, mkpts1,
                colors,
                text_lines,
                path=None,
                show_keypoints=False
            )
            out_file = out_sub / f"matches_{i}.png"
            cv2.imwrite(str(out_file), out_img)

        # --- サブフォルダごとの「平均」計算 ---
        # sub_sum_matches は全画像の matches の合計
        # sub_count_images は全画像枚数
        if sub_count_images == 0:
            ave_matches = 0
            ave_correct = 0
            ave_acc = 0.0
        else:
            ave_matches = sub_sum_matches / sub_count_images
            ave_correct = sub_sum_correct / sub_count_images
            ave_acc = sub_sum_accuracy / sub_count_images  # (average of each image's accuracy)

        print(f"[Subfolder {sub_id}] images={sub_count_images}, ave_matches={ave_matches:.2f}, "
              f"ave_correct={ave_correct:.2f}, ave_accuracy={ave_acc:.2f}%")

        # テキスト出力
        with open(accuracy_txt, 'a', encoding='utf-8') as f:
            f.write(f"Subfolder {sub_id}: images={sub_count_images}, "
                    f"ave_matches={ave_matches:.2f}, ave_correct={ave_correct:.2f}, ave_accuracy={ave_acc:.2f}%\n")


def put_text_overlay(base_img, lines):
    """
    シンプルなテキスト描画。
    base_img: (H,W,3) np.uint8
    lines: list[str]
    """
    vis = base_img.copy()
    x, y = 15, 40
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    color = (255, 255, 255)  # 白
    thickness = 2
    for line in lines:
        cv2.putText(vis, line, (x, y), font, font_scale, color, thickness)
        y += 30
    return vis


if __name__ == '__main__':
    main()


