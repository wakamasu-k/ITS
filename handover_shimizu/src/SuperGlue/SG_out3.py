#ポイントサイズ1-6ループ
from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import re

from models.matching import Matching
from models.utils import make_matching_plot_fast, frame2tensor

torch.set_grad_enabled(False)

def main():
    parser = argparse.ArgumentParser(description="Multi-folder matching demo")

    # カメラ画像フォルダ (例: C:\Users\divin\Downloads\match\79_119\camera)
    parser.add_argument('--camera_folder', type=str, required=True,
                        help='Path to the folder containing camera images (e.g. 79.jpg, 80.jpg...)')

    # 反射強度画像の親フォルダ (例: C:\Users\divin\Downloads\match\79_119\intensity)
    # この下に 1,2,3,4,5,6 のサブフォルダがあり、それぞれ "Intensity_Image_{番号}.png" が入っている想定
    parser.add_argument('--intensity_base', type=str, required=True,
                        help='Path to the base folder of intensity subfolders (1..6)')

    # 結果を保存するフォルダ (例: C:\Users\divin\Downloads\match\79_119\match)
    # ここに 1,2,3,4,5,6 のサブフォルダを作り、その中に "matches_{番号}.png" を保存
    parser.add_argument('--match_base', type=str, required=True,
                        help='Path to the base folder where match results will be saved (1..6)')

    parser.add_argument('--resize', type=int, nargs='+', default=[800, 450],
                        help='Resize the input image before running inference. '
                             'If two numbers, resize to the exact dimensions, '
                             'if one number, resize the max dimension, if -1, do not resize')

    parser.add_argument('--superglue', choices={'indoor', 'outdoor'}, default='outdoor')
    parser.add_argument('--max_keypoints', type=int, default=-1)
    parser.add_argument('--keypoint_threshold', type=float, default=0.005)
    parser.add_argument('--nms_radius', type=int, default=4)
    parser.add_argument('--sinkhorn_iterations', type=int, default=20)
    parser.add_argument('--match_threshold', type=float, default=0.2)
    parser.add_argument('--show_keypoints', action='store_true')
    parser.add_argument('--no_display', action='store_true')
    parser.add_argument('--force_cpu', action='store_true')

    args = parser.parse_args()

    # -------------------------------------------------------------
    # デバイス設定 (GPU / CPU)
    # -------------------------------------------------------------
    device = 'cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu'

    # -------------------------------------------------------------
    # SuperGlue モデル設定
    # -------------------------------------------------------------
    config = {
        'superpoint': {
            'nms_radius': args.nms_radius,
            'keypoint_threshold': args.keypoint_threshold,
            'max_keypoints': args.max_keypoints
        },
        'superglue': {
            'weights': args.superglue,
            'sinkhorn_iterations': args.sinkhorn_iterations,
            'match_threshold': args.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)

    # -------------------------------------------------------------
    # 画像読み込み＆リサイズ用の関数
    # -------------------------------------------------------------
    def load_and_resize_image(image_path, resize_args):
        """
        画像を読み込み、指定されたリサイズ設定に従ってリサイズして返す。
        グレースケールで読み込む。
        """
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image from {image_path}")

        # リサイズ処理
        if len(resize_args) == 2 and resize_args[0] != -1 and resize_args[1] != -1:
            image = cv2.resize(image, (resize_args[0], resize_args[1]))
        elif len(resize_args) == 1 and resize_args[0] > 0:
            h, w = image.shape[:2]
            scale = resize_args[0] / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        # resize_args が [-1] の場合はリサイズせずそのまま
        return image

    # -------------------------------------------------------------
    # フォルダパスの取得
    # -------------------------------------------------------------
    camera_folder = Path(args.camera_folder)
    intensity_base = Path(args.intensity_base)
    match_base = Path(args.match_base)

    # -------------------------------------------------------------
    # フォルダの存在確認 & 結果フォルダ作成
    # -------------------------------------------------------------
    if not camera_folder.is_dir():
        print(f"Camera folder not found: {camera_folder}")
        return
    if not intensity_base.is_dir():
        print(f"Intensity base folder not found: {intensity_base}")
        return

    match_base.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # カメラ画像のファイル名から番号を抽出 (例: 79.jpg → 79)
    # -------------------------------------------------------------
    pattern_cam = re.compile(r'^(\d+)\.jpg$')
    camera_indices = []
    for p in camera_folder.glob('*.jpg'):
        m = pattern_cam.match(p.name)
        if m:
            num = int(m.group(1))
            camera_indices.append(num)
    camera_indices = sorted(camera_indices)

    if not camera_indices:
        print("No .jpg files found in camera folder.")
        return

    # -------------------------------------------------------------
    # 1～6 のサブフォルダを順次処理
    # -------------------------------------------------------------
    for sub_id in range(1, 7):
        # 反射強度画像のサブフォルダ例: intensity_base / "1"
        reflect_folder = intensity_base / str(sub_id)
        if not reflect_folder.is_dir():
            print(f"Reflect folder not found: {reflect_folder}")
            continue  # 存在しない場合はスキップ

        # 結果保存先フォルダ例: match_base / "1"
        output_dir = match_base / str(sub_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------
        # 反射強度画像のファイル名から番号を抽出 (例: Intensity_Image_79.png → 79)
        # ---------------------------------------------------------
        pattern_ref = re.compile(r'^Intensity_Image_(\d+)\.png$')
        reflect_indices = []
        for rp in reflect_folder.glob('*.png'):
            m = pattern_ref.match(rp.name)
            if m:
                rnum = int(m.group(1))
                reflect_indices.append(rnum)
        reflect_indices = sorted(reflect_indices)

        # ---------------------------------------------------------
        # マッチング可能な番号 (カメラ画像と反射強度画像の共通番号) を取得
        # ---------------------------------------------------------
        common_indices = sorted(set(camera_indices).intersection(reflect_indices))
        if not common_indices:
            print(f"No matching indices found for subfolder: {sub_id}")
            continue

        print("-" * 50)
        print(f"Matching folder '{sub_id}' -> Found {len(common_indices)} pairs to process.")

        # ---------------------------------------------------------
        # 同じ番号 i を持つファイル同士でマッチング実行
        # ---------------------------------------------------------
        for i in common_indices:
            # カメラ画像 (例: camera/79.jpg)
            cam_path = camera_folder / f"{i}.jpg"

            # 反射強度画像 (例: intensity/1/Intensity_Image_79.png)
            ref_path = reflect_folder / f"Intensity_Image_{i}.png"

            # 画像を読み込み & リサイズ
            image1 = load_and_resize_image(cam_path, args.resize)
            image2 = load_and_resize_image(ref_path, args.resize)

            # Tensor に変換
            tensor1 = frame2tensor(image1, device)
            tensor2 = frame2tensor(image2, device)

            # SuperGlue でマッチング
            pred = matching({'image0': tensor1, 'image1': tensor2})
            kpts0 = pred['keypoints0'][0].cpu().numpy()
            kpts1 = pred['keypoints1'][0].cpu().numpy()
            matches = pred['matches0'][0].cpu().numpy()
            confidence = pred['matching_scores0'][0].cpu().numpy()

            # マッチが有効な部分のみ抽出
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            color = cm.jet(confidence[valid])

            text = [
                f'SuperGlue subfolder={sub_id}',
                f'Keypoints: {len(kpts0)}:{len(kpts1)}',
                f'Matches: {len(mkpts0)}'
            ]

            # マッチング結果の可視化
            out = make_matching_plot_fast(
                image1, image2, kpts0, kpts1, mkpts0, mkpts1, color, text,
                path=None, show_keypoints=args.show_keypoints)

            # 表示オプション
            if not args.no_display:
                cv2.imshow('SuperGlue matches', out)
                cv2.waitKey(500)  # 0にすればキー待ちで停止

            # 結果画像を保存 (例: match_base/1/matches_79.png)
            out_file = output_dir / f"matches_{i}.png"
            cv2.imwrite(str(out_file), out)
            print(f"[subfolder {sub_id}] Saved match result: {out_file}")

        # subfolder終了時にウィンドウを閉じる (no_display でなければ)
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
