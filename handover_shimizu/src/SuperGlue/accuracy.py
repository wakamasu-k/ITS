from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import numpy as np

from models.matching import Matching
from models.utils import make_matching_plot_fast, frame2tensor

torch.set_grad_enabled(False)

def main():
    # -------------------------------------------------------------
    # コマンドライン引数の設定
    # -------------------------------------------------------------
    parser = argparse.ArgumentParser(description='SuperGlue demo with distance-based evaluation')
    parser.add_argument('--image1', type=str, required=True, help='Path to the first image file')
    parser.add_argument('--image2', type=str, required=True, help='Path to the second image file')
    parser.add_argument('--resize', type=int, nargs='+', default=[800, 450],
                        help='Resize the input image before running inference. If two numbers, '
                             'resize to the exact dimensions, if one number, resize the max '
                             'dimension, if -1, do not resize')
    parser.add_argument('--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
                        help='SuperGlue weights')
    parser.add_argument('--max_keypoints', type=int, default=-1,
                        help='Maximum number of keypoints detected by Superpoint (-1 keeps all keypoints)')
    parser.add_argument('--keypoint_threshold', type=float, default=0.005,
                        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument('--nms_radius', type=int, default=4,
                        help='SuperPoint NMS radius (Must be positive)')
    parser.add_argument('--sinkhorn_iterations', type=int, default=20,
                        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument('--match_threshold', type=float, default=0.2,
                        help='SuperGlue match threshold')
    parser.add_argument('--distance_threshold', type=float, default=15.0,
                        help='Pixel distance threshold to judge correctness')
    parser.add_argument('--show_keypoints', action='store_true',
                        help='Show the detected keypoints')
    parser.add_argument('--no_display', action='store_true',
                        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force pytorch to run in CPU mode.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory where to write output frames')
    args = parser.parse_args()

    # -------------------------------------------------------------
    # デバイスの設定
    # -------------------------------------------------------------
    device = 'cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu'

    # -------------------------------------------------------------
    # SuperGlue モデルの設定
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
    # 画像の読み込みとリサイズ用関数
    # -------------------------------------------------------------
    def load_and_resize_image(image_path, resize_args):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image from {image_path}")

        if len(resize_args) == 2:
            if resize_args[0] != -1 and resize_args[1] != -1:
                image = cv2.resize(image, (resize_args[0], resize_args[1]))
        elif len(resize_args) == 1 and resize_args[0] > 0:
            h, w = image.shape[:2]
            scale = resize_args[0] / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        return image

    # 画像1・画像2の読み込み
    image1 = load_and_resize_image(args.image1, args.resize)
    image2 = load_and_resize_image(args.image2, args.resize)

    # テンソルへの変換
    tensor1 = frame2tensor(image1, device)
    tensor2 = frame2tensor(image2, device)

    # -------------------------------------------------------------
    # マッチングの実行
    # -------------------------------------------------------------
    pred = matching({'image0': tensor1, 'image1': tensor2})
    kpts0 = pred['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()  # -1: no match
    confidence = pred['matching_scores0'][0].cpu().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]             # image1 の対応点
    mkpts1 = kpts1[matches[valid]]    # image2 の対応点
    conf = confidence[valid]          # confidence (match score)

    num_matches = len(mkpts0)

    # -------------------------------------------------------------
    # ここから距離を計算して「しきい値」以内かどうかを判定
    # -------------------------------------------------------------
    threshold = args.distance_threshold

    # 対応点ごとのピクセル距離を計算
    distances = np.sqrt(np.sum((mkpts0 - mkpts1) ** 2, axis=1))

    # しきい値を下回るかどうかの真偽配列
    within_thresh = distances < threshold
    num_correct = np.sum(within_thresh)
    accuracy = (num_correct / num_matches) if num_matches > 0 else 0.0

    # -------------------------------------------------------------
    # 描画用の色を作成: しきい値以内を緑, しきい値超過を赤
    # -------------------------------------------------------------
    # shape=(num_matches, 4) → RGBA の想定
    colors = []
    for d in distances:
        if d < threshold:
            # 緑 (R=0,G=1,B=0)
            colors.append([0.0, 1.0, 0.0, 1.0])
        else:
            # 赤 (R=1,G=0,B=0)
            colors.append([1.0, 0.0, 0.0, 1.0])
    colors = np.array(colors)

    # ラベルテキスト (画面左上に表示する簡易情報)
    text = [
        'SuperGlue',
        f'Keypoints: {len(kpts0)}:{len(kpts1)}',
        f'Matches: {num_matches}',
        f'<{threshold}px: {num_correct}  ({accuracy*100:.1f}%)'
    ]

    # -------------------------------------------------------------
    # 可視化: make_matching_plot_fast() でラインを描画
    # -------------------------------------------------------------
    out = make_matching_plot_fast(
        image1, image2,
        kpts0, kpts1,
        mkpts0, mkpts1,
        colors,            # confidence の代わりに作成した colors を指定
        text,
        path=None,
        show_keypoints=args.show_keypoints
    )

    # -------------------------------------------------------------
    # 結果の表示
    # -------------------------------------------------------------
    print(f"Total Matches: {num_matches}")
    print(f"Distance < {threshold}: {num_correct} → Accuracy: {accuracy*100:.1f}%")

    if not args.no_display:
        cv2.imshow('SuperGlue matches', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # -------------------------------------------------------------
    # 結果の保存
    # -------------------------------------------------------------
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / 'matches.png'
        cv2.imwrite(str(out_file), out)
        print(f"Saved match result: {out_file}")

if __name__ == '__main__':
    main()

"""
python accuracy.py --image1 "C:\Users\shimizu.k\Downloads\camera\0001.jpg" --image2 "C:\Users\shimizu.k\Downloads\intensity\0001.png" --no_display --distance_threshold 15 --output_dir "C:\Users\shimizu.k\Downloads\sg"
"""