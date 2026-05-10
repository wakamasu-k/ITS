#画像ファイルを2つ指定
from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch

from models.matching import Matching
from models.utils import make_matching_plot_fast, frame2tensor

torch.set_grad_enabled(False)

def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='SuperGlue demo')
    parser.add_argument('--image1', type=str, required=True, help='Path to the first image file')
    parser.add_argument('--image2', type=str, required=True, help='Path to the second image file')
    parser.add_argument('--resize', type=int, nargs='+', default=[800, 450],
                        help='Resize the input image before running inference. If two numbers, '
                             'resize to the exact dimensions, if one number, resize the max '
                             'dimension, if -1, do not resize')
    parser.add_argument('--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
                        help='SuperGlue weights')
    parser.add_argument('--max_keypoints', type=int, default=-1,
                        help='Maximum number of keypoints detected by Superpoint (\'-1\' keeps all keypoints)')
    parser.add_argument('--keypoint_threshold', type=float, default=0.005, # def0.005
                        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument('--nms_radius', type=int, default=4,
                        help='SuperPoint Non Maximum Suppression (NMS) radius (Must be positive)')
    parser.add_argument('--sinkhorn_iterations', type=int, default=20,
                        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument('--match_threshold', type=float, default=0.2, # def0.2
                        help='SuperGlue match threshold')
    parser.add_argument('--show_keypoints', action='store_true',
                        help='Show the detected keypoints')
    parser.add_argument('--no_display', action='store_true',
                        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force pytorch to run in CPU mode.')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory where to write output frames')


    

    args = parser.parse_args()

    # デバイスの設定
    device = 'cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu'

    # モデルの設定
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

    # 画像の読み込みとリサイズ
    def load_and_resize_image(image_path, resize_args):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if len(resize_args) == 2:
            image = cv2.resize(image, (resize_args[0], resize_args[1]))
        elif len(resize_args) == 1 and resize_args[0] > 0:
            h, w = image.shape[:2]
            scale = resize_args[0] / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        return image

    image1 = load_and_resize_image(args.image1, args.resize)
    image2 = load_and_resize_image(args.image2, args.resize)

    # テンソルへの変換
    tensor1 = frame2tensor(image1, device)
    tensor2 = frame2tensor(image2, device)

    # マッチングの実行
    pred = matching({'image0': tensor1, 'image1': tensor2})
    kpts0 = pred['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    color = cm.jet(confidence[valid])
    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0))
    ]

    out = make_matching_plot_fast(
        image1, image2, kpts0, kpts1, mkpts0, mkpts1, color, text,
        path=None, show_keypoints=args.show_keypoints)

    # 結果の表示
    if not args.no_display:
        cv2.imshow('SuperGlue matches', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if args.output_dir is not None:
        # 出力ディレクトリが存在しない場合は作成
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存するファイル名を設定（例：'matches.png'）
        out_file = output_dir / 'matches.png'
        cv2.imwrite(str(out_file), out)

if __name__ == '__main__':
    main()

