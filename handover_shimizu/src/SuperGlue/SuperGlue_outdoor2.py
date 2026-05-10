#二つのフォルダ(画像が入った)指定
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_folder', type=str, required=True)
    parser.add_argument('--reflect_folder', type=str, required=True)
    parser.add_argument('--resize', type=int, nargs='+', default=[800, 450])
    parser.add_argument('--superglue', choices={'indoor', 'outdoor'}, default='outdoor')
    parser.add_argument('--max_keypoints', type=int, default=-1)
    parser.add_argument('--keypoint_threshold', type=float, default=0.005)
    parser.add_argument('--nms_radius', type=int, default=4)
    parser.add_argument('--sinkhorn_iterations', type=int, default=20)
    parser.add_argument('--match_threshold', type=float, default=0.2)
    parser.add_argument('--show_keypoints', action='store_true')
    parser.add_argument('--no_display', action='store_true')
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu'

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

    def load_and_resize_image(image_path, resize_args):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image from {image_path}")

        if len(resize_args) == 2 and resize_args[0] != -1 and resize_args[1] != -1:
            image = cv2.resize(image, (resize_args[0], resize_args[1]))
        elif len(resize_args) == 1 and resize_args[0] > 0:
            h, w = image.shape[:2]
            scale = resize_args[0] / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        return image

    camera_folder = Path(args.camera_folder)
    reflect_folder = Path(args.reflect_folder)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None

    # --- (1) カメラ画像 & 反射強度画像の名前から番号リストを取得 ---
    #     例: 39.jpg → 39, 40.jpg → 40
    #     例: Intensity_Image_39.png → 39 など、正規表現で数値を抽出。
    #     どこから番号が始まっているかを自動判定したい。

    # 正規表現で "数字のみ" を抽出する例
    # ファイル名末尾を想定した簡易な書き方です。
    pattern_cam = re.compile(r'^(\d+)\.jpg$')  # 例: "39.jpg" → group(1)="39"
    pattern_ref = re.compile(r'^Intensity_Image_(\d+)\.png$')  # 例: "Intensity_Image_39.png"

    # カメラ画像フォルダにある jpg を列挙して番号を取得
    camera_indices = []
    for p in camera_folder.glob('*.jpg'):
        m = pattern_cam.match(p.name)
        if m:
            num = int(m.group(1))
            camera_indices.append(num)

    # 反射画像フォルダにある png を列挙して番号を取得
    reflect_indices = []
    for p in reflect_folder.glob('*.png'):
        m = pattern_ref.match(p.name)
        if m:
            num = int(m.group(1))
            reflect_indices.append(num)

    # マッチング対象は「両方に存在する番号」
    common_indices = sorted(set(camera_indices).intersection(set(reflect_indices)))

    if not common_indices:
        print("マッチングできる (番号.jpg) と (Intensity_Image_番号.png) の組み合わせが見つかりません。")
        return

    # --- (2) 対象となる番号ごとにマッチング ---
    for i in common_indices:
        cam_path = camera_folder / f"{i}.jpg"
        ref_path = reflect_folder / f"Intensity_Image_{i}.png"

        image1 = load_and_resize_image(cam_path, args.resize)
        image2 = load_and_resize_image(ref_path, args.resize)

        tensor1 = frame2tensor(image1, device)
        tensor2 = frame2tensor(image2, device)

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
            f'Keypoints: {len(kpts0)}:{len(kpts1)}',
            f'Matches: {len(mkpts0)}'
        ]

        out = make_matching_plot_fast(
            image1, image2, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=args.show_keypoints)

        if not args.no_display:
            cv2.imshow('SuperGlue matches', out)
            cv2.waitKey(500)

        if output_dir is not None:
            out_file = output_dir / f"matches_{i}.png"
            cv2.imwrite(str(out_file), out)
            print(f"Saved match result: {out_file}")

    if not args.no_display:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

