#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperPoint + SuperGlue で
  • 「左 : camera_gray/*.jpg」, 「右 : intensity/*.png」 の全ペアを一括処理
  • 15 px 未満を TrueMatch として線色を緑 (True) / 赤 (False) で描画
  • 各ペアごとに <OUT_DIR>/match_<stem>.png を保存し，最後に総合統計を表示
"""

# ────────────────────────── 依存 ──────────────────────────
from pathlib import Path
import argparse, cv2, torch, numpy as np
from tqdm import tqdm

from models.matching import Matching
from models.utils import make_matching_plot_fast, frame2tensor
# ─────────────────────────────────────────────────────────

torch.set_grad_enabled(False)

# ────────── 1. コマンドライン引数 ──────────
parser = argparse.ArgumentParser()
parser.add_argument('--folder_png',  type=str, required=True,
                    help='intensity PNG フォルダ (右側)')
parser.add_argument('--folder_jpg',  type=str, required=True,
                    help='camera_gray JPG フォルダ (左側)')
parser.add_argument('--out_dir',     type=str, required=True,
                    help='結果保存フォルダ')
parser.add_argument('--resize', nargs='+', type=int, default=[800,450])
parser.add_argument('--distance_thr', type=float, default=15.0)
parser.add_argument('--superglue', choices={'indoor','outdoor'},
                    default='outdoor')
parser.add_argument('--device', choices={'cuda','cpu'}, default='cuda')
parser.add_argument('--no_display', action='store_true')
args = parser.parse_args()

# ────────── 2. デバイス・モデル ──────────
device = 'cuda' if args.device=='cuda' and torch.cuda.is_available() else 'cpu'
matching = Matching({
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1},
    'superglue': {
        'weights': args.superglue,
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2}
    }).eval().to(device)

# ────────── 3. 入出力フォルダ ──────────
PNG_DIR = Path(args.folder_png)
JPG_DIR = Path(args.folder_jpg)
OUT_DIR = Path(args.out_dir); OUT_DIR.mkdir(parents=True, exist_ok=True)

# ────────── 4. 画像ロード関数 ──────────
def load_gray(path, resize):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(path)
    if len(resize)==2 and resize[0]!=-1 and resize[1]!=-1:
        img = cv2.resize(img, tuple(resize))
    elif len(resize)==1 and resize[0]>0:
        h,w = img.shape[:2]; scale = resize[0]/max(h,w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
    return img

# ────────── 5. 集計変数 ──────────
total_match = total_true = 0
thr = args.distance_thr

# ────────── 6. バッチループ ──────────
for png_path in tqdm(sorted(PNG_DIR.glob("*.png")), desc="Batch"):
    stem = png_path.stem
    jpg_path = JPG_DIR / f"{stem}.jpg"
    if not jpg_path.exists():
        print(f"skip {stem}: JPG 無し"); continue

    # 6-1 読み込み (左=JPG, 右=PNG)
    imgL = load_gray(jpg_path,  args.resize)
    imgR = load_gray(png_path,  args.resize)

    # 6-2 Tensor 変換
    tL, tR = frame2tensor(imgL, device), frame2tensor(imgR, device)

    # 6-3 マッチング
    pred = matching({'image0':tL, 'image1':tR})
    kL, kR = pred['keypoints0'][0].cpu().numpy(), pred['keypoints1'][0].cpu().numpy()
    m     = pred['matches0'][0].cpu().numpy()
    vmask = m > -1
    mkL   = kL[vmask]
    mkR   = kR[m[vmask]]
    numM  = len(mkL)

    # 6-4 15px 判定
    d = np.linalg.norm(mkL-mkR, axis=1)
    ok = d < thr
    numT = int(ok.sum()); acc = numT/numM if numM else 0
    total_match += numM; total_true += numT

    # 6-5 色 (緑/赤)
    colors = np.zeros((numM,4))
    colors[:,3] = 1.0
    colors[ ok,1] = 1.0          # G
    colors[~ok,0] = 1.0          # R

    # 6-6 ラベル
    text = [
        'SuperGlue',
        f'Keypoints: {len(kL)}:{len(kR)}',
        f'Matches : {numM}',
        f'<{thr:.0f}px : {numT} ({acc*100:.1f}%)'
    ]

    # 6-7 描画・保存
    out = make_matching_plot_fast(imgL, imgR,
                                  kL, kR, mkL, mkR,
                                  colors, text,
                                  path=None, show_keypoints=False)
    cv2.imwrite(str(OUT_DIR/f"match_{stem}.png"), out)

    if not args.no_display:
        cv2.imshow('matches', out); cv2.waitKey(1)

# ────────── 7. 総合統計 ──────────
if not args.no_display: cv2.destroyAllWindows()
if total_match:
    print(f"\n=== Summary ===")
    print(f"Total Matches   : {total_match}")
    print(f"Total TrueMatch : {total_true}")
    print(f"Overall Accuracy: {total_true/total_match*100:.2f}%")
else:
    print("対応ペアが見つかりませんでした。")

