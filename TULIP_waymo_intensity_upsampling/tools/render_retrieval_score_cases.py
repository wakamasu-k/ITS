"""Diagnostic contact sheets of existing images; no inference or input edits."""
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd


def read(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    return im


def tile(im, title, subtitle=''):
    w, h = 640, 426
    ratio = min(w/im.shape[1], h/im.shape[0])
    resized = cv2.resize(im, (round(im.shape[1]*ratio), round(im.shape[0]*ratio)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h+62, w, 3), dtype=np.uint8)
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    y, x = 62+(h-resized.shape[0])//2, (w-resized.shape[1])//2
    canvas[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
    cv2.putText(canvas, title, (8,24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1)
    cv2.putText(canvas, subtitle, (8,48), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
    return canvas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0, help='0: all selected cases')
    args = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    df = pd.read_csv(project/'outputs/retrieval_score_margin_audit_20261021/per_query.csv')
    selected = []
    for pair, g in df.groupby('pair_id', sort=True):
        lost = g[g.group=='B_lost'].sort_values(['margin_delta','q_idx']).head(1)
        gained = g[g.group=='B_gained'].sort_values(['margin_delta','q_idx'])
        # Typical gain within the same scene, not the largest gain.
        if len(gained):
            gained = gained.iloc[[len(gained)//2]]
        selected.extend([lost, gained])
    cases = pd.concat(selected)
    if args.limit > 0:
        cases = cases.head(args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = []
    for _, r in cases.iterrows():
        a, b = read(r.gt_image_A), read(r.gt_image_B)
        assert a.shape == b.shape
        diff = np.abs(a.astype(np.int16)-b.astype(np.int16)).astype(np.uint8)
        panels = [tile(read(r.q_cam_png), 'Query camera', f'q={r.q_idx} {r.group}'),
                  tile(a, 'GT anchor A', f'rank={r.gt_rank_A} score={r.gt_score_A:.4f}'),
                  tile(b, 'Same GT anchor B', f'rank={r.gt_rank_B} score={r.gt_score_B:.4f}'),
                  tile(diff, 'Absolute PNG difference |B-A|', 'Fixed 0..255; includes missing pixels; not geometry error'),
                  tile(read(r.top_competitor_image_A), 'Highest non-Strict-GT competitor A', f'score={r.top_competitor_score_A:.4f} distance={r.top_competitor_distance_A:.2f}m'),
                  tile(read(r.top_competitor_image_B), 'Highest non-Strict-GT competitor B', f'score={r.top_competitor_score_B:.4f} distance={r.top_competitor_distance_B:.2f}m')]
        sheet = np.concatenate([np.concatenate(panels[:3], axis=1), np.concatenate(panels[3:], axis=1)], axis=0)
        path = args.output_dir/f'{r.group}_q{int(r.q_idx):03d}.png'
        if not cv2.imwrite(str(path), sheet):
            raise OSError(path)
        outputs.append(str(path))
        print(path, flush=True)
    cases = cases.copy()
    cases['comparison_png'] = outputs
    cases.to_csv(args.output_dir/'cases.csv', index=False)


if __name__ == '__main__':
    main()
