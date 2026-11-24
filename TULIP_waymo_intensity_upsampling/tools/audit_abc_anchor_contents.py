#!/usr/bin/env python3
"""Validate all DB image/index/depth contents and compare identical A/B/C rows.

Depth is camera forward-axis depth, not radial LiDAR range. PNG error is
postprocessed 8-bit error, not raw reflectance error. Inputs are read-only.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def compare(image, depth, mask, ref_image, ref_depth, ref_mask):
    common, union = mask & ref_mask, mask | ref_mask
    d = np.abs(depth[common].astype(np.float64) - ref_depth[common])
    delta = np.abs(image.astype(np.float32) - ref_image.astype(np.float32))
    close = np.zeros_like(common)
    close[common] = d <= 1.0
    return dict(valid_pixels=int(mask.sum()), reference_valid_pixels=int(ref_mask.sum()),
                common_pixels=int(common.sum()), union_pixels=int(union.sum()),
                coverage=float(mask.mean()), reference_coverage=float(ref_mask.mean()),
                geometry_iou=float(common.sum() / union.sum()) if union.any() else None,
                depth_mae_m=float(d.mean()) if d.size else None,
                depth_median_m=float(np.median(d)) if d.size else None,
                depth_p95_m=float(np.percentile(d, 95)) if d.size else None,
                depth_over_1m_fraction=float((d > 1).mean()) if d.size else None,
                png_mae_common=float(delta[common].mean()) if common.any() else None,
                png_mae_union=float(delta[union].mean()) if union.any() else None,
                depth_within_1m_pixels=int(close.sum()),
                png_mae_depth_within_1m=float(delta[close].mean()) if close.any() else None)


def load(row):
    folder = Path(row['anchor_png']).parent
    meta = json.loads((folder / 'anchor.json').read_text())
    image = cv2.imread(str(folder / 'anchor.png'), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError('PNG must be readable grayscale uint8')
    with np.load(folder / 'anchor_index.npz', allow_pickle=False) as archive:
        idx, depth = archive['point_index'], archive['depth']
    if image.shape != idx.shape or image.shape != depth.shape:
        raise ValueError('PNG/index/depth shape mismatch')
    if image.shape != (meta['height'], meta['width']):
        raise ValueError('PNG shape differs from metadata')
    if not np.issubdtype(idx.dtype, np.integer) or not np.issubdtype(depth.dtype, np.floating):
        raise ValueError('Unexpected index/depth dtype')
    covered = idx >= 0
    valid = covered & np.isfinite(depth) & (depth > 0)
    stats = dict(covered_pixels=int(covered.sum()), invalid_covered_depth=int((covered & ~valid).sum()),
                 invalid_negative_index=int((idx < -1).sum()),
                 uncovered_nonzero_png=int(((~covered) & (image != 0)).sum()),
                 covered_zero_png=int((covered & (image == 0)).sum()),
                 coverage_metadata_error=abs(float(covered.mean()) - float(meta['coverage'])))
    return image, depth, valid, stats


def aggregate(rows):
    metrics = ('coverage', 'reference_coverage', 'geometry_iou', 'depth_mae_m',
               'depth_median_m', 'depth_p95_m', 'depth_over_1m_fraction',
               'png_mae_common', 'png_mae_union', 'png_mae_depth_within_1m')
    out = dict(anchor_count=len(rows))
    for k in metrics:
        values = [r[k] for r in rows if r[k] is not None]
        out['mean_anchor_' + k] = float(np.mean(values)) if values else None
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0, help='0 means all rows')
    args = p.parse_args()
    tables = {}
    for v in 'ABC':
        with (args.manifest_dir / f'{v}_db.csv').open(newline='') as f:
            tables[v] = list(csv.DictReader(f))
    if len({len(t) for t in tables.values()}) != 1:
        raise ValueError('Manifest lengths differ')
    for v in 'AB':
        for a, c in zip(tables[v], tables['C']):
            if {k: val for k, val in a.items() if k != 'anchor_png'} != {k: val for k, val in c.items() if k != 'anchor_png'}:
                raise ValueError('Manifest alignment mismatch')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, checks, errors = [], [], []
    count = min(args.limit, len(tables['C'])) if args.limit > 0 else len(tables['C'])
    for i in range(count):
        loaded = {}
        identity = dict(row=i, pair_id=tables['C'][i]['pair_id'], anchor_id=tables['C'][i]['anchor_id'])
        for v in 'ABC':
            try:
                loaded[v] = load(tables[v][i])
                stats = loaded[v][3]
                checks.append(dict(**identity, variant=v, **stats))
                if stats['invalid_covered_depth'] or stats['invalid_negative_index'] or stats['coverage_metadata_error'] > 1e-8 or stats['uncovered_nonzero_png']:
                    errors.append(dict(**identity, variant=v, error='Content consistency issue', stats=stats))
            except Exception as exc:
                errors.append(dict(**identity, variant=v, error=str(exc)))
        if 'C' in loaded:
            for v in 'AB':
                if v in loaded:
                    rows.append(dict(**identity, variant=v, **compare(*loaded[v][:3], *loaded['C'][:3])))
        if (i + 1) % 25 == 0:
            print(f'Checked {i + 1}/{count}; issues={len(errors)}', flush=True)
    for name, content in [('per_anchor', rows), ('content_checks', checks)]:
        if content:
            with (args.output_dir / (name + '.csv')).open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(content[0]))
                w.writeheader()
                w.writerows(content)
    summary = dict(requested_anchors=count, loaded_filesets=len(checks), issue_count=len(errors), issues=errors,
                   comparisons={v: aggregate([r for r in rows if r['variant'] == v]) for v in 'AB'},
                   by_pair={pair: {v: aggregate([r for r in rows if r['variant'] == v and r['pair_id'] == pair]) for v in 'AB'} for pair in sorted({r['pair_id'] for r in rows})},
                   notes=['Arithmetic mean of per-anchor metrics, not pooled pixels.',
                          'Mask = point_index >= 0 and finite positive depth; zero intensity is valid.',
                          'Same-pixel forward-depth differences; not 3D point correspondences.',
                          'PNG MAE is after normalization/CLAHE; not raw intensity error.',
                          'Point index bounds against source map and raw intensity not checked.'])
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({k: summary[k] for k in ('requested_anchors', 'loaded_filesets', 'issue_count', 'comparisons')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
