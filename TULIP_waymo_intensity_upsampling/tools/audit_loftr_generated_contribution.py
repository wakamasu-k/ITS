"""Attribute original-B matches and rerun the inherited PnP on source subsets.

No training, image changes or GT-based candidate selection. Python 3.8 compatible.
The subset experiment retains features inferred from the complete B image.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
REF = PROJECT.parent / 'handover_shimizu/src/LoFTR'
BASE = PROJECT / 'outputs/shimizu_protocol_507q_754db_abc'


def load_evaluator():
    sys.path.insert(0, str(REF / 'src'))
    path = REF / 'tools/eval_cross_match_to_pose_zalign_multi_visall_v2.py'
    spec = importlib.util.spec_from_file_location('contribution_inherited_eval', str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def classify(mk0, index, point_index, generated, a_valid):
    h, w = index.depth_x.shape
    u = np.clip(np.round(mk0[:, 0]).astype(np.int64), 0, w - 1)
    v = np.clip(np.round(mk0[:, 1]).astype(np.int64), 0, h - 1)
    depth = index.depth_x[v, u]
    valid = index.valid_mask[v, u] & np.isfinite(depth) & (depth >= .5) & (depth <= 200)
    ids = point_index[v, u]
    if np.any((ids[valid] < 0) | (ids[valid] >= len(generated))):
        raise ValueError('Point index does not match B provenance map')
    labels = np.zeros(len(mk0), dtype=np.uint8)
    labels[valid] = 1  # measured
    gen = np.zeros(len(mk0), dtype=bool)
    gen[valid] = generated[ids[valid]]
    labels[gen & a_valid[v, u]] = 2  # generated replacing A coverage
    labels[gen & ~a_valid[v, u]] = 3  # generated filling A holes
    return labels, valid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=PROJECT / 'outputs/B_dense_distance_protocol_20261019/B_strict_507.csv')
    p.add_argument('--checkpoint', type=Path, default=PROJECT.parent / 'waymo_outputs/old32_dense/loftr_retrain_stage1_paper_wdrive/stage1_final_gstep715958.pth')
    p.add_argument('--icp', type=Path, default=Path('/mnt/w/32line/datasets/splits/cross/all/icp_startend_staticmaps.csv'))
    p.add_argument('--scope', choices=['pilot10', 'all507'], default='pilot10')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--pnp-refine', action='store_true')
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this Python environment; use .venv_tulip/bin/python')
    ev = load_evaluator()
    df = pd.read_csv(args.manifest)
    if len(df) != 507 or df.duplicated(['pair_id', 'q_frame_index', 'anchor_id']).any():
        raise ValueError('Expected 507 unique Strict pairs')
    if args.scope == 'pilot10':
        df = pd.concat([g.iloc[sorted(set([0, len(g)//2]))] for _, g in df.groupby('pair_id', sort=True)])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'matches').mkdir()
    (args.output_dir / 'protocol.json').write_text(json.dumps(dict(
        manifest=str(args.manifest), checkpoint=str(args.checkpoint), icp=str(args.icp),
        scope=args.scope, seed=args.seed, pnp_refine=args.pnp_refine,
        labels={'0': 'invalid_depth', '1': 'measured', '2': 'generated_on_A_covered', '3': 'generated_on_A_missing'},
        note='Fixed Strict anchors. Subset PnP uses features from full B. No z alignment. Success rates use all evaluated pairs.'
    ), indent=2))
    matcher = ev.load_matcher(str(args.checkpoint), device=torch.device('cuda'), enable_fine=True, enable_repeatability=False)
    icp = ev.load_icp_db_to_q(str(args.icp), db_col=None, q_col=None, mat_col=None, mat_cols=None)
    records, sources = [], {}
    for number, (_, row) in enumerate(df.iterrows(), 1):
        seg, aid = str(row.db_seg), str(row.anchor_id)
        if seg not in sources:
            with np.load(BASE / 'maps/B_tulip32/training' / seg / 'map_static.npz') as z:
                sources[seg] = z['source_is_generated'].astype(bool)
        anchor = json.loads(Path(row.anchor_json).read_text())
        query = json.loads(Path(row.q_meta_json).read_text())
        a_path = BASE / 'anchors_A/training' / seg / aid / 'anchor_index.npz'
        a_meta = json.loads(a_path.with_name('anchor.json').read_text())
        if not np.allclose(ev._parse_Twc(anchor), ev._parse_Twc(a_meta)) or not np.allclose(ev._parse_intrinsic(anchor), ev._parse_intrinsic(a_meta)):
            raise ValueError('A/B anchor pose or intrinsics differ')
        with np.load(a_path) as z:
            a_valid = z['point_index'] >= 0
        with np.load(row.anchor_index_npz) as z:
            point_index = z['point_index']
        index = ev.load_anchor_index_npz(row.anchor_index_npz)
        if a_valid.shape != index.depth_x.shape:
            raise ValueError('A/B raster dimensions differ')
        matches = ev.infer_matches(matcher, ev._read_gray_u8(row.anchor_png), ev._read_gray_u8(row.q_image_png),
            resize_long=840, stride=8, cell_point='center', device=torch.device('cuda'),
            amp_enabled=True, amp_dtype=torch.float16, match_source='auto', conf_th=0., max_matches=4096)
        mk0, mk1 = matches['mkpts0'], matches['mkpts1']
        labels, valid = classify(mk0, index, point_index, sources[seg], a_valid)
        transform = icp.get(seg, str(row.q_seg))
        if transform is None:
            raise ValueError('No ICP transform; cannot compare pose errors')
        saved = dict(mkpts0=mk0, mkpts1=mk1, source_label=labels, valid_3d=valid)
        for condition, mask in [('all', np.ones(len(labels), bool)), ('measured_only', labels == 1), ('generated_only', labels >= 2)]:
            selected = np.flatnonzero(mask)
            valid_selected = selected[valid[selected]]
            captured = {}
            original = ev.cv2.solvePnPRansac
            def capture(*aa, **kw):
                result = original(*aa, **kw)
                captured['indices'] = result[3]
                return result
            ev.cv2.setRNGSeed(args.seed)
            ev.cv2.solvePnPRansac = capture
            try:
                pose = ev.estimate_pose_pnp_xforward(mk0[mask], mk1[mask], index,
                    ev._parse_Twc(anchor), ev._parse_intrinsic(anchor), ev._parse_intrinsic(query), transform,
                    min_depth=.5, max_depth=200., ransac_reproj_th=4., ransac_max_iter=2000,
                    ransac_conf=.999, refine=args.pnp_refine)
            finally:
                ev.cv2.solvePnPRansac = original
            inliers = np.empty(0, dtype=np.int64)
            if pose.success:
                inliers = valid_selected[np.asarray(captured['indices']).reshape(-1)]
                assert len(inliers) == pose.n_inlier
            saved[condition + '_inlier_indices'] = inliers
            xy = float(np.linalg.norm((pose.C_pred - ev._camera_center_world(ev._parse_Twc(query)))[:2])) if pose.success else np.nan
            records.append(dict(pair_id=row.pair_id, q_frame_index=row.q_frame_index, anchor_id=row.anchor_id,
                condition=condition, pnp_success=pose.success, xy_error_m=xy,
                matches=pose.n_match_in, correspondences_3d=pose.n_3d, inliers=pose.n_inlier,
                measured_inliers=int(np.sum(labels[inliers] == 1)),
                generated_replacement_inliers=int(np.sum(labels[inliers] == 2)),
                generated_new_coverage_inliers=int(np.sum(labels[inliers] == 3)),
                measured_matches=int(np.sum(labels == 1)), generated_replacement_matches=int(np.sum(labels == 2)),
                generated_new_coverage_matches=int(np.sum(labels == 3))))
        np.savez_compressed(args.output_dir / 'matches' / ('%03d.npz' % number), **saved)
        pd.DataFrame(records).to_csv(args.output_dir / 'per_pair.csv', index=False)
        print('[%d/%d] %s frame=%s: all XY=%s' % (number, len(df), row.pair_id, row.q_frame_index, records[-3]['xy_error_m']), flush=True)
    result = pd.DataFrame(records)
    summary = result.groupby('condition').agg(pairs=('pnp_success', 'size'), pnp_success_rate=('pnp_success', 'mean'),
        xy_median_success_only=('xy_error_m', 'median'), matches_median=('matches', 'median'),
        inliers_median=('inliers', 'median'))
    for threshold in [1, 2, 5]:
        summary['success_within_%dm_all_pairs' % threshold] = result.assign(hit=result.xy_error_m.le(threshold)).groupby('condition').hit.mean()
    summary.to_csv(args.output_dir / 'summary.csv')
    print(summary.to_string())
    print('Saved:', args.output_dir)


if __name__ == '__main__':
    main()
