"""Same-pixel depth audit, not pointwise reconstruction ground truth."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = project/'outputs/shimizu_protocol_507q_754db_abc'
    folders = {}
    for v in 'ABC':
        table = pd.read_csv(root/f'retrieval_manifests/{v}_db.csv', dtype={'anchor_id': str})
        folders[v] = {(r.db_seg, r.anchor_id): Path(r.anchor_png).parent for r in table.itertuples()}
    cases = pd.read_csv(project/'outputs/gt_vs_anchor102_audit_ax6mcoi2/anchor_comparison.csv', dtype={'anchor': str})
    if args.limit:
        cases = cases.head(args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for segment, group in cases.groupby('segment'):
        with np.load(root/'maps/B_tulip32/training'/segment/'map_static.npz', allow_pickle=False) as z:
            generated = z['source_is_generated'].astype(bool)
        for _, case in group.iterrows():
            dep, valid, idx, meta = {}, {}, {}, {}
            for v in 'ABC':
                folder = folders[v][segment, case.anchor]
                meta[v] = json.loads((folder/'anchor.json').read_text())
                with np.load(folder/'anchor_index.npz', allow_pickle=False) as z:
                    dep[v] = z['depth'].astype(np.float64)
                    idx[v] = z['point_index']
                valid[v] = (idx[v]>=0)&np.isfinite(dep[v])&(dep[v]>0)
            assert dep['A'].shape == dep['B'].shape == dep['C'].shape
            for v in 'AB':
                for field in ('intrinsic','world_to_cam'):
                    assert np.allclose(meta[v][field], meta['C'][field], atol=1e-8, rtol=0), field
            covered = idx['B']>=0
            assert idx['B'][covered].max() < len(generated)
            gen = np.zeros(covered.shape, dtype=bool)
            gen[covered] = generated[idx['B'][covered]]
            masks = {'B_generated': valid['B']&gen,
                     'B_measured': valid['B']&~gen,
                     'generated_on_A_covered': valid['B']&gen&valid['A'],
                     'generated_on_A_missing': valid['B']&gen&~valid['A']}
            height = covered.shape[0]
            y = np.arange(height)[:,None]
            regions = {'all': np.ones_like(covered),
                       'upper_half': np.broadcast_to(y<height//2, covered.shape),
                       'lower_half': np.broadcast_to(y>=height//2, covered.shape)}
            for name, base in masks.items():
                for region, area in regions.items():
                    source = base&area
                    common = source&valid['C']
                    matched = common&valid['A']
                    errors = np.abs(dep['B'][common]-dep['C'][common])
                    ea = np.abs(dep['A'][matched]-dep['C'][matched])
                    eb = np.abs(dep['B'][matched]-dep['C'][matched])
                    rows.append(dict(role=case.role, segment=segment, anchor=case.anchor,
                        mask=name, region=region, source_pixels=int(source.sum()),
                        comparable_pixels=int(common.sum()),
                        comparable_fraction=float(common.sum()/source.sum()) if source.any() else None,
                        B_C_depth_mae=float(errors.mean()) if errors.size else None,
                        B_C_depth_p95=float(np.percentile(errors,95)) if errors.size else None,
                        B_C_over2m_fraction=float((errors>2).mean()) if errors.size else None,
                        ABC_common_pixels=int(matched.sum()),
                        A_C_mae_matched=float(ea.mean()) if ea.size else None,
                        B_C_mae_matched=float(eb.mean()) if eb.size else None,
                        delta_BC_minus_AC=float((eb-ea).mean()) if ea.size else None))
            print(f'Checked {case.role} anchor={case.anchor}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir/'per_anchor_region.csv', index=False)
    cols = ['comparable_fraction','B_C_depth_mae','B_C_depth_p95','B_C_over2m_fraction',
            'A_C_mae_matched','B_C_mae_matched','delta_BC_minus_AC']
    summary = df[df.region=='all'].groupby(['role','mask'])[cols].median()
    summary.to_csv(args.output_dir/'median_per_anchor.csv')
    print(summary.round(4).to_string())
    print('Units: depth=m; fractions=0..1. Regions are image halves, NOT semantic labels.')
    print('Same-pixel forward-depth discrepancy includes occlusion; C is a reference, not perfect truth.')


if __name__ == '__main__':
    main()
