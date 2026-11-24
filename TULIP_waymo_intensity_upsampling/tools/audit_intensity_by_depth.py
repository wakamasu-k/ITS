"""Read-only same-pixel intensity comparison; C is a reference, not ground truth."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0, help='0: all fixed cases; positive: first N cases')
    args = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = project/'outputs/shimizu_protocol_507q_754db_abc'
    cases = pd.read_csv(project/'outputs/gt_vs_anchor102_audit_ax6mcoi2/anchor_comparison.csv', dtype={'anchor': str})
    if args.limit > 0:
        cases = cases.head(args.limit)
    folders = {}
    for v in 'ABC':
        df = pd.read_csv(root/f'retrieval_manifests/{v}_db.csv', dtype={'anchor_id': str})
        folders[v] = {(r.db_seg, r.anchor_id): Path(r.anchor_png).parent for r in df.itertuples()}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, checks = [], []
    for segment, group in cases.groupby('segment'):
        paths = {'A': root/'maps/A_observed16/training'/segment/'map_static.npz',
                 'B': root/'maps/B_tulip32/training'/segment/'map_static.npz',
                 'C': Path('/mnt/w/32line/datasets/maps/cross/original_val/db/training')/segment/'map_static.npz'}
        maps = {}
        for v, path in paths.items():
            with np.load(path, allow_pickle=False) as z:
                maps[v] = (z['xyz'], z['intensity'].reshape(-1))
                if v == 'B':
                    generated = z['source_is_generated'].reshape(-1).astype(bool)
        for case in group.itertuples():
            depth, raw, valid, meta, indices, png = {}, {}, {}, {}, {}, {}
            for v in 'ABC':
                folder = folders[v][segment, case.anchor]
                meta[v] = json.loads((folder/'anchor.json').read_text())
                with np.load(folder/'anchor_index.npz', allow_pickle=False) as z:
                    idx = z['point_index']; depth[v] = z['depth'].astype(float)
                indices[v] = idx
                image = cv2.imread(str(folder/'anchor.png'), cv2.IMREAD_GRAYSCALE)
                assert image is not None and image.shape == idx.shape, folder
                png[v] = image.astype(np.float64)
                ok = (idx >= 0) & np.isfinite(depth[v]) & (depth[v] > 0)
                xyz, intensity = maps[v]
                assert ok.any() and idx[ok].max() < len(intensity)
                # Check the referenced map really agrees with stored forward depth.
                ids = idx[ok]; transform = np.asarray(meta[v]['world_to_cam'])
                reconstructed = xyz[ids].astype(float) @ transform[0, :3] + transform[0, 3]
                error = np.abs(reconstructed-depth[v][ok])
                checks.append(dict(anchor=case.anchor, variant=v, map=str(paths[v]), depth_check_max=float(error.max())))
                assert error.max() < 0.01, (v, case.anchor, error.max())
                raw[v] = np.full(idx.shape, np.nan)
                raw[v][ok] = intensity[ids]
                valid[v] = ok & np.isfinite(raw[v])
            for v in 'AB':
                assert depth[v].shape == depth['C'].shape
                for field in ('intrinsic', 'world_to_cam'):
                    assert np.allclose(meta[v][field], meta['C'][field], atol=1e-8, rtol=0)
            gen = np.zeros(indices['B'].shape, bool)
            ok = indices['B'] >= 0
            gen[ok] = generated[indices['B'][ok]]
            with np.errstate(invalid='ignore'):
                d = np.abs(depth['B']-depth['C'])
                da = np.abs(depth['A']-depth['C'])
            for source, mask in [('generated',gen), ('measured',~gen)]:
                common = mask & valid['B'] & valid['C']
                for label, lo, hi in [('le0.1m',-1,0.1), ('0.1to0.5m',0.1,0.5), ('0.5to2m',0.5,2), ('over2m',2,np.inf)]:
                    selected = common & (d>lo) & (d<=hi)
                    paired = selected & valid['A']
                    # Stricter subset: both A and B depths close to C.
                    near = paired & (da<=0.1) & (d<=0.1)
                    row = dict(role=case.role, anchor=case.anchor, source=source, depth_bin=label,
                               pixels=int(selected.sum()), fraction_of_BC_common=float(selected.sum()/common.sum()) if common.any() else np.nan,
                               BC_raw_mae=float(np.abs(raw['B'][selected]-raw['C'][selected]).mean()) if selected.any() else np.nan)
                    errors = np.abs(raw['B'][selected]-raw['C'][selected])
                    row['BC_raw_median'] = float(np.median(errors)) if errors.size else np.nan
                    row['BC_raw_p95'] = float(np.percentile(errors,95)) if errors.size else np.nan
                    for name, subset in [('paired',paired), ('both_near',near)]:
                        ea = np.abs(raw['A'][subset]-raw['C'][subset]); eb = np.abs(raw['B'][subset]-raw['C'][subset])
                        row[name+'_pixels'] = int(subset.sum())
                        row[name+'_AC_median'] = float(np.median(ea)) if ea.size else np.nan
                        row[name+'_BC_median'] = float(np.median(eb)) if eb.size else np.nan
                        for key, values in [('AC_mae',ea),('BC_mae',eb),('delta',eb-ea)]:
                            row[name+'_'+key] = float(values.mean()) if values.size else np.nan
                        pa = np.abs(png['A'][subset]-png['C'][subset])
                        pb = np.abs(png['B'][subset]-png['C'][subset])
                        ab = np.abs(png['B'][subset]-png['A'][subset])
                        for key, values in [('AC_mae',pa), ('BC_mae',pb), ('delta',pb-pa), ('AB_mae',ab)]:
                            row[name+'_png_'+key] = float(values.mean()) if values.size else np.nan
                        row[name+'_png_B_worse_fraction'] = float((pb>pa).mean()) if pa.size else np.nan
                    rows.append(row)
            print('Checked anchor='+case.anchor, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir/'per_anchor.csv', index=False)
    pd.DataFrame(checks).to_csv(args.output_dir/'map_index_checks.csv', index=False)
    summary = df.groupby(['role','source','depth_bin']).median(numeric_only=True)
    summary.to_csv(args.output_dir/'median_per_anchor.csv')
    png_cols = ['paired_pixels','paired_png_AC_mae','paired_png_BC_mae','paired_png_delta',
                'paired_png_AB_mae','paired_png_B_worse_fraction','both_near_pixels',
                'both_near_png_AC_mae','both_near_png_BC_mae','both_near_png_delta']
    summary[png_cols].to_csv(args.output_dir/'png_median_per_anchor.csv')
    print(summary[png_cols].round(6).to_string())
    print('PNG differences: 0..255 grayscale levels. Each row is a median across anchors, not pooled pixels.')
    print('Raw intensity units, not PNG levels. Delta=BC minus AC on identical pixels; positive means farther from C. No causal attribution.')


if __name__ == '__main__':
    main()
