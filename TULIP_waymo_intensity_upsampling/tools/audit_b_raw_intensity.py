"""Check stored intensities and raster-selected pixels without modifying inputs."""
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd


def stats(values):
    v = np.asarray(values).reshape(-1)
    good = np.isfinite(v)
    f = v[good]
    return dict(count=len(v), nonfinite_count=int((~good).sum()),
                zero_fraction=float((f==0).mean()) if len(f) else None,
                negative_fraction=float((f<0).mean()) if len(f) else None,
                p05=float(np.percentile(f,5)) if len(f) else None,
                median=float(np.median(f)) if len(f) else None,
                p95=float(np.percentile(f,95)) if len(f) else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    base = Path(__file__).resolve().parents[1]/'outputs'
    root = base/'shimizu_protocol_507q_754db_abc'
    cases = pd.read_csv(base/'gt_vs_anchor102_audit_ax6mcoi2/anchor_comparison.csv', dtype={'anchor':str})
    if args.limit:
        cases = cases.head(args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, maps = [], []
    for seg, group in cases.groupby('segment'):
        with np.load(root/'maps/B_tulip32/training'/seg/'map_static.npz', allow_pickle=False) as z:
            intensity = z['intensity'].reshape(-1)
            generated = z['source_is_generated'].reshape(-1).astype(bool)
        assert len(intensity) == len(generated)
        for label, mask in [('generated',generated),('measured',~generated)]:
            maps.append(dict(segment=seg, source=label, **stats(intensity[mask])))
        for _, case in group.iterrows():
            folder = root/'anchors_B/training'/seg/case.anchor
            with np.load(folder/'anchor_index.npz', allow_pickle=False) as z:
                idx = z['point_index']
            png = cv2.imread(str(folder/'anchor.png'), cv2.IMREAD_GRAYSCALE)
            assert png is not None and png.shape == idx.shape
            covered = idx>=0
            assert idx[covered].max() < len(intensity)
            ids = idx[covered]
            raw, rendered = intensity[ids], png[covered]
            for label, mask in [('generated',generated[ids]),('measured',~generated[ids])]:
                r, image = raw[mask], rendered[mask]
                finite_positive = np.isfinite(r)&(r>0)
                rows.append(dict(role=case.role, anchor=case.anchor, source=label,
                    **{'raw_'+k:v for k,v in stats(r).items()},
                    png_zero_fraction=float((image==0).mean()) if len(image) else None,
                    png_255_fraction=float((image==255).mean()) if len(image) else None,
                    positive_raw_count=int(finite_positive.sum()),
                    positive_raw_png_zero_fraction=float((image[finite_positive]==0).mean()) if finite_positive.any() else None))
            print('Checked anchor='+case.anchor, flush=True)
    df = pd.DataFrame(rows)
    pd.DataFrame(maps).to_csv(args.output_dir/'map_intensity.csv', index=False)
    df.to_csv(args.output_dir/'rendered_intensity.csv', index=False)
    print('\nMAP POINTS:')
    print(pd.DataFrame(maps).to_string(index=False))
    cols = ['raw_zero_fraction','raw_negative_fraction','raw_median','raw_p95','png_zero_fraction','png_255_fraction','positive_raw_png_zero_fraction']
    print('\nMEDIAN PER ANCHOR:')
    print(df.groupby(['role','source'])[cols].median().round(6).to_string())
    print('Nonfinite rendered intensity count:', int(df.raw_nonfinite_count.sum()))
    print('Zero intensity is not proof of a missing point. Raster statistics are pixel-weighted.')


if __name__ == '__main__':
    main()
