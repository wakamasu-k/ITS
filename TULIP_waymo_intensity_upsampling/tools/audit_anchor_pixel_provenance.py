#!/usr/bin/env python3
"""Inspect existing A/B raster provenance for one anchor without rerendering."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser(); p.add_argument('--segment',required=True)
    p.add_argument('--anchor',default='0'); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); root=Path(__file__).resolve().parents[1]/'outputs/shimizu_protocol_507q_754db_abc'
    if args.output.exists(): raise FileExistsError(args.output)
    with np.load(root/'maps/B_tulip32/training'/args.segment/'map_static.npz',allow_pickle=False) as z:
        generated=z['source_is_generated'].astype(bool)
    a_to_b=np.flatnonzero(~generated)
    images={}; indices={}; meta={}
    for v in 'AB':
        folder=root/f'anchors_{v}/training'/args.segment/args.anchor
        with np.load(folder/'anchor_index.npz',allow_pickle=False) as z: indices[v]=z['point_index']
        images[v]=cv2.imread(str(folder/'anchor.png'),cv2.IMREAD_GRAYSCALE)
        meta[v]=json.loads((folder/'anchor.json').read_text())
    a,b=indices['A'],indices['B']; av=a>=0; bv=b>=0
    assert a[av].max()<len(a_to_b) and b[bv].max()<len(generated)
    assert images['A'].shape==images['B'].shape==a.shape==b.shape
    bg=np.zeros_like(bv); bg[bv]=generated[b[bv]]
    mapped=np.full(a.shape,-1,dtype=np.int64); mapped[av]=a_to_b[a[av]]
    same=av & bv & (mapped==b)
    both=av & bv; newly=bv & ~av
    diff=np.abs(images['A'].astype(np.int16)-images['B'].astype(np.int16))
    out=dict(segment=args.segment,anchor=args.anchor,a_valid=int(av.sum()),b_valid=int(bv.sum()),
        b_generated_pixels=int(bg.sum()),newly_covered=int(newly.sum()),a_coverage_lost=int((av & ~bv).sum()),
        a_covered_now_generated=int((av & bg).sum()),same_measured_point_pixels=int(same.sum()),
        same_measured_point_png_changed=int((same & (diff>0)).sum()),
        same_measured_point_png_mae=float(diff[same].mean()) if same.any() else None,
        common_other_measured_point_pixels=int((both & ~bg & ~same).sum()),
        normalization_upper={v:meta[v]['render_profile']['intensity_up_val'] for v in 'AB'},
        note='Uses exact measured-point order preservation established in separate map audit; postprocessed PNG differences, not raw intensity errors.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))


if __name__=='__main__': main()
