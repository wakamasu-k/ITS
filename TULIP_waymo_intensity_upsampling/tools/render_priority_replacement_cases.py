"""Diagnostic contact sheets. Pixel masks are not model attribution maps."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    abc = project/'outputs/shimizu_protocol_507q_754db_abc'
    images = pd.read_csv(args.run_root/'images/images.csv', dtype={'anchor':str})
    scores = pd.read_csv(args.run_root/'scores/scores.csv')
    cases = {50:45, 59:53, 47:42, 102:None}
    chosen = images[images.anchor.isin([str(a) for a in cases])]
    assert chosen.segment.nunique()==1
    segment = chosen.segment.iloc[0]
    with np.load(abc/'maps/B_tulip32/training'/segment/'map_static.npz') as z:
        generated = z['source_is_generated'].reshape(-1).astype(bool)
    records = [json.loads(s) for s in (project/'outputs/paperlike_retrieval_A_507q/paperlike_A_507q_topk_top50.jsonl').read_text().splitlines()]
    queries = {r['q_idx']:r for r in records}
    args.output_dir.mkdir(parents=True,exist_ok=False)
    metrics = []

    def tile(image, title, mask=False):
        if image.ndim==2:
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        thumb = cv2.resize(image,(576,384),interpolation=cv2.INTER_NEAREST if mask else cv2.INTER_AREA)
        out = np.full((420,576,3),245,np.uint8)
        out[36:] = thumb
        cv2.putText(out,title,(8,24),cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,0,0),1,cv2.LINE_AA)
        return out

    for anchor,query in cases.items():
        rasters = {}
        for condition in ['A_original','B_original','AB_holes_joint_post']:
            row = chosen[(chosen.anchor==str(anchor))&(chosen.condition==condition)]
            assert len(row)==1
            image = cv2.imread(row.iloc[0].image,cv2.IMREAD_GRAYSCALE)
            assert image is not None
            rasters[condition] = image
        idx,dep = {},{}
        for v in 'AB':
            folder = abc/f'anchors_{v}/training'/segment/str(anchor)
            with np.load(folder/'anchor_index.npz') as z:
                idx[v] = z['point_index']; dep[v] = z['depth'].astype(float)
        av,bv = idx['A']>=0,idx['B']>=0
        assert av.shape==bv.shape==rasters['A_original'].shape
        gen = np.zeros_like(bv)
        assert idx['B'][bv].max()<len(generated)
        gen[bv] = generated[idx['B'][bv]]
        replaced = av & bv & gen
        added = ~av & bv
        mask = np.zeros((*av.shape,3),np.uint8)
        mask[av&bv&~gen] = (100,100,100)
        mask[replaced] = (0,0,255)  # red
        mask[added] = (0,200,0)    # green
        mask[av&~bv] = (255,0,255)
        delta = np.zeros(av.shape,float)
        common = av&bv&np.isfinite(dep['A'])&np.isfinite(dep['B'])
        delta[common] = np.abs(dep['B'][common]-dep['A'][common])
        heat = cv2.applyColorMap(np.uint8(np.clip(delta/2,0,1)*255),cv2.COLORMAP_TURBO)
        heat[~(replaced&common)] = 0
        diff = cv2.absdiff(rasters['A_original'],rasters['B_original'])
        diff[~replaced]=0
        panels = [tile(rasters['A_original'],'A: observed16'),tile(rasters['B_original'],'B: reconstructed32'),
                  tile(rasters['AB_holes_joint_post'],'A preserved + B holes + joint post'),
                  tile(mask,'Red=replaced; green=new; gray=B measured',True),
                  tile(heat,'Replaced: |depth B-A| blue=0 red>=2m',True),
                  tile(diff,'Replaced: |PNG B-A| black=0 white=255')]
        sheet=np.vstack([np.hstack(panels[:3]),np.hstack(panels[3:])])
        assert cv2.imwrite(str(args.output_dir/f'anchor_{anchor}_comparison.png'),sheet)
        assert cv2.imwrite(str(args.output_dir/f'anchor_{anchor}_replacement_mask.png'),mask)
        metrics.append(dict(anchor=anchor,query=query,a_covered=int(av.sum()),replaced=int(replaced.sum()),
            replaced_fraction_of_A=float(replaced.sum()/av.sum()),added=int(added.sum()),
            replaced_depth_AB_median=float(np.median(delta[replaced&common])),
            replaced_png_AB_mae=float(diff[replaced].mean())))
        if query is not None:
            camera=cv2.imread(queries[query]['q_cam_png'])
            assert camera is not None
            assert cv2.imwrite(str(args.output_dir/f'query_{query}_camera.png'),camera)
        print('Wrote anchor',anchor,flush=True)
    pd.DataFrame(metrics).to_csv(args.output_dir/'pixel_summary.csv',index=False)
    selected=scores[scores['query'].isin([45,53,42])].copy()
    base=selected[selected.condition=='B_original'].set_index('query')
    selected['gt_gain_vs_B']=selected.gt_score-selected['query'].map(base.gt_score)
    selected['rival_gain_vs_B']=selected.rival_score-selected['query'].map(base.rival_score)
    selected.to_csv(args.output_dir/'selected_scores.csv',index=False)
    show=selected[selected.condition.isin(['A_original','B_original','AB_holes_joint_post'])]
    print(show[['query','gt_anchor','condition','gt_score','rival_score','margin','gt_gain_vs_B','rival_gain_vs_B']].round(6).to_string(index=False))
    (args.output_dir/'README.txt').write_text('Masks show raster provenance, NOT network attention. Depth difference is B versus A, not C or GT error. Heat saturates at 2m. Gray means selected B point is measured, not verified identical to A point. Diagnostic composites are not physical rerenders. Queries are consecutive, not independent scenes.\n')


if __name__=='__main__':
    main()
