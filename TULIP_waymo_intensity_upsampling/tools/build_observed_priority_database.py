"""Build an isolated observed-priority raster DB with matching depth/index.

Diagnostic image-space composition, not a physical rerender or thesis baseline.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0, help='Smoke test only; 0 builds full DB')
    p.add_argument('--input-root', type=Path)
    p.add_argument('--b-maps-root', type=Path)
    p.add_argument('--expected-db', type=int, default=754)
    p.add_argument('--expected-queries', type=int, default=507)
    args = p.parse_args()
    args.output_dir = args.output_dir.resolve()
    project = Path(__file__).resolve().parents[1]
    root = args.input_root or project/'outputs/shimizu_protocol_507q_754db_abc'
    map_roots = {'A': root/'maps/A_observed16',
                 'B': args.b_maps_root or root/'maps/B_tulip32'}
    renderer = project.parent/'handover_shimizu/src/data_make/valid_test/render_anchor_intensity_txt_se3_fix.py'
    source = renderer.read_text()
    names = {'apply_gamma_clip','masked_hist_equalize_uint8','apply_clahe_uint8'}
    nodes = [n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==3
    scope = {'np':np,'cv2':cv2}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(renderer),'exec'),scope)
    db = pd.read_csv(root/'retrieval_manifests/B_db.csv', dtype={'anchor_id':str})
    pairs = pd.read_csv(root/'retrieval_manifests/B_pairs.csv', dtype={'anchor_id':str})
    assert len(db)==args.expected_db and len(pairs)==args.expected_queries
    assert not db.duplicated(['db_seg','anchor_id']).any()
    a_db = pd.read_csv(root/'retrieval_manifests/A_db.csv', dtype={'anchor_id':str})
    a_folders = {(r.db_seg,r.anchor_id):Path(r.anchor_png).parent for r in a_db.itertuples()}
    work = db.head(args.limit) if args.limit>0 else db
    args.output_dir.mkdir(parents=True,exist_ok=False)
    checks, pathmap = [],{}

    def post(img,valid,profile):
        pp=profile['postprocess']; mask=valid if pp['mask_covered_only'] else np.ones_like(valid)
        if pp['hist_equalize']:
            img=scope['masked_hist_equalize_uint8'](img,mask)
        if pp['clahe']:
            img=scope['apply_clahe_uint8'](img,mask,pp['clahe_clip_limit'],pp['clahe_tile_grid'])
        return img

    for segment, group in work.groupby('db_seg',sort=False):
        maps={}
        for v, map_root in map_roots.items():
            with np.load(map_root/'training'/segment/'map_static.npz',allow_pickle=False) as z:
                maps[v]=(z['xyz'],z['intensity'].reshape(-1))
                if v=='B': generated=z['source_is_generated'].reshape(-1).astype(bool)
        a_to_b=np.flatnonzero(~generated)
        assert np.array_equal(maps['A'][0],maps['B'][0][a_to_b]), 'Observed XYZ/order mismatch'
        assert np.array_equal(maps['A'][1],maps['B'][1][a_to_b]), 'Observed intensity mismatch'
        for r in group.itertuples():
            idx,depth,meta,raw={}, {}, {}, {}
            for v,folder in [('A',a_folders[segment,r.anchor_id]),('B',Path(r.anchor_png).parent)]:
                meta[v]=json.loads((folder/'anchor.json').read_text())
                with np.load(folder/'anchor_index.npz',allow_pickle=False) as z:
                    idx[v]=z['point_index']; depth[v]=z['depth']
                profile=meta[v]['render_profile']
                assert profile['int_mode']=='global_map' and profile['range_corr_power']==0
                valid=idx[v]>=0
                assert valid.any() and idx[v][valid].max()<len(maps[v][1])
                assert np.isfinite(depth[v][valid]).all() and (depth[v][valid]>0).all()
                values=scope['apply_gamma_clip'](np.clip(maps[v][1]/(profile['intensity_up_val']+1e-6),0,1),profile['gamma'])
                base=np.zeros(idx[v].shape,np.float32); base[valid]=values[idx[v][valid]]
                image=np.clip((base*255).round(),0,255).astype(np.uint8)
                original=cv2.imread(str(folder/'anchor.png'),cv2.IMREAD_GRAYSCALE)
                assert original is not None
                assert np.array_equal(post(image,valid,profile),original), f'Baseline mismatch {v}/{segment}/{r.anchor_id}'
                upper=meta['A']['render_profile']['intensity_up_val']
                values=scope['apply_gamma_clip'](np.clip(maps[v][1]/(upper+1e-6),0,1),profile['gamma'])
                base=np.zeros(idx[v].shape,np.float32); base[valid]=values[idx[v][valid]]
                raw[v]=np.clip((base*255).round(),0,255).astype(np.uint8)
            for field in ['intrinsic','world_to_cam']:
                assert np.allclose(meta['A'][field],meta['B'][field],atol=1e-8,rtol=0)
            assert idx['A'].shape==idx['B'].shape
            av,bv=idx['A']>=0,idx['B']>=0
            added=~av&bv
            image=raw['A'].copy(); image[added]=raw['B'][added]
            image=post(image,av|bv,meta['A']['render_profile'])
            # Point indices refer to the original B map. A points exist in that map.
            mixed_idx=np.full(idx['B'].shape,-1,dtype=np.int32)
            mixed_idx[av]=a_to_b[idx['A'][av]]; mixed_idx[added]=idx['B'][added]
            mixed_depth=np.full(depth['B'].shape,np.inf,dtype=np.float32)
            mixed_depth[av]=depth['A'][av]; mixed_depth[added]=depth['B'][added]
            assert np.array_equal(mixed_idx>=0,av|bv)
            assert np.array_equal(mixed_depth[av],depth['A'][av])
            assert np.array_equal(mixed_depth[added],depth['B'][added])
            folder=args.output_dir/'anchors'/segment/r.anchor_id
            folder.mkdir(parents=True)
            assert cv2.imwrite(str(folder/'anchor.png'),image)
            np.savez_compressed(folder/'anchor_index.npz',point_index=mixed_idx,depth=mixed_depth,
                                source_is_generated=np.where(av|bv,generated[np.maximum(mixed_idx,0)],False))
            m=meta['A']; m['image_path']=str(folder/'anchor.png'); m['coverage']=float((av|bv).mean())
            m['diagnostic_composition']={'rule':'A covered first, then B holes; joint A-profile postprocessing',
                'point_index_map':str(map_roots['B']/'training'/segment/'map_static.npz'),
                'depth_source':'A on A-covered pixels; B on added pixels',
                'not_physical_rerender':True}
            (folder/'anchor.json').write_text(json.dumps(m,indent=2))
            pathmap[r.anchor_png]=str(folder/'anchor.png')
            checks.append(dict(segment=segment,anchor=r.anchor_id,baseline_exact=True,
                observed_pixels=int(av.sum()),added_pixels=int(added.sum()),valid_pixels=int((av|bv).sum())))
            print(f'[{len(checks)}/{len(work)}] {segment} anchor={r.anchor_id}',flush=True)
    for segment, group in work.groupby('db_seg'):
        layout = Path(group.iloc[0].anchor_png).parent.parent/'anchors.csv'
        if layout.is_file():
            shutil.copy2(layout,args.output_dir/'anchors'/segment/'anchors.csv')
    pd.DataFrame(checks).to_csv(args.output_dir/'checks.csv',index=False)
    if args.limit>0:
        print('SMOKE ONLY: full evaluation manifests were not written.'); return
    for name,table in [('db',db),('pairs',pairs)]:
        table=table.copy(); table['anchor_png']=table['anchor_png'].map(pathmap)
        assert table.anchor_png.notna().all()
        table.to_csv(args.output_dir/f'{name}.csv',index=False)
        table.to_parquet(args.output_dir/f'{name}.parquet',index=False)
    (args.output_dir/'completed.json').write_text(json.dumps(dict(db=len(db),queries=len(pairs),
        renderer_sha256=hashlib.sha256(source.encode()).hexdigest(),
        note='Diagnostic observed-priority rasters, matching depth, B-map indices. No retraining; no claim of thesis equivalence.'),indent=2))
    print('DONE: full DB and matching depth/index prepared')


if __name__=='__main__': main()
