"""Post-PNG A restoration probes. Not pure geometry/intensity interventions."""
import argparse
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(); args.output_dir=args.output_dir.resolve()
    project=Path(__file__).resolve().parents[1]
    root=project/'outputs/shimizu_protocol_507q_754db_abc'
    original=pd.read_csv(args.source_root/'images.csv',dtype={'anchor':str})
    check=pd.read_csv(args.source_root/'baseline_check.csv')
    assert (check.changed_pixels==0).all() and (args.source_root/'completed.json').exists()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.source_root/'baseline_check.csv',args.output_dir/'baseline_check.csv')
    rows=original.to_dict('records'); audit=[]
    for segment,group in original.groupby('segment'):
        with np.load(root/'maps/B_tulip32/training'/segment/'map_static.npz') as z:
            generated=z['source_is_generated'].reshape(-1).astype(bool)
        for anchor, sub in group.groupby('anchor',sort=False):
            images={}
            for v in 'AB':
                r=sub[sub.condition==v+'_original'].iloc[0]
                images[v]=cv2.imread(r.image,cv2.IMREAD_GRAYSCALE)
                assert images[v] is not None
            idx,depth={},{}
            for v in 'AB':
                with np.load(root/f'anchors_{v}/training'/segment/anchor/'anchor_index.npz') as z:
                    idx[v]=z['point_index']; depth[v]=z['depth']
            av,bv=idx['A']>=0,idx['B']>=0
            gen=np.zeros_like(bv); gen[bv]=generated[idx['B'][bv]]
            replaced=av&bv&gen
            valid=replaced&np.isfinite(depth['A'])&np.isfinite(depth['B'])
            d=np.full(av.shape,np.inf)
            d[valid]=np.abs(depth['A'][valid]-depth['B'][valid])
            masks={'probe_restore_all_replaced':replaced,
                   'probe_restore_depth_le01':valid&(d<=0.1),
                   'probe_restore_depth_01to2':valid&(d>0.1)&(d<=2),
                   'probe_restore_depth_gt2':valid&(d>2)}
            h,w=av.shape
            for y in range(2):
                for x in range(3):
                    m=np.zeros_like(av); m[y*h//2:(y+1)*h//2,x*w//3:(x+1)*w//3]=True
                    masks[f'probe_restore_tile_r{y}_c{x}']=replaced&m
            for name,m in masks.items():
                image=images['B'].copy(); image[m]=images['A'][m]
                dest=args.output_dir/name/segment/anchor/'anchor.png'
                dest.parent.mkdir(parents=True,exist_ok=True)
                assert cv2.imwrite(str(dest),image)
                rows.append(dict(role=sub.iloc[0].role,segment=segment,anchor=anchor,condition=name,
                    variant='PNG_restoration_probe',upper=None,post=True,image=str(dest)))
                audit.append(dict(segment=segment,anchor=anchor,condition=name,pixels=int(m.sum()),
                    fraction_of_replaced=float(m.sum()/replaced.sum()) if replaced.any() else 0))
            print('Prepared probes anchor='+anchor,flush=True)
    pd.DataFrame(rows).to_csv(args.output_dir/'images.csv',index=False)
    pd.DataFrame(audit).to_csv(args.output_dir/'probe_pixel_counts.csv',index=False)
    (args.output_dir/'completed.json').write_text(json.dumps(dict(
        note='Restores A PNG inside selected B replacement masks. Both GT and rival are changed. Tiles are image regions, not semantic labels. Different mask sizes; not a pure position/intensity separation.')))


if __name__=='__main__': main()
