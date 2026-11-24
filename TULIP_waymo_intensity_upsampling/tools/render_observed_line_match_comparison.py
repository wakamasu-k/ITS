"""Render same-pair observed 8/16/32-line LoFTR/PnP diagnostics."""
import argparse
import os
import json
import subprocess
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--its-root', type=Path, default=Path('/home/wakamatsu/ITS'))
    parser.add_argument('--out-parent', type=Path, default=Path('/mnt/w/its_outputs'))
    args = parser.parse_args()
    args.out_parent.mkdir(parents=True, exist_ok=True)
    its = args.its_root.resolve()
    project = its/'TULIP_waymo_intensity_upsampling'
    data = project/'outputs'
    ref = its/'handover_shimizu/src/LoFTR'
    out = Path(tempfile.mkdtemp(prefix='observed_line_match_compare_', dir=args.out_parent))
    print('OUTPUT', out, flush=True)
    man = data/'shimizu_protocol_507q_754db_abc/retrieval_manifests'
    keys = ['pair_id', 'db_seg', 'anchor_id']
    a, c = [pd.read_parquet(man/f'{v}_db.parquet').set_index(keys) for v in 'AC']
    assert a.index.is_unique and c.index.is_unique
    pairs = pd.read_csv(data/'B_dense_distance_protocol_20261019/B_strict_507.csv')
    paths = []
    chosen = []
    for prefix in ('128443', '641049', '126816'):
        subset = pairs[pairs.pair_id.str.startswith('segpair_'+prefix)].copy()
        subset['choice'] = (subset.anchor_id - 50).abs()
        for _, row in subset.sort_values(['choice','dist_xy_m']).iterrows():
            key = tuple(row[k] for k in keys)
            if key not in a.index or key not in c.index:
                continue
            p16, p32 = [Path(d.loc[key,'anchor_png']) for d in (a,c)]
            p8 = data/'observed8_localization_20261024/data/test/anchors'/row.db_subset/row.db_seg/str(row.anchor_id)/'anchor.png'
            variants = {8:p8,16:p16,32:p32}
            if all(p.is_file() and p.with_name('anchor.json').is_file() and p.with_name('anchor_index.npz').is_file() for p in variants.values()):
                metas = [json.loads(p.with_name('anchor.json').read_text()) for p in variants.values()]
                common = set.intersection(*(set(m) for m in metas))
                for field in ('world_to_cam','intrinsic','T_wv_rowmajor','anchor_T_wv_rowmajor'):
                    if field in common and not isinstance(metas[0][field],dict):
                        assert all(np.allclose(metas[0][field],m[field],atol=1e-5,rtol=0) for m in metas[1:]), (key,field)
                pose = 'anchor_T_wv_rowmajor'
                def matrix(value):
                    return np.asarray(json.loads(value) if isinstance(value,str) else value)
                assert np.allclose(matrix(a.loc[key,pose]),matrix(c.loc[key,pose]),atol=1e-5,rtol=0), key
                print('PAIR', key, 'frame',row.q_frame_index,'common metadata', sorted(common),flush=True)
                paths.append(variants)
                chosen.append(row.drop(labels='choice'))
                break
        else:
            raise RuntimeError(f'No common pair in {prefix}')
    cases = pd.DataFrame(chosen)
    cases[['pair_id','db_seg','q_frame_index','anchor_id','q_image_png','dist_xy_m']].to_csv(out/'cases.csv',index=False)
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    def font(size):
        return ImageFont.truetype(fontpath,size) if Path(fontpath).exists() else ImageFont.load_default()
    def load(path):
        with Image.open(path) as im:
            return im.convert('RGB')
    def fit(im,w):
        return im.resize((w,round(im.height*w/im.width)),Image.Resampling.LANCZOS)
    for i,(variants, (_,case)) in enumerate(zip(paths,cases.iterrows()),1):
        for target in (16,32):
            panels = [fit(load(variants[line]),900) for line in (8,target)]
            canvas=Image.new('RGB',(1800,90+max(p.height for p in panels)), '#151b25')
            draw=ImageDraw.Draw(canvas)
            draw.text((20,10),f"Scene {i} | {case.pair_id} | frame {case.q_frame_index} | anchor {case.anchor_id}",font=font(26),fill='white')
            for col,(line,panel) in enumerate(zip((8,target),panels)):
                draw.text((col*900+20,50),f'Observed {line} lines',font=font(24),fill='white')
                canvas.paste(panel,(col*900,90))
            canvas.save(out/f'scene{i}_8_vs_{target}_intensity.png')
    env=dict(os.environ,PYTHONPATH=str(ref/'src'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    metrics={}
    for line in (8,16,32):
        dest=out/f'line{line}'
        df=cases.copy()
        for col,name in [('anchor_png','anchor.png'),('anchor_json','anchor.json'),('anchor_index_npz','anchor_index.npz')]:
            df[col]=[str(p[line].with_name(name)) for p in paths]
        df=df[['pair_id','q_seg','db_seg','q_frame_index','anchor_id','q_image_png','q_meta_json','anchor_png','anchor_json','anchor_index_npz','dist_xy_m','is_strict_gt','is_gt_t1m','is_gt_t2m','is_gt_t5m','is_gt_t10m','error']]
        manifest=out/f'line{line}.csv'
        df.to_csv(manifest,index=False)
        subprocess.run([str(its/'.venv_tulip/bin/python'),str(ref/'tools/eval_cross_match_to_pose_zalign_multi_visall_v2.py'),'--manifest',str(manifest),'--ckpt',str(its/'waymo_outputs/old32_dense/loftr_retrain_stage1_paper_wdrive/stage1_final_gstep715958.pth'),'--out-dir',str(dest),'--device','cuda','--amp-dtype','fp16','--gt-type','strict','--icp-csv','/mnt/w/32line/datasets/splits/cross/all/icp_startend_staticmaps.csv','--save-match-vis','--match-vis-max-draw','180'],env=env,check=True)
        d=pd.read_csv(dest/'per_pair.csv')
        assert len(d)==3 and d.comparable.eq(True).all()
        metrics[line]=d.assign(lines=line)
    table=pd.concat(metrics.values())
    table.to_csv(out/'match_comparison.csv',index=False)
    for i,(_,case) in enumerate(cases.iterrows(),1):
        for target in (16,32):
            panels=[]
            for line in (8,target):
                ans=metrics[line]
                row=ans[(ans.pair_id==case.pair_id)&(ans.q_frame_index==case.q_frame_index)].iloc[0]
                prefix=f'q{int(row.q_frame_index):06d}_'
                candidates=list((out/f'line{line}/match_vis/strict').rglob(prefix+'*.png'))
                if len(candidates)!=1:
                    raise RuntimeError((prefix,candidates))
                panels.append((line,row,fit(load(candidates[0]),1540)))
            height=125+sum(p.height+56 for _,_,p in panels)
            canvas=Image.new('RGB',(1540,height),'#151b25')
            draw=ImageDraw.Draw(canvas)
            draw.text((20,12),f"Scene {i} | {case.pair_id} | frame {case.q_frame_index} | anchor {case.anchor_id}",font=font(29),fill='white')
            draw.text((20,54),'Each row: LiDAR intensity (left) and camera (right)',font=font(23),fill='white')
            draw.text((20,86),'Green: GT reprojection <= 4 px. Red: outside this criterion. Up to 180 lines shown.',font=font(22),fill='#cee1ef')
            y=125
            for line,r,panel in panels:
                error=f'{r.t_err_xy:.3f} m' if r.pnp_success and np.isfinite(r.t_err_xy) else 'n/a'
                draw.text((20,y+10),f'Observed {line} lines  |  LoFTR {r.n_match_in}  |  3D pairs {r.n_3d}  |  PnP inliers {r.n_inlier}  |  XY error {error}',font=font(24),fill='white')
                canvas.paste(panel,(0,y+52))
                y+=panel.height+56
            canvas.save(out/f'scene{i}_8_vs_{target}_matches.png')
    print(table[['lines','pair_id','anchor_id','n_match_in','n_3d','n_inlier','pnp_success','t_err_xy']].to_string(index=False),flush=True)
    print('FINISHED',out,flush=True)


if __name__ == '__main__':
    main()
