"""Pair measured-16 and original TULIP-32 anchor PNGs at identical poses.

Uses A/B retrieval manifests. Does not rerun TULIP, rerender maps, normalize
intensity, apply observed-priority composition, or resize the source images.
B includes the retained measured points and generated points.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--manifest-dir',type=Path,default=PROJECT/
                   'outputs/shimizu_protocol_507q_754db_abc/retrieval_manifests')
    p.add_argument('--pair-ids',nargs='+',default=['segpair_126816__366934',
                    'segpair_138235__663760','segpair_848780__183339'])
    p.add_argument('--anchor-ids',nargs='+',type=int,default=[25,50])
    a=p.parse_args()
    out=a.output_dir.resolve()
    if out.exists(): raise FileExistsError(f'Refusing to overwrite: {out}')
    tables={v:pd.read_csv(a.manifest_dir/f'{v}_db.csv',dtype={'anchor_id':int}) for v in 'AB'}
    keys=['pair_id','db_seg','anchor_id']
    for v,t in tables.items():
        if t.duplicated(keys).any(): raise ValueError(f'Duplicate A/B anchor identity in {v}')
    joined=tables['A'].merge(tables['B'],on=keys,suffixes=('_A','_B'),validate='one_to_one')
    if len(joined)!=len(tables['A']) or len(joined)!=len(tables['B']):
        raise ValueError('A/B anchor sets differ')
    jobs=[]
    for pair in dict.fromkeys(a.pair_ids):
        for aid in dict.fromkeys(a.anchor_ids):
            sub=joined[(joined.pair_id==pair)&(joined.anchor_id==aid)]
            if len(sub)!=1: raise ValueError(f'Expected one anchor: {pair}/{aid}')
            row=sub.iloc[0]; paths={v:Path(row[f'anchor_png_{v}']) for v in 'AB'}
            meta={v:json.loads(paths[v].with_name('anchor.json').read_text()) for v in 'AB'}
            for v in 'AB':
                if meta[v]['segment_id']!=row.db_seg or int(meta[v]['submap_id'])!=aid:
                    raise ValueError('Anchor metadata differs from manifest')
            for k in ('intrinsic','world_to_cam','anchor_pose_T_wv'):
                if not np.allclose(meta['A'][k],meta['B'][k],atol=1e-8,rtol=0):
                    raise ValueError(f'Pose/calibration differs: {pair}/{aid}/{k}')
            jobs.append((pair,aid,row.db_seg,paths))
    out.mkdir(parents=True)
    fontpath=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(fontpath),24) if fontpath.is_file() else ImageFont.load_default()
    rows=[]
    for pair,aid,segment,paths in jobs:
        images={}
        for v,path in paths.items():
            with Image.open(path) as image:
                if image.mode!='L': raise ValueError(f'Expected original grayscale PNG: {path}')
                images[v]=image.copy()
        if images['A'].size!=images['B'].size: raise ValueError('A/B image sizes differ')
        w,h=images['A'].size
        header,gap=100,16
        canvas=Image.new('RGB',(2*w+gap,h+header),'white')
        canvas.paste(images['A'],(0,header));canvas.paste(images['B'],(w+gap,header))
        draw=ImageDraw.Draw(canvas)
        draw.text((12,8),f'{pair}  /  anchor {aid}  /  identical position and view',font=font,fill='black')
        draw.text((12,53),'A: measured 16-line',font=font,fill='black')
        draw.text((w+gap+12,53),'B: measured16 + TULIP-generated points (32-line)',font=font,fill='black')
        name=f'{pair}__anchor_{aid:03d}__A16_vs_B32.png'
        canvas.save(out/name)
        # Also retain convenient byte-identical copies for opening separately.
        folder=out/f'{pair}__anchor_{aid:03d}'
        folder.mkdir()
        copied={}
        for v,label in [('A','A_measured16.png'),('B','B_original_tulip32.png')]:
            dest=folder/label;shutil.copy2(paths[v],dest)
            copied[v]=hashlib.sha256(dest.read_bytes()).hexdigest()
        rows.append(dict(pair_id=pair,segment=segment,anchor_id=aid,width=w,height=h,
            source_A=str(paths['A']),source_B=str(paths['B']),sha256_A=copied['A'],sha256_B=copied['B'],
            comparison=str(out/name)))
        print(out/name,flush=True)
    with (out/'sources.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'README.txt').write_text(
        '左: A 実測16ライン。右: B 観測16ライン＋TULIP生成点による32ライン相当。\n'
        'Bは最初のRetrieval評価に使用した通常画像です。観測部分保持方式ではありません。\n'
        '同一アンカー位置・向き・内部パラメータを検査。元PNGをリサイズ・明暗調整せず横並びにしています。\n'
        'Aは観測16ラインの参照画像であり、Bの復元先32ラインのGTではありません。\n'
        '各サブフォルダのA/Bは元ファイルと同じ内容のコピーです。\n',encoding='utf-8')
    print(f'DONE: {len(rows)} comparisons; sources.csv records original paths and hashes.')


if __name__=='__main__': main()
