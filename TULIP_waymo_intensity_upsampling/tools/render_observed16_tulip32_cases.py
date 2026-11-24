"""Show original anchor rasters side by side, preserving their pixel values.

Default: measured16 (A) versus measured16 + generated points (B), in four
segments other than the previously inspected 641049 scene. This compares
anchor-view images, not native LiDAR range images or generated-only points.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
PREVIOUS = 'segment-6410495600874495447_5287_500_5307_500_with_camera_labels'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--segment',action='append',help='Full segment name or unique initial number, e.g. 848780')
    p.add_argument('--anchor-ids',type=int,nargs='+',help='Explicit anchor IDs in each selected segment')
    p.add_argument('--samples-per-scene',type=int,default=2)
    p.add_argument('--include-c',action='store_true',help='Add observed32 reference as a third panel')
    args=p.parse_args()
    if args.samples_per_scene<1: p.error('--samples-per-scene must be positive')
    base=PROJECT/'outputs/shimizu_protocol_507q_754db_abc/retrieval_manifests'
    variants=['A','B']+(['C'] if args.include_c else [])
    tables={v:pd.read_csv(base/f'{v}_db.csv') for v in variants}
    keys=['pair_id','db_seg','anchor_id']
    for v,table in tables.items():
        if table.duplicated(keys).any(): raise ValueError(f'Duplicate anchors in {v}')
    joined=tables['A'][keys+['anchor_png']].rename(columns={'anchor_png':'A'})
    for v in variants[1:]:
        joined=joined.merge(tables[v][keys+['anchor_png']].rename(columns={'anchor_png':v}),
                            on=keys,validate='one_to_one',how='outer')
    if joined[variants].isna().any().any(): raise ValueError('Variant anchor sets differ')
    segments=sorted(joined.db_seg.unique())
    if args.segment:
        selected=[]
        for token in args.segment:
            matches=[s for s in segments if s==token or s.startswith('segment-'+token)]
            if len(matches)!=1: raise ValueError(f'Segment must identify one scene: {token}: {matches}')
            if matches[0] not in selected: selected.append(matches[0])
    else: selected=[s for s in segments if s!=PREVIOUS]
    cases=[]
    for segment in selected:
        group=joined[joined.db_seg==segment].sort_values('anchor_id').reset_index(drop=True)
        if args.anchor_ids:
            wanted=list(dict.fromkeys(args.anchor_ids))
            missing=set(wanted)-set(group.anchor_id)
            if missing: raise ValueError(f'Anchor IDs {sorted(missing)} do not exist in {segment}')
            group=group[group.anchor_id.isin(wanted)]
        else:
            count=min(args.samples_per_scene,len(group))
            # Uniform interior samples rather than examples chosen by score.
            positions=np.linspace(0,len(group)-1,count+2)[1:-1].round().astype(int)
            group=group.iloc[positions]
        cases.extend(group.to_dict('records'))
    args.output_dir=args.output_dir.resolve()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    font_path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(font_path),21) if font_path.is_file() else ImageFont.load_default()
    labels={'A':'A: observed16 (measured points)',
            'B':'B: measured16 + TULIP-generated points (32-line equivalent)',
            'C':'C: observed32 reference'}
    report=[];header=76
    for row in cases:
        images={};metadata={}
        for v in variants:
            path=Path(row[v])
            with Image.open(path) as image: images[v]=image.convert('RGB')
            metadata[v]=json.loads(path.with_name('anchor.json').read_text())
        for v in variants[1:]:
            if images[v].size!=images['A'].size: raise ValueError('Image sizes differ')
            for key in ('world_to_cam','intrinsic','anchor_pose_T_wv'):
                if not np.allclose(metadata['A'][key],metadata[v][key],atol=1e-8,rtol=0):
                    raise ValueError(f'Anchor camera differs: {row["db_seg"]}/{row["anchor_id"]}: {key}')
        w,h=images['A'].size
        canvas=Image.new('RGB',(w*len(variants),h+header),'white')
        draw=ImageDraw.Draw(canvas)
        anchor=int(row['anchor_id']);segment=row['db_seg']
        for i,v in enumerate(variants):
            canvas.paste(images[v],(i*w,header))
            draw.text((i*w+12,8),labels[v],fill='black',font=font)
            draw.text((i*w+12,39),f'{segment} / anchor {anchor}',fill='black',font=font)
            # Confirm no changes to the original image pixels, apart from RGB expansion.
            if not np.array_equal(np.asarray(canvas.crop((i*w,header,(i+1)*w,header+h))),np.asarray(images[v])):
                raise AssertionError('Source image pixels changed')
        name=f'{segment}_anchor_{anchor:03d}_'+('_'.join(variants))+'.png'
        target=args.output_dir/name
        canvas.save(target)
        item=dict(segment=segment,anchor_id=anchor,comparison=str(target),
                  same_pose_and_intrinsics=True,source_pixels_unchanged=True)
        for v in variants:
            item[f'{v}_source']=row[v]
            item[f'{v}_sha256']=hashlib.sha256(Path(row[v]).read_bytes()).hexdigest()
        report.append(item)
        print('Wrote:',target,flush=True)
    with (args.output_dir/'images.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(report[0]));writer.writeheader();writer.writerows(report)
    with (args.output_dir/'README.txt').open('x') as f:
        f.write('Left=A measured16; center/right=B measured16 plus TULIP-generated points.\n'
                'Optional third panel=C measured32 reference. A is the measured baseline, not the 32-line target.\n'
                'These are saved Retrieval anchor rasters after their original postprocessing.\n'
                'No new intensity normalization, resizing, cropping or contrast adjustment was applied.\n'
                'The source anchor poses and camera intrinsics were checked.\n'
                'Default samples are uniform interior anchor IDs in scenes other than 641049.\n')
    print(f'DONE: {len(report)} comparisons in {len(selected)} scenes. Sources: images.csv')


if __name__=='__main__': main()
