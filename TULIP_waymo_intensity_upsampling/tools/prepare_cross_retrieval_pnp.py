#!/usr/bin/env python3
"""Build cross-drive Top-5 manifests; retain wrong-pair retrieval candidates."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    abc = root/'outputs/shimizu_protocol_507q_754db_abc'
    sources = {'A': abc/'retrieval_A_retry1/shimizu_507q_754db_A_topk_top50.jsonl',
               'B': abc/'retrieval_B/shimizu_507q_754db_B_topk_top50.jsonl',
               'C': root/'outputs/shimizu_retrieval_reproduction_epoch16_retry1/shimizu_test4_reproduced_topk_top50.jsonl'}
    items = {v:[json.loads(s) for s in path.read_text().splitlines()] for v,path in sources.items()}
    assert all(len(x)==507 for x in items.values())
    with Path('/mnt/w/32line/datasets/splits/cross/all/icp_startend_staticmaps.csv').open() as f:
        icp_rows=list(csv.DictReader(f))
    transforms, matrices = {}, []
    for r in icp_rows:
        a=math.radians(float(r['final_yaw_deg'])); c,s=math.cos(a),math.sin(a)
        T=np.eye(4); T[:2,:2]=[[c,-s],[s,c]]
        T[:3,3]=[float(r['final_'+k]) for k in ('tx','ty','tz')]
        transforms[r['db_seg'],r['q_seg']]=T
        matrices.append(dict(db_seg=r['db_seg'],q_seg=r['q_seg'],**{f'T{i}{j}':T[i,j] for i in range(4) for j in range(4)}))
    # Fixed, outcome-independent pilot: first and middle query of each pair.
    selected=[]
    for pair in sorted({x['pair_id'] for x in items['C']}):
        group=[x['q_idx'] for x in items['C'] if x['pair_id']==pair]
        selected.extend([group[0],group[len(group)//2]])
    args.output_dir.mkdir(parents=True,exist_ok=False)
    write_csv(args.output_dir/'icp_db_to_q.csv',matrices)
    for v, records in items.items():
        rows=[]
        for q,ref in zip(records,items['C']):
            assert q['q_idx']==ref['q_idx'] and q['q_cam_png']==ref['q_cam_png']
            qp=Path(q['q_cam_png']); qm=json.loads(qp.with_suffix('.json').read_text())
            qseg=qp.parent.parent.name
            assert len(q['topk'])>=5
            for candidate in q['topk'][:5]:
                ap=Path(candidate['anchor_png']); am=json.loads(ap.with_name('anchor.json').read_text())
                same=bool(candidate['is_same_pair'])
                if same:
                    T=transforms[am['segment_id'],qseg]
                    xy=(T@np.array(am['anchor_pose_T_wv']).reshape(4,4)[:,3])[:2]
                    dist=float(np.linalg.norm(xy-np.array(q['q_xy_m'])))
                    if abs(dist-float(candidate['dist_m']))>1e-3:
                        raise ValueError('ICP distance disagrees with retrieval')
                else:
                    dist=float('nan')
                rows.append(dict(pair_id=q['pair_id'],db_seg=am['segment_id'],q_seg=qseg,
                    q_frame_index=qm['frame_index'],anchor_id=am['submap_id'],
                    q_image_png=str(qp),q_meta_json=str(qp.with_suffix('.json')),
                    anchor_png=str(ap),anchor_json=str(ap.with_name('anchor.json')),
                    anchor_index_npz=str(ap.with_name('anchor_index.npz')),dist_xy_m=dist,
                    is_strict_gt=int(candidate['is_strict_gt']),
                    **{f'is_gt_t{t}m':int(same and candidate['soft_gt'][f't{t}m']) for t in (1,2,5,10)},
                    retrieval_q_idx=q['q_idx'],retrieval_rank=candidate['rank'],
                    retrieval_is_same_pair=int(same),error=''))
        write_csv(args.output_dir/f'{v}_top5_all507.csv',rows)
        pilot=[r for r in rows if r['retrieval_q_idx'] in selected]
        write_csv(args.output_dir/f'{v}_top5_pilot10.csv',pilot)
    (args.output_dir/'selection.json').write_text(json.dumps(dict(query_ids=selected,
        rule='first and middle Query per pair, independent of outcome',sources={v:str(p) for v,p in sources.items()},
        note='All Top-5 retained, including wrong-pair candidates. No GT filtering.'),indent=2)+'\n')
    print('Created 507 Query x 5 candidates and fixed pilot10 for A/B/C; ICP distances verified.')


if __name__=='__main__': main()
