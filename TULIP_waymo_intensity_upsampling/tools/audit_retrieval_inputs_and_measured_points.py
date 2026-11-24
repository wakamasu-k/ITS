#!/usr/bin/env python3
"""Audit saved retrieval references and measured-point preservation, without changes."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def csv_rows(p):
    with p.open() as f: return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(); root=Path(__file__).resolve().parents[1]
    abc=root/'outputs/shimizu_protocol_507q_754db_abc'
    runs={'A':(abc/'retrieval_A_retry1','shimizu_507q_754db_A'),
          'B':(abc/'retrieval_B','shimizu_507q_754db_B'),
          'C':(root/'outputs/shimizu_retrieval_reproduction_epoch16_retry1','shimizu_test4_reproduced')}
    result={'retrieval':{},'maps':[],'limitations':[
        'Saved paths/indices verified, not historical image bytes or feature tensors.',
        'No inference, model, image, or evaluation settings changed.',
        'Measured-point comparison uses stored ordering and exact float equality.']}
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for v,(folder,name) in runs.items():
        db=csv_rows(abc/f'retrieval_manifests/{v}_db.csv')
        per=csv_rows(folder/f'{name}_per_query.csv')
        issues=[]; n=0
        for i,line in enumerate((folder/f'{name}_topk_top50.jsonl').open()):
            q=json.loads(line); r=per[i]
            if str(q['q_idx'])!=r['q_idx'] or q['q_cam_png']!=r['q_cam_png']:
                issues.append(f'Query row {i}')
            g=q['strict_gt']; idx=int(g['db_index'])
            if g['anchor_png']!=db[idx]['anchor_png'] or idx!=int(r['gt_index_db']) or g['anchor_png']!=r['gt_anchor_png']:
                issues.append(f'GT row {i}')
            if int(g['strict_rank'])!=int(r['strict_rank']) or abs(float(g['score'])-float(r['gt_score']))>1e-7:
                issues.append(f'GT score/rank {i}')
            for rank,c in enumerate(q['topk'],1):
                if c['anchor_png']!=db[int(c['db_index'])]['anchor_png'] or int(c['rank'])!=rank:
                    issues.append(f'Candidate {i}/{rank}')
                n+=1
            t=q['topk'][0]
            if int(t['db_index'])!=int(r['top1_index_db']) or t['anchor_png']!=r['top1_anchor_png'] or abs(float(t['score'])-float(r['top1_score']))>1e-7:
                issues.append(f'Top1 {i}')
        result['retrieval'][v]=dict(queries=len(per),db=len(db),candidate_references=n,issues=issues,
            saved_embedding_files=[str(x) for x in folder.glob('*embedding*.npy')])
        print(v,'retrieval issues',len(issues),flush=True)
    for ap in sorted((abc/'maps/A_observed16/training').glob('*/map_static.npz')):
        bp=abc/'maps/B_tulip32/training'/ap.parent.name/'map_static.npz'
        with np.load(ap,allow_pickle=False) as a,np.load(bp,allow_pickle=False) as b:
            ax,ai=a['xyz'],a['intensity']; bx,bi=b['xyz'],b['intensity']; generated=b['source_is_generated'].astype(bool)
            mx,mi=bx[~generated],bi[~generated]
            item=dict(segment=ap.parent.name,a_points=len(ax),b_points=len(bx),b_measured=len(mx),b_generated=int(generated.sum()),
                      measured_xyz_exact=bool(np.array_equal(ax,mx)),measured_intensity_exact=bool(np.array_equal(ai,mi)))
            result['maps'].append(item); print(item,flush=True)
        del ax,ai,bx,bi,mx,mi,generated
    (args.output_dir/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
