#!/usr/bin/env python3
"""Summarize unfiltered Top-5 PnP selections, never GT-filtered summaries."""
import argparse
import csv
import json
import math
from statistics import median


def read(path):
    with path.open() as f: return list(csv.DictReader(f))


def flag(value): return str(value).lower() in ('true','1')
def num(r,k):
    try: return float(r[k])
    except (KeyError,ValueError): return float('nan')


def main():
    from pathlib import Path
    p=argparse.ArgumentParser(); p.add_argument('--run-root',type=Path,required=True)
    args=p.parse_args(); root=args.run_root
    details=[]; summary={}
    for v in 'ABC':
        manifest=read(root/f'{v}_top5_pilot10.csv')
        results=read(root/f'{v}_pilot10_pnp/per_query.csv')
        pairs=read(root/f'{v}_pilot10_pnp/per_pair.csv')
        assert len(manifest)==50 and len(pairs)==50 and len(results)==10
        errors=[]; rotations=[]; out=[]
        for r in results:
            candidates=[m for m in manifest if m['q_seg']==r['q_seg'] and m['q_frame_index']==r['q_frame_index']]
            assert len(candidates)==5
            chosen=[m for m in candidates if m['db_seg']==r['db_seg'] and m['anchor_id']==r['anchor_id']]
            assert len(chosen)==1
            e=num(r,'t_err_xy'); rotation=num(r,'rot_err_deg')
            comparable=flag(r.get('comparable',False)); solved=flag(r['pnp_success'])
            valid=comparable and solved and math.isfinite(e)
            if valid: errors.append(e); rotations.append(rotation)
            pose_pairs=[a for a in pairs if a['q_seg']==r['q_seg'] and a['q_frame_index']==r['q_frame_index']]
            best_possible=min([num(a,'t_err_xy') for a in pose_pairs if flag(a.get('comparable',False)) and flag(a['pnp_success']) and math.isfinite(num(a,'t_err_xy'))], default=float('nan'))
            row=dict(variant=v,query_index=chosen[0]['retrieval_q_idx'],pair_id=r['pair_id'],
                selected_rank=chosen[0]['retrieval_rank'],pnp_returned=solved,comparable=comparable,
                top5_has_2m_gt=any(flag(m['is_gt_t2m']) for m in candidates),
                xy_error_m=e if valid else None,rotation_error_deg=rotation if valid else None,
                n_matches=r.get('n_match_in',''),n_3d=r.get('n_3d',''),n_inliers=r.get('n_inlier',''),
                success_xy2m=bool(valid and e<=2),success_xy2m_rot5deg=bool(valid and e<=2 and rotation<=5),
                success_xy5m=bool(valid and e<=5),
                diagnostic_best_candidate_xy_m=best_possible if math.isfinite(best_possible) else None)
            out.append(row); details.append(row)
        summary[v]=dict(queries=10,candidate_pairs=50,pnp_returned=sum(r['pnp_returned'] for r in out),
             comparable_solutions=len(errors),xy_median_m=median(errors) if errors else None,
             rotation_median_deg=median(rotations) if rotations else None,
             success_xy2m=sum(r['success_xy2m'] for r in out),
             success_xy2m_rot5deg=sum(r['success_xy2m_rot5deg'] for r in out),
             success_xy5m=sum(r['success_xy5m'] for r in out),
             top5_has_2m_gt=sum(r['top5_has_2m_gt'] for r in out),
             fails_xy2m_with_2m_candidate=sum(r['top5_has_2m_gt'] and not r['success_xy2m'] for r in out),
             fails_xy2m_without_2m_candidate=sum(not r['top5_has_2m_gt'] and not r['success_xy2m'] for r in out),
             pair_exception_count=sum(bool(r.get('error','')) for r in pairs))
    (root/'pilot_comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    with (root/'pilot_query_details.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(details[0])); w.writeheader(); w.writerows(details)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
