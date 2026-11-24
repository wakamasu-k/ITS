#!/usr/bin/env python3
"""Compare saved retrieval outcomes without changing or rerunning retrieval."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def hit(row, key):
    value = row[key].lower()
    if value not in ('true', 'false'):
        raise ValueError(f'Invalid hit flag: {value}')
    return value == 'true'


def transition(rows, baseline, key):
    counts = dict(both_success=0, both_failure=0, B_lost=0, B_gained=0)
    for r in rows:
        old, new = hit(r[baseline], key), hit(r['B'], key)
        name = ('both_success' if new else 'B_lost') if old else ('B_gained' if new else 'both_failure')
        counts[name] += 1
    return counts


def stats(rows):
    out = {}
    for v in 'ABC':
        out[v] = dict(query_count=len(rows), strict_median_rank=median(int(r[v]['strict_rank']) for r in rows),
                      soft2_median_rank=median(float(r[v]['soft_t2m_best_rank']) for r in rows),
                      top1_other_pair=sum(not hit(r[v], 'top1_is_same_pair') for r in rows))
        for key in ('strict_hit@1', 'strict_hit@5', 'strict_hit@10', 'soft_t2m_hit@1', 'soft_t2m_hit@5', 'soft_t2m_hit@10'):
            n = sum(hit(r[v], key) for r in rows)
            out[v][key] = dict(success=n, failure=len(rows)-n, percent=100*n/len(rows))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for v in 'ABC':
        p.add_argument('--' + v.lower(), type=Path, required=True)
    p.add_argument('--db', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    tables = {v: read(getattr(args, v.lower())) for v in 'ABC'}
    db = read(args.db)
    if len(db) != 754 or any(len(t) != 507 for t in tables.values()):
        raise ValueError('Expected 507 Query / 754 DB')
    identities = ('q_idx', 'pair_id', 'q_cam_png', 'gt_index_db', 'q_x_m', 'q_y_m', 'gt_dist_m',
                  'soft_t1m_gt_size', 'soft_t2m_gt_size', 'soft_t5m_gt_size', 'soft_t10m_gt_size')
    rows = []
    for i in range(507):
        r = {v: tables[v][i] for v in 'ABC'}
        if any(any(r[v][k] != r['C'][k] for k in identities) for v in 'AB'):
            raise ValueError(f'Query/GT alignment mismatch at {i}')
        for v in 'ABC':
            if int(r[v]['soft_t2m_gt_size']) <= 0:
                raise ValueError('Empty soft GT')
            for k in (1,5,10):
                if hit(r[v], f'strict_hit@{k}') != (int(r[v]['strict_rank']) <= k):
                    raise ValueError('Rank/hit inconsistency')
        rows.append(r)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = dict(mode='Saved-result analysis; no new inference or ranking changes', query_count=507, db_count=754,
                   sources={v:str(getattr(args,v.lower()).resolve()) for v in 'ABC'}, overall=stats(rows),
                   by_pair={pair:stats([r for r in rows if r['C']['pair_id']==pair]) for pair in sorted({r['C']['pair_id'] for r in rows})},
                   transitions={base:{k:transition(rows,base,k) for k in ('strict_hit@5','soft_t2m_hit@5')} for base in ('A','C')})
    flat = []
    for r in rows:
        entry = {k:r['C'][k] for k in ('q_idx','pair_id','q_cam_png','gt_index_db')}
        for v in 'ABC':
            for k in ('strict_rank','soft_t2m_best_rank','strict_hit@5','soft_t2m_hit@5','gt_anchor_png','top1_anchor_png','top1_dist_m'):
                entry[v+'_'+k] = r[v][k]
        entry['B_minus_C_strict_rank'] = int(r['B']['strict_rank'])-int(r['C']['strict_rank'])
        flat.append(entry)
    perdb = []
    for i, d in enumerate(db):
        subset = [r for r in rows if int(r['C']['gt_index_db'])==i]
        entry = dict(db_index=i, pair_id=d['pair_id'], anchor_id=d['anchor_id'], reference_png=d['anchor_png'], strict_gt_query_count=len(subset))
        for v in 'ABC':
            entry[v+'_strict_top5_success'] = sum(hit(r[v],'strict_hit@5') for r in subset)
            entry[v+'_top1_selected_count'] = sum(int(r[v]['top1_index_db'])==i for r in rows)
        perdb.append(entry)
    for name, records in [('per_query',flat),('per_db_754',perdb),('largest_rank_regressions',sorted(flat,key=lambda x:x['B_minus_C_strict_rank'],reverse=True)[:20])]:
        with (args.output_dir/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    summary['db_anchors_with_strict_gt_queries'] = sum(r['strict_gt_query_count']>0 for r in perdb)
    (args.output_dir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('overall','transitions','db_anchors_with_strict_gt_queries')},indent=2))


if __name__ == '__main__':
    main()
