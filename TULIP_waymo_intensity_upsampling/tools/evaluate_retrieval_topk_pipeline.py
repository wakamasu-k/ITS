#!/usr/bin/env python3
"""Prepare/run/summarize unfiltered Retrieval Top-K -> LoFTR/PnP.

Uses fixed local checkpoints. This is an integrated evaluation of the thesis
method, not a claim that the thesis published an identical integrated experiment.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

REF = Path('/home/wakamatsu/ITS/handover_shimizu/src/LoFTR')
PYTHON = '/home/wakamatsu/ITS/.venv_tulip/bin/python'
CKPT = '/home/wakamatsu/ITS/waymo_outputs/old32_dense/loftr_retrain_stage1_paper_wdrive/stage1_final_gstep715958.pth'
ICP = '/mnt/w/32line/datasets/splits/cross/all/icp_startend_staticmaps.csv'


def flag(x):
    return str(x).lower() in ('true', '1')


def number(x, fallback=float('inf')):
    try:
        v = float(x)
        return v if math.isfinite(v) else fallback
    except (ValueError, TypeError):
        return fallback


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def key(row):
    return (row['q_seg'], int(row['q_frame_index']), row['db_seg'], int(row['anchor_id']))


def selection_key(row):
    # Thesis: maximum RANSAC inliers. Ties follow the inherited evaluator,
    # then Retrieval rank to make fully tied candidates deterministic.
    return (-int(flag(row['pnp_success'])), -number(row.get('n_inlier'), 0),
            number(row.get('reproj_med')), -number(row.get('n_3d'), 0),
            int(row['retrieval_rank']))


def candidate_manifest(topk, k, expected_queries):
    raw = topk.read_bytes()
    records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(records) != expected_queries or len({r['q_idx'] for r in records}) != expected_queries:
        raise ValueError(f'Expected {expected_queries} unique Retrieval queries')
    cache = {}
    checked = set()

    def metadata(path):
        if path not in cache:
            cache[path] = json.loads(path.read_text())
        return cache[path]

    rows = []
    for q in records:
        qp = Path(q['q_cam_png'])
        qm = metadata(qp.with_suffix('.json'))
        candidates = q['topk'][:k]
        if len(candidates) != k:
            raise ValueError('Not enough saved candidates')
        for rank, candidate in enumerate(candidates, 1):
            if int(candidate['rank']) != rank:
                raise ValueError('Unexpected ranking order')
            ap = Path(candidate['anchor_png'])
            am = metadata(ap.with_name('anchor.json'))
            paths = dict(q_image_png=qp, q_meta_json=qp.with_suffix('.json'),
                         anchor_png=ap, anchor_json=ap.with_name('anchor.json'),
                         anchor_index_npz=ap.with_name('anchor_index.npz'))
            for path in paths.values():
                if path not in checked:
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    checked.add(path)
            rows.append(dict(pair_id=q['pair_id'], q_seg=qp.parent.parent.name,
                db_seg=am['segment_id'], q_frame_index=int(qm['frame_index']),
                anchor_id=int(am['submap_id']), **{k: str(v) for k, v in paths.items()},
                dist_xy_m=candidate.get('dist_m') if candidate.get('dist_m') is not None else '',
                is_strict_gt=int(flag(candidate['is_strict_gt'])),
                **{f'is_gt_t{t}m': int(flag(candidate['soft_gt'][f't{t}m'])) for t in (1, 2, 5, 10)},
                retrieval_q_idx=int(q['q_idx']), retrieval_rank=rank,
                retrieval_is_same_pair=int(flag(candidate['is_same_pair'])), error=''))
    if len({key(r) for r in rows}) != len(rows):
        raise ValueError('Duplicate query/candidate keys')
    return raw, records, rows


def prepare(args):
    if args.root.exists():
        raise FileExistsError(f'Refusing to overwrite {args.root}')
    raw, records, rows = candidate_manifest(args.topk, args.k, 507)
    pilot_ids = []
    for pair in sorted({q['pair_id'] for q in records}):
        group = [q['q_idx'] for q in records if q['pair_id'] == pair]
        pilot_ids.extend([group[0], group[len(group)//2]])
    if len(set(pilot_ids)) != 10:
        raise ValueError('Expected five pairs and ten pilot queries')
    args.root.mkdir(parents=True)
    write_csv(args.root/'all507.csv', rows)
    write_csv(args.root/'pilot10.csv', [r for r in rows if r['retrieval_q_idx'] in pilot_ids])
    protocol = dict(topk_source=str(args.topk.resolve()), topk_sha256=hashlib.sha256(raw).hexdigest(),
        k=args.k, queries=507, pairs=len(rows), pilot_query_ids=pilot_ids,
        candidate_filter='none: retain wrong-pair and distant candidates',
        selection='PnP success, maximum inliers; ties: reproj_med, n_3d, Retrieval rank',
        precision='fp16 CUDA', checkpoint=CKPT, icp=ICP,
        location_success_denominator='all queries including failed/wrong-pair selections',
        median_denominator='selected PnP-successful same-pair, comparable, finite-error queries only',
        thesis='K=10 section 3.4.5; max-inliers section 2.1.2.3; section 3.5 evaluates isolated pairs')
    (args.root/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print(json.dumps(protocol, indent=2))


def summarize(root, scope):
    manifest = read_csv(root/f'{scope}.csv')
    output = root/f'{scope}_pnp'
    results = read_csv(output/'per_pair.csv')
    lookup = {key(r): r for r in manifest}
    if len(results) != len(manifest) or {key(r) for r in results} != set(lookup):
        raise ValueError('Missing/duplicate candidate results; cannot report a complete evaluation')
    if any(r.get('error', '').strip() for r in results):
        raise ValueError('Candidate execution errors found; inspect per_pair.csv before summarizing')
    groups = {}
    for r in results:
        m = lookup[key(r)]
        r = dict(r, retrieval_rank=m['retrieval_rank'],
                 retrieval_q_idx=m['retrieval_q_idx'],
                 retrieval_is_same_pair=m['retrieval_is_same_pair'])
        groups.setdefault(int(m['retrieval_q_idx']), []).append(r)
    details = []
    errors = []
    for qi, group in sorted(groups.items()):
        chosen = min(group, key=selection_key)
        solved = flag(chosen['pnp_success'])
        comparable = flag(chosen.get('comparable'))
        same_pair = flag(chosen['retrieval_is_same_pair'])
        error = number(chosen.get('t_err_xy'))
        valid = solved and comparable and same_pair and math.isfinite(error)
        if valid:
            errors.append(error)
        details.append(dict(query_index=qi, pair_id=chosen['pair_id'],
            q_seg=chosen['q_seg'], q_frame_index=chosen['q_frame_index'],
            selected_rank=chosen['retrieval_rank'] if solved else '',
            selected_db_seg=chosen['db_seg'] if solved else '',
            selected_anchor_id=chosen['anchor_id'] if solved else '',
            pnp_returned=solved, all_candidates_failed=not solved,
            selected_wrong_pair=solved and not same_pair,
            selected_comparable=comparable if solved else False,
            xy_error_m=error if valid else '',
            **{f'success_xy_{t}m': valid and error <= t for t in (.5, 1, 2, 5, 10)}))
    n = len(details)
    report = dict(queries=n, candidate_pairs=len(results),
        pnp_returned=sum(r['pnp_returned'] for r in details),
        all_candidates_failed=sum(r['all_candidates_failed'] for r in details),
        selected_wrong_pair=sum(r['selected_wrong_pair'] for r in details),
        finite_same_pair_pose_count=len(errors),
        xy_median_success_only=statistics.median(errors) if errors else None,
        location_success_denominator=n,
        **{f'success_xy_{t}m_all_queries': sum(r[f'success_xy_{t}m'] for r in details)/n
           for t in (.5, 1, 2, 5, 10)},
        note='Wrong-pair selections count as location failures; no cross-scene metric error is invented. No GT used to select candidates.')
    # Exclusive creation protects a previously reviewed summary.
    write_csv(output/'integrated_per_query.csv', details)
    with (output/'integrated_summary.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(report, indent=2, allow_nan=False))


def show_completed(root, scope):
    """Display an existing complete report without running inference or writing."""
    out = root/f'{scope}_pnp'
    report_path = out/'integrated_summary.json'
    if not report_path.is_file():
        return False
    required = [root/f'{scope}.csv', out/'per_pair.csv', out/'integrated_per_query.csv']
    if not all(path.is_file() for path in required):
        raise SystemExit(f'Incomplete saved report: {out}. Files have been preserved.')
    manifest, pairs, selected = [read_csv(path) for path in required]
    report = json.loads(report_path.read_text())
    expected_queries = {int(r['retrieval_q_idx']) for r in manifest}
    if (len(pairs) != len(manifest)
            or len({key(r) for r in pairs}) != len(pairs)
            or {key(r) for r in pairs} != {key(r) for r in manifest}
            or len(selected) != len(expected_queries)
            or {int(r['query_index']) for r in selected} != expected_queries
            or report.get('queries') != len(expected_queries)
            or report.get('candidate_pairs') != len(manifest)):
        raise SystemExit(f'Saved report does not match the manifest: {out}. Files have been preserved.')
    print(f'[ALREADY COMPLETED] Showing saved results; inference was not rerun: {report_path}')
    print(json.dumps(report, indent=2, allow_nan=False))
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run', 'summarize'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--topk', type=Path)
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--scope', choices=['pilot10', 'all507'], default='pilot10')
    args = p.parse_args()
    if args.action == 'prepare':
        if args.topk is None or not 1 <= args.k <= 50:
            p.error('prepare requires --topk and K in [1,50]')
        prepare(args)
    elif args.action == 'summarize':
        if not show_completed(args.root, args.scope):
            summarize(args.root, args.scope)
    else:
        manifest = args.root/f'{args.scope}.csv'
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        out = args.root/f'{args.scope}_pnp'
        if show_completed(args.root, args.scope):
            return
        if out.exists():
            raise SystemExit(
                f'Output directory already exists but has no complete integrated report: {out}\n'
                'If per_pair.csv is complete, use the summarize action. Otherwise inspect the existing run; '
                'use a new --root for a fresh evaluation. Files have been preserved.')
        subprocess.run([PYTHON, '-c', 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'], check=True)
        out.mkdir(exist_ok=False)
        cmd = [PYTHON, '-u', str(REF/'tools/eval_cross_match_to_pose_zalign_multi_visall_v2.py'),
            '--manifest', str(manifest.resolve()), '--ckpt', CKPT, '--out-dir', str(out.resolve()),
            '--device', 'cuda', '--amp-dtype', 'fp16', '--gt-type', 'all',
            '--z-align', 'none', '--icp-csv', ICP, '--no-write-per-radius']
        (out/'command.json').write_text(json.dumps(cmd, indent=2)+'\n')
        env = dict(os.environ, PYTHONPATH=str(REF/'src'), PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        subprocess.run(cmd, env=env, check=True)
        summarize(args.root, args.scope)


if __name__ == '__main__':
    main()
