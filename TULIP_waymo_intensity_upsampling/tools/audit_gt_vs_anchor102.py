"""Inspect fixed GT anchors for queries 40..53 and rival 102; no inference."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd
from audit_abc_anchor_contents import load, compare


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    base = project/'outputs/shimizu_protocol_507q_754db_abc'
    tables = {v: pd.read_csv(base/f'retrieval_manifests/{v}_db.csv') for v in 'ABC'}
    records = [json.loads(s) for s in (project/'outputs/paperlike_retrieval_A_507q/paperlike_A_507q_topk_top50.jsonl').read_text().splitlines() if s.strip()]
    targets = {}
    segment = 'segment-6410495600874495447_5287_500_5307_500_with_camera_labels'
    for q in records:
        if 40 <= int(q['q_idx']) <= 53:
            path = Path(q['strict_gt']['anchor_png'])
            assert path.parent.parent.name == segment
            targets.setdefault(int(path.parent.name), []).append(int(q['q_idx']))
    assert sum(map(len, targets.values())) == 14 and 102 not in targets
    targets[102] = []
    items = list(targets.items())
    if args.limit:
        items = items[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for anchor, queries in items:
        report = args.output_dir/f'anchor_{anchor}_provenance.json'
        subprocess.run([sys.executable, str(project/'tools/audit_anchor_pixel_provenance.py'), '--segment', segment,
                        '--anchor', str(anchor), '--output', str(report)], check=True)
        z = json.loads(report.read_text())
        loaded = {}
        for v in 'ABC':
            t = tables[v]
            match = t[(t.db_seg == segment)&(t.anchor_id == anchor)]
            assert len(match) == 1
            loaded[v] = load(match.iloc[0].to_dict())
        common = loaded['A'][2]&loaded['B'][2]&loaded['C'][2]
        row = dict(anchor=anchor, role='rival102' if anchor==102 else 'GT', queries=str(queries),
                   replacement_percent=100*z['a_covered_now_generated']/z['a_valid'],
                   same_point_png_mae=z['same_measured_point_png_mae'],
                   new_pixels=z['newly_covered'], common_ABC_pixels=int(common.sum()))
        for v in 'AB':
            # Identical valid pixel population for A-C and B-C comparisons.
            result = compare(loaded[v][0], loaded[v][1], common,
                             loaded['C'][0], loaded['C'][1], common)
            for field in ['depth_mae_m','depth_p95_m','png_mae_common']:
                row[f'{v}_vs_C_{field}'] = result[field]
        rows.append(row)
        print(f'Completed anchor {anchor}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir/'comparison.csv', index=False)
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
