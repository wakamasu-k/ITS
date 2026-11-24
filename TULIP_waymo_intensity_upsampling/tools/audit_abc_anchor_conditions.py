#!/usr/bin/env python3
"""Read-only A/B/C manifest and anchor metadata audit; writes new reports only."""
import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path


def same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        return math.isclose(a, b, rel_tol=0, abs_tol=1e-8)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    # Refuse overwriting a previous audit.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tables = {}
    issues = []
    for kind in ('db', 'pairs'):
        for v in 'ABC':
            with (args.manifest_dir / f'{v}_{kind}.csv').open(newline='') as f:
                tables[v, kind] = list(csv.DictReader(f))
        reference = tables['C', kind]
        for v in 'AB':
            rows = tables[v, kind]
            if len(rows) != len(reference):
                issues.append(dict(variant=v, table=kind, field='row_count'))
            for i, (row, ref) in enumerate(zip(rows, reference)):
                for key in row.keys() | ref.keys():
                    if key != 'anchor_png' and row.get(key) != ref.get(key):
                        issues.append(dict(variant=v, table=kind, row=i, field=key))
    fields = ('segment_id', 'subset', 'submap_id', 'anchor_frame_index',
              'anchor_pose_xyyaw', 'anchor_pose_T_wv', 'camera_name',
              'width', 'height', 'intrinsic', 'world_to_cam')
    details = []
    counts = Counter()
    for i, refrow in enumerate(tables['C', 'db']):
        metadata = {}
        for v in 'ABC':
            if i >= len(tables[v, 'db']):
                continue
            row = tables[v, 'db'][i]
            folder = Path(row['anchor_png']).parent
            for name in ('anchor.png', 'anchor.json', 'anchor_index.npz'):
                if not (folder / name).is_file():
                    issues.append(dict(variant=v, row=i, field='missing_file', path=str(folder / name)))
            try:
                metadata[v] = json.loads((folder / 'anchor.json').read_text())
            except (OSError, ValueError) as e:
                issues.append(dict(variant=v, row=i, field='metadata_read', error=str(e)))
        if 'C' not in metadata:
            continue
        for v in 'AB':
            if v not in metadata:
                continue
            a, c = metadata[v], metadata['C']
            mismatches = [k for k in fields if k not in a or k not in c or not same(a[k], c[k])]
            ap, cp = dict(a.get('render_profile', {})), dict(c.get('render_profile', {}))
            au, cu = ap.pop('intensity_up_val', None), cp.pop('intensity_up_val', None)
            if not ap or not cp or not same(ap, cp):
                mismatches.append('render_profile_except_intensity_up_val')
            for k in mismatches:
                issues.append(dict(variant=v, row=i, field=k))
            counts[v + '_checked'] += 1
            counts[v + '_matched'] += not mismatches
            counts[v + '_intensity_up_diff'] += not same(au, cu)
            details.append(dict(variant=v, row=i, segment=refrow['db_seg'],
                                anchor_id=refrow['anchor_id'], mismatches=mismatches,
                                intensity_up=au, reference_intensity_up=cu,
                                coverage=a.get('coverage'), reference_coverage=c.get('coverage')))
        if (i + 1) % 100 == 0:
            print(f'Checked {i + 1} anchors', flush=True)
    summary = dict(created=datetime.now().isoformat(), manifest_dir=str(args.manifest_dir.resolve()),
                   rows={f'{v}_{k}': len(t) for (v, k), t in tables.items()},
                   counts=dict(counts), issue_count=len(issues), issues=issues,
                   limitations=['Metadata and file existence only; NPZ contents and PNG sizes not decoded.',
                                'intensity_up_val is reported separately as a data-dependent parameter.',
                                'Does not prove equivalence of map construction, ICP, or model training.'])
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (args.output_dir / 'per_anchor.json').write_text(json.dumps(details, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
