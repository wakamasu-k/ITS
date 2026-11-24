"""Stratified diagnostic sampling, not a causal test or model-selection procedure."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--per-scene', type=int, default=2)
    p.add_argument('--select-only', action='store_true')
    args = p.parse_args()
    assert args.per_scene > 0
    project = Path(__file__).resolve().parents[1]
    frames = {}
    for v in 'AB':
        frames[v] = pd.read_csv(project / f'outputs/paperlike_retrieval_{v}_507q/paperlike_{v}_507q_per_query.csv')
    keys = ['pair_id', 'q_cam_png']
    df = frames['A'].merge(frames['B'], on=keys, suffixes=('_A', '_B'), validate='one_to_one')
    assert len(df) == len(frames['A']) == len(frames['B']) == 507
    hit = lambda col: df[col].astype(str).str.lower().isin(['true', '1'])
    a, b = hit('strict_hit@10_A'), hit('strict_hit@10_B')
    df['group'] = 'both_failure'
    df.loc[a & ~b, 'group'] = 'B_lost'
    df.loc[~a & b, 'group'] = 'B_gained'
    df.loc[a & b, 'group'] = 'both_success'
    for v in 'AB':
        df[f'segment_{v}'] = df[f'gt_anchor_png_{v}'].map(lambda x: Path(x).parent.parent.name)
        df[f'anchor_{v}'] = df[f'gt_anchor_png_{v}'].map(lambda x: Path(x).parent.name)
    assert (df.segment_A == df.segment_B).all() and (df.anchor_A == df.anchor_B).all()
    df['rank_delta_B_minus_A'] = df.strict_rank_B - df.strict_rank_A
    samples = []
    for _, g in df.groupby(['group', 'pair_id'], sort=True):
        g = g.sort_values('q_cam_png').drop_duplicates(['segment_A', 'anchor_A'])
        samples.append(g.sample(n=min(args.per_scene, len(g)), random_state=20261021))
    chosen = pd.concat(samples).sort_values(['group', 'pair_id', 'q_cam_png'])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    df.to_csv(args.output_dir / 'all_query_groups.csv', index=False)
    chosen.to_csv(args.output_dir / 'selected_queries.csv', index=False)
    print('All queries grouped by Strict R@10:')
    print(df.groupby(['group', 'pair_id']).size().unstack(fill_value=0).to_string())
    print('Selected:', len(chosen), flush=True)
    if args.select_only:
        return
    cache, records = {}, []
    for _, r in chosen.iterrows():
        key = (r.segment_A, r.anchor_A)
        if key not in cache:
            path = args.output_dir / f'anchor_{len(cache):03d}.json'
            subprocess.run([sys.executable, str(project / 'tools/audit_anchor_pixel_provenance.py'),
                            '--segment', key[0], '--anchor', key[1], '--output', str(path)], check=True)
            cache[key] = json.loads(path.read_text())
        z = cache[key]
        records.append(dict(group=r.group, pair_id=r.pair_id, q_cam_png=r.q_cam_png,
            segment=key[0], anchor=key[1], rank_A=r.strict_rank_A, rank_B=r.strict_rank_B,
            rank_delta_B_minus_A=r.rank_delta_B_minus_A,
            replacement_percent=100*z['a_covered_now_generated']/z['a_valid'],
            newly_covered_percent_of_A=100*z['newly_covered']/z['a_valid'],
            same_point_changed_percent=(100*z['same_measured_point_png_changed']/z['same_measured_point_pixels']
                                        if z['same_measured_point_pixels'] else None),
            same_point_png_mae=z['same_measured_point_png_mae']))
    out = pd.DataFrame(records)
    out.to_csv(args.output_dir / 'pixel_comparison.csv', index=False)
    cols = ['replacement_percent', 'newly_covered_percent_of_A', 'same_point_changed_percent', 'same_point_png_mae', 'rank_delta_B_minus_A']
    stats = out.groupby('group')[cols].agg(['count', 'median', 'min', 'max'])
    stats.to_csv(args.output_dir / 'group_statistics.csv')
    out.groupby(['pair_id', 'group'])[cols].median().to_csv(args.output_dir / 'scene_group_medians.csv')
    print(stats.to_string())


if __name__ == '__main__':
    main()
