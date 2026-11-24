"""Describe saved Strict Top-10 score changes; not causal attribution."""
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    base = Path(__file__).resolve().parents[1] / 'outputs'
    data = {}
    for v in 'AB':
        path = base / f'paperlike_retrieval_{v}_507q/paperlike_{v}_507q_topk_top50.jsonl'
        records = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        data[v] = {(r['pair_id'], r['q_cam_png']): r for r in records}
        assert len(records) == len(data[v]) == 507
    assert data['A'].keys() == data['B'].keys()
    rows = []
    for key, a in data['A'].items():
        b = data['B'][key]
        assert a['strict_gt']['db_index'] == b['strict_gt']['db_index']
        rec = dict(pair_id=key[0], q_cam_png=key[1], q_idx=a['q_idx'])
        for v, r in [('A', a), ('B', b)]:
            gt = r['strict_gt']
            competitors = sorted([c for c in r['topk'] if not c['is_strict_gt']], key=lambda c: c['rank'])
            assert len(competitors) >= 10
            rec.update({f'gt_score_{v}': gt['score'], f'gt_rank_{v}': gt['strict_rank'],
                        f'gt_image_{v}': gt['anchor_png'],
                        f'rival10_score_{v}': competitors[9]['score'],
                        f'top_competitor_score_{v}': competitors[0]['score'],
                        f'top_competitor_image_{v}': competitors[0]['anchor_png'],
                        f'top_competitor_distance_{v}': competitors[0].get('dist_m'),
                        f'margin_{v}': gt['score']-competitors[9]['score']})
        sa, sb = rec['gt_rank_A'] <= 10, rec['gt_rank_B'] <= 10
        rec['group'] = ('both_success' if sb else 'B_lost') if sa else ('B_gained' if sb else 'both_failure')
        rec['gt_score_delta'] = rec['gt_score_B']-rec['gt_score_A']
        rec['rival10_score_delta'] = rec['rival10_score_B']-rec['rival10_score_A']
        rec['margin_delta'] = rec['margin_B']-rec['margin_A']
        rec['rank_delta'] = rec['gt_rank_B']-rec['gt_rank_A']
        rows.append(rec)
    df = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    df.to_csv(args.output_dir/'per_query.csv', index=False)
    cols = ['gt_score_delta', 'rival10_score_delta', 'margin_delta', 'rank_delta']
    report = {}
    for name, g in df.groupby('group'):
        report[name] = dict(count=len(g), medians=g[cols].median().to_dict(),
            gt_down=int((g.gt_score_delta < 0).sum()),
            rival10_up=int((g.rival10_score_delta > 0).sum()),
            both_down=int(((g.gt_score_delta < 0)&(g.rival10_score_delta < 0)).sum()))
    (args.output_dir/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    df.groupby(['pair_id','group'])[cols].median().to_csv(args.output_dir/'by_scene.csv')
    df[df.group=='B_lost'].sort_values('margin_delta').head(10).to_csv(args.output_dir/'worst10.csv', index=False)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
