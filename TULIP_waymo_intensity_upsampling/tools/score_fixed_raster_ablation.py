"""Compare fixed GT/rival scores; not full-DB recall or ranking evaluation."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    ref = project.parent/'handover_shimizu/src/retrieval'
    sys.path.insert(0, str(ref))
    import evaluate as ev
    assert (args.images_root/'completed.json').exists(), 'Image preparation not complete'
    check = pd.read_csv(args.images_root/'baseline_check.csv')
    assert (check.changed_pixels == 0).all()
    images = pd.read_csv(args.images_root/'images.csv', dtype={'anchor': str})
    cases = pd.read_csv(project/'outputs/gt_vs_anchor102_audit_ax6mcoi2/query_gt_comparison.csv', dtype={'anchor': str})
    assert len(cases) == 14 and cases['query'].is_unique
    conditions = ['A_original','B_original','B_Aupper_post','A_Aupper_no_post','B_Aupper_no_post']
    conditions += [c for c in ['AB_holes_no_post','AB_holes_joint_post','AB_holes_final_png'] if c in set(images.condition)]
    conditions += sorted(c for c in set(images.condition) if c.startswith('probe_restore_'))
    lookup = {}
    for r in images.itertuples():
        key = (r.condition,r.segment,r.anchor)
        assert key not in lookup
        lookup[key] = r.image
        assert Path(r.image).is_file()
    saved = {}
    for v in 'AB':
        path = project/f'outputs/paperlike_retrieval_{v}_507q/paperlike_{v}_507q_topk_top50.jsonl'
        saved[v] = {r['q_idx']: r for r in map(json.loads, path.read_text().splitlines())}
    qpaths = []
    for r in cases.itertuples():
        a,b = saved['A'][r.query],saved['B'][r.query]
        assert a['q_cam_png'] == b['q_cam_png']
        for record in (a,b):
            gt = Path(record['strict_gt']['anchor_png'])
            assert gt.parent.name == r.anchor and gt.parent.parent.name == r.segment
        qpaths.append(a['q_cam_png'])
        for condition in conditions:
            for anchor in (r.anchor,'102'):
                assert (condition,r.segment,anchor) in lookup
    args.output_dir.mkdir(parents=True, exist_ok=False)
    ckpt = project.parent/'waymo_outputs/old32_retrieval/train_wdrive_paper80_b16/checkpoints/best.ckpt'
    config = SimpleNamespace(checkpoint=str(ckpt),backbone='resnet34',embed_dim=256,clusters=64,freeze_stages=2)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        model = ev._load_model(config)
    print(log.getvalue(), end='')
    (args.output_dir/'model_load.txt').write_text(log.getvalue())
    assert '[WARN]' not in log.getvalue(), 'Check model_load.txt before proceeding'
    model = model.to(args.device)
    stats_path = project/'configs/paper80_effective_modality_stats.json'
    stats = ev._load_modality_stats(str(stats_path))
    tq = ev._build_transform(512,864,stats['cam_mean'],stats['cam_std'])
    td = ev._build_transform(512,864,stats['anc_mean'],stats['anc_std'])
    qf = ev._embed_paths(model,qpaths,'cam',tq,args.device,16,2,True)
    paths = list(dict.fromkeys(lookup.values()))
    df = ev._embed_paths(model,paths,'int',td,args.device,16,2,True)
    feats = dict(zip(paths,df))
    rows = []
    for i,r in enumerate(cases.itertuples()):
        for condition in conditions:
            gt = float(qf[i] @ feats[lookup[condition,r.segment,r.anchor]])
            rival = float(qf[i] @ feats[lookup[condition,r.segment,'102']])
            row = dict(query=r.query,gt_anchor=r.anchor,rival_anchor=102,condition=condition,
                       gt_score=gt,rival_score=rival,margin=gt-rival)
            if condition in ('A_original','B_original'):
                record = saved[condition[0]][r.query]
                rivals = [x for x in record['topk'] if Path(x['anchor_png']).parent.name=='102'
                          and Path(x['anchor_png']).parent.parent.name==r.segment]
                assert len(rivals)==1
                row['saved_gt_score'] = record['strict_gt']['score']
                row['saved_rival_score'] = rivals[0]['score']
                row['baseline_max_abs_diff'] = max(abs(gt-row['saved_gt_score']),abs(rival-row['saved_rival_score']))
            rows.append(row)
    result = pd.DataFrame(rows)
    base = result[result.condition=='B_original'].set_index('query').margin
    result['margin_delta_vs_B_original'] = result.margin-result['query'].map(base)
    result.to_csv(args.output_dir/'scores.csv',index=False)
    maxdiff = float(result.baseline_max_abs_diff.max())
    print('Baseline max absolute score difference:',maxdiff)
    if maxdiff > 0.003:
        raise RuntimeError('Baseline scores differ >0.003; inspect scores.csv before interpreting conditions')
    summary = result.groupby('condition').agg(queries=('query','count'),gt_median=('gt_score','median'),
        rival_median=('rival_score','median'),margin_median=('margin','median'),
        margin_delta_median=('margin_delta_vs_B_original','median'),gt_beats_rival=('margin',lambda x:int((x>0).sum())))
    summary.to_csv(args.output_dir/'summary.csv')
    (args.output_dir/'completed.json').write_text(json.dumps(dict(checkpoint=str(ckpt),stats=stats,
        baseline_max_abs_diff=maxdiff,baseline_tolerance=0.003,images_root=str(args.images_root),
        evaluator_sha256=hashlib.sha256((ref/'evaluate.py').read_bytes()).hexdigest(),
        note='Fixed rival102, 14 queries. No full database ranking or recall. AMP batch changes can cause small baseline differences.'),indent=2))
    print(summary.round(6).to_string())


if __name__ == '__main__':
    main()
