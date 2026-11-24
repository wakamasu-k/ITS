"""Read-only progress report; never runs training/inference or changes files."""
import argparse
import csv
import json
import math
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1] /
                        'outputs/observed8_localization_20261024')
    args = parser.parse_args()
    root = args.root.resolve()
    print('保存先:', root)
    plan = root / 'plan.json'
    if not plan.is_file():
        print('実験計画: なし')
        return
    jobs = load(plan)['jobs']
    print('\n【地図生成：ファイルの存在確認】')
    for role in ('train', 'val', 'test'):
        group = [j for j in jobs if j['role'] == role]
        count = sum((root / 'data' / role / 'maps' / j['subset'] / j['segment'] /
                     'map_static.npz').is_file() for j in group)
        print(f'{role}: {count}/{len(group)} セグメント')
    print('\n【画像・学習一覧の準備記録】')
    for name in ('test_manifests', 'manifests'):
        path = root / name / 'completed.json'
        print(name + ':', load(path) if path.is_file() else '完了記録なし')
    print('正規化統計:', 'あり' if (root / 'modality_stats.json').is_file() else 'なし')
    print('※完了記録は準備時の検査結果です。全画像の現在の破損検査は行いません。')

    print('\n【Retrieval学習】')
    metrics = root / 'retrieval_train/logs/train_metrics.csv'
    invalid_soft = False
    if metrics.is_file():
        with metrics.open(newline='') as f:
            rows = list(csv.DictReader(f))
        valid_rows = [r for r in rows if r.get('epoch')]
        if valid_rows:
            epochs = {int(float(r['epoch'])) for r in valid_rows}
            print(f'記録済み: {len(epochs)} epoch、最新: {max(epochs)}/80')
            last = max(valid_rows, key=lambda r: int(float(r['epoch'])))
            print('最新の検証Strict R@10:', last.get('val1/R@10'))
            key = 'val1/Soft_t5m_R@10'
            values = [float(r[key]) for r in valid_rows if r.get(key)]
            invalid_soft = bool(values) and all(v == 0 for v in values)
            print('検証Soft(5m) R@10の最大:', max(values) if values else '記録なし')
            if invalid_soft:
                print('要修正: 全epochでSoftが0です。この指標による重み選択は未検証です。')
        else:
            print('epoch記録なし')
    else:
        print('学習ログなし')
    selected = root / 'selected_retrieval.json'
    print('\n【重み選択】')
    if selected.is_file():
        d = load(selected)
        print('選択epoch:', d.get('epoch'), '指標:', d.get('metric'), '値:', d.get('value'))
        print('重みファイル:', d.get('checkpoint'))
        if invalid_soft:
            print('注意: 全epoch同点のためepoch 1を選んだ可能性があります。最良重みとは判断できません。')
    else:
        print('選択記録なし')

    print('\n【Retrieval評価】')
    paths = [('従来重み固定', 'retrieval_fixed/observed8_fixed_metrics.json'),
             ('8ライン学習後', 'retrieval_test/observed8_metrics.json')]
    for label, name in paths:
        path = root / name
        if not path.is_file():
            print(label + ': 完了結果なし')
            continue
        d = load(path)
        strict = d['strict']
        soft5 = d['soft']['R_soft']['5.0']['R@']['10']
        print(f"{label}: Q={strict['Q']}, DB={strict['DB']}, "
              f"Strict R@10={strict['R@']['10']:.2%}, Soft5m R@10={soft5:.2%}")
        if label == '8ライン学習後' and invalid_soft:
            print('  重み選択の修正前の参考値です。最終性能として使用しないでください。')

    print('\n【LoFTR＋PnP】')
    for folder in ('dense_fixed', 'dense'):
        for scope in ('pilot10', 'all507'):
            out = root / folder / (scope + '_pnp')
            path = out / 'integrated_summary.json'
            if path.is_file():
                d = load(path)
                median = d.get('xy_median_success_only')
                print(f"{folder}/{scope}: 完了、Query={d['queries']}, XY中央値={median} m")
            else:
                print(f'{folder}/{scope}: ' + ('出力途中・要確認' if out.exists() else '完了結果なし'))
    dependency = Path('/home/wakamatsu/ITS/handover_shimizu/src/LoFTR/tools/'
                      'make_loftr_matches_gt_step2a_paper_final_fixed_cam_v6.py')
    print('\n既存LoFTR教師生成器のv6依存:', 'あり' if dependency.is_file() else '欠落')
    print('この報告では学習・推論の再実行やファイル変更を行っていません。')


if __name__ == '__main__':
    main()
