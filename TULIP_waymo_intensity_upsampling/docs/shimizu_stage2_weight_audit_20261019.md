# 清水手法の重みとStage 2 fine再現（2026-10-19）

## 重みの関係

| 処理 | 入力重み | 出力・備考 |
| --- | --- | --- |
| Retrieval | ResNet34のImageNet初期化とNetVLAD学習 | `waymo_outputs/old32_retrieval/train_wdrive_paper80_b16/checkpoints/{best,epoch_058}.ckpt` |
| LoFTR Stage 1 | `handover_shimizu/src/LoFTR/weights/mega_depth/outdoor_ds.ckpt` | `waymo_outputs/old32_dense/loftr_retrain_stage1_paper_wdrive/stage1_final_gstep715958.pth` |
| LoFTR Stage 2 fine | 上記Stage 1の最終重み | `tools/train_shimizu_stage2_fine_bn_fixed.py`で新しい出力先へ保存 |
| PnP | 学習重みなし | LoFTR対応点、深度・index、カメラ情報、ICP変換を利用 |

Retrievalの80 epoch実験は、壊れた`modality_stats.json`のため学習時にmean/std=0.5へフォールバックした。評価する際は`configs/paper80_effective_modality_stats.json`を用いる。`best.ckpt`はstrict R@10で選択され、修論のSoft指標による選択とは異なる。また881件のvalidationに507件の評価クエリが含まれるため、独立testとして扱わない。

## 既存の数値の出所

- Strict 507件でマッチ中央値1135、3D対応中央値1013、PnP成功率98.62%は、`/mnt/w/32line/experiment/loftr/train/run/2026_01_28/coarse_fine_bacbone/latest_model.pth`による評価。自分で再学習したStage 1ではない。
- 再学習Stage 1の同じ507件ではマッチ中央値783、3D対応中央値713、PnP成功率97.63%。Soft(10m) 9759件ではマッチ中央値488、3D対応中央値439。
- 継承元の修論値はStrictで約1582マッチ・約1285 3D対応、Soft(10m)で約1368マッチ・約1084 3D対応。
- 100 stepの旧Stage 2スクリプトはfine重み24個のほかに凍結backboneのBatchNorm統計102個を変化させていた。5ペアから4件ずつ選んだStrict 20件で、マッチ中央値はStage 1が732.5、旧Stage 2が333.5、BatchNorm統計を戻したStage 2が732.5。これは小規模診断で、修論との最終比較ではない。
- BatchNorm固定版の1 step学習を実行済み。Stage 1重みのロード欠落0・余剰0、fine重み24個が変化、fine以外の変化0。保存先は`waymo_outputs/old32_dense/loftr_stage2_bnfixed_smoke1_20261019/latest_model.pth`。この重みはsmoke専用。

## Stage 2 fineの段階実行

`handover_shimizu`のコードを変更せず、TULIP側の修正版を使用する。コマンドはWSLのbashで実行する。

```bash
source /home/wakamatsu/ITS/.venv_waymo/bin/activate
cd /home/wakamatsu/ITS/handover_shimizu/src/LoFTR
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling/tools/train_shimizu_stage2_fine_bn_fixed.py \
  --manifest /mnt/w/32line/datasets/loftr_pairs_shifted/training/manifest_matches_step2a_paper_final_wdrive.csv \
  --init-ckpt /home/wakamatsu/ITS/waymo_outputs/old32_dense/loftr_retrain_stage1_paper_wdrive/stage1_final_gstep715958.pth \
  --out-dir /home/wakamatsu/ITS/waymo_outputs/old32_dense/loftr_stage2_bnfixed_1000_20261019 \
  --device cuda --epochs 1 --max-steps 1000 --save-every 100 \
  --max-gt-points 256 --num-workers 0
```

このスクリプトはoptimizer状態を保存しないため、`latest_model.pth`から厳密な学習再開はできない。上記はStage 1から新規に1000 step進める。`--max-steps 1000`に達すると`latest_model.pth`を保存して終了する。既存の100 step版を上書きしない。

## 同じ評価設定でマッチ数とPnPを比較

まず1000 step重みを少数ペアで確認する。20件は`/tmp/shimizu_strict_probe20_20261019.csv`に作成済みだが一時ファイルなので、WSL再起動後は作り直すこと。最終比較では全件manifestを使う。

```bash
cd /home/wakamatsu/ITS/handover_shimizu/src/LoFTR
source /home/wakamatsu/ITS/.venv_waymo/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python tools/eval_cross_match_to_pose_zalign_multi_visall_v2.py \
  --manifest /home/wakamatsu/ITS/waymo_outputs/old32_dense/manifest_cross_eval_gt_v3_wdrive_test4.csv \
  --ckpt /home/wakamatsu/ITS/waymo_outputs/old32_dense/loftr_stage2_bnfixed_1000_20261019/latest_model.pth \
  --out-dir /home/wakamatsu/ITS/waymo_outputs/old32_dense/eval_stage2_bnfixed_1000_test4 \
  --gt-type t10m \
  --icp-csv /mnt/w/32line/datasets/splits/cross/all/icp_startend_staticmaps.csv \
  --device cuda --amp-dtype bf16 --write-oracle
```

`--gt-type t10m`で9759ペアを処理し、`per_pair.csv`中の`is_strict_gt=True`で507件のStrictを取り出す。重み以外の引数は既存のStage 1評価`eval_dense_pose_retrain_stage1_final/summary.json`と揃える。まず`summary.json`の`n_pairs=9759`を確認する。

```bash
python - <<'PY'
import csv, statistics
from pathlib import Path
root = Path('/home/wakamatsu/ITS/waymo_outputs/old32_dense')
for label, subdir in (
    ('inherited', 'eval_loftr_gt_manifest_test4_full'),
    ('stage1', 'eval_dense_pose_retrain_stage1_final'),
    ('stage2', 'eval_stage2_bnfixed_1000_test4'),
):
    with (root / subdir / 'per_pair.csv').open(newline='') as handle:
        all_rows = list(csv.DictReader(handle))
    if len(all_rows) != 9759:
        raise ValueError(f'{label}: expected 9759 pairs, got {len(all_rows)}')
    strict = [r for r in all_rows if r['is_strict_gt'].lower() in ('true', '1')]
    if len(strict) != 507:
        raise ValueError(f'{label}: expected 507 Strict pairs, got {len(strict)}')
    for scope, rows in (('strict', strict), ('soft10', all_rows)):
        print(label, scope,
              'matches_median=', statistics.median(float(r['n_match_in']) for r in rows),
              'n3d_median=', statistics.median(float(r['n_3d']) for r in rows),
              'pnp_success_rate=', sum(r['pnp_success'].lower() in ('true', '1') for r in rows) / len(rows))
PY
```

Stage 2はfine座標の精度向上を狙う段階であり、coarse候補の数を直接増やすものではない。マッチ数だけでなくPnP成功率と位置誤差も`summary.json`で比較する。修論との厳密な比較には同じ評価条件とチェックポイント選択根拠が必要。
