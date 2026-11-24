# TULIP準拠標準2ch baseline: 30 epoch

## 条件

- checkpoint: `outputs/waymo_16_to_32_paper_baseline_e30/checkpoint.pth`
- train samples: 160
- epochs: 30
- batch size: 1
- learning rate: `1e-4`
- loss: 標準2ch masked L1
- intensity transform: raw
- tail weighting: なし
- Transformer本体: 変更なし

## 学習

- initial mean loss: `11.1360`
- final mean loss: `6.2437`
- loss decreased: `true`

## validation 40

| 方法 | range MAE | intensity MAE |
|---|---:|---:|
| TULIP | 3.9983 | 6.7931 |
| nearest | 1.1749 | 5.3903 |
| linear | 1.4165 | 6.4840 |

## test 40

| 方法 | range MAE | intensity MAE |
|---|---:|---:|
| TULIP | 4.0357 | 13.4095 |
| nearest | 1.3855 | 9.2368 |
| linear | 1.6972 | 10.7297 |

## 解釈

標準TULIP準拠baselineでは、rangeの学習は進んだが、raw intensityはnearest/linearを上回らなかった。
これは現在のraw intensityに非常に大きな外れ値が含まれ、標準L1の最適化へ影響しているためと考えられる。

したがって、このcheckpointを「TULIP構造の基準結果」として保持し、tail-weighted、log1pは構造変更ではなく
データ・損失条件の追加実験として別に報告する。

次は、標準baselineとtail-weightedを同一test点群評価で比較し、range重視とintensity重視のトレードオフを確認する。

---

作成日（ファイル更新日時から推定）: 2026-08-20
