# Waymo 16-to-32 log1p + Huberスモーク実験（2026-09-08）

## 結論

高反射画素を削除せず、全有効生成画素を使い、反射強度のlog1p誤差に標準Huber
loss（delta 0.2）を適用した。20 trainフレーム・5 epochではbaselineを改善せず、
本学習への採用条件を満たさなかった。

## 条件

- train先頭20フレーム、validation 40フレーム
- 16ライン入力から32ラインを予測
- 評価対象は有効な生成対象奇数ringのみ（1,537,686画素）
- range scale: 75 m
- intensity transform: log1p
- intensity loss: Huber、delta 0.2
- 高反射画素を含む全有効画素を保持
- AdamW、learning rate 1e-4、batch size 1、seed 0

## 40フレーム評価

| 指標 | L1 baseline | Huber delta 0.2 | 変化 |
|---|---:|---:|---:|
| range MAE [m] | 3.895878 | 4.618495 | +18.5483% |
| intensity raw MAE | 6.305852 | 6.310118 | +0.0677% |
| intensity log1p MAE | 0.090722 | 0.093910 | +3.5139% |
| 負のrange予測数 | 2,695 | 768 | -71.50% |
| 負のintensity予測数 | 36,840 | 22,315 | -39.43% |

閾値20、100、1000以上の反射強度について、Huberモデルのpredicted positiveと
recallは全て0だった。負値は減ったが、高反射復元とMAEは改善していない。

## 画像確認

validation sample 0の生成対象画素MAEはrange 4.474975 m、raw intensity
0.678181だった。予測は32ラインの密な画像になったが、rangeには横縞が残り、
intensityはGTより平滑で明るいピークを再現できなかった。

画像保存先:

- `outputs/controlled_huber_smoke_20260908/visualization_validation_sample0/range_gt_low_pred.png`
- `outputs/controlled_huber_smoke_20260908/visualization_validation_sample0/intensity_gt_low_pred_linear.png`
- `outputs/controlled_huber_smoke_20260908/visualization_validation_sample0/intensity_gt_low_pred_log1p.png`
- 生配列: `outputs/controlled_huber_smoke_20260908/visualization_validation_sample0/comparison_raw.npz`

## ログと数値

- 学習ログ: `outputs/controlled_huber_smoke_20260908/huber_train.log`
- 評価ログ: `outputs/controlled_huber_smoke_20260908/huber_eval.log`
- テストログ: `outputs/controlled_moderate_tail_smoke_20260908/huber_tests.log`
- 集計JSON: `outputs/controlled_huber_smoke_20260908/huber_validation40/summary.json`
- フレーム別CSV: `outputs/controlled_huber_smoke_20260908/huber_validation40/per_frame.csv`

## 解釈と次の判断

標準Huberは大きなlog誤差の勾配をdelta 0.2に制限するため、高反射画素の悪影響は
抑えられる。しかし同時に反射強度headの学習信号が弱まり、5 epochでは通常領域も
改善しなかった。また2チャネル共通backboneのため、反射強度loss変更がrangeにも
影響し、range MAEが悪化した。

次はHuberを長時間学習へ拡大せず、rangeとintensityのloss干渉を分離する。
具体的には、学習済みrange側を固定してintensity headだけをfine-tuningし、
高反射ピークは閾値分類の補助lossで扱う。最初の合格条件は20フレームoverfitで
強度20以上のrecallが0を脱することとする。
