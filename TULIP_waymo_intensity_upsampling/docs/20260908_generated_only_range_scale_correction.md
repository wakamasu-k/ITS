# Waymo 16→32生成ring評価の距離スケール修正（2026-09-08）

## 目的

`outputs/waymo_16_to_32_full_e300/checkpoint.pth`は距離を`75.0 m`で正規化して学習したが、生成ring限定評価ではDataset生成時とMAE計算時にこのスケールが明示されていなかった。学習時と評価時の条件を一致させ、raw距離をメートル単位で記録する。

## 修正

- チェックポイントの`args.range_scale_m`を評価時に自動取得する。
- CLIで`--range-scale-m`を指定した場合、チェックポイント値との不一致をエラーにする。
- Datasetへ同じ`range_scale_m`を渡す。
- MAE計算前に距離チャンネルをrawメートルへ戻す。
- `summary.json`へ`range_scale_m`を記録する。
- 強度帯評価にも同じ処理を適用する。

## テスト

```bash
source /home/wakamatsu/ITS/.venv_tulip/bin/activate
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
python -m unittest tests.test_waymo_16_to_32_generated_only -v
```

結果: 7 tests, OK。

## test 40フレーム再評価コマンド

```bash
source /home/wakamatsu/ITS/.venv_tulip/bin/activate
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling

python tools/evaluate_waymo_16_to_32_generated_only.py \
  --index outputs/training_pairs_16_to_32_full/index.csv \
  --tulip-root /home/wakamatsu/ITS/TULIP_backup/tulip \
  --checkpoint outputs/waymo_16_to_32_full_e300/checkpoint.pth \
  --output-root outputs/waymo_16_to_32_full_e300/generated_only_corrected_scale75_test40_20260908 \
  --intensity-transform log1p \
  --dataset-role test \
  --max-frames 40 \
  --device cuda
```

## 修正後の結果

- checkpoint epoch: `299`（300 epoch学習の0始まり表記）
- 評価範囲: 生成された奇数ring `[1,3,...,31]`のみ
- test frames: `40`
- generated valid pixels: `1,491,970`
- range scale: `75.0 m`
- intensity transform: `log1p`

| 手法 | 距離MAE [m] | raw反射強度MAE |
|---|---:|---:|
| TULIP | 1.224831 | 17.817501 |
| 最近傍補間 | 2.772037 | 18.497297 |
| 線形補間 | 2.418986 | 18.193589 |

修正前のTULIP距離MAE `0.016331`は正規化領域の値であり、`75.0`を掛けた`1.224831 m`が正しい。反射強度の復元処理は以前からraw領域へ戻していたため、反射強度MAEと強度帯別MAEは修正前後で変化しなかった。

## 強度帯別の主要結果

| GT raw反射強度 | 画素数 | TULIP MAE | 最近傍MAE | 線形MAE |
|---|---:|---:|---:|---:|
| 0–0.5 | 1,417,517 | 0.0907 | 2.2043 | 2.0041 |
| 0.5–1 | 59,405 | 0.3365 | 7.1149 | 7.4360 |
| 1–2 | 5,094 | 1.4136 | 36.1575 | 34.5088 |
| 2–5 | 3,000 | 3.2693 | 158.3958 | 126.0872 |
| 5–20 | 2,142 | 11.2198 | 73.6081 | 88.8775 |
| 20–100 | 1,658 | 42.9572 | 408.5745 | 339.5391 |
| 100–1000 | 1,386 | 405.2457 | 924.9314 | 802.3719 |
| 1000以上 | 1,768 | 14,586.2976 | 12,055.6138 | 12,148.2748 |

## 解釈と課題

1. 生成ring全体ではTULIPが最近傍・線形補間を上回る。
2. 0～1000の各強度帯ではTULIPのMAEが最小である。
3. 1000以上だけはTULIPが両補間より悪く、高反射領域の復元が残る主課題である。
4. 1000以上は全生成有効画素の約0.119%しかなく、データ不均衡が大きい。
5. 次は最大weight 2程度の穏やかな高反射重み付けを、同じtest評価条件で比較する。

## 出力

- `outputs/waymo_16_to_32_full_e300/generated_only_corrected_scale75_test40_20260908/summary.json`
- `outputs/waymo_16_to_32_full_e300/generated_only_corrected_scale75_test40_20260908/per_frame.csv`
- `outputs/waymo_16_to_32_full_e300/generated_only_corrected_scale75_test40_20260908/run.log`
