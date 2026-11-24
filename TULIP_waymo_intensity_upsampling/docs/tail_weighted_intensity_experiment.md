# Waymo 16→32 高intensity loss実験

## 1. 目的

Waymo TOP LiDARの16ライン入力から32ラインへ反射強度を復元する際、
`1000以上`の高intensity画素がraw MAEを支配しているため、
TULIP本体を変更せず、lossだけで高強度画素の学習寄与を増やす。

## 2. 事前診断

test 40フレームの生成ring（奇数行）を対象に、target intensityが高い画素と、
同じ列の隣接する観測ringのintensityを比較した。

| target条件 | 対象画素数 | 隣接観測ringにも同じ閾値超の値がある割合 |
|---|---:|---:|
| `>0.5` | 72,853 | 59.38% |
| `>1` | 14,913 | 61.54% |
| `>2` | 9,931 | 58.28% |
| `>5` | 6,935 | 56.37% |
| `>20` | 4,808 | 62.98% |
| `>100` | 3,154 | 66.77% |
| `>1000` | 1,767 | 68.19% |

`>1000`画素の約68%には隣接観測ringに同じ列の手掛かりがある。
残りは隣接ringに同じ高値がなく、入力からの直接推定が難しい可能性がある。
高intensity画素のrange有効率は100%だった。

## 3. 変更点

新規ファイルのみを追加した。

- `tools/train_waymo_16_to_32_tail_weighted.py`
- `tests/test_waymo_16_to_32_tail_weighted.py`

既存のTULIP、既存adapter、既存Dataset、既存学習スクリプトは変更しない。

### loss仕様

- 入力・出力shapeは既存と同じ
  - input: `[B,2,16,2688]`
  - target/prediction: `[B,2,32,2688]`
- `target_mask & generated_mask`だけをloss対象にする
- 観測済み偶数ringはloss対象から除外する
- rangeは通常masked L1
- intensityはraw値のままmasked L1
- `target intensity > tail_threshold`の画素に`tail_weight`を掛ける
- Transformer、Encoder、Decoder、Attention、patch処理は変更しない

`tail_threshold`と`tail_weight`は実験引数であり、正規化値ではない。
研究上の最終値として固定しない。

## 4. 実行例

最初の探索例:

```bash
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
source /home/wakamatsu/ITS/.venv_tulip/bin/activate

python tools/train_waymo_16_to_32_tail_weighted.py \
  --index outputs/training_pairs_16_to_32_full/index.csv \
  --tulip-root /home/wakamatsu/ITS/TULIP_backup/tulip \
  --output outputs/waymo_16_to_32_tail_weighted_t20_w10/checkpoint.pth \
  --max-train-samples 160 \
  --epochs 30 \
  --batch-size 1 \
  --lr 1e-4 \
  --range-weight 0.05 \
  --intensity-weight 1.0 \
  --tail-threshold 20 \
  --tail-weight 10 \
  --device cuda
```

学習後は、既存の評価スクリプトでvalidationを先に評価し、testは候補選択後に一度だけ評価する。

## 5. 判定指標

全体raw MAEだけでなく、以下を併記する。

- `0〜20` raw intensity MAE
- `20〜100` raw intensity MAE
- `100〜1000` raw intensity MAE
- `1000以上` raw intensity MAE
- 高intensity画素の有効数

全体MAEが改善しても、通常強度の復元を壊している場合は採用しない。
逆に、全体MAEが大きく変わらなくても、通常強度を維持しながら高強度帯が改善する場合は、
反射強度画像の用途別に評価する。

## 6. 現時点の判断

raw priority e30は通常強度帯ではnearest・linearより良いが、`1000以上`のtest外れ値で劣る。
log1p e30はtestの高強度帯を実質的に改善しなかった。

したがって、次の実験は正規化方式の変更ではなく、同一モデルでのtarget依存lossの比較とする。

## 7. 追加実装・smoke test結果 (2026-08-20)

追加したloss関数の単体テストは4件すべて成功した。

- py_compile: 成功
- pytest: `4 passed`
- 実データGPU smoke test: 1 frame、1 epoch、CUDAで完走
- input: `[1,2,16,2688]`
- target: `[1,2,32,2688]`
- tail threshold: `20`
- tail weight: `10`
- tail fraction: `2.5696e-05`
- loss: `1.13146`
- checkpoint保存: `outputs/waymo_16_to_32_tail_weighted_smoke/checkpoint.pth`

このsmoke testは実行経路の確認であり、性能評価ではない。
学習時には既存TULIP初期化由来の
`zero-element tensors is a no-op`
警告が出たが、forward、loss、checkpoint保存に失敗はなかった。

`.venv_tulip`にはpytestがないため、py_compileとGPU実行は
`.venv_tulip`、pytestは`.venv_waymo`で実施した。

## 8. tail-weighted 160フレーム学習結果 (2026-08-20)

実データ160フレーム、30epochの学習が完了した。

- checkpoint: outputs/waymo_16_to_32_tail_weighted_t20_w10/checkpoint.pth
- initial loss: 84.14327498190104
- final loss: 83.49174985634163
- loss decreased: true

lossは約0.77%低下した。
ただし、このlossは生成ring限定かつtarget intensity依存重み付きであるため、
従来のraw priority学習のloss値とは直接比較しない。
モデル性能の判定にはvalidationのraw MAEと強度帯別MAEを使用する。

## 9. tail-weighted validation結果 (2026-08-20)

tail threshold 20、tail weight 10、生成ring限定でvalidation 40フレームを評価した。

- range MAE: 9.85875 m
- intensity MAE: 7.49619
- nearest intensity MAE: 5.39029
- linear intensity MAE: 6.48401

raw priority e30との比較では、

- range: 3.13228 mから9.85875 mへ悪化
- intensity: 6.75837から7.49619へ悪化

強度帯別でも、0〜0.5、0.5〜1、1〜2、2〜5、5〜20、
20〜100、100〜1000、1000以上の全帯域でraw priority e30を改善しなかった。
特に1000以上は16601.22で、raw priority e30の16600.18より悪化した。

この設定は不採用とする。test評価は実施しない。

今回の実験では、tail重み付けだけでなく、loss対象を全valid画素から生成ringだけへ変更している。
したがって、tail重みとloss対象row変更の影響が分離されていない。

次のアブレーションでは、同じTULIP構造のまま、
既存raw priorityと同じ全valid画素をloss対象に戻し、
tail重みだけを追加する条件を比較する。

## 10. loss対象rowを分離するための追加変更 (2026-08-20)

tail-weightedスクリプトに --loss-rows all|generated を追加した。

- generated: 有効な生成奇数ringだけをloss対象にする従来の実験条件
- all: 観測済み偶数ringを含む全valid画素をloss対象にする条件

tail重みの効果だけを比較する場合は、既存raw priority学習と同じく
--loss-rows all を使用する。これにより、loss対象rowの変更とtail重みの変更を分離できる。

公平な比較では、train/validation split、epoch、learning rate、batch size、
range/intensity weight、tail threshold、tail weightを固定する。

## 11. loss-rows all smoke test (2026-08-20)

全valid画素をloss対象にする条件で、実データ1フレーム・1epochのGPU smoke testを実施した。

- loss rows: all
- input: [1,2,16,2688]
- target: [1,2,32,2688]
- tail threshold: 20
- tail weight: 10
- loss: 0.987445
- tail fraction: 3.8436e-05
- checkpoint: outputs/waymo_16_to_32_tail_weighted_all_smoke/checkpoint.pth
- result: PASS

構文確認とtail-weighted単体テスト4件も成功した。

## 12. loss-rows all 学習結果 (2026-08-20)

全valid画素をloss対象にし、tail threshold 20、tail weight 10で
160フレーム・30epochの学習を実施した。

- checkpoint: outputs/waymo_16_to_32_tail_weighted_all_t20_w10/checkpoint.pth
- initial loss: 81.50460186451673
- final loss: 80.77309272037819
- loss decreased: true
- loss低下率: 約0.90%

lossは低下したが、tail重み付きlossの値はraw priorityのlossと直接比較しない。
validationのraw range/intensity MAEと強度帯別MAEで判定する。

## 13. loss-rows all validation結果 (2026-08-20)

全valid画素をloss対象にしたtail threshold 20、tail weight 10のvalidation結果。

- range MAE: 3.47409 m
- intensity MAE: 6.75330
- nearest intensity MAE: 5.39029
- linear intensity MAE: 6.48401

raw priority e30との比較:

- range: 3.13228 mから3.47409 mへ悪化
- intensity: 6.75837から6.75330へ微小改善

強度帯別では、0〜0.5、2〜5、5〜20、20〜100、
100〜1000、1000以上で改善した。
一方、0.5〜1と1〜2は悪化した。

高強度帯は改善したが、1000以上のMAEは16595.31であり、
nearestの6817.64、linearの8428.53を上回っている。
全体intensity MAEもlinearを下回っていない。

したがって、tail weight 10は現時点で最終採用しない。
range悪化を抑えながら高強度帯を維持できるか確認するため、
次はtail weightを下げた小規模ablationを行う。
test評価はまだ実施しない。

## 14. tail weight 2 学習結果 (2026-08-20)

全valid画素をloss対象にし、tail threshold 20、tail weight 2で
160フレーム・30epochの学習を実施した。

- checkpoint: outputs/waymo_16_to_32_tail_weighted_all_t20_w2/checkpoint.pth
- initial loss: 17.427696770429613
- final loss: 16.800614495575427
- loss decreased: true
- loss低下率: 約3.60%

lossは低下したが、tail weight 10やraw priorityのloss値とは直接比較しない。
validationのrange/intensity MAEと強度帯別MAEで判定する。

## 15. tail weight 2 validation結果 (2026-08-20)

全valid画素をloss対象にしたtail threshold 20、tail weight 2のvalidation結果。

- range MAE: 3.24418 m
- intensity MAE: 6.74267
- nearest intensity MAE: 5.39029
- linear intensity MAE: 6.48401

raw priority e30との比較:

- range: 3.13228 mから3.24418 mへ悪化
- intensity: 6.75837から6.74267へ改善

強度帯別では、0〜0.5、2〜5、5〜20、20〜100、
100〜1000、1000以上でraw priority e30より改善した。
1000以上は16600.18から16598.91へ微小改善したが、
nearest 6817.64、linear 8428.53には届かない。

tail weight 10よりrange悪化が小さく、現時点のtail-weighted候補ではweight 2が最良である。
ただし全体intensity MAEはlinear補間を下回っていないため、最終採用ではない。
validationで微小な改善が確認できたため、次にtest 40フレームで汎化を確認する。

## 16. tail weight 2 test結果 (2026-08-20)

tail threshold 20、tail weight 2、loss rows allのtest 40フレーム結果。

- range MAE: 3.46026 m
- intensity MAE: 13.36872
- nearest intensity MAE: 9.23681
- linear intensity MAE: 10.72966

raw priority e30との比較:

- range: 3.30982 mから3.46026 mへ悪化
- intensity: 13.38831から13.36872へ微小改善

強度帯別では、0〜0.5、2〜5、5〜20、20〜100、
100〜1000、1000以上でraw priority e30より改善した。
ただし、1000以上のMAEは12101.60であり、
nearest 6715.29、linear 7468.50を大きく上回る。

tail weight 2はraw priority e30よりintensityが約0.15%改善しただけで、
補間baselineを上回る性能ではない。最終モデルとしては未採用とする。

現行の標準評価とbin評価は観測済み偶数ringを含むため、
次は生成対象の奇数ringだけを対象とした評価指標を追加し、
実際の16→32復元性能を再比較する。

## 17. 生成ring専用validation評価 (2026-08-20)

観測済み偶数ringを除外し、生成対象の奇数ring 1,3,...,31だけを評価した。
validation 40フレーム、生成valid画素数は1,537,686である。

### raw priority e30

- range MAE: 3.18412 m
- intensity MAE: 6.34363
- nearest range MAE: 2.34750 m
- nearest intensity MAE: 10.77449
- linear range MAE: 1.97087 m
- linear intensity MAE: 9.99226

### tail weight 2

- range MAE: 3.25764 m
- intensity MAE: 6.32778
- nearest range MAE: 2.34750 m
- nearest intensity MAE: 10.77449
- linear range MAE: 1.97087 m
- linear intensity MAE: 9.99226

生成ringだけで見ると、TULIPはintensityでnearest・linearより良い。
tail weight 2はraw priority e30よりintensityを約0.25%改善したが、
rangeは悪化した。

強度帯別には、tail weight 2は0〜0.5、2〜5、5〜20、20〜100、
100〜1000でraw priority e30より改善した。
1000以上は16952.03で、nearest 14966.17、linear 14318.75より悪い。

この結果から、通常強度の生成ring復元はTULIPが有効である一方、
巨大intensity外れ値の完全再現は未達であることが確認できた。

## 18. 生成ring専用test評価 (2026-08-20)

観測済み偶数ringを除外し、生成奇数ringだけをtest 40フレームで評価した。
生成valid画素数は1,491,970である。

### raw priority e30

- range MAE: 3.32911 m
- intensity MAE: 17.99652
- nearest range MAE: 2.77204 m
- nearest intensity MAE: 18.49730
- linear range MAE: 2.41899 m
- linear intensity MAE: 18.19359

### tail weight 2

- range MAE: 3.43299 m
- intensity MAE: 17.97485
- nearest range MAE: 2.77204 m
- nearest intensity MAE: 18.49730
- linear range MAE: 2.41899 m
- linear intensity MAE: 18.19359

生成ringのintensityでは、tail weight 2がnearestより約2.8%、
linearより約1.2%良かった。
raw priority e30よりも約0.12%改善した。

一方、rangeはnearest・linearより悪く、tail weight 2はraw priority e30より悪化した。
1000以上のintensityはTULIPがnearest・linearより悪く、
巨大外れ値の数値再現は未達である。

生成ring専用評価により、通常intensityの16→32復元はTULIPで有効だが、
rangeと極端なintensity外れ値は別課題であることが確認できた。

## 19. 16→32疑似カラー可視化 (2026-08-20)

新規ツール tools/visualize_waymo_16_to_32_tulip.py を追加し、
同一validation frameについてraw priority e30とtail weight 2を可視化した。

対象:

- dataset role: validation
- sample index: 0
- segment: segment-10206293520369375008_2796_800_2816_800_with_camera_labels
- frame index: 0
- generated valid pixels: 39,354
- generated rows: 1,3,...,31

各checkpointで次の3枚を生成した。

- range_comparison.png
- intensity_comparison_linear.png
- intensity_comparison_log1p.png

全PNGは読み込み可能で、サイズは3230 x 1360、RGBAである。
線形表示のintensity vmaxは20、log1p表示の元値vmaxは20000であり、
これらは表示専用でrawデータを変更しない。

### sample 0の生成ring MAE

| checkpoint | range MAE | intensity MAE |
|---|---:|---:|
| raw priority e30 | 2.87231 m | 0.64863 |
| tail weight 2 | 3.00531 m | 0.64103 |

このフレームではtail weight 2がintensityで少し良く、
raw priority e30がrangeで良かった。

---

作成日（ファイル更新日時から推定）: 2026-08-20
