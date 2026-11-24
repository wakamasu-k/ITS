# Waymo 16→32 高反射強度の統制比較

作成日: 2026-09-08

## 目的

TULIPのTransformer、Encoder、Decoder、Attention、skip connectionを変更せず、
Waymo TOP LiDARの16→32・range/intensity 2chモデルで高反射強度の復元を改善する。
高反射重み以外の条件を同一にしたbaselineとmoderateを比較する。

## 履歴baselineの位置付け

`outputs/waymo_16_to_32_full_reference/checkpoint.pth`は次の条件で学習済みである。

- train 160件、300 epoch、batch size 1
- learning rate `5e-4`
- range scale `75 m`
- intensity transform `log1p`
- AdamWのweight decayは未指定で、PyTorch既定値`0.01`
- seedは未固定
- lossは全有効行、range/intensity等重み
- shuffle、scheduler、warmup、AMPは使用していない

このcheckpointは履歴比較用として保持する。生成行maskとseedが今回の統制実験と
異なるため、moderateモデルとの唯一の対照実験にはしない。

## 新しい学習条件

共通条件は次の通り。

- 入力 `[B,2,16,2688]`
- 出力・GT `[B,2,32,2688]`
- native評価幅 `2650`
- 損失対象はGT有効かつ生成奇数行`1,3,...,31`のみ
- rangeは`75 m`で除算
- intensityは`log1p`
- range/intensity channel weightは`1:1`
- AdamW、learning rate `5e-4`、weight decay `0.01`
- seed `0`、shuffleなし、worker `0`
- scheduler、warmup、AMPなし
- gradient clip norm `1.0`

baseline profileは全intensity weightを`1.0`とする。moderate profileはraw GT
intensityを基準に次を用いる。

| raw intensity | weight |
|---|---:|
| 0以上2未満 | 1.0 |
| 2以上20未満 | 1.2 |
| 20以上100未満 | 1.5 |
| 100以上1000未満 | 2.0 |
| 1000以上 | 2.0 |

強度帯はraw値の意味を持つが、float32の`log1p→expm1`境界丸めを避けるため、
実装上はlog1p targetを`log1p(threshold)`と比較する。誤差はlog1p領域で計算し、
重み付き平均はweight合計で正規化する。

## 追加したコード

- `tools/train_waymo_16_to_32_log1p_moderate_tail.py`
- `tools/evaluate_waymo_16_to_32_controlled.py`
- `tests/test_waymo_16_to_32_controlled.py`

既存TULIP、既存Waymo Dataset、既存model adapterは変更していない。

## 統一評価

評価器はcheckpointに保存されたrange scaleとintensity transformを自動使用し、
CLI overrideがcheckpointと異なる場合はエラーにする。評価対象は有効な生成奇数行のみ。

保存項目:

- range MAE `[m]`
- raw intensity MAE
- log1p intensity MAE
- raw GT intensity帯別MAE
- 閾値20、100、1000のPrecision、Recall、F1
- 負range予測率
- clamp前の負intensity予測率
- TULIP、nearest、linearの同一mask比較
- pixel-weighted集計とframe-mean集計

raw intensity評価時だけ負予測を0へclampし、clamp前の負値率を別に記録する。
nearest/linearは従来結果との連続性のためraw range/raw intensity領域で補間する。

## 検証結果

- `.venv_tulip`による3ファイルの`py_compile`: PASS
- `.venv_waymo`による関連単体テスト: `18 passed`
- 実データ1件、baseline、1 epoch、CUDA学習: PASS
- 生成checkpointの統一評価: PASS
- smoke testの評価対象: 生成有効画素38,398点

1 epochのsmoke数値は未学習に近いため、性能判断には使用しない。

## 実行順

### 1. controlled baseline

```bash
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
source /home/wakamatsu/ITS/.venv_tulip/bin/activate

python tools/train_waymo_16_to_32_log1p_moderate_tail.py \
  --index outputs/training_pairs_16_to_32_full/index.csv \
  --tulip-root /home/wakamatsu/ITS/TULIP_backup/tulip \
  --output outputs/waymo_16_to_32_controlled_baseline_s0/checkpoint.pth \
  --profile baseline \
  --max-train-samples 160 \
  --epochs 300 \
  --batch-size 1 \
  --lr 5e-4 \
  --weight-decay 0.01 \
  --range-weight 1.0 \
  --intensity-weight 1.0 \
  --range-scale-m 75.0 \
  --seed 0 \
  --device cuda
```

### 2. moderate high-intensity weight

```bash
python tools/train_waymo_16_to_32_log1p_moderate_tail.py \
  --index outputs/training_pairs_16_to_32_full/index.csv \
  --tulip-root /home/wakamatsu/ITS/TULIP_backup/tulip \
  --output outputs/waymo_16_to_32_controlled_moderate_s0/checkpoint.pth \
  --profile moderate \
  --max-train-samples 160 \
  --epochs 300 \
  --batch-size 1 \
  --lr 5e-4 \
  --weight-decay 0.01 \
  --range-weight 1.0 \
  --intensity-weight 1.0 \
  --range-scale-m 75.0 \
  --seed 0 \
  --device cuda
```

### 3. validation評価

```bash
python tools/evaluate_waymo_16_to_32_controlled.py \
  --index outputs/training_pairs_16_to_32_full/index.csv \
  --tulip-root /home/wakamatsu/ITS/TULIP_backup/tulip \
  --checkpoint outputs/waymo_16_to_32_controlled_baseline_s0/checkpoint.pth \
  --output-root outputs/waymo_16_to_32_controlled_baseline_s0/validation40 \
  --dataset-role validation \
  --max-frames 40 \
  --device cuda

python tools/evaluate_waymo_16_to_32_controlled.py \
  --index outputs/training_pairs_16_to_32_full/index.csv \
  --tulip-root /home/wakamatsu/ITS/TULIP_backup/tulip \
  --checkpoint outputs/waymo_16_to_32_controlled_moderate_s0/checkpoint.pth \
  --output-root outputs/waymo_16_to_32_controlled_moderate_s0/validation40 \
  --dataset-role validation \
  --max-frames 40 \
  --device cuda
```

test splitはvalidationで採用条件を決めた後、一度だけ同じ評価器で実行する。

## 次の判定

moderate採用可否は、全体MAEだけでなく以下を同時に比較する。

1. 生成行のrange MAE
2. 生成行のraw/log1p intensity MAE
3. 0〜2、20〜100、100〜1000、1000以上のMAE
4. 閾値100・1000のRecall、Precision、F1
5. 負range/intensity予測率

moderateで高強度Recallだけが上がり、Precision、低強度、rangeが大きく悪化する場合は
採用しない。次段階はモデル変更ではなく、高反射点を含むフレームの学習サンプリングを
統制比較する。分類ヘッド、Focal Loss、残差予測、出力ヘッド分離はその後の候補とする。

## 作業ツリー上のHuber試作

2026-09-08時点で、同じtrainerとtestにHuber intensity lossの未commit試作が存在する。
構文確認と専用テストは`13 passed`だが、これは高反射点を強く学習させる処理ではなく、
大誤差の勾配を`huber_delta`で制限するロバスト化である。baseline対moderateの比較には
既定のL1を使用し、Huberは結果が揃った後の第3条件として分離する。
