# 帯別バランス反射強度実験（2026-09-09）

## 背景

`low_0_1`は0～1 raw MAEを0.06670まで改善したが、20・100・1000以上の
Recallが0になった。moderate 30epochは高反射F1を改善したが、0～1 raw MAEが
0.10273へ悪化した。その後の`low_0_1` 5epochで0～1 raw MAEは0.06756へ戻った
一方、高反射F1はほぼ0へ戻った。逐次fine-tuneでは両者を同時に維持できない。

## 実験条件

モデル構造と既存2ch出力は変更せず、生成奇数行のlog1p intensity誤差をraw GT
intensity帯ごとに平均してから合成する。

```text
L_intensity = 0.90 L_0_1
            + 0.05 L_1_20
            + 0.02 L_20_100
            + 0.02 L_100_1000
            + 0.01 L_1000_inf
```

バッチ内に存在しない帯は合成から除外し、存在する帯の係数で再正規化する。
これにより高反射画素が少ないことによる勾配消失を防ぎつつ、0～1を主目的として
維持する。

## 最初の実行

```bash
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
source /home/wakamatsu/ITS/.venv_tulip/bin/activate

python -u tools/train_waymo_16_to_32_log1p_moderate_tail.py \
  --index outputs/training_pairs_16_to_32_full/index.csv \
  --tulip-root /home/wakamatsu/ITS/TULIP_backup/tulip \
  --init-checkpoint outputs/waymo_16_to_32_controlled_low_0_1_s0/checkpoint.pth \
  --output outputs/waymo_16_to_32_band_balanced_s0_e10/checkpoint.pth \
  --profile band_balanced \
  --max-train-samples 160 \
  --epochs 10 \
  --batch-size 1 \
  --lr 1e-4 \
  --weight-decay 0.01 \
  --range-weight 1.0 \
  --intensity-weight 1.0 \
  --range-scale-m 75.0 \
  --seed 0 \
  --device cuda
```

## 評価基準

- 0～1 raw MAE: 0.080以下
- range MAE: 0.825 m以下
- 20以上F1: 0.35以上
- 100以上F1: 0.35以上
- 1000以上F1: 0.25以上
- Precision・Recallを必ず併記する

validationで条件を決めるまでtestは使用しない。
