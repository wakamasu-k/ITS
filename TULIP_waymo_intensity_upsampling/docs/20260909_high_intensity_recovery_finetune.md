# 高反射強度帯の段階的回復実験（2026-09-09）

## 目的

Waymo 16→32・2ch TULIPについて、`low_0_1`モデルで得られた低強度性能を
できるだけ維持しながら、20・100・1000以上の高反射検出を回復する。
TULIPのTransformer、Encoder、Decoder、Attention、入出力headは変更しない。

## 出発点

- `low_0_1` validation 40フレームの0～1 raw intensity MAE: 0.06670
- baselineの同等値（既存binからの概算）: 0.09320
- `low_0_1`のrange MAE: 0.82473 m
- `low_0_1`の20・100・1000以上Recall: すべて0
- moderateはbaselineより100・1000以上のRecall/F1を改善したが、0～1 MAEを悪化させた

## 最小変更の仮説

`low_0_1` checkpointのモデル重みだけを初期値として読み込み、既存moderate
profileを低い学習率・短いepochでfine-tuneする。optimizer stateは復元しない。
元checkpointとは別のファイルへ保存し、比較可能性と復帰可能性を保つ。

## 第1条件

```bash
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
source /home/wakamatsu/ITS/.venv_tulip/bin/activate

python -u tools/train_waymo_16_to_32_log1p_moderate_tail.py \
  --index outputs/training_pairs_16_to_32_full/index.csv \
  --tulip-root /home/wakamatsu/ITS/TULIP_backup/tulip \
  --init-checkpoint outputs/waymo_16_to_32_controlled_low_0_1_s0/checkpoint.pth \
  --output outputs/waymo_16_to_32_low01_to_moderate_lr1e4_e30/checkpoint.pth \
  --profile moderate \
  --max-train-samples 160 \
  --epochs 30 \
  --batch-size 1 \
  --lr 1e-4 \
  --weight-decay 0.01 \
  --range-weight 1.0 \
  --intensity-weight 1.0 \
  --range-scale-m 75.0 \
  --seed 0 \
  --device cuda
```

## 合否基準

validation 40フレームで次を同時に確認する。

1. 0～1 raw MAEが0.06670から大幅に悪化しない。
2. range MAEが0.82473 mから大幅に悪化しない。
3. 閾値20・100・1000のRecallが0より大きくなる。
4. PrecisionとF1を併記し、Recallだけの改善を成功としない。
5. baseline・moderate・low_0_1と同じ生成奇数行maskで比較する。

この条件で両立しない場合は、学習率を下げるかepochを短くする。モデルheadや
Transformerの変更、分類head追加は、この最小実験の結果を確認するまで行わない。
