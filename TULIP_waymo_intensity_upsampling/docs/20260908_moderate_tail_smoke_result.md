# Waymo 16-to-32 中強度重み付けスモーク実験（2026-09-08）

## 結論

今回の `moderate` 条件は本学習へ採用しない。通常領域の `log1p` 反射強度MAEと
負値出力は改善した一方、目的である高反射点の検出recallは改善せず、距離MAEも
悪化した。これは実装失敗ではなく、20学習フレーム・5 epoch・最大2倍の重みでは、
全体の約0.304%しかない高反射画素を学習させる信号が不足した結果と判断する。

## 固定した比較条件

- 入力・GT: Waymo TOP LiDAR 16-to-32
- 学習: train先頭20フレーム、5 epoch、batch size 1
- 評価: 学習に使わないvalidation 40フレーム
- 評価対象: GTが有効な生成対象の奇数ringのみ、1,537,686画素
- range: 75 mで正規化
- intensity: `log1p`
- optimizer: AdamW、learning rate `1e-4`
- seed: 0、shuffleなし
- baseline: 全画素のintensity weight 1.0
- moderate: `[0,2)=1.0`, `[2,20)=1.2`, `[20,100)=1.5`,
  `[100,1000)=2.0`, `[1000,inf)=2.0`

## validation 40フレーム結果

| 指標 | baseline | moderate | 変化 |
|---|---:|---:|---:|
| range MAE [m] | 3.895878 | 4.305982 | +10.5266%（悪化） |
| intensity raw MAE | 6.305852 | 6.302520 | -0.0529% |
| intensity log1p MAE | 0.090722 | 0.087353 | -3.7143% |
| 負のrange予測数 | 2,695 | 134 | -95.03% |
| 負のintensity予測数 | 36,840 | 16,665 | -54.76% |

高反射検出では、baselineとmoderateの両方が閾値20、100、1000の全てで
predicted positive 0、recall 0だった。したがってraw MAEの小さな改善を、
高反射領域の復元成功とは解釈しない。

強度帯別raw MAEも差は小さい。

| GT raw intensity | baseline | moderate |
|---|---:|---:|
| 0–0.5 | 0.067332 | 0.063860 |
| 20–100 | 45.362073 | 45.351996 |
| 100–1000 | 388.138404 | 388.131098 |
| 1000以上 | 16954.080472 | 16954.066190 |

比較用の線形補間はrange MAE 1.959374 mで、両TULIP条件より良い。現時点では
距離側も学習不足であり、この5 epochモデルを研究成果の最終値には使用しない。

## 実行コード

```bash
source /home/wakamatsu/ITS/.venv_tulip/bin/activate
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling

python tools/train_waymo_16_to_32_log1p_moderate_tail.py \
  --index outputs/training_pairs_16_to_32_full/index.csv \
  --tulip-root /home/wakamatsu/ITS/TULIP_backup/tulip \
  --output outputs/controlled_moderate_tail_smoke_20260908/baseline_e5.pth \
  --profile baseline --max-train-samples 20 --epochs 5 --batch-size 1 \
  --lr 0.0001 --range-weight 1.0 --intensity-weight 1.0 \
  --range-scale-m 75 --seed 0 --device cuda

python tools/train_waymo_16_to_32_log1p_moderate_tail.py \
  --index outputs/training_pairs_16_to_32_full/index.csv \
  --tulip-root /home/wakamatsu/ITS/TULIP_backup/tulip \
  --output outputs/controlled_moderate_tail_smoke_20260908/moderate_e5.pth \
  --profile moderate --max-train-samples 20 --epochs 5 --batch-size 1 \
  --lr 0.0001 --range-weight 1.0 --intensity-weight 1.0 \
  --range-scale-m 75 --seed 0 --device cuda

python tools/evaluate_waymo_16_to_32_controlled.py \
  --index outputs/training_pairs_16_to_32_full/index.csv \
  --tulip-root /home/wakamatsu/ITS/TULIP_backup/tulip \
  --checkpoint outputs/controlled_moderate_tail_smoke_20260908/baseline_e5.pth \
  --output-root outputs/controlled_moderate_tail_smoke_20260908/baseline_validation40 \
  --dataset-role validation --max-frames 40 --device cuda

python tools/evaluate_waymo_16_to_32_controlled.py \
  --index outputs/training_pairs_16_to_32_full/index.csv \
  --tulip-root /home/wakamatsu/ITS/TULIP_backup/tulip \
  --checkpoint outputs/controlled_moderate_tail_smoke_20260908/moderate_e5.pth \
  --output-root outputs/controlled_moderate_tail_smoke_20260908/moderate_validation40 \
  --dataset-role validation --max-frames 40 --device cuda
```

## 保存先

- 学習ログ: `outputs/controlled_moderate_tail_smoke_20260908/*_train.log`
- 評価ログ: `outputs/controlled_moderate_tail_smoke_20260908/*_eval.log`
- 詳細指標: `outputs/controlled_moderate_tail_smoke_20260908/*_validation40/summary.json`
- フレーム別指標: `outputs/controlled_moderate_tail_smoke_20260908/*_validation40/per_frame.csv`

## 次の最短手順

1. rangeとintensityを同時に最適化した影響を切り分けるため、range重みを固定した
   intensity専用headまたは二段階fine-tuningを比較する。
2. 高反射を単なる2倍重みではなく、閾値分類補助lossと回帰lossの組合せで直接学習する。
3. まず20フレームoverfitで閾値20以上のrecallが0を脱することを合格条件にする。
4. 合格後だけ160フレーム・複数seedへ拡大する。

学習時に出たCuBLAS再現性警告に対して、実験後のコード版1.0.1では
`CUBLAS_WORKSPACE_CONFIG=:4096:8` をCUDA初期化前に設定した。今回保存した
チェックポイントは警告が出た版1.0.0なので、単一seedの傾向確認用として扱う。
