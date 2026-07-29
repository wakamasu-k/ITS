#!/usr/bin/env bash
set -euo pipefail

cd ~/ITS/handover_shimizu/src/retrieval
source ~/ITS/.venv_waymo/bin/activate

OUT_DIR="$HOME/ITS/waymo_outputs/old32_retrieval/train_wdrive_paper80_b16"

TRAIN_PAIRS="$HOME/ITS/waymo_outputs/old32_retrieval/training_wdrive.parquet"
MODALITY_STATS="$HOME/ITS/waymo_outputs/old32_retrieval/modality_stats.json"

VAL_PAIRS="$HOME/ITS/waymo_outputs/old32_retrieval/val4_test4_all_pairs_wdrive.parquet"
VAL_DB="$HOME/ITS/waymo_outputs/old32_retrieval/val4_test4_all_db_wdrive.parquet"
VAL_ICP="/mnt/w/32line/datasets/splits/cross/all/icp_startend_staticmaps.csv"

mkdir -p "$OUT_DIR"

python train.py \
  --pairs "$TRAIN_PAIRS" \
  --out-dir "$OUT_DIR" \
  --epochs 80 \
  --batch-size 16 \
  --lr 0.0003 \
  --wd 0.0001 \
  --backbone resnet34 \
  --clusters 64 \
  --embed-dim 256 \
  --freeze-stages 2 \
  --image-h 512 \
  --image-w 864 \
  --seed 0 \
  --device cuda \
  --temperature 0.12 \
  --val-pairs "$VAL_PAIRS" \
  --val-db "$VAL_DB" \
  --val-icp-csv "$VAL_ICP" \
  --val-invert-icp false \
  --val1-name val4_test4 \
  --best-on val1 \
  --best-by R@10 \
  --num-workers 4 \
  --amp \
  --scheduler cosine \
  --grad-clip 1.0 \
  --patience 0 \
  --jitter 0.2 \
  --train-ri-mode epoch_cycle \
  --train-ri-count 8 \
  --train-ri-glob "ri_*.png" \
  --train-ri-min 1 \
  --train-ri-seed 0 \
  --modality-stats "$MODALITY_STATS" \
  --save-every-epochs 1 \
  2>&1 | tee "$OUT_DIR/train.log"
