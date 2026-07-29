#!/usr/bin/env bash
set -euo pipefail

source "$HOME/ITS/.venv_waymo/bin/activate"

OUT_DIR="$HOME/ITS/waymo_outputs/timing_recheck_15"
mkdir -p "$OUT_DIR"

CKPT="$HOME/ITS/waymo_outputs/old32_retrieval/train_wdrive_paper80_b16/checkpoints/epoch_058.ckpt"
PAIRS="$HOME/ITS/waymo_outputs/timing_recheck_15/mini_retrieval_manifest/test4_pairs_15.parquet"
DB="$HOME/ITS/waymo_outputs/timing_recheck_15/mini_retrieval_manifest/test4_db_15.parquet"
ICP="/mnt/w/32line/datasets/splits/cross/all/icp_startend_staticmaps.csv"

python evaluate_timing_15.py \
  --checkpoint "$CKPT" \
  --pairs "$PAIRS" \
  --db "$DB" \
  --icp_csv "$ICP" \
  --out_dir "$OUT_DIR/retrieval_15" \
  --name "paper80_epoch058_timing15" \
  --backbone resnet34 \
  --clusters 64 \
  --embed_dim 256 \
  --image_h 512 \
  --image_w 864 \
  --batch_size 1 \
  --num_workers 0 \
  --device cuda \
  --amp true \
  --max_queries 15 \
  2>&1 | tee "$OUT_DIR/netvlad_timing_15.log"
