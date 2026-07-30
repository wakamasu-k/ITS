#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling"

cd "${PROJECT_ROOT}"
python waymo_preprocess/analyze_waymo_32_to_64_statistics.py \
  --input-root /mnt/w/waymo_tf/perception_v1.4.3/individual_files/training \
  --output-dir outputs/statistics_32_to_64 \
  --max-scenes 10 \
  --frames-per-scene 20 \
  --sample-values-per-frame 10000
