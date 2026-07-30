#!/usr/bin/env bash
set -euo pipefail
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
python tools/build_shimizu_frame_manifest.py \
  --input-manifest /mnt/w/32line/datasets/loftr_gt/manifest_cross_eval_gt_v3.csv \
  --line32-root /mnt/w/32line \
  --waymo-root /mnt/w/waymo_tf/perception_v1.4.3/individual_files \
  --output outputs/shimizu_frame_manifest.csv \
  "$@"
