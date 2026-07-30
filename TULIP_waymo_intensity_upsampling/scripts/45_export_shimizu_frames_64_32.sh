#!/usr/bin/env bash
set -euo pipefail
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
python waymo_preprocess/export_shimizu_frames_64_32.py \
  --manifest outputs/shimizu_frame_manifest.csv \
  --output-root outputs/shimizu_frames_64_32 \
  "$@"
