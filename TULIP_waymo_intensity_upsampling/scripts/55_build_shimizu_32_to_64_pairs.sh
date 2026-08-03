#!/usr/bin/env bash
set -euo pipefail
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
python tools/build_shimizu_32_to_64_pairs.py \
  --manifest outputs/shimizu_frame_manifest.csv \
  --input-root outputs/shimizu_frames_64_32 \
  --index-output outputs/shimizu_pairs_32_to_64/index.csv \
  --failures outputs/shimizu_pairs_32_to_64/failures.csv \
  "$@"
