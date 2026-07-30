#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling"

cd "${PROJECT_ROOT}"
python tools/visualize_waymo_32_to_64_pair.py \
  --input outputs/one_frame_000000/tulip_pair_32_to_64.npz \
  --output-dir outputs/one_frame_000000/visualization \
  --range-max 75 \
  --intensity-max 0.75 \
  --intensity-log-max 22016
