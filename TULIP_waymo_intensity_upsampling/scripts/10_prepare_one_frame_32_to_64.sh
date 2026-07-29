#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling"

cd "${PROJECT_ROOT}"
python tools/prepare_32_to_64_pair.py \
  --input outputs/one_frame_000000/waymo_top_frame_000000.npz \
  --output outputs/one_frame_000000/tulip_pair_32_to_64.npz \
  --metadata-output outputs/one_frame_000000/tulip_pair_32_to_64.json
