#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling"
cd "${PROJECT_ROOT}"
python tools/build_handover_frame_manifest.py \
  --input-manifest /mnt/w/32line/datasets/loftr_pairs_shifted/training/manifest_matches_step2a_paper_final.csv \
  --line32-root /mnt/w/32line \
  --waymo-root /mnt/w/waymo_tf/perception_v1.4.3/individual_files \
  --output outputs/handover_frames.csv \
  "$@"
