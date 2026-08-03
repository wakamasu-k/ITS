#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python tools/build_training_segment_manifest.py \
  --waymo-root /mnt/w/waymo_tf/perception_v1.4.3/individual_files \
  --evaluation-manifest outputs/shimizu_frame_manifest.csv \
  --output outputs/training_segment_manifest.csv \
  "$@"
