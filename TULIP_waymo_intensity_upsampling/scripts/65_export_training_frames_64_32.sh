#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python waymo_preprocess/export_training_frames_64_32.py \
  --segment-manifest outputs/training_segment_manifest.csv \
  --evaluation-manifest outputs/shimizu_frame_manifest.csv \
  --output-root outputs/training_frames_64_32 \
  --frame-manifest outputs/training_frames_64_32/index.csv \
  --failures outputs/training_frames_64_32/failures.csv \
  "$@"
