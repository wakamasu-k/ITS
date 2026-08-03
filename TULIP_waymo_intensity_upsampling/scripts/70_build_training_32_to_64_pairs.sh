#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python tools/build_training_32_to_64_pairs.py \
  --frame-index outputs/training_frames_64_32/index.csv \
  --pair-index outputs/training_pairs_32_to_64/index.csv \
  --failures outputs/training_pairs_32_to_64/failures.csv \
  "$@"
