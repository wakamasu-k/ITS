#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python tools/evaluate_range_only_tulip.py \
  --index outputs/training_pairs_32_to_64/index.csv \
  --tulip-root /home/wakamatsu/ITS/TULIP_backup/tulip \
  --checkpoint outputs/range_only_overfit/checkpoint.pth \
  --output-root outputs/range_only_evaluation \
  "$@"
