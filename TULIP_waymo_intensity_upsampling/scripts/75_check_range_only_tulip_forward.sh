#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python tools/check_range_only_tulip_forward.py \
  --index outputs/training_pairs_32_to_64/index.csv \
  --tulip-root /home/wakamatsu/ITS/TULIP_backup/tulip \
  "$@"
