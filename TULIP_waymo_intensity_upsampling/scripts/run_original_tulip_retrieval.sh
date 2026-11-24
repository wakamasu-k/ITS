#!/usr/bin/env bash
# Stage-by-stage original TULIP B adaptation; no observed-priority composition.
set -euo pipefail
PROJECT="/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling"
PY="/home/wakamatsu/ITS/.venv_waymo/bin/python"
EXPERIMENT_ROOT="${ORIGINAL_B_ROOT:-$PROJECT/outputs/original_tulip_B_retraining_20261029}"
ACTION="${1:-status}"
if [ "$#" -gt 0 ]; then shift; fi
exec "$PY" "$PROJECT/tools/run_original_tulip_retrieval.py" \
  "$ACTION" --root "$EXPERIMENT_ROOT" "$@"
