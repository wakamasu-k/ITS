#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling"
ROOT="$PROJECT/outputs/observed8_epoch052_top10_stage1_fp16_20261029"
SCOPE="${1:-pilot10}"

case "$SCOPE" in
  pilot10|all507) ;;
  *) printf 'Usage: bash %s [pilot10|all507]\n' "$0" >&2; exit 2 ;;
esac

if [[ "$SCOPE" == all507 && ! -s "$ROOT/pilot10_pnp/integrated_summary.json" ]]; then
  printf 'Run pilot10 first.\n' >&2
  exit 1
fi

LOG=$(mktemp "$ROOT/${SCOPE}_console_XXXXXX.log")
printf 'Log: %s\n' "$LOG"
/home/wakamatsu/ITS/.venv_waymo/bin/python -u \
  "$PROJECT/tools/evaluate_retrieval_topk_pipeline.py" run \
  --root "$ROOT" --scope "$SCOPE" 2>&1 | tee "$LOG"
