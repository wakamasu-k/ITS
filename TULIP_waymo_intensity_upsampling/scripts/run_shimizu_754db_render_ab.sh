#!/usr/bin/env bash
set -euo pipefail

# Render A/B anchor databases for Shimizu's 507Q/754DB protocol.
# Source TFRecords and reference metadata are read-only. Outputs are isolated
# under the TULIP project. Default mode is dry-run.

MODE="dry-run"
if [[ "${1:-}" == "--execute" ]]; then
  MODE="execute"
elif [[ $# -gt 0 && "${1:-}" != "--dry-run" ]]; then
  echo "Usage: $0 [--dry-run|--execute]" >&2
  exit 2
fi

PROJECT_ROOT="/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling"
PYTHON="/home/wakamatsu/ITS/.venv_waymo/bin/python"
RENDERER="/home/wakamatsu/ITS/handover_shimizu/src/data_make/valid_test/render_anchor_intensity_txt_se3_fix.py"
TFRECORD_ROOT="/mnt/w/waymo_tf/perception_v1.4.3/individual_files/training"
REFERENCE_ROOT="/mnt/w/32line/datasets/submaps/cross/original_val_test_all/db/training"
RUN_ROOT="$PROJECT_ROOT/outputs/shimizu_protocol_507q_754db_abc"
MAPS_ROOT="$RUN_ROOT/maps"
LOG_ROOT="$RUN_ROOT/render_logs"
MAX_JOBS="${MAX_JOBS:-5}"

SEGMENTS=(
  "segment-6410495600874495447_5287_500_5307_500_with_camera_labels"
  "segment-12844373518178303651_2140_000_2160_000_with_camera_labels"
  "segment-13823509240483976870_1514_190_1534_190_with_camera_labels"
  "segment-12681651284932598380_3585_280_3605_280_with_camera_labels"
  "segment-8487809726845917818_4779_870_4799_870_with_camera_labels"
)

declare -A MAP_VARIANT=( [A]="A_observed16" [B]="B_tulip32" )
declare -A ANCHOR_ROOT=( [A]="anchors_A" [B]="anchors_B" )

copy_metadata_if_needed() {
  local variant="$1"
  local segment="$2"
  local source="$REFERENCE_ROOT/$segment"
  local destination="$RUN_ROOT/${ANCHOR_ROOT[$variant]}/training/$segment"
  mkdir -p "$destination"
  for name in anchors.csv frames_poses.csv frames_submap_ids.csv submaps.json; do
    if [[ ! -f "$source/$name" ]]; then
      echo "[ERROR] missing reference metadata: $source/$name" >&2
      exit 1
    fi
    if [[ ! -f "$destination/$name" ]]; then
      cp "$source/$name" "$destination/$name"
    elif ! cmp -s "$source/$name" "$destination/$name"; then
      echo "[ERROR] existing metadata differs; refusing overwrite: $destination/$name" >&2
      exit 1
    fi
  done
}

render_one() {
  local variant="$1"
  local segment="$2"
  local anchor_root="$RUN_ROOT/${ANCHOR_ROOT[$variant]}"
  local map_root="$MAPS_ROOT/${MAP_VARIANT[$variant]}"
  local tfrecord="$TFRECORD_ROOT/$segment.tfrecord"
  local expected
  expected=$(awk -F, 'NR==2 {print $1}' "$anchor_root/training/$segment/anchors.csv" >/dev/null; wc -l < "$anchor_root/training/$segment/anchors.csv")
  expected=$((expected - 1))
  local actual
  actual=$(find "$anchor_root/training/$segment" -mindepth 2 -maxdepth 2 -type f -name anchor.png | wc -l)
  if [[ "$actual" -eq "$expected" ]]; then
    echo "[SKIP] variant=$variant segment=$segment anchors=$actual/$expected"
    return
  fi
  echo "[RUN] variant=$variant segment=$segment anchors=$actual/$expected mode=$MODE"
  if [[ "$MODE" == "dry-run" ]]; then
    return
  fi
  TF_FORCE_GPU_ALLOW_GROWTH=true "$PYTHON" "$RENDERER" \
    --tfrecord "$tfrecord" \
    --subset training \
    --maps-root "$map_root" \
    --submaps-root "$anchor_root" \
    --cam FRONT \
    --int-percentile 99.5 \
    --gamma 1.0 \
    --int-mode global_map \
    --range-corr-power 0.0 \
    --point-size 2 \
    --write-index true \
    --crop-mode none \
    --crop-radius-m 10.0 \
    --crop-margin-m 2.0 \
    --post-hist-eq true \
    --post-clahe true \
    --clahe-clip-limit 2.0 \
    --clahe-tile-grid 8 \
    --post-mask-covered-only true \
    --overwrite false
}

mkdir -p "$LOG_ROOT"
for variant in A B; do
  for segment in "${SEGMENTS[@]}"; do
    copy_metadata_if_needed "$variant" "$segment"
  done
done

running=0
for variant in A B; do
  for segment in "${SEGMENTS[@]}"; do
    log="$LOG_ROOT/${variant}_${segment}.log"
    render_one "$variant" "$segment" >"$log" 2>&1 &
    running=$((running + 1))
    if [[ "$running" -ge "$MAX_JOBS" ]]; then
      wait -n
      running=$((running - 1))
    fi
  done
done
wait

echo "[DONE] mode=$MODE logs=$LOG_ROOT"
