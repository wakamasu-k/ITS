#!/usr/bin/env bash
set -euo pipefail

# Build A/B/C maps for the five DB drives in Shimizu's 507Q/754DB protocol.
# The default is a dry run. Pass --execute to perform inference.

MODE="dry-run"
if [[ "${1:-}" == "--execute" ]]; then
  MODE="execute"
elif [[ $# -gt 0 && "${1:-}" != "--dry-run" ]]; then
  echo "Usage: $0 [--dry-run|--execute]" >&2
  exit 2
fi

PROJECT_ROOT="/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling"
PYTHON="/home/wakamatsu/ITS/.venv_tulip/bin/python"
TFRECORD_ROOT="/mnt/w/waymo_tf/perception_v1.4.3/individual_files/training"
TULIP_ROOT="/home/wakamatsu/ITS/TULIP_backup/tulip"
CHECKPOINT="$PROJECT_ROOT/outputs/waymo_16_to_32_baseline_to_moderate_s0_e20/checkpoint.pth"
OUTPUT_ROOT="$PROJECT_ROOT/outputs/shimizu_protocol_507q_754db_abc/maps"
BUILDER="$PROJECT_ROOT/tools/build_tulip_16_to_32_static_map.py"

SEGMENTS=(
  "segment-6410495600874495447_5287_500_5307_500_with_camera_labels"
  "segment-12844373518178303651_2140_000_2160_000_with_camera_labels"
  "segment-13823509240483976870_1514_190_1534_190_with_camera_labels"
  "segment-12681651284932598380_3585_280_3605_280_with_camera_labels"
  "segment-8487809726845917818_4779_870_4799_870_with_camera_labels"
)

for required in "$PYTHON" "$CHECKPOINT" "$BUILDER"; do
  if [[ ! -f "$required" ]]; then
    echo "[ERROR] required file not found: $required" >&2
    exit 1
  fi
done
if [[ ! -d "$TULIP_ROOT" ]]; then
  echo "[ERROR] TULIP root not found: $TULIP_ROOT" >&2
  exit 1
fi

for segment in "${SEGMENTS[@]}"; do
  tfrecord="$TFRECORD_ROOT/$segment.tfrecord"
  if [[ ! -f "$tfrecord" ]]; then
    echo "[ERROR] TFRecord not found: $tfrecord" >&2
    exit 1
  fi

  complete=true
  for variant in A_observed16 B_tulip32 C_gt32; do
    if [[ ! -f "$OUTPUT_ROOT/$variant/training/$segment/map_static.npz" ]]; then
      complete=false
    fi
  done
  if [[ "$complete" == "true" ]]; then
    echo "[SKIP] all A/B/C maps already exist: $segment"
    continue
  fi

  command=(
    "$PYTHON" "$BUILDER"
    --tfrecord "$tfrecord"
    --subset training
    --tulip-root "$TULIP_ROOT"
    --checkpoint "$CHECKPOINT"
    --output-root "$OUTPUT_ROOT"
    --range-scale-m 75
    --intensity-transform log1p
    --device cuda
    --map-variants A B C
  )
  if [[ "$MODE" == "dry-run" ]]; then
    command+=(--dry-run)
  fi

  echo "[RUN] mode=$MODE segment=$segment"
  TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONPATH="$PROJECT_ROOT" "${command[@]}"
done

echo "[DONE] mode=$MODE output_root=$OUTPUT_ROOT"
