#!/usr/bin/env bash
set -euo pipefail
cd /home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
python tools/audit_shimizu_frames.py \
  --manifest outputs/shimizu_frame_manifest.csv \
  --input-root outputs/shimizu_frames_64_32 \
  --report-dir outputs/shimizu_frames_audit \
  "$@"
