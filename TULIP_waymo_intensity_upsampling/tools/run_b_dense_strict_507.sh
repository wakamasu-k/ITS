#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling
REF=/home/wakamatsu/ITS/handover_shimizu/src/LoFTR
PYTHON=/home/wakamatsu/ITS/.venv_tulip/bin/python
MANIFEST="$PROJECT/outputs/B_dense_distance_protocol_20261019/B_strict_507.csv"
CKPT=/home/wakamatsu/ITS/waymo_outputs/old32_dense/loftr_retrain_stage1_paper_wdrive/stage1_final_gstep715958.pth
ICP=/mnt/w/32line/datasets/splits/cross/all/icp_startend_staticmaps.csv
OUT="$PROJECT/outputs/B_dense_strict_507_stage1_fp16"

if [[ ! -f "$MANIFEST" || ! -f "$CKPT" || ! -f "$ICP" ]]; then
    echo 'Required manifest, checkpoint, or ICP file is missing.' >&2
    exit 1
fi
if [[ -e "$OUT/per_pair.csv" || -e "$OUT/summary.json" ]]; then
    echo "Output already exists; refusing to overwrite: $OUT" >&2
    exit 1
fi

PYTHONPATH="$REF/src" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
"$PYTHON" "$REF/tools/eval_cross_match_to_pose_zalign_multi_visall_v2.py" \
    --manifest "$MANIFEST" \
    --ckpt "$CKPT" \
    --out-dir "$OUT" \
    --device cuda \
    --amp-dtype fp16 \
    --gt-type strict \
    --icp-csv "$ICP"

"$PYTHON" - "$OUT/per_pair.csv" <<'PY'
import csv
import statistics
import sys

with open(sys.argv[1], newline='') as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 507:
    raise SystemExit(f'Expected 507 rows, got {len(rows)}')
print('Strict pairs:', len(rows))
print('LoFTR matches median:', statistics.median(int(r['n_match_in']) for r in rows))
print('3D correspondences median:', statistics.median(int(r['n_3d']) for r in rows))
print('PnP success:', sum(r['pnp_success'] == 'True' for r in rows) / len(rows))
print('Zero-match pairs:', sum(int(r['n_match_in']) == 0 for r in rows))
PY
