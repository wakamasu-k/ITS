#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling"
PY="/home/wakamatsu/ITS/.venv_waymo/bin/python"
ROOT="/mnt/w/its_outputs/observed8_loftr"
mountpoint -q /mnt/w
mkdir -p "$ROOT"
RUN=$(mktemp -d "$ROOT/masked_label_pilot_XXXXXX")
printf 'Output: %s\n' "$RUN"
export CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4

(
  for index in 0 1; do
    readarray -t job < <("$PY" -c '
import json, sys
with open(sys.argv[1]) as file:
    jobs = [j for j in json.load(file)["jobs"] if j["role"] == "train"]
job = jobs[int(sys.argv[2])]
print(job["subset"])
print(job["segment"])
' "$PROJECT/outputs/observed8_localization_20261024/plan.json" "$index")
    [[ ${#job[@]} == 2 ]]
    DEPTH="$ROOT/depth_pilot"
    if [[ "$index" == 1 ]]; then
      DEPTH="$RUN/depth"
      "$PY" "$PROJECT/tools/prepare_observed8_loftr_depth.py" \
        --train-segment-index "$index" --out-root "$DEPTH"
    fi
    markers=("$DEPTH/${job[0]}/${job[1]}"/*/_done.json)
    [[ ${#markers[@]} == 1 && -f "${markers[0]}" ]]
    "$PY" "$PROJECT/tools/check_observed8_loftr_labels.py" \
      --anchor "$(dirname "${markers[0]}")" --out "$RUN/segment_$index" \
      --mask-camera-objects
  done
) 2>&1 | tee "$RUN/console.log"
printf 'Done: %s\n' "$RUN"
