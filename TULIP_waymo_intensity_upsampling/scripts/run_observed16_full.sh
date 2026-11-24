#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling"
SOURCE="$PROJECT/outputs/observed16_localization"
DEST="/mnt/w/its_outputs/observed16_localization"
PY="/home/wakamatsu/ITS/.venv_waymo/bin/python"
SCRIPT="$PROJECT/tools/run_observed16_retrieval.py"

mountpoint -q /mnt/w
mkdir -p "$DEST"
exec 9>"$DEST/.full_run.lock"
flock -n 9 || { printf 'Another full run is active.\n' >&2; exit 1; }

if [[ ! -f "$DEST/.initial_copy_done" ]]; then
    rsync -rt --info=progress2 "$SOURCE/" "$DEST/"
    touch "$DEST/.initial_copy_done"
fi

LOG=$(mktemp "$DEST/full_console_XXXXXX.log")
printf 'Log: %s\n' "$LOG"
exec > >(tee -a "$LOG") 2>&1
for stage in data manifests stats check train select evaluate; do
    printf '\n[%s] %s\n' "$(date -Is)" "$stage"
    "$PY" -u "$SCRIPT" "$stage" --root "$DEST"
done
