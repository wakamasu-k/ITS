#!/usr/bin/env bash
set -u

# This script is intentionally read-only. It reports which runtime can be used
# before TFRecord preprocessing and TULIP inference are connected.
echo "project=/home/wakamatsu/ITS/TULIP_waymo_intensity_upsampling"
PYTHON_BIN="$(command -v python || command -v python3 || true)"
echo "python=${PYTHON_BIN}"
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "error=python and python3 are both missing"
    exit 1
fi
"${PYTHON_BIN}" --version 2>&1

"${PYTHON_BIN}" - <<'PY'
import importlib.util

for name in ("numpy", "torch", "tensorflow", "waymo_open_dataset"):
    available = importlib.util.find_spec(name) is not None
    print(f"{name}={'available' if available else 'missing'}")
PY
