#!/usr/bin/env bash

# This file must be sourced so the activated environment remains in the
# caller's shell:
#   source scripts/01_activate_waymo_gpu.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Run this script with: source scripts/01_activate_waymo_gpu.sh"
    exit 2
fi

WAYMO_VENV="/home/wakamatsu/ITS/.venv_waymo"
WAYMO_SITE="${WAYMO_VENV}/lib/python3.10/site-packages"
WAYMO_CUDNN_LIB="${WAYMO_SITE}/nvidia/cudnn/lib"
WAYMO_CUBLAS_LIB="${WAYMO_SITE}/nvidia/cublas/lib"
WAYMO_CUDA_LIB="/usr/local/cuda/targets/x86_64-linux/lib"
WSL_CUDA_DRIVER_LIB="/usr/lib/wsl/lib"

source "${WAYMO_VENV}/bin/activate"

for required_dir in \
    "${WAYMO_CUDNN_LIB}" \
    "${WAYMO_CUBLAS_LIB}" \
    "${WAYMO_CUDA_LIB}" \
    "${WSL_CUDA_DRIVER_LIB}"; do
    if [[ ! -d "${required_dir}" ]]; then
        echo "Missing GPU library directory: ${required_dir}" >&2
        return 1
    fi
done

export LD_LIBRARY_PATH="${WAYMO_CUDNN_LIB}:${WAYMO_CUBLAS_LIB}:${WAYMO_CUDA_LIB}:${WSL_CUDA_DRIVER_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

python - <<'PY'
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print(f"tensorflow={tf.__version__}")
print(f"tensorflow_gpus={gpus}")
if not gpus:
    raise SystemExit("TensorFlow GPU is unavailable.")
print("waymo_gpu_environment=ready")
PY
