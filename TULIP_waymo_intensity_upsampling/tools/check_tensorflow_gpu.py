#!/usr/bin/env python3
"""Fail unless TensorFlow can execute a real operation on GPU:0."""

from __future__ import annotations

import tensorflow as tf


def main() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("TensorFlow does not detect a GPU.")

    with tf.device("/GPU:0"):
        left = tf.random.uniform((1024, 1024), seed=1)
        right = tf.random.uniform((1024, 1024), seed=2)
        result = tf.matmul(left, right)
        checksum = float(tf.reduce_sum(result))

    if "GPU:0" not in result.device:
        raise RuntimeError(f"TensorFlow operation ran on unexpected device: {result.device}")
    print(f"tensorflow={tf.__version__}")
    print(f"gpu={gpus[0]}")
    print(f"operation_device={result.device}")
    print(f"checksum={checksum}")
    print("tensorflow_gpu_test=passed")


if __name__ == "__main__":
    main()
