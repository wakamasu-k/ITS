#!/usr/bin/env python3
"""Run a one-sample Waymo 16→32 TULIP forward smoke test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tulip_adapter.waymo_16_to_32_dataset import Waymo16To32Dataset
from tulip_adapter.waymo_16_to_32_tulip import build_waymo_16_to_32_model


def parse_args() -> argparse.Namespace:
    """Parse smoke-test arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--tulip-root", required=True, type=Path)
    parser.add_argument("--dataset-role", default="train")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    """Run one forward, loss, finite-value, and native-width crop check."""
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    dataset = Waymo16To32Dataset(args.index, dataset_role=args.dataset_role)
    sample = dataset[0]
    model = build_waymo_16_to_32_model(args.tulip_root).to(device).eval()
    input_signal = sample["input"].unsqueeze(0).to(device)
    target_signal = sample["target"].unsqueeze(0).to(device)
    target_mask = sample["target_mask"].unsqueeze(0).to(device)
    with torch.inference_mode():
        prediction, loss = model(input_signal, target_signal, target_mask)
    result = {
        "device": str(device),
        "input_shape": list(input_signal.shape),
        "target_shape": list(target_signal.shape),
        "output_shape": list(prediction.shape),
        "crop_shape": list(model.crop_native(prediction).shape),
        "loss": float(loss),
        "output_finite": bool(torch.isfinite(prediction).all()),
        "loss_finite": bool(torch.isfinite(loss)),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["output_shape"] != [1, 2, 32, 2688]:
        raise RuntimeError(f"unexpected output shape: {result['output_shape']}")
    if result["crop_shape"] != [1, 2, 32, 2650]:
        raise RuntimeError(f"unexpected crop shape: {result['crop_shape']}")
    if not result["output_finite"] or not result["loss_finite"]:
        raise RuntimeError("output or loss contains NaN/Inf")


if __name__ == "__main__":
    main()
