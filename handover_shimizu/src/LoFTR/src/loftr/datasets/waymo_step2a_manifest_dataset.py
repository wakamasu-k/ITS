#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


def _read_gray(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"image not found: {path}")
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"cv2 failed to read: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _to_float01(img: np.ndarray) -> np.ndarray:
    if np.issubdtype(img.dtype, np.integer):
        mx = float(np.iinfo(img.dtype).max)
        mx = max(mx, 1.0)
        return (img.astype(np.float32) / mx).astype(np.float32)
    return img.astype(np.float32)


def _resize_keep_aspect(h: int, w: int, long_side: Optional[int]) -> Tuple[int, int]:
    if long_side is None or int(long_side) <= 0:
        return h, w
    long_side = int(long_side)
    s = float(long_side) / float(max(h, w))
    nh = max(1, int(round(h * s)))
    nw = max(1, int(round(w * s)))
    return nh, nw


def _floor_to_stride(x: int, stride: int) -> int:
    x2 = (int(x) // int(stride)) * int(stride)
    return max(int(stride), x2)


def _cell_offset(stride: int, cell_point: str) -> Tuple[float, float]:
    if cell_point == "topleft":
        return 0.0, 0.0
    if cell_point == "center":
        return float(stride) / 2.0, float(stride) / 2.0
    raise ValueError(f"unknown cell_point: {cell_point}")


def _remap_ids(ids: np.ndarray, old_h: int, old_w: int, new_h: int, new_w: int, stride: int, cell_point: str) -> np.ndarray:
    """
    coarse線形index（Hc*Wc系）を、リサイズ後のcoarse線形indexへ写像する。
    """
    if old_h == new_h and old_w == new_w:
        return ids.astype(np.int64)

    if (old_h % stride) != 0 or (old_w % stride) != 0:
        raise ValueError(f"old size not divisible by stride: old=({old_h},{old_w}) stride={stride}")
    if (new_h % stride) != 0 or (new_w % stride) != 0:
        raise ValueError(f"new size not divisible by stride: new=({new_h},{new_w}) stride={stride}")

    offx, offy = _cell_offset(stride, cell_point)

    old_Wc = old_w // stride
    new_Wc = new_w // stride
    new_Hc = new_h // stride

    ids = ids.astype(np.int64)
    y = (ids // old_Wc).astype(np.int64)
    x = (ids % old_Wc).astype(np.int64)

    # representative pixel coord
    u = x.astype(np.float64) * float(stride) + offx
    v = y.astype(np.float64) * float(stride) + offy

    sx = float(new_w) / float(old_w)
    sy = float(new_h) / float(old_h)
    u2 = u * sx
    v2 = v * sy

    x2 = np.rint((u2 - offx) / float(stride)).astype(np.int64)
    y2 = np.rint((v2 - offy) / float(stride)).astype(np.int64)

    x2 = np.clip(x2, 0, new_Wc - 1)
    y2 = np.clip(y2, 0, new_Hc - 1)
    return (y2 * new_Wc + x2).astype(np.int64)


class WaymoStep2AManifestDataset(Dataset):
    """
    Step2A v6 manifest（anc_png, cam_png, gt_npz, n_matches, error...）を読む。
    さらに resize-long を指定した場合は、画像をリサイズしつつ i_ids/j_ids を再マップする。
    v6 は「学習側でcoarseセルを再計算しない」思想なので、この再マップで整合を取る。:contentReference[oaicite:1]{index=1}
    """
    def __init__(
        self,
        manifest_csv: str,
        min_matches: int = 8,
        skip_error_rows: bool = True,
        verify_files: bool = True,
        resize_long_side: Optional[int] = None,
        cell_point: str = "topleft",
    ):
        self.manifest_csv = str(manifest_csv)
        self.min_matches = int(min_matches)
        self.skip_error_rows = bool(skip_error_rows)
        self.verify_files = bool(verify_files)
        self.resize_long_side = int(resize_long_side) if resize_long_side is not None else None
        self.cell_point = str(cell_point)

        path = Path(self.manifest_csv)
        if not path.exists():
            raise FileNotFoundError(f"manifest not found: {self.manifest_csv}")

        rows: List[Dict[str, Any]] = []
        dropped_err = dropped_low = dropped_miss = 0

        with open(path, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                err = (row.get("error", "") or "").strip()
                if self.skip_error_rows and err != "":
                    dropped_err += 1
                    continue
                n_matches = int(float(row.get("n_matches", "0") or "0"))
                if n_matches < self.min_matches:
                    dropped_low += 1
                    continue

                anc = (row.get("anc_png", "") or "").strip()
                cam = (row.get("cam_png", "") or "").strip()
                gt  = (row.get("gt_npz", "") or "").strip()
                if not anc or not cam or not gt:
                    dropped_miss += 1
                    continue
                if self.verify_files:
                    if (not Path(anc).exists()) or (not Path(cam).exists()) or (not Path(gt).exists()):
                        dropped_miss += 1
                        continue

                rows.append({"anc_png": anc, "cam_png": cam, "gt_npz": gt})

        self.rows = rows
        print("[dataset] manifest:", self.manifest_csv)
        print("[dataset] total usable rows:", len(self.rows))
        print("[dataset] dropped_error_rows:", dropped_err)
        print("[dataset] dropped_low_matches:", dropped_low)
        print("[dataset] dropped_missing_files/fields:", dropped_miss)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        gt_path = row["gt_npz"]

        with np.load(gt_path, allow_pickle=False) as z:
            for k in ["i_ids", "j_ids", "hw0", "hw1", "stride"]:
                if k not in z.files:
                    raise KeyError(f"gt_npz missing {k}: {gt_path}")
            i_ids = z["i_ids"].astype(np.int64)
            j_ids = z["j_ids"].astype(np.int64)
            hw0 = z["hw0"].astype(np.int32)
            hw1 = z["hw1"].astype(np.int32)
            stride = int(z["stride"])

        h0, w0 = int(hw0[0]), int(hw0[1])
        h1, w1 = int(hw1[0]), int(hw1[1])

        img0 = _to_float01(_read_gray(row["anc_png"]))
        img1 = _to_float01(_read_gray(row["cam_png"]))

        if img0.shape[:2] != (h0, w0):
            raise ValueError(f"size mismatch image0 vs gt hw0: img={img0.shape[:2]} gt={(h0,w0)} gt={gt_path}")
        if img1.shape[:2] != (h1, w1):
            raise ValueError(f"size mismatch image1 vs gt hw1: img={img1.shape[:2]} gt={(h1,w1)} gt={gt_path}")

        nh0, nw0 = _resize_keep_aspect(h0, w0, self.resize_long_side)
        nh1, nw1 = _resize_keep_aspect(h1, w1, self.resize_long_side)

        # stride(=8) に揃える（coarse grid 整合のため）
        nh0 = _floor_to_stride(nh0, stride)
        nw0 = _floor_to_stride(nw0, stride)
        nh1 = _floor_to_stride(nh1, stride)
        nw1 = _floor_to_stride(nw1, stride)

        if (nh0, nw0) != (h0, w0):
            img0 = cv2.resize(img0, (nw0, nh0), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        if (nh1, nw1) != (h1, w1):
            img1 = cv2.resize(img1, (nw1, nh1), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        i2 = _remap_ids(i_ids, h0, w0, nh0, nw0, stride, self.cell_point)
        j2 = _remap_ids(j_ids, h1, w1, nh1, nw1, stride, self.cell_point)

        # 重複削除（丸めで被る）
        if i2.size > 0:
            pairs = np.unique(np.stack([i2, j2], axis=1), axis=0)
            i2 = pairs[:, 0]
            j2 = pairs[:, 1]

        t0 = torch.from_numpy(img0).float().unsqueeze(0)  # (1,H,W)
        t1 = torch.from_numpy(img1).float().unsqueeze(0)

        return {
            "image0": t0,
            "image1": t1,
            "i_ids_gt": torch.from_numpy(i2).long(),
            "j_ids_gt": torch.from_numpy(j2).long(),
            "stride": int(stride),
        }
