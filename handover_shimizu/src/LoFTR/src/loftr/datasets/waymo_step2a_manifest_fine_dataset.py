#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Waymo Step2A (paper_final v6) manifest dataset for LoFTR Stage-2 (fine enabled).

- Reads:
    - anc_png (reflectivity / intensity)  uint8 gray (H,W)
    - cam_png (camera gray)              uint8 gray (H,W)
    - gt_npz  (produced by make_loftr_matches_gt_step2a_paper_final_fixed_cam_v6.py)

- Outputs per sample:
    sample = {
        "image0": torch.FloatTensor [1,H0,W0]  (0..1),
        "image1": torch.FloatTensor [1,H1,W1]  (0..1),
        "i_ids_gt": torch.LongTensor [M]       (coarse cell indices, resized domain),
        "j_ids_gt": torch.LongTensor [M],
        "expec_f_gt": torch.FloatTensor [M,2]  (normalized offsets for fine loss; [-1,1] expected),
        "stride": int                          (coarse stride, typically 8),
        "pair_name": str                       (debug),
    }

Notes:
- This dataset re-computes (i_ids, j_ids, fine offsets) AFTER resize.
  That keeps supervision consistent with training input resolution.
- It assumes cell_point="topleft" by default (matching our Step2A GT generation).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Avoid OpenCV over-threading when using DataLoader workers
try:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


def _read_gray_u8(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"cv2.imread failed: {path}")
    if img.ndim == 3:
        # BGR -> gray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        # keep as uint8-like for now; later normalize
        img = img.astype(np.uint8)
    return img


def _round_to_multiple(x: float, m: int) -> int:
    # nearest multiple (ties -> up)
    return int(max(m, math.floor((x + m / 2.0) / m) * m))


def _compute_resize_hw(h: int, w: int, long_side: Optional[int], divisor: int) -> Tuple[int, int]:
    if long_side is None or long_side <= 0:
        h2, w2 = h, w
    else:
        if w >= h:
            s = float(long_side) / float(w)
        else:
            s = float(long_side) / float(h)
        h2 = int(round(h * s))
        w2 = int(round(w * s))

    h2 = _round_to_multiple(h2, divisor)
    w2 = _round_to_multiple(w2, divisor)
    h2 = max(divisor, h2)
    w2 = max(divisor, w2)
    return h2, w2


def _resize_u8(img_u8: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    h2, w2 = int(hw[0]), int(hw[1])
    h, w = img_u8.shape[:2]
    if (h2, w2) == (h, w):
        return img_u8
    return cv2.resize(img_u8, (w2, h2), interpolation=cv2.INTER_AREA)


def _to_tensor01(img_u8: np.ndarray) -> torch.Tensor:
    # [H,W] uint8 -> [1,H,W] float32 in 0..1
    x = torch.from_numpy(np.ascontiguousarray(img_u8)).float() / 255.0
    return x.unsqueeze(0)


def _cell_offsets(cell_point: str, stride: int) -> Tuple[float, float]:
    cp = (cell_point or "topleft").lower()
    if cp in ("topleft", "tl", "top_left"):
        return 0.0, 0.0
    if cp in ("center", "centre"):
        return float(stride) * 0.5, float(stride) * 0.5
    raise ValueError(f"Unsupported cell_point: {cell_point}")


class WaymoStep2AFineManifestDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str,
        *,
        min_matches: int = 8,
        min_fine_ok: int = 200,
        skip_error_rows: bool = True,
        verify_files: bool = True,
        resize_long_side: Optional[int] = 840,
        # GT / model consistency
        coarse_stride: int = 8,
        fine_stride: int = 2,
        fine_window_size: int = 5,
        cell_point: str = "topleft",
        only_fine_ok: bool = True,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.manifest_csv = str(manifest_csv)
        self.min_matches = int(min_matches)
        self.min_fine_ok = int(min_fine_ok)
        self.skip_error_rows = bool(skip_error_rows)
        self.verify_files = bool(verify_files)
        self.resize_long_side = int(resize_long_side) if resize_long_side is not None else None

        self.coarse_stride = int(coarse_stride)
        self.fine_stride = int(fine_stride)
        self.fine_window_size = int(fine_window_size)
        if self.fine_window_size % 2 != 1:
            raise ValueError("fine_window_size must be odd.")
        self.radius = self.fine_window_size // 2

        self.cell_point = str(cell_point)
        self.only_fine_ok = bool(only_fine_ok)
        self.verbose = bool(verbose)

        self.rows: List[Dict[str, str]] = []
        self._load_manifest()

    def _load_manifest(self) -> None:
        path = Path(self.manifest_csv)
        if not path.exists():
            raise FileNotFoundError(f"manifest not found: {path}")

        kept = 0
        dropped_error = 0
        dropped_low = 0
        dropped_missing = 0

        with path.open("r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if self.skip_error_rows and (row.get("error", "") or "").strip() != "":
                    dropped_error += 1
                    continue

                try:
                    n_matches = int(float(row.get("n_matches", "0") or "0"))
                except Exception:
                    n_matches = 0
                try:
                    n_fine = int(float(row.get("n_fine", "0") or "0"))
                except Exception:
                    n_fine = 0

                if n_matches < self.min_matches:
                    dropped_low += 1
                    continue
                # For stage2 we want enough fine-supervised GT
                if n_fine < self.min_fine_ok:
                    dropped_low += 1
                    continue

                anc = row.get("anc_png", "")
                cam = row.get("cam_png", "")
                gt = row.get("gt_npz", "")

                if self.verify_files:
                    ok = True
                    for p in (anc, cam, gt):
                        if not p or (not Path(p).exists()):
                            ok = False
                            break
                    if not ok:
                        dropped_missing += 1
                        continue

                self.rows.append(row)
                kept += 1

        if self.verbose:
            print(f"[dataset] manifest: {self.manifest_csv}")
            print(f"[dataset] total usable rows: {kept}")
            print(f"[dataset] dropped_error_rows: {dropped_error}")
            print(f"[dataset] dropped_low_matches/fine: {dropped_low}")
            print(f"[dataset] dropped_missing_files/fields: {dropped_missing}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[int(idx)]
        anc_path = row["anc_png"]
        cam_path = row["cam_png"]
        gt_path = row["gt_npz"]

        # ---- read images ----
        img0_u8 = _read_gray_u8(anc_path)
        img1_u8 = _read_gray_u8(cam_path)

        h0, w0 = img0_u8.shape[:2]
        h1, w1 = img1_u8.shape[:2]

        # ---- resize (keep aspect; make divisible by coarse stride) ----
        h0r, w0r = _compute_resize_hw(h0, w0, self.resize_long_side, self.coarse_stride)
        h1r, w1r = _compute_resize_hw(h1, w1, self.resize_long_side, self.coarse_stride)

        img0r_u8 = _resize_u8(img0_u8, (h0r, w0r))
        img1r_u8 = _resize_u8(img1_u8, (h1r, w1r))

        sx0, sy0 = float(w0r) / float(w0), float(h0r) / float(h0)
        sx1, sy1 = float(w1r) / float(w1), float(h1r) / float(h1)

        # ---- load GT npz ----
        npz = np.load(gt_path, allow_pickle=False)

        # meta consistency check (best-effort)
        if "meta_json" in npz:
            try:
                meta = json.loads(str(npz["meta_json"].tolist()))
                # only warn, don't crash (user may regenerate)
                stride_gt = int(meta.get("stride", self.coarse_stride))
                fine_stride_gt = int(meta.get("fine_stride", self.fine_stride))
                fine_w_gt = int(meta.get("fine_window_size", self.fine_window_size))
                cell_point_gt = str(meta.get("cell_point", self.cell_point))
                if stride_gt != self.coarse_stride or fine_stride_gt != self.fine_stride or fine_w_gt != self.fine_window_size:
                    # small, but make it visible in logs
                    print(
                        f"[WARN][gt-meta] mismatch stride/fine: "
                        f"gt(stride={stride_gt}, fine_stride={fine_stride_gt}, fine_w={fine_w_gt}) "
                        f"!= dataset(stride={self.coarse_stride}, fine_stride={self.fine_stride}, fine_w={self.fine_window_size})"
                    )
                if cell_point_gt.lower() != self.cell_point.lower():
                    print(
                        f"[WARN][gt-meta] cell_point mismatch: gt={cell_point_gt} != dataset={self.cell_point}"
                    )
            except Exception:
                pass

        mkpts0 = npz["mkpts0"].astype(np.float32)      # [M,2] (x,y) in original image0 pixels
        mkpts1_f = npz["mkpts1_f"].astype(np.float32)  # [M,2] (x,y) in original image1 pixels

        # ---- scale points to resized domain ----
        mkpts0r = mkpts0.copy()
        mkpts0r[:, 0] *= sx0
        mkpts0r[:, 1] *= sy0

        mkpts1fr = mkpts1_f.copy()
        mkpts1fr[:, 0] *= sx1
        mkpts1fr[:, 1] *= sy1

        # ---- compute coarse indices in resized domain ----
        offx, offy = _cell_offsets(self.cell_point, self.coarse_stride)

        # image0: src cell from (scaled) representative pixel
        ix = np.rint((mkpts0r[:, 0] - offx) / float(self.coarse_stride)).astype(np.int64)
        iy = np.rint((mkpts0r[:, 1] - offy) / float(self.coarse_stride)).astype(np.int64)

        # image1: dst cell from (scaled) continuous reprojection (more stable after resize)
        jx = np.rint((mkpts1fr[:, 0] - offx) / float(self.coarse_stride)).astype(np.int64)
        jy = np.rint((mkpts1fr[:, 1] - offy) / float(self.coarse_stride)).astype(np.int64)

        w0c = w0r // self.coarse_stride
        h0c = h0r // self.coarse_stride
        w1c = w1r // self.coarse_stride
        h1c = h1r // self.coarse_stride

        ix = np.clip(ix, 0, w0c - 1)
        iy = np.clip(iy, 0, h0c - 1)
        jx = np.clip(jx, 0, w1c - 1)
        jy = np.clip(jy, 0, h1c - 1)

        i_ids = iy * w0c + ix
        j_ids = jy * w1c + jx

        # patch center (coarse representative pixel) in resized image1
        j_rep_x = jx.astype(np.float32) * float(self.coarse_stride) + float(offx)
        j_rep_y = jy.astype(np.float32) * float(self.coarse_stride) + float(offy)
        j_rep = np.stack([j_rep_x, j_rep_y], axis=1)  # [M,2]

        # offset in pixels (resized domain)
        offset = mkpts1fr - j_rep  # [M,2]

        # normalized offset for LoFTR fine supervision
        # expec_f_gt = offset / fine_stride / radius
        if self.radius <= 0:
            raise RuntimeError("fine_window_size too small.")
        expec = offset / float(self.fine_stride) / float(self.radius)

        fine_ok = (np.max(np.abs(expec), axis=1) < 1.0)  # consistent with LoFTR fine_correct_thr=1.0

        if self.only_fine_ok:
            keep = fine_ok
            i_ids = i_ids[keep]
            j_ids = j_ids[keep]
            expec = expec[keep]
            offset = offset[keep]

        # ---- deduplicate pairs (downscale can create collisions) ----
        if i_ids.size > 0:
            pair_key = i_ids.astype(np.int64) * 1_000_000 + j_ids.astype(np.int64)  # safe
            err = np.linalg.norm(offset, axis=1).astype(np.float32)

            # (1) unique by (i,j) first (keep smallest reproj error)
            order = np.lexsort((err, pair_key))  # pair_key asc, err asc
            pair_key_s = pair_key[order]
            _, first = np.unique(pair_key_s, return_index=True)
            pick = order[first]
            i_ids = i_ids[pick]
            j_ids = j_ids[pick]
            expec = expec[pick]
            err = err[pick]

            # (2) enforce 1-to-1 after resize (unique i and unique j): greedy lowest-error
            HW0 = int(h0c * w0c)
            HW1 = int(h1c * w1c)
            used_i = np.zeros(HW0, dtype=np.bool_)
            used_j = np.zeros(HW1, dtype=np.bool_)
            pick2 = []
            for k in np.argsort(err):
                ii = int(i_ids[k]); jj = int(j_ids[k])
                if used_i[ii] or used_j[jj]:
                    continue
                used_i[ii] = True
                used_j[jj] = True
                pick2.append(k)
            if len(pick2) > 0:
                i_ids = i_ids[pick2]
                j_ids = j_ids[pick2]
                expec = expec[pick2]
            else:
                i_ids = i_ids[:0]
                j_ids = j_ids[:0]
                expec = expec[:0]
        # final tensors
        i_ids_t = torch.from_numpy(i_ids.astype(np.int64)).long()
        j_ids_t = torch.from_numpy(j_ids.astype(np.int64)).long()
        expec_t = torch.from_numpy(expec.astype(np.float32)).float()

        sample: Dict[str, Any] = {
            "image0": _to_tensor01(img0r_u8),
            "image1": _to_tensor01(img1r_u8),
            "i_ids_gt": i_ids_t,
            "j_ids_gt": j_ids_t,
            "expec_f_gt": expec_t,
            "stride": int(self.coarse_stride),
            "pair_name": row.get("pair_name", "") or f"{Path(anc_path).name}__{Path(cam_path).name}",
        }
        return sample