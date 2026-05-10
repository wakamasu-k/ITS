#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 使い方:
#   python make_loftr_matches_gt_cross_eval_v3.py --help
#   必要な入力パスと出力先を引数で指定して実行する。

# 目的: クロス評価用に RI ↔ Camera の GT 対応点を生成する。
"""
make_loftr_matches_gt_cross_eval_v2.py

Waymo LoFTR 向けに、cross-time 評価用の GT 対応点（query camera vs database anchor RI）を生成する。

このスクリプトは「堅牢寄り」の実装で、依存を標準ライブラリ + numpy + tqdm に絞り、
次の幾何ベース GT 生成スタイルを cross-segment / cross-time 設定へ拡張することを意図している:

  tools/make_loftr_matches_gt_step2a_paper_final_fixed_cam_v6.py

対象は以下の組み合わせ:

  - Query 側 (q): query の静的地図から描画した camera image + depth
      （render_query_cam_depth_txt.py が生成）
  - Database 側 (db): db の静的地図から描画した anchor reflectivity image + index/depth
      （render_anchor_intensity_txt_se3_fix.py / render_anchor_intensity_from_segments_auto_subset.py が生成）

対応付けと座標合わせ:
  - Pair CSV: q_to_anchor_softgt_max10m_val4test4.csv など
      (pair_id, db_seg, q_seg, *_subset, q_frame_index, anchor_id, dist_xy_m, flags...) を含む
  - ICP CSV: icp_startend_staticmaps.csv
      pair_id をキーにした db -> q 変換（Z 軸 yaw + 並進）を与える

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


# -----------------------------
# 小さな補助関数群
# -----------------------------

def str2bool(v: str) -> bool:
    """CLI などで受け取った真偽値表現を bool に正規化する。"""
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {v}")


def _norm_wsl_path(p: str) -> str:
    """
    WSL で使えるパスへ正規化する。優先する形は /mnt/<drive>/...。

    対応内容:
      - Windows ドライブパス: D:\waymo\... -> /mnt/d/waymo/...
      - バックスラッシュ -> スラッシュ
      - 先頭末尾の引用符を除去
    """
    if p is None:
        return ""
    s = str(p).strip().strip('"').strip("'")
    if s == "":
        return ""
    s = s.replace("\\", "/")
    # D:/... を /mnt/d/... へ直す
    if len(s) >= 3 and s[1] == ":" and s[2] == "/":
        drive = s[0].lower()
        s = f"/mnt/{drive}{s[2:]}"
    return s


def atomic_savez_compressed(path: Path, **kwargs) -> None:
    """NumPy 配列群を一時ファイル経由で安全に圧縮保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **kwargs)
    os.replace(tmp, path)


def atomic_write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    """CSV を一時ファイル経由で安全に保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)


def load_index_depth(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """index/depth NPZ を読み込む。"""
    with np.load(npz_path) as z:
        # 最も一般的: point_index/depth
        if "point_index" in z.files and "depth" in z.files:
            return z["point_index"], z["depth"]
        # フォールバック用キー（念のため）
        if "point_index" in z.files and "depth_x" in z.files:
            return z["point_index"], z["depth_x"]
        raise KeyError(f"expected keys (point_index, depth) in {npz_path}, got {z.files}")


def load_cam_meta_any(json_path: Path) -> Dict[str, Any]:
    """形式差を吸収しながらカメラメタ JSON を読み込む。"""
    m = json.loads(json_path.read_text())
    # よくある表記ゆれを正規化する
    if "intrinsic" not in m and "cam_intrinsic" in m:
        m["intrinsic"] = m["cam_intrinsic"]
    if "width" not in m or "height" not in m:
        # 例: camera_image_shape が [H, W, C] または [H, W] の場合
        shape = m.get("camera_image_shape", None) or m.get("image_shape", None) or m.get("img_shape", None)
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            m["height"] = int(shape[0])
            m["width"] = int(shape[1])
    # 一部の export では world_to_cam を別キーで保存していることがある（まれ）。
    if "world_to_cam" not in m:
        for k in ["T_world_cam", "T_world_to_cam", "world2cam"]:
            if k in m:
                m["world_to_cam"] = m[k]
                break
    return m



def _parse_intrinsic4(meta: Dict[str, Any], who: str = "cam") -> Tuple[float, float, float, float]:
    """カメラ内部パラメータを頑健に解釈する。

    対応形式:
      - [fx, fy, cx, cy]
      - 3x3 の K を一次元化したもの（len=9）: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
      - [fx, fy, cx, cy, ...]（歪み係数付き） -> 先頭 4 要素を使う
      - ネストした 3x3 list
      - fx/fy/cx/cy キーを持つ dict
    """
    if meta is None:
        raise ValueError(f"{who}: meta is None")

    intr = meta.get("intrinsic", None)
    if intr is None:
        raise KeyError(f"{who}: intrinsic not found in meta keys={list(meta.keys())[:40]}")

    # dict 形式
    if isinstance(intr, dict):
        for keys in [("fx", "fy", "cx", "cy"), ("f_x", "f_y", "c_x", "c_y")]:
            if all(k in intr for k in keys):
                fx, fy, cx, cy = (float(intr[keys[0]]), float(intr[keys[1]]), float(intr[keys[2]]), float(intr[keys[3]]))
                return fx, fy, cx, cy
        raise ValueError(f"{who}: intrinsic dict missing fx/fy/cx/cy. keys={list(intr.keys())}")

    # ネストした 3x3
    if isinstance(intr, (list, tuple)) and len(intr) == 3 and all(isinstance(r, (list, tuple)) and len(r) == 3 for r in intr):
        K = np.array(intr, dtype=np.float64)
        return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    # 平坦な list/tuple
    if isinstance(intr, (list, tuple)):
        vals = [float(x) for x in intr]
        if len(vals) == 4:
            return vals[0], vals[1], vals[2], vals[3]
        if len(vals) >= 9:
            # 経験則: 行優先で平坦化された K に見える場合
            if (abs(vals[8] - 1.0) < 1e-3 and abs(vals[1]) < 1e-3 and abs(vals[3]) < 1e-3
                and abs(vals[6]) < 1e-3 and abs(vals[7]) < 1e-3):
                fx = vals[0]
                fy = vals[4]
                cx = vals[2]
                cy = vals[5]
                return float(fx), float(fy), float(cx), float(cy)
            # それ以外は [fx, fy, cx, cy, k1, k2, ...] とみなす
            return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
        if len(vals) > 4:
            return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])

        raise ValueError(f"{who}: intrinsic list too short: len={len(vals)} vals={vals}")

    raise ValueError(f"{who}: unsupported intrinsic type={type(intr)}")


def _parse_world_to_cam(meta: Dict[str, Any], who: str = "cam") -> np.ndarray:
    """4x4 の world_to_cam を np.ndarray(float64) として返す。"""
    if meta is None:
        raise ValueError(f"{who}: meta is None")
    if "world_to_cam" in meta:
        Twc = meta["world_to_cam"]
        arr = np.array(Twc, dtype=np.float64)
        if arr.size == 16:
            return arr.reshape(4, 4)
        if arr.shape == (4, 4):
            return arr
        raise ValueError(f"{who}: world_to_cam shape unexpected: {arr.shape} (size={arr.size})")
    # フォールバック: cam_to_world を反転する
    if "cam_to_world" in meta:
        Tcw = np.array(meta["cam_to_world"], dtype=np.float64)
        if Tcw.size == 16:
            Tcw = Tcw.reshape(4, 4)
        if Tcw.shape != (4, 4):
            raise ValueError(f"{who}: cam_to_world shape unexpected: {Tcw.shape}")
        return np.linalg.inv(Tcw)

    raise KeyError(f"{who}: neither world_to_cam nor cam_to_world found. keys={list(meta.keys())[:50]}")


def se3_from_yaw_xyz(yaw_deg: float, tx: float, ty: float, tz: float) -> np.ndarray:
    """Rz(yaw) と並進から 4x4 変換を組み立てる。"""
    yaw = math.radians(float(yaw_deg))
    c = math.cos(yaw)
    s = math.sin(yaw)
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    T[0, 3] = float(tx)
    T[1, 3] = float(ty)
    T[2, 3] = float(tz)
    return T


# -----------------------------
# Coarse 代表グリッド
# -----------------------------

@dataclass
class RepGrid:
    """反射強度グリッド上のセル配置と幾何計算を扱う。"""
    H: int
    W: int
    stride: int
    cell_point: str
    Hc: int
    Wc: int
    rep_x: np.ndarray      # float32, (Hc*Wc,)
    rep_y: np.ndarray      # float32, (Hc*Wc,)
    rep_pid: np.ndarray    # int32,   (Hc*Wc,)
    rep_depth: np.ndarray  # float32, (Hc*Wc,)
    rep_valid: np.ndarray  # bool,    (Hc*Wc,)


def cell_offsets(stride: int, cell_point: str) -> Tuple[float, float]:
    """近傍セルの相対オフセット列を生成する。"""
    if cell_point == "topleft":
        return 0.0, 0.0
    if cell_point == "center":
        return float(stride) / 2.0, float(stride) / 2.0
    raise ValueError(f"unknown cell_point: {cell_point}")


def build_rep_grid(
    point_index: np.ndarray,
    depth: np.ndarray,
    stride: int,
    cell_point: str,
    margin: int,
    min_depth: float,
    max_depth: float,
) -> RepGrid:
    """反射強度マップ用の 2D グリッド構造を構築する。"""
    assert point_index.shape == depth.shape, (point_index.shape, depth.shape)
    H, W = depth.shape
    if (H % stride) != 0 or (W % stride) != 0:
        raise ValueError(f"depth map shape must be divisible by stride. got {(H, W)} stride={stride}")
    Hc, Wc = H // stride, W // stride

    offx, offy = cell_offsets(stride, cell_point)
    xs = (np.arange(Wc, dtype=np.int32) * stride + offx).astype(np.int32)
    ys = (np.arange(Hc, dtype=np.int32) * stride + offy).astype(np.int32)
    gx, gy = np.meshgrid(xs, ys)  # (Hc,Wc)

    rep_x = gx.reshape(-1).astype(np.float32)
    rep_y = gy.reshape(-1).astype(np.float32)

    pid = point_index[gy, gx].astype(np.int32).reshape(-1)
    rep_depth = depth[gy, gx].astype(np.float32).reshape(-1)

    finite = np.isfinite(rep_depth)
    in_range = (rep_depth >= float(min_depth)) & (rep_depth <= float(max_depth))
    in_margin = (
        (gx.reshape(-1) >= margin)
        & (gx.reshape(-1) < (W - margin))
        & (gy.reshape(-1) >= margin)
        & (gy.reshape(-1) < (H - margin))
    )
    # 重要: 空画素を埋まっていると誤認しないよう pid>=0 を要求する
    valid = (pid >= 0) & finite & in_range & in_margin

    return RepGrid(
        H=int(H), W=int(W),
        stride=int(stride), cell_point=str(cell_point),
        Hc=int(Hc), Wc=int(Wc),
        rep_x=rep_x, rep_y=rep_y,
        rep_pid=pid, rep_depth=rep_depth,
        rep_valid=valid,
    )


# -----------------------------
# 幾何ユーティリティ（x-forward pinhole。Step2A v6 スクリプトと同じ）
# -----------------------------

def backproject_xforward(u: np.ndarray, v: np.ndarray, depth_x: np.ndarray,
                         fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    x-forward pinhole:
      u = fx * (-Y/X) + cx
      v = fy * (-Z/X) + cy
    u, v と X（=depth_x）から (X, Y, Z) を復元する。
    """
    X = depth_x
    Y = -(u - cx) * X / fx
    Z = -(v - cy) * X / fy
    return np.stack([X, Y, Z], axis=1)


def project_xforward(xyz: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x-forward カメラモデルで 3D 点を画像平面へ投影する。"""
    X = xyz[:, 0]
    Y = xyz[:, 1]
    Z = xyz[:, 2]
    u = fx * (-Y / X) + cx
    v = fy * (-Z / X) + cy
    return u, v, X


def _depth_consistent(pred: np.ndarray, ref: np.ndarray, abs_th: float, rel_th: float) -> np.ndarray:
    """投影結果と深度値が整合しているかを判定する。"""
    ref = ref.astype(np.float64)
    pred = pred.astype(np.float64)
    th = np.maximum(float(abs_th), np.abs(ref) * float(rel_th))
    return np.abs(pred - ref) <= th


# -----------------------------
# クロスマッピング（src world -> dst world の変換を挿入する）
# -----------------------------

def compute_nn_map_to_grid_cross(
    rep_src: RepGrid,
    rep_dst: RepGrid,
    Twc_src: np.ndarray,
    Twc_dst: np.ndarray,
    T_dst_from_src_world: np.ndarray,
    fx0: float, fy0: float, cx0: float, cy0: float,
    fx1: float, fy1: float, cx1: float, cy1: float,
    point_index_dst: np.ndarray,
    depth_dst: np.ndarray,
    reproj_th: float,
    vis_mode: str,
    depth_abs_th: float,
    depth_rel_th: float,
) -> np.ndarray:
    """
    各 src coarse cell に対して、対応する dst coarse cell index を 1 つ返す（無ければ -1）。

    変換の流れ:
      src_cam -> src_world -> dst_world -> dst_cam
      X_dst_cam = Twc_dst @ T_dst_from_src_world @ inv(Twc_src) @ X_src_cam
    """
    Nc = rep_src.Hc * rep_src.Wc
    out = np.full((Nc,), -1, dtype=np.int32)

    idx = np.where(rep_src.rep_valid)[0]
    if idx.size == 0:
        return out

    u0 = rep_src.rep_x[idx].astype(np.float64)
    v0 = rep_src.rep_y[idx].astype(np.float64)
    d0 = rep_src.rep_depth[idx].astype(np.float64)

    xyz0 = backproject_xforward(u0, v0, d0, fx0, fy0, cx0, cy0)  # (N,3)
    xyz0_h = np.concatenate([xyz0, np.ones((xyz0.shape[0], 1), dtype=np.float64)], axis=1)  # (N,4)

    # src カメラ座標 -> src world 座標
    Tcw_src = np.linalg.inv(Twc_src)
    xyz_w_src = (Tcw_src @ xyz0_h.T).T  # (N,4)

    # src world 座標 -> dst world 座標
    xyz_w_dst = (T_dst_from_src_world @ xyz_w_src.T).T  # (N,4)

    # dst world 座標 -> dst カメラ座標
    xyz1_h = (Twc_dst @ xyz_w_dst.T).T
    xyz1 = xyz1_h[:, :3]  # (N,3)

    u1f, v1f, d1f = project_xforward(xyz1, fx1, fy1, cx1, cy1)

    good = d1f > 1e-6
    if not np.any(good):
        return out

    u1f = u1f[good]
    v1f = v1f[good]
    d1f = d1f[good]
    idx_good = idx[good]

    offx, offy = cell_offsets(rep_dst.stride, rep_dst.cell_point)
    cx1i = np.rint((u1f - offx) / float(rep_dst.stride)).astype(np.int32)
    cy1i = np.rint((v1f - offy) / float(rep_dst.stride)).astype(np.int32)

    inb = (cx1i >= 0) & (cx1i < rep_dst.Wc) & (cy1i >= 0) & (cy1i < rep_dst.Hc)
    if not np.any(inb):
        return out

    cx1i = cx1i[inb]
    cy1i = cy1i[inb]
    u1f = u1f[inb]
    v1f = v1f[inb]
    d1f = d1f[inb]
    idx_good = idx_good[inb]

    dst_idx = cy1i * rep_dst.Wc + cx1i
    u1r = rep_dst.rep_x[dst_idx].astype(np.float64)
    v1r = rep_dst.rep_y[dst_idx].astype(np.float64)

    # 再投影誤差
    if reproj_th > 0:
        err = np.sqrt((u1f - u1r) ** 2 + (v1f - v1r) ** 2)
        ok = err <= float(reproj_th)
        if not np.any(ok):
            return out
        dst_idx = dst_idx[ok]
        idx_good = idx_good[ok]
        d1f = d1f[ok]
        u1r = u1r[ok]
        v1r = v1r[ok]

    # 可視性判定
    if vis_mode != "none":
        if vis_mode != "depth":
            raise ValueError("cross-eval supports only vis_mode=none/depth (id is meaningless across maps)")
        ix1 = u1r.astype(np.int32)  # trunc
        iy1 = v1r.astype(np.int32)
        inb2 = (ix1 >= 0) & (ix1 < rep_dst.W) & (iy1 >= 0) & (iy1 < rep_dst.H)
        if not np.any(inb2):
            return out
        dst_idx = dst_idx[inb2]
        idx_good = idx_good[inb2]
        d1f = d1f[inb2]
        ix1 = ix1[inb2]
        iy1 = iy1[inb2]

        d1_ref = depth_dst[iy1, ix1].astype(np.float64)
        pid1_ref = point_index_dst[iy1, ix1].astype(np.int32)
        has_depth = (pid1_ref >= 0) & np.isfinite(d1_ref) & (d1_ref > 1e-6)

        ok = np.zeros_like(has_depth, dtype=bool)
        if np.any(has_depth):
            ok[has_depth] = _depth_consistent(d1f[has_depth], d1_ref[has_depth], depth_abs_th, depth_rel_th)

        if not np.any(ok):
            return out
        dst_idx = dst_idx[ok]
        idx_good = idx_good[ok]

    out[idx_good] = dst_idx.astype(np.int32)
    return out


def mutual_matches_from_maps(fwd: np.ndarray, bwd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """相互最近傍条件で対応点集合を生成する。"""
    src = np.where(fwd >= 0)[0].astype(np.int32)
    if src.size == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)
    dst = fwd[src].astype(np.int32)
    ok = (dst >= 0) & (dst < bwd.shape[0])
    src = src[ok]
    dst = dst[ok]
    back = bwd[dst].astype(np.int32)
    ok2 = back == src
    return src[ok2], dst[ok2]


def compute_mkpts1_f_cross(
    rep_src: RepGrid,
    src_idx: np.ndarray,
    Twc_src: np.ndarray,
    Twc_dst: np.ndarray,
    T_dst_from_src_world: np.ndarray,
    fx0: float, fy0: float, cx0: float, cy0: float,
    fx1: float, fy1: float, cx1: float, cy1: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """cross 評価向けに query 側の対応点座標を計算する。"""
    u0 = rep_src.rep_x[src_idx].astype(np.float64)
    v0 = rep_src.rep_y[src_idx].astype(np.float64)
    d0 = rep_src.rep_depth[src_idx].astype(np.float64)
    xyz0 = backproject_xforward(u0, v0, d0, fx0, fy0, cx0, cy0)
    xyz0_h = np.concatenate([xyz0, np.ones((xyz0.shape[0], 1), dtype=np.float64)], axis=1)

    Tcw_src = np.linalg.inv(Twc_src)
    xyz_w_src = (Tcw_src @ xyz0_h.T).T
    xyz_w_dst = (T_dst_from_src_world @ xyz_w_src.T).T
    xyz1_h = (Twc_dst @ xyz_w_dst.T).T
    xyz1 = xyz1_h[:, :3]

    u1f, v1f, d1f = project_xforward(xyz1, fx1, fy1, cx1, cy1)
    ok = d1f > 1e-6
    u1f = np.where(ok, u1f, np.nan).astype(np.float32)
    v1f = np.where(ok, v1f, np.nan).astype(np.float32)
    d1f = np.where(ok, d1f, np.nan).astype(np.float32)
    return u1f, v1f, d1f


# -----------------------------
# パス構築ヘルパ（現在のディレクトリ構成に基づく）
# -----------------------------

def build_query_cam_paths(q_cam_root: Path, subset: str, seg: str, cam: str, frame_idx: int) -> Tuple[Path, Path]:
    """query カメラ画像の関連パスを組み立てる。"""
    fr = f"{int(frame_idx):05d}"
    d = q_cam_root / subset / seg / cam
    return d / f"{fr}.png", d / f"{fr}.json"


def build_query_depth_path(q_submaps_root: Path, subset: str, seg: str, cam: str, frame_idx: int) -> Path:
    """query 深度画像のパスを組み立てる。"""
    fr = f"{int(frame_idx):05d}"
    return q_submaps_root / subset / seg / cam / fr / "ri_000_index_depth.npz"


def build_db_anchor_paths(db_submaps_root: Path, subset: str, seg: str, anchor_id: int) -> Tuple[Path, Path, Path]:
    """DB アンカー側の関連パス群を組み立てる。"""
    d = db_submaps_root / subset / seg / str(int(anchor_id))
    return d / "anchor.png", d / "anchor.json", d / "anchor_index.npz"


def stable_npz_path(out_root: Path, pair_id: str, key: str) -> Path:
    """
    /mnt/d で極端に長いファイル名になるのを避ける。
    ファイルは out_root/<pair_id>/<h0>/<h1>/<hash>.npz 配下へ置く。
    """
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    sub0 = h[:2]
    sub1 = h[2:4]
    return out_root / pair_id / sub0 / sub1 / f"{h}.npz"


# -----------------------------
# メイン処理
# -----------------------------

def parse_args() -> argparse.Namespace:
    """CLI 引数を定義して解析結果を返す。"""
    ap = argparse.ArgumentParser(description="Generate evaluation GT (cross-time) in Step2A style (v2).")

    ap.add_argument("--pairs-csv", type=str, required=True,
                    help="Pair list CSV (q_to_anchor_softgt_max10m_*.csv). Windows paths OK; will be normalized to /mnt/..")
    ap.add_argument("--icp-csv", type=str, required=True,
                    help="ICP CSV (db->q), icp_startend_staticmaps.csv. Windows paths OK; will be normalized to /mnt/..")

    ap.add_argument("--q-cam-root", type=str, required=True,
                    help="Query cam_gray root (the '.../cam_gray/.../q' directory)")
    ap.add_argument("--q-submaps-root", type=str, required=True,
                    help="Query submaps root (the '.../submaps/.../q' directory)")
    ap.add_argument("--db-submaps-root", type=str, required=True,
                    help="DB submaps root (the '.../submaps/.../db' directory)")

    ap.add_argument("--out-root", type=str, required=True, help="Output root directory for GT npz files")
    ap.add_argument("--manifest-out", type=str, required=True, help="Output manifest CSV path")

    ap.add_argument("--cam", type=str, default="FRONT", help="Camera name directory (default: FRONT)")

    ap.add_argument("--filter", type=str, default="all",
                    choices=["all", "strict", "t1m", "t2m", "t5m", "t10m"],
                    help="Which rows to process (based on is_* columns in pair CSV).")

    ap.add_argument("--write-filtered-manifests", type=str2bool, default=True,
                    help="Also write filtered manifests next to manifest-out (strict/t1m/t2m/t5m/t10m).")

    ap.add_argument("--skip-existing", type=str2bool, default=True)
    ap.add_argument("--max-rows", type=int, default=-1, help="Debug: process only first N rows (-1=all)")
    ap.add_argument("--sort-by-query", type=str2bool, default=True,
                    help="Sort rows by (q_seg, q_subset, q_frame_index) to reuse IO better.")
    ap.add_argument("--no-verify-files", type=str2bool, default=False,
                    help="Skip existence checks (faster; assumes your file structure is correct).")

    # ICP 列名の対応付け
    ap.add_argument("--icp-yaw-col", type=str, default="final_yaw_deg")
    ap.add_argument("--icp-tx-col", type=str, default="final_tx")
    ap.add_argument("--icp-ty-col", type=str, default="final_ty")
    ap.add_argument("--icp-tz-col", type=str, default="final_tz")

    # GT パラメータ（Step2A の既定値に合わせる）
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--cell-point", type=str, default="topleft", choices=["topleft", "center"])
    ap.add_argument("--reproj-th", type=float, default=4.0)

    ap.add_argument("--min-depth", type=float, default=0.5)
    ap.add_argument("--max-depth", type=float, default=200.0)
    ap.add_argument("--margin", type=int, default=8)

    ap.add_argument("--vis-mode", type=str, default="depth", choices=["none", "depth"])
    ap.add_argument("--depth-abs-th", type=float, default=1.0)
    ap.add_argument("--depth-rel-th", type=float, default=0.01)

    ap.add_argument("--fine-window-size", type=int, default=5)
    ap.add_argument("--fine-stride", type=int, default=2)

    return ap.parse_args()


def main() -> None:
    """CLI 引数を解釈し、入力の読み込みから結果保存までの一連の処理を実行する。"""
    args = parse_args()

    pairs_csv = Path(_norm_wsl_path(args.pairs_csv))
    icp_csv = Path(_norm_wsl_path(args.icp_csv))

    q_cam_root = Path(_norm_wsl_path(args.q_cam_root))
    q_submaps_root = Path(_norm_wsl_path(args.q_submaps_root))
    db_submaps_root = Path(_norm_wsl_path(args.db_submaps_root))

    out_root = Path(_norm_wsl_path(args.out_root))
    manifest_out = Path(_norm_wsl_path(args.manifest_out))

    cam = str(args.cam)

    if not pairs_csv.exists():
        raise FileNotFoundError(f"pairs_csv not found: {pairs_csv}")
    if not icp_csv.exists():
        raise FileNotFoundError(f"icp_csv not found: {icp_csv}")

    # ICP 変換を読む（pair_id -> T_q_db）
    icp_rows: List[Dict[str, Any]] = []
    with open(icp_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            icp_rows.append(row)

    T_by_pair: Dict[str, np.ndarray] = {}
    for row in icp_rows:
        pid = str(row.get("pair_id", "")).strip()
        if pid == "":
            continue
        yaw = float(row[args.icp_yaw_col])
        tx = float(row[args.icp_tx_col])
        ty = float(row[args.icp_ty_col])
        tz = float(row[args.icp_tz_col])
        T_by_pair[pid] = se3_from_yaw_xyz(yaw, tx, ty, tz)

    if len(T_by_pair) == 0:
        raise RuntimeError("No transforms loaded from icp_csv. Check column names.")

    # pair CSV を読む
    rows_in: List[Dict[str, Any]] = []
    with open(pairs_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows_in.append(row)

    # フィルタ条件
    def row_selected(row: Dict[str, Any]) -> bool:
        """現在の行が処理対象条件を満たすか判定する。"""
        if args.filter == "all":
            return True
        if args.filter == "strict":
            return str2bool(row.get("is_strict_gt", "0"))
        if args.filter == "t1m":
            return str2bool(row.get("is_gt_t1m", "0"))
        if args.filter == "t2m":
            return str2bool(row.get("is_gt_t2m", "0"))
        if args.filter == "t5m":
            return str2bool(row.get("is_gt_t5m", "0"))
        if args.filter == "t10m":
            return str2bool(row.get("is_gt_t10m", "0"))
        return True

    rows_sel = [r for r in rows_in if row_selected(r)]
    if args.max_rows > 0:
        rows_sel = rows_sel[: int(args.max_rows)]

    if args.sort_by_query:
        def _k(r: Dict[str, Any]) -> Tuple:
            """ソートやグルーピングに使う複合キーを作る。"""
            return (r.get("q_subset", ""), r.get("q_seg", ""), int(float(r.get("q_frame_index", 0))), r.get("pair_id",""), int(float(r.get("anchor_id", 0))))
        rows_sel = sorted(rows_sel, key=_k)

    # Manifest のスキーマ
    fieldnames = [
        # 識別子
        "pair_id", "lane_tag",
        "db_seg", "q_seg", "db_subset", "q_subset",
        "q_frame_index", "anchor_id",
        # パス
        "q_image_png", "q_meta_json", "q_depth_npz",
        "anchor_png", "anchor_json", "anchor_index_npz",
        "gt_npz",
        # フラグ / 距離
        "dist_xy_m", "is_strict_gt", "is_gt_t1m", "is_gt_t2m", "is_gt_t5m", "is_gt_t10m",
        # 統計値
        "n_matches", "n_fine",
        "w0", "h0", "w1", "h1",
        # エラー情報
        "error",
    ]

    rows_out: List[Dict[str, Any]] = []

    # 直前に読み込んだ query depth と cam meta をキャッシュする（ソート時の IO 削減用）
    cache_q_key = None
    cache_q_cam_meta = None
    cache_q_pi = None
    cache_q_dep = None
    cache_q_rep = None

    cache_db_key = None
    cache_db_cam_meta = None
    cache_db_pi = None
    cache_db_dep = None
    cache_db_rep = None

    # GT パラメータ
    stride = int(args.stride)
    cell_point = str(args.cell_point)
    reproj_th = float(args.reproj_th)
    vis_mode = str(args.vis_mode)
    margin = int(args.margin)
    min_depth = float(args.min_depth)
    max_depth = float(args.max_depth)
    depth_abs_th = float(args.depth_abs_th)
    depth_rel_th = float(args.depth_rel_th)
    fine_wsize = int(args.fine_window_size)
    if fine_wsize < 1:
        fine_wsize = 1
    if (fine_wsize % 2) == 0:
        fine_wsize += 1
    fine_stride = int(args.fine_stride)
    fine_half_px = (fine_wsize // 2) * fine_stride

    for row in tqdm(rows_sel, desc="pairs", ncols=120):
        # 基本 ID
        pair_id = str(row.get("pair_id", "")).strip()
        lane_tag = row.get("lane_tag", "")
        db_seg = str(row.get("db_seg", "")).strip()
        q_seg = str(row.get("q_seg", "")).strip()
        db_subset = str(row.get("db_subset", "")).strip()
        q_subset = str(row.get("q_subset", "")).strip()
        q_frame_index = int(float(row.get("q_frame_index", 0)))
        anchor_id = int(float(row.get("anchor_id", 0)))
        dist_xy_m = row.get("dist_xy_m", "")

        out_row = {k: "" for k in fieldnames}
        out_row.update({
            "pair_id": pair_id,
            "lane_tag": lane_tag,
            "db_seg": db_seg,
            "q_seg": q_seg,
            "db_subset": db_subset,
            "q_subset": q_subset,
            "q_frame_index": q_frame_index,
            "anchor_id": anchor_id,
            "dist_xy_m": dist_xy_m,
            "is_strict_gt": row.get("is_strict_gt", ""),
            "is_gt_t1m": row.get("is_gt_t1m", ""),
            "is_gt_t2m": row.get("is_gt_t2m", ""),
            "is_gt_t5m": row.get("is_gt_t5m", ""),
            "is_gt_t10m": row.get("is_gt_t10m", ""),
            "n_matches": 0,
            "n_fine": 0,
            "w0": 0, "h0": 0, "w1": 0, "h1": 0,
            "error": "",
        })

        try:
            if pair_id not in T_by_pair:
                raise KeyError(f"pair_id not in icp_csv: {pair_id}")
            T_q_db = T_by_pair[pair_id]
            T_db_q = np.linalg.inv(T_q_db)

            # パスを組み立てる（root から再構成し、CSV 内の省略パスには依存しない）
            q_png, q_json = build_query_cam_paths(q_cam_root, q_subset, q_seg, cam, q_frame_index)
            q_depth = build_query_depth_path(q_submaps_root, q_subset, q_seg, cam, q_frame_index)
            a_png, a_json, a_idx = build_db_anchor_paths(db_submaps_root, db_subset, db_seg, anchor_id)

            out_row["q_image_png"] = str(q_png)
            out_row["q_meta_json"] = str(q_json)
            out_row["q_depth_npz"] = str(q_depth)
            out_row["anchor_png"] = str(a_png)
            out_row["anchor_json"] = str(a_json)
            out_row["anchor_index_npz"] = str(a_idx)

            # 存在確認（任意）
            if not args.no_verify_files:
                for p in [q_png, q_json, q_depth, a_png, a_json, a_idx]:
                    if not p.exists():
                        raise FileNotFoundError(str(p))

            # 出力パス
            key = f"{pair_id}|{q_subset}|{q_seg}|{q_frame_index}|{db_subset}|{db_seg}|{anchor_id}"
            gt_npz = stable_npz_path(out_root, pair_id, key)
            out_row["gt_npz"] = str(gt_npz)

            if args.skip_existing and gt_npz.exists() and gt_npz.stat().st_size > 200:
                try:
                    with np.load(gt_npz, allow_pickle=False) as z:
                        out_row["n_matches"] = int(z["mkpts0"].shape[0]) if "mkpts0" in z.files else 0
                        out_row["n_fine"] = int(z["fine_ok"].sum()) if "fine_ok" in z.files else 0
                        out_row["h0"] = int(z["hw0"][0]) if "hw0" in z.files else 0
                        out_row["w0"] = int(z["hw0"][1]) if "hw0" in z.files else 0
                        out_row["h1"] = int(z["hw1"][0]) if "hw1" in z.files else 0
                        out_row["w1"] = int(z["hw1"][1]) if "hw1" in z.files else 0
                    rows_out.append(out_row)
                    continue
                except Exception:
                    # 無効なら再生成する
                    pass

            # -------- query を読み込む（キャッシュあり） --------
            q_key = (q_subset, q_seg, q_frame_index)
            need_reload_q = (q_key != cache_q_key) or (cache_q_cam_meta is None) or ("_parsed" not in cache_q_cam_meta)
            if need_reload_q:
                # 注意: パース成功後にだけキャッシュへ反映する（poison-cache による KeyError('_parsed') を避けるため）。
                q_cam_meta = load_cam_meta_any(q_json)

                w1 = int(q_cam_meta["width"])
                h1 = int(q_cam_meta["height"])
                fx1, fy1, cx1, cy1 = _parse_intrinsic4(q_cam_meta, who="q_cam")
                Twc_q = _parse_world_to_cam(q_cam_meta, who="q_cam")

                q_cam_meta["_parsed"] = {
                    "w": w1, "h": h1,
                    "fx": fx1, "fy": fy1, "cx": cx1, "cy": cy1,
                    "Twc": Twc_q,
                }

                cache_q_pi, cache_q_dep = load_index_depth(q_depth)
                cache_q_rep = build_rep_grid(
                    point_index=cache_q_pi,
                    depth=cache_q_dep,
                    stride=stride,
                    cell_point=cell_point,
                    margin=margin,
                    min_depth=min_depth,
                    max_depth=max_depth,
                )

                cache_q_key = q_key
                cache_q_cam_meta = q_cam_meta

            w1 = int(cache_q_cam_meta["_parsed"]["w"])
            h1 = int(cache_q_cam_meta["_parsed"]["h"])
            fx1 = float(cache_q_cam_meta["_parsed"]["fx"])
            fy1 = float(cache_q_cam_meta["_parsed"]["fy"])
            cx1 = float(cache_q_cam_meta["_parsed"]["cx"])
            cy1 = float(cache_q_cam_meta["_parsed"]["cy"])
            Twc_q = cache_q_cam_meta["_parsed"]["Twc"]

            rep_dst = cache_q_rep
# -------- db anchor を読み込む（キャッシュあり） --------
            db_key = (db_subset, db_seg, anchor_id)
            need_reload_db = (db_key != cache_db_key) or (cache_db_cam_meta is None) or ("_parsed" not in cache_db_cam_meta)
            if need_reload_db:
                # 注意: パース成功後にだけキャッシュへ反映する（poison-cache による KeyError('_parsed') を避けるため）。
                db_cam_meta = load_cam_meta_any(a_json)

                w0 = int(db_cam_meta["width"])
                h0 = int(db_cam_meta["height"])
                fx0, fy0, cx0, cy0 = _parse_intrinsic4(db_cam_meta, who="db_anchor")
                Twc_a = _parse_world_to_cam(db_cam_meta, who="db_anchor")

                db_cam_meta["_parsed"] = {
                    "w": w0, "h": h0,
                    "fx": fx0, "fy": fy0, "cx": cx0, "cy": cy0,
                    "Twc": Twc_a,
                }

                cache_db_pi, cache_db_dep = load_index_depth(a_idx)
                cache_db_rep = build_rep_grid(
                    point_index=cache_db_pi,
                    depth=cache_db_dep,
                    stride=stride,
                    cell_point=cell_point,
                    margin=margin,
                    min_depth=min_depth,
                    max_depth=max_depth,
                )

                cache_db_key = db_key
                cache_db_cam_meta = db_cam_meta

            w0 = int(cache_db_cam_meta["_parsed"]["w"])
            h0 = int(cache_db_cam_meta["_parsed"]["h"])
            fx0 = float(cache_db_cam_meta["_parsed"]["fx"])
            fy0 = float(cache_db_cam_meta["_parsed"]["fy"])
            cx0 = float(cache_db_cam_meta["_parsed"]["cx"])
            cy0 = float(cache_db_cam_meta["_parsed"]["cy"])
            Twc_a = cache_db_cam_meta["_parsed"]["Twc"]

            rep_src = cache_db_rep
            rep_dst = cache_q_rep

            # -------- 相互 coarse match を計算する --------
            fwd = compute_nn_map_to_grid_cross(
                rep_src=rep_src, rep_dst=rep_dst,
                Twc_src=Twc_a, Twc_dst=Twc_q,
                T_dst_from_src_world=T_q_db,
                fx0=fx0, fy0=fy0, cx0=cx0, cy0=cy0,
                fx1=fx1, fy1=fy1, cx1=cx1, cy1=cy1,
                point_index_dst=cache_q_pi,
                depth_dst=cache_q_dep,
                reproj_th=reproj_th,
                vis_mode=vis_mode,
                depth_abs_th=depth_abs_th,
                depth_rel_th=depth_rel_th,
            )
            bwd = compute_nn_map_to_grid_cross(
                rep_src=rep_dst, rep_dst=rep_src,
                Twc_src=Twc_q, Twc_dst=Twc_a,
                T_dst_from_src_world=T_db_q,
                fx0=fx1, fy0=fy1, cx0=cx1, cy0=cy1,
                fx1=fx0, fy1=fy0, cx1=cx0, cy1=cy0,
                point_index_dst=cache_db_pi,
                depth_dst=cache_db_dep,
                reproj_th=reproj_th,
                vis_mode=vis_mode,
                depth_abs_th=depth_abs_th,
                depth_rel_th=depth_rel_th,
            )

            src_idx, dst_idx = mutual_matches_from_maps(fwd, bwd)

            mkpts0 = np.stack([rep_src.rep_x[src_idx], rep_src.rep_y[src_idx]], axis=1).astype(np.float32)
            mkpts1 = np.stack([rep_dst.rep_x[dst_idx], rep_dst.rep_y[dst_idx]], axis=1).astype(np.float32)

            # fine GT（連続再投影）
            u1f, v1f, d1f = compute_mkpts1_f_cross(
                rep_src=rep_src, src_idx=src_idx,
                Twc_src=Twc_a, Twc_dst=Twc_q,
                T_dst_from_src_world=T_q_db,
                fx0=fx0, fy0=fy0, cx0=cx0, cy0=cy0,
                fx1=fx1, fy1=fy1, cx1=cx1, cy1=cy1,
            )
            mkpts1_f = np.stack([u1f, v1f], axis=1).astype(np.float32)
            offset1 = (mkpts1_f - mkpts1).astype(np.float32)

            fine_ok = ((np.abs(offset1[:, 0]) <= float(fine_half_px)) &
                       (np.abs(offset1[:, 1]) <= float(fine_half_px))).astype(np.uint8)

            valid0_coarse = rep_src.rep_valid.reshape(rep_src.Hc, rep_src.Wc).astype(np.uint8)
            valid1_coarse = rep_dst.rep_valid.reshape(rep_dst.Hc, rep_dst.Wc).astype(np.uint8)

            meta = {
                "pair_id": pair_id,
                "db_seg": db_seg,
                "q_seg": q_seg,
                "db_subset": db_subset,
                "q_subset": q_subset,
                "q_frame_index": int(q_frame_index),
                "anchor_id": int(anchor_id),
                "stride": int(stride),
                "cell_point": str(cell_point),
                "pix_rounding": "trunc",
                "coarse_assign": "round",
                "reproj_th": float(reproj_th),
                "vis_mode": str(vis_mode),
                "depth_abs_th": float(depth_abs_th),
                "depth_rel_th": float(depth_rel_th),
                "fine_window_size": int(fine_wsize),
                "fine_stride": int(fine_stride),
                "T_q_db": T_q_db.tolist(),
            }

            atomic_savez_compressed(
                gt_npz,
                mkpts0=mkpts0,
                mkpts1=mkpts1,
                mkpts1_f=mkpts1_f,
                offset1=offset1,
                fine_ok=fine_ok,
                valid0_coarse=valid0_coarse,
                valid1_coarse=valid1_coarse,
                i_ids=src_idx.astype(np.int32),
                j_ids=dst_idx.astype(np.int32),
                hw0=np.array([h0, w0], dtype=np.int32),
                hw1=np.array([h1, w1], dtype=np.int32),
                stride=np.int32(stride),
                meta_json=json.dumps(meta, ensure_ascii=False),
            )

            out_row["n_matches"] = int(mkpts0.shape[0])
            out_row["n_fine"] = int(fine_ok.sum())
            out_row["w0"] = int(w0)
            out_row["h0"] = int(h0)
            out_row["w1"] = int(w1)
            out_row["h1"] = int(h1)

        except Exception as e:
            out_row["error"] = repr(e)

        rows_out.append(out_row)

    
    # -------- サマリ（path / meta の問題を素早く見つけるため） --------
    n_total = len(rows_out)
    n_err = sum(1 for r in rows_out if str(r.get("error", "")).strip() != "")
    n_ok = n_total - n_err
    if n_err == 0:
        print(f"[SUMMARY] total={n_total} ok={n_ok} err=0")
    else:
        from collections import Counter
        c = Counter(str(r.get("error", "")).strip() for r in rows_out if str(r.get("error", "")).strip() != "")
        print(f"[SUMMARY] total={n_total} ok={n_ok} err={n_err}")
        for e, cnt in c.most_common(10):
            print(f"  - {cnt:6d} : {e}")

# メイン manifest を書き出す
    atomic_write_csv(manifest_out, fieldnames=fieldnames, rows=rows_out)
    print("[OK] wrote manifest:", str(manifest_out))

    # 便宜上、必要ならフィルタ済み manifest も併せて書き出す
    if args.write_filtered_manifests:
        base = manifest_out
        stem = base.stem
        parent = base.parent
        suffix = base.suffix

        def _write(name: str, keep: List[Dict[str, Any]]) -> None:
            """ローカル補助関数として 1 件分の出力を書き込む。"""
            p = parent / f"{stem}_{name}{suffix}"
            atomic_write_csv(p, fieldnames=fieldnames, rows=keep)
            print("[OK] wrote:", str(p))

        def _b(v: Any) -> bool:
            """任意の値を bool として解釈する。"""
            try:
                return str2bool(v)
            except Exception:
                return False

        _write("strict", [r for r in rows_out if _b(r.get("is_strict_gt","0"))])
        _write("t1m",    [r for r in rows_out if _b(r.get("is_gt_t1m","0"))])
        _write("t2m",    [r for r in rows_out if _b(r.get("is_gt_t2m","0"))])
        _write("t5m",    [r for r in rows_out if _b(r.get("is_gt_t5m","0"))])
        _write("t10m",   [r for r in rows_out if _b(r.get("is_gt_t10m","0"))])


if __name__ == "__main__":
    main()
