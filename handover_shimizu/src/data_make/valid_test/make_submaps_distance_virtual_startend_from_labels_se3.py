#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 使い方:
#   python make_submaps_distance_virtual_startend_from_labels_se3.py --help
#   必要な入力パスと出力先を引数で指定して実行する。

"""start/end ラベルから仮想アンカー submap を生成する。

start/end ラベルを使って、SE(3)（6DoF）の「仮想アンカー」submap を生成するスクリプト。

背景 / 目的
-----------
- DB 軌跡上の start から end までに、一定間隔（例: 1m）で *仮想* アンカーを置きたい。
- それらのアンカーには完全な 4x4 姿勢（world<-vehicle）を持たせ、実フレームの DB カメラ index がなくても、
  後段の RI レンダリングをその *仮想* カメラ姿勢（カメラ外部パラメータ経由）で行えるようにしたい。

このスクリプトは、元の `make_submaps_distance_txt.py` / 以前の仮想アンカー版と
出力レイアウトと主要な約束事をできるだけそろえている:
  out_root/<subset>/<segment>/
    - frames_poses.csv
    - anchors.csv
    - frames_submap_ids.csv
    - submaps.json

注意
----
このスクリプトが行うのは SE(3) アンカーの書き出しだけ。
実際にその仮想アンカー位置で RI を描画するには、`render_anchor_intensity_txt.py`
（またはそのラッパー側）が、アンカー姿勢として `frame.pose.transform` ではなく
`anchor_pose_T_wv` を使う必要がある。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 重い import（Waymo 関連）
import tensorflow as tf  # type: ignore
from waymo_open_dataset import dataset_pb2  # type: ignore


# -------------------------
# 補助関数: SE(3) / 回転処理
# -------------------------

def _normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    """四元数を正規化して単位四元数にそろえる。"""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n <= 0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """3x3 回転行列を四元数 (x,y,z,w) に変換する。"""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        # 最大の対角成分を持つ軸を探す
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S

    return _normalize_quat_xyzw(np.array([x, y, z, w], dtype=np.float64))


def quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    """四元数 (x,y,z,w) を 3x3 回転行列へ変換する。"""
    x, y, z, w = _normalize_quat_xyzw(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    return R


def slerp_quat_xyzw(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """四元数 (x,y,z,w) 同士の球面線形補間（SLERP）を行う。"""
    q0 = _normalize_quat_xyzw(q0)
    q1 = _normalize_quat_xyzw(q1)
    t = float(np.clip(t, 0.0, 1.0))

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    # 十分近い場合は lerp を使う
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return _normalize_quat_xyzw(q)

    theta0 = math.acos(dot)
    sin_theta0 = math.sin(theta0)
    theta = theta0 * t
    sin_theta = math.sin(theta)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta0
    s1 = sin_theta / sin_theta0
    q = (s0 * q0) + (s1 * q1)
    return _normalize_quat_xyzw(q)


def rotmat_to_rpy_zyx_deg(R: np.ndarray) -> Tuple[float, float, float]:
    """R = Rz(yaw)*Ry(pitch)*Rx(roll) に対して、(roll, pitch, yaw) を度単位で返す。

    R is world<-vehicle.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)

    # pitch を求める
    sp = -float(R[2, 0])
    sp = float(np.clip(sp, -1.0, 1.0))
    pitch = math.asin(sp)

    # ジンバルロックを処理する
    cp = math.cos(pitch)
    if abs(cp) < 1e-8:
        # pitch を求める ~ +-90deg
        roll = 0.0
        yaw = math.atan2(-float(R[0, 1]), float(R[1, 1]))
    else:
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))

    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def compose_T_wv(t_xyz: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    """平行移動と回転から world<-vehicle の姿勢行列を構成する。"""
    R = quat_xyzw_to_rotmat(q_xyzw)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t_xyz, dtype=np.float64).reshape(3)
    return T


# -------------------------
# セグメント一覧とラベルの解析
# -------------------------

SEG_RE = re.compile(r"^segment-[0-9]+_.*_with_camera_labels$")


def load_segments_txt(p: Path) -> List[str]:
    """セグメント一覧テキストを読み込む。"""
    lines = [ln.strip() for ln in p.read_text().splitlines()]
    segs: List[str] = []
    for ln in lines:
        if not ln or ln.startswith("#"):
            continue
        segs.append(ln)
    return segs


@dataclass
class LabelWindow:
    """開始・中間・終了ラベルから得たフレーム範囲を表す。"""
    db_segment: str
    q_segment: str
    db_i_start: int
    db_i_end: int


def _try_parse_label_json(obj: dict) -> Optional[LabelWindow]:
    """v7 ラベラの JSON を解析する（必要な項目だけ読む）。"""
    if not isinstance(obj, dict):
        return None
    db_seg = obj.get("db_segment")
    q_seg = obj.get("q_segment")
    if not isinstance(db_seg, str) or not isinstance(q_seg, str):
        return None

    start = obj.get("start") or {}
    end = obj.get("end") or {}

    try:
        db_i_start = int(start.get("db_i"))
        db_i_end = int(end.get("db_i"))
    except Exception:
        return None

    return LabelWindow(db_segment=db_seg, q_segment=q_seg, db_i_start=db_i_start, db_i_end=db_i_end)


def load_labels_db_index(labels_root: Path) -> Dict[str, LabelWindow]:
    """db_segment -> LabelWindow の対応を作る。

    It scans for files named like labels_*.json under labels_root.
    """
    idx: Dict[str, LabelWindow] = {}
    if not labels_root.exists():
        return idx

    for p in labels_root.rglob("labels_*.json"):
        try:
            obj = json.loads(p.read_text())
        except Exception:
            continue
        lw = _try_parse_label_json(obj)
        if lw is None:
            continue

        if lw.db_segment in idx:
            prev = idx[lw.db_segment]
            # 重複がある場合は (end-start) の幅が大きいものを残す。
            prev_len = prev.db_i_end - prev.db_i_start
            new_len = lw.db_i_end - lw.db_i_start
            if new_len > prev_len:
                idx[lw.db_segment] = lw
        else:
            idx[lw.db_segment] = lw

    return idx


# -------------------------
# TFRecord の探索と姿勢読み込み
# -------------------------


def find_tfrecord_auto(tfrecord_root: Path, segment: str) -> Tuple[str, Path]:
    """どの subset に segment.tfrecord があるかを探す。

    Returns (subset_name, tfrecord_path).
    """
    cand = [
        "training",
        "validation",
        "testing",
        "domain_adaptation",
        "testing_3d_camera_only_detection",
    ]
    for subset in cand:
        p = tfrecord_root / subset / f"{segment}.tfrecord"
        if p.exists():
            return subset, p

    # 見つからない場合は 1 階層深くまで探す
    for d in tfrecord_root.iterdir():
        if not d.is_dir():
            continue
        p = d / f"{segment}.tfrecord"
        if p.exists():
            return d.name, p

    raise FileNotFoundError(f"TFRecord not found: {segment}.tfrecord under {tfrecord_root}")


def read_all_frame_poses(tfrecord_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """全フレームの timestamp（int64）と pose 行列 (N,4,4, world<-vehicle) を読む。"""
    ds = tf.data.TFRecordDataset([str(tfrecord_path)], compression_type="")
    ts: List[int] = []
    Ts: List[np.ndarray] = []

    for raw in ds:
        fr = dataset_pb2.Frame.FromString(bytearray(raw.numpy()))
        ts.append(int(fr.timestamp_micros))
        T = np.array(fr.pose.transform, dtype=np.float64).reshape(4, 4)
        Ts.append(T)

    if not Ts:
        raise RuntimeError(f"No frames in tfrecord: {tfrecord_path}")

    return np.asarray(ts, dtype=np.int64), np.stack(Ts, axis=0)


# -------------------------
# 軌跡ウィンドウとアンカー生成
# -------------------------


def build_window_traj(
    ts_all: np.ndarray,
    T_all: np.ndarray,
    start_i: int,
    end_i: int,
    stride: int,
) -> Dict[str, np.ndarray]:
    """SE(3) 情報を保ったまま、必要なら stride を入れた windowed trajectory を作る。

    Returns dict of arrays aligned on traj points.
    """
    n = int(T_all.shape[0])
    start_i = int(max(0, min(start_i, n - 1)))
    end_i = int(max(0, min(end_i, n - 1)))
    if end_i < start_i:
        start_i, end_i = end_i, start_i

    stride = int(max(1, stride))

    traj_idx = list(range(start_i, end_i + 1, stride))
    if traj_idx[-1] != end_i:
        traj_idx.append(end_i)

    traj_idx_arr = np.asarray(traj_idx, dtype=np.int32)

    ts = ts_all[traj_idx_arr]
    T = T_all[traj_idx_arr]
    t_xyz = T[:, :3, 3]
    R = T[:, :3, :3]

    quat = np.stack([rotmat_to_quat_xyzw(R[i]) for i in range(R.shape[0])], axis=0)
    rpy_deg = np.stack([np.asarray(rotmat_to_rpy_zyx_deg(R[i]), dtype=np.float64) for i in range(R.shape[0])], axis=0)

    x = t_xyz[:, 0]
    y = t_xyz[:, 1]

    # XY 平面での累積距離を計算する（既存コードと同じ規約）
    if len(x) <= 1:
        cumdist = np.zeros((len(x),), dtype=np.float64)
    else:
        dx = np.diff(x)
        dy = np.diff(y)
        step = np.sqrt(dx * dx + dy * dy)
        cumdist = np.concatenate([np.array([0.0], dtype=np.float64), np.cumsum(step)])

    return {
        "traj_idx": traj_idx_arr,
        "ts": ts,
        "T_wv": T,
        "t_xyz": t_xyz,
        "quat_xyzw": quat,
        "rpy_deg": rpy_deg,
        "cumdist_m": cumdist,
        "start_i": np.array([start_i], dtype=np.int32),
        "end_i": np.array([end_i], dtype=np.int32),
    }


def interp_anchor_pose_by_cumdist(traj: Dict[str, np.ndarray], target_d: float) -> Dict[str, np.ndarray]:
    """累積距離 `target_d` 上にある SE(3) アンカー姿勢を補間する。

    Returns dict with T_wv, t_xyz, quat_xyzw, rpy_deg.
    """
    cum = traj["cumdist_m"].astype(np.float64)
    t_xyz = traj["t_xyz"].astype(np.float64)
    quat = traj["quat_xyzw"].astype(np.float64)

    if len(cum) == 0:
        raise RuntimeError("empty trajectory")

    d = float(target_d)
    if d <= cum[0] + 1e-9:
        q = quat[0]
        t = t_xyz[0]
        T = compose_T_wv(t, q)
        R = T[:3, :3]
        rpy = np.asarray(rotmat_to_rpy_zyx_deg(R), dtype=np.float64)
        return {"T_wv": T, "t_xyz": t, "quat_xyzw": q, "rpy_deg": rpy}

    if d >= cum[-1] - 1e-9:
        q = quat[-1]
        t = t_xyz[-1]
        T = compose_T_wv(t, q)
        R = T[:3, :3]
        rpy = np.asarray(rotmat_to_rpy_zyx_deg(R), dtype=np.float64)
        return {"T_wv": T, "t_xyz": t, "quat_xyzw": q, "rpy_deg": rpy}

    hi = int(np.searchsorted(cum, d, side="right"))
    lo = max(0, hi - 1)
    hi = min(hi, len(cum) - 1)

    d0 = float(cum[lo])
    d1 = float(cum[hi])
    if abs(d1 - d0) < 1e-9:
        a = 0.0
    else:
        a = (d - d0) / (d1 - d0)
        a = float(np.clip(a, 0.0, 1.0))

    t = (1.0 - a) * t_xyz[lo] + a * t_xyz[hi]
    q = slerp_quat_xyzw(quat[lo], quat[hi], a)

    T = compose_T_wv(t, q)
    R = T[:3, :3]
    rpy = np.asarray(rotmat_to_rpy_zyx_deg(R), dtype=np.float64)

    return {"T_wv": T, "t_xyz": t, "quat_xyzw": q, "rpy_deg": rpy}


def make_virtual_anchors_se3(traj: Dict[str, np.ndarray], spacing_m: float) -> List[dict]:
    """start（cumdist=0）から一定間隔で仮想アンカーを作る。

    Each anchor stores full SE(3) pose (T_wv).
    """
    spacing_m = float(spacing_m)
    if spacing_m <= 0:
        raise ValueError("spacing_m must be > 0")

    cum = traj["cumdist_m"].astype(np.float64)
    total_d = float(cum[-1]) if len(cum) else 0.0

    if total_d <= 1e-9:
        d_list = [0.0]
    else:
        # 以前のスクリプトと同じ規約で、0, spacing, 2*spacing, ... <= total にアンカーを置く
        d_list = list(np.arange(0.0, total_d + 1e-6, spacing_m).astype(np.float64))
        if not d_list:
            d_list = [0.0]

    anchors: List[dict] = []
    traj_idx = traj["traj_idx"].astype(np.int32)
    ts = traj["ts"].astype(np.int64)
    t_traj = traj["t_xyz"].astype(np.float64)

    for aid, d in enumerate(d_list):
        pose = interp_anchor_pose_by_cumdist(traj, float(d))
        T = pose["T_wv"]
        t = pose["t_xyz"]
        q = pose["quat_xyzw"]
        rpy = pose["rpy_deg"]

        # 参照用の実フレームを 1 つ選ぶ（軌跡サンプル中で XY が最も近いもの）
        # これは後で calibration を取るためだけに使い、姿勢そのものは仮想のまま。
        xy = t[:2]
        d2 = np.sum((t_traj[:, :2] - xy[None, :]) ** 2, axis=1)
        k = int(np.argmin(d2))
        ref_frame_index = int(traj_idx[k])
        ref_ts = int(ts[k])

        anchors.append(
            {
                "anchor_id": int(aid),
                "anchor_distance_m": float(d),
                "anchor_ref_frame_index": ref_frame_index,
                "anchor_ref_timestamp_micros": ref_ts,
                # 互換性用フィールド（下流スクリプトがこの名前を期待する場合がある）
                "anchor_frame_index": ref_frame_index,
                "anchor_timestamp_micros": ref_ts,
                # 姿勢表現
                "anchor_pose_T_wv": T.reshape(-1).tolist(),
                "anchor_pose_xyz": [float(t[0]), float(t[1]), float(t[2])],
                "anchor_pose_quat_xyzw": [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
                "anchor_pose_rpy_deg": [float(rpy[0]), float(rpy[1]), float(rpy[2])],
                "anchor_pose_xyyaw": [float(t[0]), float(t[1]), float(rpy[2])],
                "is_virtual": True,
            }
        )

    return anchors


def assign_frames_to_anchors_xy(traj: Dict[str, np.ndarray], anchors: List[dict]) -> np.ndarray:
    """各軌跡フレームを、XY 距離で最も近いアンカーへ割り当てる。"""
    frames_xy = traj["t_xyz"][:, :2].astype(np.float64)
    anchors_xy = np.array([a["anchor_pose_xyz"][:2] for a in anchors], dtype=np.float64)

    # 形状は (N_frames, N_anchors)
    diff = frames_xy[:, None, :] - anchors_xy[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    return np.argmin(d2, axis=1).astype(np.int32)


# -------------------------
# 出力書き込み
# -------------------------


def _ensure_dir(p: Path) -> None:
    """出力先ディレクトリを作成して存在を保証する。"""
    p.mkdir(parents=True, exist_ok=True)


def write_frames_poses_csv(out_path: Path, traj: Dict[str, np.ndarray]) -> None:
    """各フレームの姿勢一覧を CSV に書き出す。"""
    idx = traj["traj_idx"].tolist()
    ts = traj["ts"].tolist()
    t = traj["t_xyz"]
    quat = traj["quat_xyzw"]
    rpy = traj["rpy_deg"]
    cum = traj["cumdist_m"]

    header = [
        "frame_index",
        "timestamp_micros",
        "x",
        "y",
        "z",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "qx",
        "qy",
        "qz",
        "qw",
        "cumdist_m",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(idx)):
            w.writerow(
                [
                    int(idx[i]),
                    int(ts[i]),
                    float(t[i, 0]),
                    float(t[i, 1]),
                    float(t[i, 2]),
                    float(rpy[i, 0]),
                    float(rpy[i, 1]),
                    float(rpy[i, 2]),
                    float(quat[i, 0]),
                    float(quat[i, 1]),
                    float(quat[i, 2]),
                    float(quat[i, 3]),
                    float(cum[i]),
                ]
            )


def write_anchors_csv(out_path: Path, anchors: List[dict]) -> None:
    """仮想アンカーの一覧を CSV に書き出す。"""
    header = [
        "anchor_id",
        "anchor_distance_m",
        "anchor_ref_frame_index",
        "anchor_ref_timestamp_micros",
        "x",
        "y",
        "z",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "qx",
        "qy",
        "qz",
        "qw",
        "T_wv_rowmajor",
        "is_virtual",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for a in anchors:
            xyz = a["anchor_pose_xyz"]
            rpy = a["anchor_pose_rpy_deg"]
            q = a["anchor_pose_quat_xyzw"]
            T = a["anchor_pose_T_wv"]
            w.writerow(
                [
                    int(a["anchor_id"]),
                    float(a["anchor_distance_m"]),
                    int(a["anchor_ref_frame_index"]),
                    int(a["anchor_ref_timestamp_micros"]),
                    float(xyz[0]),
                    float(xyz[1]),
                    float(xyz[2]),
                    float(rpy[0]),
                    float(rpy[1]),
                    float(rpy[2]),
                    float(q[0]),
                    float(q[1]),
                    float(q[2]),
                    float(q[3]),
                    json.dumps([float(x) for x in T]),
                    bool(a.get("is_virtual", True)),
                ]
            )


def write_frames_submap_ids_csv(out_path: Path, traj: Dict[str, np.ndarray], assign_ids: np.ndarray) -> None:
    """各フレームと所属 submap の対応を CSV に保存する。"""
    idx = traj["traj_idx"].tolist()
    ts = traj["ts"].tolist()

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "timestamp_micros", "submap_id"])
        for i in range(len(idx)):
            w.writerow([int(idx[i]), int(ts[i]), int(assign_ids[i])])


def write_submaps_json(
    out_path: Path,
    segment: str,
    subset: str,
    traj: Dict[str, np.ndarray],
    anchors: List[dict],
    assign_ids: np.ndarray,
    spacing_m: float,
    stride: int,
    submap_radius_m: float,
    labels_window: Tuple[int, int],
    tfrecord_path: Path,
) -> None:
    # anchor ごとの member frame index を作る（軌跡フレームのみ。make_submaps_distance_txt.py と整合させる）
    """submap ごとの設定と姿勢情報を JSON に保存する。"""
    traj_idx = traj["traj_idx"].astype(np.int32)
    ts = traj["ts"].astype(np.int64)

    members_by_anchor: List[List[int]] = [[] for _ in anchors]
    ts_by_anchor: List[List[int]] = [[] for _ in anchors]
    for i in range(len(traj_idx)):
        aid = int(assign_ids[i])
        if 0 <= aid < len(anchors):
            members_by_anchor[aid].append(int(traj_idx[i]))
            ts_by_anchor[aid].append(int(ts[i]))

    submaps = []
    for a in anchors:
        aid = int(a["anchor_id"])
        entry = {
            "submap_id": aid,
            "anchor_distance_m": float(a["anchor_distance_m"]),
            # 既存 renderer コードとの後方互換のため
            "anchor_frame_index": int(a["anchor_frame_index"]),
            "anchor_timestamp_micros": int(a["anchor_timestamp_micros"]),
            "anchor_pose_xyyaw": [float(x) for x in a["anchor_pose_xyyaw"]],
            # 新しい SE(3) フィールド
            "anchor_pose_T_wv": [float(x) for x in a["anchor_pose_T_wv"]],
            "anchor_pose_xyz": [float(x) for x in a["anchor_pose_xyz"]],
            "anchor_pose_quat_xyzw": [float(x) for x in a["anchor_pose_quat_xyzw"]],
            "anchor_pose_rpy_deg": [float(x) for x in a["anchor_pose_rpy_deg"]],
            "anchor_ref_frame_index": int(a["anchor_ref_frame_index"]),
            "anchor_ref_timestamp_micros": int(a["anchor_ref_timestamp_micros"]),
            "is_virtual": True,
            # membership（軌跡フレーム。anchors > frames の場合は空になりうる）
            "frames": members_by_anchor[aid],
            "frame_timestamps_micros": ts_by_anchor[aid],
        }
        submaps.append(entry)

    meta = {
        "segment": segment,
        "subset": subset,
        "tfrecord": str(tfrecord_path),
        "spacing_m": float(spacing_m),
        "stride": int(stride),
        "submap_radius_m": float(submap_radius_m),
        "window_db_i_start": int(labels_window[0]),
        "window_db_i_end": int(labels_window[1]),
        "n_traj_frames": int(traj_idx.shape[0]),
        "n_anchors": int(len(anchors)),
        "anchor_pose_format": "T_wv_rowmajor_16",  # world<-vehicle
    }

    out = {"meta": meta, "submaps": submaps}
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")


# -------------------------
# メイン処理
# -------------------------


def main() -> None:
    """CLI 引数を解釈し、入力の読み込みから結果保存までの一連の処理を実行する。"""
    ap = argparse.ArgumentParser(description="Virtual SE(3) anchors from start-end labels")
    ap.add_argument("--segments_txt", type=str, required=True, help="one segment name per line (DB segments)")
    ap.add_argument("--tfrecord_root", type=str, required=True, help="Waymo individual_files root")
    ap.add_argument("--labels_root", type=str, required=True, help="dir containing labels_*.json per pair")
    ap.add_argument("--out_root", type=str, required=True, help="output root")
    ap.add_argument("--spacing_m", type=float, default=1.0, help="anchor spacing in meters")
    ap.add_argument("--stride", type=int, default=1, help="trajectory sampling stride inside [start,end]")
    ap.add_argument("--submap_radius_m", type=float, default=10.0, help="submap radius (metadata; same meaning as r-sub-m)")
    args = ap.parse_args()

    segments_txt = Path(args.segments_txt)
    tfrecord_root = Path(args.tfrecord_root)
    labels_root = Path(args.labels_root)
    out_root = Path(args.out_root)

    segs = load_segments_txt(segments_txt)
    labels_idx = load_labels_db_index(labels_root)

    # DB ラベルがあるものに絞る
    segs_ok = [s for s in segs if s in labels_idx]

    print(f"[INFO] segments={len(segs)} labels_db_found={len(segs_ok)}")
    print(f"[INFO] out_root={out_root}")

    ok = 0
    fail = 0

    for seg in segs:
        if seg not in labels_idx:
            print(f"[WARN] skip (no labels for db segment): {seg}")
            continue

        lw = labels_idx[seg]

        # TFRecord を見つけて姿勢を読む
        try:
            subset, tfrec_path = find_tfrecord_auto(tfrecord_root, seg)
        except Exception as e:
            print(f"[ERR] tfrecord not found for {seg}: {e}")
            fail += 1
            continue

        try:
            ts_all, T_all = read_all_frame_poses(tfrec_path)
        except Exception as e:
            print(f"[ERR] failed to read poses for {seg}: {e}")
            fail += 1
            continue

        n_frames = int(T_all.shape[0])
        start_i = int(lw.db_i_start)
        end_i = int(lw.db_i_end)
        start_i = max(0, min(start_i, n_frames - 1))
        end_i = max(0, min(end_i, n_frames - 1))
        if end_i < start_i:
            start_i, end_i = end_i, start_i

        # ウィンドウ軌跡を構築する（stride を適用しつつ end は必ず含める）
        traj = build_window_traj(ts_all, T_all, start_i=start_i, end_i=end_i, stride=args.stride)

        # SE(3) 仮想アンカーを生成する
        anchors = make_virtual_anchors_se3(traj, spacing_m=args.spacing_m)

        # stride 後の window フレームを最近傍アンカーへ割り当てる（XY）
        assign_ids = assign_frames_to_anchors_xy(traj, anchors)

        # 出力を書き出す
        out_dir = out_root / subset / seg
        _ensure_dir(out_dir)

        write_frames_poses_csv(out_dir / "frames_poses.csv", traj)
        write_anchors_csv(out_dir / "anchors.csv", anchors)
        write_frames_submap_ids_csv(out_dir / "frames_submap_ids.csv", traj, assign_ids)
        write_submaps_json(
            out_dir / "submaps.json",
            segment=seg,
            subset=subset,
            traj=traj,
            anchors=anchors,
            assign_ids=assign_ids,
            spacing_m=args.spacing_m,
            stride=args.stride,
            submap_radius_m=args.submap_radius_m,
            labels_window=(start_i, end_i),
            tfrecord_path=tfrec_path,
        )

        # 以前の実行と同じようにログ出力する
        window_len = int(end_i - start_i + 1)
        print(
            f"[OK] {seg} | subset={subset} | window=[{start_i}:{end_i}] | frames_window={window_len} | "
            f"anchors={len(anchors)} -> {out_dir}"
        )

        ok += 1

    print(f"[DONE] ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
