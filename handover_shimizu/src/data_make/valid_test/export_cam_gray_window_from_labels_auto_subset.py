#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 使い方:
#   python export_cam_gray_window_from_labels_auto_subset.py --help
#   必要な入力パスと出力先を引数で指定して実行する。

"""ラベル JSON に基づいて Q セグメントの時間窓画像を書き出す。

Qセグメントのカメラ画像（グレースケール）を書き出す。
ただし「labels_segpair_*.json（v7ラベラの start/mid/end）」にある
q_i_start〜q_i_end の範囲だけを書き出す。

目的 / 方針
-----------
- 元コード `export_cam_gray_txt.py` の処理（JPEG decode → gray化 → (任意)CLAHE → PNG保存、
  4x4 pose / intrinsics / extrinsics のメタJSON保存）に準拠。
- subset は tfrecord_root 配下を探索して自動判定（training/validation/testing...）。
- 「仮想アンカー」方針でも、Q側は実在フレームのカメラ画像/姿勢が必要なので、
  start-end の実フレーム範囲だけをまず吐き出す。



"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 重い import（tensorflow / waymo / cv2）は必要になるまで遅延させる


# ------------------------
# ユーティリティ
# ------------------------

def parse_bool(x: str | bool) -> bool:
    """CLI などで受け取った真偽値表現を bool に正規化する。"""
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ["1", "true", "t", "yes", "y", "on"]:
        return True
    if s in ["0", "false", "f", "no", "n", "off"]:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {x}")


def read_segments_txt(path: Path) -> List[str]:
    """テキストファイルからセグメント名一覧を読み込む。

    各行には次のいずれかを指定できる:
      - segment stem: 'segment-..._with_camera_labels'
      - tfrecord path: '/.../segment-..._with_camera_labels.tfrecord'
    """
    segs: List[str] = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if ln.endswith(".tfrecord"):
            segs.append(Path(ln).stem)
        else:
            segs.append(ln)
    # 重複を除去しつつ順序は維持する
    out: List[str] = []
    seen = set()
    for s in segs:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def find_tfrecord_for_segment(tfrecord_root: Path, segment: str) -> Tuple[str, Path]:
    """(subset, tfrecord_path) を返す。"""
    # まず代表的な subset を優先して調べる
    cand_subsets = [
        "training",
        "validation",
        "testing",
        "domain_adaptation",
        "testing_3d_camera_only_detection",
    ]
    # そのほかの直下ディレクトリも候補に加える
    try:
        for p in tfrecord_root.iterdir():
            if p.is_dir() and p.name not in cand_subsets:
                cand_subsets.append(p.name)
    except Exception:
        pass

    hits: List[Tuple[str, Path]] = []
    for subset in cand_subsets:
        p = tfrecord_root / subset / f"{segment}.tfrecord"
        if p.exists():
            hits.append((subset, p))
    if not hits:
        raise FileNotFoundError(f"TFRecord not found for segment={segment} under {tfrecord_root}")
    # 複数見つかった場合は training -> validation -> testing を優先する
    priority = {"training": 0, "validation": 1, "testing": 2}
    hits.sort(key=lambda x: priority.get(x[0], 999))
    return hits[0]


# ------------------------
# ラベル由来の開始・終了ウィンドウ
# ------------------------

@dataclass
class WindowInfo:
    """ラベルから切り出したカメラ出力ウィンドウの情報を保持する。"""
    pair_id: str
    labels_json: str
    q_start: int
    q_end: int


def _safe_int(v) -> Optional[int]:
    """欠損や例外を吸収しながら値を int に変換する。"""
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def load_q_windows_from_labels(labels_root: Path) -> Dict[str, WindowInfo]:
    """labels_root 配下の labels*.json を走査し、q_segment ごとに index 化する。"""
    idx: Dict[str, List[WindowInfo]] = {}

    # v7 ラベラの出力形式: labels_segpair_*.json（現在の命名規則）
    # ただし厳密に固定せず、'labels' で始まる json は広く拾う
    jsons = list(labels_root.rglob("labels*.json"))

    for jp in jsons:
        try:
            obj = json.loads(jp.read_text())
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue

        q_seg = obj.get("q_segment")
        if not isinstance(q_seg, str) or not q_seg:
            continue

        start = obj.get("start") or {}
        end = obj.get("end") or {}
        q_start = _safe_int(start.get("q_i"))
        q_end = _safe_int(end.get("q_i"))
        if q_start is None or q_end is None:
            continue

        if q_start > q_end:
            q_start, q_end = q_end, q_start

        pair_id = obj.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            # 代替策として、親フォルダ名が segpair_xxx__yyy であることが多い
            pair_id = jp.parent.name

        wi = WindowInfo(pair_id=pair_id, labels_json=str(jp), q_start=q_start, q_end=q_end)
        idx.setdefault(q_seg, []).append(wi)

    # q_seg ごとに 1 件を選ぶ（重複時は span が最も広いものを採用）
    out: Dict[str, WindowInfo] = {}
    for q_seg, lst in idx.items():
        lst.sort(key=lambda w: (w.q_end - w.q_start), reverse=True)
        out[q_seg] = lst[0]
        if len(lst) > 1:
            # 挙動は決定的に保ちつつ、警告は後で main で出す
            pass

    return out


# ------------------------
# カメラ処理用の補助関数（export_cam_gray_txt.py に合わせる）
# ------------------------

def decode_gray(jpeg_bytes: bytes) -> np.ndarray:
    """エンコード済み画像をグレースケール配列へ復元する。"""
    import tensorflow as tf  # type: ignore
    import cv2  # type: ignore

    img = tf.image.decode_jpeg(jpeg_bytes).numpy()  # RGB
    # グレースケールへ変換する（OpenCV は BGR 前提）
    gray = cv2.cvtColor(img[..., ::-1], cv2.COLOR_BGR2GRAY)
    return gray


def apply_clahe(gray_u8: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """CLAHE を適用して局所コントラストを補正する。"""
    import cv2  # type: ignore

    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    return clahe.apply(gray_u8)


def vehicle_to_cam_from(fr, cam_name: str) -> np.ndarray:
    """車両座標系からカメラ座標系への変換行列を構成する。"""
    from waymo_open_dataset import dataset_pb2  # type: ignore

    cam_enum = getattr(dataset_pb2.CameraName, cam_name)
    cal = None
    for c in fr.context.camera_calibrations:
        if c.name == cam_enum:
            cal = c
            break
    if cal is None:
        raise ValueError(f"camera calibration not found: {cam_name}")

    T_cam_to_vehicle = np.array(cal.extrinsic.transform, dtype=np.float64).reshape(4, 4)
    # vehicle->camera を返す
    return np.linalg.inv(T_cam_to_vehicle)


def _camera_calib(fr, cam_name: str):
    """指定カメラのキャリブレーション情報を取得する。"""
    from waymo_open_dataset import dataset_pb2  # type: ignore

    cam_enum = getattr(dataset_pb2.CameraName, cam_name)
    for c in fr.context.camera_calibrations:
        if c.name == cam_enum:
            return c
    raise ValueError(f"camera calibration not found: {cam_name}")


def get_camera_image(fr, cam_name: str) -> Optional[bytes]:
    """Waymo Frame から指定カメラの画像を取得する。"""
    from waymo_open_dataset import dataset_pb2  # type: ignore

    cam_enum = getattr(dataset_pb2.CameraName, cam_name)
    for im in fr.images:
        if im.name == cam_enum:
            return im.image
    return None


# ------------------------
# 書き出し処理
# ------------------------

def export_one_segment(
    segment: str,
    subset: str,
    tfrecord_path: Path,
    window: WindowInfo,
    out_root: Path,
    cam: str,
    stride: int,
    gray: bool,
    clahe: bool,
    clahe_clip: float,
    clahe_grid: int,
    overwrite: bool,
) -> Tuple[int, int]:
    """1 セグメント分の画像を書き出し、(exported_frames, skipped_frames) を返す。"""

    import tensorflow as tf  # type: ignore
    from waymo_open_dataset import dataset_pb2  # type: ignore
    import cv2  # type: ignore

    seg_out = out_root / subset / segment
    cam_dir = seg_out / cam
    cam_dir.mkdir(parents=True, exist_ok=True)

    # セグメント単位のメタ情報
    seg_meta_path = seg_out / "export_meta.json"
    if not seg_meta_path.exists() or overwrite:
        seg_meta = {
            "segment": segment,
            "subset": subset,
            "tfrecord": str(tfrecord_path),
            "cam": cam,
            "window": {"q_start": int(window.q_start), "q_end": int(window.q_end)},
            "pair_id": window.pair_id,
            "labels_json": window.labels_json,
            "stride": int(stride),
            "gray": bool(gray),
            "clahe": bool(clahe),
            "clahe_clip": float(clahe_clip),
            "clahe_grid": int(clahe_grid),
        }
        seg_meta_path.write_text(json.dumps(seg_meta, indent=2), encoding="utf-8")

    ds = tf.data.TFRecordDataset([str(tfrecord_path)], compression_type="")

    exported = 0
    skipped = 0

    # 最初のフレームから intrinsics/extrinsics を先読みする（Waymo では通常一定）
    first_frame = True
    v2c0 = None
    cal0 = None

    for i, raw in enumerate(ds):
        if i < window.q_start or i > window.q_end:
            continue
        # stride は window の開始位置に合わせる
        if stride > 1 and ((i - window.q_start) % stride != 0):
            continue

        fr = dataset_pb2.Frame()
        fr.ParseFromString(bytes(raw.numpy()))

        if first_frame:
            v2c0 = vehicle_to_cam_from(fr, cam)
            cal0 = _camera_calib(fr, cam)
            first_frame = False

        # ファイルパス
        out_png = cam_dir / f"{i:05d}.png"
        out_json = cam_dir / f"{i:05d}.json"
        if (out_png.exists() and out_json.exists()) and not overwrite:
            skipped += 1
            continue

        jpeg = get_camera_image(fr, cam)
        if jpeg is None:
            skipped += 1
            continue

        if gray:
            g = decode_gray(jpeg)
            if clahe:
                g = apply_clahe(g, clip=clahe_clip, grid=clahe_grid)
            cv2.imwrite(str(out_png), g)
            img_shape = [int(g.shape[0]), int(g.shape[1]), 1]
        else:
            img = tf.image.decode_jpeg(jpeg).numpy()  # RGB
            cv2.imwrite(str(out_png), img[..., ::-1])  # BGR
            img_shape = [int(img.shape[0]), int(img.shape[1]), int(img.shape[2])]

        # 姿勢 / 行列
        v2w = np.array(fr.pose.transform, dtype=np.float64).reshape(4, 4)
        w2v = np.linalg.inv(v2w)
        if v2c0 is None:
            v2c0 = vehicle_to_cam_from(fr, cam)
        w2c = v2c0 @ w2v

        meta = {
            "frame_index": int(i),
            "frame_timestamp_micros": int(getattr(fr, "timestamp_micros", 0)),
            "camera_name": cam,
            "camera_image_shape": img_shape,
            "vehicle_to_cam": v2c0.tolist(),
            "cam_intrinsic": (list(cal0.intrinsic) if cal0 is not None else []),
            "cam_extrinsic": (list(cal0.extrinsic.transform) if cal0 is not None else []),
            "frame_pose": v2w.tolist(),
            "world_to_cam": w2c.tolist(),
            # window info (later pairingで使いやすいように)
            "window": {"q_start": int(window.q_start), "q_end": int(window.q_end)},
            "pair_id": window.pair_id,
        }
        out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        exported += 1

    return exported, skipped


# ------------------------
# メイン処理
# ------------------------

def main() -> None:
    """CLI 引数を解釈し、入力の読み込みから結果保存までの一連の処理を実行する。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments_txt", type=str, required=True, help="q segments list (one per line)")
    ap.add_argument("--tfrecord_root", type=str, required=True, help="Waymo individual_files root")
    ap.add_argument("--labels_root", type=str, required=True, help="root that contains labels_segpair_*.json")
    ap.add_argument("--out_root", type=str, required=True, help="output root dir")

    ap.add_argument("--cam", type=str, default="FRONT", help="camera name (e.g., FRONT)")
    ap.add_argument("--stride", type=int, default=1, help="export every N frames within [start,end]")

    ap.add_argument("--gray", type=parse_bool, default=True, help="write grayscale PNG")
    ap.add_argument("--clahe", type=parse_bool, default=True, help="apply CLAHE (grayscale only)")
    ap.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clipLimit")
    ap.add_argument("--clahe-grid", type=int, default=8, help="CLAHE tileGridSize")
    ap.add_argument("--overwrite", type=parse_bool, default=False, help="overwrite existing files")

    args = ap.parse_args()

    segs_path = Path(args.segments_txt)
    tfroot = Path(args.tfrecord_root)
    labels_root = Path(args.labels_root)
    out_root = Path(args.out_root)

    segs = read_segments_txt(segs_path)
    if not segs:
        raise SystemExit(f"no segments in: {segs_path}")

    if not labels_root.exists():
        raise SystemExit(f"labels_root not found: {labels_root}")

    q_windows = load_q_windows_from_labels(labels_root)

    print(f"[INFO] segments={len(segs)} from {segs_path}")
    print(f"[INFO] tfrecord_root={tfroot}")
    print(f"[INFO] labels_root={labels_root} (labels with q_window={len(q_windows)})")
    print(f"[INFO] out_root={out_root}")
    print(f"[INFO] cam={args.cam} stride={args.stride} gray={args.gray} clahe={args.clahe}")

    ok = 0
    fail = 0
    total_exported = 0
    total_skipped = 0

    # 重複警告: 同じ q segment が複数の labels json に見つかった場合
    # （load_q_windows_from_labels では span が最も広いものを選んでいる）
    # 軽量さを優先し、欠損セグメントもここでは警告だけにとどめる。

    for seg in segs:
        if seg not in q_windows:
            print(f"[WARN] no q window found in labels for segment: {seg} (skip)")
            fail += 1
            continue
        win = q_windows[seg]

        try:
            subset, tfrec = find_tfrecord_for_segment(tfroot, seg)
        except Exception as e:
            print(f"[ERR] tfrecord not found for {seg}: {e}")
            fail += 1
            continue

        print("\n" + "=" * 80)
        print(f"[INFO] {seg} | subset={subset} | window=[{win.q_start}:{win.q_end}] | pair_id={win.pair_id}")

        try:
            exported, skipped = export_one_segment(
                segment=seg,
                subset=subset,
                tfrecord_path=tfrec,
                window=win,
                out_root=out_root,
                cam=args.cam,
                stride=max(1, int(args.stride)),
                gray=bool(args.gray),
                clahe=bool(args.clahe),
                clahe_clip=float(args.clahe_clip),
                clahe_grid=int(args.clahe_grid),
                overwrite=bool(args.overwrite),
            )
            print(f"[OK] exported={exported} skipped={skipped} -> {out_root/subset/seg/args.cam}")
            ok += 1
            total_exported += exported
            total_skipped += skipped
        except Exception as e:
            print(f"[ERR] failed to export {seg}: {e}")
            fail += 1

    print("\n" + "=" * 80)
    print(f"[DONE] ok={ok} fail={fail} exported={total_exported} skipped={total_skipped}")


if __name__ == "__main__":
    main()
