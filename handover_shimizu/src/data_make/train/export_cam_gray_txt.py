#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 使い方:
#   python export_cam_gray_txt.py --help
#   必要な入力パスと出力先を引数で指定して実行する。
#"\\172.22.40.107\e\waymo_tf\perception_v1.4.3\individual_files\training\segment-15832924468527961_1564_160_1584_160_with_camera_labels.tfrecord"

"""
Step 4: カメラ画像（Gray）＋メタの一括出力（全フレーム）
（オプションで CLAHE を適用して保存）

出力:
  <out_root>/<subset>/<segment>/<CAM>/<frame>_cam.png
  <out_root>/<subset>/<segment>/<CAM>/<frame>_cam.json
  （--save-raw true かつ --apply-clahe true のとき）
  <out_root>/<subset>/<segment>/<CAM>/<frame>_cam_raw.png

入力（今回追加）:
  --tfrecord には以下のどちらも指定できる
    1) Waymo セグメントの .tfrecord（従来通り）
    2) .txt ファイル（1行1セグメント）。各行は以下いずれでも良い:
         - tfrecord のフルパス
         - tfrecord の相対パス（txtのあるディレクトリ基準）
         - セグメントstem（例: segment-..._with_camera_labels）
           → <tfrecord_root>/<subset>/<stem>.tfrecord を探す

"""

import sys
import argparse
import json
import math
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset import dataset_pb2 as open_dataset


# txt内がstemの場合に探索する Waymo individual_files のルート（例: <root>/<subset>/<stem>.tfrecord）
# 単一tfrecord指定（従来）には影響しない（txt入力のときだけ使用）。
DEFAULT_TFRECORD_ROOT = "\\172.22.40.107\e\waymo_tf\perception_v1.4.3\individual_files"


def ensure_dir(p: Path):
    """出力先ディレクトリを作成して存在を保証する。"""
    p.mkdir(parents=True, exist_ok=True)


def str2bool(v):
    """CLI などで受け取った真偽値表現を bool に正規化する。"""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {v}")


def mat4_list_to_np(m):
    """行優先の 16 要素配列を 4x4 の NumPy 行列に変換する。"""
    return np.array(m, dtype=np.float64).reshape(4, 4)


def camera_name_from_str(name: str) -> int:
    """文字列から Waymo のカメラ種別 ID を解決する。"""
    nm = name.strip().upper()
    if not hasattr(open_dataset.CameraName, nm):
        raise ValueError(f"Unknown camera name: {name}")
    return getattr(open_dataset.CameraName, nm)


def find_camera_calib(frame, cam_name: int):
    """対象カメラに対応するキャリブレーションをフレーム内から探す。"""
    for c in frame.context.camera_calibrations:
        if c.name == cam_name:
            return c
    return None


def intrinsic_from(calib) -> Tuple[float, float, float, float]:
    """キャリブレーションやメタ情報から内部パラメータを取り出す。"""
    intr = list(calib.intrinsic)
    if len(intr) < 4:
        raise ValueError("camera intrinsic has less than 4 elements")
    return float(intr[0]), float(intr[1]), float(intr[2]), float(intr[3])  # fx, fy, cx, cy


def world_to_cam_from(frame, calib) -> np.ndarray:
    """
    world_to_cam = (camera<-vehicle)^{-1} * (vehicle<-world)^{-1}
                 = (T_vc)^{-1} * (T_wv)^{-1} = T_cv * T_vw
    """
    T_wv = mat4_list_to_np(frame.pose.transform)  # world <- vehicle
    T_vw = np.linalg.inv(T_wv)
    T_vc = mat4_list_to_np(calib.extrinsic.transform)  # vehicle <- camera
    T_cv = np.linalg.inv(T_vc)
    T_wc = T_cv @ T_vw  # world -> camera
    return T_wc


def yaw_from_T_wv(T_wv: np.ndarray) -> float:
    """水平ヨー角 [deg]（右手系, -180..180 正規化）"""
    R = T_wv[:3, :3]
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    while yaw > 180.0:
        yaw -= 360.0
    while yaw < -180.0:
        yaw += 360.0
    return float(yaw)


def decode_gray(image_bytes: bytes) -> Optional[np.ndarray]:
    """エンコード済み画像をグレースケール配列へ復元する。"""
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray


def apply_clahe(gray_u8: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    """CLAHE を適用して局所コントラストを補正する。"""
    if gray_u8.dtype != np.uint8:
        raise ValueError("apply_clahe expects uint8 grayscale image")
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))
    return clahe.apply(gray_u8)


def _read_nonempty_lines(txt_path: Path) -> List[str]:
    """txtを1行ずつ読み、空行・コメント行（#で開始）を除外して返す（順序維持で重複除去）。"""
    lines: List[str] = []
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for raw in f:
                s = raw.strip()
                if not s:
                    continue
                if s.startswith("#"):
                    continue
                lines.append(s)
    except UnicodeDecodeError:
        with open(txt_path, "r", encoding="utf-8-sig") as f:
            for raw in f:
                s = raw.strip()
                if not s:
                    continue
                if s.startswith("#"):
                    continue
                lines.append(s)

    uniq: List[str] = []
    seen = set()
    for s in lines:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def _looks_like_tfrecord_path(s: str) -> bool:
    """文字列が TFRecord パス表記かどうかを判定する。"""
    ls = s.lower()
    return ls.endswith(".tfrecord") or ls.endswith(".tfrecord.gz")


def resolve_tfrecord_path(
    item: str,
    subset: str,
    tfrecord_root: Path,
    txt_dir: Optional[Path] = None,
) -> Path:
    """
    txt内の1行（item）を tfrecord の実パスに解決する。

    解決順:
      1) item が絶対パスで存在
      2) txt_dir/item が存在（相対パス対応）
      3) item が相対パスで存在（カレントディレクトリ基準）
      4) stem とみなして <tfrecord_root>/<subset>/<stem>.tfrecord を探索
      5) item が .tfrecord 名（存在しない）なら <tfrecord_root>/<subset>/<name> を探索
    """
    s = item.strip()
    if not s:
        raise FileNotFoundError("Empty line")

    p = Path(s).expanduser()

    # 1) 絶対パスで存在
    if p.is_absolute() and p.exists():
        return p

    # 2) txt_dir基準の相対パス
    if txt_dir is not None:
        cand = (txt_dir / p).expanduser()
        if cand.exists():
            return cand.resolve()

    # 3) カレントディレクトリ基準の相対パス
    if p.exists():
        return p.resolve()

    # 4) stem から構築（.tfrecord 付与）
    tfroot = Path(tfrecord_root)
    if _looks_like_tfrecord_path(s):
        name = Path(s).name
    else:
        name = f"{s}.tfrecord"

    cand2 = tfroot / subset / name
    if cand2.exists():
        return cand2.resolve()

    raise FileNotFoundError(
        "tfrecord が見つかりません。\n"
        f"  item: {item}\n"
        f"  tried: {p}\n"
        f"  tried: {cand2}\n"
        "  ヒント: txtにフルパスを書くか、--tfrecord-root が正しいか確認してください。"
    )


def _process_one_tfrecord(tfrecord_path: Path, args: argparse.Namespace, strict: bool) -> str:
    """単一tfrecordの処理（元の処理をそのまま関数化）。"""

    try:
        if not tfrecord_path.exists():
            raise FileNotFoundError(f"tfrecord not found: {tfrecord_path}")

        seg_stem = tfrecord_path.stem
        cam_name = camera_name_from_str(args.cam)

        # 出力ディレクトリ: <out_root>/<subset>/<segment>/<CAM>
        out_dir = Path(args.out_root) / args.subset / seg_stem / args.cam.upper()
        ensure_dir(out_dir)

        # フレーム全読み
        frames = []
        for rec in tf.data.TFRecordDataset(str(tfrecord_path), compression_type=""):
            fr = open_dataset.Frame()
            fr.ParseFromString(rec.numpy())
            frames.append(fr)
        n_frames = len(frames)
        print(f"[INFO] segment: {seg_stem} | frames: {n_frames} | cam: {args.cam}")

        if n_frames == 0:
            raise RuntimeError("no frames in tfrecord")

        z = max(4, len(str(n_frames - 1)))

        calib0 = find_camera_calib(frames[0], cam_name)
        if calib0 is None:
            raise RuntimeError(f"camera calibration not found for {args.cam}")

        ok_count = 0
        skip_count = 0
        for i, fr in enumerate(tqdm(frames, desc="export", unit="frame")):
            img_rec = next((im for im in fr.images if im.name == cam_name), None)
            if img_rec is None:
                continue

            gray_raw = decode_gray(img_rec.image)
            if gray_raw is None:
                print(f"[WARN] decode failed at frame {i}")
                continue

            calib = find_camera_calib(fr, cam_name)
            if calib is None:
                print(f"[WARN] calib missing at frame {i}")
                continue

            W, H = int(calib.width), int(calib.height)
            fx, fy, cx, cy = intrinsic_from(calib)
            T_wv = mat4_list_to_np(fr.pose.transform)  # world <- vehicle
            T_wc = world_to_cam_from(fr, calib)

            x, y = float(T_wv[0, 3]), float(T_wv[1, 3])
            yaw_deg = yaw_from_T_wv(T_wv)

            stem = f"{i:0{z}d}_cam"
            png_path = out_dir / f"{stem}.png"
            json_path = out_dir / f"{stem}.json"
            raw_png_path = out_dir / f"{stem}_raw.png"

            if (not args.overwrite) and png_path.exists() and json_path.exists():
                skip_count += 1
                continue

            gray_out = gray_raw
            if args.apply_clahe:
                try:
                    gray_out = apply_clahe(gray_raw, args.clahe_clip_limit, args.clahe_tile_grid)
                except Exception as e:
                    print(f"[WARN] CLAHE failed at frame {i}: {e}")
                    continue

            if args.apply_clahe and args.save_raw:
                if (args.overwrite or (not raw_png_path.exists())):
                    if not cv2.imwrite(str(raw_png_path), gray_raw):
                        print(f"[WARN] failed to write raw {raw_png_path}")
                        continue

            if not cv2.imwrite(str(png_path), gray_out):
                print(f"[WARN] failed to write {png_path}")
                continue

            meta = {
                "segment_id": seg_stem,
                "frame_index": int(i),
                "timestamp_us": int(fr.timestamp_micros),
                "camera_name": args.cam.upper(),
                "image_path": str(png_path),
                "image_path_raw": str(raw_png_path) if (args.apply_clahe and args.save_raw) else None,
                "width": int(W),
                "height": int(H),
                "intrinsic": [fx, fy, cx, cy],
                "world_to_cam": T_wc.tolist(),
                "vehicle_pose": T_wv.tolist(),  # world <- vehicle
                "pose_xyyaw": {"x": float(x), "y": float(y), "yaw_deg": float(yaw_deg)},
                "preprocess": {
                    "grayscale": True,
                    "clahe": {
                        "enabled": bool(args.apply_clahe),
                        "clip_limit": float(args.clahe_clip_limit),
                        "tile_grid": int(args.clahe_tile_grid),
                    }
                }
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            ok_count += 1

        print(f"[OK] wrote {ok_count} frames to: {out_dir} (skipped existing: {skip_count})")
        return "ok"

    except Exception as e:
        print(f"[ERR] failed: {tfrecord_path}")
        print(f"      {type(e).__name__}: {e}")
        if strict:
            raise
        return "err"


def main():
    """CLI 引数を解釈し、入力の読み込みから結果保存までの一連の処理を実行する。"""
    ap = argparse.ArgumentParser(description="Step 4: カメラGray＋メタ出力（全フレーム） + (optional) CLAHE")
    ap.add_argument(
        "--tfrecord",
        type=str,
        required=True,
        help="Waymo segment .tfrecord または tfrecordリスト .txt（WSL: /mnt/e/...）",
    )
    ap.add_argument(
        "--tfrecord-root",
        type=str,
        default=DEFAULT_TFRECORD_ROOT,
        help=(
            "txt内がstemの場合に探索するWaymo individual_filesのルート（既定: "
            f"{DEFAULT_TFRECORD_ROOT}）。例: <root>/<subset>/<stem>.tfrecord"
        ),
    )
    ap.add_argument("--out-root", type=str, required=True,
                    help="出力ルート（例: /mnt/e/waymo/data/cam_gray）")
    ap.add_argument("--subset", type=str, choices=["training", "validation", "testing"], required=True)
    ap.add_argument("--cam", type=str, default="FRONT",
                    help="カメラ名（FRONT/FRONT_LEFT/FRONT_RIGHT/SIDE_LEFT/SIDE_RIGHT）")

    # 前処理
    ap.add_argument("--apply-clahe", type=str2bool, default=False,
                    help="true で CLAHE を適用して保存")
    ap.add_argument("--clahe-clip-limit", type=float, default=2.0,
                    help="CLAHE clip limit（OpenCV createCLAHE の clipLimit）")
    ap.add_argument("--clahe-tile-grid", type=int, default=8,
                    help="CLAHE tile grid（tileGridSize の 1 辺サイズ。例: 8 -> 8x8）")
    ap.add_argument("--save-raw", type=str2bool, default=False,
                    help="true の場合、CLAHE 前の生 Gray も *_cam_raw.png として保存（apply-clahe=true のときのみ有効）")

    # io
    ap.add_argument("--overwrite", type=str2bool, default=False,
                    help="true で既存 png/json を上書き。false の場合は存在すればスキップ")

    args = ap.parse_args()

    in_path = Path(args.tfrecord)
    if not in_path.exists():
        print(f"[ERR] input not found: {in_path}")
        sys.exit(1)

    is_list_mode = in_path.suffix.lower() == ".txt"

    if not is_list_mode:
        # 従来通り: 単一tfrecord
        _process_one_tfrecord(tfrecord_path=in_path, args=args, strict=True)
        return

    # 追加: txtで複数tfrecord
    items = _read_nonempty_lines(in_path)
    if len(items) == 0:
        print(f"[ERR] txt が空です: {in_path}")
        sys.exit(1)

    tfrecord_root = Path(args.tfrecord_root)

    print(f"[INFO] list mode: {in_path}")
    print(f"[INFO] entries: {len(items)}")
    print(f"[INFO] tfrecord_root: {tfrecord_root}")
    print(f"[INFO] subset: {args.subset}")

    ok = 0
    err = 0

    for i, item in enumerate(items):
        print("\n" + "=" * 80)
        print(f"[INFO] ({i + 1}/{len(items)}) item: {item}")

        try:
            tf_path = resolve_tfrecord_path(
                item=item,
                subset=str(args.subset),
                tfrecord_root=tfrecord_root,
                txt_dir=in_path.parent,
            )
        except Exception as e:
            print(f"[ERR] tfrecord 解決に失敗: {item}")
            print(f"      {type(e).__name__}: {e}")
            err += 1
            continue

        st = _process_one_tfrecord(tfrecord_path=tf_path, args=args, strict=False)
        if st == "ok":
            ok += 1
        else:
            err += 1

    print("\n" + "=" * 80)
    print("[INFO] finished list mode")
    print(f"[INFO] ok={ok}, err={err}")

    if err > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
