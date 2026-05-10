#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 使い方:
#   python make_static_map_txt.py --help
#   必要な入力パスと出力先を引数で指定して実行する。

"""

make_static_map_txt.py は Step2 の静的点群地図生成スクリプトです。
各フレームの TOP LiDAR を world 座標に重畳し、動的物体のBBox内点と自車ルーフ近傍の点を除去して、`map_static.npz` を作ります。
保存するのは `xyz` と `intensity` で、画像投影はここでは行いません。

主な可変要素は次の3つです。
- `lidar-lines` / `ring-keep-even`: 64/32/16 line 化の設定
- `bbox-vehicle-y-scale`: 車両BBoxの除去強さ
- `ego-x/y/z*`: ルーフ除去範囲

出力は `<out_root>/<subset>/<segment>/` 配下の
- `map_static.npz`
- `render_profile.json`
- `stats.json`
- `intensity_summary.json`（有効時）
です。

運用面では、`overwrite=false` なら既存完了分はスキップします。
`tfrecord` には単一 `.tfrecord` だけでなく `.txt` も渡せて、txt の各行はフルパス・相対パス・segment stem に対応しています。

注意点は2つです。
- このスクリプトでは voxel downsample や ground 除去はしていません
- list mode で一部失敗しても最後に強制的に異常終了しないので、ログの `ok/skip/err` を確認する必要があります

この `map_static.npz` は後段の anchor RI 生成、retrieval データ生成、query depth レンダで直接使われるため、点数や intensity 分布が変わると下流結果も変わります。


Step 2: Waymo セグメントから静的点群マップ（map_static.npz）を生成
 - 全フレームの LiDAR (TOP) を world 座標へ重畳し、動的物体とルーフを除去
 - 64 line / 32 line / 16 line（ring 残し方を指定）を選択可能
 - 強度（intensity）も同次元で保存
 - 反射強度の“画像投影”は本スクリプトでは行わない（Step 3 以降）

追加（今回）:
 - intensity_summary.json を出力（サンプルで分布確認）
 - 既存出力の有無を見てレジューム（--overwrite で上書き制御）
 - --tfrecord に .txt を渡した場合、txt に書かれたすべてのセグメントを順に処理（他処理は同じ）

出力:
 <out_root>/<subset>/<segment>/map_static.npz
 <out_root>/<subset>/<segment>/render_profile.json
 <out_root>/<subset>/<segment>/stats.json
 <out_root>/<subset>/<segment>/intensity_summary.json   # optional




"""

import os
import json
import argparse
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Union, Optional

import numpy as np
from tqdm import tqdm
import tensorflow as tf

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset, label_pb2


# ---------------------- デフォルト設定（CLIで上書き可） ---------------------- #

# txt内がstemの場合に探索する Waymo individual_files のルート（例: <root>/<subset>/<stem>.tfrecord）
# ※既存のコマンド例に合わせたデフォルト。単一tfrecord指定（従来）には影響しない。
DEFAULT_TFRECORD_ROOT = "/mnt/e/waymo/perception_v1.4.3/individual_files"

# クラス別 BBox 拡張率（inside 拡大判定用）
#        長さ(x)      幅(y)      高さ(z)
DEFAULT_BBOX_SCALE = {
    label_pb2.Label.TYPE_VEHICLE   : dict(x=1.0, y=2.5, z=1.0),  # 車両は幅方向を広めに
    label_pb2.Label.TYPE_PEDESTRIAN: dict(x=1.0, y=1.0, z=1.0),
    label_pb2.Label.TYPE_CYCLIST   : dict(x=1.0, y=1.0, z=1.0),
}
# 残すクラス（例：標識）
DEFAULT_KEEP_TYPES = {label_pb2.Label.TYPE_SIGN}

# ルーフ付近の立ち上がり点を除去（車載LiDAR周り）
DEFAULT_EGO_BOX = {'x': [-3.1, 5.0], 'y': [-3.1, 3.1], 'z': [-0.2, 0.2]}

DEFAULT_PERCENTILES = [0.0, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.5, 99.9, 99.99, 100.0]


# --------------------------------------------------------------------------- #

def _str2bool(s: str) -> bool:
    """CLI などで受け取った真偽値表現を bool に正規化する。"""
    return str(s).strip().lower() in ["1", "true", "t", "yes", "y"]


def _resolve_lidar_lines(lidar_lines: Optional[int], downsample_32l: bool) -> int:
    """設定値から使用する LiDAR ライン数を解決する。"""
    if lidar_lines is not None:
        if int(lidar_lines) not in (64, 32, 16):
            raise ValueError(f"unsupported --lidar-lines: {lidar_lines}")
        return int(lidar_lines)
    return 32 if downsample_32l else 64


def _safe_float(x) -> float:
    """欠損や例外を吸収しながら値を float に変換する。"""
    try:
        return float(x)
    except Exception:
        return float("nan")


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
        # 万一BOM付きなどだった場合の保険
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


def _np_percentile_compat(x: np.ndarray, q: Union[float, List[float], np.ndarray]):
    """
    NumPy互換:
      - 新: np.percentile(..., method="linear")
      - 旧: np.percentile(..., interpolation="linear")
    """
    try:
        return np.percentile(x, q, method="linear")
    except TypeError:
        try:
            return np.percentile(x, q, interpolation="linear")
        except TypeError:
            return np.percentile(x, q)


def _percentile_stats(x: np.ndarray, percentiles: List[float]) -> Dict[str, float]:
    """数値配列のパーセンタイル統計を計算する。"""
    x = np.asarray(x)
    if x.size == 0:
        return {f"p{p:g}": float("nan") for p in percentiles}
    vals = _np_percentile_compat(x, percentiles)
    out = {}
    for p, v in zip(percentiles, vals):
        out[f"p{p:g}"] = _safe_float(v)
    return out


def _sample_array(arr: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    """大きな配列から確認用の代表サンプルを抜き出す。"""
    arr = np.asarray(arr)
    n = arr.shape[0]
    if sample_size <= 0 or sample_size >= n:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=sample_size, dtype=np.int64)
    return arr[idx]


def _ascii_hist(counts: np.ndarray, edges: np.ndarray, width: int = 48) -> str:
    """数値分布を簡易な ASCII ヒストグラムへ整形する。"""
    counts = np.asarray(counts, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if counts.size == 0:
        return "(empty hist)"
    m = counts.max() if counts.max() > 0 else 1.0
    lines = []
    for i, c in enumerate(counts):
        lo = edges[i]
        hi = edges[i + 1]
        bar = int(round(width * (c / m)))
        lines.append(f"[{lo: .4g},{hi: .4g}) | {'#' * bar} {int(c)}")
    return "\n".join(lines)


def compute_intensity_summary(intensity: np.ndarray,
                              sample_size: int,
                              bins: int,
                              hist_log1p: bool,
                              seed: int,
                              percentiles: List[float]) -> Dict[str, Any]:
    """反射強度画像の要約統計をまとめて計算する。"""
    inten = np.asarray(intensity).reshape(-1)
    finite = np.isfinite(inten)
    inten_f = inten[finite]
    n_all = int(inten.size)
    n_finite = int(inten_f.size)

    sample = _sample_array(inten_f, sample_size, seed).astype(np.float64)
    n_sample = int(sample.size)

    basic = {
        "min": _safe_float(np.min(sample)) if n_sample > 0 else float("nan"),
        "max": _safe_float(np.max(sample)) if n_sample > 0 else float("nan"),
        "mean": _safe_float(np.mean(sample)) if n_sample > 0 else float("nan"),
        "std": _safe_float(np.std(sample)) if n_sample > 0 else float("nan"),
        "zero_ratio": float(np.mean(sample == 0.0)) if n_sample > 0 else float("nan"),
        "neg_ratio": float(np.mean(sample < 0.0)) if n_sample > 0 else float("nan"),
    }

    p_all = _percentile_stats(sample, percentiles)
    sample_nz = sample[sample != 0.0]
    p_nz = _percentile_stats(sample_nz, percentiles)

    hsrc = np.log1p(sample) if hist_log1p else sample
    if hsrc.size > 0:
        lo = float(_np_percentile_compat(hsrc, 0.1))
        hi = float(_np_percentile_compat(hsrc, 99.9))
        if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
            lo, hi = float(np.min(hsrc)), float(np.max(hsrc))
    else:
        lo, hi = 0.0, 1.0

    counts, edges = np.histogram(np.clip(hsrc, lo, hi), bins=int(bins), range=(lo, hi))
    ascii_hist = _ascii_hist(counts, edges)

    return {
        "n_all": n_all,
        "n_finite": n_finite,
        "n_sample": n_sample,
        "sample_size_arg": int(sample_size),
        "seed": int(seed),
        "basic": basic,
        "percentiles_all": p_all,
        "percentiles_nonzero": p_nz,
        "hist": {
            "bins": int(bins),
            "log1p": bool(hist_log1p),
            "clip_lo": lo,
            "clip_hi": hi,
            "counts": counts.astype(int).tolist(),
            "edges": edges.astype(float).tolist(),
            "ascii": ascii_hist,
        },
    }


def mat4_list_to_np(m):
    """行優先の 16 要素配列を 4x4 の NumPy 行列に変換する。"""
    return np.array(m, dtype=np.float64).reshape(4, 4)


def build_static_map(frames,
                     lidar_lines: int,
                     ring_keep_even: bool,
                     bbox_scale: dict,
                     keep_types: set,
                     ego_box: dict):
    """
    すべてのフレームから LiDAR(TOP) の点群を world 座標に重畳し、
    動的物体（BBox）とルーフ近傍を除去した xyz/intensity を返す。
    """
    pts_world_acc = []
    inten_acc = []

    REMOVE_TYPES = set(bbox_scale.keys())

    for fr in tqdm(frames, desc="  静的マップ生成", unit="frame"):
        # ---------- Range Image を取得 ----------
        rng, proj, _, ri_pose = frame_utils.parse_range_image_and_camera_projection(fr)
        ri_top = rng[open_dataset.LaserName.TOP][0]  # [range_image, range_image_pose]
        H, W, C = [d if isinstance(d, int) else d.size for d in ri_top.shape.dims]
        ri_np = np.asarray(ri_top.data, dtype=np.float32).reshape(H, W, C)

        valid = ri_np[..., 0] > 0
        ring_id = np.repeat(np.arange(H)[:, None], W, axis=1)[valid]  # (num_valid,)
        inten = ri_np[..., 1][valid]

        # ---------- レンジ画像 -> 点群 ----------
        pts_list, _ = frame_utils.convert_range_image_to_point_cloud(
            fr, rng, proj, ri_pose, keep_polar_features=False
        )
        xyz = pts_list[0]  # LaserName.TOP
        if xyz.shape[0] != ring_id.size:
            # 稀に一致しない場合は最小長に合わせて切る（安全策）
            n = min(xyz.shape[0], ring_id.size)
            xyz = xyz[:n]
            inten = inten[:n]
            ring_id = ring_id[:n]

        # ---------- line 数に応じた疎化 ----------
        if lidar_lines == 32:
            keep = (ring_id % 2 == (0 if ring_keep_even else 1))
            xyz, inten = xyz[keep], inten[keep]
        elif lidar_lines == 16:
            # keep_even=true なら 0,4,8,... / false なら 1,5,9,...
            keep = (ring_id % 4 == (0 if ring_keep_even else 1))
            xyz, inten = xyz[keep], inten[keep]
        elif lidar_lines != 64:
            raise ValueError(f"unsupported lidar_lines: {lidar_lines}")

        # ---------- BBox による動的物体除去 ----------
        keep_mask = np.ones(len(xyz), dtype=bool)
        if len(fr.laser_labels) > 0:
            for lbl in fr.laser_labels:
                # 残すクラス or 除去対象でないクラスはスキップ
                if (lbl.type in keep_types) or (lbl.type not in REMOVE_TYPES):
                    continue
                scale = bbox_scale[lbl.type]
                c = np.array([lbl.box.center_x, lbl.box.center_y, lbl.box.center_z], dtype=np.float32)
                d = np.array([lbl.box.length * scale["x"],
                              lbl.box.width  * scale["y"],
                              lbl.box.height * scale["z"]], dtype=np.float32)
                ch, sh = np.cos(lbl.box.heading), np.sin(lbl.box.heading)
                Rz = np.array([[ ch,  sh, 0],
                               [-sh,  ch, 0],
                               [  0,   0, 1]], dtype=np.float32)
                inside = np.all(np.abs((xyz - c) @ Rz) <= d / 2.0, axis=1)
                keep_mask &= ~inside

        # ---------- ルーフ除去 ----------
        xmin, xmax = ego_box["x"]
        ymin, ymax = ego_box["y"]
        zmin, zmax = ego_box["z"]
        roof = np.all((xyz > [xmin, ymin, zmin]) & (xyz < [xmax, ymax, zmax]), axis=1)
        keep_mask &= ~roof

        if not np.any(keep_mask):
            continue

        # ---------- 車両座標 -> ワールド座標 ----------
        T_wv = mat4_list_to_np(fr.pose.transform)  # world <- vehicle
        pts_h = np.hstack([xyz[keep_mask], np.ones((keep_mask.sum(), 1), dtype=np.float32)])
        pts_w = (pts_h @ T_wv.T)[:, :3]

        pts_world_acc.append(pts_w.astype(np.float32))
        inten_acc.append(inten[keep_mask].astype(np.float32))

    if not pts_world_acc:
        return None, None

    xyz_world = np.concatenate(pts_world_acc, axis=0)
    intensity = np.concatenate(inten_acc, axis=0)
    return xyz_world, intensity


def _process_one_tfrecord(tfrecord_path: Path, args: argparse.Namespace) -> str:
    """単一tfrecordの処理（元の main の処理をそのまま関数化）。"""

    if not tfrecord_path.exists():
        print(f"[ERR] tfrecord が見つかりません: {tfrecord_path}")
        return "err"

    seg_stem = tfrecord_path.stem
    out_dir = Path(args.out_root) / args.subset / seg_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 既存出力の確認（レジューム対応）
    map_path = out_dir / "map_static.npz"
    render_profile_path = out_dir / "render_profile.json"
    stats_path = out_dir / "stats.json"
    int_summary_path = out_dir / "intensity_summary.json"

    if not args.overwrite:
        exists_all = map_path.exists() and render_profile_path.exists() and stats_path.exists()
        if args.write_intensity_summary:
            exists_all = exists_all and int_summary_path.exists()
        if exists_all:
            print(f"[SKIP] 既に map_static.npz / json が存在するためスキップします: {out_dir}")
            return "skip"
        else:
            # どれか1つでも存在するなら「中途半端な状態」とみなして上書き再生成
            if map_path.exists() or render_profile_path.exists() or stats_path.exists() or \
               (args.write_intensity_summary and int_summary_path.exists()):
                print(f"[INFO] 出力が一部だけ存在するため上書きして再生成します: {out_dir}")

    # 設定まとめ
    bbox_scale = DEFAULT_BBOX_SCALE.copy()
    bbox_scale[label_pb2.Label.TYPE_VEHICLE] = dict(x=1.0, y=args.bbox_vehicle_y_scale, z=1.0)
    keep_types = DEFAULT_KEEP_TYPES
    ego_box = {"x": [args.ego_xmin, args.ego_xmax],
               "y": [args.ego_ymin, args.ego_ymax],
               "z": [args.ego_zmin, args.ego_zmax]}

    # フレーム列挙
    ds = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="")
    frames = [open_dataset.Frame.FromString(r.numpy()) for r in ds]
    total_frames = len(frames)
    print(f"[INFO] segment: {seg_stem} | frames: {total_frames}")
    print(f"[INFO] lidar_lines: {args.lidar_lines} | keep_even: {args.ring_keep_even}")
    print(f"[INFO] BBox scale (vehicle y): {args.bbox_vehicle_y_scale}")

    # 静的マップ生成
    xyz, intensity = build_static_map(
        frames,
        lidar_lines=args.lidar_lines,
        ring_keep_even=args.ring_keep_even,
        bbox_scale=bbox_scale,
        keep_types=keep_types,
        ego_box=ego_box,
    )

    if xyz is None or intensity is None:
        print("[WARN] 有効な点群が得られませんでした。終了します。")
        return "err"

    # 保存（npz）
    np.savez_compressed(out_dir / "map_static.npz", xyz=xyz, intensity=intensity)

    # 再現用プロファイル
    render_profile = {
        "segment_id": seg_stem,
        "subset": args.subset,
        "lidar_lines": int(args.lidar_lines),
        "downsample_32l": bool(args.downsample_32l),
        "ring_keep_even": bool(args.ring_keep_even),
        "bbox_scale": {str(k): v for k, v in bbox_scale.items()},
        "keep_types": [int(t) for t in keep_types],
        "ego_box": ego_box,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tool": "make_static_map.py",
        "version": "v1.1-resume",
    }
    with open(out_dir / "render_profile.json", "w", encoding="utf-8") as f:
        json.dump(render_profile, f, indent=2, ensure_ascii=False)

    # 簡易統計
    stats = {
        "num_points": int(xyz.shape[0]),
        "min_xyz": np.asarray(xyz, dtype=float).min(axis=0).tolist(),
        "max_xyz": np.asarray(xyz, dtype=float).max(axis=0).tolist(),
        "intensity_min": float(np.min(intensity)),
        "intensity_max": float(np.max(intensity)),
        "frames": int(total_frames),
    }

    # intensity summary（追加）
    if args.write_intensity_summary:
        percentiles = [float(s) for s in args.int_percentiles.split(",") if s.strip() != ""]
        summary = compute_intensity_summary(
            intensity=intensity,
            sample_size=int(args.int_sample_size),
            bins=int(args.int_bins),
            hist_log1p=bool(args.int_hist_log1p),
            seed=int(args.int_seed),
            percentiles=percentiles,
        )
        out_json = out_dir / "intensity_summary.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        stats["intensity_summary_json"] = str(out_json)

    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"[OK] map_static.npz を保存: {out_dir}")
    print(f"     points={stats['num_points']}, frames={total_frames}")
    print(f"     render_profile.json / stats.json を保存しました。")
    if args.write_intensity_summary:
        print(f"     intensity_summary.json も保存しました。")

    return "ok"


def main():
    """CLI 引数を解釈し、入力の読み込みから結果保存までの一連の処理を実行する。"""
    ap = argparse.ArgumentParser(description="Step 2: 静的点群マップ生成（map_static.npz のみ）")
    ap.add_argument(
        "--tfrecord",
        type=str,
        required=True,
        help=(
            "Waymo segment .tfrecord または tfrecordリスト .txt（1行1セグメント）。"
            " txtの各行は tfrecordのフルパス / 相対パス / セグメントstem を許可。"
        ),
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
                    help="出力ルート（例: /mnt/e/waymo/data/maps）")
    ap.add_argument("--subset", type=str, choices=["training", "validation", "testing"], required=True)
    ap.add_argument("--downsample-32l", type=_str2bool, default=True,
                    help="true で 32 line 化（偶数/奇数 ring を片側だけ残す）")
    ap.add_argument("--lidar-lines", type=int, choices=[64, 32, 16], default=None,
                    help="LiDAR line 数。指定時は downsample-32l より優先（64/32/16）")
    ap.add_argument("--ring-keep-even", type=_str2bool, default=True,
                    help="32 line 時に偶数 ring を残すか（false で奇数）")
    ap.add_argument("--bbox-vehicle-y-scale", type=float, default=2.5,
                    help="車両 BBox の幅拡大率（y）")
    ap.add_argument("--ego-xmin", type=float, default=-3.1)
    ap.add_argument("--ego-xmax", type=float, default=5.0)
    ap.add_argument("--ego-ymin", type=float, default=-3.1)
    ap.add_argument("--ego-ymax", type=float, default=3.1)
    ap.add_argument("--ego-zmin", type=float, default=-0.2)
    ap.add_argument("--ego-zmax", type=float, default=0.2)

    # intensity summary（追加）
    ap.add_argument("--write-intensity-summary", type=_str2bool, default=True,
                    help="true で intensity_summary.json を保存")
    ap.add_argument("--int-sample-size", type=int, default=2_000_000,
                    help="強度分布サンプル数（0で全点）")
    ap.add_argument("--int-bins", type=int, default=30, help="ヒスト bin 数")
    ap.add_argument("--int-hist-log1p", type=_str2bool, default=True,
                    help="true で log1p(intensity) のヒスト")
    ap.add_argument("--int-seed", type=int, default=0, help="サンプル乱数 seed")
    ap.add_argument("--int-percentiles", type=str,
                    default=",".join(str(p) for p in DEFAULT_PERCENTILES),
                    help="percentiles をカンマ区切り指定")

    # 上書き制御（レジューム用）
    ap.add_argument("--overwrite", type=_str2bool, default=False,
                    help="true なら既存の map_static.npz / json を上書きして再計算。"
                         "false なら出力がそろっていればスキップしてレジュームに使う。")

    args = ap.parse_args()
    args.lidar_lines = _resolve_lidar_lines(args.lidar_lines, args.downsample_32l)

    tfrecord_arg = Path(args.tfrecord)
    if not tfrecord_arg.exists():
        print(f"[ERR] 入力が見つかりません: {tfrecord_arg}")
        return

    is_list_mode = tfrecord_arg.suffix.lower() == ".txt"

    if not is_list_mode:
        # 従来通り: 単一tfrecord
        _process_one_tfrecord(tfrecord_path=tfrecord_arg, args=args)
        return

    # 今回追加: txtで複数tfrecord
    items = _read_nonempty_lines(tfrecord_arg)
    if len(items) == 0:
        print(f"[ERR] txt が空です: {tfrecord_arg}")
        return

    tfrecord_root = Path(args.tfrecord_root)

    print(f"[INFO] list mode: {tfrecord_arg}")
    print(f"[INFO] entries: {len(items)}")
    print(f"[INFO] tfrecord_root: {tfrecord_root}")
    print(f"[INFO] subset: {args.subset}")

    ok = 0
    skip = 0
    err = 0

    for i, item in enumerate(items):
        print("\n" + "=" * 80)
        print(f"[INFO] ({i + 1}/{len(items)}) item: {item}")

        try:
            tfrecord_path = resolve_tfrecord_path(
                item=item,
                subset=str(args.subset),
                tfrecord_root=tfrecord_root,
                txt_dir=tfrecord_arg.parent,
            )
        except Exception as e:
            print(f"[ERR] tfrecord 解決に失敗: {item}")
            print(f"      {type(e).__name__}: {e}")
            err += 1
            continue

        status = _process_one_tfrecord(tfrecord_path=tfrecord_path, args=args)
        if status == "ok":
            ok += 1
        elif status == "skip":
            skip += 1
        else:
            err += 1

    print("\n" + "=" * 80)
    print("[INFO] finished list mode")
    print(f"[INFO] ok={ok}, skip={skip}, err={err}")


if __name__ == "__main__":
    main()
