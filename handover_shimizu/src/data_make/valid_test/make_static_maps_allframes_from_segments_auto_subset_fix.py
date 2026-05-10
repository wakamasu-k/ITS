#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 使い方:
#   python make_static_maps_allframes_from_segments_auto_subset_fix.py --help
#   必要な入力パスと出力先を引数で指定して実行する。

"""
make_static_maps_allframes_from_segments_auto_subset.py

目的
----
`make_static_map_txt.py` (元コード) を “セグメント一覧(txt)” に対してバッチ実行するラッパーです。
セグメントごとに TFRecord が training / validation / testing のどこにあるかを自動判定し、
subset ごとのリストに分けてから `make_static_map_txt.py` を呼びます。

重要
----
あなたが実行したコマンドで指定していた以下のような引数は、元の `make_static_map_txt.py` には存在しません:
  --min-dist, --max-dist, --z-min, --z-max, --bbox-extra-m, --ego-x-range, --ego-y-range, --ego-z-range
そのため、以前の版のラッパーが “未知の引数をそのまま make_static_map_txt.py に転送” してしまい、
`unrecognized arguments` で落ちていました。

この版では:
- `make_static_map_txt.py` が受け付ける引数だけを転送します。
- 互換のために `--ego-*-range` は受け付け、`--ego-*-min/max` に変換して転送します。
- `--min-dist` 等は “受け付けるが無視” し、必ず WARN を出します（黙って無視しません）。

  """
from __future__ import annotations

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


def str2bool(v):
    """CLI などで受け取った真偽値表現を bool に正規化する。"""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {v}")


def read_segments_txt(p: Path) -> List[str]:
    """セグメント一覧テキストを読み込む。"""
    segs: List[str] = []
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith("#"):
            continue
        # tfrecord のフルパスでも segment stem だけでも受け付ける
        if ln.endswith(".tfrecord"):
            ln = ln[:-len(".tfrecord")]
        segs.append(ln)
    return segs


def detect_subset(tfrecord_root: Path, segment: str, subset_order: List[str]) -> str:
    """
    Find the subset folder under tfrecord_root that contains `<segment>.tfrecord`.
    Returns the subset name.
    """
    # segment がすでにフルパスである場合もある
    sp = Path(segment)
    if sp.exists() and sp.suffix == ".tfrecord":
        # ユーザーがフルパスを渡した場合は、親フォルダ名から subset を推定する。
        return sp.parent.name

    for subset in subset_order:
        cand = tfrecord_root / subset / f"{segment}.tfrecord"
        if cand.exists():
            return subset
    raise FileNotFoundError(f"TFRecord not found for segment '{segment}' under {tfrecord_root} (checked {subset_order})")


def write_list(path: Path, items: List[str]) -> None:
    """パスや ID の一覧をテキストファイルへ書き出す。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(items) + ("\n" if items else ""), encoding="utf-8")


def main() -> None:
    """CLI 引数を解釈し、入力の読み込みから結果保存までの一連の処理を実行する。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments_txt", required=True, help="1行1セグメント（.tfrecord なし推奨）。フルパスも可")
    ap.add_argument("--tfrecord_root", required=True, help=".../individual_files")
    ap.add_argument("--out_root", required=True, help="出力 root（training/validation/... の下に作られます）")
    ap.add_argument("--make_static_map_py", default="make_static_map_txt.py", help="元コードへのパス")
    ap.add_argument("--python", default=sys.executable, help="呼び出しに使う python（デフォルト: この python）")

    # ---- make_static_map_txt.py が受け付ける引数群 ----
    ap.add_argument("--downsample-32l", dest="downsample_32l", type=str2bool, default=True)
    ap.add_argument("--lidar-lines", dest="lidar_lines", type=int, choices=[64, 32, 16], default=None,
                    help="LiDAR line count (64/32/16)。指定時は downsample-32l より優先")
    ap.add_argument("--ring-keep-even", dest="ring_keep_even", type=str2bool, default=True)
    ap.add_argument("--bbox-vehicle-y-scale", dest="bbox_vehicle_y_scale", type=float, default=1.2)

    ap.add_argument("--ego-xmin", dest="ego_xmin", type=float, default=None)
    ap.add_argument("--ego-xmax", dest="ego_xmax", type=float, default=None)
    ap.add_argument("--ego-ymin", dest="ego_ymin", type=float, default=None)
    ap.add_argument("--ego-ymax", dest="ego_ymax", type=float, default=None)
    ap.add_argument("--ego-zmin", dest="ego_zmin", type=float, default=None)
    ap.add_argument("--ego-zmax", dest="ego_zmax", type=float, default=None)

    ap.add_argument("--write-intensity-summary", dest="write_intensity_summary", type=str2bool, default=False)
    ap.add_argument("--int-sample-size", dest="int_sample_size", type=int, default=200000)
    ap.add_argument("--int-bins", dest="int_bins", type=int, default=64)
    ap.add_argument("--int-hist-log1p", dest="int_hist_log1p", type=str2bool, default=True)
    ap.add_argument("--int-seed", dest="int_seed", type=int, default=0)
    ap.add_argument("--int-percentiles", dest="int_percentiles", type=str, default="0,1,5,10,25,50,75,90,95,99,100")
    ap.add_argument("--overwrite", dest="overwrite", type=str2bool, default=False)

    # ---- 互換/無視する引数（あなたのコマンド互換のため）----
    ap.add_argument("--ego-x-range", dest="ego_x_range", type=float, default=None,
                    help="(互換) ego_xmin=-r, ego_xmax=+r に変換して使用")
    ap.add_argument("--ego-y-range", dest="ego_y_range", type=float, default=None,
                    help="(互換) ego_ymin=-r, ego_ymax=+r に変換して使用")
    ap.add_argument("--ego-z-range", dest="ego_z_range", type=float, default=None,
                    help="(互換) ego_zmin=-r, ego_zmax=+r に変換して使用")

    ap.add_argument("--min-dist", dest="min_dist", type=float, default=None,
                    help="(無視) make_static_map_txt.py に該当オプションが無いので無視")
    ap.add_argument("--max-dist", dest="max_dist", type=float, default=None,
                    help="(無視) make_static_map_txt.py に該当オプションが無いので無視")
    ap.add_argument("--z-min", dest="z_min", type=float, default=None,
                    help="(無視) make_static_map_txt.py に該当オプションが無いので無視")
    ap.add_argument("--z-max", dest="z_max", type=float, default=None,
                    help="(無視) make_static_map_txt.py に該当オプションが無いので無視")
    ap.add_argument("--bbox-extra-m", dest="bbox_extra_m", type=float, default=None,
                    help="(無視) make_static_map_txt.py に該当オプションが無いので無視")

    args = ap.parse_args()

    # ライン数指定（新）: 未指定なら従来互換で downsample-32l から決定
    if args.lidar_lines is None:
        args.lidar_lines = 32 if args.downsample_32l else 64
    else:
        # 互換用: 既存ログ/引数の意味と矛盾しないよう同期
        args.downsample_32l = (args.lidar_lines != 64)

    segments_txt = Path(args.segments_txt)
    tfrecord_root = Path(args.tfrecord_root)
    out_root = Path(args.out_root)
    make_static_map_py = Path(args.make_static_map_py)
    py = args.python

    out_root.mkdir(parents=True, exist_ok=True)
    lists_dir = out_root / "_lists"
    lists_dir.mkdir(parents=True, exist_ok=True)

    # WARN: 無視される引数
    ignored = []
    for name in ("min_dist", "max_dist", "z_min", "z_max", "bbox_extra_m"):
        if getattr(args, name) is not None:
            ignored.append((name, getattr(args, name)))
    if ignored:
        for k, v in ignored:
            print(f"[WARN] ignoring --{k.replace('_','-')} {v} (make_static_map_txt.py does not support it)")

    # ユーザーが ego range を渡し、かつ明示的な min/max を渡していない場合は変換する
    if args.ego_x_range is not None and (args.ego_xmin is None and args.ego_xmax is None):
        args.ego_xmin = -float(args.ego_x_range)
        args.ego_xmax = +float(args.ego_x_range)
    if args.ego_y_range is not None and (args.ego_ymin is None and args.ego_ymax is None):
        args.ego_ymin = -float(args.ego_y_range)
        args.ego_ymax = +float(args.ego_y_range)
    if args.ego_z_range is not None and (args.ego_zmin is None and args.ego_zmax is None):
        args.ego_zmin = -float(args.ego_z_range)
        args.ego_zmax = +float(args.ego_z_range)

    # セグメントごとに subset を判定する
    segs = read_segments_txt(segments_txt)
    if not segs:
        raise RuntimeError(f"no segments in {segments_txt}")

    subset_order = ["training", "validation", "testing"]
    buckets: Dict[str, List[str]] = {s: [] for s in subset_order}
    fails: List[Tuple[str, str]] = []

    for seg in segs:
        try:
            subset = detect_subset(tfrecord_root, seg, subset_order=subset_order)
            if subset not in buckets:
                buckets[subset] = []
            buckets[subset].append(seg)
        except Exception as e:
            fails.append((seg, str(e)))

    for seg, msg in fails:
        print(f"[WARN] subset detect failed: {seg} | {msg}")

    # list file を書き出し、subset ごとに make_static_map_txt.py を実行する
    ok_total = 0
    for subset in subset_order:
        items = buckets.get(subset, [])
        if not items:
            continue

        list_path = lists_dir / f"segments_{subset}.txt"
        write_list(list_path, items)

        cmd: List[str] = [
            py,
            str(make_static_map_py),
            "--tfrecord", str(list_path),
            "--subset", subset,
            "--out-root", str(out_root),
            "--tfrecord-root", str(tfrecord_root),
            "--downsample-32l", str(args.downsample_32l).lower(),
            "--lidar-lines", str(args.lidar_lines),
            "--ring-keep-even", str(args.ring_keep_even).lower(),
            "--bbox-vehicle-y-scale", str(args.bbox_vehicle_y_scale),
            "--write-intensity-summary", str(args.write_intensity_summary).lower(),
            "--int-sample-size", str(args.int_sample_size),
            "--int-bins", str(args.int_bins),
            "--int-hist-log1p", str(args.int_hist_log1p).lower(),
            "--int-seed", str(args.int_seed),
            "--int-percentiles", str(args.int_percentiles),
            "--overwrite", str(args.overwrite).lower(),
        ]

        # ego min/max はユーザーが指定したときだけ転送し、それ以外は make_static_map_txt.py の既定値を維持する
        if args.ego_xmin is not None:
            cmd += ["--ego-xmin", str(args.ego_xmin)]
        if args.ego_xmax is not None:
            cmd += ["--ego-xmax", str(args.ego_xmax)]
        if args.ego_ymin is not None:
            cmd += ["--ego-ymin", str(args.ego_ymin)]
        if args.ego_ymax is not None:
            cmd += ["--ego-ymax", str(args.ego_ymax)]
        if args.ego_zmin is not None:
            cmd += ["--ego-zmin", str(args.ego_zmin)]
        if args.ego_zmax is not None:
            cmd += ["--ego-zmax", str(args.ego_zmax)]

        print("[RUN]", " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            raise RuntimeError(f"make_static_map_txt.py failed (subset={subset}) rc={rc}")
        ok_total += len(items)

    print(f"[DONE] ok={ok_total} fail={len(fails)}")
    if fails:
        print("[DONE] failed segments list (first 20):")
        for seg, msg in fails[:20]:
            print("  -", seg, "|", msg)


if __name__ == "__main__":
    main()
