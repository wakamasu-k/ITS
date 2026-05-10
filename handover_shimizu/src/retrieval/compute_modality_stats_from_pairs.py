#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 使い方:
#   python compute_modality_stats_from_pairs.py --help
#   必要な入力パスと出力先を引数で指定して実行する。


"""pairs から modality_stats を再計測する。

学習/評価で使う Normalize 用の modality_stats.json（cam/anchor の mean/std）を再計測する。

前提:
- 画像は PNG 想定
- 学習コードと同様に、統計は "グレースケール(L)" として 0..1 に正規化した値で計算する
  （学習では L→RGB にしているが、3chが同一値なので mean/std は実質同じ）



"""

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def img_to_gray01(path: str, resize_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """PNG等を開いて L(1ch) にし、float64 の 0..1 配列で返す"""
    img = Image.open(path).convert("L")
    if resize_hw is not None:
        # PILは size=(W,H)
        h, w = resize_hw
        img = img.resize((w, h), resample=Image.BILINEAR)
    x = np.asarray(img, dtype=np.float64) / 255.0
    return x


def update_sums(x: np.ndarray, s: float, ss: float, n: int) -> Tuple[float, float, int]:
    """平均や分散を求めるための累積和を更新する。"""
    s += float(x.sum())
    ss += float((x * x).sum())
    n += int(x.size)
    return s, ss, n


def safe_std(sum_: float, sumsq: float, n: int) -> float:
    """数値誤差に配慮しながら標準偏差を計算する。"""
    if n <= 0:
        return 0.0
    mean = sum_ / n
    var = (sumsq / n) - (mean * mean)
    if var < 0:
        var = 0.0
    return float(np.sqrt(var))


def list_ri_candidates(anchor_dir: Path, ri_count: int, ri_glob: str) -> List[str]:
    """anchor_dir 内の RI 候補を列挙。まず ri_000.. を試し、見つからなければglob。"""
    cands: List[str] = []
    if ri_count and ri_count > 0:
        for k in range(ri_count):
            p = anchor_dir / f"ri_{k:03d}.png"
            if p.is_file():
                cands.append(str(p))
    if (not cands) and ri_glob:
        try:
            cands = [str(p) for p in sorted(anchor_dir.glob(ri_glob)) if p.is_file()]
        except Exception:
            cands = []
    return cands


def main():
    """CLI 引数を解釈し、入力の読み込みから結果保存までの一連の処理を実行する。"""
    ap = argparse.ArgumentParser(description="Compute modality mean/std for cam and anchor images")
    ap.add_argument("--pairs", required=True, help="pairs parquet (cam_png, anchor_png, label optional)")
    ap.add_argument("--out-json", required=True, help="output json path")

    ap.add_argument("--cam-col", default="cam_png")
    ap.add_argument("--anchor-col", default="anchor_png")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--only-label1", action="store_true", help="label列がある場合 label==1 だけ使う")

    ap.add_argument("--sample-every", type=int, default=1, help="N行に1回サンプル（1なら全行）")
    ap.add_argument("--max-rows", type=int, default=-1, help="-1なら全行。正なら先頭max_rows行だけ")

    ap.add_argument(
        "--anchor-mode",
        choices=["anchor_png", "all_ri", "one_ri_per_dir"],
        default="anchor_png",
        help=(
            "anchor統計の対象。"
            "anchor_png=parquetのanchor_pngのみ / "
            "all_ri=anchor_dir内のriを全て / "
            "one_ri_per_dir=anchor_dirごとにriを1枚（seedで決定）"
        ),
    )
    ap.add_argument("--ri-count", type=int, default=8, help="ri_000.. の枚数（0以下ならglob）")
    ap.add_argument("--ri-glob", type=str, default="ri_*.png", help="ri-count<=0 のときのglob")
    ap.add_argument("--seed", type=int, default=0, help="one_ri_per_dir の選択seed")
    ap.add_argument("--dedup-anchors", action="store_true", help="anchor側の同一パス重複を除外")

    ap.add_argument("--resize-h", type=int, default=0, help=">0なら統計前にリサイズするH")
    ap.add_argument("--resize-w", type=int, default=0, help=">0なら統計前にリサイズするW")

    args = ap.parse_args()

    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        raise FileNotFoundError(f"--pairs not found: {pairs_path}")

    df = pd.read_parquet(str(pairs_path))
    if args.only_label1 and (args.label_col in df.columns):
        df = df[df[args.label_col] == 1].copy()

    if args.cam_col not in df.columns or args.anchor_col not in df.columns:
        raise RuntimeError(
            f"pairs parquet must have columns: {args.cam_col}, {args.anchor_col} (got {df.columns.tolist()})"
        )

    if args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    cam_paths = df[args.cam_col].astype(str).tolist()
    anchor_base_paths = df[args.anchor_col].astype(str).tolist()

    step = max(1, int(args.sample_every))

    resize_hw = None
    if args.resize_h > 0 and args.resize_w > 0:
        resize_hw = (int(args.resize_h), int(args.resize_w))

    cam_sum = cam_sumsq = 0.0
    anc_sum = anc_sumsq = 0.0
    cam_pixels = anc_pixels = 0
    cam_count = anc_count = 0
    cam_missing = anc_missing = 0

    # anchorの重複除外
    seen_anchor_paths: Optional[Set[str]] = set() if args.dedup_anchors else None

    rng = np.random.RandomState(int(args.seed) & 0xFFFFFFFF)

    n_rows = len(cam_paths)
    idxs = list(range(0, n_rows, step))

    pbar = tqdm(idxs, desc="rows", ncols=100)
    for i in pbar:
        cam_p = cam_paths[i]
        anc_base = anchor_base_paths[i]

        # --- cam 側 ---
        if Path(cam_p).is_file():
            try:
                x = img_to_gray01(cam_p, resize_hw=resize_hw)
                cam_sum, cam_sumsq, cam_pixels = update_sums(x, cam_sum, cam_sumsq, cam_pixels)
                cam_count += 1
            except Exception:
                cam_missing += 1
        else:
            cam_missing += 1

        # --- anchor 側 ---
        anc_paths: List[str] = []
        if args.anchor_mode == "anchor_png":
            anc_paths = [anc_base]
        else:
            adir = Path(anc_base).parent
            cands = list_ri_candidates(adir, int(args.ri_count), str(args.ri_glob))
            if not cands:
                cands = [anc_base]
            if args.anchor_mode == "all_ri":
                anc_paths = cands
            else:
                # ディレクトリごとに 1 つの RI を使う
                anc_paths = [cands[int(rng.randint(0, len(cands)))] if len(cands) > 0 else anc_base]

        for apath in anc_paths:
            if seen_anchor_paths is not None:
                if apath in seen_anchor_paths:
                    continue
                seen_anchor_paths.add(apath)

            if Path(apath).is_file():
                try:
                    x = img_to_gray01(apath, resize_hw=resize_hw)
                    anc_sum, anc_sumsq, anc_pixels = update_sums(x, anc_sum, anc_sumsq, anc_pixels)
                    anc_count += 1
                except Exception:
                    anc_missing += 1
            else:
                anc_missing += 1

        if cam_pixels > 0 and anc_pixels > 0:
            cam_mean = cam_sum / cam_pixels
            anc_mean = anc_sum / anc_pixels
            pbar.set_postfix(cam_mean=f"{cam_mean:.4f}", anc_mean=f"{anc_mean:.4f}")

    cam_mean = float(cam_sum / cam_pixels) if cam_pixels > 0 else 0.0
    anc_mean = float(anc_sum / anc_pixels) if anc_pixels > 0 else 0.0
    cam_std = safe_std(cam_sum, cam_sumsq, cam_pixels)
    anc_std = safe_std(anc_sum, anc_sumsq, anc_pixels)

    out = {
        "sample_every": int(step),
        "max_rows": int(args.max_rows),
        "anchor_mode": str(args.anchor_mode),
        "ri_count": int(args.ri_count),
        "ri_glob": str(args.ri_glob),
        "resize_h": int(args.resize_h),
        "resize_w": int(args.resize_w),
        "cam_count": int(cam_count),
        "anc_count": int(anc_count),
        "cam_pixels": int(cam_pixels),
        "anc_pixels": int(anc_pixels),
        "cam_missing": int(cam_missing),
        "anc_missing": int(anc_missing),
        "cam_mean": float(cam_mean),
        "cam_std": float(cam_std),
        "anc_mean": float(anc_mean),
        "anc_std": float(anc_std),
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("[DONE] wrote:", str(out_path))
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
