#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 使い方:
#   python make_loftr_matches_gt_step2a_paper_final_fixed_cam_v7.py --help
#   必要な入力パスと出力先を引数で指定して実行する。

# 目的: Step2A GT 生成器 v7。v6 を破壊せずに包み、検証と delta 解釈をより厳密にする。
"""

1) ds/dl/dyaw はトップレベルではなく ri_meta['delta'] から読む
2) is_valid_gt_npz() を厳密化し、古い / 不完全な GT を skip-existing で見逃さないようにする
3) 旧 run と出力が混ざらないよう、安全側の既定値を使う:
   - --gt-dirname の既定値: gt_matches_step2a_paper_v7
   - --manifest-name の既定値: manifest_matches_step2a_paper_v7.csv

それ以外のロジックは v6 実装と同一のまま。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


V6_PATH = Path(__file__).resolve().parent / "make_loftr_matches_gt_step2a_paper_final_fixed_cam_v6.py"


def _safe_float(v, default: float = 0.0) -> float:
    """欠損や例外を吸収しながら値を float に変換する。"""
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return float(x)


def read_shift_delta_v7(ri_meta: Dict) -> Tuple[float, float, float]:
    """厳密な delta 読み取り器。ri_meta['delta'] のキーだけを正とみなす。"""
    delta = ri_meta.get("delta", {})
    if not isinstance(delta, dict):
        delta = {}
    ds = _safe_float(delta.get("ds_m", 0.0), 0.0)
    dl = _safe_float(delta.get("dl_m", 0.0), 0.0)
    dyaw = _safe_float(delta.get("dyaw_deg", 0.0), 0.0)
    return ds, dl, dyaw


def is_valid_gt_npz_v7(path: Path) -> bool:
    """古い/不完全な npz を必ず再生成させるため、GT 妥当性判定をより厳密に行う。"""
    required = [
        "mkpts0",
        "mkpts1",
        "mkpts1_f",
        "offset1",
        "fine_ok",
        "i_ids",
        "j_ids",
        "meta_json",
        "hw0",
        "hw1",
        "stride",
    ]
    try:
        if (not path.exists()) or path.stat().st_size < 100:
            return False
        with np.load(path, allow_pickle=False) as z:
            for k in required:
                if k not in z.files:
                    return False

            mk0 = z["mkpts0"]
            mk1 = z["mkpts1"]
            mk1f = z["mkpts1_f"]
            off = z["offset1"]
            fine = z["fine_ok"]
            i_ids = z["i_ids"]
            j_ids = z["j_ids"]
            hw0 = z["hw0"]
            hw1 = z["hw1"]

            if mk0.ndim != 2 or mk1.ndim != 2 or mk1f.ndim != 2 or off.ndim != 2:
                return False
            if mk0.shape[1] != 2 or mk1.shape[1] != 2 or mk1f.shape[1] != 2 or off.shape[1] != 2:
                return False
            m = mk0.shape[0]
            if mk1.shape[0] != m or mk1f.shape[0] != m or off.shape[0] != m:
                return False

            if fine.reshape(-1).shape[0] != m:
                return False
            if i_ids.reshape(-1).shape[0] != m or j_ids.reshape(-1).shape[0] != m:
                return False

            if tuple(hw0.shape) != (2,) or tuple(hw1.shape) != (2,):
                return False

            # meta_json は JSON 文字列として復号できる必要がある
            try:
                _ = json.loads(str(z["meta_json"].tolist()))
            except Exception:
                return False

        return True
    except Exception:
        return False


def _load_v6_module(path: Path):
    """互換性維持のため旧 v6 実装モジュールを読み込む。"""
    spec = importlib.util.spec_from_file_location("step2a_v6_mod", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _inject_default_arg(argv: List[str], key: str, value: str) -> List[str]:
    """後方互換のため不足している既定引数を補う。"""
    if key in argv:
        return argv
    return argv + [key, value]


def main() -> None:
    """CLI 引数を解釈し、入力の読み込みから結果保存までの一連の処理を実行する。"""
    if not V6_PATH.exists():
        raise FileNotFoundError(f"v6 script not found: {V6_PATH}")

    mod = _load_v6_module(V6_PATH)

    # 重要な挙動パッチ
    mod.read_shift_delta = read_shift_delta_v7
    mod.is_valid_gt_npz = is_valid_gt_npz_v7

    argv = sys.argv[1:]
    argv = _inject_default_arg(argv, "--gt-dirname", "gt_matches_step2a_paper_v7")
    argv = _inject_default_arg(argv, "--manifest-name", "manifest_matches_step2a_paper_v7.csv")

    print("[INFO][v7] patched read_shift_delta: use ri_meta['delta'] only", flush=True)
    print("[INFO][v7] patched is_valid_gt_npz: strict required-key check", flush=True)
    print("[INFO][v7] default --gt-dirname=gt_matches_step2a_paper_v7", flush=True)
    print("[INFO][v7] default --manifest-name=manifest_matches_step2a_paper_v7.csv", flush=True)

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + argv
        mod.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
