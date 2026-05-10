#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 使い方:
#   python evaluate.py --help
#   必要な入力パスと出力先を引数で指定して実行する。


"""evaluatev2（拡張版 evaluator）

- Strict 指標（Recall/MRR/MedianRank）と、ICPで整列した距離に基づく Soft 指標（例: t=2m,5m の Recall）を同時に計算します。
- 解析用に per-query CSV/JSONL、topK JSONL、failure dump、anchor failure summary、（任意で）埋め込み .npy を出力します。

"""


from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import csv
import hashlib
from functools import lru_cache
import pandas as pd
try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore



def _str2bool(v: str | bool) -> bool:
    """CLI などで受け取った真偽値表現を bool に正規化する。"""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on"}



def _fmt_m(t_m: float) -> str:
    """距離しきい値をタグや列名向けの文字列へ整形する。

    Examples:
      2.0  -> '2'
      2.5  -> '2p5'
      10.0 -> '10'
    """
    s = f"{float(t_m):.6f}".rstrip("0").rstrip(".")
    s = s.replace("-", "m").replace(".", "p")
    return s


def _read_table(path: Path):
    """CSV や Parquet などの表形式ファイルを DataFrame として読み込む。"""
    import pandas as pd

    if path.suffix.lower() in {".parquet"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _ensure_columns(df, required: List[str], where: str) -> None:
    """必要な列が揃っているか確認し、不足時は例外を投げる。"""
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"{where}: missing required columns: {miss}. available={list(df.columns)}")


def _load_modality_stats(stats_path: Optional[str]) -> Dict[str, float]:
    # 想定する主要キー（スカラー統計）: cam_mean/cam_std/anc_mean/anc_std
    """モダリティ別の mean/std 設定を読み込む。"""
    if not stats_path:
        # 未指定時は ImageNet 系の値へフォールバックする
        return {
            "cam_mean": 0.485,
            "cam_std": 0.229,
            "anc_mean": 0.485,
            "anc_std": 0.229,
        }
    p = Path(stats_path)
    if not p.exists():
        raise FileNotFoundError(f"modality_stats not found: {p}")
    d = json.loads(p.read_text())
    # 多少の表記ゆれは許容する
    out = {}
    out["cam_mean"] = float(d.get("cam_mean", 0.485))
    out["cam_std"] = float(d.get("cam_std", 0.229))
    out["anc_mean"] = float(d.get("anc_mean", d.get("ri_mean", 0.485)))
    out["anc_std"] = float(d.get("anc_std", d.get("ri_std", 0.229)))
    return out


def _build_transform(image_h: int, image_w: int, mean: float, std: float):
    """評価や学習で使う画像前処理パイプラインを組み立てる。"""
    from torchvision import transforms

    # 学習時とそろえるため、前処理はシンプルに保つ:
    # - (H,W) へリサイズ
    # - tensor へ変換
    # - 正規化（3ch すべて同じスカラー mean/std を使う）
    return transforms.Compose(
        [
            transforms.Resize((image_h, image_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean, mean, mean], std=[std, std, std]),
        ]
    )


class _ImagePathDataset:
    """画像パス一覧から遅延読み込みでテンソルを返す Dataset。"""
    def __init__(self, paths: List[str], tfm):
        """インスタンス生成時に必要な設定値と内部状態を初期化する。"""
        self.paths = paths
        self.tfm = tfm

    def __len__(self):
        """保持しているサンプル数を返す。"""
        return len(self.paths)

    def __getitem__(self, idx):
        """指定インデックスのサンプルを読み込んで返す。"""
        from PIL import Image

        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        x = self.tfm(img)
        return x, idx


def _embed_paths(
    model,
    paths: List[str],
    kind: str,
    tfm,
    device: str,
    batch_size: int,
    num_workers: int,
    amp: bool,
) -> np.ndarray:
    """画像パスのリストを、L2 正規化済み NumPy 特徴量へ埋め込む。"""
    import torch
    from torch.utils.data import DataLoader

    ds = _ImagePathDataset(paths, tfm)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.startswith("cuda")),
        drop_last=False,
    )

    enc = model.cam if kind == "cam" else model.int
    enc.eval()

    feats = None

    # AMP の文脈設定は torch の版差を吸収できるようにする。
    # - amp=false -> autocast なし
    # - CUDA では torch.cuda.amp.autocast を使う
    # - cpu  -> torch.cpu.amp.autocast（利用可能なら）
    from contextlib import nullcontext

    if bool(amp) and str(device).startswith("cuda"):
        autocast = torch.cuda.amp.autocast
    elif bool(amp) and (str(device) == "cpu" or str(device).startswith("cpu")):
        cpu_amp = getattr(getattr(torch, "cpu", None), "amp", None)
        autocast = getattr(cpu_amp, "autocast", nullcontext) if cpu_amp is not None else nullcontext
    else:
        autocast = nullcontext

    with torch.no_grad():
        for x, idx in dl:
            x = x.to(device, non_blocking=True)
            with autocast():
                y = enc(x)
            y = y.float().detach().cpu().numpy()
            if feats is None:
                feats = np.zeros((len(paths), y.shape[1]), dtype=np.float32)
            feats[idx.numpy()] = y

    assert feats is not None
    # L2 正規化する
    n = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / np.clip(n, 1e-12, None)
    return feats


def _topk_from_sim(sim: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    # sim の形状: [Nq, Ndb]
    # 戻り値は idx [Nq,k]（降順）と sim [Nq,k]
    """類似度行列から上位 K 件の候補を抽出する。"""
    if k >= sim.shape[1]:
        idx = np.argsort(-sim, axis=1)
        return idx, np.take_along_axis(sim, idx, axis=1)

    # まず部分ソートし、その中で再ソートする
    part = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    part_sim = np.take_along_axis(sim, part, axis=1)
    order = np.argsort(-part_sim, axis=1)
    idx = np.take_along_axis(part, order, axis=1)
    top_sim = np.take_along_axis(sim, idx, axis=1)
    return idx, top_sim


def _recall_at_k(ranks_1based: np.ndarray, k: int) -> float:
    """Recall@K を計算する。"""
    if ranks_1based.size == 0:
        return float("nan")
    return float(np.mean(ranks_1based <= k))


def _mrr(ranks_1based: np.ndarray) -> float:
    """Mean Reciprocal Rank を計算する。"""
    if ranks_1based.size == 0:
        return float("nan")
    return float(np.mean(1.0 / ranks_1based.astype(np.float64)))


def _median_rank(ranks_1based: np.ndarray) -> float:
    """順位の中央値を計算する。"""
    if ranks_1based.size == 0:
        return float("nan")
    return float(np.median(ranks_1based.astype(np.float64)))


def _describe_floats(values: List[Optional[float]], *, prefix: str = "") -> dict:
    """ログや JSON 向けの簡潔な数値要約を作る。

    - Filters out None / NaN / inf
    - Returns count + mean/std + min/p10/p50/p90/max
    """
    clean = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if len(clean) == 0:
        return {prefix + "count": 0}

    arr = np.asarray(clean, dtype=np.float64)
    qs = np.percentile(arr, [10, 50, 90]).tolist()
    return {
        prefix + "count": int(arr.size),
        prefix + "mean": float(arr.mean()),
        prefix + "std": float(arr.std()),
        prefix + "min": float(arr.min()),
        prefix + "p10": float(qs[0]),
        prefix + "p50": float(qs[1]),
        prefix + "p90": float(qs[2]),
        prefix + "max": float(arr.max()),
    }


def _read_q_xy_from_cam_json(cam_png: str) -> Optional[Tuple[float, float]]:
    # /path/000010.png -> /path/000010.json へ対応付ける
    """query カメラ JSON から xy 位置を読み取る。"""
    jp = Path(cam_png).with_suffix(".json")
    if not jp.exists():
        return None
    try:
        obj = json.loads(jp.read_text())
    except Exception:
        return None

    # exporter 形式: {"frame_pose": [[...],[...],[...],[...]], ...}
    fp = obj.get("frame_pose")
    if not (isinstance(fp, list) and len(fp) == 4 and isinstance(fp[0], list) and len(fp[0]) == 4):
        return None

    try:
        x = float(fp[0][3])
        y = float(fp[1][3])
        return x, y
    except Exception:
        return None



def _html_escape(s: str) -> str:
    """HTML 出力向けに文字列をエスケープする。"""
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )


def _hash_path(s: str) -> str:
    """パス文字列から安定したハッシュ値を生成する。"""
    return hashlib.md5((s or "").encode("utf-8")).hexdigest()[:16]


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    """辞書列を JSONL 形式で書き出す。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@lru_cache(maxsize=20000)
def _img_stats_gray(path: str) -> Dict[str, float]:
    """解析用の簡易画像統計を計算する（学習用ではない）。

    Returns {} if PIL is unavailable or the file can't be read.
    """
    if Image is None:
        return {}
    try:
        im = Image.open(path).convert("L")
        arr = np.array(im, dtype=np.float32)
    except Exception:
        return {}
    if arr.ndim != 2:
        return {}
    h, w = arr.shape
    mean = float(arr.mean())
    std = float(arr.std())
    nonzero_frac = float((arr > 0).mean())
    # エントロピー（自然対数）
    hist, _ = np.histogram(arr, bins=256, range=(0.0, 255.0), density=True)
    hist = hist[hist > 0]
    entropy = float(-(hist * np.log(hist + 1e-12)).sum())
    # Laplacian 分散（単純な 4 近傍）
    lap = (-4.0 * arr
           + np.roll(arr, 1, axis=0) + np.roll(arr, -1, axis=0)
           + np.roll(arr, 1, axis=1) + np.roll(arr, -1, axis=1))
    lap_var = float(lap.var())
    return {
        "w": float(w),
        "h": float(h),
        "mean": mean,
        "std": std,
        "nonzero_frac": nonzero_frac,
        "entropy": entropy,
        "lap_var": lap_var,
    }


def _make_thumb(src_path: str, thumbs_dir: Path, thumb_w: int) -> Optional[str]:
    """サムネイルを作る（既存があれば再利用する）。戻り値は thumbs_dir からの相対ファイル名。"""
    if Image is None:
        return None
    if not src_path:
        return None
    sp = Path(src_path)
    if not sp.exists():
        return None
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    ext = sp.suffix.lower() or ".png"
    out_name = f"{_hash_path(src_path)}{ext}"
    out_p = thumbs_dir / out_name
    if out_p.exists():
        return out_name
    try:
        im = Image.open(sp)
        w, h = im.size
        if w > thumb_w:
            new_h = max(1, int(round(h * (thumb_w / float(w)))))
            im = im.resize((thumb_w, new_h))
        im.save(out_p)
        return out_name
    except Exception:
        return None


def _write_failure_gallery(
    out_html: Path,
    failures: List[dict],
    out_dir: Path,
    thumb_w: int = 512,
) -> None:
    """failure case 用の軽量な HTML ギャラリーを書き出す。"""
    if Image is None:
        print("[WARN] PIL not available; skip failure gallery")
        return
    thumbs_dir = out_dir / "_thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    parts: List[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append("<title>Failure gallery</title>")
    parts.append(
        "<style>"
        "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:16px;}"
        ".case{border:1px solid #ddd;border-radius:10px;padding:12px;margin:14px 0;}"
        ".row{display:grid;grid-template-columns:520px 1fr;gap:12px;align-items:start;}"
        "img{max-width:100%;height:auto;border:1px solid #eee;border-radius:6px;}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px;}"
        ".card{border:1px solid #eee;border-radius:10px;padding:8px;}"
        ".small{font-size:12px;color:#444;word-break:break-all;}"
        "code{font-size:11px;}"
        "</style>"
    )
    parts.append("</head><body>")
    parts.append(f"<h1>Failure cases: {len(failures)}</h1>")
    parts.append("<div class='small'>Query / strict GT / top retrieved anchors.</div>")

    for c, rec in enumerate(failures, 1):
        q_path = rec.get("q_cam_png", "")
        gt_path = rec.get("gt_anchor_png", "")
        q_thumb = _make_thumb(q_path, thumbs_dir, thumb_w)
        gt_thumb = _make_thumb(gt_path, thumbs_dir, thumb_w)

        pair_id = _html_escape(str(rec.get("pair_id", "")))
        strict_rank = rec.get("strict_rank", "")
        q_idx = rec.get("q_idx", "")

        parts.append(f"<div class='case' id='c{c}'>")
        parts.append(f"<h2>#{c} {pair_id} | strict_rank={strict_rank} | q_idx={q_idx}</h2>")
        parts.append("<div class='row'>")

        # 左側: query + GT
        parts.append("<div>")
        parts.append("<h3>Query</h3>")
        if q_thumb:
            parts.append(f"<img src='{_html_escape('_thumbs/' + q_thumb)}'>")
        parts.append(f"<div class='small'><code>{_html_escape(q_path)}</code></div>")
        parts.append("<h3>Strict GT</h3>")
        if gt_thumb:
            parts.append(f"<img src='{_html_escape('_thumbs/' + gt_thumb)}'>")
        parts.append(f"<div class='small'><code>{_html_escape(gt_path)}</code></div>")
        parts.append("</div>")  # left

        # 右側: 上位 retrieved 候補
        parts.append("<div>")
        parts.append("<h3>Top retrieved</h3>")
        parts.append("<div class='grid'>")
        for p in rec.get("top", []) or []:
            a_path = p.get("anchor_png", "")
            a_thumb = _make_thumb(a_path, thumbs_dir, 320)
            title = f"rank={p.get('rank','')} score={p.get('score','')} pair={p.get('pair_id','')}"
            dist = p.get("dist_m", None)
            if dist is not None:
                try:
                    title += f" dist={float(dist):.3f}m"
                except Exception:
                    pass
            parts.append("<div class='card'>")
            parts.append(f"<div class='small'>{_html_escape(title)}</div>")
            if a_thumb:
                parts.append(f"<img src='{_html_escape('_thumbs/' + a_thumb)}'>")
            parts.append(f"<div class='small'><code>{_html_escape(a_path)}</code></div>")
            parts.append("</div>")
        parts.append("</div>")  # grid
        parts.append("</div>")  # right

        parts.append("</div>")  # row
        parts.append("</div>")  # case

    parts.append("</body></html>")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text("\n".join(parts), encoding="utf-8")

def _pose_rowmajor_to_xyz(rowmajor: str) -> Optional[Tuple[float, float, float]]:
    # "a,b,c,..." の 16 要素形式を想定する
    """行優先の 4x4 姿勢行列から xyz 平行移動を抽出する。"""
    if not isinstance(rowmajor, str):
        return None
    parts = [p.strip() for p in rowmajor.split(",") if p.strip()]
    if len(parts) != 16:
        return None
    try:
        m = [float(p) for p in parts]
        # 行優先の 4x4 行列
        x = m[3]
        y = m[7]
        z = m[11]
        return x, y, z
    except Exception:
        return None


def _yaw_tx_ty_tz_to_T(yaw_deg: float, tx: float, ty: float, tz: float) -> np.ndarray:
    """yaw と並進成分から 4x4 変換行列を組み立てる。"""
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


def _normalize_wsl_path(p: str) -> str:
    """WSL と Windows が混在するパス表記を読める形に正規化する。"""
    p = (p or "").strip()
    if not p:
        return p
    if len(p) >= 3 and p[1] == ":" and (p[2] == "\\" or p[2] == "/"):
        drive = p[0].lower()
        rest = p[2:].lstrip("\\/").replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p


def _load_icp_T_map(icp_csv: Path) -> Dict[str, np.ndarray]:
    """pair_id -> T_db_to_q（4x4）の辞書を返す。"""
    import csv

    if not icp_csv.exists():
        raise FileNotFoundError(f"icp_csv not found: {icp_csv}")

    out: Dict[str, np.ndarray] = {}
    with icp_csv.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            pair_id = (row.get("pair_id") or "").strip()
            if not pair_id:
                # 代替として db_seg/q_seg から構成する
                db_seg = (row.get("db_seg") or row.get("db_segment") or "").strip()
                q_seg = (row.get("q_seg") or row.get("q_segment") or "").strip()
                if db_seg and q_seg:
                    m1 = re.match(r"segment-([0-9]{6})", db_seg)
                    m2 = re.match(r"segment-([0-9]{6})", q_seg)
                    if m1 and m2:
                        pair_id = f"segpair_{m1.group(1)}__{m2.group(1)}"
            if not pair_id:
                continue

            try:
                yaw = float(row.get("final_yaw_deg"))
                tx = float(row.get("final_tx"))
                ty = float(row.get("final_ty"))
                tz = float(row.get("final_tz"))
            except Exception:
                continue

            out[pair_id] = _yaw_tx_ty_tz_to_T(yaw, tx, ty, tz)

    if not out:
        raise RuntimeError(f"failed to read any ICP rows from: {icp_csv}")
    return out


def _load_model(args):
    """モデル構造を組み立て、チェックポイント重みを読み込む。

    NOTE:
      We intentionally define the model architecture *inside this script*.
      Reason: importing the training script may require extra optional deps
      (e.g. tensorboard) that are not installed in evaluation-only envs.

      The architecture below mirrors the one in:
        diff_train_contrastive_vlad_one_val2_epochri_v3_resume_fix1.py
      (ResNet + NetVLAD + MLP; dual encoders for cam/intensity).
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models

    class NetVLAD(nn.Module):
        """2D 特徴マップを NetVLAD(K*C) で集約し、L2 正規化する。"""

        def __init__(
            self,
            num_clusters: int = 32,
            dim: int = 512,
            alpha: float = 1.0,
            normalize_input: bool = True,
            intra_norm: bool = True,
        ):
            """インスタンス生成時に必要な設定値と内部状態を初期化する。"""
            super().__init__()
            self.K = int(num_clusters)
            self.D = int(dim)
            self.alpha = float(alpha)
            self.normalize_input = bool(normalize_input)
            self.intra_norm = bool(intra_norm)

            self.conv = nn.Conv2d(self.D, self.K, kernel_size=(1, 1), bias=True)
            self.centroids = nn.Parameter(torch.rand(self.K, self.D))
            self._init_params()

        def _init_params(self) -> None:
            """層パラメータを初期化する。"""
            with torch.no_grad():
                self.conv.weight.copy_((2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1))
                self.conv.bias.copy_(-self.alpha * self.centroids.norm(dim=1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x の形状: [B,C,H,W]
            """順伝播を実行して必要な特徴量または損失を計算する。"""
            B, C, H, W = x.shape
            if self.normalize_input:
                x = F.normalize(x, p=2, dim=1)

            soft = self.conv(x).view(B, self.K, -1)  # [B,K,N]
            soft = F.softmax(soft, dim=1)
            x_flat = x.view(B, C, -1)  # [B,C,N]

            vlad = torch.zeros(B, self.K, C, device=x.device, dtype=x.dtype)
            for k in range(self.K):
                a = soft[:, k, :].unsqueeze(1)  # [B,1,N]
                residual = x_flat - self.centroids[k].view(1, C, 1)  # [B,C,N]
                vlad[:, k, :] = (a * residual).sum(dim=2)  # [B,C]

            if self.intra_norm:
                vlad = F.normalize(vlad, p=2, dim=2)
            vlad = vlad.view(B, -1)
            vlad = F.normalize(vlad, p=2, dim=1)
            return vlad

    def _build_resnet(backbone: str):
        """バージョン差異に強い ResNet 構築関数（重みダウンロードは行わない）。"""

        backbone = str(backbone).lower().strip()
        if backbone not in {"resnet18", "resnet34", "resnet50"}:
            raise ValueError(f"unsupported backbone: {backbone}")

        ctor = getattr(models, backbone)
        try:
            # torchvision>=0.13 の場合
            net = ctor(weights=None)
        except TypeError:
            # 古い torchvision の場合
            net = ctor(pretrained=False)
        return net

    class ResNetVLAD(nn.Module):
        """ResNet バックボーンと NetVLAD を組み合わせた埋め込み抽出器。"""
        def __init__(self, backbone: str = "resnet18", out_dim: int = 256, clusters: int = 32, freeze_stages: int = 2):
            """インスタンス生成時に必要な設定値と内部状態を初期化する。"""
            super().__init__()
            net = _build_resnet(backbone)

            if backbone == "resnet50":
                feat_dim = 2048
            else:
                feat_dim = 512

            # stem + layer1..4 を使う（avgpool/fc は除外）
            self.conv1, self.bn1, self.relu, self.maxpool = net.conv1, net.bn1, net.relu, net.maxpool
            self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4

            # resnet50 の場合は 2048->512 へ圧縮する
            if feat_dim != 512:
                self.reduce = nn.Conv2d(feat_dim, 512, kernel_size=1, bias=False)
                feat_dim = 512
            else:
                self.reduce = None

            self._freeze_stages_base_compatible(int(freeze_stages))

            self.vlad = NetVLAD(num_clusters=int(clusters), dim=int(feat_dim), alpha=1.0, normalize_input=True, intra_norm=True)
            self.proj = nn.Sequential(
                nn.Linear(int(clusters) * int(feat_dim), 1024),
                nn.BatchNorm1d(1024),
                nn.GELU(),
                nn.Linear(1024, int(out_dim)),
            )

        def _freeze_stages_base_compatible(self, k: int) -> None:
            # k>=1: conv1, bn1 / k>=2: +layer1 / k>=3: +layer2 を凍結する
            """互換性を保ちながら backbone の一部ステージを凍結する。"""
            names_to_freeze = []
            if k >= 1:
                names_to_freeze += ["conv1", "bn1"]
            if k >= 2:
                names_to_freeze += ["layer1"]
            if k >= 3:
                names_to_freeze += ["layer2"]

            for n, p in self.named_parameters():
                if any(n.startswith(prefix) for prefix in names_to_freeze):
                    p.requires_grad = False

            # BN は eval のまま保つ（学習スクリプトと同じ挙動）
            for m in self.modules():
                cls = m.__class__.__name__.lower()
                if "batchnorm" in cls:
                    m.eval()

            # 後段ステージは学習可能なままにする
            if hasattr(self, "layer3"):
                self.layer3.train()
                for p in self.layer3.parameters():
                    p.requires_grad = True
            if hasattr(self, "layer4"):
                self.layer4.train()
                for p in self.layer4.parameters():
                    p.requires_grad = True

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """順伝播を実行して必要な特徴量または損失を計算する。"""
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            if self.reduce is not None:
                x = self.reduce(x)
            v = self.vlad(x)
            e = self.proj(v)
            e = F.normalize(e, p=2, dim=1)
            return e

    class DualEncoder(nn.Module):
        """Camera encoder と intensity encoder を別重みで持つ。"""

        def __init__(self, backbone: str = "resnet18", out_dim: int = 256, clusters: int = 32, freeze_stages: int = 2):
            """インスタンス生成時に必要な設定値と内部状態を初期化する。"""
            super().__init__()
            self.cam = ResNetVLAD(backbone, out_dim, clusters, freeze_stages)
            self.int = ResNetVLAD(backbone, out_dim, clusters, freeze_stages)

        def forward(self, cam_img: torch.Tensor, anc_img: torch.Tensor):
            """順伝播を実行して必要な特徴量または損失を計算する。"""
            zc = self.cam(cam_img)
            za = self.int(anc_img)
            return zc, za

    model = DualEncoder(
        backbone=str(args.backbone),
        out_dim=int(args.embed_dim),
        clusters=int(args.clusters),
        freeze_stages=int(args.freeze_stages),
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    # よくある 'module.' prefix を考慮する
    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[WARN] missing keys in checkpoint: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[WARN] unexpected keys in checkpoint: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    model.eval()
    return model


def main() -> None:
    """CLI 引数を解釈し、入力の読み込みから結果保存までの一連の処理を実行する。"""
    ap = argparse.ArgumentParser()

    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--pairs", type=str, required=True, help="pairs parquet/csv (cam_png + anchor_png GT)")
    ap.add_argument("--db", type=str, required=True, help="db parquet/csv (anchor_png gallery)")
    ap.add_argument("--icp_csv", type=str, default="", help="icp csv for soft metrics")
    ap.add_argument("--invert_icp", type=str, default="false")

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--name", type=str, default="test")

    # モデル / 入力設定
    ap.add_argument("--backbone", type=str, default="resnet34")
    ap.add_argument("--clusters", type=int, default=64)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--freeze_stages", type=int, default=0)
    ap.add_argument("--image_h", type=int, default=512)
    ap.add_argument("--image_w", type=int, default=864)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--amp", type=_str2bool, default=False, help="Enable AMP (fp16). Accepts true/false.")

    ap.add_argument("--modality_stats", type=str, default="")

    # strict 指標
    ap.add_argument("--strict_ks", type=str, default="1,5,10,30,50")
    ap.add_argument("--include_r1pct", type=str, default="true")

    # --- failure / 解析用ダンプ ---
    ap.add_argument("--write_failures", type=_str2bool, default=True,
                    help="Write failure analysis files (csv/jsonl/html).")
    ap.add_argument("--failure_k", type=int, default=5,
                    help="Define failure as strict GT not in top-K (default: 5).")
    ap.add_argument("--failure_topn", type=int, default=10,
                    help="Store top-N retrieved anchors per failure.")
    ap.add_argument("--failure_max_cases", type=int, default=500,
                    help="Max failure cases to dump (0 = no limit).")
    ap.add_argument("--write_failure_gallery", type=_str2bool, default=True,
                    help="Write HTML gallery for failures (requires PIL).")
    ap.add_argument("--failure_thumb_w", type=int, default=512,
                    help="Thumbnail width for failure gallery.")
    ap.add_argument("--failure_compute_image_stats", type=_str2bool, default=False,
                    help="Compute simple image stats for query/GT/topN in failure dumps.")
    ap.add_argument("--write_anchor_failure_summary", type=_str2bool, default=True,
                    help="Write per-anchor failure summary csv.")
    ap.add_argument("--write_failure_summary", type=_str2bool, default=True,
                    help="Write aggregated failure summary json (rank/segment/pair breakdown).")
    ap.add_argument("--write_failure_by_pair", type=_str2bool, default=True,
                    help="Write per-pair_id failure rate csv.")
    ap.add_argument("--write_failure_pair_confusion", type=_str2bool, default=True,
                    help="Write top1 pair-id confusion csv for failures.")
    ap.add_argument("--write_per_query_csv", type=_str2bool, default=True,
                    help="Write per-query CSV with strict + soft diagnostics (hits/ranks), top1, distances, etc. Strongly recommended for analysis.")
    ap.add_argument("--write_per_query_jsonl", type=_str2bool, default=False,
                    help="Write per-query JSONL (same content as per_query.csv but JSONL).")
    ap.add_argument("--write_embeddings_npy", type=_str2bool, default=False,
                    help="Write query/DB embeddings to .npy (+ path list txt) for later analysis without re-embedding.")
    ap.add_argument("--write_topk_jsonl", type=_str2bool, default=True,
                    help="Write per-query topK retrieval list as JSONL (rank, sim, dist_m, pair_id, and soft-GT flags). Useful for visual/debug analysis.")
    ap.add_argument("--topk_dump_k", type=int, default=50,
                    help="Top-K to dump into JSONL. If <=0, disables topK dump (also disabled when write_topk_jsonl=false).")

    # soft 指標
    ap.add_argument("--soft_thresholds_m", type=str, default="1,2,5,10")
    ap.add_argument("--soft_ks", type=str, default="1,5,10")
    ap.add_argument("--soft_rank_t_main_m", type=float, default=2.0)

    # その他の設定
    ap.add_argument("--max_queries", type=int, default=0, help="debug: if >0, evaluate only first N queries")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name = str(args.name)


    strict_ks = [int(x) for x in args.strict_ks.split(",") if x.strip()]
    soft_ks = [int(x) for x in args.soft_ks.split(",") if x.strip()]
    soft_ts = [float(x) for x in args.soft_thresholds_m.split(",") if x.strip()]

    include_r1pct = _str2bool(args.include_r1pct)
    invert_icp = _str2bool(args.invert_icp)

    pairs_path = Path(_normalize_wsl_path(args.pairs))
    db_path = Path(_normalize_wsl_path(args.db))

    df_pairs = _read_table(pairs_path)
    df_db = _read_table(db_path)

    # 2 通りの命名揺れをどちらも受け付ける
    if "gt_anchor_png" not in df_pairs.columns and "anchor_png" in df_pairs.columns:
        df_pairs = df_pairs.copy()
        df_pairs["gt_anchor_png"] = df_pairs["anchor_png"]

    _ensure_columns(df_pairs, ["cam_png", "gt_anchor_png", "pair_id"], where="pairs")
    _ensure_columns(df_db, ["anchor_png", "pair_id"], where="db")

    # 後段で DB lookup に numpy index（0..N-1）を使うため、index を整合させる。
    # parquet/csv が RangeIndex 以外で保存されている場合は、微妙なバグを避けるため index を振り直す
    # （例: df_db.loc[db_idx] が意図しない行を拾うケース）
    df_pairs = df_pairs.reset_index(drop=True)
    df_db = df_db.reset_index(drop=True)

    # 画像パス（WSL 形式）は早めに正規化し、文字列照合（GT->DB）と画像読み込みの結果を一致させる。
    df_pairs = df_pairs.copy()
    df_pairs["cam_png"] = df_pairs["cam_png"].astype(str).apply(_normalize_wsl_path)
    df_pairs["gt_anchor_png"] = df_pairs["gt_anchor_png"].astype(str).apply(_normalize_wsl_path)
    df_db = df_db.copy()
    df_db["anchor_png"] = df_db["anchor_png"].astype(str).apply(_normalize_wsl_path)

    # anchor xyz 列
    if not all(c in df_db.columns for c in ["anchor_x_db", "anchor_y_db", "anchor_z_db"]):
        if "anchor_pose_rowmajor" in df_db.columns:
            xyz = df_db["anchor_pose_rowmajor"].apply(_pose_rowmajor_to_xyz)
            df_db = df_db.copy()
            df_db["anchor_x_db"] = xyz.apply(lambda t: t[0] if t else np.nan)
            df_db["anchor_y_db"] = xyz.apply(lambda t: t[1] if t else np.nan)
            df_db["anchor_z_db"] = xyz.apply(lambda t: t[2] if t else np.nan)
        else:
            raise ValueError(
                "db: need anchor_x_db/anchor_y_db/anchor_z_db columns or anchor_pose_rowmajor"
            )


    # DB anchor の xyz/xy 配列を前計算する（高速な距離判定 / dump 用）
    anchor_x_db = df_db["anchor_x_db"].astype(float).to_numpy()
    anchor_y_db = df_db["anchor_y_db"].astype(float).to_numpy()
    anchor_z_db = df_db["anchor_z_db"].astype(float).to_numpy()
    anchor_xy_db = np.stack([anchor_x_db, anchor_y_db], axis=1)

    # 必要なら query 数を制限する（デバッグ用）
    if args.max_queries and args.max_queries > 0:
        df_pairs = df_pairs.iloc[: args.max_queries].copy()

    q_paths = df_pairs["cam_png"].astype(str).tolist()
    gt_paths = df_pairs["gt_anchor_png"].astype(str).tolist()

    db_paths = df_db["anchor_png"].astype(str).tolist()

    # GT アンカーを DB インデックスへ対応付ける
    anchor2idx: Dict[str, int] = {}
    for i, p in enumerate(db_paths):
        if p in anchor2idx:
            # 通常は起きない想定だが、念のため先頭を採用する。
            continue
        anchor2idx[p] = i

    gt_idx = np.full((len(gt_paths),), -1, dtype=np.int64)
    miss_gt = 0
    for i, gp in enumerate(gt_paths):
        j = anchor2idx.get(gp, None)
        if j is None:
            miss_gt += 1
            continue
        gt_idx[i] = j

    if miss_gt:
        print(f"[WARN] strict GT anchor not found in DB for {miss_gt}/{len(gt_paths)} queries")

    keep = gt_idx >= 0
    if not np.all(keep):
        df_pairs = df_pairs.loc[keep].reset_index(drop=True)
        q_paths = [p for p, k in zip(q_paths, keep) if k]
        gt_idx = gt_idx[keep]

    n_q = len(q_paths)
    n_db = len(db_paths)
    print(f"[INFO] Q={n_q} DB={n_db} pairs_file={pairs_path.name} db_file={db_path.name}")

    # 1% K
    k_1pct = int(math.ceil(0.01 * n_db))
    if k_1pct < 1:
        k_1pct = 1

    max_k = 0
    if strict_ks:
        max_k = max(max_k, max(strict_ks))
    if soft_ks:
        max_k = max(max_k, max(soft_ks))
    if include_r1pct:
        max_k = max(max_k, k_1pct)
    # failure dump や debug dump に十分な top-k を確保しておく。
    if args.failure_topn and args.failure_topn > 0:
        max_k = max(max_k, int(args.failure_topn))
    if getattr(args, "topk_dump_k", 0) and args.topk_dump_k and args.topk_dump_k > 0:
        max_k = max(max_k, int(args.topk_dump_k))

    # モデルを構築する
    model = _load_model(args)

    device = args.device
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] cuda requested but not available; falling back to cpu")
        device = "cpu"

    model = model.to(device)

    stats = _load_modality_stats(args.modality_stats)
    tfm_q = _build_transform(args.image_h, args.image_w, stats["cam_mean"], stats["cam_std"])
    tfm_db = _build_transform(args.image_h, args.image_w, stats["anc_mean"], stats["anc_std"])

    # 埋め込みを計算する
    print("[INFO] embedding Q...")
    q_feat = _embed_paths(
        model,
        q_paths,
        kind="cam",
        tfm=tfm_q,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=args.amp,
    )

    print("[INFO] embedding DB...")
    db_feat = _embed_paths(
        model,
        db_paths,
        kind="int",
        tfm=tfm_db,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=args.amp,
    )

    # 任意: 埋め込みを dump する（再埋め込みせず後段解析を速くしたいときに便利）。
    if getattr(args, "write_embeddings_npy", False) and bool(args.write_embeddings_npy):
        q_npy = out_dir / f"{name}_q_embeddings.npy"
        db_npy = out_dir / f"{name}_db_embeddings.npy"
        np.save(q_npy, q_feat)
        np.save(db_npy, db_feat)

        # インデックス再現性のため、パス順序も保存する。
        (out_dir / f"{name}_q_paths.txt").write_text("\n".join(map(str, q_paths)) + "\n", encoding="utf-8")
        (out_dir / f"{name}_db_paths.txt").write_text("\n".join(map(str, db_paths)) + "\n", encoding="utf-8")
        meta = {
            "name": name,
            "q": int(len(q_paths)),
            "db": int(len(db_paths)),
            "embed_dim": int(q_feat.shape[1]) if (isinstance(q_feat, np.ndarray) and q_feat.ndim == 2) else None,
            "q_embeddings_npy": str(q_npy.name),
            "db_embeddings_npy": str(db_npy.name),
            "q_paths_txt": f"{name}_q_paths.txt",
            "db_paths_txt": f"{name}_db_paths.txt",
        }
        (out_dir / f"{name}_embeddings_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] wrote embeddings: {q_npy} {db_npy}")

    # 類似度を計算する
    print("[INFO] computing similarity + topk...")
    sim = q_feat @ db_feat.T  # cosine because L2-normalized

    top_idx, top_sim = _topk_from_sim(sim, max_k)
    top1_idx = top_idx[:, 0].astype(np.int64, copy=False)
    top1_sim = top_sim[:, 0].astype(np.float32, copy=False)


    # Strict の順位（1 始まり）
    gt_scores = sim[np.arange(n_q), gt_idx]
    # rank = gt_score 以上の score を持つ DB 件数
    strict_rank = (sim >= gt_scores[:, None]).sum(axis=1).astype(np.int64)

    strict_metrics = {
        "Q": int(n_q),
        "DB": int(n_db),
        "R@": {str(k): _recall_at_k(strict_rank, k) for k in strict_ks},
        "MRR": _mrr(strict_rank),
        "MedianRank": _median_rank(strict_rank),
    }
    if include_r1pct:
        strict_metrics["R@1%"] = _recall_at_k(strict_rank, k_1pct)
        strict_metrics["K@1%"] = int(k_1pct)

    # -------------------------------------------------------------------------
    # 共通メタ情報（--icp-csv を省略しても必ず持たせる）
    # -------------------------------------------------------------------------
    q_pair_ids = df_pairs["pair_id"].astype(str).tolist()
    db_pair_ids = df_db["pair_id"].astype(str).tolist()
    # Top-1 が同じ segpair (pair_id) に属するか (セグメントペア内の取り違えか、別ペアに飛んだか)
    top1_same_pair = (np.asarray(db_pair_ids, dtype=object)[top1_idx] == np.asarray(q_pair_ids, dtype=object))

    # Q フレーム上の query xy（cam.json 由来）。取れるときだけ埋める。
    q_xy = np.full((n_q, 2), np.nan, dtype=np.float64)
    q_has_pose = np.zeros((n_q,), dtype=bool)
    # pair ごとに Q フレームへ変換した DB アンカー群（icp_csv がある場合のみ埋める）
    db_by_pair: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    # soft GT 集合（threshold -> n_q 個の配列リスト）
    gt_sets: Dict[float, List[np.ndarray]] = {}

    # Soft 指標
    soft_metrics = {}
    if args.icp_csv:
        icp_map = _load_icp_T_map(Path(_normalize_wsl_path(args.icp_csv)))

        # pair ごとの DB アンカー位置を Q フレームで準備する
        # db_by_pair: pair_id -> (global_db_idx (N,), anchor_xy_q (N,2)) の対応
        # df_db には pair_id と anchor_xyz_db が入っている前提で扱う
        for pair_id, g in df_db.groupby("pair_id"):
            pid = str(pair_id)
            T = icp_map.get(pid)
            if T is None:
                # ある pair に ICP が無ければ、その pair の soft GT は計算できない。
                continue
            if invert_icp:
                T = np.linalg.inv(T)

            idxs = g.index.to_numpy(dtype=np.int64)
            xyz_db = g[["anchor_x_db", "anchor_y_db", "anchor_z_db"]].to_numpy(dtype=np.float64)
            # 同次座標化
            ones = np.ones((xyz_db.shape[0], 1), dtype=np.float64)
            pts = np.concatenate([xyz_db, ones], axis=1)  # (N,4)
            pts_q = (T @ pts.T).T  # (N,4)
            xy_q = pts_q[:, :2].astype(np.float64)
            db_by_pair[pid] = (idxs, xy_q)

        # query xy を先読みする
        miss_pose = 0
        for i, p in enumerate(q_paths):
            xy = _read_q_xy_from_cam_json(p)
            if xy is None:
                q_has_pose[i] = False
                miss_pose += 1
                continue
            q_has_pose[i] = True
            q_xy[i, 0] = xy[0]
            q_xy[i, 1] = xy[1]

        if miss_pose:
            print(f"[WARN] missing q pose json for {miss_pose}/{n_q} queries -> excluded from soft")

        # 各 t に対する GT 集合を構築する
        # gt_sets[t][qi] = global DB index の np.ndarray
        gt_sets = {t: [np.empty((0,), dtype=np.int64) for _ in range(n_q)] for t in soft_ts}
        empty_counts: Dict[float, int] = {t: 0 for t in soft_ts}

        for qi in range(n_q):
            if not q_has_pose[qi]:
                for t in soft_ts:
                    gt_sets[t][qi] = np.empty((0,), dtype=np.int64)
                    empty_counts[t] += 1
                continue

            pid = q_pair_ids[qi]
            if pid not in db_by_pair:
                for t in soft_ts:
                    gt_sets[t][qi] = np.empty((0,), dtype=np.int64)
                    empty_counts[t] += 1
                continue

            db_idxs, db_xy = db_by_pair[pid]
            dx = db_xy[:, 0] - q_xy[qi, 0]
            dy = db_xy[:, 1] - q_xy[qi, 1]
            d = np.sqrt(dx * dx + dy * dy)

            for t in soft_ts:
                mask = d <= float(t)
                sel = db_idxs[mask]
                gt_sets[t][qi] = sel
                if sel.size == 0:
                    empty_counts[t] += 1

        # Soft recall を計算する
        soft_R = {}
        for t in soft_ts:
            # この t で有効な query は GT が空でないものだけ
            valid = [i for i in range(n_q) if gt_sets[t][i].size > 0]
            if not valid:
                soft_R[str(t)] = {"valid": 0}
                continue

            # 各 K ごとに処理する
            rec = {}
            for k in soft_ks:
                hit = 0
                for qi in valid:
                    topk = top_idx[qi, :k]
                    gt = gt_sets[t][qi]
                    # 交差があるかを調べる
                    if np.intersect1d(topk, gt, assume_unique=False).size > 0:
                        hit += 1
                rec[str(k)] = hit / len(valid)

            # Recall@1% を計算する
            if include_r1pct:
                hit = 0
                for qi in valid:
                    topk = top_idx[qi, :k_1pct]
                    gt = gt_sets[t][qi]
                    if np.intersect1d(topk, gt, assume_unique=False).size > 0:
                        hit += 1
                rec["1%"] = hit / len(valid)

            soft_R[str(t)] = {
                "valid": len(valid),
                "empty_gt": int(empty_counts[t]),
                "R@": rec,
            }

        soft_metrics["R_soft"] = soft_R

        # 主しきい値に対する Soft MRR / MedianRank
        t_main = float(args.soft_rank_t_main_m)
        if t_main in gt_sets:
            ranks = []
            for qi in range(n_q):
                gt = gt_sets[t_main][qi]
                if gt.size == 0:
                    continue
                # 集合内で最良の GT score
                best = float(np.max(sim[qi, gt]))
                r = int(np.sum(sim[qi] >= best))
                ranks.append(r)
            if ranks:
                ranks_arr = np.array(ranks, dtype=np.int64)
                soft_metrics["SoftMRR_t2m"] = _mrr(ranks_arr)
                soft_metrics["SoftMedianRank_t2m"] = _median_rank(ranks_arr)
                soft_metrics["SoftValid_t2m"] = int(len(ranks))
            else:
                soft_metrics["SoftMRR_t2m"] = float("nan")
                soft_metrics["SoftMedianRank_t2m"] = float("nan")
                soft_metrics["SoftValid_t2m"] = 0

    metrics = {
        "name": args.name,
        "strict": strict_metrics,
        "soft": soft_metrics,
    }

    out_json = out_dir / f"{args.name}_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # -------------------------------------------------------------------------
    # 解析用の query 単位成果物（推奨）
    # -------------------------------------------------------------------------
    # 注意:
    # - per_query.csv は解析コード側の一次情報として使う想定。
    # - topk.jsonl は eval を再実行せずに詳細確認や gallery 生成を行うためのもの。

    strict_ks_sorted = sorted({int(k) for k in strict_ks})
    soft_ks_sorted = sorted({int(k) for k in soft_ks})
    soft_ts_sorted = sorted({float(t) for t in soft_ts})

    # query 単位の診断量を一度だけ前計算する（per_query.csv と topk.jsonl の両方で使う）。
    # strict hit フラグ
    strict_hit_at_k: Dict[int, np.ndarray] = {
        k: (strict_rank <= k) for k in strict_ks_sorted
    }
    strict_hit_1pct: Optional[np.ndarray] = None
    if include_r1pct:
        strict_hit_1pct = (strict_rank <= k_1pct)

    # 距離（query frame 基準。計算にはその pair の pose と ICP が必要）
    # 重要:
    #   q_xy は query segment の座標系にある。
    #   anchor_xy_db は各 DB segment の座標系にある。
    #   したがって、距離は pair ごとの ICP 変換を適用した後でのみ意味を持つ。
    gt_dist_m = np.full(n_q, np.nan, dtype=np.float32)
    top1_dist_m = np.full(n_q, np.nan, dtype=np.float32)
    if q_has_pose.any() and len(db_by_pair) > 0:
        _cache_idx2xy: Dict[str, Dict[int, Tuple[float, float]]] = {}

        def _get_idx2xy_qframe(pair_id: str) -> Optional[Dict[int, Tuple[float, float]]]:
            """query フレーム索引から xy 座標への対応表を構築する。"""
            if pair_id in _cache_idx2xy:
                return _cache_idx2xy[pair_id]
            if pair_id not in db_by_pair:
                return None
            db_idxs, db_xy = db_by_pair[pair_id]
            m = {int(ii): (float(xy[0]), float(xy[1])) for ii, xy in zip(db_idxs, db_xy)}
            _cache_idx2xy[pair_id] = m
            return m

        for qi in range(n_q):
            if not bool(q_has_pose[qi]):
                continue
            pid = str(q_pair_ids[qi])
            idx2xy = _get_idx2xy_qframe(pid)
            if idx2xy is None:
                continue

            qx, qy = float(q_xy[qi, 0]), float(q_xy[qi, 1])
            gt_global = int(gt_idx[qi])
            if gt_global in idx2xy:
                ax, ay = idx2xy[gt_global]
                gt_dist_m[qi] = float(np.hypot(ax - qx, ay - qy))

            # top1 距離は、取得された top1 が同じ pair に属するときだけ計算する
            if bool(top1_same_pair[qi]):
                top1_global = int(top1_idx[qi])
                if top1_global in idx2xy:
                    ax, ay = idx2xy[top1_global]
                    top1_dist_m[qi] = float(np.hypot(ax - qx, ay - qy))

    # query ごとの soft 診断量（しきい値ごと）
    # soft_diag[t] = 配列辞書
    soft_diag: Dict[float, Dict[str, Any]] = {}
    if soft_ts_sorted:
        for t in soft_ts_sorted:
            gt_list = gt_sets.get(t, [])
            # 安全策: 通常は必ず存在するが、堅牢性のため防御しておく。
            if not gt_list or len(gt_list) != n_q:
                gt_list = [np.asarray([], dtype=np.int64) for _ in range(n_q)]

            gt_size = np.asarray([int(len(g)) for g in gt_list], dtype=np.int32)
            best_score = np.full(n_q, np.nan, dtype=np.float32)
            best_rank = np.full(n_q, np.nan, dtype=np.float32)

            # hit 配列
            hit_at_k: Dict[int, np.ndarray] = {
                k: np.zeros(n_q, dtype=bool) for k in soft_ks_sorted
            }
            hit_1pct = np.zeros(n_q, dtype=bool) if include_r1pct else None
            first_hit_rank = np.full(n_q, np.nan, dtype=np.float32)

            # top_idx に対する membership（max_k までは計算済み）
            for qi in range(n_q):
                gt = gt_list[qi]
                if gt.size == 0:
                    continue

                # 最良 GT score とその global rank（1 始まり）
                bs = float(np.max(sim[qi, gt]))
                best_score[qi] = bs
                best_rank[qi] = float(int(np.sum(sim[qi] >= bs)))

                # top-k への所属判定
                is_gt = np.isin(top_idx[qi], gt)
                if np.any(is_gt):
                    first_hit_rank[qi] = float(int(np.argmax(is_gt)) + 1)

                for k in soft_ks_sorted:
                    kk = min(int(k), is_gt.shape[0])
                    hit_at_k[k][qi] = bool(np.any(is_gt[:kk]))

                if include_r1pct and hit_1pct is not None:
                    kk = min(int(k_1pct), is_gt.shape[0])
                    hit_1pct[qi] = bool(np.any(is_gt[:kk]))

            soft_diag[t] = {
                "gt_size": gt_size,
                "best_score": best_score,
                "best_rank": best_rank,
                "first_hit_rank": first_hit_rank,
                "hit_at_k": hit_at_k,
                "hit_1pct": hit_1pct,
            }

    # 任意: query 単位テーブル（CSV/JSONL）
    if args.write_per_query_csv or getattr(args, "write_per_query_jsonl", False):
        rows: List[Dict[str, Any]] = []
        for qi in range(n_q):
            pid = str(df_pairs.iloc[qi]["pair_id"])
            q_cam = str(q_paths[qi])
            gt_png = str(db_paths[int(gt_idx[qi])])

            row: Dict[str, Any] = {
                "q_idx": int(qi),
                "pair_id": pid,
                "q_cam_png": q_cam,
                "gt_anchor_png": gt_png,
                "gt_index_db": int(gt_idx[qi]),
                "gt_score": float(gt_scores[qi]),
                "gt_dist_m": (None if not np.isfinite(gt_dist_m[qi]) else float(gt_dist_m[qi])),
                "strict_rank": int(strict_rank[qi]),
                "top1_index_db": int(top1_idx[qi]),
                "top1_score": float(top1_sim[qi]),
                "top1_anchor_png": str(db_paths[int(top1_idx[qi])]),
                "top1_pair_id": str(df_db.iloc[int(top1_idx[qi])]["pair_id"]),
                "top1_is_same_pair": bool(top1_same_pair[qi]),
                "top1_dist_m": (None if not np.isfinite(top1_dist_m[qi]) else float(top1_dist_m[qi])),
            }

            if q_has_pose[qi]:
                row["q_x_m"] = float(q_xy[qi, 0])
                row["q_y_m"] = float(q_xy[qi, 1])
            else:
                row["q_x_m"] = None
                row["q_y_m"] = None

            # strict hit フラグ
            for k in strict_ks_sorted:
                row[f"strict_hit@{k}"] = bool(strict_hit_at_k[k][qi])
            if include_r1pct and strict_hit_1pct is not None:
                row["strict_hit@1pct"] = bool(strict_hit_1pct[qi])
                row["strict_k@1pct"] = int(k_1pct)

            # soft 診断量
            for t in soft_ts_sorted:
                tag = f"t{_fmt_m(t)}m"  # e.g., t2m, t5m, t2p5m
                sd = soft_diag.get(t, None)
                if sd is None:
                    continue

                row[f"soft_{tag}_gt_size"] = int(sd["gt_size"][qi])
                row[f"soft_{tag}_best_rank"] = (None if not np.isfinite(sd["best_rank"][qi]) else int(sd["best_rank"][qi]))
                row[f"soft_{tag}_best_score"] = (None if not np.isfinite(sd["best_score"][qi]) else float(sd["best_score"][qi]))
                row[f"soft_{tag}_first_hit_rank"] = (None if not np.isfinite(sd["first_hit_rank"][qi]) else int(sd["first_hit_rank"][qi]))

                hit_at_k = sd["hit_at_k"]
                for k in soft_ks_sorted:
                    row[f"soft_{tag}_hit@{k}"] = bool(hit_at_k[k][qi])

                if include_r1pct and sd.get("hit_1pct", None) is not None:
                    row[f"soft_{tag}_hit@1pct"] = bool(sd["hit_1pct"][qi])
                    row[f"soft_{tag}_k@1pct"] = int(k_1pct)

            rows.append(row)

        if getattr(args, "write_per_query_jsonl", False) and args.write_per_query_jsonl:
            out_path = out_dir / f"{name}_per_query.jsonl"
            _write_jsonl(out_path, rows)
            print(f"[OK] wrote per-query JSONL: {out_path} rows={len(rows)}")

        if args.write_per_query_csv:
            df_perq = pd.DataFrame(rows)
            out_path = out_dir / f"{name}_per_query.csv"
            df_perq.to_csv(out_path, index=False)
            print(f"[OK] wrote per-query CSV: {out_path} rows={len(df_perq)}")

    # 任意: top-k の JSONL dump（eval を再実行せずに詳細 / 可視 debugging を行うため）
    if getattr(args, "write_topk_jsonl", False) and args.write_topk_jsonl:
        dump_k = int(getattr(args, "topk_dump_k", 0) or 0)
        if dump_k <= 0:
            print("[INFO] write_topk_jsonl=true but topk_dump_k<=0 -> skip topk dump")
        else:
            dump_k = min(dump_k, int(top_idx.shape[1]))
            out_path = out_dir / f"{name}_topk_top{dump_k}.jsonl"

            db_pair_ids = df_db["pair_id"].astype(str).tolist()

            with open(out_path, "w", encoding="utf-8") as f:
                for qi in range(n_q):
                    pid = str(df_pairs.iloc[qi]["pair_id"])

                    rec: Dict[str, Any] = {
                        "q_idx": int(qi),
                        "pair_id": pid,
                        "q_cam_png": str(q_paths[qi]),
                        "q_xy_m": (
                            [float(q_xy[qi, 0]), float(q_xy[qi, 1])]
                            if q_has_pose[qi] else None
                        ),
                        "strict_gt": {
                            "db_index": int(gt_idx[qi]),
                            "anchor_png": str(db_paths[int(gt_idx[qi])]),
                            "score": float(gt_scores[qi]),
                            "strict_rank": int(strict_rank[qi]),
                            "dist_m": (None if not np.isfinite(gt_dist_m[qi]) else float(gt_dist_m[qi])),
                        },
                        "topk": [],
                        "soft": {},
                    }

                    # この query に対する soft 診断サマリ（全 GT index は列挙しない）
                    for t in soft_ts_sorted:
                        tag = f"t{_fmt_m(t)}m"
                        sd = soft_diag.get(t, None)
                        if sd is None:
                            continue
                        rec["soft"][tag] = {
                            "gt_size": int(sd["gt_size"][qi]),
                            "best_rank": (None if not np.isfinite(sd["best_rank"][qi]) else int(sd["best_rank"][qi])),
                            "best_score": (None if not np.isfinite(sd["best_score"][qi]) else float(sd["best_score"][qi])),
                            "first_hit_rank": (None if not np.isfinite(sd["first_hit_rank"][qi]) else int(sd["first_hit_rank"][qi])),
                            "hit_at_k": {str(k): bool(sd["hit_at_k"][k][qi]) for k in soft_ks_sorted},
                            "hit_1pct": (None if (not include_r1pct or sd.get("hit_1pct", None) is None) else bool(sd["hit_1pct"][qi])),
                            "k_1pct": (None if not include_r1pct else int(k_1pct)),
                        }

                    # エントリ単位の dump
                    # GT membership 判定用に、この query ではしきい値ごとの小さな python set を作る（ここで扱う dump サイズなら十分速い）。
                    gt_sets_py: Dict[float, set] = {}
                    for t in soft_ts_sorted:
                        gt = gt_sets.get(t, [np.asarray([], dtype=np.int64) for _ in range(n_q)])[qi]
                        gt_sets_py[t] = set(int(x) for x in gt.tolist())

                    # 距離は次の場合に限って *query frame* 上で計算できる。
                    # (1) query pose があり、(2) その pair の ICP（db_by_pair）がある場合。
                    idx2xy_q: Optional[Dict[int, Tuple[float, float]]] = None
                    qx = float(q_xy[qi, 0]) if bool(q_has_pose[qi]) else float("nan")
                    qy = float(q_xy[qi, 1]) if bool(q_has_pose[qi]) else float("nan")
                    if bool(q_has_pose[qi]) and pid in db_by_pair:
                        db_idxs_q, db_xy_q = db_by_pair[pid]
                        idx2xy_q = {int(ii): (float(xy[0]), float(xy[1])) for ii, xy in zip(db_idxs_q, db_xy_q)}

                    for r in range(dump_k):
                        dbi = int(top_idx[qi, r])
                        score = float(top_sim[qi, r])
                        entry: Dict[str, Any] = {
                            "rank": int(r + 1),
                            "db_index": int(dbi),
                            "score": score,
                            "anchor_png": str(db_paths[dbi]),
                            "pair_id": db_pair_ids[dbi],
                            "is_same_pair": bool(db_pair_ids[dbi] == pid),
                            "is_strict_gt": bool(dbi == int(gt_idx[qi])),
                            "dist_m": None,
                            "soft_gt": {},
                        }

                        if idx2xy_q is not None and dbi in idx2xy_q:
                            ax, ay = idx2xy_q[dbi]
                            entry["dist_m"] = float(np.hypot(ax - qx, ay - qy))

                        for t in soft_ts_sorted:
                            tag = f"t{_fmt_m(t)}m"
                            entry["soft_gt"][tag] = bool(dbi in gt_sets_py[t])

                        rec["topk"].append(entry)

                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(f"[OK] wrote topK JSONL: {out_path} rows={n_q}")

    # -------------------------------------------------------------------------
    # failure 解析用 dump（後で確認 / フィルタするため）
    # -------------------------------------------------------------------------
    if args.write_failures:
        fail_k = max(1, int(args.failure_k))
        topn = max(1, int(args.failure_topn))
        max_cases = int(args.failure_max_cases)

        # failure は STRICT GT が top-K に入っていない状態と定義する。
        fail_mask = (strict_rank > fail_k)
        fail_indices_all = np.where(fail_mask)[0].tolist()

        # --- *すべて* の failure 用に軽量 CSV を書く（top-N 一覧は持たない） ---
        fail_rows_all: List[dict] = []
        for qi in fail_indices_all:
            pid = str(df_pairs.loc[qi, "pair_id"])
            row = {
                "q_idx": int(qi),
                "pair_id": pid,
                "q_cam_png": str(q_paths[qi]),
                "gt_anchor_png": str(db_paths[int(gt_idx[qi])]),
                "strict_rank": int(strict_rank[qi]),
            }
            # 任意メタデータをそのまま流す（存在する場合）
            for col in ["q_seg", "db_seg", "q_daypart", "db_daypart", "q_weather", "db_weather"]:
                if col in df_pairs.columns:
                    row[col] = df_pairs.loc[qi, col]
            # pose / distance（取れる場合のみ）
            if bool(q_has_pose[qi]):
                qx, qy = float(q_xy[qi, 0]), float(q_xy[qi, 1])
                row["q_x"] = qx
                row["q_y"] = qy
                gt_dist = None
                if pid in db_by_pair:
                    db_idxs, db_xy = db_by_pair[pid]
                    try:
                        # この pair 内で GT anchor の位置を求める
                        gt_global = int(gt_idx[qi])
                        # idx->xy の辞書を作る（pair サイズは小さいので解析用途なら十分軽い）
                        idx2xy = {int(ii): (float(db_xy[k, 0]), float(db_xy[k, 1])) for k, ii in enumerate(db_idxs)}
                        if gt_global in idx2xy:
                            ax, ay = idx2xy[gt_global]
                            gt_dist = float(np.hypot(ax - qx, ay - qy))
                    except Exception:
                        gt_dist = None
                row["gt_dist_m"] = gt_dist
            else:
                row["q_x"] = None
                row["q_y"] = None
                row["gt_dist_m"] = None
            fail_rows_all.append(row)

        fail_all_csv = out_dir / f"{args.name}_fail_strictR@{fail_k}_all.csv"
        pd.DataFrame(fail_rows_all).to_csv(fail_all_csv, index=False)
        print(f"[OK] wrote all-fail CSV: {fail_all_csv} rows={len(fail_rows_all)}")

        # --- 追加の集約解析（pair confusion, pair ごとの failure rate など） ---
        # 注意: This uses the *full* failure set (not truncated by --max-cases).
        try:
            if top_idx.shape[1] >= 1:
                top1_global_all = top_idx[:, 0].astype(np.int64)
                top1_pair_all = df_db.iloc[top1_global_all]["pair_id"].astype(str).to_numpy()
            else:
                top1_global_all = np.full((n_q,), -1, dtype=np.int64)
                top1_pair_all = np.asarray([""] * n_q, dtype=object)

            q_pair_all = np.asarray(q_pair_ids, dtype=object)
            fail_q_pair = q_pair_all[fail_mask]
            fail_top1_pair = top1_pair_all[fail_mask]

            # Pair confusion（failure のみ）: query-pair ごとの retrieved top1 pair 分布。
            if args.write_failure_pair_confusion and fail_q_pair.size > 0:
                df_conf = pd.DataFrame({
                    "q_pair_id": fail_q_pair,
                    "top1_pair_id": fail_top1_pair,
                })
                conf = df_conf.value_counts().reset_index(name="count")
                conf["frac_within_q_pair"] = conf["count"] / conf.groupby("q_pair_id")["count"].transform("sum")
                conf = conf.sort_values(["q_pair_id", "count"], ascending=[True, False])
                conf_csv = out_dir / f"{args.name}_fail_strictR@{fail_k}_top1_pair_confusion.csv"
                conf.to_csv(conf_csv, index=False)
                print(f"[OK] wrote failure top1 pair confusion: {conf_csv} rows={len(conf)}")

            # pair_id ごとの failure rate（failure のみ、strict 定義）
            if args.write_failure_by_pair:
                df_pf = pd.DataFrame({
                    "pair_id": q_pair_all,
                    "is_fail": fail_mask.astype(np.int64),
                })
                df_pf = df_pf.groupby("pair_id").agg(
                    num_queries=("is_fail", "size"),
                    num_failures=("is_fail", "sum"),
                ).reset_index()
                df_pf["failure_rate"] = df_pf["num_failures"] / df_pf["num_queries"].clip(lower=1)
                df_pf = df_pf.sort_values(["failure_rate", "num_failures"], ascending=[False, False])
                pf_csv = out_dir / f"{args.name}_fail_strictR@{fail_k}_by_pair.csv"
                df_pf.to_csv(pf_csv, index=False)
                print(f"[OK] wrote failure-by-pair: {pf_csv} rows={len(df_pf)}")

            # 集約した failure summary JSON（rank 統計、same-pair top1 率、任意の soft 統計）。
            if args.write_failure_summary:
                fail_ranks = strict_rank[fail_mask].astype(np.int64)

                # failure 集合における same-pair top1
                same_pair_top1 = int(np.sum(fail_top1_pair == fail_q_pair)) if fail_q_pair.size else 0
                same_pair_rate = float(same_pair_top1 / max(int(fail_q_pair.size), 1))

                # 距離統計（pose が取れる場合のみ、かつ same pair 内でのみ意味を持つ）
                fail_gt_dist = [r.get("gt_dist_m") for r in fail_rows_all]
                # 可能な failure について top1 距離を計算する
                fail_top1_dist: List[Optional[float]] = []
                if fail_q_pair.size and np.any(q_has_pose[fail_mask]):
                    # 高速化のため pair ごとの idx->xy をキャッシュする
                    _cache_idx2xy: Dict[str, Dict[int, Tuple[float, float]]] = {}

                    def _get_idx2xy(pair_id: str) -> Optional[Dict[int, Tuple[float, float]]]:
                        """索引から xy 座標への対応表を構築する。"""
                        if pair_id in _cache_idx2xy:
                            return _cache_idx2xy[pair_id]
                        if pair_id not in db_by_pair:
                            return None
                        db_idxs, db_xy = db_by_pair[pair_id]
                        m = {int(ii): (float(xy[0]), float(xy[1])) for ii, xy in zip(db_idxs, db_xy)}
                        _cache_idx2xy[pair_id] = m
                        return m

                    for qi in fail_indices_all:
                        if not bool(q_has_pose[qi]):
                            fail_top1_dist.append(None)
                            continue
                        pid = str(q_pair_ids[qi])
                        if top_idx.shape[1] < 1:
                            fail_top1_dist.append(None)
                            continue
                        top1_i = int(top_idx[qi, 0])
                        top1_pid = str(df_db.iloc[top1_i]["pair_id"])
                        if top1_pid != pid:
                            fail_top1_dist.append(None)
                            continue
                        idx2xy = _get_idx2xy(pid)
                        if idx2xy is None or top1_i not in idx2xy:
                            fail_top1_dist.append(None)
                            continue
                        qx, qy = float(q_xy[qi, 0]), float(q_xy[qi, 1])
                        ax, ay = idx2xy[top1_i]
                        fail_top1_dist.append(float(np.hypot(ax - qx, ay - qy)))

                summary: dict = {
                    "name": args.name,
                    "fail_k": int(fail_k),
                    "num_queries": int(n_q),
                    "num_failures": int(len(fail_indices_all)),
                    "failure_rate": float(len(fail_indices_all) / max(int(n_q), 1)),
                    "fail_rank": _describe_floats([float(r) for r in fail_ranks.tolist()], prefix="rank_"),
                    "fail_top1_same_pair_count": int(same_pair_top1),
                    "fail_top1_same_pair_rate": float(same_pair_rate),
                    "fail_gt_dist_m": _describe_floats(fail_gt_dist, prefix="gt_dist_m_"),
                    "fail_top1_dist_m_same_pair": _describe_floats(fail_top1_dist, prefix="top1_dist_m_"),
                }

                # Soft-on-fail: STRICT failure のうち、soft 指標では実は「近い」ものがどれだけあるか。
                # （ICP CSV が必要。）
                if gt_sets:
                    soft_on_fail: Dict[str, dict] = {}
                    for t in soft_ts:
                        key = f"t{t:.3g}"
                        hits: Dict[str, float] = {}
                        # この t で GT 集合が空でない failure だけを対象にする
                        valid_fail = [qi for qi in fail_indices_all if gt_sets.get(t) is not None and gt_sets[t][qi].size > 0]
                        denom = max(len(valid_fail), 1)
                        for k in sorted(set(list(soft_ks) + [fail_k])):
                            k_eff = int(min(k, top_idx.shape[1]))
                            if k_eff <= 0:
                                continue
                            h = 0
                            for qi in valid_fail:
                                gtset = set(map(int, gt_sets[t][qi].tolist()))
                                topk = set(map(int, top_idx[qi, :k_eff].tolist()))
                                if gtset.intersection(topk):
                                    h += 1
                            hits[f"soft_hit@{k_eff}"] = float(h / denom)
                        soft_on_fail[key] = {
                            "valid_fail": int(len(valid_fail)),
                            "hits": hits,
                        }
                    summary["soft_on_fail"] = soft_on_fail

                summary_json = out_dir / f"{args.name}_fail_strictR@{fail_k}_summary.json"
                summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
                print(f"[OK] wrote failure summary json: {summary_json}")
        except Exception as e:
            print(f"[WARN] failure extra analyses skipped due to error: {type(e).__name__}: {e}")

        # --- 先頭 N 件の failure について詳細 dump を出す（top-N 一覧、任意統計、任意 HTML） ---
        # strict_rank の降順で並べる（難しいものを先にする）
        fail_indices = sorted(fail_indices_all, key=lambda i: int(strict_rank[i]), reverse=True)
        if max_cases > 0 and len(fail_indices) > max_cases:
            fail_indices = fail_indices[:max_cases]

        failures_detail: List[dict] = []
        for qi in fail_indices:
            pid = str(df_pairs.loc[qi, "pair_id"])
            has_pose = bool(q_has_pose[qi])
            qx = float(q_xy[qi, 0]) if has_pose else float("nan")
            qy = float(q_xy[qi, 1]) if has_pose else float("nan")

            # この pair 用の idx->xy を構築する（各 record で 1 回だけ）。
            idx2xy = None
            if has_pose and pid in db_by_pair:
                db_idxs, db_xy = db_by_pair[pid]
                idx2xy = {int(ii): (float(db_xy[k, 0]), float(db_xy[k, 1])) for k, ii in enumerate(db_idxs)}

            gt_global = int(gt_idx[qi])
            gt_png = str(db_paths[gt_global])
            gt_dist = None
            if idx2xy is not None and gt_global in idx2xy:
                ax, ay = idx2xy[gt_global]
                gt_dist = float(np.hypot(ax - qx, ay - qy))

            rec: dict = {
                "q_idx": int(qi),
                "pair_id": pid,
                "q_cam_png": str(q_paths[qi]),
                "gt_anchor_png": gt_png,
                "strict_rank": int(strict_rank[qi]),
                "q_xy": [float(qx), float(qy)] if has_pose else None,
                "gt_dist_m": gt_dist,
            }

            # score 群（margin や ambiguity の診断に有用）
            gt_score = float(gt_scores[qi])
            top1_global = int(top_idx[qi, 0]) if top_idx.shape[1] >= 1 else -1
            top1_score = float(sim[qi, top1_global]) if top1_global >= 0 else float("nan")
            top1_pid = str(df_db.iloc[top1_global]["pair_id"]) if top1_global >= 0 else ""
            top1_png = str(db_paths[top1_global]) if top1_global >= 0 else ""
            top1_dist = None
            if idx2xy is not None and top1_pid == pid and top1_global in idx2xy:
                ax, ay = idx2xy[top1_global]
                top1_dist = float(np.hypot(ax - qx, ay - qy))

            rec["gt_score"] = gt_score
            rec["top1"] = {
                "db_idx": int(top1_global),
                "anchor_png": top1_png,
                "pair_id": top1_pid,
                "score": top1_score,
                "same_pair": bool(top1_pid == pid),
                "dist_m": top1_dist,
            }
            rec["score_margin_top1_minus_gt"] = float(top1_score - gt_score)

            # soft 診断量 for this failure (only if ICP CSV is provided)
            if gt_sets:
                soft_diag: Dict[str, dict] = {}
                for t in soft_ts:
                    g = gt_sets[t][qi]
                    if g.size == 0:
                        soft_diag[f"t{t:.3g}"] = {"gt_size": 0, "hit@K": False, "best_rank": None}
                        continue
                    k_eff = int(min(fail_k, top_idx.shape[1]))
                    gtset = set(map(int, g.tolist()))
                    topk = set(map(int, top_idx[qi, :k_eff].tolist()))
                    hit = bool(gtset.intersection(topk))
                    # soft GT 集合内での最良順位
                    best_score = float(np.max(sim[qi, g]))
                    best_rank = int(np.sum(sim[qi] >= best_score))
                    # 最良距離（same-pair 座標が取れる場合）
                    best_dist = None
                    if idx2xy is not None:
                        best_local = int(g[int(np.argmax(sim[qi, g]))])
                        if best_local in idx2xy:
                            ax, ay = idx2xy[best_local]
                            best_dist = float(np.hypot(ax - qx, ay - qy))
                    soft_diag[f"t{t:.3g}"] = {
                        "gt_size": int(g.size),
                        f"hit@{k_eff}": hit,
                        "best_rank": best_rank,
                        "best_dist_m": best_dist,
                    }
                rec["soft_diag"] = soft_diag
            for col in ["q_seg", "db_seg", "q_daypart", "db_daypart", "q_weather", "db_weather"]:
                if col in df_pairs.columns:
                    rec[col] = df_pairs.loc[qi, col]

            if args.failure_compute_image_stats:
                rec["q_stats"] = _img_stats_gray(str(q_paths[qi]))
                rec["gt_stats"] = _img_stats_gray(gt_png)

            top_list: List[dict] = []
            topn_eff = min(topn, int(top_idx.shape[1]))
            soft_main_set = None
            # tag / diagnostics には「主」soft 半径しきい値を使う（既定: --soft_rank_t_main_m）。
            # 注意: gt_sets keys are floats (meters), so we use float(args.soft_rank_t_main_m).
            _t_main = float(args.soft_rank_t_main_m)
            if gt_sets and _t_main in gt_sets:
                g_main = gt_sets[_t_main][qi]
                if g_main.size > 0:
                    soft_main_set = set(map(int, g_main.tolist()))
            for r in range(topn_eff):
                di = int(top_idx[qi, r])
                a_png = str(db_paths[di])
                a_pid = str(df_db.iloc[di]["pair_id"])
                score = float(sim[qi, di])
                dist = None
                if idx2xy is not None and a_pid == pid and di in idx2xy:
                    ax, ay = idx2xy[di]
                    dist = float(np.hypot(ax - qx, ay - qy))
                item = {
                    "rank": int(r + 1),
                    "db_idx": int(di),
                    "pair_id": a_pid,
                    "anchor_png": a_png,
                    "score": score,
                    "score_minus_gt": float(score - gt_score),
                    "dist_m": dist,
                    "is_gt": bool(di == gt_global),
                    "is_soft_gt_main": (bool(di in soft_main_set) if soft_main_set is not None else None),
                }
                if args.failure_compute_image_stats:
                    item["stats"] = _img_stats_gray(a_png)
                top_list.append(item)

            rec["top"] = top_list
            failures_detail.append(rec)

        if failures_detail:
            fail_detail_jsonl = out_dir / f"{args.name}_fail_strictR@{fail_k}_detail_top{topn}.jsonl"
            _write_jsonl(fail_detail_jsonl, failures_detail)
            print(f"[OK] wrote failure detail jsonl: {fail_detail_jsonl} rows={len(failures_detail)}")

            # コンパクトな CSV（top1 のみ + top リストは JSON 文字列）
            fail_detail_csv = out_dir / f"{args.name}_fail_strictR@{fail_k}_detail_top{topn}.csv"
            flat_rows: List[dict] = []
            for rec in failures_detail:
                top1 = (rec.get("top") or [{}])[0]
                flat_rows.append({
                    "q_idx": rec.get("q_idx"),
                    "pair_id": rec.get("pair_id"),
                    "q_cam_png": rec.get("q_cam_png"),
                    "gt_anchor_png": rec.get("gt_anchor_png"),
                    "strict_rank": rec.get("strict_rank"),
                    "gt_dist_m": rec.get("gt_dist_m"),
                    "top1_anchor_png": top1.get("anchor_png", ""),
                    "top1_pair_id": top1.get("pair_id", ""),
                    "top1_score": top1.get("score", None),
                    "top1_dist_m": top1.get("dist_m", None),
                    "top_json": json.dumps(rec.get("top", []), ensure_ascii=False),
                })
            pd.DataFrame(flat_rows).to_csv(fail_detail_csv, index=False)
            print(f"[OK] wrote failure detail csv: {fail_detail_csv} rows={len(flat_rows)}")

            if args.write_failure_gallery:
                fail_html = out_dir / f"{args.name}_fail_strictR@{fail_k}_detail_top{topn}.html"
                _write_failure_gallery(fail_html, failures_detail, out_dir=out_dir, thumb_w=int(args.failure_thumb_w))
                print(f"[OK] wrote failure gallery: {fail_html}")

        # --- 任意の anchor 単位 failure summary（切り詰めない *全* failure を使う） ---
        if args.write_anchor_failure_summary:
            counts_gt = np.bincount(gt_idx.astype(np.int64), minlength=n_db).astype(np.int64)
            counts_fail_gt = np.bincount(gt_idx[fail_mask].astype(np.int64), minlength=n_db).astype(np.int64)

            top1_fail = top_idx[fail_mask, 0] if top_idx.shape[1] >= 1 else np.array([], dtype=np.int64)
            counts_top1_in_fail = np.bincount(top1_fail.astype(np.int64), minlength=n_db).astype(np.int64) if top1_fail.size else np.zeros(n_db, dtype=np.int64)

            top5 = top_idx[:, :min(5, top_idx.shape[1])]
            top5_fail = top_idx[fail_mask, :min(5, top_idx.shape[1])] if top_idx.shape[1] else top5[:0]
            counts_top5_in_fail = np.bincount(top5_fail.reshape(-1).astype(np.int64), minlength=n_db).astype(np.int64) if top5_fail.size else np.zeros(n_db, dtype=np.int64)

            # この anchor が GT のときの平均順位
            sum_rank = np.zeros(n_db, dtype=np.float64)
            np.add.at(sum_rank, gt_idx.astype(np.int64), strict_rank.astype(np.float64))
            mean_rank = np.divide(sum_rank, np.maximum(counts_gt, 1), dtype=np.float64)

            df_anchor = df_db.copy()
            df_anchor["gt_count"] = counts_gt
            df_anchor["gt_fail_count_strictR@{}".format(fail_k)] = counts_fail_gt
            df_anchor["gt_mean_rank"] = mean_rank
            df_anchor["top1_in_fail_count"] = counts_top1_in_fail
            df_anchor["top5_in_fail_count"] = counts_top5_in_fail


            # 任意: 画像品質統計（failure に現れた anchor のみ）。
            if args.failure_compute_image_stats:
                use_mask = (df_anchor["gt_fail_count_strictR@{}".format(fail_k)] > 0) | (df_anchor["top1_in_fail_count"] > 0)
                # NaN / None で事前に埋めておく
                for k in ["w", "h", "mean", "std", "nonzero_frac", "entropy", "lap_var"]:
                    df_anchor["img_" + k] = None
                idxs_use = df_anchor.index[use_mask].tolist()
                for ii in idxs_use:
                    a_path = str(df_anchor.loc[ii, "anchor_png"])
                    st = _img_stats_gray(a_path)
                    if not st:
                        continue
                    for k, v in st.items():
                        df_anchor.at[ii, "img_" + k] = v

                # カバレッジが怪しく低い RI を見るための小さな candidate list。
                try:
                    low_cov = df_anchor[(df_anchor["top1_in_fail_count"] > 0) & (df_anchor["img_nonzero_frac"].notna()) & (df_anchor["img_nonzero_frac"].astype(float) < 0.03)]
                    if len(low_cov) > 0:
                        cand_csv = out_dir / f"{args.name}_ri_low_coverage_candidates_strictR@{fail_k}.csv"
                        low_cov.sort_values(["top1_in_fail_count", "gt_fail_count_strictR@{}".format(fail_k)], ascending=False).to_csv(cand_csv, index=False)
                        print(f"[OK] wrote RI low-coverage candidates: {cand_csv} rows={len(low_cov)}")
                except Exception:
                    pass

            anchor_csv = out_dir / f"{args.name}_anchor_failure_summary_strictR@{fail_k}.csv"
            df_anchor.to_csv(anchor_csv, index=False)
            print(f"[OK] wrote anchor failure summary: {anchor_csv} rows={len(df_anchor)}")


    print("\n====================")
    print(f"[OK] wrote: {out_json}")
    print("[RESULT] Strict:")
    print(json.dumps(strict_metrics, indent=2))
    if soft_metrics:
        print("[RESULT] Soft:")
        print(json.dumps(soft_metrics, indent=2))
    print("[DONE]")


if __name__ == "__main__":
    main()
