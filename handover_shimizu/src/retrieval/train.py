#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 使い方:
#   python train.py --help
#   必要な入力パスと出力先を引数で指定して実行する。


"""
diff_train_contrastive_vlad_one_val2.py

Dual-encoder (cam/intensity) + NetVLAD (global retrieval) with InfoNCE.

本版のポイント
- 学習器・前処理は diff_train_contrastive_vlad_one.py と同等（ImageNet初期化のResNet、RGB化、枝別NetVLAD+MLP、cam/anchorで別Normalize）
- バリデーションを2系統（val1, val2）同時評価
- ログは val1/*, val2/* で分離（stdout, CSV, JSONL, TensorBoard）
- ベスト判定は --best-on {val1,val2} × --best-by {R@1,R@5,R@10,loss}
- 入力は pairs 形式（cam_png, anchor_png, label=1）
- Normalize は --modality-stats の JSON (cam/anchor の mean/std) を使用
- 既存のCLIと互換（device/patience/scheduler/amp/grad-clip 等あり）
"""

import argparse, json, random, shutil, hashlib, multiprocessing as mp
import os
from math import inf
from pathlib import Path
from functools import lru_cache
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms as T
from tqdm import tqdm


# ============ ユーティリティ ============

def set_seed(seed: int = 0):
    """乱数シードを固定して再現性を高める。"""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def str2bool(v):
    """argparse 用: いろいろな真偽表現を bool に変換"""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off", ""):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v}")

def to_rgb_tensor(path: str, tfm):
    """画像を読み込み、モデル入力用の RGB テンソルへ変換する。"""
    img = Image.open(path).convert("L").convert("RGB")  # 1ch→3ch
    return tfm(img)

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: [N,d], b: [M,d]（どちらも L2 正規化済み）
    """L2 正規化済み埋め込み同士のコサイン類似度行列を計算する。"""
    return a @ b.t()


# --- 修正：モダリティ別 mean/std ローダ（堅牢 & 互換） ---
def load_modality_stats(path: Optional[str]) -> Tuple[float,float,float,float]:
    """
    いろいろな JSON 形式を許容して cam_mean, cam_std, anc_mean, anc_std を返す。
    未指定・不足時は 0.5/0.5 にフォールバック。

    受け入れる例:
      フラット: {"cam_mean":0.42,"cam_std":0.23,"anc_mean":0.17,"anc_std":0.15}
      ネスト  : {"cam":{"mean":...,"std":...}, "anchor":{"mean":...,"std":...}}
      別名対応: "camera","cam_gray","intensity","lidar","anc","anchor_gray" など
    """
    cam_m = cam_s = anc_m = anc_s = 0.5
    if not path:
        return cam_m, cam_s, anc_m, anc_s
    try:
        d = json.load(open(path, "r"))
    except Exception as e:
        print(f"[WARN] failed to load modality stats from {path}: {e}")
        return cam_m, cam_s, anc_m, anc_s

    def pick(dct, keys, default=None):
        """候補キー列から最初に見つかった値を返す。"""
        if isinstance(dct, dict):
            for k in keys:
                if k in dct: return dct[k]
        return default

    # フラット優先
    cm = pick(d, ["cam_mean","camera_mean","cam_m","camMean"], None)
    cs = pick(d, ["cam_std","camera_std","cam_s","camStd"], None)
    am = pick(d, ["anc_mean","anchor_mean","intensity_mean","lidar_mean","anc_m"], None)
    asd= pick(d, ["anc_std","anchor_std","intensity_std","lidar_std","anc_s"], None)

    # ネスト（cam / anchor / intensity / lidar）
    if cm is None or cs is None:
        cam_d = pick(d, ["cam","camera","cam_gray","camera_gray"], None)
        if isinstance(cam_d, dict):
            cm = pick(cam_d, ["mean","m"], cm)
            cs = pick(cam_d, ["std","s"], cs)
    if am is None or asd is None:
        anc_d = pick(d, ["anchor","anc","intensity","lidar","intensity_gray"], None)
        if isinstance(anc_d, dict):
            am = pick(anc_d, ["mean","m"], am)
            asd= pick(anc_d, ["std","s"], asd)

    try:
        cam_m = float(cm) if cm is not None else cam_m
        cam_s = float(cs) if cs is not None else cam_s
        anc_m = float(am) if am is not None else anc_m
        anc_s = float(asd) if asd is not None else anc_s
    except Exception as e:
        print(f"[WARN] invalid values in {path}: {e} (fallback to 0.5)")
        cam_m = cam_s = anc_m = anc_s = 0.5
    return cam_m, cam_s, anc_m, anc_s


# ============ データセット ============

class PosPairsDataset(Dataset):
    """
    pairs.parquet から label==1 を抽出し、
    (カメラ, 正解アンカー) を1:1で返す。
    バッチ内の他アンカーが自然な負例として機能（InfoNCE）。

    追加機能（Train向け）:
      - ri_mode="epoch_cycle" の場合、parquet の anchor_png から anchor_dir（親ディレクトリ）を求め、
        anchor_dir 内の RI 画像（既定では ri_000.png ...）から "各epochで1枚" を選んで正例として返します。
      - "偏りにくい" 選び方:
          各サンプル（anchor_dir）ごとに RI の巡回順をランダムにシャッフルした順序を固定し、
          epoch に応じてその順序を循環させます。
          → 例えば 8枚あるなら、8epoch 回す間は同じRIが連続して選ばれにくく、
            各RIがほぼ均等に正例として使われます（epochが8を超えたら再巡回）。

    注意:
      - 損失関数は "対角だけ正例" の InfoNCE のままです。
        したがって「同じ cam_png を複数行に増やす」データ構造は避けてください。
        （in-batch negatives の前提が崩れます）
    """

    def __init__(
        self,
        pairs_parquet: str,
        image_size: Tuple[int, int] = (384, 640),
        jitter: float = 0.2,
        cam_mean: float = 0.5,
        cam_std: float = 0.5,
        anc_mean: float = 0.5,
        anc_std: float = 0.5,
        # --- RI 選択（trainのみ使う想定。valは固定にするのが基本）---
        ri_mode: str = "fixed",            # "fixed" | "epoch_cycle"
        ri_count: int = 8,                 # >0: ri_000..ri_{count-1} を想定。<=0: globで探索
        ri_glob: str = "ri_*.png",         # ri_count<=0 のときに使用
        ri_min_candidates: int = 1,        # 見つかった候補がこれ未満なら anchor_png にフォールバック
        ri_seed: int = 0,                  # 巡回順を決めるseed（anchor_dirごとに派生）
    ):
        """インスタンス生成時に必要な設定値と内部状態を初期化する。"""
        df = pd.read_parquet(pairs_parquet)
        if "label" not in df.columns:
            raise RuntimeError(f"'label' column not found in {pairs_parquet}")
        df = df[df["label"] == 1].copy().reset_index(drop=True)
        if not {"cam_png", "anchor_png"}.issubset(df.columns):
            raise RuntimeError("pairs に cam_png/anchor_png/label が必要です")

        # pandas row access を避ける（少し速くなる）
        self.cam_paths = df["cam_png"].astype(str).tolist()
        self.anchor_base_paths = df["anchor_png"].astype(str).tolist()
        self.anchor_dirs = [str(Path(p).parent) for p in self.anchor_base_paths]

        H, W = image_size
        # ★ モダリティ別 Normalize（元スクリプトと同等）
        self.tfm_cam = T.Compose([
            T.Resize((H, W)),
            T.ColorJitter(brightness=0.1, contrast=0.1) if jitter > 0 else T.Lambda(lambda x: x),
            T.ToTensor(),
            T.Normalize([cam_mean] * 3, [cam_std] * 3),
        ])
        self.tfm_anc = T.Compose([
            T.Resize((H, W)),
            T.ToTensor(),
            T.Normalize([anc_mean] * 3, [anc_std] * 3),
        ])

        # --- RI 選択の設定 ---
        if ri_mode not in ("fixed", "epoch_cycle"):
            raise ValueError(f"ri_mode must be 'fixed' or 'epoch_cycle', got: {ri_mode}")
        self.ri_mode = ri_mode
        self.ri_count = int(ri_count)
        self.ri_glob = str(ri_glob)
        self.ri_min_candidates = int(ri_min_candidates)
        self.ri_seed = int(ri_seed)

        # epoch を worker から見えるように共有メモリで保持する
        # （DataLoader の worker は別プロセスなので、普通の self.epoch では更新が伝播しません）
        self._epoch = mp.Value("i", 0)

        # anchor_dir -> candidates / permutation キャッシュ（各workerプロセス内でのみ有効）
        self._ri_candidates_cache: Dict[str, list] = {}
        self._ri_perm_cache: Dict[Tuple[str, int], list] = {}

    def __len__(self) -> int:
        """保持しているサンプル数を返す。"""
        return len(self.cam_paths)

    def set_epoch(self, epoch0: int) -> None:
        """0 始まりの epoch を設定する（train loop の先頭で呼ぶ）。"""
        with self._epoch.get_lock():
            self._epoch.value = int(epoch0)

    @staticmethod
    def _stable_hash32(s: str) -> int:
        """Pythonのhash()はプロセスごとに変わるので、安定ハッシュを使う"""
        h = hashlib.md5(s.encode("utf-8")).digest()
        return int.from_bytes(h[:4], "little", signed=False)

    def _get_epoch(self) -> int:
        """ファイル名や状態からエポック番号を推定する。"""
        return int(self._epoch.value)

    def _get_ri_candidates(self, anchor_dir: str) -> list:
        """anchor_dir 内の RI 候補パス（文字列）を返す"""
        if anchor_dir in self._ri_candidates_cache:
            return self._ri_candidates_cache[anchor_dir]

        pdir = Path(anchor_dir)
        cands = []

        # まずは ri_000.. の存在チェックで集める（dir listing を避けられて速いことが多い）
        if self.ri_count and self.ri_count > 0:
            for k in range(self.ri_count):
                p = pdir / f"ri_{k:03d}.png"
                if p.is_file():
                    cands.append(str(p))

        # 期待どおり見つからない場合だけ glob で探索（保険）
        if (not cands) and self.ri_glob:
            try:
                cands = [str(p) for p in sorted(pdir.glob(self.ri_glob)) if p.is_file()]
            except Exception:
                cands = []

        self._ri_candidates_cache[anchor_dir] = cands
        return cands

    def _get_perm(self, anchor_dir: str, n: int) -> list:
        """anchor_dir ごとに固定の permutation を作る（偏りにくい巡回のため）"""
        key = (anchor_dir, int(n))
        if key in self._ri_perm_cache:
            return self._ri_perm_cache[key]

        seed = (self.ri_seed + self._stable_hash32(anchor_dir)) & 0xFFFFFFFF
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n).tolist()
        self._ri_perm_cache[key] = perm
        return perm

    def _choose_anchor_path(self, base_anchor_png: str, anchor_dir: str) -> str:
        """ri_mode に従って anchor_png を選ぶ"""
        if self.ri_mode == "fixed":
            return base_anchor_png

        cands = self._get_ri_candidates(anchor_dir)
        if len(cands) < max(1, self.ri_min_candidates):
            return base_anchor_png

        perm = self._get_perm(anchor_dir, len(cands))
        e = self._get_epoch()  # 0-indexed epoch
        idx = perm[e % len(perm)]
        return cands[idx]

    def __getitem__(self, idx: int):
        """指定インデックスのサンプルを読み込んで返す。"""
        cam_path = self.cam_paths[idx]
        base_anchor = self.anchor_base_paths[idx]
        anchor_dir = self.anchor_dirs[idx]

        chosen_anchor = self._choose_anchor_path(base_anchor, anchor_dir)

        cam = to_rgb_tensor(cam_path, self.tfm_cam)

        # chosen_anchor が欠損していた場合などは base_anchor にフォールバック
        try:
            anc = to_rgb_tensor(chosen_anchor, self.tfm_anc)
        except Exception:
            chosen_anchor = base_anchor
            anc = to_rgb_tensor(chosen_anchor, self.tfm_anc)

        return cam, anc, cam_path, chosen_anchor


# ============ NetVLAD 層 ============

class NetVLAD(nn.Module):
    """
    2D特徴マップ -> NetVLAD(K×C) -> L2 normalize
    """
    def __init__(self, num_clusters=32, dim=512, alpha=1.0, normalize_input=True, intra_norm=True):
        """インスタンス生成時に必要な設定値と内部状態を初期化する。"""
        super().__init__()
        self.K = num_clusters
        self.D = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.intra_norm = intra_norm

        self.conv = nn.Conv2d(dim, self.K, kernel_size=(1,1), bias=True)
        self.centroids = nn.Parameter(torch.rand(self.K, dim))
        self._init_params()

    def _init_params(self):
        """層パラメータを初期化する。"""
        with torch.no_grad():
            self.conv.weight.copy_((2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1))
            self.conv.bias.copy_(- self.alpha * self.centroids.norm(dim=1))

    def forward(self, x):
        """順伝播を実行して必要な特徴量または損失を計算する。"""
        B, C, H, W = x.shape
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)

        soft = self.conv(x).view(B, self.K, -1)        # [B,K,N]
        soft = F.softmax(soft, dim=1)                  # assign over K
        x_flat = x.view(B, C, -1)                      # [B,C,N]

        vlad = torch.zeros(B, self.K, C, device=x.device, dtype=x.dtype)
        for k in range(self.K):
            a = soft[:, k, :].unsqueeze(1)                            # [B,1,N]
            residual = x_flat - self.centroids[k].view(1, C, 1)       # [B,C,N]
            vlad[:, k, :] = (a * residual).sum(dim=2)                 # [B,C]

        if self.intra_norm:
            vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(B, -1)                   # [B, K*C]
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad


# ============ エンコーダ（ResNet + NetVLAD + MLP） ============

class ResNetVLAD(nn.Module):
    """ResNet バックボーンと NetVLAD を組み合わせた埋め込み抽出器。"""
    def __init__(self, backbone='resnet18', out_dim=256, clusters=32, freeze_stages=2):
        """インスタンス生成時に必要な設定値と内部状態を初期化する。"""
        super().__init__()
        assert backbone in ['resnet18','resnet34','resnet50']
        if backbone=='resnet18':
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            feat_dim = 512
        elif backbone=='resnet34':
            net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            feat_dim = 512
        else:
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            feat_dim = 2048

        # stem + layer1..4（avgpool/fcは除去）
        self.conv1, self.bn1, self.relu, self.maxpool = net.conv1, net.bn1, net.relu, net.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4

        # ResNet50は 2048→512 に圧縮
        if feat_dim != 512:
            self.reduce = nn.Conv2d(feat_dim, 512, kernel_size=1, bias=False)
            feat_dim = 512
        else:
            self.reduce = None

        # ★ BASE と同一のフリーズ仕様
        self._freeze_stages_base_compatible(freeze_stages)

        self.vlad = NetVLAD(num_clusters=clusters, dim=feat_dim, alpha=1.0, normalize_input=True, intra_norm=True)
        self.proj = nn.Sequential(
            nn.Linear(clusters*feat_dim, 1024), nn.BatchNorm1d(1024), nn.GELU(),
            nn.Linear(1024, out_dim)
        )

    def _freeze_stages_base_compatible(self, k: int):
        # k>=1 で conv1, bn1 / k>=2 で +layer1 / k>=3 で +layer2 を凍結する
        """互換性を保ちながら backbone の一部ステージを凍結する。"""
        names_to_freeze = []
        if k >= 1: names_to_freeze += ["conv1", "bn1"]
        if k >= 2: names_to_freeze += ["layer1"]
        if k >= 3: names_to_freeze += ["layer2"]

        for n, p in self.named_parameters():
            if any(n.startswith(prefix) for prefix in names_to_freeze):
                p.requires_grad = False

        for m in self.modules():
            cls = m.__class__.__name__.lower()
            if "batchnorm" in cls:
                m.eval()

        if hasattr(self, "layer3"):
            self.layer3.train()
            for p in self.layer3.parameters(): p.requires_grad = True
        if hasattr(self, "layer4"):
            self.layer4.train()
            for p in self.layer4.parameters(): p.requires_grad = True

    def forward(self, x):
        """順伝播を実行して必要な特徴量または損失を計算する。"""
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        if self.reduce is not None: x = self.reduce(x)
        v = self.vlad(x)          # [B, K*C]
        e = self.proj(v)          # [B, D]
        e = F.normalize(e, p=2, dim=1)
        return e


class DualEncoder(nn.Module):
    """camera encoder と intensity encoder を別々に持つ（重み分離）。"""
    def __init__(self, backbone='resnet18', out_dim=256, clusters=32, freeze_stages=2):
        """インスタンス生成時に必要な設定値と内部状態を初期化する。"""
        super().__init__()
        self.cam = ResNetVLAD(backbone, out_dim, clusters, freeze_stages)
        self.int = ResNetVLAD(backbone, out_dim, clusters, freeze_stages)

    def forward(self, cam_img, anc_img):
        """順伝播を実行して必要な特徴量または損失を計算する。"""
        zc = self.cam(cam_img)
        za = self.int(anc_img)
        return zc, za


# ============ 損失関数 ============

class InfoNCELoss(nn.Module):
    """対照学習で使う InfoNCE 損失を計算する。"""
    def __init__(self, temperature: float = 0.07):
        """インスタンス生成時に必要な設定値と内部状態を初期化する。"""
        super().__init__()
        self.t = temperature

    def forward(self, zc, za):
        # zc, za: [B, D] で L2 正規化済み
        """順伝播を実行して必要な特徴量または損失を計算する。"""
        sim = (zc @ za.t()) / self.t  # [B,B]
        labels = torch.arange(zc.size(0), device=zc.device)
        loss1 = F.cross_entropy(sim, labels)          # cam->anchor
        loss2 = F.cross_entropy(sim.t(), labels)      # anchor->cam
        return (loss1 + loss2) * 0.5


# ============ 評価（retrieval 指標） ============

def _read_table(path: str) -> pd.DataFrame:
    """依存を増やしすぎずに csv / parquet を読む。"""
    path = str(path)
    if not path:
        raise ValueError("empty path")
    if path.endswith(".parquet"):
        try:
            return pd.read_parquet(path)
        except Exception:
            # parquet engine が使えない場合は CSV 側を参照できるようにする。
            csv_path = path[:-8] + ".csv"
            if os.path.exists(csv_path):
                return pd.read_csv(csv_path)
            raise
    return pd.read_csv(path)


def _infer_sibling_manifest(pairs_path: str, kind: str) -> Optional[str]:
    """`*_pairs.(csv|parquet)` の隣にある対応 manifest のパスを推定する。"""
    if not pairs_path:
        return None
    p = Path(pairs_path)
    stem = p.name
    if "_pairs" not in stem:
        return None
    cand = p.with_name(stem.replace("_pairs", f"_{kind}"))
    if cand.exists():
        return str(cand)
    # parquet がなくて csv がある場合、またはその逆も許容する。
    if cand.suffix == ".parquet":
        cand2 = cand.with_suffix(".csv")
        if cand2.exists():
            return str(cand2)
    if cand.suffix == ".csv":
        cand2 = cand.with_suffix(".parquet")
        if cand2.exists():
            return str(cand2)
    return None


def _se3_yaw_tx_ty_tz_to_T(yaw_deg: float, tx: float, ty: float, tz: float) -> np.ndarray:
    """yaw と並進成分から 4x4 変換行列を組み立てる。"""
    yaw = np.deg2rad(yaw_deg)
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    T[0, 3] = float(tx)
    T[1, 3] = float(ty)
    T[2, 3] = float(tz)
    return T


@lru_cache(maxsize=8)
def _load_icp_transforms(csv_path: str, invert: bool) -> Dict[str, np.ndarray]:
    """ICP 変換を csv から {pair_id: 4x4} 形式で読み込む。結果は epoch をまたいでキャッシュする。"""
    if not csv_path or (not os.path.exists(csv_path)):
        return {}
    df = pd.read_csv(csv_path)
    required = {"pair_id", "final_yaw_deg", "final_tx", "final_ty", "final_tz"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"icp_csv missing columns. required={sorted(required)} got={sorted(df.columns)}"
        )
    out: Dict[str, np.ndarray] = {}
    for _, r in df.iterrows():
        pid = str(r["pair_id"])
        T = _se3_yaw_tx_ty_tz_to_T(
            float(r["final_yaw_deg"]), float(r["final_tx"]), float(r["final_ty"]), float(r["final_tz"])
        )
        if invert:
            T = np.linalg.inv(T)
        out[pid] = T
    return out


@lru_cache(maxsize=65536)
def _load_cam_xy_from_sidecar(cam_png: str) -> Optional[Tuple[float, float]]:
    """同名 sidecar json から q フレーム姿勢の並進 (x,y) を読む。"""
    try:
        jpath = str(Path(cam_png).with_suffix(".json"))
        if not os.path.exists(jpath):
            return None
        with open(jpath, "r") as f:
            meta = json.load(f)
        pose = meta.get("frame_pose", None)
        if pose is None or len(pose) != 4:
            return None
        x = float(pose[0][3])
        y = float(pose[1][3])
        return (x, y)
    except Exception:
        return None


def _find_anchors_csv_from_anchor_png(anchor_png: str) -> Optional[str]:
    """anchor 画像パスから親ディレクトリをたどって anchors.csv を探す。"""
    p = Path(anchor_png)
    for up in range(6):
        cand = p.parents[up] / "anchors.csv"
        if cand.exists():
            return str(cand)
    return None


@lru_cache(maxsize=256)
def _load_anchors_xy(anchors_csv: str) -> pd.DataFrame:
    """anchors.csv を読み込み、必要な列だけ残す。"""
    df = pd.read_csv(anchors_csv)
    required = {"anchor_id", "x", "y"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"anchors.csv missing columns. required={sorted(required)} got={sorted(df.columns)}"
        )
    out = df[["anchor_id", "x", "y"]].copy()
    out["anchor_id"] = out["anchor_id"].astype(int)
    return out


def _fmt_t(t: float) -> str:
    """経過時間を見やすい文字列に整形する。"""
    t = float(t)
    if abs(t - round(t)) < 1e-9:
        return str(int(round(t)))
    return str(t).replace(".", "p")


def _build_soft_gt_sets(
    pairs_df: pd.DataFrame,
    db_df: pd.DataFrame,
    *,
    icp_csv: str,
    invert_icp: bool,
    thresholds_m: Tuple[float, ...],
) -> Optional[Dict[float, list]]:
    """query ごとに soft GT 集合を作る。正しい pair に属し、Q world 座標系で距離しきい値以内にある anchor を集める。

    Returns:
        {t: [set(db_anchor_indices), ...]} aligned with the query order (pairs_df rows).
    """
    if not icp_csv:
        return None
    transforms = _load_icp_transforms(icp_csv, invert_icp)
    if not transforms:
        return None

    if "anchor_png" not in db_df.columns:
        # よくある別名も試す
        if "ri_png" in db_df.columns:
            db_df = db_df.rename(columns={"ri_png": "anchor_png"})
        else:
            raise ValueError("db_df must contain 'anchor_png' column")

    # 全体 gallery index
    gallery = db_df["anchor_png"].astype(str).tolist()
    a2i = {p: i for i, p in enumerate(gallery)}

    # 先に確保する
    gt: Dict[float, list] = {t: [set() for _ in range(len(pairs_df))] for t in thresholds_m}

    # 効率化のため pair_id ごとにまとめる
    if "pair_id" not in pairs_df.columns:
        raise ValueError("pairs_df must contain 'pair_id' for soft evaluation")
    if "db_seg" not in pairs_df.columns:
        raise ValueError("pairs_df must contain 'db_seg' for soft evaluation")
    if "anchor_png" not in pairs_df.columns:
        raise ValueError("pairs_df must contain 'anchor_png' for strict GT")

    # db_seg ごとの anchor 表を一度だけ作る
    if "db_seg" not in db_df.columns or "anchor_id" not in db_df.columns:
        raise ValueError("db_df must contain 'db_seg' and 'anchor_id' columns for soft evaluation")
    db_df = db_df.copy()
    db_df["anchor_id"] = db_df["anchor_id"].astype(int)

    for pair_id, q_idx in pairs_df.groupby("pair_id").groups.items():
        pid = str(pair_id)
        if pid not in transforms:
            continue
        T = transforms[pid]
        R = T[:2, :2].astype(np.float64)
        txy = T[:2, 3].astype(np.float64)

        # この pair に対応する db segment
        db_seg = str(pairs_df.loc[list(q_idx)[0], "db_seg"])
        seg_anchors = db_df[db_df["db_seg"] == db_seg]
        if len(seg_anchors) == 0:
            continue

        anchors_csv = _find_anchors_csv_from_anchor_png(str(seg_anchors.iloc[0]["anchor_png"]))
        if anchors_csv is None:
            continue
        anc_xy_tbl = _load_anchors_xy(anchors_csv)
        # anchor_id -> (x,y) の対応を作る
        xy_map = dict(zip(anc_xy_tbl["anchor_id"].tolist(), zip(anc_xy_tbl["x"].tolist(), anc_xy_tbl["y"].tolist())))

        # アンカー位置（db world）
        seg_aids = seg_anchors["anchor_id"].astype(int).tolist()
        seg_gidx = [a2i[str(p)] for p in seg_anchors["anchor_png"].astype(str).tolist()]
        seg_xy_db = np.array([xy_map.get(aid, (np.nan, np.nan)) for aid in seg_aids], dtype=np.float64)

        good = np.isfinite(seg_xy_db).all(axis=1)
        if not np.any(good):
            continue
        seg_gidx = np.array(seg_gidx, dtype=np.int64)[good]
        seg_xy_db = seg_xy_db[good]

        # XY 平面で db->q 変換を適用する（yaw + 並進）
        seg_xy_q = (seg_xy_db @ R.T) + txy[None, :]

        # この pair 内の各 query について、この db_seg に属する全 anchor との距離を計算する
        for qi in q_idx:
            cam_png = str(pairs_df.loc[qi, "cam_png"])
            qxy = _load_cam_xy_from_sidecar(cam_png)
            if qxy is None:
                # 計算できない場合は strict-only セットへフォールバックする。
                qxy_np = None
            else:
                qxy_np = np.array([qxy[0], qxy[1]], dtype=np.float64)

            # データ作成方針に合わせ、t>=0.5m では集合が空にならないよう strict GT を必ず含める。
            strict_gt_png = str(pairs_df.loc[qi, "anchor_png"])
            strict_gt_idx = a2i.get(strict_gt_png, None)

            if qxy_np is None:
                for t in thresholds_m:
                    if strict_gt_idx is not None:
                        gt[t][qi].add(int(strict_gt_idx))
                continue

            d = np.linalg.norm(seg_xy_q - qxy_np[None, :], axis=1)
            for t in thresholds_m:
                idxs = seg_gidx[d <= float(t)]
                if idxs.size:
                    gt[t][qi].update(map(int, idxs.tolist()))
                if strict_gt_idx is not None:
                    gt[t][qi].add(int(strict_gt_idx))

    return gt


@torch.no_grad()
def eval_retrieval_metrics(
    model: DualEncoder,
    pairs_df: pd.DataFrame,
    db_df: Optional[pd.DataFrame],
    device,
    *,
    image_size=(384, 640),
    batch_size: int = 32,
    strict_Ks: Tuple[int, ...] = (1, 5, 10, 30, 50),
    soft_thresholds_m: Tuple[float, ...] = (1.0, 2.0, 5.0, 10.0),
    soft_Ks: Tuple[int, ...] = (1, 5, 10),
    soft_mrr_threshold_m: float = 2.0,
    icp_csv: str = "",
    invert_icp: bool = False,
    cam_mean=0.5,
    cam_std=0.5,
    anc_mean=0.5,
    anc_std=0.5,
) -> Dict[str, float]:
    """Strict / Soft の検索指標を計算する。

    Strict:
      - GT is a single anchor (the matched nearest anchor used to build the manifest)
      - R@{1,5,10,30,50}, MRR, MedianRank

    Soft:
      - GT is a *set* of anchors within distance threshold t (in meters) in the Q world
      - We report R@{1,5,10} for t in {1,2,5,10}
      - Additionally, for t=2m we report Soft MRR / MedianRank
    """
    model.eval()

    # query ごとに 1 行であることを前提に安全確認する
    if "cam_png" not in pairs_df.columns or "anchor_png" not in pairs_df.columns:
        raise ValueError("pairs_df must have 'cam_png' and 'anchor_png'")
    if "label" in pairs_df.columns:
        pairs_df = pairs_df[pairs_df["label"] == 1].copy()
    pairs_df = pairs_df.drop_duplicates(subset=["cam_png"]).reset_index(drop=True)

    cam_list = pairs_df["cam_png"].astype(str).tolist()
    strict_gt_png = pairs_df["anchor_png"].astype(str).tolist()

    # Gallery 側 anchor はまず db_df（完全な DB）を使い、なければ pairs に出現した anchor だけへフォールバックする。
    if db_df is not None and len(db_df) > 0:
        if "anchor_png" not in db_df.columns:
            if "ri_png" in db_df.columns:
                db_df = db_df.rename(columns={"ri_png": "anchor_png"})
            else:
                raise ValueError("db_df must contain 'anchor_png' (or 'ri_png')")
        anchors = db_df["anchor_png"].astype(str).drop_duplicates().tolist()
    else:
        anchors = sorted(list({p for p in strict_gt_png}))

    a2i = {p: i for i, p in enumerate(anchors)}
    gt_idx = [a2i.get(p, -1) for p in strict_gt_png]
    if any(i < 0 for i in gt_idx):
        missing = sum(1 for i in gt_idx if i < 0)
        raise RuntimeError(f"{missing} strict GT anchors are missing from the DB gallery")

    tfm_cam = T.Compose([T.Resize(image_size), T.ToTensor(), T.Normalize([cam_mean] * 3, [cam_std] * 3)])
    tfm_anc = T.Compose([T.Resize(image_size), T.ToTensor(), T.Normalize([anc_mean] * 3, [anc_std] * 3)])

    def _embed(paths: list, encoder: nn.Module, tfm: Any, bs: int = 32) -> torch.Tensor:
        """モデルを通して埋め込みを計算し、必要な形に整える。"""
        out = []
        for i in range(0, len(paths), bs):
            ps = paths[i : i + bs]
            xb = torch.stack([to_rgb_tensor(p, tfm) for p in ps], 0).to(device)
            eb = encoder(xb)
            out.append(eb)
        return torch.cat(out, 0)

    # 埋め込み計算
    A = _embed(anchors, model.int, tfm_anc, bs=int(batch_size))          # [M,D] on device
    C = _embed(cam_list, model.cam, tfm_cam, bs=int(batch_size))         # [Q,D] on device

    sim = C @ A.t()                                         # [Q,M]
    order = torch.argsort(sim, dim=1, descending=True).cpu().numpy()  # [Q,M]

    # Strict の順位計算
    strict_ranks0 = []  # 0-based
    for i, gi in enumerate(gt_idx):
        pos = int(np.where(order[i] == gi)[0][0])
        strict_ranks0.append(pos)
    strict_ranks0 = np.asarray(strict_ranks0, dtype=np.int64)

    metrics: Dict[str, float] = {}

    for k in strict_Ks:
        metrics[f"Strict_R@{k}"] = float(np.mean(strict_ranks0 < int(k)))
    metrics["Strict_MRR"] = float(np.mean(1.0 / (strict_ranks0 + 1)))
    metrics["Strict_MedianRank"] = float(np.median(strict_ranks0 + 1))

    # 後方互換の別名（旧コードがこれらを監視していた）
    for k in (1, 5, 10):
        if f"Strict_R@{k}" in metrics:
            metrics[f"R@{k}"] = metrics[f"Strict_R@{k}"]

    # Soft GT 集合（任意）
    soft_gt = _build_soft_gt_sets(
        pairs_df,
        db_df if db_df is not None else pd.DataFrame({"anchor_png": anchors}),
        icp_csv=icp_csv,
        invert_icp=invert_icp,
        thresholds_m=soft_thresholds_m,
    )

    if soft_gt is not None:
        # Soft recall の計算
        for t in soft_thresholds_m:
            t_key = _fmt_t(t)
            gt_sets = soft_gt[t]
            for k in soft_Ks:
                hit = 0
                kk = int(k)
                for i in range(len(cam_list)):
                    topk = order[i, :kk]
                    if any(int(a) in gt_sets[i] for a in topk):
                        hit += 1
                metrics[f"Soft_t{t_key}m_R@{k}"] = float(hit / len(cam_list))

        # 単一しきい値（既定: 2m）に対する Soft MRR / MedianRank
        t0 = float(soft_mrr_threshold_m)
        if t0 in soft_gt:
            gt_sets = soft_gt[t0]
            first_pos = []  # 0-based
            for i in range(len(cam_list)):
                found = None
                for r, a in enumerate(order[i]):
                    if int(a) in gt_sets[i]:
                        found = r
                        break
                if found is None:
                    found = 10**9
                first_pos.append(found)
            first_pos = np.asarray(first_pos, dtype=np.int64)
            t_key = _fmt_t(t0)
            metrics[f"Soft_t{t_key}m_MRR"] = float(np.mean(1.0 / (first_pos + 1)))
            metrics[f"Soft_t{t_key}m_MedianRank"] = float(np.median(first_pos + 1))

    return metrics


@torch.no_grad()
def eval_recall_at_k(
    model: DualEncoder,
    pos_df: pd.DataFrame,
    device,
    image_size=(384, 640),
    Ks=(1, 5, 10),
    cam_mean=0.5,
    cam_std=0.5,
    anc_mean=0.5,
    anc_std=0.5,
) -> Dict[str, float]:
    """後方互換のためのラッパー（Strict のみ）。"""
    m = eval_retrieval_metrics(
        model,
        pos_df,
        db_df=None,
        device=device,
        image_size=image_size,
        strict_Ks=tuple(int(k) for k in Ks),
        soft_thresholds_m=(),
        soft_Ks=(),
        icp_csv="",
        cam_mean=cam_mean,
        cam_std=cam_std,
        anc_mean=anc_mean,
        anc_std=anc_std,
    )
    return {f"R@{k}": float(m.get(f"Strict_R@{k}", float("nan"))) for k in Ks}


# ============ 学習ランナー ============

def _atomic_torch_save(obj: dict, path: Path) -> None:
    """クラッシュ時の破損を避けるため、torch checkpoint を atomic に保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def capture_rng_state() -> Dict[str, Any]:
    """再現可能に resume できるよう、RNG 状態を保存する。"""
    st: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            st["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return st


def restore_rng_state(st: Dict[str, Any]) -> None:
    """`capture_rng_state` で保存した RNG 状態を復元する。可能な範囲で行う。"""
    try:
        if st.get("python") is not None:
            random.setstate(st["python"])
    except Exception:
        pass
    try:
        if st.get("numpy") is not None:
            np.random.set_state(st["numpy"])
    except Exception:
        pass
    try:
        if st.get("torch") is not None:
            torch.set_rng_state(st["torch"])
    except Exception:
        pass
    if torch.cuda.is_available() and st.get("cuda") is not None:
        try:
            torch.cuda.set_rng_state_all(st["cuda"])
        except Exception:
            pass


def save_ckpt(
    path: Path,
    model: torch.nn.Module,
    args,
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """学習 checkpoint を保存する。

    Backward compatible with the old signature (path, model, args, epoch).
    If optimizer/scheduler/scaler are provided, their states are also saved.
    """
    ckpt: Dict[str, Any] = {
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "args": vars(args),
        "rng_state": capture_rng_state(),
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        try:
            ckpt["scheduler"] = scheduler.state_dict()
        except Exception:
            pass
    if scaler is not None:
        try:
            ckpt["scaler"] = scaler.state_dict()
        except Exception:
            pass
    if extra:
        ckpt.update(extra)

    _atomic_torch_save(ckpt, path)


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """optimizer の状態テンソルを指定 device へ移す。可能な範囲で行う。"""
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)


def resolve_resume_path(resume: str, ckpt_dir: Path) -> Optional[Path]:
    """再開学習に使うチェックポイントパスを解決する。"""
    if not resume:
        return None
    r = str(resume).strip()
    if not r:
        return None
    rl = r.lower()
    if rl in ["auto", "last", "latest"]:
        p = ckpt_dir / "last.ckpt"
        if p.exists():
            return p
        # 代替として一番新しい ckpt を選ぶ
        cands = sorted(ckpt_dir.glob("*.ckpt"), key=lambda x: x.stat().st_mtime, reverse=True)
        return cands[0] if cands else None
    return Path(r)


def load_ckpt(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
    restore_rng: bool = True,
) -> Dict[str, Any]:
    """checkpoint を読み込み、各状態を復元する。戻り値は checkpoint 辞書。"""
    ckpt = torch.load(path, map_location="cpu")

    # 後方互換: 古い ckpt は生の state_dict の場合がある
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=strict)
        if (missing or unexpected) and not strict:
            print(f"[WARN] resume non-strict: missing={len(missing)} unexpected={len(unexpected)}")
    elif isinstance(ckpt, dict):
        # よくあるキー名を試す
        sd = ckpt.get("model") or ckpt.get("state")
        if sd is not None:
            model.load_state_dict(sd, strict=strict)
        else:
            # それ自体が state_dict とみなす
            model.load_state_dict(ckpt, strict=strict)
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    if device is not None:
        model.to(device)

    if optimizer is not None and isinstance(ckpt, dict) and ckpt.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            if device is not None:
                _optimizer_to_device(optimizer, device)
        except Exception as e:
            print(f"[WARN] failed to restore optimizer state: {e}")

    if scheduler is not None and isinstance(ckpt, dict) and ckpt.get("scheduler") is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as e:
            print(f"[WARN] failed to restore scheduler state: {e}")

    if scaler is not None and isinstance(ckpt, dict) and ckpt.get("scaler") is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as e:
            print(f"[WARN] failed to restore AMP scaler state: {e}")

    if restore_rng and isinstance(ckpt, dict) and ckpt.get("rng_state") is not None:
        restore_rng_state(ckpt["rng_state"])

    return ckpt


def append_csv(path: Path, obj: Dict[str, Any], header_order=None):
    """1 行分の記録を CSV に追記する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([obj])
    if header_order is not None:
        cols = [c for c in header_order if c in df.columns] + [c for c in df.columns if c not in header_order]
        df = df[cols]
    write_header = (not path.exists())
    df.to_csv(path, mode="a", index=False, header=write_header)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    """`path` に JSONL を 1 行追記する。

    We keep this tiny (no pandas dependency) and make sure the parent
    directory exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    """CLI 引数を解釈し、入力の読み込みから結果保存までの一連の処理を実行する。"""
    ap = argparse.ArgumentParser(description="Contrastive VLAD Training (dual-val; modality-wise normalize)")
    # 必須
    ap.add_argument("--pairs", required=True, help="train pairs parquet（cam_png, anchor_png, label=1）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--resume", type=str, default="", help="Checkpoint to resume from, or 'auto' to use out_dir/checkpoints/last.ckpt")
    ap.add_argument("--resume-strict", type=str2bool, default=True, help="strict=True enforces exact key match when loading weights")
    ap.add_argument("--no-resume-rng", type=str2bool, default=False, help="do not restore RNG states when resuming")
    ap.add_argument("--save-every-steps", type=int, default=0, help="save last checkpoint every N train steps (0 disables)")
    ap.add_argument(
        "--save-every-epochs",
        type=int,
        default=1,
        help=(
            "save an additional per-epoch checkpoint (epoch_XXX.ckpt) every N epochs. "
            "0 disables (keeps only last.ckpt and best.ckpt). Default=1 saves every epoch."
        ),
    )

    # 学習ハイパラ
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--backbone", choices=["resnet18","resnet34","resnet50"], default="resnet18")
    ap.add_argument("--clusters", type=int, default=32)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--freeze-stages", type=int, default=2)
    ap.add_argument("--image-h", type=int, default=384)
    ap.add_argument("--image-w", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, choices=["cuda","cpu"], default="cuda")
    ap.add_argument("--temperature", type=float, default=0.07, help="InfoNCE 温度（小さいほど鋭い相違）")

    # on-the-fly val（2系統）
    ap.add_argument("--val-pairs", type=str, default=None, help="validation pairs parquet（val1）")
    ap.add_argument(
        "--val-db",
        type=str,
        default=None,
        help="val1 の DB マニフェスト（csv/parquet）。未指定なら --val-pairs と同階層の '*_db.*' を推測して読み込みます。",
    )
    ap.add_argument(
        "--val-icp-csv",
        type=str,
        default=None,
        help="val1 の ICP 結果CSV（soft GT 生成用）。未指定なら soft 指標はスキップします。",
    )
    ap.add_argument(
        "--val-invert-icp",
        type=str2bool,
        default=False,
        help="val1 の ICP 変換を反転して使う（q->db がCSVに入っている場合に true）。通常は false（db->q）。",
    )
    ap.add_argument("--val1-name", type=str, default="val1", help="val1 の表示名")
    ap.add_argument("--val2-pairs", type=str, default=None, help="validation pairs parquet（val2 任意）")
    ap.add_argument(
        "--val2-db",
        type=str,
        default=None,
        help="val2 の DB マニフェスト（csv/parquet）。未指定なら --val2-pairs と同階層の '*_db.*' を推測して読み込みます。",
    )
    ap.add_argument(
        "--val2-icp-csv",
        type=str,
        default=None,
        help="val2 の ICP 結果CSV（soft GT 生成用）。未指定なら soft 指標はスキップします。",
    )
    ap.add_argument(
        "--val2-invert-icp",
        type=str2bool,
        default=False,
        help="val2 の ICP 変換を反転して使う（q->db がCSVに入っている場合に true）。通常は false（db->q）。",
    )
    ap.add_argument("--val2-name", type=str, default="val2", help="val2 の表示名")

    # ベスト判定
    ap.add_argument("--best-on", choices=["val1","val2"], default="val2", help="どちらの val を監視するか")
    ap.add_argument("--best-by", choices=["R@1","R@5","R@10","loss"], default="R@5",
                    help="best.ckpt の判定指標（R@K: 大きいほど良い / loss: 小さいほど良い）")

    # ログ/実行制御
    ap.add_argument("--no-tb", action="store_true", help="TensorBoard無効化")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", help="AMP 有効化（fp16）")
    ap.add_argument("--scheduler", choices=["none","cosine"], default="cosine")
    ap.add_argument("--grad-clip", type=float, default=0.0, help=">0 で有効")
    ap.add_argument("--patience", type=int, default=0, help="早期打ち切り patience（0 なら無効）")
    ap.add_argument("--jitter", type=float, default=0.2, help="CAM側の軽いColorJitter（0で無効）")

    # Train: RI を anchor_dir から選ぶ（損失は変えずにデータ拡張として扱う）
    ap.add_argument("--train-ri-mode", choices=["fixed","epoch_cycle"], default="epoch_cycle",
                    help="trainの正例RI選択。fixed=parquetのanchor_png固定 / epoch_cycle=anchor_dir内のRIをepochごとに巡回（偏りにくい）")
    ap.add_argument("--train-ri-count", type=int, default=8,
                    help="1 submapあたりのRI枚数（ri_000..ri_{count-1} を想定）。0以下なら --train-ri-glob で探索")
    ap.add_argument("--train-ri-glob", type=str, default="ri_*.png",
                    help="train-ri-count<=0 のときに使う glob（例: ri_*.png）")
    ap.add_argument("--train-ri-min", type=int, default=1,
                    help="anchor_dir内のRI候補がこの枚数未満なら anchor_png にフォールバック")
    ap.add_argument("--train-ri-seed", type=int, default=None,
                    help="RI巡回順のseed（未指定なら --seed を使用）")

    # モダリティ別 stats
    ap.add_argument("--modality-stats", type=str, default=None, help="JSON file with cam/anchor mean/std")

    args = ap.parse_args()

    # train-ri-seed 未指定なら --seed を使う
    if args.train_ri_seed is None:
        args.train_ri_seed = args.seed

    # 乱数固定とデバイス
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # パス確認
    if not Path(args.pairs).exists():
        raise FileNotFoundError(f"--pairs not found: {args.pairs}")
    if args.val_pairs is not None and not Path(args.val_pairs).exists():
        raise FileNotFoundError(f"--val-pairs not found: {args.val_pairs}")
    if args.val2_pairs is not None and not Path(args.val2_pairs).exists():
        raise FileNotFoundError(f"--val2-pairs not found: {args.val2_pairs}")

    # 出力ディレクトリ
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    tb = None if args.no_tb else SummaryWriter(str(out / "tb"))

    # 引数保存
    with open(out / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    image_size = (args.image_h, args.image_w)

    # ★ stats を読む（無ければ 0.5 固定）
    cam_m, cam_s, anc_m, anc_s = load_modality_stats(args.modality_stats)
    print(f"[INFO] device        : {device.type}")
    print(f"[INFO] best-by metric: {args.best_by} ({'maximize' if args.best_by!='loss' else 'minimize'})")
    print(f"[INFO] best-on       : {args.best_on}")
    print(f"[INFO] normalize stats: cam(mean={cam_m:.6f}, std={cam_s:.6f}) | anchor(mean={anc_m:.6f}, std={anc_s:.6f})")

    print(f"[INFO] train-ri-mode  : {args.train_ri_mode} (count={args.train_ri_count}, glob='{args.train_ri_glob}', min={args.train_ri_min}, seed={args.train_ri_seed})")

    # Dataset / Loader の準備
    
    ds = PosPairsDataset(
    args.pairs, image_size=image_size, jitter=args.jitter,
    cam_mean=cam_m, cam_std=cam_s, anc_mean=anc_m, anc_std=anc_s,
    ri_mode=args.train_ri_mode,
    ri_count=args.train_ri_count,
    ri_glob=args.train_ri_glob,
    ri_min_candidates=args.train_ri_min,
    ri_seed=args.train_ri_seed,
    )
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type=="cuda"), drop_last=True
    )

    # Model / Optim / Scheduler（元と同等）
    model = DualEncoder(
        backbone=args.backbone, out_dim=args.embed_dim,
        clusters=args.clusters, freeze_stages=args.freeze_stages
    ).to(device)

    # 学習可能パラメ情報（元と同じ出力）
    tot = sum(p.numel() for p in model.parameters())
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[PARAM] total={tot:,}  trainable={trn:,}  ratio={trn/tot:.3f}")
    for branch_name in ["cam", "int"]:
        b = getattr(model, branch_name)
        for blk in ["conv1","bn1","layer1","layer2","layer3","layer4"]:
            if hasattr(b, blk):
                sub = getattr(b, blk)
                n = sum(p.numel() for p in sub.parameters() if p.requires_grad)
                print(f"[PARAM] {branch_name}.{blk}: trainable={n:,}")

    loss_fn = InfoNCELoss(temperature=args.temperature)
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.lr, weight_decay=args.wd)
    if args.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    else:
        sched = None

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and torch.cuda.is_available())

    # on-the-fly 評価セット（pairs / db を事前ロードしておく）
    # - Strict (本文メイン):   R@{1,5,10,30,50} / MRR / MedianRank
    # - Soft   (本文サブ):     t={1,2,5,10}m で R@{1,5,10}、さらに t=2m の MRR / MedianRank
    STRICT_KS = (1, 5, 10, 30, 50)
    SOFT_KS = (1, 5, 10)
    SOFT_TS = (1.0, 2.0, 5.0, 10.0)
    SOFT_MRR_T = 2.0

    pos_df_val1: Optional[pd.DataFrame] = None
    pos_df_val2: Optional[pd.DataFrame] = None
    db_df_val1: Optional[pd.DataFrame] = None
    db_df_val2: Optional[pd.DataFrame] = None

    if args.val_pairs:
        pos_df_val1 = _read_table(args.val_pairs)
        need_cols = {"cam_png", "anchor_png"}
        if not need_cols.issubset(pos_df_val1.columns):
            raise RuntimeError(f"--val-pairs must have columns {need_cols}, got {pos_df_val1.columns.tolist()}")
        if "label" in pos_df_val1.columns:
            pos_df_val1 = pos_df_val1[pos_df_val1["label"] == 1].reset_index(drop=True)

        # DB gallery（未指定なら sibling を推測）
        db_path = args.val_db or _infer_sibling_manifest(args.val_pairs, "db")
        if db_path:
            db_df_val1 = _read_table(db_path)
            if "anchor_png" not in db_df_val1.columns:
                raise RuntimeError(f"val1 db manifest must have 'anchor_png' column: {db_path}")
            db_df_val1 = db_df_val1.drop_duplicates(subset=["anchor_png"]).reset_index(drop=True)
        else:
            print("[WARN] val1 db manifest is not provided and could not be inferred -> using anchors referenced by val1 pairs only")

    if args.val2_pairs:
        pos_df_val2 = _read_table(args.val2_pairs)
        need_cols = {"cam_png", "anchor_png"}
        if not need_cols.issubset(pos_df_val2.columns):
            raise RuntimeError(f"--val2-pairs must have columns {need_cols}, got {pos_df_val2.columns.tolist()}")
        if "label" in pos_df_val2.columns:
            pos_df_val2 = pos_df_val2[pos_df_val2["label"] == 1].reset_index(drop=True)

        db_path = args.val2_db or _infer_sibling_manifest(args.val2_pairs, "db")
        if db_path:
            db_df_val2 = _read_table(db_path)
            if "anchor_png" not in db_df_val2.columns:
                raise RuntimeError(f"val2 db manifest must have 'anchor_png' column: {db_path}")
            db_df_val2 = db_df_val2.drop_duplicates(subset=["anchor_png"]).reset_index(drop=True)
        else:
            print("[WARN] val2 db manifest is not provided and could not be inferred -> using anchors referenced by val2 pairs only")

    csv_path = out / "logs" / "train_metrics.csv"
    jsonl_path = out / "logs" / "train_metrics.jsonl"
    # CSVログのヘッダ（主要な指標だけ列を用意。append_csv は未定義キーも最後に追加で書きます）
    header = ["epoch", "loss", "lr"]
    for prefix, enabled in (("val1", pos_df_val1 is not None), ("val2", pos_df_val2 is not None)):
        if not enabled:
            continue
        header += [f"{prefix}/R@{k}" for k in (1, 5, 10)]
        header += [f"{prefix}/Strict_R@{k}" for k in STRICT_KS]
        header += [f"{prefix}/Strict_MRR", f"{prefix}/Strict_MedianRank"]
        for t in SOFT_TS:
            tstr = str(int(t)) if float(t).is_integer() else str(t).replace(".", "p")
            header += [f"{prefix}/Soft_t{tstr}m_R@{k}" for k in SOFT_KS]
        header += [f"{prefix}/Soft_t2m_MRR", f"{prefix}/Soft_t2m_MedianRank"]

    # best-by 用の監視
    monitor = args.best_by                     # 'R@1' / 'R@5' / 'R@10' / 'loss'
    maximize = (monitor != "loss")
    best_val = -inf if maximize else inf
    no_improve = 0

    ckpt_dir = out / "checkpoints"
    start_epoch = 1
    global_step = 0

    # ---- Resume（任意） ----
    resume_path = resolve_resume_path(args.resume, ckpt_dir)
    if resume_path is not None and resume_path.exists():
        print(f"[INFO] resuming from: {resume_path}")
        ck = load_ckpt(
            resume_path,
            model,
            optimizer=optim,
            scheduler=sched,
            scaler=scaler,
            device=device,
            strict=bool(args.resume_strict),
            restore_rng=(not args.no_resume_rng),
        )
        # 学習 bookkeeping（best metric / patience）を復元する
        if isinstance(ck, dict):
            ck_monitor = ck.get("monitor")
            ck_maximize = ck.get("maximize")
            if ck_monitor is not None and ck_monitor != monitor:
                print(f"[WARN] checkpoint monitor={ck_monitor} != current monitor={monitor}. Resetting best/patience.")
            elif ck_maximize is not None and bool(ck_maximize) != bool(maximize):
                print(f"[WARN] checkpoint maximize={ck_maximize} != current maximize={maximize}. Resetting best/patience.")
            else:
                best_val = ck.get("best_val", best_val)
                no_improve = int(ck.get("no_improve", no_improve))
            global_step = int(ck.get("global_step", global_step))
            start_epoch = int(ck.get("epoch", 0)) + 1

        print(f"[INFO] resume ok: next_epoch={start_epoch} best_val={best_val} no_improve={no_improve} global_step={global_step}")
    else:
        if args.resume:
            print(f"[WARN] resume requested but checkpoint not found: {args.resume}")

    for epoch in range(start_epoch, args.epochs + 1):

        # Train Dataset の RI 選択を epoch に追従させる（0-indexed）
        # ※ DataLoader worker からも見えるよう共有メモリで保持している
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch - 1)
        model.train()
        running = 0.0
        pbar = tqdm(dl, desc=f"epoch {epoch}/{args.epochs}", ncols=100)
        for cam, anc, _, _ in pbar:
            cam = cam.to(device, non_blocking=True)
            anc = anc.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp and torch.cuda.is_available()):
                zc, za = model(cam, anc)
                loss = loss_fn(zc, za)

            optim.zero_grad(set_to_none=True)
            if args.amp and torch.cuda.is_available():
                scaler.scale(loss).backward()
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optim); scaler.update()
            else:
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optim.step()

            running += float(loss.item())
            pbar.set_postfix(loss=f"{running / max(1, pbar.n):.4f}")

            global_step += 1
            if args.save_every_steps and args.save_every_steps > 0 and (global_step % int(args.save_every_steps) == 0):
                save_ckpt(
                    out / "checkpoints" / "last.ckpt",
                    model,
                    args,
                    epoch,
                    optimizer=optim,
                    scheduler=sched,
                    scaler=scaler,
                    extra={
                        "monitor": monitor,
                        "maximize": maximize,
                        "best_val": best_val,
                        "no_improve": no_improve,
                        "global_step": global_step,
                    },
                )

        # Scheduler を 1 ステップ進める
        if sched is not None:
            sched.step()
        lr_now = optim.param_groups[0]["lr"]

        # ------ 検証 ------
        metrics: Dict[str, Optional[float]] = {}

        if pos_df_val1 is not None and len(pos_df_val1) > 0:
            model.eval()
            with torch.no_grad():
                m1 = eval_retrieval_metrics(
                    model,
                    pairs_df=pos_df_val1,
                    db_df=db_df_val1,
                    device=device,
                    image_size=image_size,
                    batch_size=32,
                    strict_Ks=STRICT_KS,
                    soft_thresholds_m=SOFT_TS,
                    soft_Ks=SOFT_KS,
                    soft_mrr_threshold_m=SOFT_MRR_T,
                    icp_csv=args.val_icp_csv,
                    invert_icp=bool(args.val_invert_icp),
                    cam_mean=cam_m,
                    cam_std=cam_s,
                    anc_mean=anc_m,
                    anc_std=anc_s,
                )
                for k, v in m1.items():
                    metrics[f"val1/{k}"] = float(v)

        if pos_df_val2 is not None and len(pos_df_val2) > 0:
            model.eval()
            with torch.no_grad():
                m2 = eval_retrieval_metrics(
                    model,
                    pairs_df=pos_df_val2,
                    db_df=db_df_val2,
                    device=device,
                    image_size=image_size,
                    batch_size=32,
                    strict_Ks=STRICT_KS,
                    soft_thresholds_m=SOFT_TS,
                    soft_Ks=SOFT_KS,
                    soft_mrr_threshold_m=SOFT_MRR_T,
                    icp_csv=args.val2_icp_csv,
                    invert_icp=bool(args.val2_invert_icp),
                    cam_mean=cam_m,
                    cam_std=cam_s,
                    anc_mean=anc_m,
                    anc_std=anc_s,
                )
                for k, v in m2.items():
                    metrics[f"val2/{k}"] = float(v)

        # ログ出力
        log_loss = float(running/len(dl))
        log = dict(epoch=int(epoch), loss=log_loss, lr=float(lr_now), **metrics)
        print(json.dumps(log, ensure_ascii=False))
        append_csv(csv_path, log, header_order=header)
        append_jsonl(jsonl_path, log)
        if tb:
            tb.add_scalar("train/loss", log["loss"], epoch)
            tb.add_scalar("train/lr", lr_now, epoch)

            # 互換: 以前の R@{1,5,10}
            for k in ["R@1", "R@5", "R@10"]:
                v1 = metrics.get(f"val1/{k}", None)
                if v1 is not None:
                    tb.add_scalar(f"{args.val1_name}/{k}", v1, epoch)
                v2 = metrics.get(f"val2/{k}", None)
                if v2 is not None:
                    tb.add_scalar(f"{args.val2_name}/{k}", v2, epoch)

            # 追加: 本文/サブで見る指標（必要最小限だけ）
            extra_keys = [
                "Strict_R@30",
                "Strict_R@50",
                "Strict_MRR",
                "Strict_MedianRank",
                "Soft_t2m_R@1",
                "Soft_t2m_R@5",
                "Soft_t2m_R@10",
                "Soft_t2m_MRR",
                "Soft_t2m_MedianRank",
            ]
            for k in extra_keys:
                v1 = metrics.get(f"val1/{k}", None)
                if v1 is not None:
                    tb.add_scalar(f"{args.val1_name}/{k}", v1, epoch)
                v2 = metrics.get(f"val2/{k}", None)
                if v2 is not None:
                    tb.add_scalar(f"{args.val2_name}/{k}", v2, epoch)

        # ckpt 保存（last は毎回保存）
        save_ckpt(
            out / "checkpoints" / "last.ckpt",
            model,
            args,
            epoch,
            optimizer=optim,
            scheduler=sched,
            scaler=scaler,
            extra={
                "monitor": monitor,
                "maximize": maximize,
                "best_val": best_val,
                "no_improve": no_improve,
                "global_step": global_step,
            },
        )

        # ckpt 保存（epoch_{:03d}.ckpt）: --save-every-epochs > 0 のときのみ
        #  - 1 の場合: 毎 epoch 保存（ユーザ希望の「全エポック保存」）
        #  - 0 の場合: 無効（last/best のみ）
        if getattr(args, "save_every_epochs", 0) and args.save_every_epochs > 0:
            if (epoch % args.save_every_epochs) == 0:
                save_ckpt(
                    out / "checkpoints" / f"epoch_{epoch:03d}.ckpt",
                    model,
                    args,
                    epoch,
                    optimizer=optim,
                    scheduler=sched,
                    scaler=scaler,
                    extra={
                        "monitor": monitor,
                        "maximize": maximize,
                        "best_val": best_val,
                        "no_improve": no_improve,
                        "global_step": global_step,
                    },
                )

        # ベスト判定
        if monitor == "loss":
            curr = log_loss
            improved = (curr < best_val) if not maximize else (curr > best_val)  # loss は minimize
        else:
            key = f"{args.best_on}/{monitor}"
            curr = metrics.get(key, None)
            improved = (curr is not None) and ((curr > best_val) if maximize else (curr < best_val))

        if improved:
            best_val = curr if curr is not None else best_val
            save_ckpt(
                out / "checkpoints" / "best.ckpt",
                model,
                args,
                epoch,
                optimizer=optim,
                scheduler=sched,
                scaler=scaler,
                extra={
                    "monitor": monitor,
                    "maximize": maximize,
                    "best_val": best_val,
                    "no_improve": no_improve,
                    "global_step": global_step,
                },
            )
            no_improve = 0
        else:
            no_improve += 1

        # 早期打ち切り
        if (args.patience > 0) and (monitor == "loss" or (curr is not None)):
            if no_improve >= args.patience:
                print(f"[EARLY STOP] no improvement in '{monitor}' on {args.best_on} for {args.patience} epoch(s).")
                break

    if tb:
        tb.close()


if __name__ == "__main__":
    main()
