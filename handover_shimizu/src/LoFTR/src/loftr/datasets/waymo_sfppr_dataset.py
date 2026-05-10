import os
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WaymoSFPPRDataset(Dataset):
    """
    Waymo LiDAR-map (anchor) × camera 画像ペアを、
    SFPPR 風 LoFTR 学習用に読み出す Dataset クラス。

    機能:
      - manifest CSV (pf0_train_manifest_cov_pxmask.csv) からペア情報を読む
      - source (strict / close_only など) でフィルタ
      - mkpts0/mkpts1 (2D-2D 対応 GT) を読み込む
      - coverage_mask_coarse_npz から coarse coverage mask を読み込む
      - 必要なら画像・GT・マスクをリサイズする
        - 固定サイズ (H, W)
        - もしくは「長辺を L ピクセルにしてアスペクト比維持」
    """

    def __init__(
        self,
        manifest_csv: str,
        sources: Optional[Sequence[str]] = None,
        min_n_matches: int = 8,
        resize_to: Optional[Tuple[int, int]] = None,
        resize_long_side: Optional[int] = None,
    ) -> None:
        """
        Args:
            manifest_csv: pf0_train_manifest_cov_pxmask.csv のパス
            sources: 使う source 名のリスト（例: ["strict","close_only"]）
            min_n_matches: mkpts 本数の下限（"n" 列があればそれでフィルタ）
            resize_to: 画像を (H, W) にリサイズする場合に指定。
                       例: (480, 640)。None のときはこの指定は使わない。
            resize_long_side: 長辺をこのピクセル数にリサイズして
                              アスペクト比を維持したい場合に指定。
                              例: 840。None なら使わない。
        """
        super().__init__()
        self.manifest_csv = manifest_csv
        self.sources = list(sources) if sources is not None else None
        self.min_n_matches = int(min_n_matches)
        self.resize_to = resize_to
        self.resize_long_side = resize_long_side

        if self.resize_to is not None and self.resize_long_side is not None:
            print(
                "[WaymoSFPPRDataset] WARNING: resize_to と resize_long_side "
                "が両方指定されています。resize_to を優先します。"
            )

        df = pd.read_csv(manifest_csv)

        need_cols = [
            "npz",
            "anc_png",
            "cam_png",
            "w0",
            "h0",
            "w1",
            "h1",
            "source",
            "coverage_mask_coarse_npz",
        ]
        missing = [c for c in need_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"[WaymoSFPPRDataset] missing columns {missing} in {manifest_csv}"
            )

        # source でフィルタ
        if self.sources is not None:
            df = df[df["source"].isin(self.sources)]

        # mkpts 本数でフィルタ（n 列がある場合のみ）
        if "n" in df.columns:
            df = df[df["n"] >= self.min_n_matches]

        # ファイル存在チェック
        def _exists(p: str) -> bool:
            return isinstance(p, str) and os.path.exists(p)

        mask_ok = (
            df["npz"].map(_exists)
            & df["anc_png"].map(_exists)
            & df["cam_png"].map(_exists)
            & df["coverage_mask_coarse_npz"].map(_exists)
        )
        missing_rows = (~mask_ok).sum()
        if missing_rows > 0:
            print(
                f"[WaymoSFPPRDataset] WARNING: drop {missing_rows} rows with missing files"
            )
            df = df[mask_ok]

        self.df = df.reset_index(drop=True)

        print(
            f"[WaymoSFPPRDataset] loaded {len(self.df)} samples from {manifest_csv}"
        )
        print(f"[WaymoSFPPRDataset] sources filter: {self.sources}")
        print(f"[WaymoSFPPRDataset] min_n_matches: {self.min_n_matches}")
        if self.resize_to is not None:
            print(f"[WaymoSFPPRDataset] resize_to (fixed): {self.resize_to}")
        elif self.resize_long_side is not None and self.resize_long_side > 0:
            print(
                f"[WaymoSFPPRDataset] resize_long_side: {self.resize_long_side} (keep aspect ratio)"
            )
        else:
            print("[WaymoSFPPRDataset] no resize")

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _load_gray(path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"[WaymoSFPPRDataset] failed to read image: {path}")
        return img

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        anc_path = row["anc_png"]
        cam_path = row["cam_png"]
        npz_path = row["npz"]
        cov_coarse_npz_path = row["coverage_mask_coarse_npz"]

        # 画像読み込み（元解像度）
        anc = self._load_gray(anc_path)
        cam = self._load_gray(cam_path)
        H0, W0 = anc.shape

        # GT 対応読み込み
        z = np.load(npz_path)
        if "mkpts0" in z:
            mkpts0 = z["mkpts0"].astype(np.float32)
            mkpts1 = z["mkpts1"].astype(np.float32)
        elif "kpts0" in z:
            mkpts0 = z["kpts0"].astype(np.float32)
            mkpts1 = z["kpts1"].astype(np.float32)
        else:
            raise KeyError(
                f"[WaymoSFPPRDataset] {npz_path} has no mkpts0/kpts0 keys."
            )

        # coarse coverage mask 読み込み（元の coarse 解像度）
        cov = np.load(cov_coarse_npz_path)
        if "mask_coarse" in cov:
            mask_coarse = cov["mask_coarse"].astype(np.float32)
        else:
            raise KeyError(
                f"[WaymoSFPPRDataset] {cov_coarse_npz_path} has no 'mask_coarse'."
            )

        # manifest 上のサイズ（一応チェック用）
        h0 = int(row["h0"])
        w0 = int(row["w0"])
        if (H0, W0) != (h0, w0):
            print(
                f"[WaymoSFPPRDataset] WARNING: anc size mismatch at idx={idx}: "
                f"img={H0}x{W0}, csv={h0}x{w0}"
            )

        # ===== リサイズの初期値（デフォルトはリサイズ無し） =====
        image0_np = anc
        image1_np = cam
        H_used, W_used = H0, W0

        # ===== 1) 固定サイズリサイズが指定されている場合 =====
        if self.resize_to is not None:
            H_new, W_new = self.resize_to
            if H_new <= 0 or W_new <= 0:
                raise ValueError(
                    f"[WaymoSFPPRDataset] invalid resize_to={self.resize_to}"
                )

            anc_resized = cv2.resize(
                anc, (W_new, H_new), interpolation=cv2.INTER_AREA
            )
            cam_resized = cv2.resize(
                cam, (W_new, H_new), interpolation=cv2.INTER_AREA
            )

            scale_y = float(H_new) / float(H0)
            scale_x = float(W_new) / float(W0)
            mkpts0 = mkpts0.copy()
            mkpts1 = mkpts1.copy()
            mkpts0[:, 0] *= scale_x
            mkpts0[:, 1] *= scale_y
            mkpts1[:, 0] *= scale_x
            mkpts1[:, 1] *= scale_y

            Hc_new = H_new // 8
            Wc_new = W_new // 8
            if Hc_new <= 0 or Wc_new <= 0:
                raise ValueError(
                    f"[WaymoSFPPRDataset] invalid coarse size from resize_to={self.resize_to}"
                )
            mask_coarse_resized = cv2.resize(
                mask_coarse, (Wc_new, Hc_new), interpolation=cv2.INTER_AREA
            )
            mask_coarse = mask_coarse_resized.astype(np.float32)

            image0_np = anc_resized
            image1_np = cam_resized
            H_used, W_used = H_new, W_new

        # ===== 2) 長辺リサイズ（SFPPR/LoFTR 流儀） =====
        elif self.resize_long_side is not None and self.resize_long_side > 0:
            long0 = max(H0, W0)
            if long0 <= 0:
                raise ValueError(
                    f"[WaymoSFPPRDataset] invalid original size: {H0}x{W0}"
                )

            # 長辺が self.resize_long_side になるようにスケール（拡大はしない）
            scale = float(self.resize_long_side) / float(long0)
            if scale > 1.0:
                scale = 1.0

            H_new = int(round(H0 * scale))
            W_new = int(round(W0 * scale))
            if H_new <= 0 or W_new <= 0:
                raise ValueError(
                    f"[WaymoSFPPRDataset] invalid resized size: {H_new}x{W_new}"
                )

            anc_resized = cv2.resize(
                anc, (W_new, H_new), interpolation=cv2.INTER_AREA
            )
            cam_resized = cv2.resize(
                cam, (W_new, H_new), interpolation=cv2.INTER_AREA
            )

            scale_y = float(H_new) / float(H0)
            scale_x = float(W_new) / float(W0)
            mkpts0 = mkpts0.copy()
            mkpts1 = mkpts1.copy()
            mkpts0[:, 0] *= scale_x
            mkpts0[:, 1] *= scale_y
            mkpts1[:, 0] *= scale_x
            mkpts1[:, 1] *= scale_y

            Hc_new = H_new // 8
            Wc_new = W_new // 8
            if Hc_new <= 0 or Wc_new <= 0:
                raise ValueError(
                    f"[WaymoSFPPRDataset] invalid coarse size from resize_long_side={self.resize_long_side}"
                )
            mask_coarse_resized = cv2.resize(
                mask_coarse, (Wc_new, Hc_new), interpolation=cv2.INTER_AREA
            )
            mask_coarse = mask_coarse_resized.astype(np.float32)

            image0_np = anc_resized
            image1_np = cam_resized
            H_used, W_used = H_new, W_new

        # 3) リサイズ無しのときは、そのまま
        #    mask_coarse は元の coarse 解像度 (H0/8, W0/8) のまま使う。

        # ===== Tensor 変換 =====
        image0_np = image0_np.astype(np.float32) / 255.0
        image1_np = image1_np.astype(np.float32) / 255.0

        image0 = torch.from_numpy(image0_np).unsqueeze(0)  # [1, H, W]
        image1 = torch.from_numpy(image1_np).unsqueeze(0)  # [1, H, W]

        sample = {
            "image0": image0,  # LiDAR map intensity image
            "image1": image1,  # camera image
            "mkpts0": torch.from_numpy(mkpts0),  # [N, 2]
            "mkpts1": torch.from_numpy(mkpts1),  # [N, 2]
            "mask0_coarse": torch.from_numpy(mask_coarse),  # [Hc, Wc]
            "imsize0": torch.tensor([H_used, W_used], dtype=torch.long),
            "imsize1": torch.tensor([H_used, W_used], dtype=torch.long),
            "source": str(row["source"]),
            "index": int(idx),
        }

        return sample

class WaymoSFPPRValidDataset(Dataset):
    """
    評価用の Dataset。
    学習用と違い、pixel coverage mask は使わず、
    画像と mkpts0/mkpts1 だけを読み出す。

    特徴:
      - manifest CSV (valid_pseudogt/*/manifest.csv) からペア情報を読む
      - 長辺 resize_long_side ピクセルにリサイズ（アスペクト比維持）に対応
      - 出力は image0/image1/mkpts0/mkpts1/imsize0/imsize1 だけ
    """

    def __init__(
        self,
        manifest_csv: str,
        resize_long_side: Optional[int] = 840,
    ) -> None:
        """
        Args:
            manifest_csv: 評価用 manifest.csv のパス
            resize_long_side: 長辺をこのピクセル数にリサイズして
                              アスペクト比を維持。0 以下ならリサイズ無し。
        """
        super().__init__()
        self.manifest_csv = manifest_csv
        self.resize_long_side = resize_long_side
        self.base_dir = os.path.dirname(os.path.abspath(manifest_csv))

        df = pd.read_csv(manifest_csv)

        need_cols = ["npz", "anc_png", "cam_png", "w0", "h0", "w1", "h1"]
        missing = [c for c in need_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"[WaymoSFPPRValidDataset] missing columns {missing} in {manifest_csv}"
            )

        def _resolve_path(p: str) -> str:
            """WSL + Windows + 相対パスをできるだけ正しく解決する。"""
            if not isinstance(p, str):
                return p
            p = p.strip()
            if p == "":
                return p

            # バックスラッシュをスラッシュに統一
            p_norm = p.replace("\\", "/")

            # Windows 風パス: 例 "F:/waymo/..." or "F:\waymo\..."
            if len(p_norm) >= 2 and p_norm[1] == ":":
                drive = p_norm[0].lower()
                rest = p_norm[2:].lstrip("/\\")
                return os.path.join("/mnt", drive, rest)

            # Linux 絶対パス
            if p_norm.startswith("/"):
                return p_norm

            # 相対パス: base_dir と、その親ディレクトリの両方を試す
            cand1 = os.path.normpath(os.path.join(self.base_dir, p_norm))
            parent_dir = os.path.dirname(self.base_dir)
            cand2 = os.path.normpath(os.path.join(parent_dir, p_norm))

            if os.path.exists(cand1):
                return cand1
            if os.path.exists(cand2):
                return cand2

            # どちらも存在しない場合は cand1 を返しておき、
            # 後段の os.path.exists で落とす
            return cand1

        def _exists(p: str) -> bool:
            if not isinstance(p, str):
                return False
            full = _resolve_path(p)
            return os.path.exists(full)

        mask_ok = (
            df["npz"].map(_exists)
            & df["anc_png"].map(_exists)
            & df["cam_png"].map(_exists)
        )
        missing_rows = (~mask_ok).sum()
        if missing_rows > 0:
            print(
                f"[WaymoSFPPRValidDataset] WARNING: drop {missing_rows} rows with missing files"
            )
            df = df[mask_ok]

        self.df = df.reset_index(drop=True)

        print(
            f"[WaymoSFPPRValidDataset] loaded {len(self.df)} samples from {manifest_csv}"
        )
        if "source" in df.columns:
            sources = sorted(df["source"].unique().tolist())
            print(f"[WaymoSFPPRValidDataset] sources = {sources}")
        if self.resize_long_side is not None and self.resize_long_side > 0:
            print(
                f"[WaymoSFPPRValidDataset] resize_long_side: {self.resize_long_side} (keep aspect ratio)"
            )
        else:
            print("[WaymoSFPPRValidDataset] no resize")

        self._resolve_path = _resolve_path

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _load_gray(path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(
                f"[WaymoSFPPRValidDataset] failed to read image: {path}"
            )
        return img

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        anc_path_raw = row["anc_png"]
        cam_path_raw = row["cam_png"]
        npz_path_raw = row["npz"]

        anc_path = self._resolve_path(anc_path_raw)
        cam_path = self._resolve_path(cam_path_raw)
        npz_path = self._resolve_path(npz_path_raw)

        anc = self._load_gray(anc_path)
        cam = self._load_gray(cam_path)
        H0, W0 = anc.shape

        # GT 対応読み込み
        z = np.load(npz_path)
        if "mkpts0" in z:
            mkpts0 = z["mkpts0"].astype(np.float32)
            mkpts1 = z["mkpts1"].astype(np.float32)
        elif "kpts0" in z:
            mkpts0 = z["kpts0"].astype(np.float32)
            mkpts1 = z["kpts1"].astype(np.float32)
        else:
            raise KeyError(
                f"[WaymoSFPPRValidDataset] {npz_path} has no mkpts0/kpts0 keys."
            )

        h0 = int(row["h0"])
        w0 = int(row["w0"])
        if (H0, W0) != (h0, w0):
            print(
                f"[WaymoSFPPRValidDataset] WARNING: anc size mismatch at idx={idx}: "
                f"img={H0}x{W0}, csv={h0}x{w0}"
            )

        image0_np = anc
        image1_np = cam
        H_used, W_used = H0, W0

        # 長辺リサイズ（SFPPR/LoFTR と同じ流儀）
        if self.resize_long_side is not None and self.resize_long_side > 0:
            long0 = max(H0, W0)
            if long0 <= 0:
                raise ValueError(
                    f"[WaymoSFPPRValidDataset] invalid original size: {H0}x{W0}"
                )

            scale = float(self.resize_long_side) / float(long0)
            if scale > 1.0:
                scale = 1.0

            H_new = int(round(H0 * scale))
            W_new = int(round(W0 * scale))
            if H_new <= 0 or W_new <= 0:
                raise ValueError(
                    f"[WaymoSFPPRValidDataset] invalid resized size: {H_new}x{W_new}"
                )

            anc_resized = cv2.resize(
                anc, (W_new, H_new), interpolation=cv2.INTER_AREA
            )
            cam_resized = cv2.resize(
                cam, (W_new, H_new), interpolation=cv2.INTER_AREA
            )

            scale_y = float(H_new) / float(H0)
            scale_x = float(W_new) / float(W0)
            mkpts0 = mkpts0.copy()
            mkpts1 = mkpts1.copy()
            mkpts0[:, 0] *= scale_x
            mkpts0[:, 1] *= scale_y
            mkpts1[:, 0] *= scale_x
            mkpts1[:, 1] *= scale_y

            image0_np = anc_resized
            image1_np = cam_resized
            H_used, W_used = H_new, W_new

        # Tensor 化
        image0_np = image0_np.astype(np.float32) / 255.0
        image1_np = image1_np.astype(np.float32) / 255.0

        image0 = torch.from_numpy(image0_np).unsqueeze(0)  # [1, H, W]
        image1 = torch.from_numpy(image1_np).unsqueeze(0)  # [1, H, W]

        sample = {
            "image0": image0,
            "image1": image1,
            "mkpts0": torch.from_numpy(mkpts0),
            "mkpts1": torch.from_numpy(mkpts1),
            "imsize0": torch.tensor([H_used, W_used], dtype=torch.long),
            "imsize1": torch.tensor([H_used, W_used], dtype=torch.long),
            "index": int(idx),
        }

        if "source" in row.index:
            sample["source"] = str(row["source"])
        else:
            sample["source"] = "valid"

        return sample



