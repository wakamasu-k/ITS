import torch
import torch.nn as nn
from einops.einops import rearrange

from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
from .loftr_module import LocalFeatureTransformer, FinePreprocess
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching


class SFPPRLoFTR(nn.Module):
    """
    2本backbone対応のLoFTR（クロスモーダル用のベース）。

    - backbone0: image0 側（例: 反射強度）
    - backbone1: image1 側（例: カメラ）

    repeatability について:
      - 本実装では repeatability は "logits" を出す（Sigmoidは入れない）。
      - data["repeatability0"]      : [B, HWc0] logits
      - data["repeatability0_prob"] : [B, HWc0] sigmoid(logits)

    coarse-only で回すとき:
      - matcher.enable_fine = False にすると fine を完全にスキップする（速度・安定性向上）。
    """

    def __init__(self, config, enable_fine: bool = True, enable_repeatability: bool = True):
        super().__init__()
        self.config = config

        # ===== Dual backbone =====
        self.backbone0 = build_backbone(config)
        self.backbone1 = build_backbone(config)

        # ===== Positional encoding =====
        self.pos_encoding = PositionEncodingSine(
            config["coarse"]["d_model"],
            temp_bug_fix=config["coarse"]["temp_bug_fix"],
        )

        # ===== Coarse transformer + matcher =====
        self.loftr_coarse = LocalFeatureTransformer(config["coarse"])
        self.coarse_matching = CoarseMatching(config["match_coarse"])

        # ===== Fine modules =====
        self.fine_preprocess = FinePreprocess(config)
        self.loftr_fine = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching()

        # ===== Repeatability head (logits; NO sigmoid) =====
        d_model = config["coarse"]["d_model"]
        self.repeatability_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )

        # runtime flags
        self.enable_fine = bool(enable_fine)
        self.enable_repeatability = bool(enable_repeatability)

    def forward(self, data: dict):
        # ===== meta =====
        data.update(
            {
                "bs": data["image0"].size(0),
                "hw0_i": data["image0"].shape[2:],
                "hw1_i": data["image1"].shape[2:],
            }
        )
        B = data["bs"]

        # ===== dual backbone forward (always separate) =====
        feat_c0, feat_f0 = self.backbone0(data["image0"])
        feat_c1, feat_f1 = self.backbone1(data["image1"])

        data.update(
            {
                "hw0_c": feat_c0.shape[2:],
                "hw1_c": feat_c1.shape[2:],
                "hw0_f": feat_f0.shape[2:],
                "hw1_f": feat_f1.shape[2:],
            }
        )

        # ===== coarse transformer =====
        feat_c0 = rearrange(self.pos_encoding(feat_c0), "b c h w -> b (h w) c")
        feat_c1 = rearrange(self.pos_encoding(feat_c1), "b c h w -> b (h w) c")

        mask_c0 = mask_c1 = None
        if "mask0" in data:
            mask_c0 = data["mask0"].flatten(-2)
            mask_c1 = data["mask1"].flatten(-2)

        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # ===== coarse matching =====
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # ===== repeatability (optional) =====
        if self.enable_repeatability:
            rep_logits = self.repeatability_head(feat_c0).squeeze(-1)  # [B, HWc0]
            rep_prob = torch.sigmoid(rep_logits)

            data.update(
                {
                    "repeatability0": rep_logits,
                    "repeatability0_prob": rep_prob,
                }
            )

            # conf_matrix scaling (anchor side)
            if "conf_matrix" in data:
                conf_matrix = data["conf_matrix"]  # [B, HW0, HW1]
                if (
                    conf_matrix.dim() == 3
                    and conf_matrix.size(0) == B
                    and conf_matrix.size(1) == rep_prob.size(1)
                ):
                    data["conf_matrix"] = conf_matrix * rep_prob.unsqueeze(-1)

        # ===== fine stage (optional compute) =====
        if not self.enable_fine:
            device = data["image0"].device
            data.setdefault("mkpts0_f", torch.empty(0, 2, device=device))
            data.setdefault("mkpts1_f", torch.empty(0, 2, device=device))
            data.setdefault("expec_f", torch.empty(0, 3, device=device))
            return

        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(
            feat_f0, feat_f1, feat_c0, feat_c1, data
        )
        if feat_f0_unfold.size(0) != 0:
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)

        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def load_state_dict(self, state_dict, *args, **kwargs):
        # PLチェックポイント互換: "matcher." prefix を剥がす
        for k in list(state_dict.keys()):
            if k.startswith("matcher."):
                state_dict[k.replace("matcher.", "", 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
