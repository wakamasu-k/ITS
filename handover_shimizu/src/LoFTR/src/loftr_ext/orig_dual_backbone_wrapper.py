# dual_backbone_wrapper.py
# LoFTRの「backbone(torch.cat([img0,img1]))」という前提を壊さずに、
# 内部で img0→RI用Backbone、img1→CAM用Backbone に振り分けてから
# 出力を [img0群; img1群] の順で連結して返すラッパ。
from typing import Tuple
import torch
import torch.nn as nn

class DualBackboneWrapper(nn.Module):
    """
    Wrap two backbones (RI/CAM) to mimic a single backbone API:
      forward(x_cat) where x_cat has shape [2B, C, H, W]
    Returns:
      feats_c_cat, feats_f_cat with batch dim [2B, ...]
    """
    def __init__(self, backbone_ri: nn.Module, backbone_cam: nn.Module):
        super().__init__()
        self.backbone_ri = backbone_ri
        self.backbone_cam = backbone_cam

    @torch.no_grad()
    def copy_from_single(self, single_backbone: nn.Module):
        """Initialize both RI/CAM backbones from a single-backbone weights."""
        sd = single_backbone.state_dict()
        _ = self.backbone_ri.load_state_dict(sd, strict=False)
        _ = self.backbone_cam.load_state_dict(sd, strict=False)
        return self

    def forward(self, x_cat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_cat: [2B, C, H, W] where first half are img0 (RI), second half img1 (CAM)
        The wrapped backbones must return (feats_c, feats_f) with batch-first shape.
        """
        assert x_cat.dim() == 4 and x_cat.size(0) % 2 == 0, \
            f"expect [2B, C, H, W], got {list(x_cat.shape)}"
        B2 = x_cat.size(0)
        B = B2 // 2
        x0 = x_cat[:B]   # RI
        x1 = x_cat[B:]   # CAM

        feats_c0, feats_f0 = self.backbone_ri(x0)
        feats_c1, feats_f1 = self.backbone_cam(x1)

        # 期待通り [img0群; img1群] の順で連結して返す
        feats_c = torch.cat([feats_c0, feats_c1], dim=0)
        feats_f = torch.cat([feats_f0, feats_f1], dim=0)
        return feats_c, feats_f
