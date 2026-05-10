# dual_backbone_wrapper.py
# LoFTR の backbone を 2 本（RI 用 / CAM 用）に分けて持つラッパ。
# もともとは「backbone(torch.cat([img0, img1]))」という前提だったが、
# kornia 版 LoFTR は image0 / image1 を別々に self.backbone(...) に渡すため、
# それに合わせて実装を変更している。
#
# 新しい仕様:
#   - 1 回目の forward 呼び出し: image0 (RI) 用 backbone を使う
#   - 2 回目の forward 呼び出し: image1 (CAM) 用 backbone を使う
#   - 3 回目: 再び RI、4 回目: CAM ... というふうに偶奇で切り替える
#
# こうすることで、
#   (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(img0), self.backbone(img1)
# のような kornia LoFTR の実装と整合がとれる。
#
# 既存の学習済み ckpt（backbone_ri / backbone_cam を持つもの）とも
# モジュール名を変えずに互換にしている。

from typing import Tuple
import torch
import torch.nn as nn


class DualBackboneWrapper(nn.Module):
    """
    Wrap two backbones (RI/CAM) but expose a single-backbone API:

      forward(x) where x has shape [B, C, H, W]

    呼び出し回数の偶奇によって、RI 用 / CAM 用 backbone を切り替える。
    kornia.feature.LoFTR の forward 内では

        (feat_c0, feat_f0), (feat_c1, feat_f1) = \
            self.backbone(image0), self.backbone(image1)

    の順で呼ばれることを前提にしている。
    """

    def __init__(self, backbone_ri: nn.Module, backbone_cam: nn.Module):
        super().__init__()
        self.backbone_ri = backbone_ri
        self.backbone_cam = backbone_cam

        # LoFTR から何回呼ばれたかをカウントするためのカウンタ
        # 0 回目 (偶数) -> RI 用, 1 回目 (奇数) -> CAM 用, 2 回目 -> RI 用, ...
        self._call_counter = 0

    @torch.no_grad()
    def copy_from_single(self, single_backbone: nn.Module):
        """
        単一の backbone から両方の backbone_ri / backbone_cam を初期化する補助関数。
        （必要なら使う。既存の ckpt をロードするだけなら必須ではない。）
        """
        sd = single_backbone.state_dict()
        _ = self.backbone_ri.load_state_dict(sd, strict=False)
        _ = self.backbone_cam.load_state_dict(sd, strict=False)
        return self

    def reset_pair_counter(self):
        """
        呼び出しカウンタをリセットするための関数。
        通常の LoFTR 推論ループでは特に呼ぶ必要はないが、
        明示的に「次の呼び出しは必ず RI から始めたい」場合に使える。
        """
        self._call_counter = 0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, C, H, W]
          - LoFTR からは image0 / image1 が別々に渡される想定。

        戻り値:
          - feats_c, feats_f : 両方とも [B, C', H', W'] 形式の特徴マップ
        """
        if x.dim() != 4:
            raise ValueError(f"expected 4D tensor [B, C, H, W], got shape={list(x.shape)}")

        # 偶数回目の呼び出し -> RI 用 backbone
        # 奇数回目の呼び出し -> CAM 用 backbone
        if self._call_counter % 2 == 0:
            feats_c, feats_f = self.backbone_ri(x)
        else:
            feats_c, feats_f = self.backbone_cam(x)

        self._call_counter += 1
        return feats_c, feats_f
