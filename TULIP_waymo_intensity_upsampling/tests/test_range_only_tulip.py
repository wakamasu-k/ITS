import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tulip_adapter.range_only_tulip import (
    AnisotropicFinalPatchExpanding, WaymoRangeOnlyTULIP, masked_l1,
    pad_width, required_padded_width,
)


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 1
        self.norm_layer = nn.LayerNorm
        self.pixel_shuffle = False
        self.final_patch_expanding = nn.Identity()

    def forward(self, input_range, target_range, mc_drop=False):
        self.last_input_shape = tuple(input_range.shape)
        return torch.zeros_like(target_range)


class RangeOnlyTulipTest(unittest.TestCase):
    def test_waymo_width_pads_to_encoder_multiple(self):
        self.assertEqual(required_padded_width(2650), 2816)
        self.assertEqual(required_padded_width(2816), 2816)

    def test_anisotropic_head_restores_row_patch(self):
        head = AnisotropicFinalPatchExpanding(
            dim=3, vertical_scale=2, horizontal_scale=4)
        output = head(torch.zeros(1, 32, 664, 3))
        self.assertEqual(tuple(output.shape), (1, 64, 2656, 3))

    def test_input_width_padding_wraps_horizontally(self):
        value = torch.tensor([[[[1.0, 2.0, 3.0]]]])
        padded = pad_width(value, 5, circular=True)
        self.assertEqual(padded.tolist(), [[[[1.0, 2.0, 3.0, 1.0, 2.0]]]])

    def test_wrapper_crops_back_to_native_width_and_masks_loss(self):
        backbone = DummyBackbone()
        model = WaymoRangeOnlyTULIP(backbone, native_width=5, padded_width=8)
        input_range = torch.ones(2, 1, 32, 5)
        target_range = torch.ones(2, 1, 64, 5)
        mask = torch.ones(2, 64, 5, dtype=torch.bool)
        prediction, loss = model(input_range, target_range, mask)
        self.assertEqual(backbone.last_input_shape, (2, 1, 32, 8))
        self.assertEqual(tuple(prediction.shape), (2, 1, 64, 5))
        self.assertEqual(float(loss), 1.0)

    def test_masked_l1_ignores_invalid_pixels(self):
        prediction = torch.tensor([[[[5.0, 100.0]]]])
        target = torch.tensor([[[[2.0, 0.0]]]])
        mask = torch.tensor([[[True, False]]])
        self.assertEqual(float(masked_l1(prediction, target, mask)), 3.0)


if __name__ == "__main__":
    unittest.main()
