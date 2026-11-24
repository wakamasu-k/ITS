import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tulip_adapter.waymo_32_to_64_tulip import (  # noqa: E402
    Waymo32To64TULIP,
    masked_l1_2ch,
)


class DummyBackbone(nn.Module):
    """Shape-only backbone matching the adapter's TULIP contract."""

    def __init__(self, output_channels: int = 2) -> None:
        super().__init__()
        self.embed_dim = 1
        self.norm_layer = nn.LayerNorm
        self.pixel_shuffle = False
        self.decoder_pred = nn.Conv2d(1, output_channels, 1, bias=False)
        self.final_patch_expanding = nn.Identity()
        self.last_input_shape = None

    def forward(
        self,
        input_signal: torch.Tensor,
        target_signal: torch.Tensor,
        mc_drop: bool = False,
    ) -> torch.Tensor:
        self.last_input_shape = tuple(input_signal.shape)
        return torch.zeros(
            input_signal.shape[0], 2, 64, input_signal.shape[-1],
            dtype=input_signal.dtype,
        )


class WaymoTwoChannelTulipTest(unittest.TestCase):
    def test_adapter_pads_crops_and_computes_two_channel_loss(self) -> None:
        model = Waymo32To64TULIP(
            DummyBackbone(), native_width=5, padded_width=8
        )
        input_signal = torch.ones(2, 2, 32, 5)
        target_signal = torch.ones(2, 2, 64, 5)
        target_mask = torch.ones(2, 64, 5, dtype=torch.bool)
        prediction, loss = model(input_signal, target_signal, target_mask)
        self.assertEqual(model.backbone.last_input_shape, (2, 2, 32, 8))
        self.assertEqual(tuple(prediction.shape), (2, 2, 64, 5))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss), 1.0)

    def test_masked_l1_uses_shared_spatial_mask_for_both_channels(self) -> None:
        prediction = torch.tensor([[[[5.0, 100.0]], [[7.0, 200.0]]]])
        target = torch.tensor([[[[2.0, 0.0]], [[4.0, 0.0]]]])
        mask = torch.tensor([[[True, False]]])
        self.assertEqual(float(masked_l1_2ch(prediction, target, mask)), 3.0)

    def test_adapter_rejects_one_channel_decoder(self) -> None:
        with self.assertRaises(ValueError):
            Waymo32To64TULIP(
                DummyBackbone(output_channels=1), native_width=5, padded_width=8
            )


if __name__ == "__main__":
    unittest.main()
