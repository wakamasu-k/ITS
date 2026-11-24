import unittest

import torch
from torch import nn

from tulip_adapter.waymo_16_to_32_tulip import Waymo16To32TULIP


class DummyDecoderPred(nn.Module):
    out_channels = 2


class DummyBackbone(nn.Module):
    pixel_shuffle = False
    embed_dim = 96
    norm_layer = nn.LayerNorm

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(2, 96, kernel_size=(1, 8), stride=(1, 2))
        self.patch_embed.norm = nn.Identity()
        self.decoder_pred = DummyDecoderPred()
        self.final_patch_expanding = nn.Identity()

    def forward(self, input_signal, target_signal, mc_drop=False):
        del target_signal, mc_drop
        return input_signal.repeat_interleave(2, dim=2)


class Waymo16To32TULIPTest(unittest.TestCase):
    def test_adapter_contract_and_crop(self):
        model = Waymo16To32TULIP(DummyBackbone())
        input_signal = torch.zeros((1, 2, 16, 2688), dtype=torch.float32)
        target_signal = torch.ones((1, 2, 32, 2688), dtype=torch.float32)
        target_mask = torch.ones((1, 32, 2688), dtype=torch.bool)
        prediction, loss = model(input_signal, target_signal, target_mask)
        self.assertEqual(tuple(prediction.shape), (1, 2, 32, 2688))
        self.assertEqual(tuple(model.crop_native(prediction).shape), (1, 2, 32, 2650))
        self.assertTrue(torch.isfinite(loss))

    def test_rejects_wrong_height(self):
        model = Waymo16To32TULIP(DummyBackbone())
        with self.assertRaisesRegex(ValueError, "input must"):
            model(
                torch.zeros((1, 2, 32, 2688)),
                torch.zeros((1, 2, 32, 2688)),
                torch.ones((1, 32, 2688), dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
