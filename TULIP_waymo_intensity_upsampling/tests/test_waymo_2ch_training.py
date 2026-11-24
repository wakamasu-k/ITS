import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tulip_adapter.waymo_2ch_training import (  # noqa: E402
    compute_two_channel_losses,
    interpolate_32_to_64_2ch,
    normalize_two_channel,
    two_channel_metrics,
)


class WaymoTwoChannelTrainingTest(unittest.TestCase):
    def test_normalization_changes_range_only(self) -> None:
        input_signal = torch.tensor([[[[10.0]], [[4.0]]]])
        target_signal = torch.tensor([[[[20.0]], [[8.0]]]])
        normalized_input, normalized_target = normalize_two_channel(
            input_signal, target_signal, 10.0
        )
        self.assertEqual(float(normalized_input[0, 0, 0, 0]), 1.0)
        self.assertEqual(float(normalized_target[0, 0, 0, 0]), 2.0)
        self.assertEqual(float(normalized_input[0, 1, 0, 0]), 4.0)
        self.assertEqual(float(normalized_target[0, 1, 0, 0]), 8.0)

    def test_loss_uses_shared_mask_and_both_channels(self) -> None:
        prediction = torch.zeros(1, 2, 2, 2)
        target = torch.ones(1, 2, 2, 2)
        target_mask = torch.tensor([[[True, False], [True, False]]])
        generated_mask = torch.tensor([[[False, False], [True, False]]])
        losses = compute_two_channel_losses(
            prediction, target, target_mask, generated_mask
        )
        self.assertEqual(float(losses.all_valid), 1.0)
        self.assertEqual(float(losses.generated), 1.0)

    def test_interpolation_preserves_two_channels_and_shape(self) -> None:
        values = torch.zeros(1, 2, 32, 3)
        values[:, 0] = 2.0
        values[:, 1] = 5.0
        mask = torch.ones(1, 32, 3, dtype=torch.bool)
        result = interpolate_32_to_64_2ch(values, mask, mode="linear")
        self.assertEqual(tuple(result.shape), (1, 2, 64, 3))
        self.assertEqual(float(result[:, 0].mean()), 2.0)
        self.assertEqual(float(result[:, 1].mean()), 5.0)

    def test_metrics_are_separate_for_range_and_intensity(self) -> None:
        prediction = torch.zeros(1, 2, 2, 1)
        target = torch.tensor([[[[2.0], [4.0]], [[10.0], [20.0]]]])
        mask = torch.ones(1, 2, 1, dtype=torch.bool)
        metrics = two_channel_metrics(prediction, target, mask, mask)
        self.assertEqual(metrics.all_valid_range, 3.0)
        self.assertEqual(metrics.all_valid_intensity, 15.0)


if __name__ == "__main__":
    unittest.main()
