import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tulip_adapter.waymo_32_to_64_tulip import masked_l1_2ch  # noqa: E402
from tulip_adapter.waymo_2ch_weighted_training import (  # noqa: E402
    weighted_two_channel_l1,
)


class WeightedTwoChannelTrainingTest(unittest.TestCase):
    def test_equal_weights_match_existing_two_channel_l1(self) -> None:
        prediction = torch.zeros(1, 2, 2, 1)
        target = torch.tensor([[[[2.0], [4.0]], [[10.0], [20.0]]]])
        mask = torch.ones(1, 2, 1, dtype=torch.bool)
        expected = masked_l1_2ch(prediction, target, mask)
        actual = weighted_two_channel_l1(
            prediction, target, mask, range_weight=1.0,
            intensity_weight=1.0,
        )
        self.assertAlmostEqual(float(actual), float(expected), places=6)

    def test_intensity_priority_weight_reduces_range_contribution(self) -> None:
        prediction = torch.zeros(1, 2, 1, 1)
        target = torch.tensor([[[[10.0]], [[1.0]]]])
        mask = torch.ones(1, 1, 1, dtype=torch.bool)
        equal = weighted_two_channel_l1(
            prediction, target, mask, range_weight=1.0,
            intensity_weight=1.0,
        )
        priority = weighted_two_channel_l1(
            prediction, target, mask, range_weight=0.05,
            intensity_weight=1.0,
        )
        self.assertGreater(float(equal), float(priority))

    def test_rejects_negative_or_zero_total_weights(self) -> None:
        values = torch.zeros(1, 2, 1, 1)
        mask = torch.ones(1, 1, 1, dtype=torch.bool)
        with self.assertRaises(ValueError):
            weighted_two_channel_l1(
                values, values, mask, range_weight=-1.0,
                intensity_weight=1.0,
            )
        with self.assertRaises(ValueError):
            weighted_two_channel_l1(
                values, values, mask, range_weight=0.0,
                intensity_weight=0.0,
            )


if __name__ == "__main__":
    unittest.main()
