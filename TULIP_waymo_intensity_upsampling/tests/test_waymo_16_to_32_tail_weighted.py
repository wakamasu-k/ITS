import unittest

import torch

from tools.train_waymo_16_to_32_tail_weighted import tail_weighted_loss


class TailWeightedLossTest(unittest.TestCase):
    def setUp(self):
        self.target = torch.zeros((1, 2, 4, 2), dtype=torch.float32)
        self.prediction = torch.zeros_like(self.target)
        self.valid = torch.ones((1, 4, 2), dtype=torch.bool)
        self.generated = torch.zeros((1, 4, 2), dtype=torch.bool)
        self.generated[:, 1::2] = True

    def test_observed_rows_are_excluded(self):
        self.target[:, 1, 1, :] = 2.0
        self.prediction[:, 1, 0, :] = 1000.0
        self.prediction[:, 1, 1, :] = 2.0
        total, _, intensity, _, weighted = tail_weighted_loss(
            self.prediction, self.target, self.valid, self.generated,
            range_weight=0.05, intensity_weight=1.0,
            tail_threshold=20.0, tail_weight=10.0,
        )
        self.assertAlmostEqual(float(intensity), 0.0, places=6)
        self.assertAlmostEqual(float(weighted), 0.0, places=6)
        self.assertTrue(torch.isfinite(total))

    def test_tail_weight_changes_weighted_intensity_loss(self):
        self.target[:, 1, 1, :] = 100.0
        self.target[:, 1, 3, :] = 1.0
        _, _, unweighted, _, weighted = tail_weighted_loss(
            self.prediction, self.target, self.valid, self.generated,
            range_weight=0.05, intensity_weight=1.0,
            tail_threshold=20.0, tail_weight=10.0,
        )
        self.assertAlmostEqual(float(unweighted), 50.5, places=6)
        self.assertGreater(float(weighted), float(unweighted))

    def test_invalid_generated_pixels_are_excluded(self):
        self.valid[:, 1, :] = False
        self.target[:, 1, 1, :] = 1000.0
        self.prediction[:, 1, 1, :] = 1000.0
        total, range_loss, intensity, tail_fraction, weighted = tail_weighted_loss(
            self.prediction, self.target, self.valid, self.generated,
            range_weight=0.05, intensity_weight=1.0,
            tail_threshold=20.0, tail_weight=10.0,
        )
        self.assertEqual(float(range_loss), 0.0)
        self.assertEqual(float(intensity), 0.0)
        self.assertEqual(float(tail_fraction), 0.0)
        self.assertEqual(float(weighted), 0.0)
        self.assertEqual(float(total), 0.0)

    def test_rejects_empty_generated_mask(self):
        with self.assertRaisesRegex(ValueError, "contains no pixels"):
            tail_weighted_loss(
                self.prediction, self.target, self.valid,
                torch.zeros_like(self.generated),
                range_weight=0.05, intensity_weight=1.0,
                tail_threshold=20.0, tail_weight=10.0,
            )


if __name__ == "__main__":
    unittest.main()
