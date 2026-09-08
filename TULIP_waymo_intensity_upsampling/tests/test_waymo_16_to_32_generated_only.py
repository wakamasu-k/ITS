import unittest

import torch

from tools.evaluate_waymo_16_to_32_generated_only import (
    channel_mae,
    generated_only_mask,
    raw_signals,
    resolve_range_scale_m,
)


class GeneratedOnlyEvaluationTest(unittest.TestCase):
    def test_generated_only_mask_keeps_odd_rows(self):
        mask = torch.ones((1, 4, 2), dtype=torch.bool)
        result = generated_only_mask(mask)
        expected = torch.tensor([[[False, False], [True, True],
                                  [False, False], [True, True]]])
        self.assertTrue(torch.equal(result, expected))

    def test_generated_only_mask_preserves_invalid_pixels(self):
        mask = torch.ones((1, 4, 2), dtype=torch.bool)
        mask[:, 3, 1] = False
        result = generated_only_mask(mask)
        self.assertFalse(bool(result[0, 3, 1]))
        self.assertTrue(bool(result[0, 3, 0]))
        self.assertFalse(bool(result[0, 2, 0]))

    def test_channel_mae_uses_only_generated_mask(self):
        prediction = torch.zeros((1, 2, 4, 1), dtype=torch.float32)
        target = torch.zeros_like(prediction)
        prediction[:, 1, 0, 0] = 100.0
        prediction[:, 1, 1, 0] = 2.0
        prediction[:, 1, 3, 0] = 2.0
        mask = generated_only_mask(torch.ones((1, 4, 1), dtype=torch.bool))
        self.assertEqual(channel_mae(prediction, target, mask, 1), 2.0)

    def test_rejects_wrong_mask_shape(self):
        with self.assertRaisesRegex(ValueError, "mask must"):
            generated_only_mask(torch.ones((1, 15, 1), dtype=torch.bool))

    def test_resolves_range_scale_from_checkpoint(self):
        checkpoint = {"args": {"range_scale_m": 75.0}}
        self.assertEqual(resolve_range_scale_m(checkpoint, None), 75.0)

    def test_rejects_range_scale_mismatch(self):
        checkpoint = {"args": {"range_scale_m": 75.0}}
        with self.assertRaisesRegex(ValueError, "does not match"):
            resolve_range_scale_m(checkpoint, 1.0)

    def test_raw_signals_restores_range_and_log_intensity(self):
        input_signal = torch.tensor([[[[2.0]], [[3.0]]]])
        target_signal = torch.tensor([[[[4.0]], [[5.0]]]])
        prediction = torch.tensor([[[[6.0]], [[7.0]]]])
        x_raw, y_raw, pred_raw = raw_signals(
            input_signal,
            target_signal,
            prediction,
            "log1p",
            10.0,
        )
        self.assertAlmostEqual(float(x_raw[0, 0, 0, 0]), 20.0)
        self.assertAlmostEqual(float(y_raw[0, 0, 0, 0]), 40.0)
        self.assertAlmostEqual(float(pred_raw[0, 0, 0, 0]), 60.0)
        expected_intensity = float(torch.expm1(torch.tensor(3.0)))
        self.assertAlmostEqual(float(x_raw[0, 1, 0, 0]), expected_intensity)


if __name__ == "__main__":
    unittest.main()
