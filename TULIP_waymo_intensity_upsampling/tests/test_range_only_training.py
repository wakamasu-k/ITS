import sys
import unittest
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tulip_adapter.range_only_training import (
    RangeLosses, compute_losses, prepare_range_batch, select_training_loss)


class RangeOnlyTrainingTest(unittest.TestCase):
    def test_prepare_zeros_invalid_input_and_target(self):
        batch = {
            "input": torch.tensor([[[[2.0, -1.0]]]]),
            "target": torch.tensor([[[[3.0, -1.0]]]]),
            "input_mask": torch.tensor([[[True, False]]]),
            "target_mask": torch.tensor([[[True, False]]]),
            "generated_mask": torch.tensor([[[True, False]]]),
        }
        input_range, target, target_mask, generated = prepare_range_batch(
            batch, "cpu")
        self.assertEqual(input_range.tolist(), [[[[2.0, 0.0]]]])
        self.assertEqual(target.tolist(), [[[[3.0, 0.0]]]])
        self.assertTrue(bool(target_mask[0, 0, 0]))
        self.assertTrue(bool(generated[0, 0, 0]))

    def test_reports_all_and_generated_losses(self):
        prediction = torch.tensor([[[[0.0, 4.0]]]])
        target = torch.tensor([[[[2.0, 2.0]]]])
        target_mask = torch.tensor([[[True, True]]])
        generated_mask = torch.tensor([[[False, True]]])
        losses = compute_losses(
            prediction, target, target_mask, generated_mask)
        self.assertEqual(float(losses.all_valid), 2.0)
        self.assertEqual(float(losses.generated), 2.0)

    def test_selects_explicit_loss_scope(self):
        losses = RangeLosses(torch.tensor(1.0), torch.tensor(2.0))
        self.assertEqual(float(select_training_loss(losses, "all_valid")), 1.0)
        self.assertEqual(float(select_training_loss(losses, "generated")), 2.0)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            select_training_loss(losses, "bad")


if __name__ == "__main__":
    unittest.main()
