import sys, tempfile, unittest
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tulip_adapter.range_only_evaluation import (interpolate_32_to_64,
    masked_metrics, prediction_statistics, save_difference_image)

class EvaluationTest(unittest.TestCase):
    def test_baselines(self):
        values = torch.arange(32, dtype=torch.float32).view(1, 1, 32, 1)
        mask = torch.ones((1, 32, 1), dtype=torch.bool)
        nearest = interpolate_32_to_64(values, mask, mode="nearest")
        linear = interpolate_32_to_64(values, mask, mode="linear")
        self.assertEqual(tuple(linear.shape), (1, 1, 64, 1))
        self.assertEqual(float(nearest[0, 0, 1, 0]), 0.0)
        self.assertEqual(float(linear[0, 0, 1, 0]), 0.5)
        self.assertEqual(float(linear[0, 0, 63, 0]), 31.0)

    def test_invalid_neighbour_fallback(self):
        values = torch.zeros((1, 1, 32, 1)); values[0, 0, 1, 0] = 20
        mask = torch.ones((1, 32, 1), dtype=torch.bool); mask[0, 0, 0] = False
        for mode in ("nearest", "linear"):
            self.assertEqual(float(interpolate_32_to_64(
                values, mask, mode=mode)[0, 0, 1, 0]), 20.0)

    def test_metrics_and_stats(self):
        prediction = torch.tensor([[[[-1., 4.], [3., 8.]]]])
        target = torch.tensor([[[[1., 2.], [3., 4.]]]])
        valid = torch.tensor([[[True, True], [True, False]]])
        generated = torch.tensor([[[False, True], [True, False]]])
        metrics = masked_metrics(prediction, target, valid, generated)
        self.assertAlmostEqual(metrics.all_valid_l1, 4 / 3)
        self.assertEqual(metrics.generated_l1, 1.0)
        stats = prediction_statistics(prediction, valid)
        self.assertEqual(stats["prediction_valid_min_m"], -1.0)
        self.assertEqual(stats["prediction_valid_negative_count"], 1)

    def test_image(self):
        target = torch.ones((1, 1, 64, 4)); mask = torch.ones((1, 64, 4), dtype=torch.bool)
        generated = torch.zeros_like(mask); generated[:, 1::2] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "difference.png"
            save_difference_image(path, target, target + 1, target, mask, generated)
            self.assertGreater(path.stat().st_size, 0)

if __name__ == "__main__": unittest.main()
