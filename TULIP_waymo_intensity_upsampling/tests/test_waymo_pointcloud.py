import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tulip_adapter.waymo_pointcloud import pointwise_metrics, valid_range_mask


class WaymoPointCloudTest(unittest.TestCase):
    def test_valid_range_mask_rejects_invalid_values(self):
        values = np.array([[0.0, 1.0, np.nan], [2.0, -1.0, np.inf]], dtype=np.float32)
        supplied = np.ones_like(values, dtype=bool)
        supplied[1, 0] = False
        actual = valid_range_mask(values, supplied)
        np.testing.assert_array_equal(actual, [[False, True, False], [False, False, False]])

    def test_pointwise_metrics(self):
        predicted = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        target = np.array([[1, 1, 3], [2, 5, 6]], dtype=np.float32)
        predicted_i = np.array([2, 5], dtype=np.float32)
        target_i = np.array([1, 3], dtype=np.float32)
        actual = pointwise_metrics(predicted, target, predicted_i, target_i)
        self.assertEqual(actual["count"], 2)
        self.assertAlmostEqual(actual["coordinate_mae_m"], 0.5)
        self.assertAlmostEqual(actual["intensity_mae"], 1.5)

    def test_pointwise_metrics_rejects_empty(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            pointwise_metrics(
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
