import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.evaluate_waymo_16_to_32_pointcloud import aggregate_summaries


class WaymoPointCloudEvaluationTest(unittest.TestCase):
    def test_weighted_aggregation(self):
        rows = [
            {
                "tulip_metrics": {
                    "count": 2, "coordinate_mae_m": 1.0,
                    "euclidean_mae_m": 2.0, "intensity_mae": 3.0,
                },
                "nearest_metrics": None,
                "linear_metrics": {
                    "count": 1, "coordinate_mae_m": 4.0,
                    "euclidean_mae_m": 5.0, "intensity_mae": 6.0,
                },
            },
            {
                "tulip_metrics": {
                    "count": 1, "coordinate_mae_m": 4.0,
                    "euclidean_mae_m": 5.0, "intensity_mae": 6.0,
                },
                "nearest_metrics": {
                    "count": 2, "coordinate_mae_m": 2.0,
                    "euclidean_mae_m": 4.0, "intensity_mae": 8.0,
                },
                "linear_metrics": None,
            },
        ]
        result = aggregate_summaries(rows)
        self.assertEqual(result["frames"], 2)
        self.assertEqual(result["tulip"]["count"], 3)
        self.assertAlmostEqual(result["tulip"]["euclidean_mae_m"], 3.0)
        self.assertAlmostEqual(result["tulip"]["intensity_mae"], 4.0)
        self.assertEqual(result["nearest"]["count"], 2)
        self.assertAlmostEqual(result["nearest"]["intensity_mae"], 8.0)

    def test_empty_input_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            aggregate_summaries([])


if __name__ == "__main__":
    unittest.main()
