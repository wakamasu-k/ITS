import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.evaluate_waymo_16_to_32_oracle_range import aggregate


class WaymoOracleRangeTest(unittest.TestCase):
    def test_aggregate_and_geometry_penalty(self):
        rows = [
            {
                "oracle_range_metrics": {
                    "count": 2, "coordinate_mae_m": 0.0,
                    "euclidean_mae_m": 0.0, "intensity_mae": 1.0,
                },
                "predicted_range_metrics": {
                    "count": 2, "coordinate_mae_m": 1.0,
                    "euclidean_mae_m": 2.0, "intensity_mae": 1.0,
                },
            },
            {
                "oracle_range_metrics": {
                    "count": 1, "coordinate_mae_m": 0.0,
                    "euclidean_mae_m": 0.0, "intensity_mae": 2.0,
                },
                "predicted_range_metrics": {
                    "count": 1, "coordinate_mae_m": 2.0,
                    "euclidean_mae_m": 4.0, "intensity_mae": 2.0,
                },
            },
        ]
        result = aggregate(rows)
        self.assertEqual(result["oracle_range"]["count"], 3)
        self.assertAlmostEqual(result["oracle_range"]["intensity_mae"], 4.0 / 3.0)
        self.assertAlmostEqual(result["predicted_range"]["euclidean_mae_m"], 8.0 / 3.0)
        self.assertAlmostEqual(result["range_geometry_penalty_euclidean_mae_m"], 8.0 / 3.0)

    def test_empty_input_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            aggregate([])


if __name__ == "__main__":
    unittest.main()
