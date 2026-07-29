import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from inspect_waymo_npz import inspect_arrays


class InspectArraysTest(unittest.TestCase):
    def test_reports_only_valid_pixels(self) -> None:
        range_image = np.array([[0.0, 10.0], [20.0, 0.0]], dtype=np.float32)
        intensity = np.array([[0.0, 0.2], [0.8, 0.0]], dtype=np.float32)
        report = inspect_arrays(range_image, intensity)
        self.assertEqual(report["shape"], [2, 2])
        self.assertEqual(report["valid_count"], 2)
        self.assertAlmostEqual(report["valid_range_m"]["mean"], 15.0)
        self.assertAlmostEqual(report["valid_intensity"]["mean"], 0.5)

    def test_rejects_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            inspect_arrays(np.zeros((2, 2)), np.zeros((3, 2)))


if __name__ == "__main__":
    unittest.main()
