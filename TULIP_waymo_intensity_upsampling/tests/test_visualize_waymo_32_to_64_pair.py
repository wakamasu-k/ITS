import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from visualize_waymo_32_to_64_pair import expand_rows


class ExpandRowsTest(unittest.TestCase):
    def test_restores_input_to_even_target_rows(self):
        values = np.arange(32 * 4, dtype=np.float32).reshape(32, 4)
        mask = np.ones_like(values, dtype=bool)
        rows = np.arange(0, 64, 2, dtype=np.int32)
        expanded, expanded_mask = expand_rows(values, mask, rows)

        np.testing.assert_array_equal(expanded[rows], values)
        np.testing.assert_array_equal(expanded_mask[rows], mask)
        self.assertTrue(np.all(expanded[1::2] == 0))
        self.assertFalse(np.any(expanded_mask[1::2]))

    def test_rejects_duplicate_rows(self):
        values = np.zeros((2, 4), dtype=np.float32)
        mask = np.ones_like(values, dtype=bool)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            expand_rows(values, mask, np.array([0, 0]), target_height=4)


if __name__ == "__main__":
    unittest.main()
