import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from prepare_32_to_64_pair import EVEN_ROWS, build_pair


class NpzLike:
    def __init__(self, values):
        self.values = values
        self.files = list(values)

    def __getitem__(self, key):
        return self.values[key]


def make_values():
    range_64 = np.arange(64 * 5, dtype=np.float32).reshape(64, 5) + 1
    intensity_64 = range_64 / 100
    mask_64 = range_64 > 0
    return {
        "range_64": range_64,
        "intensity_64": intensity_64,
        "valid_mask_64": mask_64,
        "range_32": range_64[EVEN_ROWS],
        "intensity_32": intensity_64[EVEN_ROWS],
        "valid_mask_32": mask_64[EVEN_ROWS],
        "ring_ids_32": EVEN_ROWS,
    }


class PreparePairTest(unittest.TestCase):
    def test_builds_32_to_64_pair(self):
        arrays, metadata = build_pair(NpzLike(make_values()))
        self.assertEqual(arrays["input_32"].shape, (32, 5, 2))
        self.assertEqual(arrays["target_64"].shape, (64, 5, 2))
        self.assertEqual(metadata["generated_target_rows"], list(range(1, 64, 2)))
        self.assertTrue(metadata["exact_subset_verified"])

    def test_rejects_mismatched_scene_or_processing(self):
        values = make_values()
        values["range_32"] = values["range_32"].copy()
        values["range_32"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "exact even-row subset"):
            build_pair(NpzLike(values))


if __name__ == "__main__":
    unittest.main()
