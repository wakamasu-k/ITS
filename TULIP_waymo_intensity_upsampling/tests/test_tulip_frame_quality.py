import sys
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from waymo_preprocess.tulip_frame_quality import (
    audit_arrays, build_tulip_16_32,
)

def valid_arrays():
    rows = np.arange(64, dtype=np.float32)[:, None]
    range_64 = np.repeat(rows + 1, 5, axis=1)
    intensity_64 = range_64 * 0.1
    mask_64 = np.ones_like(range_64, dtype=bool)
    return {
        "range_64": range_64,
        "intensity_64": intensity_64,
        "valid_mask_64": mask_64,
        "range_32": range_64[::2].copy(),
        "intensity_32": intensity_64[::2].copy(),
        "valid_mask_32": mask_64[::2].copy(),
        "ring_ids_32": np.arange(0, 64, 2, dtype=np.int32),
    }

class TulipFrameQualityTest(unittest.TestCase):
    def test_audits_exact_even_subset(self):
        report = audit_arrays(valid_arrays())
        self.assertEqual(report["width"], 5)
        self.assertEqual(report["valid_count_32"], 160)

    def test_rejects_non_exact_range_subset(self):
        arrays = valid_arrays()
        arrays["range_32"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "range_32"):
            audit_arrays(arrays)

    def test_rejects_wrong_ring_ids(self):
        arrays = valid_arrays()
        arrays["ring_ids_32"][0] = 1
        with self.assertRaisesRegex(ValueError, "ring_ids_32"):
            audit_arrays(arrays)

    def test_rejects_nonfinite_valid_intensity(self):
        arrays = valid_arrays()
        arrays["intensity_64"][0, 0] = np.inf
        arrays["intensity_32"][0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            audit_arrays(arrays)

    def test_builds_16_from_32_not_directly_from_64(self):
        arrays = valid_arrays()
        pair = build_tulip_16_32(arrays)
        np.testing.assert_array_equal(pair["range_16"], arrays["range_32"][::2])
        np.testing.assert_array_equal(
            pair["intensity_16"], arrays["intensity_32"][::2])
        np.testing.assert_array_equal(
            pair["ring_ids_16"], np.arange(0, 64, 4, dtype=np.int32))
        self.assertEqual(pair["range_16"].shape, (16, 5))
        self.assertEqual(pair["range_32"].shape, (32, 5))

if __name__ == "__main__":
    unittest.main()
