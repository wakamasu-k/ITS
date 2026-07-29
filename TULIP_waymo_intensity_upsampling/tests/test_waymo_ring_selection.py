import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from waymo_preprocess.range_image import (
    extract_range_intensity,
    select_even_rings,
)


class WaymoRingSelectionTest(unittest.TestCase):
    def test_even_ring_subset_is_exact(self) -> None:
        rows = np.arange(64, dtype=np.float32)[:, None]
        range_64 = np.repeat(rows, 3, axis=1) + 1.0
        intensity_64 = range_64 * 0.01
        mask_64 = range_64 > 0

        range_32, intensity_32, mask_32, ring_ids = select_even_rings(
            range_64, intensity_64, mask_64
        )

        np.testing.assert_array_equal(ring_ids, np.arange(0, 64, 2))
        np.testing.assert_array_equal(range_32, range_64[::2])
        np.testing.assert_array_equal(intensity_32, intensity_64[::2])
        np.testing.assert_array_equal(mask_32, mask_64[::2])

    def test_extracts_physical_channels_without_normalization(self) -> None:
        native = np.zeros((64, 5, 4), dtype=np.float32)
        native[..., 0] = 12.5
        native[..., 1] = 1234.0
        range_64, intensity_64, mask_64 = extract_range_intensity(native)
        self.assertTrue(np.all(range_64 == 12.5))
        self.assertTrue(np.all(intensity_64 == 1234.0))
        self.assertTrue(np.all(mask_64))

    def test_rejects_non_native_height(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[64,W,C>=2\]"):
            extract_range_intensity(np.zeros((32, 5, 4), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
