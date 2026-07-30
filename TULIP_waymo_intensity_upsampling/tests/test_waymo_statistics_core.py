import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from waymo_preprocess.statistics_core import (
    FrameBalancedSampler,
    choose_frame_indices,
    frame_statistics,
)


class StatisticsCoreTest(unittest.TestCase):
    def test_separates_even_and_odd_rows(self):
        rows = np.arange(64, dtype=np.float32)[:, None]
        ranges = np.repeat(rows + 1, 3, axis=1)
        intensities = np.repeat(rows, 3, axis=1)
        mask = np.ones_like(ranges, dtype=bool)
        stats = frame_statistics(ranges, intensities, mask)
        self.assertEqual(stats["all_64"]["range_m"]["count"], 64 * 3)
        self.assertEqual(stats["observed_even_32"]["range_m"]["count"], 32 * 3)
        self.assertEqual(stats["generated_odd_32"]["range_m"]["count"], 32 * 3)
        self.assertLess(
            stats["observed_even_32"]["intensity"]["mean"],
            stats["generated_odd_32"]["intensity"]["mean"],
        )

    def test_frame_balanced_sampler_caps_each_frame(self):
        sampler = FrameBalancedSampler(values_per_frame=4)
        sampler.add(np.arange(100, dtype=np.float32))
        sampler.add(np.arange(2, dtype=np.float32))
        self.assertEqual(sampler.array().size, 6)

    def test_chooses_endpoints(self):
        self.assertEqual(choose_frame_indices(200, 3), [0, 99, 199])


if __name__ == "__main__":
    unittest.main()
