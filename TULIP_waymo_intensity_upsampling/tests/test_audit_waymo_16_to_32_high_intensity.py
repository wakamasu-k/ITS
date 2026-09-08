import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.audit_waymo_16_to_32_high_intensity import audit_pair, distribution


class HighIntensityAuditTest(unittest.TestCase):
    def test_audit_reports_valid_high_points_and_observed_support(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair.npz"
            target = np.zeros((4, 3, 2), dtype=np.float32)
            valid = np.ones((4, 3), dtype=bool)
            target[..., 0] = 10.0
            target[1, 1, 1] = 1500.0
            target[0, 1, 1] = 900.0
            target[2, 1, 1] = 1200.0
            target[3, 0, 1] = 2000.0
            valid[3, 0] = False
            np.savez_compressed(
                path,
                target_32=target,
                target_valid_mask_32=valid,
                ring_ids_32=np.array([0, 2, 4, 6], dtype=np.int32),
                observed_target_rows=np.array([0, 2], dtype=np.int32),
                generated_target_rows=np.array([1, 3], dtype=np.int32),
            )
            frame, points, values = audit_pair(
                path,
                {"segment_id": "segment-test", "frame_index": 7, "timestamp_micros": 9},
            )
        self.assertEqual(frame["ge_1000_count"], 1)
        self.assertEqual(frame["ge_1000_invalid_count"], 1)
        self.assertEqual(frame["ge_1000_exact_observed_support"], 1)
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["native_ring_id"], 2)
        self.assertEqual(points[0]["observed_neighbor_max"], 1200.0)
        self.assertEqual(values.size, 5)

    def test_distribution_rejects_nonfinite_values(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            distribution(np.array([1.0, np.inf]))


if __name__ == "__main__":
    unittest.main()
