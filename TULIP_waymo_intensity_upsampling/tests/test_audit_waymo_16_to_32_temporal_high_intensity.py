import unittest

import numpy as np

from tools.audit_waymo_16_to_32_temporal_high_intensity import nearest_distances, transform_points


class TemporalHighIntensityAuditTest(unittest.TestCase):
    def test_transform_points_applies_translation(self):
        points = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = [4.0, -1.0, 2.0]
        np.testing.assert_allclose(
            transform_points(points, transform),
            np.array([[5.0, 1.0, 5.0]], dtype=np.float32),
        )

    def test_nearest_distances_and_empty_reference(self):
        query = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float32)
        reference = np.array([[1.0, 0.0, 0.0], [2.0, 4.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(nearest_distances(query, reference), [1.0, 1.0])
        self.assertTrue(np.isinf(nearest_distances(query, np.empty((0, 3), dtype=np.float32))).all())

    def test_rejects_invalid_transform_shape(self):
        with self.assertRaisesRegex(ValueError, "transform"):
            transform_points(np.zeros((1, 3), dtype=np.float32), np.eye(3, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
