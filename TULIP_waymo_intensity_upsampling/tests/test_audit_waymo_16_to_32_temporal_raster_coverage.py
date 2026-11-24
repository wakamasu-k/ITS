import unittest

import numpy as np

from tools.audit_waymo_16_to_32_temporal_raster_coverage import rasterize_static_points


class TemporalRasterCoverageTest(unittest.TestCase):
    def test_rasterizes_nearest_range_and_preserves_intensity(self):
        rays = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        points = np.array([[2, 0, 0], [4, 0, 0], [0, 3, 0]], dtype=np.float32)
        intensity = np.array([10, 20, 30], dtype=np.float32)
        ranges, values, valid = rasterize_static_points(
            points, intensity, sensor_origin_vehicle=np.zeros(3, dtype=np.float32),
            input_unit_rays=rays, input_shape=(1, 2), max_angular_distance_deg=1.0,
        )
        np.testing.assert_allclose(ranges, [[2, 3]])
        np.testing.assert_allclose(values, [[10, 30]])
        np.testing.assert_array_equal(valid, [[True, True]])

    def test_angular_gate_rejects_unrelated_point(self):
        _, _, valid = rasterize_static_points(
            np.array([[0, 1, 0]], dtype=np.float32), np.array([1], dtype=np.float32),
            sensor_origin_vehicle=np.zeros(3, dtype=np.float32),
            input_unit_rays=np.array([[1, 0, 0]], dtype=np.float32), input_shape=(1, 1),
            max_angular_distance_deg=1.0,
        )
        self.assertFalse(valid.any())


if __name__ == "__main__":
    unittest.main()
