import unittest

import numpy as np

from tools.evaluate_waymo_16_to_32_controlled_pointcloud import shared_valid_mask, valid_signal_mask


class ControlledPointcloudMaskTest(unittest.TestCase):
    def test_shared_mask_requires_every_method_without_changing_target_domain(self):
        target = np.array([[True, True], [False, True]])
        good = np.array([[[1, 1], [1, 1]], [[2, 2], [2, 2]]], dtype=np.float32)
        invalid = good.copy()
        invalid[0, 0, 1] = -1
        masks, shared = shared_valid_mask(target, {"tulip": good, "nearest": good, "linear": invalid})
        np.testing.assert_array_equal(masks["tulip"], target)
        np.testing.assert_array_equal(masks["linear"], [[True, False], [False, True]])
        np.testing.assert_array_equal(shared, [[True, False], [False, True]])

    def test_valid_signal_mask_rejects_nonfinite_intensity(self):
        signal = np.array([[[1]], [[np.nan]]], dtype=np.float32)
        self.assertFalse(valid_signal_mask(signal, np.array([[True]])).any())


if __name__ == "__main__":
    unittest.main()
