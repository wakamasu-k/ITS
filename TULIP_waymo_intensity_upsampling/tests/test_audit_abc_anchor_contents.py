"""Small deterministic tests for full-anchor audit metric definitions."""
import unittest
import numpy as np
from tools.audit_abc_anchor_contents import compare, aggregate


class AuditMetricsTest(unittest.TestCase):
    def test_black_covered_pixel_is_valid(self):
        image = np.zeros((1, 2), dtype=np.uint8)
        mask = np.array([[True, False]])
        depth = np.array([[2., np.inf]])
        with np.errstate(invalid='raise'):
            result = compare(image, depth, mask, image, depth, mask)
        self.assertEqual(result['common_pixels'], 1)
        self.assertEqual(result['geometry_iou'], 1)
        self.assertEqual(result['png_mae_common'], 0)

    def test_depth_and_image_metrics_are_separate(self):
        mask = np.array([[True, True]])
        image = np.array([[0, 100]], dtype=np.uint8)
        ref = np.array([[10, 80]], dtype=np.uint8)
        result = compare(image, np.array([[2., 5.]]), mask,
                         ref, np.array([[2., 3.]]), mask)
        self.assertEqual(result['depth_mae_m'], 1)
        self.assertEqual(result['depth_over_1m_fraction'], .5)
        self.assertEqual(result['png_mae_common'], 15)
        self.assertEqual(result['png_mae_depth_within_1m'], 10)

    def test_empty_masks_are_not_perfect_matches(self):
        image = np.zeros((1, 1), dtype=np.uint8)
        mask = np.zeros((1, 1), dtype=bool)
        result = compare(image, image, mask, image, image, mask)
        self.assertIsNone(result['geometry_iou'])
        self.assertIsNone(result['depth_mae_m'])
        self.assertEqual(aggregate([])['anchor_count'], 0)


if __name__ == '__main__':
    unittest.main()
