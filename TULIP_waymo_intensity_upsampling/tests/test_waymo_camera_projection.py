"""Unit tests for the Waymo camera projection diagnostic."""
from __future__ import annotations

import numpy as np

from tools.evaluate_waymo_16_to_32_camera_projection import project_vehicle_to_camera


class _Extrinsic:
    transform = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


class _Calibration:
    intrinsic = [100.0, 100.0, 320.0, 240.0]
    extrinsic = _Extrinsic()


def test_waymo_camera_projection_uses_x_forward_and_axis_signs() -> None:
    """Identity calibration maps X-forward points with Waymo signs."""
    points = np.asarray([[10.0, 0.0, 0.0], [10.0, -1.0, -2.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    u, v, depth, front = project_vehicle_to_camera(points, _Calibration())
    np.testing.assert_allclose(u[:2], [320.0, 330.0])
    np.testing.assert_allclose(v[:2], [240.0, 260.0])
    np.testing.assert_allclose(depth, [10.0, 10.0, -1.0])
    assert front.tolist() == [True, True, False]
