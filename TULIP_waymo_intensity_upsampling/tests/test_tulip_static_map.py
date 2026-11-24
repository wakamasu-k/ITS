from __future__ import annotations

import numpy as np

from tulip_adapter.tulip_static_map import (
    fuse_16_to_32_observed_generated,
    valid_top_pixel_pose_mask,
    vehicle_to_world,
)


def test_fuse_preserves_observed_rows_and_uses_generated_prediction() -> None:
    observed = np.zeros((2, 16, 2650), dtype=np.float32)
    observed[0] = 10.0
    observed[1] = 0.25
    mask = np.ones((16, 2650), dtype=bool)
    prediction = np.zeros((2, 32, 2650), dtype=np.float32)
    prediction[0] = 20.0
    prediction[1] = 0.75

    fused, valid, stats = fuse_16_to_32_observed_generated(observed, mask, prediction)

    assert fused.shape == (2, 32, 2650)
    assert valid.shape == (32, 2650)
    assert np.all(fused[0, 0::2] == 10.0)
    assert np.all(fused[1, 0::2] == 0.25)
    assert np.all(fused[0, 1::2] == 20.0)
    assert np.all(fused[1, 1::2] == 0.75)
    assert valid.all()
    assert stats["observed_valid_pixels"] == 16 * 2650
    assert stats["generated_valid_pixels"] == 16 * 2650


def test_fuse_zeros_invalid_generated_values_and_clamps_negative_intensity() -> None:
    observed = np.zeros((2, 16, 2650), dtype=np.float32)
    observed[0] = 5.0
    mask = np.ones((16, 2650), dtype=bool)
    prediction = np.zeros((2, 32, 2650), dtype=np.float32)
    prediction[0] = 8.0
    prediction[1] = -0.5
    prediction[0, 1, 3] = -1.0
    prediction[0, 1, 4] = np.nan

    fused, valid, stats = fuse_16_to_32_observed_generated(observed, mask, prediction)

    assert not valid[1, 3]
    assert not valid[1, 4]
    assert fused[:, 1, 3].tolist() == [0.0, 0.0]
    assert fused[:, 1, 4].tolist() == [0.0, 0.0]
    assert valid[1, 5]
    assert fused[1, 1, 5] == 0.0
    assert stats["generated_negative_intensity_clamped"] == 16 * 2650 - 2


def test_fuse_removes_generated_range_above_75m_without_clipping_observed() -> None:
    observed = np.zeros((2, 16, 2650), dtype=np.float32)
    observed[0] = 100.0
    observed[1] = 0.25
    mask = np.ones((16, 2650), dtype=bool)
    prediction = np.zeros((2, 32, 2650), dtype=np.float32)
    prediction[0] = 75.0
    prediction[1] = 0.75
    prediction[0, 1, 3] = 75.01

    fused, valid, stats = fuse_16_to_32_observed_generated(observed, mask, prediction)

    assert valid[0, 3]
    assert fused[0, 0, 3] == 100.0
    assert valid[1, 2]
    assert fused[0, 1, 2] == 75.0
    assert not valid[1, 3]
    assert fused[:, 1, 3].tolist() == [0.0, 0.0]
    assert stats["generated_range_safety_removed_pixels"] == 1
    assert stats["generated_range_above_max_removed_pixels"] == 1


def test_valid_top_pixel_pose_mask_rejects_zero_world_pose() -> None:
    pose = np.broadcast_to(np.eye(4, dtype=np.float32), (32, 2650, 4, 4)).copy()
    pose[..., 0, 3] = 100.0
    pose[1, 3] = np.eye(4, dtype=np.float32)
    pose[1, 4, 0, 0] = np.nan

    valid = valid_top_pixel_pose_mask(pose)

    assert valid[0, 0]
    assert not valid[1, 3]
    assert not valid[1, 4]


def test_vehicle_to_world_applies_row_major_pose() -> None:
    points = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = [10.0, 20.0, 30.0]
    actual = vehicle_to_world(points, transform.reshape(-1))
    np.testing.assert_allclose(actual, np.array([[11.0, 22.0, 33.0]], dtype=np.float32))
