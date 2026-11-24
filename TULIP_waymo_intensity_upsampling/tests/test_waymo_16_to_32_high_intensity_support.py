"""Tests for support-stratified high-intensity evaluation helpers."""
from __future__ import annotations

import numpy as np

from tools.evaluate_waymo_16_to_32_high_intensity_support import support_groups


def test_support_groups_partition_generated_domain() -> None:
    """Exact, local-only and no-local masks are exclusive and complete."""
    target = np.zeros((4, 7, 2), dtype=np.float32)
    valid = np.ones((4, 7), dtype=bool)
    target[..., 0] = 10.0
    target[0, 2, 1] = 1200.0  # exact support for generated row 1, column 2
    target[2, 6, 1] = 1500.0  # local-only support for generated row 3, column 5
    groups = support_groups(
        target,
        valid,
        np.array([0, 2]),
        np.array([1, 3]),
        high_threshold=1000.0,
    )
    combined = sum(mask.astype(np.int8) for mask in groups.values())
    expected = np.zeros((4, 7), dtype=np.int8)
    expected[[1, 3]] = 1
    assert np.array_equal(combined, expected)
    assert groups["exact_support"][1, 2]
    assert groups["local_only_support"][3, 5]
    assert groups["no_local_support"][1, 4]
