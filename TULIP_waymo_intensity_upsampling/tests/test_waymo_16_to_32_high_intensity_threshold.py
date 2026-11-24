"""Unit tests for validation-only high-intensity threshold calibration."""
from __future__ import annotations

import pytest
import torch

from tools.calibrate_waymo_16_to_32_high_intensity_threshold import (
    best_f1_candidate,
    metrics_from_counts,
    parse_thresholds,
    threshold_counts,
)


def test_parse_thresholds_sorts_and_deduplicates() -> None:
    """Candidate thresholds form a deterministic raw-value sweep."""
    assert parse_thresholds("1000, 300, 1000, 500") == (300.0, 500.0, 1000.0)


def test_parse_thresholds_rejects_negative_values() -> None:
    """Negative raw-intensity thresholds are not meaningful."""
    with pytest.raises(ValueError, match="non-negative"):
        parse_thresholds("100,-1")


def test_threshold_counts_uses_fixed_target_and_swept_prediction_cutoff() -> None:
    """Target and prediction thresholds intentionally have different roles."""
    target = torch.tensor([[[1200.0, 1000.0, 900.0, 10.0]]])
    prediction = torch.tensor([[[800.0, 1500.0, 1200.0, 1100.0]]])
    mask = torch.tensor([[[True, True, True, False]]])
    counts = threshold_counts(
        target, prediction, mask, target_threshold=1000.0, prediction_threshold=750.0
    )
    assert counts == {
        "target_positive": 2,
        "predicted_positive": 3,
        "true_positive": 2,
        "false_positive": 1,
        "false_negative": 0,
    }


def test_best_f1_candidate_is_deterministic() -> None:
    """F1 ties prefer precision, then a lower prediction threshold."""
    rows = [
        {"prediction_threshold": 1000.0, "precision": 0.8, "f1": 0.5},
        {"prediction_threshold": 500.0, "precision": 0.8, "f1": 0.5},
        {"prediction_threshold": 750.0, "precision": 0.9, "f1": 0.5},
    ]
    assert best_f1_candidate(rows)["prediction_threshold"] == 750.0
    assert metrics_from_counts(
        {"true_positive": 2, "false_positive": 1, "false_negative": 1}
    ) == pytest.approx({"precision": 2.0 / 3.0, "recall": 2.0 / 3.0, "f1": 2.0 / 3.0})
