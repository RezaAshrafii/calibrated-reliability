"""Tests for finite-sample split-conformal intervals and C02 orchestration."""

import numpy as np
import pandas as pd
import pytest

from calibrated_reliability.data.loader import COLUMNS
from calibrated_reliability.evaluation.conformal import (
    conformal_quantile,
    interval_metrics,
    split_conformal_intervals,
)
from calibrated_reliability.experiments.c02 import C02Config, run_c02


def config_text() -> str:
    """Return a small C02 configuration for synthetic tests."""
    return """
experiment_id: C02
source: FD001
target: FD001
evaluation_unit: engine_endpoint
seeds: [13]
alphas: [0.10, 0.05]
rul_cap: 125
preprocessing:
  temporal_windows: [2]
  variance_threshold: 0.0
calibration:
  min_observed_cycles: 2
  lower_fraction: 0.40
  upper_fraction: 0.90
"""


def trajectory(engine_count: int, cycles: int) -> pd.DataFrame:
    """Create exact-schema synthetic trajectories."""
    rows: list[dict[str, float | int]] = []
    for engine_id in range(1, engine_count + 1):
        for cycle in range(1, cycles + 1):
            row: dict[str, float | int] = {
                "engine_id": engine_id,
                "cycle": cycle,
                "op_setting_1": float(engine_id % 2),
                "op_setting_2": 0.0,
                "op_setting_3": 0.0,
            }
            for sensor in range(1, 22):
                row[f"sensor_{sensor}"] = float(engine_id + cycle + sensor)
            rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def test_conformal_quantile_uses_finite_sample_rank() -> None:
    """The quantile is the declared order statistic, not an interpolated percentile."""
    assert conformal_quantile([1.0, 2.0, 5.0], alpha=0.10) == 5.0
    assert conformal_quantile([1.0, 2.0, 5.0], alpha=0.50) == 2.0
    with pytest.raises(ValueError, match="strictly between"):
        conformal_quantile([1.0], alpha=1.0)


def test_intervals_and_metrics_are_finite_and_directionally_correct() -> None:
    """Intervals are symmetric and interval score penalizes misses."""
    lower, upper, q = split_conformal_intervals([10.0, 12.0], [9.0, 13.0], [11.0], 0.10)
    assert q == 1.0
    np.testing.assert_allclose(lower, [10.0])
    np.testing.assert_allclose(upper, [12.0])
    metrics = interval_metrics([15.0], lower, upper, 0.10, 125.0)
    assert metrics["coverage"] == 0.0
    assert metrics["mean_width"] == 2.0
    assert metrics["normalized_interval_score"] > 0.0


def test_c02_uses_full_training_rul_at_truncated_calibration_endpoint() -> None:
    """Calibration truth is remaining life, not zero at the observed cut point."""
    result = run_c02(
        trajectory(10, 5),
        trajectory(2, 3),
        pd.DataFrame({"rul": [145, 10]}),
        C02Config.from_yaml(config_text()),
        seed=13,
    )
    assert len(result.predictions) == 2
    assert list(result.predictions["y_true"]) == [125.0, 10.0]
    assert set(result.quantiles) == {
        "mean_alpha_0.1",
        "mean_alpha_0.05",
        "ridge_alpha_0.1",
        "ridge_alpha_0.05",
        "hist_gradient_boosting_alpha_0.1",
        "hist_gradient_boosting_alpha_0.05",
    }
    assert all(np.isfinite(value) for value in result.quantiles.values())
    assert all(set(bounds) == {"alpha_0.1", "alpha_0.05"} for bounds in result.metrics.values())
