"""Tests for C03 conformalized quantile regression."""

import numpy as np
import pandas as pd
import pytest

from calibrated_reliability.data.loader import COLUMNS
from calibrated_reliability.evaluation.conformal import cqr_conformity_quantile, cqr_intervals
from calibrated_reliability.experiments.c03 import C03Config, run_c03


def config_text() -> str:
    return """
experiment_id: C03
source: FD001
target: FD001
evaluation_unit: engine_endpoint
seeds: [13, 37, 73, 101, 137]
alphas: [0.10, 0.05]
rul_cap: 125
preprocessing:
  temporal_windows: [5, 10, 20]
  variance_threshold: 0.0
calibration:
  min_observed_cycles: 30
  lower_fraction: 0.40
  upper_fraction: 0.90
bootstrap:
  n_resamples: 2000
  confidence_level: 0.95
  seed_policy: experiment_seed
models:
  max_iter: 50
  learning_rate: 0.05
  max_leaf_nodes: 31
  l2_regularization: 1.0
"""


def trajectory(engine_count: int, cycles: int) -> pd.DataFrame:
    rows = []
    for engine_id in range(1, engine_count + 1):
        for cycle in range(1, cycles + 1):
            row = {"engine_id": engine_id, "cycle": cycle}
            row.update({f"op_setting_{i}": float(i == 1) for i in range(1, 4)})
            row.update({f"sensor_{i}": float(engine_id + cycle + i) for i in range(1, 22)})
            rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def test_cqr_quantile_uses_nonnegative_conformity_score() -> None:
    assert cqr_conformity_quantile([10, 20, 30], [9, 19, 31], [11, 21, 32], 0.1) == 1.0
    lower, upper, q = cqr_intervals([10, 20, 30], [9, 19, 31], [11, 21, 32], [8], [12], 0.1)
    assert q == 1.0
    np.testing.assert_allclose(lower, [7.0])
    np.testing.assert_allclose(upper, [13.0])


def test_c03_is_frozen_and_uses_full_calibration_truth() -> None:
    config = C03Config.from_yaml(config_text())
    assert config.as_dict()["bootstrap"]["seed_policy"] == "experiment_seed"
    result = run_c03(
        trajectory(10, 100), trajectory(2, 50), pd.DataFrame({"rul": [145, 10]}), config, 13
    )
    assert len(result.predictions) == 2
    assert result.predictions.engine_id.tolist() == [1, 2]
    assert result.calibration_scores.rul_raw.gt(0).all()
    assert result.calibration_scores.rul_capped.le(125).all()
    assert set(result.metrics) == {"alpha_0.1", "alpha_0.05"}
    assert all(np.isfinite(value) for value in result.quantiles.values())
    for key in result.metrics:
        assert result.metrics[key]["bootstrap_ci"]


def test_c03_rejects_unknown_configuration_fields() -> None:
    with pytest.raises(ValueError, match="schema"):
        C03Config.from_yaml(config_text().replace("models:\n", "unknown: 1\nmodels:\n"))
