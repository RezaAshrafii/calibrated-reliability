"""Tests for C05 weighted conformal design and leakage boundaries."""

import numpy as np
import pytest

from calibrated_reliability.experiments.c05 import (
    C05Config,
    _clip_and_normalize_weights,
    _density_ratio,
    _weighted_quantile,
)


def c05_text() -> str:
    return """
experiment_id: C05
source: FD001
targets: [FD002, FD003, FD004]
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
weighting:
  method: logistic_density_ratio
  features: [op_setting_1, op_setting_2, op_setting_3]
  logistic_c: 1.0
  max_iter: 1000
  clip_min: 0.05
  clip_max: 1.0
bootstrap:
  n_resamples: 2000
  confidence_level: 0.95
  seed_policy: experiment_seed
"""


def test_c05_config_is_strict_and_preregistered() -> None:
    config = C05Config.from_yaml(c05_text())
    assert config.targets == ("FD002", "FD003", "FD004")
    with pytest.raises(ValueError):
        C05Config.from_yaml(c05_text().replace("logistic_density_ratio", "mean_ratio"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("logistic_c", "2.0"),
        ("max_iter", "500"),
        ("clip_min", "0.10"),
        ("clip_max", "0.90"),
        ("clip_min", "true"),
        ("clip_max", "true"),
    ],
)
def test_c05_rejects_changes_to_frozen_weighting_design(field: str, value: str) -> None:
    declared = {
        "logistic_c": "1.0",
        "max_iter": "1000",
        "clip_min": "0.05",
        "clip_max": "1.0",
    }
    with pytest.raises(ValueError):
        C05Config.from_yaml(
            c05_text().replace(
                f"{field}: {declared[field]}",
                f"{field}: {value}",
            )
        )


def test_weighted_quantile_uses_calibration_and_test_weight() -> None:
    assert _weighted_quantile([1.0, 2.0, 5.0], [1.0, 1.0, 1.0], 1.0, 0.30) == 5.0
    assert _weighted_quantile([1.0, 2.0, 5.0], [10.0, 1.0, 1.0], 1.0, 0.50) == 1.0


def test_weighted_quantile_is_pointwise_in_test_weight() -> None:
    scores = np.array([1.0, 2.0, 3.0])
    weights = np.ones(3)
    assert _weighted_quantile(scores, weights, 0.01, 0.50) == 2.0
    assert _weighted_quantile(scores, weights, 100.0, 0.50) == float("inf")


def test_weighted_quantile_returns_infinity_for_test_mass_boundary() -> None:
    assert np.isinf(_weighted_quantile([1.0, 2.0], [1.0, 1.0], 100.0, 0.10))


def test_clipped_weights_keep_fixed_design_quantiles_finite() -> None:
    calibration, target = _clip_and_normalize_weights(
        np.array([0.001, 100.0, *([1.0] * 18)]), np.array([1000.0]), 0.05, 1.0
    )
    assert calibration.mean() == pytest.approx(1.0)
    assert target.max() <= 1.0
    assert np.isfinite(_weighted_quantile(np.arange(20.0), calibration, target[0], 0.05))


def test_density_ratio_uses_only_operating_settings() -> None:
    import pandas as pd

    source = pd.DataFrame(
        {"op_setting_1": [0.0, 0.0, 1.0, 1.0], "op_setting_2": 0.0, "op_setting_3": 0.0}
    )
    target = pd.DataFrame(
        {"op_setting_1": [1.0, 1.0, 1.0, 1.0], "op_setting_2": 0.0, "op_setting_3": 0.0}
    )
    source_weights, target_weights = _density_ratio(source, target)
    assert np.isfinite(source_weights).all()
    assert np.isfinite(target_weights).all()
    assert target_weights.mean() > source_weights[source["op_setting_1"].to_numpy() == 0].mean()


def test_density_ratio_ignores_labels_and_non_weighting_columns() -> None:
    import pandas as pd

    source = pd.DataFrame(
        {
            "op_setting_1": [0.0, 1.0, 0.0, 1.0],
            "op_setting_2": [0.0] * 4,
            "op_setting_3": [0.0] * 4,
            "rul": [0.0, 0.0, 0.0, 0.0],
            "sensor_1": [1.0] * 4,
        }
    )
    target = source.copy()
    target["rul"] = [999.0] * 4
    target["sensor_1"] = [-999.0] * 4
    source_weights, target_weights = _density_ratio(source, target)
    clean_source, clean_target = _density_ratio(source.iloc[:, :3], target.iloc[:, :3])
    np.testing.assert_allclose(source_weights, clean_source)
    np.testing.assert_allclose(target_weights, clean_target)
