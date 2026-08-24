"""Leakage and correctness tests for Phase 5 features."""

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from calibrated_reliability.features.regime import RegimeAwareScaler
from calibrated_reliability.features.temporal import TemporalFeatureTransformer


def toy_frame() -> pd.DataFrame:
    """Create two short trajectories with operating settings and sensors."""
    return pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "cycle": [1, 2, 3, 4, 1, 2, 3, 4],
            "op_setting_1": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            "op_setting_2": 0.0,
            "op_setting_3": 0.0,
            "sensor_1": [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0],
            "sensor_2": 1.0,
        }
    )


def test_temporal_transform_requires_fit() -> None:
    """Transform before fit is rejected."""
    with pytest.raises(RuntimeError, match="fitted"):
        TemporalFeatureTransformer().transform(toy_frame())


def test_temporal_features_cannot_reconstruct_rul_from_lifetime() -> None:
    """No feature encodes full run-to-failure trajectory length."""
    transformed = TemporalFeatureTransformer(windows=(2,)).fit(toy_frame()).transform(toy_frame())
    assert "cycle_ratio" not in transformed.columns
    assert "observation_horizon" not in transformed.columns


def test_temporal_features_are_past_only_and_train_selected() -> None:
    """Constant sensor removal is train-only and first delta has no past value."""
    frame = toy_frame()
    transformer = TemporalFeatureTransformer(windows=(2,), variance_threshold=0.0).fit(
        frame.iloc[:4]
    )
    transformed = transformer.transform(frame.iloc[:4])
    assert "sensor_2" not in transformed.columns
    assert transformed.loc[0, "sensor_1_delta_1"] == 0.0
    assert transformed.loc[1, "sensor_1_delta_1"] == 1.0
    assert transformed.loc[0, "sensor_1_rolling_mean_2"] == 1.0


def test_temporal_transform_does_not_use_future_row() -> None:
    """Adding a future row does not change earlier rolling features."""
    transformer = TemporalFeatureTransformer(windows=(3,)).fit(toy_frame().iloc[:4])
    short = transformer.transform(toy_frame().iloc[:3])
    long = transformer.transform(toy_frame().iloc[:4]).iloc[:3]
    pd.testing.assert_frame_equal(short, long, check_dtype=False)


def test_regime_scaler_requires_fit_and_handles_unseen_settings() -> None:
    """Regime scaler is train-fitted and predicts a regime for new settings."""
    frame = toy_frame()
    scaler = RegimeAwareScaler(n_regimes=2)
    with pytest.raises(RuntimeError, match="fitted"):
        scaler.transform(frame)
    scaler.fit(frame)
    new = frame.iloc[:2].copy()
    new["op_setting_1"] = 99.0
    result = scaler.transform(new)
    assert result.shape == (2, len(frame.columns) - 1)
    assert "engine_id" not in result.columns
    assert result.notna().all().all()


def test_regime_scaler_rejects_targets_and_falls_back_for_one_regime() -> None:
    """Labels are rejected and single-condition data uses global scaling."""
    frame = toy_frame()
    with pytest.raises(ValueError, match="approved feature columns"):
        RegimeAwareScaler(n_regimes=2).fit(frame.assign(rul_raw=1.0))
    with pytest.raises(ValueError, match="approved feature columns"):
        RegimeAwareScaler(n_regimes=2).fit(frame.assign(rul=1.0))
    one_regime = frame.copy()
    one_regime["op_setting_1"] = 0.0
    scaler = RegimeAwareScaler().fit(one_regime)
    result = scaler.transform(one_regime)
    assert result.shape[0] == len(one_regime)
    assert result.notna().all().all()


@pytest.mark.parametrize(
    "column",
    [
        "rul",
        "rul_raw",
        "y_true",
        "target_rul",
        "remaining_useful_life",
        "sensor_rul",
        "sensor_1_target",
        "sensor_22",
        "op_setting_target",
    ],
)
def test_temporal_rejects_columns_outside_raw_schema(column: str) -> None:
    """Labels and prefix collisions cannot masquerade as raw C-MAPSS features."""
    contaminated = toy_frame().assign(**{column: range(8)})
    with pytest.raises(ValueError, match="Unexpected raw feature columns"):
        TemporalFeatureTransformer(windows=(2,)).fit(contaminated)


@pytest.mark.parametrize(
    "column",
    [
        "y_true",
        "target_rul",
        "remaining_useful_life",
        "sensor_rul",
        "sensor_1_target",
        "sensor_22",
        "op_setting_target",
    ],
)
def test_regime_scaler_rejects_unknown_and_prefix_collision_columns(column: str) -> None:
    """The scaler allowlist is exact rather than prefix-based."""
    contaminated = toy_frame().assign(**{column: range(8)})
    with pytest.raises(ValueError, match="approved feature columns"):
        RegimeAwareScaler(n_regimes=2).fit(contaminated)


def test_missing_fitted_sensor_is_actionable() -> None:
    """Transform reports a schema error when a fitted sensor is absent."""
    frame = toy_frame()
    transformer = TemporalFeatureTransformer().fit(frame)
    with pytest.raises(ValueError, match="Missing fitted columns"):
        transformer.transform(frame.drop(columns=["sensor_1"]))


def test_delta_resets_at_engine_boundary() -> None:
    """The first row of every engine has zero delta."""
    transformed = TemporalFeatureTransformer(windows=(2,)).fit(toy_frame()).transform(toy_frame())
    assert transformed.loc[0, "sensor_1_delta_1"] == 0.0
    assert transformed.loc[4, "sensor_1_delta_1"] == 0.0


def test_invalid_requested_regime_count_falls_back() -> None:
    """A requested cluster count that cannot be realized uses global scaling."""
    frame = toy_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        scaler = RegimeAwareScaler(n_regimes=4).fit(frame)
    result = scaler.transform(frame)
    assert scaler.n_regimes_ == 1
    assert scaler.fallback_reason_ == "no_valid_clustering"
    assert result.notna().all().all()


def test_temporal_statistics_match_manual_values_and_isolate_engines() -> None:
    """Rolling population standard deviation and slope match hand calculations."""
    frame = toy_frame()
    frame.loc[:3, "sensor_1"] = [1.0, 2.0, 4.0, 8.0]
    transformed = TemporalFeatureTransformer(windows=(3,)).fit(frame).transform(frame)
    assert transformed.loc[2, "sensor_1_rolling_std_3"] == pytest.approx(
        np.std([1.0, 2.0, 4.0], ddof=0)
    )
    assert transformed.loc[2, "sensor_1_rolling_slope_3"] == pytest.approx(1.5)
    assert transformed.loc[4, "sensor_1_rolling_mean_3"] == 10.0
    assert transformed.loc[4, "sensor_1_rolling_std_3"] == 0.0
    assert transformed.loc[4, "sensor_1_rolling_slope_3"] == 0.0


def test_temporal_and_scaler_enforce_fitted_schema() -> None:
    """Fit-time feature schemas reject missing or additional transform columns."""
    frame = toy_frame()
    temporal = TemporalFeatureTransformer(windows=(2,)).fit(frame)
    with pytest.raises(ValueError, match="Raw feature schema differs from fit"):
        temporal.transform(frame.assign(sensor_3=2.0))
    features = temporal.transform(frame)
    scaler = RegimeAwareScaler(n_regimes=2).fit(features)
    with pytest.raises(ValueError, match="Feature schema differs from fit"):
        scaler.transform(features.drop(columns=["sensor_1_delta_1"]))
    with pytest.raises(ValueError, match="approved feature columns"):
        scaler.transform(features.assign(target_rul=1.0))


def test_valid_temporal_output_is_accepted_by_exact_allowlist() -> None:
    """Every formally generated temporal feature passes the scaler contract."""
    frame = toy_frame()
    temporal = TemporalFeatureTransformer(windows=(2, 3)).fit(frame)
    features = temporal.transform(frame)
    result = RegimeAwareScaler(n_regimes=2).fit(features).transform(features)
    assert list(result.columns) == [column for column in features if column != "engine_id"]


def test_regime_selection_is_deterministic_and_transform_does_not_refit() -> None:
    """Automatic clustering is repeatable and transform preserves fitted state."""
    frame = toy_frame()
    first = RegimeAwareScaler(random_state=13).fit(frame)
    second = RegimeAwareScaler(random_state=13).fit(frame)
    assert first.n_regimes_ == second.n_regimes_
    np.testing.assert_allclose(
        first.clusterer_.cluster_centers_, second.clusterer_.cluster_centers_
    )
    centers = first.clusterer_.cluster_centers_.copy()
    setting_mean = first.setting_scaler_.mean_.copy()
    scaler_means = {regime: scaler.mean_.copy() for regime, scaler in first.scalers_.items()}
    first.transform(frame.iloc[:2])
    np.testing.assert_array_equal(first.clusterer_.cluster_centers_, centers)
    np.testing.assert_array_equal(first.setting_scaler_.mean_, setting_mean)
    for regime, mean in scaler_means.items():
        np.testing.assert_array_equal(first.scalers_[regime].mean_, mean)


def test_feature_schema_survives_serialization() -> None:
    """Fitted feature names and transformations survive a pickle round-trip."""
    frame = toy_frame()
    temporal = TemporalFeatureTransformer(windows=(2,)).fit(frame)
    assert temporal.feature_names_out_ == list(temporal.transform(frame).columns)
    restored_temporal = pickle.loads(pickle.dumps(temporal))
    transformed = restored_temporal.transform(frame)
    assert restored_temporal.feature_names_out_ == list(transformed.columns)
    scaler = RegimeAwareScaler(n_regimes=2).fit(transformed)
    expected = scaler.transform(transformed)
    restored_scaler = pickle.loads(pickle.dumps(scaler))
    pd.testing.assert_frame_equal(restored_scaler.transform(transformed), expected)
    assert restored_scaler.feature_names_out_ == list(expected.columns)


@pytest.mark.parametrize("windows", [(), (0,), (-1,), (2, 0)])
def test_temporal_rejects_invalid_windows(windows: tuple[int, ...]) -> None:
    """Rolling windows must be non-empty positive integers."""
    with pytest.raises(ValueError, match="positive integers"):
        TemporalFeatureTransformer(windows=windows)
