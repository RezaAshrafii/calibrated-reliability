"""Leakage and correctness tests for Phase 5 features."""

import pandas as pd
import pytest

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
    scaler = RegimeAwareScaler(n_regimes=4).fit(frame)
    result = scaler.transform(frame)
    assert scaler.n_regimes_ == 1
    assert result.notna().all().all()
