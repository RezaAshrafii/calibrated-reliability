"""Past-only temporal feature construction for engine trajectories."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TemporalFeatureTransformer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Create current, lag, rolling, and trajectory-index features.

    ``fit`` learns only the low-variance sensor list from the supplied training
    frame. ``transform`` computes rolling statistics in engine/cycle order and
    never accesses a future row; callers must provide data truncated at any
    calibration or evaluation cut point.
    """

    def __init__(
        self,
        windows: tuple[int, ...] = (5, 10, 20),
        variance_threshold: float = 0.0,
    ) -> None:
        if not windows or any(not isinstance(window, int) or window < 1 for window in windows):
            raise ValueError("windows must contain positive integers")
        if variance_threshold < 0:
            raise ValueError("variance_threshold must be nonnegative")
        self.windows = windows
        self.variance_threshold = variance_threshold
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: Any = None) -> TemporalFeatureTransformer:
        """Learn the sensor list and low-variance exclusions from training data."""
        del y
        self._validate_input(X)
        sensors = [column for column in X.columns if str(column).startswith("sensor_")]
        if not sensors:
            raise ValueError("No sensor columns found")
        variances = X[sensors].astype("float64").var(ddof=0)
        self.sensor_columns_ = [
            column for column in sensors if variances[column] > self.variance_threshold
        ]
        if not self.sensor_columns_:
            raise ValueError("All sensors were removed by variance threshold")
        self.feature_names_in_ = list(X.columns)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform trajectories using only current and past observations."""
        if not self._fitted:
            raise RuntimeError("TemporalFeatureTransformer must be fitted before transform")
        self._validate_input(X)
        missing = set(self.sensor_columns_).difference(X.columns)
        if missing:
            raise ValueError(f"Missing fitted columns: {sorted(missing)}")
        frame = X.sort_values(["engine_id", "cycle"], kind="stable").copy()
        groups = frame.groupby("engine_id", sort=False)
        output = frame[["engine_id", "cycle"]].copy()
        for column in ["op_setting_1", "op_setting_2", "op_setting_3", *self.sensor_columns_]:
            output[column] = frame[column].astype("float64")
        for sensor in self.sensor_columns_:
            output[f"{sensor}_delta_1"] = groups[sensor].diff().fillna(0.0)
            for window in self.windows:
                rolling = groups[sensor].rolling(window, min_periods=1)
                output[f"{sensor}_rolling_mean_{window}"] = rolling.mean().reset_index(
                    level=0, drop=True
                )
                output[f"{sensor}_rolling_std_{window}"] = (
                    rolling.std(ddof=0).fillna(0.0).reset_index(level=0, drop=True)
                )
                output[f"{sensor}_rolling_slope_{window}"] = rolling.apply(
                    self._slope, raw=True
                ).reset_index(level=0, drop=True)
        output["cycle_index"] = frame["cycle"].astype("float64")
        self.feature_names_out_ = list(output.columns)
        return output.reset_index(drop=True)

    @staticmethod
    def _slope(values: Any) -> float:
        """Compute a least-squares slope for one past-only rolling window."""
        if len(values) < 2:
            return 0.0
        x = pd.Series(range(len(values)), dtype="float64")
        y = pd.Series(values, dtype="float64")
        denominator = float(((x - x.mean()) ** 2).sum())
        return (
            0.0
            if denominator == 0.0
            else float(((x - x.mean()) * (y - y.mean())).sum() / denominator)
        )

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        """Validate the minimum trajectory schema."""
        required = {"engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"}
        if not required.issubset(X.columns):
            raise ValueError(f"Missing feature columns: {sorted(required.difference(X.columns))}")
