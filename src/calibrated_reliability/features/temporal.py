"""Past-only temporal feature construction for engine trajectories."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from calibrated_reliability.data.loader import COLUMNS, OP_SETTING_COLUMNS, SENSOR_COLUMNS

RAW_FEATURE_COLUMNS = frozenset(COLUMNS)


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
        sensors = [column for column in SENSOR_COLUMNS if column in X.columns]
        if not sensors:
            raise ValueError("No sensor columns found")
        variances = X[sensors].astype("float64").var(ddof=0)
        self.sensor_columns_ = [
            column for column in sensors if variances[column] > self.variance_threshold
        ]
        if not self.sensor_columns_:
            raise ValueError("All sensors were removed by variance threshold")
        self.feature_names_in_ = list(X.columns)
        self.feature_names_out_ = self._output_feature_names()
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
        missing_input = set(self.feature_names_in_).difference(X.columns)
        extra_input = set(X.columns).difference(self.feature_names_in_)
        if missing_input or extra_input:
            raise ValueError(
                "Raw feature schema differs from fit; "
                f"missing={sorted(missing_input)}, extra={sorted(extra_input)}"
            )
        frame = X.sort_values(["engine_id", "cycle"], kind="stable").copy()
        groups = frame.groupby("engine_id", sort=False)
        positions = groups.cumcount().astype("float64")
        output: dict[str, pd.Series] = {
            "engine_id": frame["engine_id"],
            "cycle": frame["cycle"],
        }
        for column in [*OP_SETTING_COLUMNS, *self.sensor_columns_]:
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
                output[f"{sensor}_rolling_slope_{window}"] = self._rolling_slope(
                    frame["engine_id"], frame[sensor], positions, window
                )
        output["cycle_index"] = frame["cycle"].astype("float64")
        return pd.DataFrame(output, index=frame.index).reset_index(drop=True)

    def _output_feature_names(self) -> list[str]:
        """Return the deterministic output schema learned during fit."""
        names = ["engine_id", "cycle", *OP_SETTING_COLUMNS, *self.sensor_columns_]
        for sensor in self.sensor_columns_:
            names.append(f"{sensor}_delta_1")
            for window in self.windows:
                names.extend(
                    [
                        f"{sensor}_rolling_mean_{window}",
                        f"{sensor}_rolling_std_{window}",
                        f"{sensor}_rolling_slope_{window}",
                    ]
                )
        names.append("cycle_index")
        return names

    @staticmethod
    def _rolling_slope(
        engine_ids: pd.Series,
        values: pd.Series,
        positions: pd.Series,
        window: int,
    ) -> pd.Series:
        """Compute past-only least-squares slopes with rolling sufficient statistics."""
        work = pd.DataFrame(
            {
                "engine_id": engine_ids,
                "y": values.astype("float64"),
                "x": positions,
            }
        )
        work["xy"] = work["x"] * work["y"]
        work["x2"] = work["x"] * work["x"]
        rolling = work.groupby("engine_id", sort=False).rolling(window, min_periods=1)
        count = rolling["y"].count().reset_index(level=0, drop=True)
        sum_y = rolling["y"].sum().reset_index(level=0, drop=True)
        sum_x = rolling["x"].sum().reset_index(level=0, drop=True)
        sum_xy = rolling["xy"].sum().reset_index(level=0, drop=True)
        sum_x2 = rolling["x2"].sum().reset_index(level=0, drop=True)
        numerator = count * sum_xy - sum_x * sum_y
        denominator = count * sum_x2 - sum_x * sum_x
        return numerator.div(denominator.where(denominator != 0, 1.0)).where(denominator != 0, 0.0)

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        """Validate the minimum trajectory schema."""
        if not all(isinstance(column, str) for column in X.columns):
            raise ValueError("All feature names must be strings")
        unexpected = set(X.columns).difference(RAW_FEATURE_COLUMNS)
        if unexpected:
            raise ValueError(f"Unexpected raw feature columns: {sorted(unexpected)}")
        required = {"engine_id", "cycle", *OP_SETTING_COLUMNS}
        if not required.issubset(X.columns):
            raise ValueError(f"Missing feature columns: {sorted(required.difference(X.columns))}")
