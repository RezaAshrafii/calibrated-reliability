"""Train-only operating-regime scaling."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ALLOWED_FEATURE_PREFIXES = ("sensor_", "op_setting_")
ALLOWED_FEATURES = {"cycle", "cycle_index"}


class RegimeAwareScaler(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Scale features globally or by operating regime learned on training data."""

    def __init__(self, n_regimes: int | None = None, random_state: int = 13) -> None:
        self.n_regimes = n_regimes
        self.random_state = random_state
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: Any = None) -> RegimeAwareScaler:
        """Fit clustering and per-regime scalers on training data."""
        del y
        self._validate_input(X)
        self._global_only = False
        unexpected = {
            column
            for column in X.columns
            if column != "engine_id"
            and column not in ALLOWED_FEATURES
            and not column.startswith(ALLOWED_FEATURE_PREFIXES)
        }
        if unexpected:
            raise ValueError(f"Only approved feature columns are allowed: {sorted(unexpected)}")
        self.feature_columns_ = [column for column in X.columns if column != "engine_id"]
        if not self.feature_columns_:
            raise ValueError("No model features remain after removing engine_id")
        settings = X[["op_setting_1", "op_setting_2", "op_setting_3"]].astype("float64")
        self.setting_scaler_ = StandardScaler().fit(settings)
        scaled_settings = self.setting_scaler_.transform(settings)
        if len(pd.DataFrame(scaled_settings).drop_duplicates()) < 2:
            self._fit_global(X)
            return self
        maximum = min(6, len(X) - 1)
        if maximum < 2:
            raise ValueError("At least 3 training rows are required for regime fitting")
        candidates = (
            [self.n_regimes] if self.n_regimes is not None else list(range(2, maximum + 1))
        )
        if any(
            candidate is None or candidate < 2 or candidate > maximum for candidate in candidates
        ):
            raise ValueError("n_regimes must be between 2 and number of training rows minus one")
        scores: dict[int, float] = {}
        for candidate in candidates:
            assert candidate is not None
            model = KMeans(n_clusters=candidate, random_state=self.random_state, n_init=10).fit(
                scaled_settings
            )
            if len(set(model.labels_)) != candidate:
                continue
            scores[candidate] = float(silhouette_score(scaled_settings, model.labels_))
        if not scores:
            self._fit_global(X)
            return self
        self.n_regimes_ = max(scores, key=lambda key: scores[key])
        self.clusterer_ = KMeans(
            n_clusters=self.n_regimes_, random_state=self.random_state, n_init=10
        ).fit(scaled_settings)
        self.scalers_: dict[int, StandardScaler] = {}
        labels = self.clusterer_.labels_
        for regime in range(self.n_regimes_):
            self.scalers_[regime] = StandardScaler().fit(
                X.loc[labels == regime, self.feature_columns_]
            )
        self.feature_names_out_ = list(self.feature_columns_)
        self._fitted = True
        return self

    def _fit_global(self, X: pd.DataFrame) -> None:
        """Fit a global scaler for one-regime or invalid-cluster data."""
        self.global_scaler_ = StandardScaler().fit(X[self.feature_columns_])
        self.n_regimes_ = 1
        self.scalers_ = {}
        self._global_only = True
        self._fitted = True
        self.feature_names_out_ = list(self.feature_columns_)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using train-fitted regime assignments and scalers."""
        if not self._fitted:
            raise RuntimeError("RegimeAwareScaler must be fitted before transform")
        self._validate_input(X)
        values = X[self.feature_columns_].astype("float64")
        if getattr(self, "_global_only", False):
            return pd.DataFrame(
                self.global_scaler_.transform(values),
                columns=self.feature_columns_,
                index=X.index,
            ).reset_index(drop=True)
        settings = X[["op_setting_1", "op_setting_2", "op_setting_3"]].astype("float64")
        labels = self.clusterer_.predict(self.setting_scaler_.transform(settings))
        result = pd.DataFrame(index=X.index, columns=self.feature_columns_, dtype="float64")
        for regime, scaler in self.scalers_.items():
            mask = labels == regime
            if mask.any():
                result.loc[mask, self.feature_columns_] = scaler.transform(
                    values.loc[mask, self.feature_columns_]
                )
        return result.reset_index(drop=True)

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        """Validate regime settings and numeric feature columns."""
        required = {"op_setting_1", "op_setting_2", "op_setting_3"}
        if not required.issubset(X.columns):
            raise ValueError(
                f"Missing operating settings: {sorted(required.difference(X.columns))}"
            )
        if not all(pd.api.types.is_numeric_dtype(X[column]) for column in X.columns):
            raise ValueError("All scaling features must be numeric")
