"""Deterministic point-prediction baselines for C-MAPSS RUL."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from calibrated_reliability.features.regime import (
    ALLOWED_BASE_FEATURES,
    DERIVED_SENSOR_PATTERN,
)


def _validate_features(X: pd.DataFrame) -> None:
    """Reject identifiers, labels, nonnumeric values, and non-finite features."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Model features must be a pandas DataFrame")
    if "engine_id" in X.columns:
        raise ValueError("engine_id is metadata and cannot be a model feature")
    unexpected = {
        column
        for column in X.columns
        if column not in ALLOWED_BASE_FEATURES and DERIVED_SENSOR_PATTERN.fullmatch(column) is None
    }
    if unexpected:
        raise ValueError(f"Only approved model features are allowed: {sorted(unexpected)}")
    if len(X.columns) == 0:
        raise ValueError("At least one model feature is required")
    if not all(pd.api.types.is_numeric_dtype(X[column]) for column in X.columns):
        raise ValueError("All model features must be numeric")
    if not all(math.isfinite(float(value)) for value in X.to_numpy(dtype="float64").ravel()):
        raise ValueError("Model features must be finite")


def _validate_target(y: Any) -> Any:
    """Validate and return a finite one-dimensional numeric target."""
    values = pd.to_numeric(pd.Series(y), errors="raise").to_numpy(dtype="float64")
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("RUL target must be a non-empty one-dimensional array")
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("RUL target must be finite")
    return values


def build_baseline_models(
    random_state: int = 13,
    ridge_alpha: float = 1.0,
    hgb_max_iter: int = 50,
    hgb_learning_rate: float = 0.05,
    hgb_max_leaf_nodes: int = 31,
    hgb_l2_regularization: float = 1.0,
) -> dict[str, Any]:
    """Build the fixed C01 baseline model set.

    ``mean`` is a lower-bound sanity baseline, ``ridge`` is a linear baseline,
    and ``hist_gradient_boosting`` is a nonlinear tree baseline. No tuning is
    performed here; all estimators use only their declared parameters.
    """
    return {
        "mean": DummyRegressor(strategy="mean"),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha)),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=hgb_max_iter,
            learning_rate=hgb_learning_rate,
            max_leaf_nodes=hgb_max_leaf_nodes,
            l2_regularization=hgb_l2_regularization,
            random_state=random_state,
        ),
    }


def fit_baseline_models(
    X_train: pd.DataFrame,
    y_train: Any,
    random_state: int = 13,
    ridge_alpha: float = 1.0,
    hgb_max_iter: int = 50,
    hgb_learning_rate: float = 0.05,
    hgb_max_leaf_nodes: int = 31,
    hgb_l2_regularization: float = 1.0,
) -> dict[str, Any]:
    """Fit every C01 baseline on the supplied training partition only."""
    _validate_features(X_train)
    target = _validate_target(y_train)
    if len(X_train) != len(target):
        raise ValueError("Feature and target row counts must match")
    models = build_baseline_models(
        random_state=random_state,
        ridge_alpha=ridge_alpha,
        hgb_max_iter=hgb_max_iter,
        hgb_learning_rate=hgb_learning_rate,
        hgb_max_leaf_nodes=hgb_max_leaf_nodes,
        hgb_l2_regularization=hgb_l2_regularization,
    )
    for model in models.values():
        model.fit(X_train, target)
    return models


def predict_baselines(models: dict[str, Any], X: pd.DataFrame) -> dict[str, Any]:
    """Generate predictions without fitting or mutating the supplied models."""
    _validate_features(X)
    predictions: dict[str, Any] = {}
    for name, model in models.items():
        values = pd.to_numeric(pd.Series(model.predict(X)), errors="raise").to_numpy(
            dtype="float64"
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"Model {name!r} returned non-finite predictions")
        predictions[name] = values
    return predictions
