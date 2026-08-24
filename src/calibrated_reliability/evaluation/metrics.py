"""Predeclared point-prediction metrics for endpoint RUL evaluation."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _finite_pair(y_true: Any, y_pred: Any) -> tuple[Any, Any]:
    """Validate and align true and predicted RUL arrays."""
    truth = pd.to_numeric(pd.Series(y_true), errors="raise").to_numpy(dtype="float64")
    prediction = pd.to_numeric(pd.Series(y_pred), errors="raise").to_numpy(dtype="float64")
    if truth.ndim != 1 or prediction.ndim != 1 or len(truth) == 0:
        raise ValueError("RUL arrays must be non-empty one-dimensional arrays")
    if len(truth) != len(prediction):
        raise ValueError("True and predicted RUL lengths must match")
    if not all(math.isfinite(float(value)) for value in truth) or not all(
        math.isfinite(float(value)) for value in prediction
    ):
        raise ValueError("True and predicted RUL values must be finite")
    return truth, prediction


def nasa_asymmetric_score(y_true: Any, y_pred: Any) -> float:
    """Compute the C-MAPSS asymmetric score.

    Negative error means an early prediction and uses scale 13; positive error
    means a late prediction and uses scale 10.
    """
    truth, prediction = _finite_pair(y_true, y_pred)
    error = prediction - truth
    penalties = [
        math.exp(-float(delta) / 13.0) - 1.0 if delta < 0 else math.exp(float(delta) / 10.0) - 1.0
        for delta in error
    ]
    return float(sum(penalties))


def rul_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return the preregistered point-prediction metrics."""
    truth, prediction = _finite_pair(y_true, y_pred)
    return {
        "rmse": math.sqrt(float(mean_squared_error(truth, prediction))),
        "mae": float(mean_absolute_error(truth, prediction)),
        "signed_error": float(sum(prediction - truth) / len(truth)),
        "nasa_score": nasa_asymmetric_score(truth, prediction),
    }


def evaluate_endpoint_predictions(
    endpoints: pd.DataFrame,
    prediction_column: str = "y_pred",
    truth_column: str = "y_true",
) -> dict[str, float]:
    """Evaluate one prediction per engine at the preregistered endpoint unit."""
    required = {"engine_id", truth_column, prediction_column}
    missing = required.difference(endpoints.columns)
    if missing:
        raise ValueError(f"Missing endpoint columns: {sorted(missing)}")
    if endpoints["engine_id"].duplicated().any():
        raise ValueError("Endpoint evaluation requires exactly one row per engine")
    return rul_metrics(endpoints[truth_column], endpoints[prediction_column])
