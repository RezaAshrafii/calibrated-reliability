"""Finite-sample split-conformal interval calculations for capped RUL."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def conformal_quantile(residuals: Any, alpha: float) -> float:
    """Return the finite-sample split-conformal residual quantile."""
    values = pd.to_numeric(pd.Series(residuals), errors="raise").to_numpy(dtype="float64")
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Residuals must be a non-empty finite one-dimensional array")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    if (values < 0).any():
        raise ValueError("Absolute residuals cannot be negative")
    rank = int(math.ceil((len(values) + 1) * (1.0 - alpha)))
    index = min(max(rank - 1, 0), len(values) - 1)
    return float(np.sort(values)[index])


def split_conformal_intervals(
    y_calibration: Any,
    calibration_prediction: Any,
    prediction: Any,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build symmetric split-conformal intervals without refitting a model."""
    y_cal = pd.to_numeric(pd.Series(y_calibration), errors="raise").to_numpy(dtype="float64")
    cal_pred = pd.to_numeric(pd.Series(calibration_prediction), errors="raise").to_numpy(
        dtype="float64"
    )
    center = pd.to_numeric(pd.Series(prediction), errors="raise").to_numpy(dtype="float64")
    if len(y_cal) != len(cal_pred) or len(y_cal) == 0:
        raise ValueError("Calibration targets and predictions must have equal non-zero length")
    if (
        not np.isfinite(y_cal).all()
        or not np.isfinite(cal_pred).all()
        or not np.isfinite(center).all()
    ):
        raise ValueError("Conformal targets and predictions must be finite")
    q = conformal_quantile(np.abs(y_cal - cal_pred), alpha)
    return center - q, center + q, q


def interval_metrics(
    y_true: Any,
    lower: Any,
    upper: Any,
    alpha: float,
    target_scale: float,
) -> dict[str, float]:
    """Compute coverage, width and normalized interval score."""
    truth = pd.to_numeric(pd.Series(y_true), errors="raise").to_numpy(dtype="float64")
    lo = pd.to_numeric(pd.Series(lower), errors="raise").to_numpy(dtype="float64")
    hi = pd.to_numeric(pd.Series(upper), errors="raise").to_numpy(dtype="float64")
    if len(truth) == 0 or len(truth) != len(lo) or len(lo) != len(hi):
        raise ValueError("Interval arrays must have equal non-zero length")
    if not 0 < alpha < 1 or target_scale <= 0:
        raise ValueError("alpha and target_scale are invalid")
    if not np.isfinite(truth).all() or not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("Interval values must be finite")
    if (lo > hi).any():
        raise ValueError("Lower interval bounds cannot exceed upper bounds")
    below = np.maximum(lo - truth, 0.0)
    above = np.maximum(truth - hi, 0.0)
    score = (hi - lo) + (2.0 / alpha) * (below + above)
    return {
        "coverage": float(((truth >= lo) & (truth <= hi)).mean()),
        "mean_width": float(np.mean(hi - lo)),
        "normalized_interval_score": float(np.mean(score) / target_scale),
    }
