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


def cqr_conformity_quantile(
    y_calibration: Any, lower_prediction: Any, upper_prediction: Any, alpha: float
) -> float:
    """Return the finite-sample CQR conformity quantile."""
    truth = pd.to_numeric(pd.Series(y_calibration), errors="raise").to_numpy(dtype="float64")
    lower = pd.to_numeric(pd.Series(lower_prediction), errors="raise").to_numpy(dtype="float64")
    upper = pd.to_numeric(pd.Series(upper_prediction), errors="raise").to_numpy(dtype="float64")
    if len(truth) == 0 or len(truth) != len(lower) or len(lower) != len(upper):
        raise ValueError("CQR arrays must have equal non-zero length")
    if (
        not np.isfinite(truth).all()
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
    ):
        raise ValueError("CQR arrays must be finite")
    if (lower > upper).any():
        raise ValueError("CQR lower predictions cannot exceed upper predictions")
    scores = np.maximum(lower - truth, truth - upper)
    return conformal_quantile(np.maximum(scores, 0.0), alpha)


def cqr_intervals(
    y_calibration: Any,
    lower_calibration: Any,
    upper_calibration: Any,
    lower_prediction: Any,
    upper_prediction: Any,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build unbounded CQR intervals without refitting quantile models."""
    q = cqr_conformity_quantile(y_calibration, lower_calibration, upper_calibration, alpha)
    lower = pd.to_numeric(pd.Series(lower_prediction), errors="raise").to_numpy(dtype="float64")
    upper = pd.to_numeric(pd.Series(upper_prediction), errors="raise").to_numpy(dtype="float64")
    if len(lower) == 0 or len(lower) != len(upper):
        raise ValueError("CQR prediction arrays must have equal non-zero length")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("CQR prediction arrays must be finite")
    if (lower > upper).any():
        raise ValueError("CQR lower predictions cannot exceed upper predictions")
    return lower - q, upper + q, q


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


def bootstrap_interval_metric_cis(
    y_true: Any,
    lower: Any,
    upper: Any,
    alpha: float,
    target_scale: float,
    seed: int,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Return deterministic engine-level percentile bootstrap confidence intervals."""
    truth = pd.to_numeric(pd.Series(y_true), errors="raise").to_numpy(dtype="float64")
    lo = pd.to_numeric(pd.Series(lower), errors="raise").to_numpy(dtype="float64")
    hi = pd.to_numeric(pd.Series(upper), errors="raise").to_numpy(dtype="float64")
    interval_metrics(truth, lo, hi, alpha, target_scale)
    if seed < 0 or n_resamples < 1 or not 0 < confidence_level < 1:
        raise ValueError("Bootstrap seed, resample count, or confidence level is invalid")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(truth), size=(n_resamples, len(truth)))
    sampled_truth = truth[indices]
    sampled_lower = lo[indices]
    sampled_upper = hi[indices]
    below = np.maximum(sampled_lower - sampled_truth, 0.0)
    above = np.maximum(sampled_truth - sampled_upper, 0.0)
    scores = (sampled_upper - sampled_lower) + (2.0 / alpha) * (below + above)
    samples = {
        "coverage": ((sampled_truth >= sampled_lower) & (sampled_truth <= sampled_upper)).mean(
            axis=1
        ),
        "mean_width": (sampled_upper - sampled_lower).mean(axis=1),
        "normalized_interval_score": scores.mean(axis=1) / target_scale,
    }
    tail = (1.0 - confidence_level) / 2.0
    return {
        name: {
            "lower": float(np.quantile(values, tail)),
            "upper": float(np.quantile(values, 1.0 - tail)),
        }
        for name, values in samples.items()
    }
