"""Finite-sample split-conformal interval calculations for capped RUL."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

UnattainableRankPolicy = Literal["raise", "infinite", "legacy_max_clamp"]


@dataclass(frozen=True)
class ConformalQuantileResult:
    """Self-reporting finite-sample conformal quantile decision."""

    quantile: float
    n_cal: int
    requested_rank: int
    effective_rank: int | None
    finite_rank_attainable: bool
    regime: Literal["interior", "max_statistic", "finite_quantile_unattainable"]
    unattainable_rank_policy: UnattainableRankPolicy


def conformal_quantile_result(
    residuals: Any,
    alpha: float,
    *,
    on_unattainable: UnattainableRankPolicy = "raise",
) -> ConformalQuantileResult:
    """Return a quantile together with rank attainability diagnostics."""
    values = pd.to_numeric(pd.Series(residuals), errors="raise").to_numpy(dtype="float64")
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Residuals must be a non-empty finite one-dimensional array")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    if (values < 0).any():
        raise ValueError("Absolute residuals cannot be negative")
    if on_unattainable not in {"raise", "infinite", "legacy_max_clamp"}:
        raise ValueError("Unknown unattainable-rank policy")
    n_cal = len(values)
    rank = int(math.ceil((n_cal + 1) * (1.0 - alpha)))
    if rank < n_cal:
        regime: Literal["interior", "max_statistic", "finite_quantile_unattainable"] = "interior"
    elif rank == n_cal:
        regime = "max_statistic"
    else:
        regime = "finite_quantile_unattainable"
    attainable = rank <= n_cal
    if not attainable and on_unattainable == "raise":
        raise ValueError(
            f"Requested conformal rank {rank} is unattainable with {n_cal} calibration scores"
        )
    sorted_values = np.sort(values)
    if attainable:
        effective_rank: int | None = rank
        quantile = float(sorted_values[rank - 1])
    elif on_unattainable == "infinite":
        effective_rank = None
        quantile = math.inf
    else:
        effective_rank = n_cal
        quantile = float(sorted_values[-1])
    return ConformalQuantileResult(
        quantile=quantile,
        n_cal=n_cal,
        requested_rank=rank,
        effective_rank=effective_rank,
        finite_rank_attainable=attainable,
        regime=regime,
        unattainable_rank_policy=on_unattainable,
    )


def conformal_quantile(
    residuals: Any,
    alpha: float,
    *,
    on_unattainable: UnattainableRankPolicy = "raise",
) -> float:
    """Return a finite-sample quantile under an explicit unattainable-rank policy."""
    return conformal_quantile_result(residuals, alpha, on_unattainable=on_unattainable).quantile


class ACIState:
    """Stateful ACI predictor enforcing predict-before-outcome-update order."""

    def __init__(
        self,
        residuals: Any,
        nominal_alpha: float,
        gamma: float,
        alpha_min: float,
        alpha_max: float,
        *,
        unattainable_rank_policy: UnattainableRankPolicy,
    ) -> None:
        values = pd.to_numeric(pd.Series(residuals), errors="raise").to_numpy(dtype="float64")
        if len(values) == 0 or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("ACI residuals must be non-empty, finite, and nonnegative")
        if not 0 < nominal_alpha < 1 or gamma <= 0:
            raise ValueError("ACI residuals, alpha, or gamma are invalid")
        if not 0 < alpha_min < alpha_max < 1 or not alpha_min <= nominal_alpha <= alpha_max:
            raise ValueError("ACI alpha bounds are invalid")
        if unattainable_rank_policy not in {"raise", "infinite", "legacy_max_clamp"}:
            raise ValueError("Unknown unattainable-rank policy")
        self.residuals = values
        self.nominal_alpha = nominal_alpha
        self.gamma = gamma
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.unattainable_rank_policy = unattainable_rank_policy
        self.alpha = nominal_alpha
        self._pending: tuple[float, float] | None = None
        self.last_quantile_result: ConformalQuantileResult | None = None

    def predict_interval(self, center: float) -> tuple[float, float, float, float]:
        """Create one interval without accepting or inspecting its outcome."""
        if self._pending is not None:
            raise RuntimeError("ACI outcome must update the state before the next prediction")
        if not math.isfinite(center):
            raise ValueError("ACI center must be finite")
        decision = conformal_quantile_result(
            self.residuals,
            self.alpha,
            on_unattainable=self.unattainable_rank_policy,
        )
        self.last_quantile_result = decision
        q = decision.quantile
        lower, upper = center - q, center + q
        self._pending = (lower, upper)
        return lower, upper, self.alpha, q

    def update(self, observed_truth: float) -> tuple[bool, float]:
        """Reveal one outcome and update alpha only for future predictions."""
        if self._pending is None:
            raise RuntimeError("ACI interval must be predicted before its outcome is revealed")
        if not math.isfinite(observed_truth):
            raise ValueError("ACI observed truth must be finite")
        lower, upper = self._pending
        missed = bool(observed_truth < lower or observed_truth > upper)
        self.alpha = float(
            np.clip(
                self.alpha + self.gamma * (self.nominal_alpha - float(missed)),
                self.alpha_min,
                self.alpha_max,
            )
        )
        self._pending = None
        return missed, self.alpha


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
    lower = pd.to_numeric(pd.Series(lower_prediction), errors="raise").to_numpy(dtype="float64")
    upper = pd.to_numeric(pd.Series(upper_prediction), errors="raise").to_numpy(dtype="float64")
    if len(lower) == 0 or len(lower) != len(upper):
        raise ValueError("CQR prediction arrays must have equal non-zero length")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("CQR prediction arrays must be finite")
    if (lower > upper).any():
        raise ValueError("CQR lower predictions cannot exceed upper predictions")
    q = cqr_conformity_quantile(y_calibration, lower_calibration, upper_calibration, alpha)
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
