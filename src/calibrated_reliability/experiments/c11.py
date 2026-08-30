"""Exact C11 finite-reservoir conformal audit primitives and orchestration."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.integrate import quad
from scipy.special import betaln
from scipy.stats import beta as beta_distribution
from scipy.stats import wasserstein_distance

from calibrated_reliability.data.labels import add_rul_targets
from calibrated_reliability.data.loader import COLUMNS
from calibrated_reliability.data.splitting import (
    generate_cut_points,
    restrict_to_cut_points,
    split_engine_ids,
)
from calibrated_reliability.features.temporal import TemporalFeatureTransformer
from calibrated_reliability.models.baselines import build_baseline_models

C11_ADR_PATH = "docs/decisions/ADR-0012-c11-finite-reservoir-design.md"
C11_EXCLUDED_UNATTAINABLE = "not_evaluated_due_to_unattainable_finite_rank"
C11_EXCLUDED_DEGENERATE = "not_evaluated_due_to_degenerate_full_reservoir_subset"
C11_EVALUATED = "evaluate"
C11_PROBABILITY_TOLERANCE = 1.0e-12

_C11_EXPECTED_CONFIG: dict[str, Any] = {
    "experiment_id": "C11",
    "source": "FD001",
    "target": "FD001",
    "evaluation_unit": "engine_endpoint_finite_reservoir",
    "predictor_seed": 13,
    "split_seed": 13,
    "cut_point_seed": 13,
    "roles": {"predictor_fit_engines": 60, "reservoir_engines": 40},
    "rul_cap": 125,
    "preprocessing": {"temporal_windows": [5, 10, 20], "variance_threshold": 0.0},
    "calibration": {
        "min_observed_cycles": 30,
        "lower_fraction": 0.40,
        "upper_fraction": 0.90,
    },
    "model": {
        "type": "hist_gradient_boosting",
        "max_iter": 50,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
        "random_state": 13,
    },
    "cells": [
        {"role": "primary", "n_cal": 10, "alpha": 0.10, "status": C11_EVALUATED},
        {"role": "primary", "n_cal": 15, "alpha": 0.10, "status": C11_EVALUATED},
        {"role": "primary", "n_cal": 20, "alpha": 0.10, "status": C11_EVALUATED},
        {"role": "primary", "n_cal": 30, "alpha": 0.10, "status": C11_EVALUATED},
        {
            "role": "sensitivity",
            "n_cal": 10,
            "alpha": 0.05,
            "status": C11_EXCLUDED_UNATTAINABLE,
        },
        {
            "role": "sensitivity",
            "n_cal": 15,
            "alpha": 0.05,
            "status": C11_EXCLUDED_UNATTAINABLE,
        },
        {"role": "sensitivity", "n_cal": 20, "alpha": 0.05, "status": C11_EVALUATED},
        {"role": "sensitivity", "n_cal": 30, "alpha": 0.05, "status": C11_EVALUATED},
        {
            "role": "excluded",
            "n_cal": 40,
            "alpha": 0.10,
            "status": C11_EXCLUDED_DEGENERATE,
        },
        {
            "role": "excluded",
            "n_cal": 40,
            "alpha": 0.05,
            "status": C11_EXCLUDED_DEGENERATE,
        },
    ],
    "exact_distribution": {
        "method": "combinatorial_order_statistic_multiplicity",
        "reservoir_size": 40,
        "probability_tolerance": C11_PROBABILITY_TOLERANCE,
        "tie_policy": "exact_round_trip_float64",
    },
    "references": {
        "continuous": "beta",
        "finite_evaluation": "beta_binomial",
        "evaluation_endpoints": 100,
    },
    "discrepancies": {
        "metrics": [
            "ks_distance",
            "signed_mean_difference",
            "signed_severe_tail_difference",
            "signed_population_sd_difference",
            "wasserstein_1",
        ],
        "signed_orientation": "finite_reservoir_minus_reference",
        "severe_undercoverage_margin": 0.10,
        "quadrature_atol": 1.0e-12,
        "quadrature_rtol": 1.0e-12,
        "reconstruction_atol": 1.0e-12,
        "reconstruction_rtol": 1.0e-12,
    },
    "observation_diagnostic": {
        "metric": "empirical_wasserstein_1",
        "unit": "observed_cycles",
        "weighting": "equal_engine",
    },
}


def _exact_value(actual: Any, expected: Any, path: str) -> None:
    """Recursively reject coercion, unknown fields, ordering drift, and altered values."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"{path} must be a mapping")
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            raise ValueError(
                f"{path} schema mismatch: missing={sorted(expected_keys - actual_keys)}, "
                f"extra={sorted(actual_keys - expected_keys)}"
            )
        for key, value in expected.items():
            _exact_value(actual[key], value, f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{path} list does not match the frozen design")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            _exact_value(actual_item, expected_item, f"{path}[{index}]")
        return
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{path} does not match the frozen C11 design")


@dataclass(frozen=True)
class C11Cell:
    """One declared evaluated or excluded C11 cell."""

    role: str
    n_cal: int
    alpha: float
    status: str

    @property
    def requested_rank(self) -> int:
        return int(math.ceil((self.n_cal + 1) * (1.0 - self.alpha)))


@dataclass(frozen=True)
class C11Config:
    """Strict, exact representation of the independently accepted C11 design."""

    payload: dict[str, Any]
    cells: tuple[C11Cell, ...]

    @classmethod
    def from_yaml(cls, text: str) -> C11Config:
        raw = yaml.safe_load(text)
        if not isinstance(raw, dict):
            raise ValueError("C11 configuration must be a mapping")
        expected = copy.deepcopy(_C11_EXPECTED_CONFIG)
        _exact_value(raw, expected, "C11 configuration")
        cells = tuple(C11Cell(**cell) for cell in raw["cells"])
        config = cls(payload=raw, cells=cells)
        config.validate()
        return config

    def validate(self) -> None:
        """Revalidate even manually constructed instances before scientific use."""
        _exact_value(self.payload, _C11_EXPECTED_CONFIG, "C11 configuration")
        expected_cells = tuple(C11Cell(**cell) for cell in self.payload["cells"])
        if self.cells != expected_cells:
            raise ValueError("C11 cells differ from the frozen configuration payload")

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.payload)

    @property
    def predictor_seed(self) -> int:
        return int(self.payload["predictor_seed"])

    @property
    def split_seed(self) -> int:
        return int(self.payload["split_seed"])

    @property
    def cut_point_seed(self) -> int:
        return int(self.payload["cut_point_seed"])

    @property
    def rul_cap(self) -> int:
        return int(self.payload["rul_cap"])


@dataclass(frozen=True)
class C11Result:
    """Complete in-memory result needed for independently reconstructible artifacts."""

    split_manifest: dict[str, Any]
    reservoir_scores: pd.DataFrame
    evaluation_scores: pd.DataFrame
    enumeration_cells: pd.DataFrame
    quantile_distribution: pd.DataFrame
    beta_binomial_distribution: pd.DataFrame
    reference_summary: dict[str, Any]
    distribution_summary: pd.DataFrame
    observation_mechanism: dict[str, Any]
    feature_names: list[str]
    model_specification: dict[str, Any]
    cut_points: dict[int, int]


def _is_strict_numeric(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def _numeric_scores(values: Any, name: str) -> np.ndarray:
    raw = pd.Series(values)
    if not raw.map(_is_strict_numeric).all():
        raise ValueError(f"{name} must contain strict numeric values, not strings or booleans")
    scores = np.asarray(pd.to_numeric(raw, errors="raise").to_numpy(dtype="float64"))
    if len(scores) == 0 or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError(f"{name} must be non-empty, finite, and nonnegative")
    return scores


def exact_order_statistic_distribution(
    reservoir_scores: Any,
    n_cal: int,
    alpha: float,
    *,
    probability_tolerance: float = C11_PROBABILITY_TOLERANCE,
) -> pd.DataFrame:
    """Return exact tied-quantile masses for a uniform subset without replacement."""
    scores = _numeric_scores(reservoir_scores, "reservoir scores")
    n_reservoir = len(scores)
    if isinstance(n_cal, bool) or not isinstance(n_cal, int) or not 1 <= n_cal < n_reservoir:
        raise ValueError("n_cal must be an integer strictly below the reservoir size")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError("alpha must be numeric and strictly between zero and one")
    if not math.isfinite(float(alpha)) or probability_tolerance != C11_PROBABILITY_TOLERANCE:
        raise ValueError("C11 probability tolerance is frozen to 1e-12")
    rank = int(math.ceil((n_cal + 1) * (1.0 - float(alpha))))
    if rank > n_cal:
        raise ValueError("finite observed rank is unattainable for this cell")
    sorted_scores = np.sort(scores, kind="stable")
    denominator = math.comb(n_reservoir, n_cal)
    position_rows: list[dict[str, Any]] = []
    for position in range(rank, n_reservoir - n_cal + rank + 1):
        multiplicity = math.comb(position - 1, rank - 1) * math.comb(
            n_reservoir - position, n_cal - rank
        )
        position_rows.append(
            {
                "position": position,
                "quantile": float(sorted_scores[position - 1]),
                "multiplicity": multiplicity,
            }
        )
    if sum(int(row["multiplicity"]) for row in position_rows) != denominator:
        raise ArithmeticError("C11 order-statistic multiplicities do not sum exactly")
    grouped_rows: list[dict[str, Any]] = []
    for quantile, group in pd.DataFrame(position_rows).groupby("quantile", sort=True):
        positions = group["position"].astype(int).tolist()
        multiplicities = group["multiplicity"].astype(object).tolist()
        exact_multiplicity = sum(int(value) for value in multiplicities)
        grouped_rows.append(
            {
                "quantile": float(quantile),
                "position_min": min(positions),
                "position_max": max(positions),
                "position_multiplicities_json": json.dumps(
                    dict(zip((str(value) for value in positions), multiplicities, strict=True)),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "exact_multiplicity": exact_multiplicity,
                "combination_count": denominator,
                "probability": exact_multiplicity / denominator,
            }
        )
    result = pd.DataFrame(grouped_rows)
    if abs(float(result["probability"].sum()) - 1.0) > probability_tolerance:
        raise ArithmeticError("C11 floating probability mass does not sum to one")
    return result


def beta_binomial_reference(n_cal: int, alpha: float, endpoints: int = 100) -> pd.DataFrame:
    """Return the frozen finite-evaluation beta-binomial reference PMF."""
    if isinstance(n_cal, bool) or not isinstance(n_cal, int) or n_cal < 1:
        raise ValueError("C11 beta-binomial calibration size must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("C11 beta-binomial alpha must be numeric")
    if not math.isfinite(float(alpha)) or not 0 < alpha < 1:
        raise ValueError("C11 beta-binomial alpha must be finite and between zero and one")
    if isinstance(endpoints, bool) or endpoints != 100:
        raise ValueError("C11 beta-binomial evaluation count is frozen to 100")
    rank = int(math.ceil((n_cal + 1) * (1.0 - alpha)))
    if rank > n_cal:
        raise ValueError("finite observed rank is unattainable for this cell")
    a = rank
    b = n_cal + 1 - rank
    rows = []
    for count in range(endpoints + 1):
        log_mass = (
            math.log(math.comb(endpoints, count))
            + float(betaln(count + a, endpoints - count + b))
            - float(betaln(a, b))
        )
        rows.append(
            {
                "covered_endpoints": count,
                "coverage": count / endpoints,
                "probability": math.exp(log_mass),
            }
        )
    result = pd.DataFrame(rows)
    if abs(float(result["probability"].sum()) - 1.0) > C11_PROBABILITY_TOLERANCE:
        raise ArithmeticError("C11 beta-binomial probability mass does not sum to one")
    return result


def _weighted_distribution(values: Any, probabilities: Any) -> tuple[np.ndarray, np.ndarray]:
    raw_values = pd.Series(values)
    raw_probabilities = pd.Series(probabilities)
    if (
        not raw_values.map(_is_strict_numeric).all()
        or not raw_probabilities.map(_is_strict_numeric).all()
    ):
        raise ValueError(
            "weighted distribution must contain strict numeric values, not strings or booleans"
        )
    frame = pd.DataFrame(
        {
            "value": pd.to_numeric(raw_values, errors="raise"),
            "probability": pd.to_numeric(raw_probabilities, errors="raise"),
        }
    )
    if len(frame) == 0 or not np.isfinite(frame.to_numpy(dtype="float64")).all():
        raise ValueError("weighted distribution must be non-empty and finite")
    grouped = frame.groupby("value", sort=True, as_index=False)["probability"].sum()
    if ((grouped["value"] < 0.0) | (grouped["value"] > 1.0)).any():
        raise ValueError("C11 coverage distributions must lie within [0, 1]")
    probs = grouped["probability"].to_numpy(dtype="float64")
    if (probs < 0).any() or abs(float(probs.sum()) - 1.0) > C11_PROBABILITY_TOLERANCE:
        raise ValueError("weighted distribution probabilities must sum to one")
    return grouped["value"].to_numpy(dtype="float64"), probs


def _discrete_cdf(points: np.ndarray, values: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(probabilities)
    indices = np.searchsorted(values, points, side="right") - 1
    return np.where(indices >= 0, cumulative[np.maximum(indices, 0)], 0.0)


def _discrete_cdf_left(
    points: np.ndarray, values: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    cumulative = np.cumsum(probabilities)
    indices = np.searchsorted(values, points, side="left") - 1
    return np.where(indices >= 0, cumulative[np.maximum(indices, 0)], 0.0)


def weighted_ks_distance(
    finite_values: Any,
    finite_probabilities: Any,
    reference_values: Any | None = None,
    reference_probabilities: Any | None = None,
    *,
    beta_parameters: tuple[int, int] | None = None,
) -> float:
    """Compute the frozen left/right-limit weighted KS discrepancy."""
    values, probabilities = _weighted_distribution(finite_values, finite_probabilities)
    if (reference_values is None) == (beta_parameters is None):
        raise ValueError("Specify exactly one C11 reference distribution")
    if beta_parameters is not None:
        a, b = beta_parameters
        if (
            isinstance(a, bool)
            or isinstance(b, bool)
            or not isinstance(a, int)
            or not isinstance(b, int)
            or a < 1
            or b < 1
        ):
            raise ValueError("C11 Beta parameters must be positive integers")
        right = _discrete_cdf(values, values, probabilities)
        left = _discrete_cdf_left(values, values, probabilities)
        reference = beta_distribution.cdf(values, a, b)
        return float(max(np.max(np.abs(right - reference)), np.max(np.abs(left - reference))))
    if reference_probabilities is None:
        raise ValueError("Discrete reference probabilities are required")
    ref_values, ref_probabilities = _weighted_distribution(
        reference_values, reference_probabilities
    )
    points = np.union1d(values, ref_values)
    right_difference = np.abs(
        _discrete_cdf(points, values, probabilities)
        - _discrete_cdf(points, ref_values, ref_probabilities)
    )
    left_difference = np.abs(
        _discrete_cdf_left(points, values, probabilities)
        - _discrete_cdf_left(points, ref_values, ref_probabilities)
    )
    return float(max(np.max(right_difference), np.max(left_difference)))


def _discrete_wasserstein(
    values: np.ndarray,
    probabilities: np.ndarray,
    reference_values: np.ndarray,
    reference_probabilities: np.ndarray,
) -> float:
    points = np.union1d(np.array([0.0, 1.0]), np.union1d(values, reference_values))
    total = 0.0
    for left, right in zip(points[:-1], points[1:], strict=True):
        if right <= left:
            continue
        finite_cdf = float(_discrete_cdf(np.array([left]), values, probabilities)[0])
        reference_cdf = float(
            _discrete_cdf(np.array([left]), reference_values, reference_probabilities)[0]
        )
        total += (right - left) * abs(finite_cdf - reference_cdf)
    return float(total)


def distribution_discrepancies(
    finite_values: Any,
    finite_probabilities: Any,
    alpha: float,
    *,
    beta_parameters: tuple[int, int] | None = None,
    reference_values: Any | None = None,
    reference_probabilities: Any | None = None,
) -> dict[str, float]:
    """Return the five frozen discrepancies against one declared reference."""
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("C11 discrepancy alpha must be numeric")
    if not math.isfinite(float(alpha)) or not 0 < alpha < 1:
        raise ValueError("C11 discrepancy alpha must be finite and between zero and one")
    if (beta_parameters is None) == (reference_values is None):
        raise ValueError("Specify exactly one C11 discrepancy reference")
    values, probabilities = _weighted_distribution(finite_values, finite_probabilities)
    threshold = (1.0 - alpha) - 0.10
    finite_mean = float(np.sum(values * probabilities))
    finite_sd = float(np.sqrt(np.sum(probabilities * np.square(values - finite_mean))))
    finite_tail = float(probabilities[values < threshold].sum())
    if beta_parameters is not None:
        a, b = beta_parameters
        reference_mean = float(beta_distribution.mean(a, b))
        reference_sd = float(beta_distribution.std(a, b))
        reference_tail = float(beta_distribution.cdf(threshold, a, b))
        ks = weighted_ks_distance(values, probabilities, beta_parameters=beta_parameters)

        def integrand(point: float) -> float:
            finite_cdf = float(_discrete_cdf(np.array([point]), values, probabilities)[0])
            return abs(finite_cdf - float(beta_distribution.cdf(point, a, b)))

        w1 = float(
            quad(
                integrand,
                0.0,
                1.0,
                points=values[(values > 0.0) & (values < 1.0)].tolist(),
                epsabs=C11_PROBABILITY_TOLERANCE,
                epsrel=C11_PROBABILITY_TOLERANCE,
                limit=500,
            )[0]
        )
    else:
        if reference_values is None or reference_probabilities is None:
            raise ValueError("A complete C11 reference is required")
        ref_values, ref_probabilities = _weighted_distribution(
            reference_values, reference_probabilities
        )
        reference_mean = float(np.sum(ref_values * ref_probabilities))
        reference_sd = float(
            np.sqrt(np.sum(ref_probabilities * np.square(ref_values - reference_mean)))
        )
        reference_tail = float(ref_probabilities[ref_values < threshold].sum())
        ks = weighted_ks_distance(
            values,
            probabilities,
            reference_values=ref_values,
            reference_probabilities=ref_probabilities,
        )
        w1 = _discrete_wasserstein(values, probabilities, ref_values, ref_probabilities)
    result = {
        "ks_distance": ks,
        "signed_mean_difference": finite_mean - reference_mean,
        "signed_severe_tail_difference": finite_tail - reference_tail,
        "signed_population_sd_difference": finite_sd - reference_sd,
        "wasserstein_1": w1,
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ArithmeticError("C11 discrepancy calculation produced a non-finite value")
    return result


def _cell_key(cell: C11Cell) -> str:
    return f"n{cell.n_cal}_alpha_{cell.alpha:g}"


def _validate_declared_cells(config: C11Config) -> None:
    for cell in config.cells:
        rank = cell.requested_rank
        if cell.status == C11_EVALUATED and (rank > cell.n_cal or cell.n_cal >= 40):
            raise ValueError("C11 evaluated cell has no declared finite subset distribution")
        if cell.status == C11_EXCLUDED_UNATTAINABLE and rank <= cell.n_cal:
            raise ValueError("C11 unattainable exclusion has an attainable rank")
        if cell.status == C11_EXCLUDED_DEGENERATE and cell.n_cal != 40:
            raise ValueError("C11 degenerate exclusion must use the full reservoir")


def _endpoint_rows(features: pd.DataFrame) -> pd.DataFrame:
    indices = features.groupby("engine_id", sort=True)["cycle"].idxmax()
    return features.loc[indices].reset_index(drop=True)


def _score_frame(
    features: pd.DataFrame,
    truth_frame: pd.DataFrame,
    predictions: np.ndarray,
    cap: int,
    *,
    origin_roles: dict[int, str] | None = None,
) -> pd.DataFrame:
    endpoints = _endpoint_rows(features)
    keys = endpoints[["engine_id", "cycle"]]
    truth = keys.merge(
        truth_frame[["engine_id", "cycle", "rul_raw", "rul_capped"]],
        on=["engine_id", "cycle"],
        how="left",
        validate="one_to_one",
    )
    if truth[["rul_raw", "rul_capped"]].isna().any().any():
        raise ValueError("C11 endpoint truth alignment failed")
    endpoint_indices = features.groupby("engine_id", sort=True)["cycle"].idxmax()
    raw_prediction = pd.Series(predictions, index=features.index).loc[endpoint_indices].to_numpy()
    result = truth.copy()
    result["prediction_raw"] = raw_prediction
    result["prediction"] = np.clip(raw_prediction, 0.0, cap)
    result["residual"] = np.abs(
        result["rul_capped"].to_numpy(dtype="float64")
        - result["prediction"].to_numpy(dtype="float64")
    )
    if origin_roles is not None:
        result["origin_role"] = result["engine_id"].map(origin_roles)
        if result["origin_role"].isna().any():
            raise ValueError("C11 reservoir origin-role alignment failed")
    return result


def _analyze_cells(
    config: C11Config,
    reservoir_scores: np.ndarray,
    evaluation_scores: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    cell_rows: list[dict[str, Any]] = []
    quantile_frames: list[pd.DataFrame] = []
    beta_binomial_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    reference_cells: dict[str, Any] = {}
    for cell in config.cells:
        rank = cell.requested_rank
        regime = (
            "finite_quantile_unattainable"
            if rank > cell.n_cal
            else "max_statistic"
            if rank == cell.n_cal
            else "interior"
        )
        support_min = rank if cell.status == C11_EVALUATED else None
        support_max = 40 - cell.n_cal + rank if cell.status == C11_EVALUATED else None
        cell_key = _cell_key(cell)
        cell_rows.append(
            {
                "cell_id": cell_key,
                "role": cell.role,
                "n_cal": cell.n_cal,
                "alpha": cell.alpha,
                "status": cell.status,
                "requested_rank": rank,
                "finite_rank_attainable": rank <= cell.n_cal,
                "regime": regime,
                "support_position_min": support_min,
                "support_position_max": support_max,
                "combination_count": (
                    math.comb(40, cell.n_cal) if cell.status == C11_EVALUATED else None
                ),
                "finite_population_factor": (40 - cell.n_cal) / 39,
            }
        )
        if cell.status != C11_EVALUATED:
            continue
        quantiles = exact_order_statistic_distribution(reservoir_scores, cell.n_cal, cell.alpha)
        quantiles.insert(0, "cell_id", cell_key)
        quantiles.insert(1, "n_cal", cell.n_cal)
        quantiles.insert(2, "alpha", cell.alpha)
        coverages: list[float] = []
        normalized_scores: list[float] = []
        for quantile in quantiles["quantile"].to_numpy(dtype="float64"):
            coverages.append(float(np.mean(evaluation_scores <= quantile)))
            interval_score = 2.0 * quantile + (2.0 / cell.alpha) * np.maximum(
                evaluation_scores - quantile, 0.0
            )
            normalized_scores.append(float(np.mean(interval_score) / config.rul_cap))
        quantiles["coverage"] = coverages
        quantiles["mean_width"] = 2.0 * quantiles["quantile"]
        quantiles["median_width"] = 2.0 * quantiles["quantile"]
        quantiles["normalized_interval_score"] = normalized_scores
        quantiles["severe_undercoverage"] = quantiles["coverage"] < ((1.0 - cell.alpha) - 0.10)
        quantile_frames.append(quantiles)
        beta_binomial = beta_binomial_reference(cell.n_cal, cell.alpha)
        beta_binomial.insert(0, "cell_id", cell_key)
        beta_binomial.insert(1, "n_cal", cell.n_cal)
        beta_binomial.insert(2, "alpha", cell.alpha)
        beta_binomial_frames.append(beta_binomial)
        a = rank
        b = cell.n_cal + 1 - rank
        threshold = (1.0 - cell.alpha) - 0.10
        reference_cells[cell_key] = {
            "n_cal": cell.n_cal,
            "alpha": cell.alpha,
            "a": a,
            "b": b,
            "mean": float(beta_distribution.mean(a, b)),
            "population_sd": float(beta_distribution.std(a, b)),
            "quantile_0.05": float(beta_distribution.ppf(0.05, a, b)),
            "quantile_0.10": float(beta_distribution.ppf(0.10, a, b)),
            "strict_severe_undercoverage_threshold": threshold,
            "severe_undercoverage_probability": float(beta_distribution.cdf(threshold, a, b)),
        }
        for reference_name, kwargs in (
            ("continuous_beta", {"beta_parameters": (a, b)}),
            (
                "beta_binomial",
                {
                    "reference_values": beta_binomial["coverage"],
                    "reference_probabilities": beta_binomial["probability"],
                },
            ),
        ):
            discrepancies = distribution_discrepancies(
                quantiles["coverage"],
                quantiles["probability"],
                cell.alpha,
                **kwargs,
            )
            finite_values, finite_probabilities = _weighted_distribution(
                quantiles["coverage"], quantiles["probability"]
            )
            finite_mean = float(np.sum(finite_values * finite_probabilities))
            finite_sd = float(
                np.sqrt(np.sum(finite_probabilities * np.square(finite_values - finite_mean)))
            )
            summary_rows.append(
                {
                    "cell_id": cell_key,
                    "role": cell.role,
                    "n_cal": cell.n_cal,
                    "alpha": cell.alpha,
                    "status": "evaluated",
                    "reference": reference_name,
                    "finite_mean_coverage": finite_mean,
                    "finite_population_sd_coverage": finite_sd,
                    "finite_severe_tail_probability": float(
                        finite_probabilities[finite_values < ((1.0 - cell.alpha) - 0.10)].sum()
                    ),
                    "distinct_numerical_quantiles": int(quantiles["quantile"].nunique()),
                    **discrepancies,
                }
            )
    for cell in config.cells:
        if cell.status != C11_EVALUATED:
            summary_rows.append(
                {
                    "cell_id": _cell_key(cell),
                    "role": cell.role,
                    "n_cal": cell.n_cal,
                    "alpha": cell.alpha,
                    "status": cell.status,
                    "reference": "not_evaluated",
                }
            )
    reference_summary = {
        "continuous_reference": "Beta(r, n_cal + 1 - r)",
        "finite_evaluation_reference": "BetaBinomial(M=100, a=r, b=n_cal+1-r)",
        "evaluation_endpoints": 100,
        "strict_tail_formula": "coverage < (1-alpha)-0.10",
        "cdf_convention": "left_limits_and_right_continuous_at_union_of_jump_locations",
        "wasserstein_continuous_quadrature": {
            "domain": [0.0, 1.0],
            "absolute_tolerance": C11_PROBABILITY_TOLERANCE,
            "relative_tolerance": C11_PROBABILITY_TOLERANCE,
        },
        "reconstruction_tolerance": {
            "atol": C11_PROBABILITY_TOLERANCE,
            "rtol": C11_PROBABILITY_TOLERANCE,
        },
        "cells": reference_cells,
    }
    return (
        pd.DataFrame(cell_rows),
        pd.concat(quantile_frames, ignore_index=True),
        pd.concat(beta_binomial_frames, ignore_index=True),
        reference_summary,
        pd.DataFrame(summary_rows),
    )


def run_c11(
    train: pd.DataFrame,
    test: pd.DataFrame,
    test_rul: pd.DataFrame,
    config: C11Config,
) -> C11Result:
    """Compute C11 exactly in memory without filesystem side effects."""
    config.validate()
    _validate_declared_cells(config)
    train_ids = sorted(int(value) for value in train["engine_id"].unique())
    if train_ids != list(range(1, 101)):
        raise ValueError("C11 requires exactly FD001 training engine IDs 1 through 100")
    test_ids = [int(value) for value in test["engine_id"].drop_duplicates()]
    if test_ids != list(range(1, len(test_rul) + 1)) or len(test_ids) != 100:
        raise ValueError("C11 requires 100 ordered official FD001 test engines and RUL rows")
    partitions = split_engine_ids(train_ids, seed=config.split_seed)
    fit_ids = partitions["base_train"]
    reservoir_ids = sorted(partitions["calibration"] + partitions["validation"])
    if (
        len(fit_ids) != 60
        or len(reservoir_ids) != 40
        or set(fit_ids) & set(reservoir_ids)
        or sorted(fit_ids + reservoir_ids) != train_ids
    ):
        raise ValueError("C11 predictor and reservoir roles must be disjoint 60/40 coverage")
    raw_fit = train[train["engine_id"].isin(fit_ids)][COLUMNS].copy()
    full_labeled = add_rul_targets(train[COLUMNS].copy(), cap=config.rul_cap)
    fit_labeled = add_rul_targets(raw_fit, cap=config.rul_cap)
    cut_points = generate_cut_points(
        train,
        reservoir_ids,
        seed=config.cut_point_seed,
        min_observed_cycles=30,
        lower_fraction=0.40,
        upper_fraction=0.90,
    )
    reservoir_raw = restrict_to_cut_points(
        train[train["engine_id"].isin(reservoir_ids)][COLUMNS], cut_points
    )
    temporal = TemporalFeatureTransformer(windows=(5, 10, 20), variance_threshold=0.0).fit(raw_fit)
    fit_features = temporal.transform(raw_fit)
    reservoir_features = temporal.transform(reservoir_raw)
    test_features = temporal.transform(test[COLUMNS])
    if not fit_features[["engine_id", "cycle"]].equals(
        raw_fit[["engine_id", "cycle"]].reset_index(drop=True)
    ):
        raise ValueError("C11 fitting target and feature order differ")
    model = build_baseline_models(
        random_state=config.predictor_seed,
        hgb_max_iter=50,
        hgb_learning_rate=0.05,
        hgb_max_leaf_nodes=31,
        hgb_l2_regularization=1.0,
    )["hist_gradient_boosting"]
    model.fit(
        fit_features.drop(columns=["engine_id"]),
        fit_labeled["rul_capped"].to_numpy(dtype="float64"),
    )
    reservoir_predictions = np.asarray(
        model.predict(reservoir_features.drop(columns=["engine_id"])), dtype="float64"
    )
    test_predictions = np.asarray(
        model.predict(test_features.drop(columns=["engine_id"])), dtype="float64"
    )
    if not np.isfinite(reservoir_predictions).all() or not np.isfinite(test_predictions).all():
        raise ValueError("C11 predictor returned non-finite values")
    origin_roles = {
        **{engine_id: "former_calibration" for engine_id in partitions["calibration"]},
        **{engine_id: "former_validation" for engine_id in partitions["validation"]},
    }
    reservoir_scores = _score_frame(
        reservoir_features,
        full_labeled,
        reservoir_predictions,
        config.rul_cap,
        origin_roles=origin_roles,
    )
    reservoir_scores["complete_lifetime"] = reservoir_scores["engine_id"].map(
        train.groupby("engine_id")["cycle"].max().astype(int).to_dict()
    )
    test_endpoints = _endpoint_rows(test_features)
    if test_endpoints["engine_id"].astype(int).tolist() != test_ids:
        raise ValueError("C11 official endpoint IDs do not align with RUL rows")
    test_truth = pd.DataFrame(
        {
            "engine_id": test_ids,
            "cycle": test_endpoints["cycle"].astype(int),
            "rul_raw": test_rul["rul"].astype("int64").to_numpy(),
        }
    )
    test_truth["rul_capped"] = test_truth["rul_raw"].clip(upper=config.rul_cap)
    evaluation_scores = _score_frame(test_features, test_truth, test_predictions, config.rul_cap)
    evaluation_scores["observed_cycles"] = evaluation_scores["cycle"].astype(int)
    evaluation_scores["cap_saturated"] = evaluation_scores["rul_raw"] > config.rul_cap
    (
        enumeration_cells,
        quantile_distribution,
        beta_binomial_distribution,
        reference_summary,
        distribution_summary,
    ) = _analyze_cells(
        config,
        reservoir_scores["residual"].to_numpy(dtype="float64"),
        evaluation_scores["residual"].to_numpy(dtype="float64"),
    )
    reservoir_cut_cycles = reservoir_scores["cycle"].to_numpy(dtype="float64")
    official_cycles = evaluation_scores["observed_cycles"].to_numpy(dtype="float64")
    observation_mechanism = {
        "metric": "empirical_wasserstein_1",
        "unit": "observed_cycles",
        "weighting": "equal_engine",
        "reservoir_engine_count": 40,
        "official_endpoint_count": 100,
        "reservoir_cut_cycles": reservoir_cut_cycles.astype(int).tolist(),
        "reservoir_complete_lifetimes": reservoir_scores["complete_lifetime"].astype(int).tolist(),
        "official_observed_endpoint_cycles": official_cycles.astype(int).tolist(),
        "distance": float(wasserstein_distance(reservoir_cut_cycles, official_cycles)),
        "evaluation_cap_saturation_fraction": float(evaluation_scores["cap_saturated"].mean()),
        "reservoir_residual_tied_rows": int(
            reservoir_scores["residual"].duplicated(keep=False).sum()
        ),
        "reservoir_distinct_residuals": int(reservoir_scores["residual"].nunique()),
    }
    split_manifest = {
        "seed": config.split_seed,
        "predictor_fit_engine_ids": fit_ids,
        "reservoir_engine_ids": reservoir_ids,
        "former_calibration_engine_ids": partitions["calibration"],
        "former_validation_engine_ids": partitions["validation"],
    }
    return C11Result(
        split_manifest=split_manifest,
        reservoir_scores=reservoir_scores,
        evaluation_scores=evaluation_scores,
        enumeration_cells=enumeration_cells,
        quantile_distribution=quantile_distribution,
        beta_binomial_distribution=beta_binomial_distribution,
        reference_summary=reference_summary,
        distribution_summary=distribution_summary,
        observation_mechanism=observation_mechanism,
        feature_names=[name for name in temporal.feature_names_out_ if name != "engine_id"],
        model_specification=copy.deepcopy(config.payload["model"]),
        cut_points=cut_points,
    )
