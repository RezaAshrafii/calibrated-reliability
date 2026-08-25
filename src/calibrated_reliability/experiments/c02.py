"""Pure orchestration for C02 split-conformal RUL intervals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yaml

from calibrated_reliability.data.labels import add_rul_targets
from calibrated_reliability.data.loader import COLUMNS
from calibrated_reliability.data.splitting import (
    generate_cut_points,
    restrict_to_cut_points,
    split_engine_ids,
)
from calibrated_reliability.evaluation.conformal import interval_metrics, split_conformal_intervals
from calibrated_reliability.features.temporal import TemporalFeatureTransformer
from calibrated_reliability.models.baselines import fit_baseline_models, predict_baselines


def _strict_int(value: Any, name: str) -> int:
    """Reject lossy integer coercion in executable configuration fields."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


@dataclass(frozen=True)
class C02Config:
    """Validated executable C02 configuration."""

    experiment_id: str
    source: str
    target: str
    evaluation_unit: str
    seeds: tuple[int, ...]
    alphas: tuple[float, ...]
    rul_cap: int
    temporal_windows: tuple[int, ...]
    variance_threshold: float
    min_observed_cycles: int
    lower_fraction: float
    upper_fraction: float

    @classmethod
    def from_yaml(cls, text: str) -> C02Config:
        """Parse the strict C02 configuration schema."""
        raw = yaml.safe_load(text)
        if not isinstance(raw, dict):
            raise ValueError("C02 configuration must be a mapping")
        expected = {
            "experiment_id",
            "source",
            "target",
            "evaluation_unit",
            "seeds",
            "alphas",
            "rul_cap",
            "preprocessing",
            "calibration",
        }
        if set(raw) != expected:
            raise ValueError("C02 configuration schema mismatch")
        preprocessing = raw["preprocessing"]
        calibration = raw["calibration"]
        if not isinstance(preprocessing, dict) or set(preprocessing) != {
            "temporal_windows",
            "variance_threshold",
        }:
            raise ValueError("C02 preprocessing schema mismatch")
        if not isinstance(calibration, dict) or set(calibration) != {
            "min_observed_cycles",
            "lower_fraction",
            "upper_fraction",
        }:
            raise ValueError("C02 calibration schema mismatch")
        seeds = raw["seeds"]
        alphas = raw["alphas"]
        windows = preprocessing["temporal_windows"]
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in seeds):
            raise ValueError("C02 seeds must be integers")
        if (
            not isinstance(seeds, list)
            or not isinstance(alphas, list)
            or not isinstance(windows, list)
        ):
            raise ValueError("C02 alphas and temporal windows must be lists")
        config = cls(
            experiment_id=str(raw["experiment_id"]),
            source=str(raw["source"]),
            target=str(raw["target"]),
            evaluation_unit=str(raw["evaluation_unit"]),
            seeds=tuple(seeds),
            alphas=tuple(float(value) for value in alphas),
            rul_cap=_strict_int(raw["rul_cap"], "rul_cap"),
            temporal_windows=tuple(_strict_int(value, "temporal window") for value in windows),
            variance_threshold=float(preprocessing["variance_threshold"]),
            min_observed_cycles=_strict_int(
                calibration["min_observed_cycles"], "min_observed_cycles"
            ),
            lower_fraction=float(calibration["lower_fraction"]),
            upper_fraction=float(calibration["upper_fraction"]),
        )
        if (
            config.experiment_id != "C02"
            or config.source != "FD001"
            or config.target != "FD001"
            or config.evaluation_unit != "engine_endpoint"
            or not config.seeds
            or len(set(config.seeds)) != len(config.seeds)
            or not config.alphas
            or len(set(config.alphas)) != len(config.alphas)
            or any(not 0 < alpha < 1 or not math.isfinite(alpha) for alpha in config.alphas)
            or config.rul_cap <= 0
            or not config.temporal_windows
            or any(window < 1 for window in config.temporal_windows)
            or config.variance_threshold < 0
            or config.min_observed_cycles < 1
            or not 0 < config.lower_fraction <= config.upper_fraction <= 1
        ):
            raise ValueError("C02 configuration contains invalid values")
        return config

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable resolved configuration."""
        return {
            "experiment_id": self.experiment_id,
            "source": self.source,
            "target": self.target,
            "evaluation_unit": self.evaluation_unit,
            "seeds": list(self.seeds),
            "alphas": list(self.alphas),
            "rul_cap": self.rul_cap,
            "preprocessing": {
                "temporal_windows": list(self.temporal_windows),
                "variance_threshold": self.variance_threshold,
            },
            "calibration": {
                "min_observed_cycles": self.min_observed_cycles,
                "lower_fraction": self.lower_fraction,
                "upper_fraction": self.upper_fraction,
            },
        }


@dataclass(frozen=True)
class C02Result:
    """In-memory C02 endpoint predictions and provenance."""

    predictions: pd.DataFrame
    metrics: dict[str, dict[str, dict[str, float]]]
    partitions: dict[str, list[int]]
    cut_points: dict[int, int]
    quantiles: dict[str, float]
    feature_names: list[str]


def _endpoints(features: pd.DataFrame) -> pd.DataFrame:
    """Return one final observed row per engine in sorted engine order."""
    indices = features.groupby("engine_id", sort=True)["cycle"].idxmax()
    return features.loc[indices, ["engine_id", "cycle"]].reset_index(drop=True)


def _truth_at_endpoints(labeled: pd.DataFrame, endpoints: pd.DataFrame) -> np.ndarray:
    """Align full-trajectory RUL truth to endpoint keys."""
    truth = labeled[["engine_id", "cycle", "rul_capped"]]
    aligned = endpoints.merge(truth, on=["engine_id", "cycle"], how="left", validate="one_to_one")
    if aligned["rul_capped"].isna().any():
        raise ValueError("Endpoint rows could not be aligned to RUL targets")
    return np.asarray(aligned["rul_capped"].to_numpy(dtype="float64"), dtype=np.float64)


def run_c02(
    train: pd.DataFrame,
    test: pd.DataFrame,
    test_rul: pd.DataFrame,
    config: C02Config,
    seed: int,
) -> C02Result:
    """Fit C01 point models, calibrate on truncated training engines, evaluate test endpoints."""
    if seed not in config.seeds:
        raise ValueError(f"Seed {seed} is not declared by the C02 configuration")
    test_ids = [int(value) for value in test["engine_id"].drop_duplicates()]
    if test_ids != list(range(1, len(test_rul) + 1)):
        raise ValueError("Test engine IDs must be ordered and contiguous from 1 through RUL rows")
    partitions = split_engine_ids(train["engine_id"], seed=seed)
    raw_base = train[train["engine_id"].isin(partitions["base_train"])][COLUMNS].copy()
    full_labeled = add_rul_targets(train[COLUMNS].copy(), cap=config.rul_cap)
    base_labeled = add_rul_targets(raw_base, cap=config.rul_cap)
    cut_points = generate_cut_points(
        train,
        partitions["calibration"],
        seed=seed,
        min_observed_cycles=config.min_observed_cycles,
        lower_fraction=config.lower_fraction,
        upper_fraction=config.upper_fraction,
    )
    calibration_raw = restrict_to_cut_points(
        train[train["engine_id"].isin(partitions["calibration"])][COLUMNS], cut_points
    )
    temporal = TemporalFeatureTransformer(
        windows=config.temporal_windows,
        variance_threshold=config.variance_threshold,
    ).fit(raw_base)
    base_features = temporal.transform(raw_base)
    calibration_features = temporal.transform(calibration_raw)
    test_features = temporal.transform(test[COLUMNS])
    if not base_features[["engine_id", "cycle"]].equals(
        raw_base[["engine_id", "cycle"]].reset_index(drop=True)
    ):
        raise ValueError("Training target order does not match transformed feature order")
    models = fit_baseline_models(
        base_features.drop(columns=["engine_id"]),
        base_labeled["rul_capped"],
        random_state=seed,
    )
    calibration_endpoints = _endpoints(calibration_features)
    calibration_truth = _truth_at_endpoints(full_labeled, calibration_endpoints)
    calibration_predictions = predict_baselines(
        models, calibration_features.drop(columns=["engine_id"])
    )
    test_endpoints = _endpoints(test_features)
    if test_endpoints["engine_id"].tolist() != test_ids:
        raise ValueError("Endpoint engine order does not match official RUL order")
    test_truth_raw = test_rul["rul"].astype("float64").to_numpy()
    test_truth = np.clip(test_truth_raw, 0.0, config.rul_cap)
    prediction_frame = pd.DataFrame(
        {"engine_id": test_ids, "y_true_raw": test_truth_raw, "y_true": test_truth}
    )
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    quantiles: dict[str, float] = {}
    for name, calibration_values in calibration_predictions.items():
        test_values = predict_baselines(models, test_features.drop(columns=["engine_id"]))[name]
        calibration_endpoint_values = (
            pd.Series(calibration_values, index=calibration_features.index)
            .loc[calibration_features.groupby("engine_id", sort=True)["cycle"].idxmax()]
            .to_numpy()
        )
        test_endpoint_values = (
            pd.Series(test_values, index=test_features.index)
            .loc[test_features.groupby("engine_id", sort=True)["cycle"].idxmax()]
            .to_numpy()
        )
        prediction_frame[f"{name}_raw"] = test_endpoint_values
        prediction_frame[name] = np.clip(test_endpoint_values, 0.0, config.rul_cap)
        metrics[name] = {}
        for alpha in config.alphas:
            key = f"alpha_{alpha:g}"
            lower, upper, q = split_conformal_intervals(
                calibration_truth,
                np.clip(calibration_endpoint_values, 0.0, config.rul_cap),
                prediction_frame[name].to_numpy(),
                alpha,
            )
            prediction_frame[f"{name}_{key}_lower"] = lower
            prediction_frame[f"{name}_{key}_upper"] = upper
            metrics[name][key] = interval_metrics(
                test_truth,
                lower,
                upper,
                alpha,
                float(config.rul_cap),
            )
            quantiles[f"{name}_{key}"] = q
    return C02Result(
        predictions=prediction_frame,
        metrics=metrics,
        partitions=partitions,
        cut_points=cut_points,
        quantiles=quantiles,
        feature_names=[name for name in temporal.feature_names_out_ if name != "engine_id"],
    )
