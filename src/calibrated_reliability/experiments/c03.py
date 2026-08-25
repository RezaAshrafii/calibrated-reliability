"""Pure orchestration for C03 conformalized quantile regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor

from calibrated_reliability.data.labels import add_rul_targets
from calibrated_reliability.data.loader import COLUMNS
from calibrated_reliability.data.splitting import (
    generate_cut_points,
    restrict_to_cut_points,
    split_engine_ids,
)
from calibrated_reliability.evaluation.conformal import (
    bootstrap_interval_metric_cis,
    cqr_intervals,
    interval_metrics,
)
from calibrated_reliability.features.temporal import TemporalFeatureTransformer

C03_SEEDS = (13, 37, 73, 101, 137)
C03_ALPHAS = (0.10, 0.05)
C03_RUL_CAP = 125
C03_WINDOWS = (5, 10, 20)
C03_VARIANCE_THRESHOLD = 0.0
C03_MIN_OBSERVED_CYCLES = 30
C03_LOWER_FRACTION = 0.40
C03_UPPER_FRACTION = 0.90
C03_BOOTSTRAP_RESAMPLES = 2000
C03_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
C03_BOOTSTRAP_SEED_POLICY = "experiment_seed"
C03_MODEL_SPEC = {
    "max_iter": 50,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "l2_regularization": 1.0,
}


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


@dataclass(frozen=True)
class C03Config:
    """Validated C03 configuration."""

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
    bootstrap_resamples: int
    bootstrap_confidence_level: float
    model_spec: dict[str, Any]

    @classmethod
    def from_yaml(cls, text: str) -> C03Config:
        raw = yaml.safe_load(text)
        if not isinstance(raw, dict):
            raise ValueError("C03 configuration must be a mapping")
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
            "bootstrap",
            "models",
        }
        if set(raw) != expected:
            raise ValueError("C03 configuration schema mismatch")
        preprocessing = raw["preprocessing"]
        calibration = raw["calibration"]
        bootstrap = raw["bootstrap"]
        models = raw["models"]
        if not isinstance(preprocessing, dict) or set(preprocessing) != {
            "temporal_windows",
            "variance_threshold",
        }:
            raise ValueError("C03 preprocessing schema mismatch")
        if not isinstance(calibration, dict) or set(calibration) != {
            "min_observed_cycles",
            "lower_fraction",
            "upper_fraction",
        }:
            raise ValueError("C03 calibration schema mismatch")
        if not isinstance(bootstrap, dict) or set(bootstrap) != {
            "n_resamples",
            "confidence_level",
            "seed_policy",
        }:
            raise ValueError("C03 bootstrap schema mismatch")
        if not isinstance(models, dict) or set(models) != set(C03_MODEL_SPEC):
            raise ValueError("C03 model schema mismatch")
        seeds = raw["seeds"]
        alphas = raw["alphas"]
        windows = preprocessing["temporal_windows"]
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in seeds):
            raise ValueError("C03 seeds must be integers")
        config = cls(
            experiment_id=str(raw["experiment_id"]),
            source=str(raw["source"]),
            target=str(raw["target"]),
            evaluation_unit=str(raw["evaluation_unit"]),
            seeds=tuple(seeds),
            alphas=tuple(float(x) for x in alphas),
            rul_cap=_strict_int(raw["rul_cap"], "rul_cap"),
            temporal_windows=tuple(_strict_int(x, "temporal window") for x in windows),
            variance_threshold=float(preprocessing["variance_threshold"]),
            min_observed_cycles=_strict_int(
                calibration["min_observed_cycles"], "min_observed_cycles"
            ),
            lower_fraction=float(calibration["lower_fraction"]),
            upper_fraction=float(calibration["upper_fraction"]),
            bootstrap_resamples=_strict_int(bootstrap["n_resamples"], "bootstrap.n_resamples"),
            bootstrap_confidence_level=float(bootstrap["confidence_level"]),
            model_spec=dict(models),
        )
        if (
            config.experiment_id != "C03"
            or config.source != "FD001"
            or config.target != "FD001"
            or config.evaluation_unit != "engine_endpoint"
            or config.seeds != C03_SEEDS
            or config.alphas != C03_ALPHAS
            or config.rul_cap != C03_RUL_CAP
            or config.temporal_windows != C03_WINDOWS
            or config.variance_threshold != C03_VARIANCE_THRESHOLD
            or config.min_observed_cycles != C03_MIN_OBSERVED_CYCLES
            or config.lower_fraction != C03_LOWER_FRACTION
            or config.upper_fraction != C03_UPPER_FRACTION
            or config.bootstrap_resamples != C03_BOOTSTRAP_RESAMPLES
            or config.bootstrap_confidence_level != C03_BOOTSTRAP_CONFIDENCE_LEVEL
            or bootstrap["seed_policy"] != C03_BOOTSTRAP_SEED_POLICY
            or config.model_spec != C03_MODEL_SPEC
        ):
            raise ValueError("C03 configuration does not match the preregistered design")
        return config

    def as_dict(self) -> dict[str, Any]:
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
            "bootstrap": {
                "n_resamples": self.bootstrap_resamples,
                "confidence_level": self.bootstrap_confidence_level,
                "seed_policy": C03_BOOTSTRAP_SEED_POLICY,
            },
            "models": self.model_spec,
        }


@dataclass(frozen=True)
class C03Result:
    predictions: pd.DataFrame
    calibration_scores: pd.DataFrame
    metrics: dict[str, dict[str, Any]]
    partitions: dict[str, list[int]]
    cut_points: dict[int, int]
    quantiles: dict[str, float]
    feature_names: list[str]


def _endpoints(features: pd.DataFrame) -> pd.DataFrame:
    indices = features.groupby("engine_id", sort=True)["cycle"].idxmax()
    return features.loc[indices, ["engine_id", "cycle"]].reset_index(drop=True)


def _fit_quantile_models(X: pd.DataFrame, y: Any, config: C03Config, seed: int) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for alpha in config.alphas:
        key = f"alpha_{alpha:g}"
        for side, quantile in (("lower", alpha / 2), ("upper", 1 - alpha / 2)):
            model = HistGradientBoostingRegressor(
                loss="quantile", quantile=quantile, random_state=seed, **config.model_spec
            )
            model.fit(X, y)
            models[f"{key}_{side}"] = model
    return models


def run_c03(
    train: pd.DataFrame, test: pd.DataFrame, test_rul: pd.DataFrame, config: C03Config, seed: int
) -> C03Result:
    """Fit CQR models on base-train rows and evaluate official endpoints."""
    if seed not in config.seeds:
        raise ValueError(f"Seed {seed} is not declared by the C03 configuration")
    test_ids = [int(x) for x in test["engine_id"].drop_duplicates()]
    if test_ids != list(range(1, len(test_rul) + 1)):
        raise ValueError("Test engine IDs must be ordered and contiguous from 1 through RUL rows")
    partitions = split_engine_ids(train["engine_id"], seed=seed)
    raw_base = train[train.engine_id.isin(partitions["base_train"])][COLUMNS].copy()
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
        train[train.engine_id.isin(partitions["calibration"])][COLUMNS], cut_points
    )
    temporal = TemporalFeatureTransformer(
        windows=config.temporal_windows, variance_threshold=config.variance_threshold
    ).fit(raw_base)
    base_features = temporal.transform(raw_base)
    calibration_features = temporal.transform(calibration_raw)
    test_features = temporal.transform(test[COLUMNS])
    models = _fit_quantile_models(
        base_features.drop(columns=["engine_id"]), base_labeled.rul_capped, config, seed
    )
    cal_ep = _endpoints(calibration_features)
    test_ep = _endpoints(test_features)
    if test_ep.engine_id.tolist() != test_ids:
        raise ValueError("Endpoint engine order does not match official RUL order")
    truth = full_labeled.merge(
        cal_ep, on=["engine_id", "cycle"], how="right", validate="one_to_one"
    )
    if truth.rul_capped.isna().any():
        raise ValueError("Calibration endpoints could not be aligned to full-trajectory truth")
    test_truth_raw = test_rul.rul.astype("float64").to_numpy()
    test_truth = np.clip(test_truth_raw, 0, config.rul_cap)
    predictions = pd.DataFrame(
        {"engine_id": test_ids, "y_true_raw": test_truth_raw, "y_true": test_truth}
    )
    scores = truth[["engine_id", "cycle", "rul_raw", "rul_capped"]].copy()
    metrics: dict[str, dict[str, Any]] = {}
    quantiles: dict[str, float] = {}
    for alpha in config.alphas:
        key = f"alpha_{alpha:g}"
        lower_cal = models[f"{key}_lower"].predict(
            calibration_features.drop(columns=["engine_id"])
        )
        upper_cal = models[f"{key}_upper"].predict(
            calibration_features.drop(columns=["engine_id"])
        )
        lower_test = models[f"{key}_lower"].predict(test_features.drop(columns=["engine_id"]))
        upper_test = models[f"{key}_upper"].predict(test_features.drop(columns=["engine_id"]))
        cal_idx = calibration_features.groupby("engine_id", sort=True)["cycle"].idxmax()
        test_idx = test_features.groupby("engine_id", sort=True)["cycle"].idxmax()
        lower_cal_ep = (
            pd.Series(lower_cal, index=calibration_features.index).loc[cal_idx].to_numpy()
        )
        upper_cal_ep = (
            pd.Series(upper_cal, index=calibration_features.index).loc[cal_idx].to_numpy()
        )
        lower_test_ep = pd.Series(lower_test, index=test_features.index).loc[test_idx].to_numpy()
        upper_test_ep = pd.Series(upper_test, index=test_features.index).loc[test_idx].to_numpy()
        lower, upper, q = cqr_intervals(
            truth.rul_capped, lower_cal_ep, upper_cal_ep, lower_test_ep, upper_test_ep, alpha
        )
        scores[f"{key}_lower_raw"] = lower_cal_ep
        scores[f"{key}_upper_raw"] = upper_cal_ep
        scores[f"{key}_conformity"] = np.maximum(
            np.maximum(lower_cal_ep - truth.rul_capped, truth.rul_capped - upper_cal_ep), 0
        )
        predictions[f"{key}_lower_raw"] = lower_test_ep
        predictions[f"{key}_upper_raw"] = upper_test_ep
        predictions[f"{key}_lower"] = lower
        predictions[f"{key}_upper"] = upper
        quantiles[key] = q
        point: dict[str, Any] = interval_metrics(
            test_truth, lower, upper, alpha, float(config.rul_cap)
        )
        point["bootstrap_ci"] = bootstrap_interval_metric_cis(
            test_truth,
            lower,
            upper,
            alpha,
            float(config.rul_cap),
            seed,
            config.bootstrap_resamples,
            config.bootstrap_confidence_level,
        )
        metrics[key] = point
    return C03Result(
        predictions,
        scores,
        metrics,
        partitions,
        cut_points,
        quantiles,
        [x for x in temporal.feature_names_out_ if x != "engine_id"],
    )
