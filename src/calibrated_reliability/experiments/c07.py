"""C07 train-only regime-aware scaling point-baseline experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd
import yaml

from calibrated_reliability.data.labels import add_rul_targets
from calibrated_reliability.data.loader import COLUMNS
from calibrated_reliability.data.splitting import split_engine_ids
from calibrated_reliability.evaluation.metrics import evaluate_endpoint_predictions
from calibrated_reliability.experiments.c01 import _strict_int, _validate_test_alignment
from calibrated_reliability.features.regime import RegimeAwareScaler
from calibrated_reliability.features.temporal import TemporalFeatureTransformer
from calibrated_reliability.models.baselines import fit_baseline_models, predict_baselines

C07_TARGETS = ("FD001", "FD002", "FD004")
C07_SEEDS = (13, 37, 73, 101, 137)
C07_WINDOWS = (5, 10, 20)


def _mapping(value: Any, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} schema mismatch")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    return float(value)


@dataclass(frozen=True)
class C07Config:
    """Fail-closed C07 design configuration."""

    targets: tuple[str, ...]
    seeds: tuple[int, ...]
    rul_cap: int
    clip_min: float
    clip_max: float
    temporal_windows: tuple[int, ...]
    variance_threshold: float
    regime_random_state: int
    ridge_alpha: float
    hgb_max_iter: int
    hgb_learning_rate: float
    hgb_max_leaf_nodes: int
    hgb_l2_regularization: float

    @classmethod
    def from_yaml(cls, text: str) -> C07Config:
        raw = _mapping(
            yaml.safe_load(text),
            {
                "experiment_id",
                "source",
                "targets",
                "evaluation_unit",
                "seeds",
                "rul_cap",
                "prediction_clip",
                "preprocessing",
                "models",
            },
            "C07 configuration",
        )
        preprocessing = _mapping(
            raw["preprocessing"],
            {"temporal_windows", "variance_threshold", "regime_selection", "regime_random_state"},
            "C07 preprocessing",
        )
        models = _mapping(raw["models"], {"mean", "ridge", "hist_gradient_boosting"}, "C07 models")
        mean = _mapping(models["mean"], {"strategy"}, "C07 mean model")
        ridge = _mapping(models["ridge"], {"alpha"}, "C07 ridge model")
        hgb = _mapping(
            models["hist_gradient_boosting"],
            {"max_iter", "learning_rate", "max_leaf_nodes", "l2_regularization"},
            "C07 HGB model",
        )
        if (
            not isinstance(raw["targets"], list)
            or not isinstance(raw["seeds"], list)
            or not isinstance(preprocessing["temporal_windows"], list)
        ):
            raise ValueError("C07 targets, seeds, and temporal windows must be lists")
        if not isinstance(raw["prediction_clip"], list) or len(raw["prediction_clip"]) != 2:
            raise ValueError("C07 prediction_clip must have two values")
        config = cls(
            tuple(raw["targets"]),
            tuple(_strict_int(v, "C07 seed") for v in raw["seeds"]),
            _strict_int(raw["rul_cap"], "C07 rul_cap"),
            _finite_float(raw["prediction_clip"][0], "C07 clip minimum"),
            _finite_float(raw["prediction_clip"][1], "C07 clip maximum"),
            tuple(
                _strict_int(v, "C07 temporal window") for v in preprocessing["temporal_windows"]
            ),
            _finite_float(preprocessing["variance_threshold"], "C07 variance threshold"),
            _strict_int(preprocessing["regime_random_state"], "C07 regime random state"),
            _finite_float(ridge["alpha"], "C07 ridge alpha"),
            _strict_int(hgb["max_iter"], "C07 HGB max_iter"),
            _finite_float(hgb["learning_rate"], "C07 HGB learning rate"),
            _strict_int(hgb["max_leaf_nodes"], "C07 HGB max_leaf_nodes"),
            _finite_float(hgb["l2_regularization"], "C07 HGB l2 regularization"),
        )
        if (
            raw["experiment_id"] != "C07"
            or raw["source"] != "FD001"
            or raw["evaluation_unit"] != "engine_endpoint"
            or config.targets != C07_TARGETS
            or config.seeds != C07_SEEDS
            or config.rul_cap != 125
            or (config.clip_min, config.clip_max) != (0.0, 125.0)
            or config.temporal_windows != C07_WINDOWS
            or config.variance_threshold != 0.0
            or preprocessing["regime_selection"] != "silhouette_auto_2_to_6"
            or config.regime_random_state != 13
            or mean["strategy"] != "mean"
            or config.ridge_alpha != 1.0
            or (
                config.hgb_max_iter,
                config.hgb_learning_rate,
                config.hgb_max_leaf_nodes,
                config.hgb_l2_regularization,
            )
            != (200, 0.05, 15, 0.0)
        ):
            raise ValueError("C07 configuration does not match the preregistered design")
        return config

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": "C07",
            "source": "FD001",
            "targets": list(self.targets),
            "evaluation_unit": "engine_endpoint",
            "seeds": list(self.seeds),
            "rul_cap": self.rul_cap,
            "prediction_clip": [self.clip_min, self.clip_max],
            "preprocessing": {
                "temporal_windows": list(self.temporal_windows),
                "variance_threshold": self.variance_threshold,
                "regime_selection": "silhouette_auto_2_to_6",
                "regime_random_state": self.regime_random_state,
            },
            "models": {
                "mean": {"strategy": "mean"},
                "ridge": {"alpha": self.ridge_alpha},
                "hist_gradient_boosting": {
                    "max_iter": self.hgb_max_iter,
                    "learning_rate": self.hgb_learning_rate,
                    "max_leaf_nodes": self.hgb_max_leaf_nodes,
                    "l2_regularization": self.hgb_l2_regularization,
                },
            },
        }


@dataclass
class C07FittedPipeline:
    temporal: TemporalFeatureTransformer
    scaler: RegimeAwareScaler
    models: dict[str, Any]
    partitions: dict[str, list[int]]
    selected_sensors: list[str]
    feature_names: list[str]


@dataclass(frozen=True)
class C07Result:
    predictions: pd.DataFrame
    metrics: dict[str, dict[str, float]]
    partitions: dict[str, list[int]]
    selected_sensors: list[str]
    feature_names: list[str]
    regime_metadata: dict[str, Any]


def fit_c07_pipeline(train: pd.DataFrame, config: C07Config, seed: int) -> C07FittedPipeline:
    """Fit all C07 learned state exclusively on FD001 base-training engines."""
    if seed not in config.seeds:
        raise ValueError("C07 seed is not declared")
    partitions = split_engine_ids(train["engine_id"], seed=seed)
    raw_base = train[train["engine_id"].isin(partitions["base_train"])][COLUMNS].copy()
    labels = add_rul_targets(raw_base, cap=config.rul_cap)
    temporal = TemporalFeatureTransformer(config.temporal_windows, config.variance_threshold).fit(
        raw_base
    )
    base_features = temporal.transform(raw_base)
    expected = raw_base[["engine_id", "cycle"]].reset_index(drop=True)
    if not base_features[["engine_id", "cycle"]].equals(expected):
        raise ValueError("C07 training target order does not match transformed feature order")
    scaler = RegimeAwareScaler(random_state=config.regime_random_state).fit(base_features)
    scaled_base = scaler.transform(base_features)
    models = fit_baseline_models(
        scaled_base,
        labels["rul_capped"],
        random_state=seed,
        ridge_alpha=config.ridge_alpha,
        hgb_max_iter=config.hgb_max_iter,
        hgb_learning_rate=config.hgb_learning_rate,
        hgb_max_leaf_nodes=config.hgb_max_leaf_nodes,
        hgb_l2_regularization=config.hgb_l2_regularization,
    )
    return C07FittedPipeline(
        temporal,
        scaler,
        models,
        partitions,
        list(temporal.sensor_columns_),
        list(scaler.feature_names_out_),
    )


def evaluate_c07_pipeline(
    pipeline: C07FittedPipeline, test: pd.DataFrame, test_rul: pd.DataFrame, config: C07Config
) -> C07Result:
    """Evaluate an already fitted C07 pipeline without any target-domain fitting."""
    test_ids = _validate_test_alignment(test, test_rul)
    features = pipeline.temporal.transform(test[COLUMNS])
    scaled = pipeline.scaler.transform(features)
    predictions = predict_baselines(pipeline.models, scaled)
    endpoints = features.groupby("engine_id", sort=True)["cycle"].idxmax()
    endpoint_ids = features.loc[endpoints, "engine_id"].astype("int64").tolist()
    if endpoint_ids != test_ids:
        raise ValueError("C07 endpoint engine order does not match official RUL order")
    frame = pd.DataFrame(
        {"engine_id": endpoint_ids, "y_true_raw": test_rul["rul"].astype("float64").to_numpy()}
    )
    frame["y_true"] = frame["y_true_raw"].clip(config.clip_min, config.clip_max)
    metrics: dict[str, dict[str, float]] = {}
    for name, values in predictions.items():
        raw = pd.Series(values, index=features.index).loc[endpoints].to_numpy()
        frame[f"{name}_raw"] = raw
        frame[name] = pd.Series(raw).clip(config.clip_min, config.clip_max)
        metrics[name] = evaluate_endpoint_predictions(
            frame[["engine_id", "y_true", name]].rename(columns={name: "y_pred"})
        )
    return C07Result(
        frame,
        metrics,
        pipeline.partitions,
        pipeline.selected_sensors,
        pipeline.feature_names,
        {
            "n_regimes": pipeline.scaler.n_regimes_,
            "fallback_reason": pipeline.scaler.fallback_reason_,
            "feature_names_in": pipeline.scaler.feature_names_in_,
        },
    )


def run_c07_seed(
    source_train: pd.DataFrame,
    target_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    config: C07Config,
    seed: int,
) -> dict[str, C07Result]:
    """Fit once on FD001 and evaluate the frozen state across C07 targets."""
    if seed not in config.seeds:
        raise ValueError("C07 seed is not declared")
    if tuple(target_data) != config.targets:
        raise ValueError("C07 target data must match the declared target order")
    pipeline = fit_c07_pipeline(source_train, config, seed)
    return {
        target: evaluate_c07_pipeline(pipeline, test, rul, config)
        for target, (test, rul) in target_data.items()
    }
