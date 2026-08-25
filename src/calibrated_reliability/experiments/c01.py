"""Pure, testable orchestration for C01 point-prediction baselines."""

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
from calibrated_reliability.features.temporal import TemporalFeatureTransformer
from calibrated_reliability.models.baselines import fit_baseline_models, predict_baselines


def _require_exact_keys(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    """Return a mapping only when its schema exactly matches the contract."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    missing = expected.difference(value)
    extra = set(value).difference(expected)
    if missing or extra:
        raise ValueError(
            f"{name} schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return value


def _strict_int(value: Any, name: str) -> int:
    """Reject lossy integer coercion in executable configuration fields."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


@dataclass(frozen=True)
class C01Config:
    """Validated executable C01 configuration."""

    experiment_id: str
    source: str
    target: str
    evaluation_unit: str
    seeds: tuple[int, ...]
    rul_cap: int
    clip_min: float
    clip_max: float
    temporal_windows: tuple[int, ...]
    variance_threshold: float
    training_weighting: str
    ridge_alpha: float
    hgb_max_iter: int
    hgb_learning_rate: float
    hgb_max_leaf_nodes: int
    hgb_l2_regularization: float

    @classmethod
    def from_yaml(cls, text: str) -> C01Config:
        """Parse and fail closed on the committed C01 configuration schema."""
        raw = yaml.safe_load(text)
        raw = _require_exact_keys(
            raw,
            {
                "experiment_id",
                "source",
                "target",
                "evaluation_unit",
                "seeds",
                "rul_cap",
                "prediction_clip",
                "training_weighting",
                "preprocessing",
                "models",
            },
            "C01 configuration",
        )
        try:
            preprocessing = _require_exact_keys(
                raw["preprocessing"],
                {"temporal_windows", "variance_threshold", "regime_aware_scaling"},
                "preprocessing",
            )
            models = _require_exact_keys(
                raw["models"], {"mean", "ridge", "hist_gradient_boosting"}, "models"
            )
            ridge = _require_exact_keys(models["ridge"], {"alpha"}, "models.ridge")
            hgb = _require_exact_keys(
                models["hist_gradient_boosting"],
                {"max_iter", "learning_rate", "max_leaf_nodes", "l2_regularization"},
                "models.hist_gradient_boosting",
            )
            mean = _require_exact_keys(models["mean"], {"strategy"}, "models.mean")
            clip = raw["prediction_clip"]
            seeds = raw["seeds"]
            windows = preprocessing["temporal_windows"]
            if not isinstance(seeds, list) or not isinstance(windows, list):
                raise ValueError("C01 seeds and temporal windows must be lists")
            if not isinstance(clip, list) or len(clip) != 2:
                raise ValueError("prediction_clip must contain exactly two values")
            config = cls(
                experiment_id=str(raw["experiment_id"]),
                source=str(raw["source"]),
                target=str(raw["target"]),
                evaluation_unit=str(raw["evaluation_unit"]),
                seeds=tuple(_strict_int(seed, "seed") for seed in seeds),
                rul_cap=_strict_int(raw["rul_cap"], "rul_cap"),
                clip_min=float(clip[0]),
                clip_max=float(clip[1]),
                temporal_windows=tuple(
                    _strict_int(window, "temporal window") for window in windows
                ),
                variance_threshold=float(preprocessing["variance_threshold"]),
                training_weighting=str(raw["training_weighting"]),
                ridge_alpha=float(ridge["alpha"]),
                hgb_max_iter=_strict_int(hgb["max_iter"], "hgb.max_iter"),
                hgb_learning_rate=float(hgb["learning_rate"]),
                hgb_max_leaf_nodes=_strict_int(hgb["max_leaf_nodes"], "hgb.max_leaf_nodes"),
                hgb_l2_regularization=float(hgb["l2_regularization"]),
            )
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError("C01 configuration is missing or contains invalid fields") from exc
        if config.experiment_id != "C01" or config.source != "FD001" or config.target != "FD001":
            raise ValueError("C01 source and target must both be FD001")
        if config.evaluation_unit != "engine_endpoint":
            raise ValueError("C01 evaluation unit must be engine_endpoint")
        if not config.seeds or len(set(config.seeds)) != len(config.seeds):
            raise ValueError("C01 seeds must be non-empty and unique")
        if config.rul_cap <= 0 or (config.clip_min, config.clip_max) != (
            0.0,
            float(config.rul_cap),
        ):
            raise ValueError("C01 prediction clipping must be [0, rul_cap]")
        if preprocessing.get("regime_aware_scaling") is not False:
            raise ValueError("Regime-aware scaling is prohibited in C01")
        if config.training_weighting != "cycle":
            raise ValueError("C01 training_weighting must be cycle")
        if mean.get("strategy") != "mean" or set(models) != {
            "mean",
            "ridge",
            "hist_gradient_boosting",
        }:
            raise ValueError("C01 model set does not match ADR-0003")
        if (
            not config.temporal_windows
            or any(window < 1 for window in config.temporal_windows)
            or config.variance_threshold < 0
            or config.ridge_alpha < 0
            or config.hgb_max_iter < 1
            or config.hgb_learning_rate <= 0
            or config.hgb_max_leaf_nodes < 2
            or config.hgb_l2_regularization < 0
            or not all(
                math.isfinite(value)
                for value in (
                    config.variance_threshold,
                    config.ridge_alpha,
                    config.hgb_learning_rate,
                    config.hgb_l2_regularization,
                )
            )
        ):
            raise ValueError("C01 preprocessing and model parameters must be valid")
        return config

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable resolved configuration."""
        return {
            "experiment_id": self.experiment_id,
            "source": self.source,
            "target": self.target,
            "evaluation_unit": self.evaluation_unit,
            "seeds": list(self.seeds),
            "rul_cap": self.rul_cap,
            "prediction_clip": [self.clip_min, self.clip_max],
            "training_weighting": self.training_weighting,
            "preprocessing": {
                "temporal_windows": list(self.temporal_windows),
                "variance_threshold": self.variance_threshold,
                "regime_aware_scaling": False,
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


@dataclass(frozen=True)
class C01Result:
    """In-memory outputs required to build one immutable C01 artifact."""

    predictions: pd.DataFrame
    metrics: dict[str, dict[str, float]]
    partitions: dict[str, list[int]]
    selected_sensors: list[str]
    feature_names: list[str]


def _validate_test_alignment(test: pd.DataFrame, test_rul: pd.DataFrame) -> list[int]:
    """Validate NASA's positional RUL convention before attaching labels."""
    engine_ids = [int(value) for value in test["engine_id"].drop_duplicates()]
    expected = list(range(1, len(test_rul) + 1))
    if engine_ids != expected:
        raise ValueError("Test engine IDs must be ordered and contiguous from 1 through RUL rows")
    return engine_ids


def run_c01(
    train: pd.DataFrame,
    test: pd.DataFrame,
    test_rul: pd.DataFrame,
    config: C01Config,
    seed: int,
) -> C01Result:
    """Run C01 without filesystem side effects."""
    if seed not in config.seeds:
        raise ValueError(f"Seed {seed} is not declared by the C01 configuration")
    test_engine_ids = _validate_test_alignment(test, test_rul)
    partitions = split_engine_ids(train["engine_id"], seed=seed)
    raw_base = train[train["engine_id"].isin(partitions["base_train"])][COLUMNS].copy()
    labeled_base = add_rul_targets(raw_base, cap=config.rul_cap)

    temporal = TemporalFeatureTransformer(
        windows=config.temporal_windows,
        variance_threshold=config.variance_threshold,
    ).fit(raw_base)
    base_features = temporal.transform(raw_base)
    test_features = temporal.transform(test[COLUMNS])
    expected_keys = raw_base[["engine_id", "cycle"]].reset_index(drop=True)
    if not base_features[["engine_id", "cycle"]].equals(expected_keys):
        raise ValueError("Training target order does not match transformed feature order")
    X_train = base_features.drop(columns=["engine_id"])
    X_test = test_features.drop(columns=["engine_id"])
    models = fit_baseline_models(
        X_train,
        labeled_base["rul_capped"],
        random_state=seed,
        ridge_alpha=config.ridge_alpha,
        hgb_max_iter=config.hgb_max_iter,
        hgb_learning_rate=config.hgb_learning_rate,
        hgb_max_leaf_nodes=config.hgb_max_leaf_nodes,
        hgb_l2_regularization=config.hgb_l2_regularization,
    )
    all_predictions = predict_baselines(models, X_test)
    endpoint_indices = test_features.groupby("engine_id", sort=True)["cycle"].idxmax()
    endpoint_ids = test_features.loc[endpoint_indices, "engine_id"].astype("int64").tolist()
    if endpoint_ids != test_engine_ids:
        raise ValueError("Endpoint engine order does not match official RUL order")

    prediction_frame = pd.DataFrame(
        {
            "engine_id": endpoint_ids,
            "y_true_raw": test_rul["rul"].astype("float64").to_numpy(),
        }
    )
    prediction_frame["y_true"] = prediction_frame["y_true_raw"].clip(
        lower=config.clip_min, upper=config.clip_max
    )
    metrics: dict[str, dict[str, float]] = {}
    for name, values in all_predictions.items():
        raw_values = pd.Series(values, index=test_features.index).loc[endpoint_indices].to_numpy()
        prediction_frame[f"{name}_raw"] = raw_values
        prediction_frame[name] = pd.Series(raw_values).clip(
            lower=config.clip_min, upper=config.clip_max
        )
        metrics[name] = evaluate_endpoint_predictions(
            prediction_frame[["engine_id", "y_true", name]].rename(columns={name: "y_pred"})
        )
    return C01Result(
        predictions=prediction_frame,
        metrics=metrics,
        partitions=partitions,
        selected_sensors=list(temporal.sensor_columns_),
        feature_names=[column for column in temporal.feature_names_out_ if column != "engine_id"],
    )
