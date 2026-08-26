"""C05 weighted split-conformal evaluation under C-MAPSS shift."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from calibrated_reliability.data.loader import COLUMNS, OP_SETTING_COLUMNS
from calibrated_reliability.evaluation.conformal import (
    bootstrap_interval_metric_cis,
    interval_metrics,
)
from calibrated_reliability.experiments.c02 import (
    C02FittedPipeline,
    C02Result,
    _endpoints,
)
from calibrated_reliability.models.baselines import predict_baselines

C05_TARGETS = ("FD002", "FD003", "FD004")


@dataclass(frozen=True)
class C05Config:
    """Strict executable C05 design."""

    targets: tuple[str, ...]
    c02: Any
    weighting_method: str
    weighting_features: tuple[str, ...]
    logistic_c: float
    logistic_max_iter: int
    weight_clip_min: float
    weight_clip_max: float

    @classmethod
    def from_yaml(cls, text: str) -> C05Config:
        from calibrated_reliability.experiments.c02 import C02Config

        raw = yaml.safe_load(text)
        required = {
            "experiment_id",
            "source",
            "targets",
            "evaluation_unit",
            "seeds",
            "alphas",
            "rul_cap",
            "preprocessing",
            "calibration",
            "bootstrap",
            "weighting",
        }
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("C05 configuration schema mismatch")
        if (
            raw["experiment_id"] != "C05"
            or raw["source"] != "FD001"
            or tuple(raw["targets"]) != C05_TARGETS
        ):
            raise ValueError("C05 source or targets are invalid")
        weighting = raw["weighting"]
        if not isinstance(weighting, dict) or set(weighting) != {
            "method",
            "features",
            "logistic_c",
            "max_iter",
            "clip_min",
            "clip_max",
        }:
            raise ValueError("C05 weighting schema mismatch")
        if weighting["method"] != "logistic_density_ratio" or tuple(
            weighting["features"]
        ) != tuple(OP_SETTING_COLUMNS):
            raise ValueError("C05 weighting design is not preregistered")
        if (
            isinstance(weighting["logistic_c"], bool)
            or not isinstance(weighting["logistic_c"], (int, float))
            or float(weighting["logistic_c"]) <= 0
            or isinstance(weighting["max_iter"], bool)
            or not isinstance(weighting["max_iter"], int)
            or weighting["max_iter"] < 1
        ):
            raise ValueError("C05 logistic configuration is invalid")
        if (
            not isinstance(weighting["clip_min"], (int, float))
            or not isinstance(weighting["clip_max"], (int, float))
            or not 0 < float(weighting["clip_min"]) < float(weighting["clip_max"]) <= 1.0
        ):
            raise ValueError("C05 clipping configuration is invalid")
        c02_raw = dict(raw)
        c02_raw.pop("targets")
        c02_raw.pop("weighting")
        c02_raw["experiment_id"] = "C02"
        c02_raw["target"] = "FD001"
        c02 = C02Config.from_yaml(yaml.safe_dump(c02_raw, sort_keys=False))
        return cls(
            C05_TARGETS,
            c02,
            weighting["method"],
            tuple(weighting["features"]),
            float(weighting["logistic_c"]),
            weighting["max_iter"],
            float(weighting["clip_min"]),
            float(weighting["clip_max"]),
        )

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = dict(self.c02.as_dict())
        result["experiment_id"] = "C05"
        result["targets"] = list(self.targets)
        result["weighting"] = {
            "method": self.weighting_method,
            "features": list(self.weighting_features),
            "logistic_c": self.logistic_c,
            "max_iter": self.logistic_max_iter,
            "clip_min": self.weight_clip_min,
            "clip_max": self.weight_clip_max,
        }
        return result


def _weighted_quantile(scores: Any, weights: Any, test_weight: float, alpha: float) -> float:
    values = np.asarray(scores, dtype="float64")
    calibration_weights = np.asarray(weights, dtype="float64")
    if len(values) == 0 or len(values) != len(calibration_weights):
        raise ValueError("Weighted scores and weights must have equal non-zero length")
    if not np.isfinite(values).all() or not np.isfinite(calibration_weights).all():
        raise ValueError("Weighted scores and weights must be finite")
    if (values < 0).any() or (calibration_weights <= 0).any() or test_weight <= 0:
        raise ValueError("Scores and weights must be nonnegative and weights positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(calibration_weights[order])
    threshold = (1.0 - alpha) * (calibration_weights.sum() + test_weight)
    index = int(np.searchsorted(cumulative, threshold, side="left"))
    if index == len(order):
        return float("inf")
    return float(values[order[index]])


def _density_ratio_details(
    source: pd.DataFrame, target: pd.DataFrame, logistic_c: float, max_iter: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Estimate target/source density ratios from operating settings only."""
    x_source = source[OP_SETTING_COLUMNS].to_numpy(dtype="float64")
    x_target = target[OP_SETTING_COLUMNS].to_numpy(dtype="float64")
    scaler = StandardScaler().fit(np.vstack([x_source, x_target]))
    x = scaler.transform(np.vstack([x_source, x_target]))
    labels = np.concatenate([np.zeros(len(x_source)), np.ones(len(x_target))])
    model = LogisticRegression(
        random_state=0, max_iter=max_iter, solver="lbfgs", C=logistic_c
    ).fit(x, labels)
    probabilities = model.predict_proba(x)[:, 1]
    odds = np.clip(probabilities, 1e-8, 1.0 - 1e-8) / np.clip(1.0 - probabilities, 1e-8, 1.0)
    ratios = odds * (len(x_source) / len(x_target))
    details = {
        "features": list(OP_SETTING_COLUMNS),
        "logistic_c": logistic_c,
        "max_iter": max_iter,
        "coef": model.coef_.ravel().tolist(),
        "intercept": float(model.intercept_[0]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "source_count": len(x_source),
        "target_count": len(x_target),
    }
    return ratios[: len(x_source)], ratios[len(x_source) :], details


def _density_ratio(source: pd.DataFrame, target: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Estimate deterministic target/source ratios for unit-level tests."""
    source_weights, target_weights, _ = _density_ratio_details(source, target, 1.0, 1000)
    return source_weights, target_weights


def _clip_and_normalize_weights(
    calibration_weights: np.ndarray,
    target_weights: np.ndarray,
    clip_min: float,
    clip_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Bound density ratios and normalize calibration mass to its sample count."""
    clipped_calibration = np.clip(calibration_weights, clip_min, clip_max)
    scale = float(clipped_calibration.mean())
    normalized_calibration = clipped_calibration / scale
    normalized_target = np.clip(target_weights / scale, clip_min, clip_max)
    return normalized_calibration, normalized_target


def run_c05_target(
    pipeline: C02FittedPipeline,
    target_test: pd.DataFrame,
    target_rul: pd.DataFrame,
    config: Any,
    seed: int,
) -> C02Result:
    """Apply weighted conformal calibration without refitting the frozen C02 state."""
    test_ids = [int(value) for value in target_test["engine_id"].drop_duplicates()]
    if test_ids != list(range(1, len(target_rul) + 1)):
        raise ValueError(
            "Target engine IDs must be ordered and contiguous from 1 through RUL rows"
        )
    features = pipeline.temporal.transform(target_test[COLUMNS])
    endpoints = _endpoints(features)
    if endpoints["engine_id"].tolist() != test_ids:
        raise ValueError("Target endpoint order does not match target RUL order")
    endpoint_settings = features.loc[
        features.groupby("engine_id", sort=True)["cycle"].idxmax(), OP_SETTING_COLUMNS
    ].reset_index(drop=True)
    calibration_weights, target_weights, density_ratio = _density_ratio_details(
        pipeline.calibration_operating_settings,
        endpoint_settings,
        config.logistic_c,
        config.logistic_max_iter,
    )
    calibration_weights, target_weights = _clip_and_normalize_weights(
        calibration_weights,
        target_weights,
        config.weight_clip_min,
        config.weight_clip_max,
    )
    truth_raw = target_rul["rul"].astype("float64").to_numpy()
    truth = np.clip(truth_raw, 0.0, config.c02.rul_cap)
    frame = pd.DataFrame({"engine_id": test_ids, "y_true_raw": truth_raw, "y_true": truth})
    predictions = predict_baselines(pipeline.models, features.drop(columns=["engine_id"]))
    calibration_scores = pipeline.calibration_scores.copy(deep=True)
    calibration_scores["density_ratio_weight"] = calibration_weights
    frame["density_ratio_weight"] = target_weights
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    quantiles: dict[str, Any] = {}
    for name, values in predictions.items():
        endpoint_values = (
            pd.Series(values, index=features.index)
            .loc[features.groupby("engine_id", sort=True)["cycle"].idxmax()]
            .to_numpy()
        )
        frame[f"{name}_raw"] = endpoint_values
        frame[name] = np.clip(endpoint_values, 0.0, config.c02.rul_cap)
        metrics[name] = {}
        for alpha in config.c02.alphas:
            key = f"alpha_{alpha:g}"
            q = np.array(
                [
                    _weighted_quantile(
                        calibration_scores[f"{name}_absolute_residual"],
                        calibration_weights,
                        float(weight),
                        alpha,
                    )
                    for weight in target_weights
                ]
            )
            if not np.isfinite(q).all():
                raise ValueError(
                    "Weighted conformal quantile is infinite; finite C05 interval unavailable"
                )
            frame[f"{name}_{key}_quantile"] = q
            quantiles[f"{name}_{key}"] = {
                "min": float(q.min()),
                "max": float(q.max()),
                "unique_count": int(np.unique(q).size),
            }
            lower, upper = frame[name].to_numpy() - q, frame[name].to_numpy() + q
            frame[f"{name}_{key}_lower"] = lower
            frame[f"{name}_{key}_upper"] = upper
            point: dict[str, Any] = interval_metrics(
                truth, lower, upper, alpha, float(config.c02.rul_cap)
            )
            point["bootstrap_ci"] = bootstrap_interval_metric_cis(
                truth,
                lower,
                upper,
                alpha,
                float(config.c02.rul_cap),
                seed=seed,
                n_resamples=config.c02.bootstrap_resamples,
                confidence_level=config.c02.bootstrap_confidence_level,
            )
            metrics[name][key] = point
    return C02Result(
        predictions=frame,
        calibration_scores=calibration_scores,
        metrics=metrics,
        partitions=pipeline.partitions,
        cut_points=pipeline.cut_points,
        quantiles=quantiles,
        feature_names=list(pipeline.feature_names),
        weighting={
            **density_ratio,
            "calibration_weight_sum": float(calibration_weights.sum()),
            "target_weight_min": float(target_weights.min()),
            "target_weight_max": float(target_weights.max()),
            "target_weight_mean": float(target_weights.mean()),
            "calibration_effective_sample_size": float(
                calibration_weights.sum() ** 2 / np.square(calibration_weights).sum()
            ),
            "clip_min": config.weight_clip_min,
            "clip_max": config.weight_clip_max,
        },
    )
