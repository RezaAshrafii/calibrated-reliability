"""C08 prequential adaptive conformal evaluation on a frozen C02 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yaml

from calibrated_reliability.data.loader import COLUMNS
from calibrated_reliability.evaluation.conformal import (
    ACIState,
    UnattainableRankPolicy,
    bootstrap_interval_metric_cis,
    interval_metrics,
)
from calibrated_reliability.experiments.c02 import (
    C02Config,
    C02FittedPipeline,
    C02Result,
    _endpoints,
    fit_c02_pipeline,
)
from calibrated_reliability.models.baselines import predict_baselines

C08_TARGETS = ("FD001", "FD002", "FD003", "FD004")
C08_SEEDS = (13, 37, 73, 101, 137)
C08_UNATTAINABLE_RANK_POLICY: UnattainableRankPolicy = "legacy_max_clamp"


@dataclass(frozen=True)
class C08Config:
    """Strict C08 configuration wrapping the frozen C02 design."""

    targets: tuple[str, ...]
    c02: C02Config
    gamma: float
    alpha_min: float
    alpha_max: float
    unattainable_rank_policy: UnattainableRankPolicy

    @classmethod
    def from_yaml(cls, text: str) -> C08Config:
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
            "adaptive",
        }
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("C08 configuration schema mismatch")
        if raw["experiment_id"] != "C08" or raw["source"] != "FD001":
            raise ValueError("C08 experiment or source is invalid")
        if raw["evaluation_unit"] != "engine_endpoint_prequential":
            raise ValueError("C08 evaluation unit is invalid")
        if not isinstance(raw["targets"], list) or tuple(raw["targets"]) != C08_TARGETS:
            raise ValueError("C08 targets must be FD001, FD002, FD003, FD004 in order")
        adaptive = raw["adaptive"]
        if not isinstance(adaptive, dict) or set(adaptive) != {
            "method",
            "unattainable_rank_policy",
            "gamma",
            "alpha_min",
            "alpha_max",
        }:
            raise ValueError("C08 adaptive schema mismatch")
        if any(
            isinstance(adaptive[key], bool) or not isinstance(adaptive[key], (int, float))
            for key in ("gamma", "alpha_min", "alpha_max")
        ):
            raise ValueError("C08 adaptive numeric values are invalid")
        c02_raw = dict(raw)
        c02_raw.pop("targets")
        c02_raw.pop("adaptive")
        c02_raw["experiment_id"] = "C02"
        c02_raw["target"] = "FD001"
        c02_raw["evaluation_unit"] = "engine_endpoint"
        c02 = C02Config.from_yaml(yaml.safe_dump(c02_raw, sort_keys=False))
        config = cls(
            tuple(raw["targets"]),
            c02,
            float(adaptive["gamma"]),
            float(adaptive["alpha_min"]),
            float(adaptive["alpha_max"]),
            C08_UNATTAINABLE_RANK_POLICY,
        )
        if (
            adaptive["method"] != "aci_static_calibration_scores"
            or adaptive["unattainable_rank_policy"] != C08_UNATTAINABLE_RANK_POLICY
            or (
                config.gamma,
                config.alpha_min,
                config.alpha_max,
            )
            != (0.01, 0.001, 0.999)
        ):
            raise ValueError("C08 adaptive policy is not preregistered")
        return config

    def as_dict(self) -> dict[str, Any]:
        result = self.c02.as_dict()
        result.pop("target", None)
        result.update(
            {
                "experiment_id": "C08",
                "targets": list(self.targets),
                "evaluation_unit": "engine_endpoint_prequential",
                "adaptive": {
                    "method": "aci_static_calibration_scores",
                    "unattainable_rank_policy": self.unattainable_rank_policy,
                    "gamma": self.gamma,
                    "alpha_min": self.alpha_min,
                    "alpha_max": self.alpha_max,
                },
            }
        )
        return result


def evaluate_c08_pipeline(
    pipeline: C02FittedPipeline,
    test: pd.DataFrame,
    test_rul: pd.DataFrame,
    config: C08Config,
    seed: int,
) -> C02Result:
    """Evaluate endpoints sequentially; an outcome changes only later intervals."""
    ids = [int(value) for value in test["engine_id"].drop_duplicates()]
    if ids != list(range(1, len(test_rul) + 1)):
        raise ValueError("C08 target engine IDs must be ordered and contiguous")
    features = pipeline.temporal.transform(test[COLUMNS])
    endpoints = _endpoints(features)
    if endpoints["engine_id"].tolist() != ids:
        raise ValueError("C08 endpoint order does not match target RUL order")
    truth_raw = test_rul["rul"].astype("float64").to_numpy()
    truth = np.clip(truth_raw, 0.0, config.c02.rul_cap)
    frame = pd.DataFrame({"engine_id": ids, "y_true_raw": truth_raw, "y_true": truth})
    all_predictions = predict_baselines(pipeline.models, features.drop(columns=["engine_id"]))
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    quantiles: dict[str, Any] = {}
    endpoint_indices = features.groupby("engine_id", sort=True)["cycle"].idxmax()
    for name, values in all_predictions.items():
        raw = pd.Series(values, index=features.index).loc[endpoint_indices].to_numpy()
        frame[f"{name}_raw"] = raw
        frame[name] = np.clip(raw, 0.0, config.c02.rul_cap)
        scores = pipeline.calibration_scores[f"{name}_absolute_residual"].to_numpy()
        metrics[name] = {}
        for alpha in config.c02.alphas:
            key = f"alpha_{alpha:g}"
            state = ACIState(
                scores,
                alpha,
                config.gamma,
                config.alpha_min,
                config.alpha_max,
                unattainable_rank_policy=config.unattainable_rank_policy,
            )
            lower = np.empty(len(truth))
            upper = np.empty(len(truth))
            used = np.empty(len(truth))
            next_values = np.empty(len(truth))
            q = np.empty(len(truth))
            requested_rank = np.empty(len(truth), dtype="int64")
            effective_rank = np.empty(len(truth), dtype="int64")
            attainable = np.empty(len(truth), dtype=bool)
            regime = np.empty(len(truth), dtype=object)
            missed = np.empty(len(truth), dtype=bool)
            for index, center in enumerate(frame[name].to_numpy(dtype="float64")):
                lower[index], upper[index], used[index], q[index] = state.predict_interval(
                    float(center)
                )
                decision = state.last_quantile_result
                if decision is None or decision.effective_rank is None:
                    raise RuntimeError("C08 quantile diagnostics are unavailable")
                requested_rank[index] = decision.requested_rank
                effective_rank[index] = decision.effective_rank
                attainable[index] = decision.finite_rank_attainable
                regime[index] = decision.regime
                missed[index], next_values[index] = state.update(float(truth[index]))
            for suffix, values_out in {
                "lower": lower,
                "upper": upper,
                "alpha_used": used,
                "alpha_next": next_values,
                "quantile": q,
                "requested_rank": requested_rank,
                "effective_rank": effective_rank,
                "finite_rank_attainable": attainable,
                "quantile_regime": regime,
                "miss": missed,
            }.items():
                frame[f"{name}_{key}_{suffix}"] = values_out
            point: dict[str, Any] = interval_metrics(
                truth, lower, upper, alpha, float(config.c02.rul_cap)
            )
            point["bootstrap_ci"] = bootstrap_interval_metric_cis(
                truth,
                lower,
                upper,
                alpha,
                float(config.c02.rul_cap),
                seed,
                config.c02.bootstrap_resamples,
                config.c02.bootstrap_confidence_level,
            )
            metrics[name][key] = point
            quantiles[f"{name}_{key}"] = {
                "initial": float(q[0]),
                "final": float(q[-1]),
                "n_cal": int(len(scores)),
                "unattainable_rank_policy": config.unattainable_rank_policy,
                "unattainable_rank_fraction": float((~attainable).mean()),
                "distinct_quantiles": int(np.unique(q).size),
            }
    return C02Result(
        frame,
        pipeline.calibration_scores.copy(deep=True),
        metrics,
        pipeline.partitions,
        pipeline.cut_points,
        quantiles,
        list(pipeline.feature_names),
    )


def run_c08_seed(
    source_train: pd.DataFrame,
    target_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    config: C08Config,
    seed: int,
) -> dict[str, C02Result]:
    """Fit C02 exactly once per seed, then run online C08 evaluation per target."""
    if seed not in config.c02.seeds or tuple(target_data) != config.targets:
        raise ValueError("C08 seed or target order is not declared")
    pipeline = fit_c02_pipeline(source_train, config.c02, seed)
    return {
        target: evaluate_c08_pipeline(pipeline, test, rul, config, seed)
        for target, (test, rul) in target_data.items()
    }
