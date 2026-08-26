"""Configuration and orchestration helpers for the C04 shift matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml

from calibrated_reliability.experiments.c02 import (
    C02Config,
    C02FittedPipeline,
    C02Result,
    evaluate_c02_pipeline,
    fit_c02_pipeline,
)

C04_TARGETS = ("FD001", "FD002", "FD003", "FD004")
C04_SEEDS = (13, 37, 73, 101, 137)


@dataclass(frozen=True)
class C04Config:
    """Strict shift-matrix configuration with a frozen C02 evaluator."""

    targets: tuple[str, ...]
    c02: C02Config

    @classmethod
    def from_yaml(cls, text: str) -> C04Config:
        raw = yaml.safe_load(text)
        if not isinstance(raw, dict):
            raise ValueError("C04 configuration must be a mapping")
        if set(raw) != {
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
        }:
            raise ValueError("C04 configuration schema mismatch")
        targets = raw["targets"]
        if not isinstance(targets, list) or tuple(targets) != C04_TARGETS:
            raise ValueError("C04 targets must be FD001, FD002, FD003, FD004 in order")
        if raw["source"] != "FD001" or raw["experiment_id"] != "C04":
            raise ValueError("C04 source or experiment is invalid")
        c02_raw = dict(raw)
        c02_raw.pop("targets")
        c02_raw["experiment_id"] = "C02"
        c02_raw["target"] = "FD001"
        c02 = C02Config.from_yaml(yaml.safe_dump(c02_raw, sort_keys=False))
        if c02.seeds != C04_SEEDS:
            raise ValueError("C04 seeds do not match the preregistered design")
        return cls(tuple(targets), c02)

    def as_dict(self) -> dict[str, Any]:
        result = self.c02.as_dict()
        result["experiment_id"] = "C04"
        result["targets"] = list(self.targets)
        return result


def run_c04_target_frozen(
    pipeline: C02FittedPipeline,
    target_test: Any,
    target_rul: Any,
    config: C04Config,
    seed: int,
) -> C02Result:
    """Evaluate one target domain without refitting or recalibrating."""
    if seed not in config.c02.seeds:
        raise ValueError(f"Seed {seed} is not declared by C04")
    return evaluate_c02_pipeline(pipeline, target_test, target_rul, config.c02, seed)


def fit_c04_seed(source_train: Any, config: C04Config, seed: int) -> C02FittedPipeline:
    """Fit the FD001 C02 state exactly once for a C04 seed."""
    return fit_c02_pipeline(source_train, config.c02, seed)


def run_c04_seed(
    source_train: Any,
    target_data: dict[str, tuple[Any, Any]],
    config: C04Config,
    seed: int,
) -> dict[str, C02Result]:
    """Fit once on FD001, then evaluate the same frozen state on every target."""
    if tuple(target_data) != config.targets:
        raise ValueError("C04 target data must match the declared target order")
    pipeline = fit_c04_seed(source_train, config, seed)
    return {
        target: run_c04_target_frozen(pipeline, test, rul, config, seed)
        for target, (test, rul) in target_data.items()
    }
