"""C06 preregistered cap and calibration-policy sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml

from calibrated_reliability.experiments.c02 import (
    C02_ALPHAS,
    C02_BOOTSTRAP_CONFIDENCE_LEVEL,
    C02_BOOTSTRAP_RESAMPLES,
    C02_SEEDS,
    C02_VARIANCE_THRESHOLD,
    C02_WINDOWS,
    C02Config,
    C02Result,
    run_c02,
)


@dataclass(frozen=True)
class C06Condition:
    """One fixed C06 sensitivity condition."""

    id: str
    rul_cap: int
    min_observed_cycles: int
    lower_fraction: float
    upper_fraction: float


C06_CONDITIONS = (
    C06Condition("primary", 125, 30, 0.40, 0.90),
    C06Condition("cap_130", 130, 30, 0.40, 0.90),
    C06Condition("early_calibration", 125, 30, 0.40, 0.65),
    C06Condition("late_calibration", 125, 30, 0.65, 0.90),
)


@dataclass(frozen=True)
class C06Config:
    """Strict executable C06 design."""

    seeds: tuple[int, ...]
    alphas: tuple[float, ...]
    temporal_windows: tuple[int, ...]
    variance_threshold: float
    bootstrap_resamples: int
    bootstrap_confidence_level: float
    conditions: tuple[C06Condition, ...]

    @classmethod
    def from_yaml(cls, text: str) -> C06Config:
        raw = yaml.safe_load(text)
        required = {
            "experiment_id",
            "source",
            "target",
            "evaluation_unit",
            "seeds",
            "alphas",
            "preprocessing",
            "bootstrap",
            "conditions",
        }
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("C06 configuration schema mismatch")
        if (
            raw["experiment_id"] != "C06"
            or raw["source"] != "FD001"
            or raw["target"] != "FD001"
            or raw["evaluation_unit"] != "engine_endpoint"
        ):
            raise ValueError("C06 source, target, or evaluation unit is invalid")
        preprocessing, bootstrap, conditions = (
            raw["preprocessing"],
            raw["bootstrap"],
            raw["conditions"],
        )
        if not isinstance(preprocessing, dict) or set(preprocessing) != {
            "temporal_windows",
            "variance_threshold",
        }:
            raise ValueError("C06 preprocessing schema mismatch")
        if not isinstance(bootstrap, dict) or set(bootstrap) != {
            "n_resamples",
            "confidence_level",
            "seed_policy",
        }:
            raise ValueError("C06 bootstrap schema mismatch")
        if not isinstance(conditions, list) or len(conditions) != len(C06_CONDITIONS):
            raise ValueError("C06 conditions schema mismatch")
        parsed: list[C06Condition] = []
        for value in conditions:
            if not isinstance(value, dict) or set(value) != {
                "id",
                "rul_cap",
                "min_observed_cycles",
                "lower_fraction",
                "upper_fraction",
            }:
                raise ValueError("C06 condition schema mismatch")
            if any(
                isinstance(value[key], bool)
                for key in ("rul_cap", "min_observed_cycles", "lower_fraction", "upper_fraction")
            ):
                raise ValueError("C06 condition values cannot be boolean")
            if not isinstance(value["rul_cap"], int) or not isinstance(
                value["min_observed_cycles"], int
            ):
                raise ValueError("C06 integer condition values are invalid")
            if not isinstance(value["lower_fraction"], (int, float)) or not isinstance(
                value["upper_fraction"], (int, float)
            ):
                raise ValueError("C06 fractional condition values are invalid")
            parsed.append(
                C06Condition(
                    value["id"],
                    value["rul_cap"],
                    value["min_observed_cycles"],
                    float(value["lower_fraction"]),
                    float(value["upper_fraction"]),
                )
            )
        config = cls(
            tuple(raw["seeds"]),
            tuple(float(v) for v in raw["alphas"]),
            tuple(preprocessing["temporal_windows"]),
            float(preprocessing["variance_threshold"]),
            bootstrap["n_resamples"],
            float(bootstrap["confidence_level"]),
            tuple(parsed),
        )
        if (
            config.seeds != C02_SEEDS
            or config.alphas != C02_ALPHAS
            or config.temporal_windows != C02_WINDOWS
            or config.variance_threshold != C02_VARIANCE_THRESHOLD
            or config.bootstrap_resamples != C02_BOOTSTRAP_RESAMPLES
            or config.bootstrap_confidence_level != C02_BOOTSTRAP_CONFIDENCE_LEVEL
            or bootstrap["seed_policy"] != "experiment_seed"
            or config.conditions != C06_CONDITIONS
        ):
            raise ValueError("C06 configuration does not match the preregistered design")
        return config

    def c02_config(self, condition: C06Condition) -> C02Config:
        """Materialize the strictly declared C02-compatible condition."""
        return C02Config(
            "C02",
            "FD001",
            "FD001",
            "engine_endpoint",
            self.seeds,
            self.alphas,
            condition.rul_cap,
            self.temporal_windows,
            self.variance_threshold,
            condition.min_observed_cycles,
            condition.lower_fraction,
            condition.upper_fraction,
            self.bootstrap_resamples,
            self.bootstrap_confidence_level,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": "C06",
            "source": "FD001",
            "target": "FD001",
            "evaluation_unit": "engine_endpoint",
            "seeds": list(self.seeds),
            "alphas": list(self.alphas),
            "preprocessing": {
                "temporal_windows": list(self.temporal_windows),
                "variance_threshold": self.variance_threshold,
            },
            "bootstrap": {
                "n_resamples": self.bootstrap_resamples,
                "confidence_level": self.bootstrap_confidence_level,
                "seed_policy": "experiment_seed",
            },
            "conditions": [condition.__dict__ for condition in self.conditions],
        }


def run_c06_condition(
    train: Any, test: Any, test_rul: Any, config: C06Config, condition: C06Condition, seed: int
) -> C02Result:
    """Run one declared C06 condition without data-driven policy selection."""
    if condition not in config.conditions or seed not in config.seeds:
        raise ValueError("C06 condition or seed is not declared")
    return run_c02(train, test, test_rul, config.c02_config(condition), seed)
