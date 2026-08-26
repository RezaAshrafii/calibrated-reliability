"""Tests for the preregistered C06 sensitivity matrix."""

import pytest

from calibrated_reliability.experiments.c06 import C06Config


def c06_text() -> str:
    return """
experiment_id: C06
source: FD001
target: FD001
evaluation_unit: engine_endpoint
seeds: [13, 37, 73, 101, 137]
alphas: [0.10, 0.05]
preprocessing:
  temporal_windows: [5, 10, 20]
  variance_threshold: 0.0
bootstrap:
  n_resamples: 2000
  confidence_level: 0.95
  seed_policy: experiment_seed
conditions:
  - id: primary
    rul_cap: 125
    min_observed_cycles: 30
    lower_fraction: 0.40
    upper_fraction: 0.90
  - id: cap_130
    rul_cap: 130
    min_observed_cycles: 30
    lower_fraction: 0.40
    upper_fraction: 0.90
  - id: early_calibration
    rul_cap: 125
    min_observed_cycles: 30
    lower_fraction: 0.40
    upper_fraction: 0.65
  - id: late_calibration
    rul_cap: 125
    min_observed_cycles: 30
    lower_fraction: 0.65
    upper_fraction: 0.90
"""


def test_c06_config_is_exactly_preregistered() -> None:
    config = C06Config.from_yaml(c06_text())
    assert [condition.id for condition in config.conditions] == [
        "primary",
        "cap_130",
        "early_calibration",
        "late_calibration",
    ]
    assert config.c02_config(config.conditions[1]).rul_cap == 130
    with pytest.raises(ValueError):
        C06Config.from_yaml(c06_text().replace("upper_fraction: 0.65", "upper_fraction: 0.70"))


@pytest.mark.parametrize(
    "replacement",
    ["seeds: [13]", "rul_cap: true", "lower_fraction: '0.40'", "unknown: true"],
)
def test_c06_rejects_design_drift(replacement: str) -> None:
    text = c06_text()
    if replacement == "unknown: true":
        text += "unknown: true\n"
    elif replacement == "seeds: [13]":
        text = text.replace("seeds: [13, 37, 73, 101, 137]", replacement)
    elif replacement == "rul_cap: true":
        text = text.replace("rul_cap: 125", replacement, 1)
    else:
        text = text.replace("lower_fraction: 0.40", replacement, 1)
    with pytest.raises(ValueError):
        C06Config.from_yaml(text)


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        ("alphas: [0.10, 0.05]", 'alphas: ["0.10", 0.05]'),
        ("variance_threshold: 0.0", 'variance_threshold: "0.0"'),
        ("confidence_level: 0.95", 'confidence_level: "0.95"'),
        ("confidence_level: 0.95", "confidence_level: .nan"),
    ],
)
def test_c06_rejects_lossy_or_nonfinite_numeric_values(needle: str, replacement: str) -> None:
    with pytest.raises(ValueError):
        C06Config.from_yaml(c06_text().replace(needle, replacement, 1))
