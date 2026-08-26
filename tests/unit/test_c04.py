"""Tests for C04 shift-matrix configuration and isolation contract."""

from types import SimpleNamespace

import pytest

from calibrated_reliability.experiments.c04 import (
    C04Config,
    fit_c04_seed,
    run_c04_target_frozen,
)


def test_c04_config_requires_all_target_domains() -> None:
    text = """
experiment_id: C04
source: FD001
targets: [FD001, FD002, FD003, FD004]
evaluation_unit: engine_endpoint
seeds: [13, 37, 73, 101, 137]
alphas: [0.10, 0.05]
rul_cap: 125
preprocessing:
  temporal_windows: [5, 10, 20]
  variance_threshold: 0.0
calibration:
  min_observed_cycles: 30
  lower_fraction: 0.40
  upper_fraction: 0.90
bootstrap:
  n_resamples: 2000
  confidence_level: 0.95
  seed_policy: experiment_seed
"""
    config = C04Config.from_yaml(text)
    assert config.targets == ("FD001", "FD002", "FD003", "FD004")
    assert config.as_dict()["experiment_id"] == "C04"
    with pytest.raises(ValueError):
        C04Config.from_yaml(text.replace("FD004", "FD001"))


def test_c04_frozen_evaluation_does_not_fit_again(monkeypatch: pytest.MonkeyPatch) -> None:
    config = C04Config((), SimpleNamespace(seeds=(13,)))  # type: ignore[arg-type]
    calls = {"fit": 0, "evaluate": 0}
    sentinel = object()

    def fake_fit(source: object, c02: object, seed: int) -> object:
        calls["fit"] += 1
        return sentinel

    def fake_evaluate(
        pipeline: object, test: object, rul: object, c02: object, seed: int
    ) -> object:
        assert pipeline is sentinel
        calls["evaluate"] += 1
        return sentinel

    monkeypatch.setattr("calibrated_reliability.experiments.c04.fit_c02_pipeline", fake_fit)
    monkeypatch.setattr(
        "calibrated_reliability.experiments.c04.evaluate_c02_pipeline", fake_evaluate
    )
    pipeline = fit_c04_seed(object(), config, 13)
    run_c04_target_frozen(pipeline, object(), object(), config, 13)
    run_c04_target_frozen(pipeline, object(), object(), config, 13)
    assert calls == {"fit": 1, "evaluate": 2}
