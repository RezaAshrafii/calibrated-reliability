"""Tests for C04 shift-matrix configuration and isolation contract."""

from types import SimpleNamespace

import pytest

from calibrated_reliability.experiments.c04 import (
    C04Config,
    run_c04_seed,
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


def test_c04_seed_fits_once_and_reuses_one_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = C04Config(("FD001", "FD002", "FD003", "FD004"), SimpleNamespace(seeds=(13,)))  # type: ignore[arg-type]
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
    result = run_c04_seed(
        object(),
        {target: (object(), object()) for target in config.targets},
        config,
        13,
    )
    assert set(result) == set(config.targets)
    assert calls == {"fit": 1, "evaluate": 4}


def test_c04_seed_rejects_missing_or_reordered_target_data() -> None:
    config = C04Config(("FD001", "FD002", "FD003", "FD004"), SimpleNamespace(seeds=(13,)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="declared target order"):
        run_c04_seed(object(), {"FD001": (object(), object())}, config, 13)
