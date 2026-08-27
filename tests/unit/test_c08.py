"""C08 strict-configuration and frozen-pipeline tests."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from calibrated_reliability.experiments.c08 import C08Config, run_c08_seed


def _config_text() -> str:
    return Path("configs/cmapss/adaptive_conformal.yaml").read_text(encoding="utf-8")


def test_c08_config_is_fail_closed() -> None:
    """The ACI protocol cannot silently change through YAML values."""
    config = C08Config.from_yaml(_config_text())
    assert config.targets == ("FD001", "FD002", "FD003", "FD004")
    assert config.gamma == 0.01
    with pytest.raises(ValueError, match="preregistered"):
        C08Config.from_yaml(_config_text().replace("gamma: 0.01", "gamma: 0.02"))
    with pytest.raises(ValueError, match="numeric"):
        C08Config.from_yaml(_config_text().replace("gamma: 0.01", "gamma: '0.01'"))
    with pytest.raises(ValueError, match="schema"):
        C08Config.from_yaml(
            _config_text().replace("  alpha_max: 0.999", "  alpha_max: 0.999\n  extra: 1")
        )


def test_c08_fits_fd001_once_then_reuses_the_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Target domains cannot cause a model or transformer refit."""
    config = SimpleNamespace(
        targets=("FD001", "FD002", "FD003", "FD004"), c02=SimpleNamespace(seeds=(13,))
    )
    sentinel = object()
    calls = {"fit": 0, "evaluate": 0}

    def fake_fit(train: object, c02: object, seed: int) -> object:
        assert train == "fd001 train" and c02 is config.c02 and seed == 13
        calls["fit"] += 1
        return sentinel

    def fake_evaluate(
        pipeline: object, test: object, rul: object, given_config: object, seed: int
    ) -> object:
        assert pipeline is sentinel and given_config is config and seed == 13
        calls["evaluate"] += 1
        return test

    monkeypatch.setattr("calibrated_reliability.experiments.c08.fit_c02_pipeline", fake_fit)
    monkeypatch.setattr(
        "calibrated_reliability.experiments.c08.evaluate_c08_pipeline", fake_evaluate
    )
    target_data = {target: (f"test {target}", f"rul {target}") for target in config.targets}
    assert run_c08_seed("fd001 train", target_data, config, 13) == {
        target: f"test {target}" for target in config.targets
    }  # type: ignore[arg-type]
    assert calls == {"fit": 1, "evaluate": 4}
