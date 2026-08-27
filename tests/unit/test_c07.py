"""C07 configuration and frozen-pipeline isolation tests."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from calibrated_reliability.experiments.artifacts import write_c07_run
from calibrated_reliability.experiments.c07 import C07Config, C07Result, run_c07_seed


def _config_text() -> str:
    return (Path("configs/cmapss/regime_scaling.yaml")).read_text(encoding="utf-8")


def test_c07_config_accepts_only_the_preregistered_design() -> None:
    """C07 configuration is exact and rejects silent design changes."""
    config = C07Config.from_yaml(_config_text())
    assert config.targets == ("FD001", "FD002", "FD004")
    assert config.seeds == (13, 37, 73, 101, 137)
    assert config.as_dict()["preprocessing"]["regime_selection"] == "silhouette_auto_2_to_6"
    with pytest.raises(ValueError, match="preregistered"):
        C07Config.from_yaml(_config_text().replace("max_iter: 200", "max_iter: 201"))
    with pytest.raises(ValueError, match="schema"):
        C07Config.from_yaml(
            _config_text().replace(
                "  regime_random_state: 13", "  regime_random_state: 13\n  extra: 1"
            )
        )
    with pytest.raises(ValueError, match="integer"):
        C07Config.from_yaml(_config_text().replace("rul_cap: 125", "rul_cap: '125'"))


def test_c07_seed_fits_once_and_reuses_frozen_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each target receives one already fitted FD001-only pipeline instance."""
    config = SimpleNamespace(targets=("FD001", "FD002", "FD004"), seeds=(13,))
    sentinel = object()
    calls = {"fit": 0, "evaluate": 0}

    def fake_fit(train: object, given_config: object, seed: int) -> object:
        assert train == "FD001 train"
        assert given_config is config
        assert seed == 13
        calls["fit"] += 1
        return sentinel

    def fake_evaluate(pipeline: object, test: object, rul: object, given_config: object) -> object:
        assert pipeline is sentinel
        assert given_config is config
        assert test.startswith("test ")
        assert rul.startswith("rul ")
        calls["evaluate"] += 1
        return {"target_test": test}

    monkeypatch.setattr("calibrated_reliability.experiments.c07.fit_c07_pipeline", fake_fit)
    monkeypatch.setattr(
        "calibrated_reliability.experiments.c07.evaluate_c07_pipeline", fake_evaluate
    )
    target_data = {target: (f"test {target}", f"rul {target}") for target in config.targets}
    result = run_c07_seed("FD001 train", target_data, config, 13)  # type: ignore[arg-type]
    assert result == {target: {"target_test": f"test {target}"} for target in config.targets}
    assert calls == {"fit": 1, "evaluate": 3}


def test_c07_rejects_undeclared_seed_and_reordered_targets() -> None:
    """Callers cannot silently evaluate a changed target matrix or seed set."""
    config = SimpleNamespace(targets=("FD001", "FD002", "FD004"), seeds=(13,))
    with pytest.raises(ValueError, match="declared target order"):
        run_c07_seed("train", {"FD002": ("test", "rul")}, config, 13)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="declared"):
        run_c07_seed(
            "train",
            {target: ("test", "rul") for target in config.targets},
            config,
            37,
        )  # type: ignore[arg-type]


def test_c07_artifact_is_immutable_and_records_regime_provenance(tmp_path: Path) -> None:
    """C07 artifact writing is atomic, immutable, and records fitted scaling state."""
    config = C07Config.from_yaml(_config_text())
    result = C07Result(
        predictions=pd.DataFrame(
            {
                "engine_id": [1],
                "y_true_raw": [4.0],
                "y_true": [4.0],
                "mean_raw": [3.0],
                "mean": [3.0],
            }
        ),
        metrics={"mean": {"rmse": 1.0, "mae": 1.0, "signed_error": -1.0, "nasa_score": 0.1}},
        partitions={"base_train": [1], "calibration": [2], "validation": [3]},
        selected_sensors=["sensor_1"],
        feature_names=["cycle", "sensor_1"],
        regime_metadata={
            "n_regimes": 2,
            "fallback_reason": None,
            "feature_names_in": ["engine_id", "cycle"],
        },
    )
    config_path = Path("configs/cmapss/regime_scaling.yaml")
    registry_path = Path("data/registry.yaml")
    run_dir = write_c07_run(
        tmp_path,
        "FD001",
        13,
        "a" * 40,
        config,
        result,
        config_path,
        registry_path,
        {"train_FD001.txt": {"sha256": "x"}},
    )
    assert (run_dir / "manifest.json").is_file()
    assert "regime_aware_scaling" in (run_dir / "manifest.json").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_c07_run(
            tmp_path,
            "FD001",
            13,
            "a" * 40,
            config,
            result,
            config_path,
            registry_path,
            {"train_FD001.txt": {"sha256": "x"}},
        )
