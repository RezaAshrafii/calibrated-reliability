"""C08 strict-configuration and frozen-pipeline tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from calibrated_reliability.data.loader import COLUMNS
from calibrated_reliability.experiments import artifacts as artifacts_module
from calibrated_reliability.experiments.artifacts import write_c08_run
from calibrated_reliability.experiments.c02 import C02Result
from calibrated_reliability.experiments.c08 import C08Config, evaluate_c08_pipeline, run_c08_seed


def _config_text() -> str:
    return Path("configs/cmapss/adaptive_conformal.yaml").read_text(encoding="utf-8")


def test_c08_config_is_fail_closed() -> None:
    """The ACI protocol cannot silently change through YAML values."""
    config = C08Config.from_yaml(_config_text())
    assert config.targets == ("FD001", "FD002", "FD003", "FD004")
    assert config.gamma == 0.01
    assert config.unattainable_rank_policy == "legacy_max_clamp"
    with pytest.raises(ValueError, match="preregistered"):
        C08Config.from_yaml(_config_text().replace("gamma: 0.01", "gamma: 0.02"))
    with pytest.raises(ValueError, match="numeric"):
        C08Config.from_yaml(_config_text().replace("gamma: 0.01", "gamma: '0.01'"))
    with pytest.raises(ValueError, match="schema"):
        C08Config.from_yaml(
            _config_text().replace("  alpha_max: 0.999", "  alpha_max: 0.999\n  extra: 1")
        )
    with pytest.raises(ValueError, match="preregistered"):
        C08Config.from_yaml(_config_text().replace("legacy_max_clamp", "infinite"))


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


def _endpoint_trajectories(engine_count: int = 3) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for engine_id in range(1, engine_count + 1):
        for cycle in (1, 2):
            row: dict[str, float | int] = {
                "engine_id": engine_id,
                "cycle": cycle,
                "op_setting_1": 0.0,
                "op_setting_2": 0.0,
                "op_setting_3": 0.0,
            }
            row.update({f"sensor_{index}": 0.0 for index in range(1, 22)})
            rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def test_c08_full_evaluator_isolates_future_truth_and_reports_rank_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing the final truth cannot alter its own or any earlier interval decision."""

    class IdentityTemporal:
        def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
            return frame.reset_index(drop=True)

    monkeypatch.setattr(
        "calibrated_reliability.experiments.c08.predict_baselines",
        lambda models, features: {"mean": np.full(len(features), 10.0)},
    )
    calibration_scores = pd.DataFrame({"mean_absolute_residual": np.arange(1.0, 21.0)})
    pipeline = SimpleNamespace(
        temporal=IdentityTemporal(),
        models={"mean": object()},
        calibration_scores=calibration_scores,
        partitions={"base_train": [1], "calibration": [2], "validation": [3]},
        cut_points={2: 30},
        feature_names=["cycle"],
    )
    config = C08Config.from_yaml(_config_text())
    test = _endpoint_trajectories()
    first = evaluate_c08_pipeline(pipeline, test, pd.DataFrame({"rul": [10, 10, 10]}), config, 13)
    second = evaluate_c08_pipeline(
        pipeline, test, pd.DataFrame({"rul": [10, 10, 125]}), config, 13
    )
    pd.testing.assert_frame_equal(first.predictions.iloc[:2], second.predictions.iloc[:2])
    decision_columns = [
        name
        for name in first.predictions
        if name.endswith(("_lower", "_upper", "_alpha_used", "_quantile", "_requested_rank"))
    ]
    pd.testing.assert_series_equal(
        first.predictions.loc[2, decision_columns],
        second.predictions.loc[2, decision_columns],
    )
    assert any(name.endswith("_finite_rank_attainable") for name in first.predictions)
    assert any(name.endswith("_quantile_regime") for name in first.predictions)
    pd.testing.assert_frame_equal(pipeline.calibration_scores, calibration_scores)


def test_c08_artifact_records_fixed_path_bootstrap_and_is_immutable(tmp_path: Path) -> None:
    """C08 manifests make the conditional bootstrap interpretation explicit."""
    config_text = _config_text()
    config_path = tmp_path / "adaptive.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text("version: 1\nfiles: []\n", encoding="utf-8")
    config = C08Config.from_yaml(config_text)
    result = C02Result(
        predictions=pd.DataFrame({"engine_id": [1], "y_true": [1.0]}),
        calibration_scores=pd.DataFrame({"engine_id": [1]}),
        metrics={},
        partitions={"base_train": [1], "calibration": [2], "validation": [3]},
        cut_points={2: 30},
        quantiles={},
        feature_names=["cycle"],
    )
    output = tmp_path / "outputs"
    run_dir = write_c08_run(
        output, "FD001", 13, "a" * 40, config, result, config_path, registry_path, {}
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["bootstrap_interpretation"] == (
        "conditional_fixed_path_summary; ACI trajectory not rerun"
    )
    with pytest.raises(FileExistsError):
        write_c08_run(
            output, "FD001", 13, "a" * 40, config, result, config_path, registry_path, {}
        )


def test_c08_artifact_write_cleans_partial_directory_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed C08 write leaves no published or temporary directory."""
    config_text = _config_text()
    config_path = tmp_path / "adaptive.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text("version: 1\nfiles: []\n", encoding="utf-8")
    config = C08Config.from_yaml(config_text)
    result = C02Result(
        predictions=pd.DataFrame({"engine_id": [1], "y_true": [1.0]}),
        calibration_scores=pd.DataFrame({"engine_id": [1]}),
        metrics={},
        partitions={},
        cut_points={},
        quantiles={},
        feature_names=["cycle"],
    )
    original_write = artifacts_module._write_bytes

    def fail_on_metrics(path: Path, content: bytes) -> str:
        if path.name == "metrics.json":
            raise OSError("simulated C08 artifact failure")
        return original_write(path, content)

    monkeypatch.setattr(artifacts_module, "_write_bytes", fail_on_metrics)
    output = tmp_path / "outputs"
    with pytest.raises(OSError, match="simulated C08 artifact failure"):
        write_c08_run(
            output, "FD001", 13, "a" * 40, config, result, config_path, registry_path, {}
        )
    assert list(output.iterdir()) == []
