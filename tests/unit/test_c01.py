"""End-to-end and provenance tests for the corrected C01 experiment."""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import calibrated_reliability.experiments.artifacts as artifacts_module
from calibrated_reliability.data.loader import COLUMNS
from calibrated_reliability.experiments.artifacts import write_c01_run
from calibrated_reliability.experiments.c01 import C01Config, C01Result, run_c01


def config_text(regime_aware: bool = False) -> str:
    """Return a small executable C01 configuration for tests."""
    return f"""
experiment_id: C01
source: FD001
target: FD001
evaluation_unit: engine_endpoint
seeds: [13]
rul_cap: 125
prediction_clip: [0, 125]
training_weighting: cycle
preprocessing:
  temporal_windows: [2]
  variance_threshold: 0.0
  regime_aware_scaling: {str(regime_aware).lower()}
models:
  mean: {{strategy: mean}}
  ridge: {{alpha: 1.0}}
  hist_gradient_boosting:
    max_iter: 2
    learning_rate: 0.05
    max_leaf_nodes: 3
    l2_regularization: 1.0
"""


def trajectory(engine_count: int, cycles: int) -> pd.DataFrame:
    """Create a complete exact-schema C-MAPSS trajectory."""
    rows: list[dict[str, float | int]] = []
    for engine_id in range(1, engine_count + 1):
        for cycle in range(1, cycles + 1):
            row: dict[str, float | int] = {
                "engine_id": engine_id,
                "cycle": cycle,
                "op_setting_1": float(engine_id % 2),
                "op_setting_2": 0.0,
                "op_setting_3": 0.0,
            }
            for sensor in range(1, 22):
                row[f"sensor_{sensor}"] = float(engine_id + cycle + sensor)
            rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def test_c01_caps_truth_and_predictions_and_uses_endpoint_rows() -> None:
    """Primary metrics use capped support and exactly one row per test engine."""
    config = C01Config.from_yaml(config_text())
    result = run_c01(
        trajectory(engine_count=5, cycles=5),
        trajectory(engine_count=2, cycles=3),
        pd.DataFrame({"rul": [145, 10]}),
        config,
        seed=13,
    )
    assert list(result.predictions["engine_id"]) == [1, 2]
    assert list(result.predictions["y_true_raw"]) == [145.0, 10.0]
    assert list(result.predictions["y_true"]) == [125.0, 10.0]
    for model in ["mean", "ridge", "hist_gradient_boosting"]:
        assert result.predictions[model].between(0, 125).all()
        assert f"{model}_raw" in result.predictions
    assert set(result.partitions) == {"base_train", "calibration", "validation"}


def test_c01_rejects_unaligned_test_engine_ids() -> None:
    """Positional RUL labels cannot attach to noncontiguous test IDs."""
    test = trajectory(engine_count=2, cycles=3)
    test["engine_id"] = test["engine_id"].replace({1: 2, 2: 3})
    with pytest.raises(ValueError, match="ordered and contiguous"):
        run_c01(
            trajectory(engine_count=5, cycles=5),
            test,
            pd.DataFrame({"rul": [20, 10]}),
            C01Config.from_yaml(config_text()),
            seed=13,
        )


def test_c01_config_prohibits_regime_scaling_and_undeclared_seed() -> None:
    """The executable configuration preserves the C01/C07 separation."""
    with pytest.raises(ValueError, match="prohibited"):
        C01Config.from_yaml(config_text(regime_aware=True))
    config = C01Config.from_yaml(config_text())
    with pytest.raises(ValueError, match="not declared"):
        run_c01(
            trajectory(engine_count=5, cycles=5),
            trajectory(engine_count=2, cycles=3),
            pd.DataFrame({"rul": [20, 10]}),
            config,
            seed=37,
        )


def test_c01_config_rejects_unknown_fields_and_lossy_integer_coercion() -> None:
    """Executable configuration does not silently accept schema drift."""
    with pytest.raises(ValueError, match="schema mismatch"):
        C01Config.from_yaml(config_text() + "unexpected: true\n")
    with pytest.raises(ValueError, match="seed must be an integer"):
        C01Config.from_yaml(config_text().replace("seeds: [13]", "seeds: [13.5]"))


def test_artifact_manifest_is_complete_and_run_directory_is_immutable(tmp_path: Path) -> None:
    """Artifacts include required provenance and cannot overwrite an existing run."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text(), encoding="utf-8")
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text("version: 1\nfiles: []\n", encoding="utf-8")
    result = C01Result(
        predictions=pd.DataFrame(
            {
                "engine_id": [1],
                "y_true_raw": [145.0],
                "y_true": [125.0],
                "mean_raw": [130.0],
                "mean": [125.0],
            }
        ),
        metrics={"mean": {"rmse": 0.0}},
        partitions={"base_train": [1], "calibration": [2], "validation": [3]},
        selected_sensors=["sensor_1"],
        feature_names=["cycle", "sensor_1"],
    )
    output = tmp_path / "outputs"
    run_dir = write_c01_run(
        output,
        13,
        "a" * 40,
        C01Config.from_yaml(config_text()),
        result,
        config_path,
        registry_path,
        {"train_FD001.txt": {"sha256": "b" * 64}},
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["git"] == {"dirty": False, "sha": "a" * 40}
    assert manifest["lockfile"]["path"] == "uv.lock"
    assert {"configuration", "data", "environment", "lockfile", "split_manifest"}.issubset(
        manifest
    )
    assert {"predictions.csv", "metrics.json", "split_manifest.json", "run.log"}.issubset(
        manifest["artifacts"]
    )
    with pytest.raises(FileExistsError):
        write_c01_run(
            output,
            13,
            "a" * 40,
            C01Config.from_yaml(config_text()),
            result,
            config_path,
            registry_path,
            {"train_FD001.txt": {"sha256": "b" * 64}},
        )


def test_artifact_write_cleans_partial_directory_on_failure(tmp_path: Path) -> None:
    """A failed write leaves no partial run directory or temporary sibling."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text(), encoding="utf-8")
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text("version: 1\nfiles: []\n", encoding="utf-8")
    result = C01Result(
        predictions=pd.DataFrame({"engine_id": [1], "y_true": [125.0]}),
        metrics={"mean": {"rmse": 0.0}},
        partitions={"base_train": [1], "calibration": [2], "validation": [3]},
        selected_sensors=["sensor_1"],
        feature_names=["cycle", "sensor_1"],
    )
    output = tmp_path / "outputs"
    original_write = artifacts_module._write_bytes

    def fail_on_manifest(path: Path, content: bytes) -> str:
        if path.name == "manifest.json":
            raise OSError("simulated manifest failure")
        return original_write(path, content)

    with patch.object(artifacts_module, "_write_bytes", side_effect=fail_on_manifest):
        with pytest.raises(OSError, match="simulated manifest failure"):
            write_c01_run(
                output,
                13,
                "c" * 40,
                C01Config.from_yaml(config_text()),
                result,
                config_path,
                registry_path,
                {"train_FD001.txt": {"sha256": "b" * 64}},
            )
    assert not (output / "C01_cccccccccccc_seed_13").exists()
    assert list(output.iterdir()) == []
