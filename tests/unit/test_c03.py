"""Tests for C03 conformalized quantile regression."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibrated_reliability.data.loader import COLUMNS
from calibrated_reliability.evaluation.conformal import cqr_conformity_quantile, cqr_intervals
from calibrated_reliability.experiments.artifacts import write_c03_run
from calibrated_reliability.experiments.c03 import C03Config, C03Result, run_c03


def config_text() -> str:
    return """
experiment_id: C03
source: FD001
target: FD001
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
models:
  max_iter: 50
  learning_rate: 0.05
  max_leaf_nodes: 31
  l2_regularization: 1.0
"""


def trajectory(engine_count: int, cycles: int) -> pd.DataFrame:
    rows = []
    for engine_id in range(1, engine_count + 1):
        for cycle in range(1, cycles + 1):
            row = {"engine_id": engine_id, "cycle": cycle}
            row.update({f"op_setting_{i}": float(i == 1) for i in range(1, 4)})
            row.update({f"sensor_{i}": float(engine_id + cycle + i) for i in range(1, 22)})
            rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def test_cqr_quantile_uses_nonnegative_conformity_score() -> None:
    assert cqr_conformity_quantile([10, 20, 30], [9, 19, 31], [11, 21, 32], 0.25) == 1.0
    lower, upper, q = cqr_intervals([10, 20, 30], [9, 19, 31], [11, 21, 32], [8], [12], 0.25)
    assert q == 1.0
    np.testing.assert_allclose(lower, [7.0])
    np.testing.assert_allclose(upper, [13.0])
    with pytest.raises(ValueError, match="cannot exceed"):
        cqr_intervals([10.0], [9.0], [11.0], [12.0], [11.0], 0.10)


def test_c03_is_frozen_and_uses_full_calibration_truth() -> None:
    config = C03Config.from_yaml(config_text())
    assert config.as_dict()["bootstrap"]["seed_policy"] == "experiment_seed"
    result = run_c03(
        trajectory(100, 40), trajectory(2, 50), pd.DataFrame({"rul": [145, 10]}), config, 13
    )
    assert len(result.predictions) == 2
    assert result.predictions.engine_id.tolist() == [1, 2]
    assert result.calibration_scores.rul_raw.gt(0).all()
    assert result.calibration_scores.rul_capped.le(125).all()
    assert set(result.metrics) == {"alpha_0.1", "alpha_0.05"}
    assert all(np.isfinite(value) for value in result.quantiles.values())
    for key in result.metrics:
        assert result.metrics[key]["bootstrap_ci"]


def test_c03_rejects_unknown_configuration_fields() -> None:
    with pytest.raises(ValueError, match="schema"):
        C03Config.from_yaml(config_text().replace("models:\n", "unknown: 1\nmodels:\n"))


def test_c03_rejects_lossy_configuration_values() -> None:
    with pytest.raises(ValueError):
        C03Config.from_yaml(
            config_text().replace("confidence_level: 0.95", 'confidence_level: "0.95"')
        )
    with pytest.raises(ValueError):
        C03Config.from_yaml(config_text().replace("seeds: [13, 37, 73, 101, 137]", "seeds: 13"))


def test_c03_artifact_is_immutable_and_records_provenance(tmp_path: Path) -> None:
    config = C03Config.from_yaml(config_text())
    config_path = tmp_path / "cqr.yaml"
    registry_path = tmp_path / "registry.yaml"
    config_path.write_text(config_text(), encoding="utf-8")
    registry_path.write_text("version: 1\nfiles: []\n", encoding="utf-8")
    result = C03Result(
        predictions=pd.DataFrame({"engine_id": [1], "y_true": [1.0]}),
        calibration_scores=pd.DataFrame({"engine_id": [1], "rul_capped": [1]}),
        metrics={"alpha_0.1": {"coverage": 1.0}},
        partitions={"base_train": [1], "calibration": [2], "validation": [3]},
        cut_points={2: 30},
        quantiles={"alpha_0.1": 1.0},
        feature_names=["cycle"],
    )
    output = tmp_path / "outputs"
    run_dir = write_c03_run(
        output,
        13,
        "a" * 40,
        config,
        result,
        config_path,
        registry_path,
        {"train": {"sha256": "b" * 64}},
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "calibration_scores.csv" in manifest["artifacts"]
    assert manifest["models"] == config.as_dict()["models"]
    with pytest.raises(FileExistsError):
        write_c03_run(
            output,
            13,
            "a" * 40,
            config,
            result,
            config_path,
            registry_path,
            {"train": {"sha256": "b" * 64}},
        )
