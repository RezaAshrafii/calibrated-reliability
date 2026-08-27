"""Tests for finite-sample split-conformal intervals and C02 orchestration."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibrated_reliability.data.loader import COLUMNS
from calibrated_reliability.evaluation.conformal import (
    aci_prequential_intervals,
    bootstrap_interval_metric_cis,
    conformal_quantile,
    interval_metrics,
    split_conformal_intervals,
)
from calibrated_reliability.experiments import artifacts as artifacts_module
from calibrated_reliability.experiments.artifacts import write_c02_run
from calibrated_reliability.experiments.c02 import C02Config, C02Result, run_c02


def config_text() -> str:
    """Return a small C02 configuration for synthetic tests."""
    return """
experiment_id: C02
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
"""


def trajectory(engine_count: int, cycles: int) -> pd.DataFrame:
    """Create exact-schema synthetic trajectories."""
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


def test_conformal_quantile_uses_finite_sample_rank() -> None:
    """The quantile is the declared order statistic, not an interpolated percentile."""
    assert conformal_quantile([1.0, 2.0, 5.0], alpha=0.10) == 5.0
    assert conformal_quantile([1.0, 2.0, 5.0], alpha=0.50) == 2.0
    with pytest.raises(ValueError, match="strictly between"):
        conformal_quantile([1.0], alpha=1.0)


def test_intervals_and_metrics_are_finite_and_directionally_correct() -> None:
    """Intervals are symmetric and interval score penalizes misses."""
    lower, upper, q = split_conformal_intervals([10.0, 12.0], [9.0, 13.0], [11.0], 0.10)
    assert q == 1.0
    np.testing.assert_allclose(lower, [10.0])
    np.testing.assert_allclose(upper, [12.0])
    metrics = interval_metrics([15.0], lower, upper, 0.10, 125.0)
    assert metrics["coverage"] == 0.0
    assert metrics["mean_width"] == 2.0
    assert metrics["normalized_interval_score"] > 0.0


def test_interval_score_and_bootstrap_are_exact_and_deterministic() -> None:
    """Both miss directions receive the declared interval-score penalty."""
    metrics = interval_metrics([0.0, 10.0], [1.0, 8.0], [3.0, 9.0], 0.10, 125.0)
    assert metrics == {
        "coverage": 0.0,
        "mean_width": 1.5,
        "normalized_interval_score": pytest.approx(21.5 / 125.0),
    }
    first = bootstrap_interval_metric_cis(
        [0.0, 10.0], [1.0, 8.0], [3.0, 9.0], 0.10, 125.0, seed=13
    )
    second = bootstrap_interval_metric_cis(
        [0.0, 10.0], [1.0, 8.0], [3.0, 9.0], 0.10, 125.0, seed=13
    )
    assert first == second
    lower, upper, _ = split_conformal_intervals([0.0], [10.0], [0.0], 0.10)
    np.testing.assert_allclose(lower, [-10.0])
    np.testing.assert_allclose(upper, [10.0])


def test_c02_config_rejects_undeclared_design_values() -> None:
    """Sensitivity settings cannot masquerade as the preregistered C02 run."""
    for changed in [
        config_text().replace("alphas: [0.10, 0.05]", "alphas: [0.20]"),
        config_text().replace("rul_cap: 125", "rul_cap: 130"),
        config_text().replace("seeds: [13, 37, 73, 101, 137]", "seeds: [999]"),
        config_text().replace("n_resamples: 2000", "n_resamples: 1000"),
        config_text().replace("seed_policy: experiment_seed", "seed_policy: fixed_42"),
        config_text().replace("confidence_level: 0.95", 'confidence_level: "0.95"'),
    ]:
        with pytest.raises(ValueError):
            C02Config.from_yaml(changed)


def test_c02_uses_full_training_rul_at_truncated_calibration_endpoint() -> None:
    """Calibration truth is remaining life, not zero at the observed cut point."""
    result = run_c02(
        trajectory(10, 100),
        trajectory(2, 50),
        pd.DataFrame({"rul": [145, 10]}),
        C02Config.from_yaml(config_text()),
        seed=13,
    )
    assert len(result.predictions) == 2
    assert list(result.predictions["y_true"]) == [125.0, 10.0]
    assert set(result.quantiles) == {
        "mean_alpha_0.1",
        "mean_alpha_0.05",
        "ridge_alpha_0.1",
        "ridge_alpha_0.05",
        "hist_gradient_boosting_alpha_0.1",
        "hist_gradient_boosting_alpha_0.05",
    }
    assert all(np.isfinite(value) for value in result.quantiles.values())
    assert all(set(bounds) == {"alpha_0.1", "alpha_0.05"} for bounds in result.metrics.values())
    assert (result.calibration_scores["rul_raw"] > 0).all()
    assert np.allclose(
        result.calibration_scores["rul_raw"],
        100 - result.calibration_scores["cycle"],
    )
    assert {"mean_absolute_residual", "ridge_absolute_residual"}.issubset(
        result.calibration_scores
    )


def test_c02_rejects_unaligned_official_test_engines() -> None:
    """Official RUL labels cannot attach to noncontiguous test engine identifiers."""
    test = trajectory(2, 3)
    test["engine_id"] = test["engine_id"].replace({1: 2, 2: 3})
    with pytest.raises(ValueError, match="ordered and contiguous"):
        run_c02(
            trajectory(10, 100),
            test,
            pd.DataFrame({"rul": [10, 20]}),
            C02Config.from_yaml(config_text()),
            seed=13,
        )


def test_c02_artifact_records_scores_models_and_is_immutable(tmp_path: Path) -> None:
    """C02 artifacts retain the data required to audit conformal quantiles."""
    config_path = tmp_path / "conformal.yaml"
    config_path.write_text(config_text(), encoding="utf-8")
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text("version: 1\nfiles: []\n", encoding="utf-8")
    result = C02Result(
        predictions=pd.DataFrame({"engine_id": [1], "y_true": [1.0]}),
        calibration_scores=pd.DataFrame(
            {"engine_id": [1], "cycle": [30], "rul_capped": [70], "mean_absolute_residual": [2.0]}
        ),
        metrics={"mean": {"alpha_0.1": {"coverage": 1.0}}},
        partitions={"base_train": [1], "calibration": [2], "validation": [3]},
        cut_points={2: 30},
        quantiles={"mean_alpha_0.1": 2.0},
        feature_names=["cycle", "sensor_1"],
    )
    output = tmp_path / "outputs"
    config = C02Config.from_yaml(config_text())
    run_dir = write_c02_run(
        output,
        13,
        "a" * 40,
        config,
        result,
        config_path,
        registry_path,
        {"train_FD001.txt": {"sha256": "b" * 64}},
    )
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "calibration_scores.csv" in manifest["artifacts"]
    assert manifest["models"] == config.as_dict()["models"]
    assert manifest["bootstrap"] == config.as_dict()["bootstrap"]
    with pytest.raises(FileExistsError):
        write_c02_run(
            output,
            13,
            "a" * 40,
            config,
            result,
            config_path,
            registry_path,
            {"train_FD001.txt": {"sha256": "b" * 64}},
        )


def test_c02_artifact_write_cleans_partial_directory_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed C02 write leaves neither a published nor a temporary run directory."""
    config_path = tmp_path / "conformal.yaml"
    config_path.write_text(config_text(), encoding="utf-8")
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text("version: 1\nfiles: []\n", encoding="utf-8")
    config = C02Config.from_yaml(config_text())
    result = C02Result(
        predictions=pd.DataFrame({"engine_id": [1], "y_true": [1.0]}),
        calibration_scores=pd.DataFrame({"engine_id": [1], "cycle": [30]}),
        metrics={},
        partitions={"base_train": [1], "calibration": [2], "validation": [3]},
        cut_points={2: 30},
        quantiles={},
        feature_names=["cycle", "sensor_1"],
    )
    original_write = artifacts_module._write_bytes

    def fail_on_metrics(path: Path, content: bytes) -> str:
        if path.name == "metrics.json":
            raise OSError("simulated artifact failure")
        return original_write(path, content)

    monkeypatch.setattr(artifacts_module, "_write_bytes", fail_on_metrics)
    output = tmp_path / "outputs"
    with pytest.raises(OSError, match="simulated artifact failure"):
        write_c02_run(
            output,
            13,
            "a" * 40,
            config,
            result,
            config_path,
            registry_path,
            {"train_FD001.txt": {"sha256": "b" * 64}},
        )
    assert list(output.iterdir()) == []


def test_aci_updates_only_after_the_current_endpoint_outcome() -> None:
    """ACI uses the nominal alpha first, then widens after an observed miss."""
    lower, upper, used, next_values, quantiles, missed = aci_prequential_intervals(
        residuals=[1.0, 2.0, 3.0, 4.0],
        centers=[10.0, 10.0, 10.0],
        truth=[20.0, 10.0, 10.0],
        nominal_alpha=0.10,
        gamma=0.01,
        alpha_min=0.001,
        alpha_max=0.999,
    )
    assert used[0] == pytest.approx(0.10)
    assert missed.tolist() == [True, False, False]
    assert next_values[0] == pytest.approx(0.091)
    assert used[1] == pytest.approx(next_values[0])
    assert next_values[1] == pytest.approx(0.092)
    assert (lower <= upper).all()
    assert (quantiles >= 0).all()


def test_aci_rejects_invalid_adaptation_parameters() -> None:
    """ACI fails closed rather than silently accepting an invalid online policy."""
    with pytest.raises(ValueError, match="alpha bounds"):
        aci_prequential_intervals([1.0], [1.0], [1.0], 0.1, 0.01, 0.5, 0.4)
