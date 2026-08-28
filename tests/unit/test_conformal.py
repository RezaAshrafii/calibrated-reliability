"""Tests for finite-sample split-conformal intervals and C02 orchestration."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calibrated_reliability.data.loader import COLUMNS
from calibrated_reliability.evaluation.conformal import (
    ACIState,
    bootstrap_interval_metric_cis,
    conformal_quantile,
    conformal_quantile_result,
    interval_metrics,
    split_conformal_intervals,
)
from calibrated_reliability.experiments import artifacts as artifacts_module
from calibrated_reliability.experiments.artifacts import write_c02_run
from calibrated_reliability.experiments.c02 import (
    C02Config,
    C02Result,
    evaluate_c02_pipeline,
    fit_c02_pipeline,
    run_c02,
)


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
    interior = conformal_quantile_result([1.0, 2.0, 5.0], alpha=0.50)
    assert interior.quantile == 2.0
    assert interior.requested_rank == 2
    assert interior.regime == "interior"
    maximum = conformal_quantile_result([1.0, 2.0, 5.0], alpha=0.25)
    assert maximum.quantile == 5.0
    assert maximum.requested_rank == 3
    assert maximum.regime == "max_statistic"
    with pytest.raises(ValueError, match="strictly between"):
        conformal_quantile([1.0], alpha=1.0)


def test_unattainable_conformal_rank_requires_an_explicit_policy() -> None:
    """Unattainable finite ranks fail closed unless a named policy is selected."""
    with pytest.raises(ValueError, match="unattainable"):
        conformal_quantile([1.0, 2.0, 5.0], alpha=0.10)
    infinite = conformal_quantile_result([1.0, 2.0, 5.0], alpha=0.10, on_unattainable="infinite")
    assert np.isinf(infinite.quantile)
    assert infinite.effective_rank is None
    assert infinite.regime == "finite_quantile_unattainable"
    legacy = conformal_quantile_result(
        [1.0, 2.0, 5.0], alpha=0.10, on_unattainable="legacy_max_clamp"
    )
    assert legacy.quantile == 5.0
    assert legacy.requested_rank == 4
    assert legacy.effective_rank == 3
    assert legacy.finite_rank_attainable is False


def test_intervals_and_metrics_are_finite_and_directionally_correct() -> None:
    """Intervals are symmetric and interval score penalizes misses."""
    lower, upper, q = split_conformal_intervals([10.0, 12.0], [9.0, 13.0], [11.0], 0.50)
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
    lower, upper, _ = split_conformal_intervals([0.0, 0.0], [10.0, 10.0], [0.0], 0.50)
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
        trajectory(100, 40),
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
        40 - result.calibration_scores["cycle"],
    )
    assert {"mean_absolute_residual", "ridge_absolute_residual"}.issubset(
        result.calibration_scores
    )


def test_c02_fitted_pipeline_is_frozen_and_reused_across_evaluations() -> None:
    """Direct fit/evaluate APIs preserve splits, full truth, models, and transformer state."""
    config = C02Config.from_yaml(config_text())
    pipeline = fit_c02_pipeline(trajectory(100, 40), config, seed=13)
    partitions = [
        set(pipeline.partitions[name]) for name in ("base_train", "calibration", "validation")
    ]
    assert not (
        partitions[0] & partitions[1]
        or partitions[0] & partitions[2]
        or partitions[1] & partitions[2]
    )
    assert set.union(*partitions) == set(range(1, 101))
    assert np.allclose(
        pipeline.calibration_scores["rul_raw"],
        40 - pipeline.calibration_scores["cycle"],
    )
    temporal_id = id(pipeline.temporal)
    model_ids = {name: id(model) for name, model in pipeline.models.items()}
    scores_before = pipeline.calibration_scores.copy(deep=True)
    test = trajectory(2, 20)
    rul = pd.DataFrame({"rul": [10, 20]})
    first = evaluate_c02_pipeline(pipeline, test, rul, config, seed=13)
    second = evaluate_c02_pipeline(pipeline, test, rul, config, seed=13)
    assert id(pipeline.temporal) == temporal_id
    assert {name: id(model) for name, model in pipeline.models.items()} == model_ids
    pd.testing.assert_frame_equal(pipeline.calibration_scores, scores_before)
    pd.testing.assert_frame_equal(first.predictions, second.predictions)


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
    state = ACIState(
        [1.0, 2.0, 3.0, 4.0],
        0.10,
        0.01,
        0.001,
        0.999,
        unattainable_rank_policy="legacy_max_clamp",
    )
    lower, upper, used, quantile = state.predict_interval(10.0)
    assert used == pytest.approx(0.10)
    assert lower <= upper and quantile >= 0
    with pytest.raises(RuntimeError, match="outcome"):
        state.predict_interval(10.0)
    missed, next_alpha = state.update(20.0)
    assert missed is True and next_alpha == pytest.approx(0.091)
    with pytest.raises(RuntimeError, match="predicted"):
        state.update(10.0)
    _, _, used_next, _ = state.predict_interval(10.0)
    assert used_next == pytest.approx(next_alpha)


def test_aci_rejects_invalid_adaptation_parameters() -> None:
    """ACI fails closed rather than silently accepting an invalid online policy."""
    with pytest.raises(ValueError, match="alpha bounds"):
        ACIState(
            [1.0],
            0.1,
            0.01,
            0.5,
            0.4,
            unattainable_rank_policy="legacy_max_clamp",
        )


def test_aci_rejects_update_before_prediction_and_future_outcomes_do_not_change_first_interval():
    """The first interval is independent of outcomes revealed later."""
    first = ACIState(
        [1.0, 2.0, 3.0],
        0.10,
        0.01,
        0.001,
        0.999,
        unattainable_rank_policy="legacy_max_clamp",
    )
    second = ACIState(
        [1.0, 2.0, 3.0],
        0.10,
        0.01,
        0.001,
        0.999,
        unattainable_rank_policy="legacy_max_clamp",
    )
    with pytest.raises(RuntimeError, match="predicted"):
        first.update(100.0)
    assert first.predict_interval(10.0) == second.predict_interval(10.0)


def test_aci_future_outcomes_do_not_change_prior_intervals_or_alpha_states() -> None:
    """Changing later revealed outcomes cannot retroactively affect the path."""
    first = ACIState(
        [1.0, 2.0, 3.0],
        0.10,
        0.01,
        0.001,
        0.999,
        unattainable_rank_policy="legacy_max_clamp",
    )
    second = ACIState(
        [1.0, 2.0, 3.0],
        0.10,
        0.01,
        0.001,
        0.999,
        unattainable_rank_policy="legacy_max_clamp",
    )
    first_path = []
    second_path = []
    for current_first, current_second in zip((10.0, 10.0, 10.0), (10.0, 10.0, 10.0), strict=True):
        first_path.append(first.predict_interval(current_first))
        second_path.append(second.predict_interval(current_second))
        first.update(10.0)
        second.update(10.0)
    assert first_path == second_path

    altered = ACIState(
        [1.0, 2.0, 3.0],
        0.10,
        0.01,
        0.001,
        0.999,
        unattainable_rank_policy="legacy_max_clamp",
    )
    altered_path = []
    for index, center in enumerate((10.0, 10.0, 10.0)):
        altered_path.append(altered.predict_interval(center))
        altered.update(10.0 if index == 0 else 1000.0)
    assert altered_path[0] == first_path[0]
    assert altered_path[1] == first_path[1]


def test_aci_alpha_clips_at_both_declared_bounds() -> None:
    """Repeated misses and hits respect the frozen alpha projection bounds."""
    low = ACIState(
        [1.0],
        0.10,
        1.0,
        0.001,
        0.999,
        unattainable_rank_policy="legacy_max_clamp",
    )
    low.predict_interval(0.0)
    _, alpha_after_miss = low.update(100.0)
    assert alpha_after_miss == pytest.approx(0.001)

    high = ACIState(
        [1.0],
        0.90,
        1.0,
        0.001,
        0.999,
        unattainable_rank_policy="legacy_max_clamp",
    )
    high.predict_interval(0.0)
    _, alpha_after_hit = high.update(0.0)
    assert alpha_after_hit == pytest.approx(0.999)
