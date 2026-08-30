"""Behavioral tests for the non-executed C11 implementation checkpoint."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
import random
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner
from scipy.special import beta as beta_function
from scipy.stats import beta as beta_distribution
from scipy.stats import wasserstein_distance

from calibrated_reliability.data.loader import COLUMNS
from calibrated_reliability.data.splitting import generate_cut_points, split_engine_ids
from calibrated_reliability.experiments import artifacts as artifacts_module
from calibrated_reliability.experiments.artifacts import write_c11_run
from calibrated_reliability.experiments.c11 import (
    C11_EVALUATED,
    C11_EXCLUDED_DEGENERATE,
    C11_EXCLUDED_UNATTAINABLE,
    C11Config,
    C11Result,
    beta_binomial_reference,
    distribution_discrepancies,
    exact_order_statistic_distribution,
    run_c11,
    weighted_ks_distance,
)

_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "c11_runner_for_test", Path("scripts/run_c11_finite_reservoir.py")
)
if _RUNNER_SPEC is None or _RUNNER_SPEC.loader is None:
    raise RuntimeError("Could not load the C11 runner for behavioral testing")
runner_module = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(runner_module)


def _config_text() -> str:
    return Path("configs/cmapss/finite_reservoir.yaml").read_text(encoding="utf-8")


def _alter_config(path: tuple[str | int, ...], value: object) -> str:
    payload = yaml.safe_load(_config_text())
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    return yaml.safe_dump(payload, sort_keys=False)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("predictor_seed",), 37),
        (("predictor_seed",), True),
        (("predictor_seed",), "13"),
        (("rul_cap",), 130),
        (("preprocessing", "variance_threshold"), "0.0"),
        (("calibration", "lower_fraction"), float("nan")),
        (("calibration", "upper_fraction"), float("inf")),
        (("model", "max_iter"), 51),
        (("cells", 0, "alpha"), 0.05),
        (("cells", 4, "status"), C11_EVALUATED),
        (("references", "evaluation_endpoints"), 99),
        (("discrepancies", "quadrature_atol"), 1.0e-8),
        (("observation_diagnostic", "metric"), "ks_distance"),
    ],
)
def test_c11_configuration_rejects_every_frozen_design_change(
    path: tuple[str | int, ...], value: object
) -> None:
    with pytest.raises(ValueError, match="frozen C11 design"):
        C11Config.from_yaml(_alter_config(path, value))


def test_c11_configuration_rejects_unknown_and_malformed_fields() -> None:
    payload = yaml.safe_load(_config_text())
    payload["unknown"] = 1
    with pytest.raises(ValueError, match="schema mismatch"):
        C11Config.from_yaml(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="must be a mapping"):
        C11Config.from_yaml("- C11\n")


def test_c11_configuration_revalidation_rejects_manual_payload_drift() -> None:
    config = C11Config.from_yaml(_config_text())
    payload = config.as_dict()
    payload["predictor_seed"] = 37
    with pytest.raises(ValueError, match="frozen C11 design"):
        C11Config(payload=payload, cells=config.cells).validate()


@pytest.mark.parametrize(
    "operation",
    [
        lambda: exact_order_statistic_distribution([False, True, 2.0], 2, 0.50),
        lambda: weighted_ks_distance([True], [1.0], [0.5], [1.0]),
        lambda: weighted_ks_distance([0.5], [True], [0.5], [1.0]),
        lambda: weighted_ks_distance([0.5], [1.0], beta_parameters=(True, 2)),
        lambda: distribution_discrepancies([0.5], [1.0], 0.10, beta_parameters=(2, False)),
    ],
)
def test_c11_mathematical_apis_reject_boolean_inputs(operation: Callable[[], object]) -> None:
    with pytest.raises(ValueError, match="boolean|positive integers"):
        operation()


def test_c11_declared_cells_have_exact_ranks_supports_and_exclusions() -> None:
    config = C11Config.from_yaml(_config_text())
    observed = [
        (cell.n_cal, cell.alpha, cell.requested_rank, cell.status) for cell in config.cells
    ]
    assert observed == [
        (10, 0.10, 10, C11_EVALUATED),
        (15, 0.10, 15, C11_EVALUATED),
        (20, 0.10, 19, C11_EVALUATED),
        (30, 0.10, 28, C11_EVALUATED),
        (10, 0.05, 11, C11_EXCLUDED_UNATTAINABLE),
        (15, 0.05, 16, C11_EXCLUDED_UNATTAINABLE),
        (20, 0.05, 20, C11_EVALUATED),
        (30, 0.05, 30, C11_EVALUATED),
        (40, 0.10, 37, C11_EXCLUDED_DEGENERATE),
        (40, 0.05, 39, C11_EXCLUDED_DEGENERATE),
    ]
    supports = [
        len(exact_order_statistic_distribution(np.arange(40.0), cell.n_cal, cell.alpha))
        for cell in config.cells
        if cell.status == C11_EVALUATED
    ]
    assert supports == [31, 26, 21, 11, 21, 11]


def test_exact_order_distribution_matches_brute_force_and_integer_identity() -> None:
    scores = np.arange(1.0, 8.0)
    result = exact_order_statistic_distribution(scores, n_cal=4, alpha=0.40)
    rank = math.ceil(5 * 0.60)
    counts: dict[int, int] = {}
    for subset in itertools.combinations(range(1, 8), 4):
        position = sorted(subset)[rank - 1]
        counts[position] = counts.get(position, 0) + 1
    reported = {
        int(row.position_min): int(row.exact_multiplicity)
        for row in result.itertuples(index=False)
    }
    assert reported == counts
    assert sum(reported.values()) == math.comb(7, 4)
    assert result["probability"].sum() == pytest.approx(1.0, abs=1.0e-12)


def test_fixed_seed_monte_carlo_is_only_a_check_on_exact_distribution() -> None:
    scores = np.arange(1.0, 9.0)
    exact = exact_order_statistic_distribution(scores, n_cal=4, alpha=0.40)
    expected = dict(zip(exact["quantile"], exact["probability"], strict=True))
    rank = math.ceil(5 * 0.60)
    rng = random.Random(13)
    counts = {value: 0 for value in expected}
    draws = 20_000
    for _ in range(draws):
        subset = sorted(rng.sample(scores.tolist(), 4))
        counts[subset[rank - 1]] += 1
    for value, probability in expected.items():
        assert counts[value] / draws == pytest.approx(probability, abs=0.012)


def test_exact_ties_aggregate_only_identical_float64_values() -> None:
    scores = np.array([1.0, 1.0, np.nextafter(1.0, 2.0), 2.0, 3.0])
    result = exact_order_statistic_distribution(scores, n_cal=2, alpha=0.50)
    assert result["quantile"].nunique() == len(result)
    tied = result[result["quantile"] == 1.0]
    assert len(tied) == 1
    assert int(tied.iloc[0]["position_min"]) == 2
    assert np.nextafter(1.0, 2.0) in result["quantile"].to_numpy()
    position_map = json.loads(str(tied.iloc[0]["position_multiplicities_json"]))
    assert position_map == {"2": 1}


def test_unattainable_and_degenerate_distribution_requests_fail_closed() -> None:
    with pytest.raises(ValueError, match="unattainable"):
        exact_order_statistic_distribution(np.arange(40.0), 10, 0.05)
    with pytest.raises(ValueError, match="strictly below"):
        exact_order_statistic_distribution(np.arange(40.0), 40, 0.10)
    with pytest.raises(ValueError, match="tolerance"):
        exact_order_statistic_distribution(np.arange(40.0), 20, 0.10, probability_tolerance=1.0e-6)


def test_beta_binomial_reference_matches_frozen_formula_and_strict_tail() -> None:
    result = beta_binomial_reference(20, 0.10)
    assert len(result) == 101
    assert result["probability"].sum() == pytest.approx(1.0, abs=1.0e-12)
    rank, a, b, endpoints = 19, 19, 2, 100
    expected = (
        math.comb(endpoints, 80) * beta_function(80 + a, endpoints - 80 + b) / beta_function(a, b)
    )
    assert result.loc[80, "probability"] == pytest.approx(expected, rel=1.0e-12)
    threshold = (1.0 - 0.10) - 0.10
    strict = result.loc[result["coverage"] < threshold, "probability"].sum()
    non_strict = result.loc[result["coverage"] <= threshold, "probability"].sum()
    assert strict < non_strict
    assert rank == math.ceil(21 * 0.90)


def test_all_continuous_beta_reference_values_match_the_accepted_design() -> None:
    expected = {
        (10, 0.10): (0.9091, 0.0830, 0.7411, 0.7943, 0.1074),
        (15, 0.10): (0.9375, 0.0587, 0.8190, 0.8577, 0.0352),
        (20, 0.10): (0.9048, 0.0626, 0.7839, 0.8190, 0.0692),
        (30, 0.10): (0.9032, 0.0523, 0.8047, 0.8322, 0.0442),
        (20, 0.05): (0.9524, 0.0454, 0.8609, 0.8913, 0.0388),
        (30, 0.05): (0.9677, 0.0312, 0.9050, 0.9261, 0.0076),
    }
    for (n_cal, alpha), reference in expected.items():
        rank = math.ceil((n_cal + 1) * (1.0 - alpha))
        a, b = rank, n_cal + 1 - rank
        threshold = (1.0 - alpha) - 0.10
        observed = (
            beta_distribution.mean(a, b),
            beta_distribution.std(a, b),
            beta_distribution.ppf(0.05, a, b),
            beta_distribution.ppf(0.10, a, b),
            beta_distribution.cdf(threshold, a, b),
        )
        assert observed == pytest.approx(reference, abs=5.0e-5)


def test_weighted_ks_uses_both_jump_limits() -> None:
    finite_values = np.array([0.25, 0.75])
    probabilities = np.array([0.5, 0.5])
    reference_values = np.array([0.50])
    reference_probabilities = np.array([1.0])
    assert weighted_ks_distance(
        finite_values,
        probabilities,
        reference_values,
        reference_probabilities,
    ) == pytest.approx(0.5)
    beta_ks = weighted_ks_distance(finite_values, probabilities, beta_parameters=(2, 2))
    manual = max(
        abs(0.0 - beta_distribution.cdf(0.25, 2, 2)),
        abs(0.5 - beta_distribution.cdf(0.25, 2, 2)),
        abs(0.5 - beta_distribution.cdf(0.75, 2, 2)),
        abs(1.0 - beta_distribution.cdf(0.75, 2, 2)),
    )
    assert beta_ks == pytest.approx(manual)


def test_weighted_ks_handles_shared_boundary_and_disjoint_jumps() -> None:
    assert weighted_ks_distance([0.0, 1.0], [0.3, 0.7], [0.0, 1.0], [0.8, 0.2]) == pytest.approx(
        0.5
    )
    assert weighted_ks_distance(
        [0.0, 0.5, 1.0],
        [0.2, 0.3, 0.5],
        [0.25, 0.75],
        [0.4, 0.6],
    ) == pytest.approx(0.5)


def test_continuous_beta_wasserstein_matches_closed_form_uniform_case() -> None:
    result = distribution_discrepancies([0.5], [1.0], 0.10, beta_parameters=(1, 1))
    assert result["wasserstein_1"] == pytest.approx(0.25, abs=1.0e-12)


def test_discrepancy_orientation_and_wasserstein_units_are_frozen() -> None:
    finite_values = np.array([0.6, 0.8])
    probabilities = np.array([0.5, 0.5])
    reference_values = np.array([0.5, 0.7])
    result = distribution_discrepancies(
        finite_values,
        probabilities,
        0.10,
        reference_values=reference_values,
        reference_probabilities=probabilities,
    )
    assert result["signed_mean_difference"] == pytest.approx(0.1)
    assert result["signed_population_sd_difference"] == pytest.approx(0.0)
    assert result["wasserstein_1"] == pytest.approx(0.1)


def _trajectory_frame(engine_count: int, cycles: int) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for engine_id in range(1, engine_count + 1):
        for cycle in range(1, cycles + 1):
            row: dict[str, float | int] = {
                "engine_id": engine_id,
                "cycle": cycle,
                "op_setting_1": float(engine_id % 3),
                "op_setting_2": 0.0,
                "op_setting_3": 0.0,
            }
            row.update({f"sensor_{index}": float(cycle + index) for index in range(1, 22)})
            rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


class _IdentityTemporal:
    fit_ids: list[int] = []
    fit_calls = 0

    def __init__(self, windows: tuple[int, ...], variance_threshold: float) -> None:
        assert windows == (5, 10, 20) and variance_threshold == 0.0
        self.feature_names_out_ = list(COLUMNS)

    def fit(self, frame: pd.DataFrame) -> _IdentityTemporal:
        type(self).fit_calls += 1
        type(self).fit_ids = sorted(frame["engine_id"].unique().astype(int).tolist())
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.reset_index(drop=True)


class _FakeHGB:
    fit_calls = 0
    fit_rows = 0

    def fit(self, features: pd.DataFrame, target: np.ndarray) -> _FakeHGB:
        type(self).fit_calls += 1
        type(self).fit_rows = len(features)
        assert len(features) == len(target)
        assert not {
            "engine_id",
            "cycle_ratio",
            "observation_horizon",
            "terminal_lifetime",
            "rul_raw",
            "rul_capped",
        }.intersection(features.columns)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return features["sensor_1"].to_numpy(dtype="float64") * 0.1


def _install_fake_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "calibrated_reliability.experiments.c11.TemporalFeatureTransformer",
        _IdentityTemporal,
    )

    def fake_models(**kwargs: object) -> dict[str, _FakeHGB]:
        assert kwargs == {
            "random_state": 13,
            "hgb_max_iter": 50,
            "hgb_learning_rate": 0.05,
            "hgb_max_leaf_nodes": 31,
            "hgb_l2_regularization": 1.0,
        }
        return {"hist_gradient_boosting": _FakeHGB()}

    monkeypatch.setattr(
        "calibrated_reliability.experiments.c11.build_baseline_models", fake_models
    )


@pytest.fixture
def synthetic_c11(monkeypatch: pytest.MonkeyPatch) -> tuple[C11Config, C11Result, pd.DataFrame]:
    _IdentityTemporal.fit_calls = 0
    _FakeHGB.fit_calls = 0
    train = _trajectory_frame(100, 35)
    test = _trajectory_frame(100, 2)
    config = C11Config.from_yaml(_config_text())
    _install_fake_pipeline(monkeypatch)
    result = run_c11(train, test, pd.DataFrame({"rul": np.arange(100)}), config)
    return config, result, train


def test_c11_orchestration_fits_once_and_preserves_data_roles(
    synthetic_c11: tuple[C11Config, C11Result, pd.DataFrame],
) -> None:
    config, result, train = synthetic_c11
    partitions = split_engine_ids(list(range(1, 101)), 13)
    assert _IdentityTemporal.fit_calls == 1
    assert _IdentityTemporal.fit_ids == partitions["base_train"]
    assert _FakeHGB.fit_calls == 1
    assert _FakeHGB.fit_rows == 60 * 35
    assert len(result.reservoir_scores) == 40
    assert len(result.evaluation_scores) == 100
    assert result.split_manifest["reservoir_engine_ids"] == sorted(
        partitions["calibration"] + partitions["validation"]
    )
    assert set(result.reservoir_scores["origin_role"]) == {
        "former_calibration",
        "former_validation",
    }
    expected_cuts = generate_cut_points(
        train,
        result.split_manifest["reservoir_engine_ids"],
        seed=13,
        min_observed_cycles=30,
        lower_fraction=0.40,
        upper_fraction=0.90,
    )
    assert result.cut_points == expected_cuts
    lifetimes = train.groupby("engine_id")["cycle"].max().astype(int).to_dict()
    rng = random.Random(13)
    manual_cuts = {
        engine_id: rng.randint(
            max(30, math.ceil(0.40 * lifetimes[engine_id])),
            math.floor(0.90 * lifetimes[engine_id]),
        )
        for engine_id in result.split_manifest["reservoir_engine_ids"]
    }
    assert result.cut_points == manual_cuts
    assert len(result.enumeration_cells) == 10
    assert set(result.distribution_summary["status"]) == {
        "evaluated",
        C11_EXCLUDED_UNATTAINABLE,
        C11_EXCLUDED_DEGENERATE,
    }
    assert result.observation_mechanism["distance"] == pytest.approx(
        wasserstein_distance(
            result.reservoir_scores["cycle"], result.evaluation_scores["observed_cycles"]
        )
    )
    assert config.predictor_seed == 13


def test_c11_future_reservoir_rows_and_labels_cannot_change_frozen_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_pipeline(monkeypatch)
    config = C11Config.from_yaml(_config_text())
    train = _trajectory_frame(100, 35)
    test = _trajectory_frame(100, 2)
    rul = pd.DataFrame({"rul": np.arange(100)})
    first = run_c11(train, test, rul, config)
    perturbed = train.copy()
    for engine_id, cut_point in first.cut_points.items():
        mask = (perturbed["engine_id"] == engine_id) & (perturbed["cycle"] > cut_point)
        perturbed.loc[mask, "sensor_1"] = 1.0e9
    second = run_c11(perturbed, test, rul, config)
    pd.testing.assert_frame_equal(first.reservoir_scores, second.reservoir_scores)
    pd.testing.assert_frame_equal(first.quantile_distribution, second.quantile_distribution)
    pd.testing.assert_frame_equal(first.distribution_summary, second.distribution_summary)
    altered_rul = pd.DataFrame({"rul": np.arange(100)[::-1]})
    third = run_c11(train, test, altered_rul, config)
    assert first.split_manifest == third.split_manifest
    assert first.cut_points == third.cut_points
    pd.testing.assert_frame_equal(first.reservoir_scores, third.reservoir_scores)
    pd.testing.assert_frame_equal(first.enumeration_cells, third.enumeration_cells)
    pd.testing.assert_frame_equal(
        first.beta_binomial_distribution, third.beta_binomial_distribution
    )
    assert first.reference_summary == third.reference_summary
    assert first.observation_mechanism["distance"] == third.observation_mechanism["distance"]
    assert (
        first.observation_mechanism["official_observed_endpoint_cycles"]
        == third.observation_mechanism["official_observed_endpoint_cycles"]
    )


def test_c11_repeated_in_memory_execution_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_pipeline(monkeypatch)
    config = C11Config.from_yaml(_config_text())
    train = _trajectory_frame(100, 35)
    test = _trajectory_frame(100, 2)
    rul = pd.DataFrame({"rul": np.arange(100)})
    first = run_c11(train, test, rul, config)
    second = run_c11(train, test, rul, config)
    for name in (
        "reservoir_scores",
        "evaluation_scores",
        "enumeration_cells",
        "quantile_distribution",
        "beta_binomial_distribution",
        "distribution_summary",
    ):
        pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name))
    assert first.reference_summary == second.reference_summary
    assert first.observation_mechanism == second.observation_mechanism


def test_c11_official_endpoint_alignment_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    config = C11Config.from_yaml(_config_text())
    test = _trajectory_frame(100, 2)
    bad_test = pd.concat(
        [test[test["engine_id"] != 100], test[test["engine_id"] == 100].assign(engine_id=101)]
    )
    with pytest.raises(ValueError, match="official FD001 test"):
        run_c11(
            _trajectory_frame(100, 35),
            bad_test,
            pd.DataFrame({"rul": np.arange(100)}),
            config,
        )


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "finite_reservoir.yaml"
    config_path.write_text(_config_text(), encoding="utf-8")
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text("version: 1\nfiles: []\n", encoding="utf-8")
    return config_path, registry_path


def test_c11_artifact_is_complete_hashed_and_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_c11: tuple[C11Config, C11Result, pd.DataFrame],
) -> None:
    config, result, _ = synthetic_c11
    config_path, registry_path = _write_inputs(tmp_path)
    output = tmp_path / "outputs"
    monkeypatch.setattr(artifacts_module, "_validate_c11_publication_state", lambda *args: None)
    run_dir = write_c11_run(output, "a" * 40, config, result, config_path, registry_path, {})
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == "C11"
    assert manifest["accepted_design_adr"]["path"].endswith(
        "ADR-0012-c11-finite-reservoir-design.md"
    )
    assert manifest["split_manifest"] == result.split_manifest
    assert set(manifest["artifacts"]) == {
        "split_manifest.json",
        "reservoir_scores.csv",
        "evaluation_scores.csv",
        "enumeration_cells.csv",
        "quantile_distribution.csv",
        "beta_binomial_distribution.csv",
        "reference_summary.json",
        "distribution_summary.csv",
        "observation_mechanism.json",
        "resolved_config.json",
        "run.log",
    }
    for name, expected_hash in manifest["artifacts"].items():
        assert hashlib.sha256((run_dir / name).read_bytes()).hexdigest() == expected_hash
    with pytest.raises(FileExistsError):
        write_c11_run(output, "a" * 40, config, result, config_path, registry_path, {})


def test_c11_failed_artifact_write_cleans_temporary_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_c11: tuple[C11Config, C11Result, pd.DataFrame],
) -> None:
    config, result, _ = synthetic_c11
    config_path, registry_path = _write_inputs(tmp_path)
    original = artifacts_module._write_bytes
    monkeypatch.setattr(artifacts_module, "_validate_c11_publication_state", lambda *args: None)

    def fail(path: Path, content: bytes) -> str:
        if path.name == "distribution_summary.csv":
            raise OSError("simulated C11 failure")
        return original(path, content)

    monkeypatch.setattr(artifacts_module, "_write_bytes", fail)
    output = tmp_path / "outputs"
    with pytest.raises(OSError, match="simulated C11 failure"):
        write_c11_run(output, "a" * 40, config, result, config_path, registry_path, {})
    assert list(output.iterdir()) == []


def test_c11_final_publication_revalidation_failure_cleans_temporary_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_c11: tuple[C11Config, C11Result, pd.DataFrame],
) -> None:
    config, result, _ = synthetic_c11
    config_path, registry_path = _write_inputs(tmp_path)
    calls = 0

    def reject_final(*args: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated final provenance change")

    monkeypatch.setattr(artifacts_module, "_validate_c11_publication_state", reject_final)
    output = tmp_path / "outputs"
    with pytest.raises(RuntimeError, match="final provenance"):
        write_c11_run(output, "a" * 40, config, result, config_path, registry_path, {})
    assert calls == 2
    assert list(output.iterdir()) == []


def test_c11_publication_state_checks_sha_cleanliness_and_input_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("frozen", encoding="utf-8")
    expected_hash = hashlib.sha256(tracked.read_bytes()).hexdigest()
    outputs = iter(["a" * 40 + "\n", ""])
    monkeypatch.setattr(
        artifacts_module.subprocess,
        "check_output",
        lambda *args, **kwargs: next(outputs),
    )
    artifacts_module._validate_c11_publication_state("a" * 40, {tracked: expected_hash})

    outputs = iter(["b" * 40 + "\n"])
    with pytest.raises(RuntimeError, match="revision changed"):
        artifacts_module._validate_c11_publication_state("a" * 40, {tracked: expected_hash})

    outputs = iter(["a" * 40 + "\n", " M README.md\n"])
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        artifacts_module._validate_c11_publication_state("a" * 40, {tracked: expected_hash})

    tracked.write_text("changed", encoding="utf-8")
    outputs = iter(["a" * 40 + "\n", ""])
    with pytest.raises(RuntimeError, match="provenance input changed"):
        artifacts_module._validate_c11_publication_state("a" * 40, {tracked: expected_hash})


def test_c11_result_tables_reconstruct_distributions(
    synthetic_c11: tuple[C11Config, C11Result, pd.DataFrame],
) -> None:
    _, result, _ = synthetic_c11
    for cell_id, rows in result.quantile_distribution.groupby("cell_id"):
        assert rows["probability"].sum() == pytest.approx(1.0, abs=1.0e-12), cell_id
        assert rows["exact_multiplicity"].sum() == rows["combination_count"].iloc[0]
    for cell_id, rows in result.beta_binomial_distribution.groupby("cell_id"):
        assert len(rows) == 101, cell_id
        assert rows["probability"].sum() == pytest.approx(1.0, abs=1.0e-12)
    evaluated = result.distribution_summary[result.distribution_summary["status"] == "evaluated"]
    assert len(evaluated) == 12
    assert set(evaluated["reference"]) == {"continuous_beta", "beta_binomial"}
    assert np.isfinite(
        evaluated[
            [
                "ks_distance",
                "signed_mean_difference",
                "signed_severe_tail_difference",
                "signed_population_sd_difference",
                "wasserstein_1",
            ]
        ].to_numpy(dtype="float64")
    ).all()


def test_c11_serialized_tables_independently_reconstruct_metrics_and_discrepancies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    synthetic_c11: tuple[C11Config, C11Result, pd.DataFrame],
) -> None:
    config, result, _ = synthetic_c11
    config_path, registry_path = _write_inputs(tmp_path)
    monkeypatch.setattr(artifacts_module, "_validate_c11_publication_state", lambda *args: None)
    run_dir = write_c11_run(
        tmp_path / "outputs", "a" * 40, config, result, config_path, registry_path, {}
    )
    quantiles = pd.read_csv(run_dir / "quantile_distribution.csv", float_precision="round_trip")
    evaluation = pd.read_csv(run_dir / "evaluation_scores.csv", float_precision="round_trip")
    beta_binomial = pd.read_csv(
        run_dir / "beta_binomial_distribution.csv", float_precision="round_trip"
    )
    summary = pd.read_csv(run_dir / "distribution_summary.csv", float_precision="round_trip")
    residuals = evaluation["residual"].to_numpy(dtype="float64")
    for row in quantiles.itertuples(index=False):
        assert row.coverage == pytest.approx(float(np.mean(residuals <= row.quantile)))
        expected_nis = (
            np.mean(
                2.0 * row.quantile + (2.0 / row.alpha) * np.maximum(residuals - row.quantile, 0.0)
            )
            / 125.0
        )
        assert row.normalized_interval_score == pytest.approx(expected_nis)
    for row in summary.loc[summary["status"] == "evaluated"].itertuples(index=False):
        finite = quantiles.loc[quantiles["cell_id"] == row.cell_id]
        if row.reference == "continuous_beta":
            rank = math.ceil((row.n_cal + 1) * (1.0 - row.alpha))
            kwargs = {"beta_parameters": (rank, row.n_cal + 1 - rank)}
        else:
            reference = beta_binomial.loc[beta_binomial["cell_id"] == row.cell_id]
            kwargs = {
                "reference_values": reference["coverage"],
                "reference_probabilities": reference["probability"],
            }
        rebuilt = distribution_discrepancies(
            finite["coverage"], finite["probability"], row.alpha, **kwargs
        )
        for name, value in rebuilt.items():
            assert getattr(row, name) == pytest.approx(value, abs=1.0e-12, rel=1.0e-12)


def test_c11_runner_validates_inputs_twice_and_calls_run_and_write_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, registry_path = _write_inputs(tmp_path)
    records = [
        SimpleNamespace(
            filename=name,
            sha256="0" * 64,
            expected_bytes=1,
            expected_rows=1,
            expected_engines=1,
        )
        for name in ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")
    ]
    validation_calls: list[str] = []
    monkeypatch.setattr(runner_module, "load_registry", lambda path: records)
    monkeypatch.setattr(
        runner_module,
        "validate_file",
        lambda root, record: (
            validation_calls.append(record.filename) or SimpleNamespace(valid=True)
        ),
    )
    sentinel_frame = pd.DataFrame()
    monkeypatch.setattr(runner_module, "load_train", lambda path: sentinel_frame)
    monkeypatch.setattr(runner_module, "load_test", lambda path: sentinel_frame)
    monkeypatch.setattr(runner_module, "load_rul", lambda path: sentinel_frame)
    run_calls: list[object] = []
    result = object()
    monkeypatch.setattr(
        runner_module,
        "run_c11",
        lambda *args: run_calls.append(args) or result,
    )
    write_calls: list[object] = []
    monkeypatch.setattr(
        runner_module,
        "write_c11_run",
        lambda *args: write_calls.append(args) or (tmp_path / "published"),
    )
    monkeypatch.setattr(
        runner_module.subprocess,
        "check_output",
        lambda command, **kwargs: "a" * 40 + "\n" if "rev-parse" in command else "",
    )
    outcome = CliRunner().invoke(
        runner_module.main,
        ["--config", str(config_path), "--registry", str(registry_path)],
    )
    assert outcome.exit_code == 0, outcome.output
    assert len(run_calls) == 1
    assert len(write_calls) == 1
    assert sorted(validation_calls) == sorted([record.filename for record in records] * 2)
