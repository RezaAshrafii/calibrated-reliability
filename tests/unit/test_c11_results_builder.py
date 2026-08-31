"""Behavioral tests for the separate C11 deterministic reporting path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from calibrated_reliability.reporting import c11_results


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _fixture_repository(tmp_path: Path) -> tuple[Path, Path, Path]:
    repository = tmp_path / "repository"
    run_dir = repository / "outputs" / "c11_official" / "C11_FD001_producer_seed_13"
    run_dir.mkdir(parents=True)

    cell_specs = [
        ("n10_alpha_0.1", "primary", 10, 0.1, "evaluate"),
        ("n15_alpha_0.1", "primary", 15, 0.1, "evaluate"),
        ("n20_alpha_0.1", "primary", 20, 0.1, "evaluate"),
        ("n30_alpha_0.1", "primary", 30, 0.1, "evaluate"),
        (
            "n10_alpha_0.05",
            "sensitivity",
            10,
            0.05,
            "not_evaluated_due_to_unattainable_finite_rank",
        ),
        (
            "n15_alpha_0.05",
            "sensitivity",
            15,
            0.05,
            "not_evaluated_due_to_unattainable_finite_rank",
        ),
        ("n20_alpha_0.05", "sensitivity", 20, 0.05, "evaluate"),
        ("n30_alpha_0.05", "sensitivity", 30, 0.05, "evaluate"),
        (
            "n40_alpha_0.1",
            "excluded",
            40,
            0.1,
            "not_evaluated_due_to_degenerate_full_reservoir_subset",
        ),
        (
            "n40_alpha_0.05",
            "excluded",
            40,
            0.05,
            "not_evaluated_due_to_degenerate_full_reservoir_subset",
        ),
    ]
    cells = pd.DataFrame(
        [
            {"cell_id": cell_id, "role": role, "n_cal": n_cal, "alpha": alpha, "status": status}
            for cell_id, role, n_cal, alpha, status in cell_specs
        ]
    )
    cells.to_csv(run_dir / "enumeration_cells.csv", index=False)
    evaluated_indices = [0, 1, 2, 3, 6, 7]
    summaries = pd.DataFrame(
        [
            {
                "cell_id": cell_specs[evaluated_indices[index // 2]][0],
                "status": "evaluated",
                "reference": "continuous_beta" if index % 2 == 0 else "beta_binomial",
            }
            for index in range(12)
        ]
        + [
            {
                "cell_id": cell_specs[[4, 5, 8, 9][index]][0],
                "status": cell_specs[[4, 5, 8, 9][index]][4],
                "reference": "not_evaluated",
            }
            for index in range(4)
        ]
    )
    summaries.to_csv(run_dir / "distribution_summary.csv", index=False)
    for name, count in {
        "quantile_distribution.csv": 121,
        "beta_binomial_distribution.csv": 606,
        "reservoir_scores.csv": 40,
        "evaluation_scores.csv": 100,
    }.items():
        pd.DataFrame({"row": range(count)}).to_csv(run_dir / name, index=False)
    _write_json(
        run_dir / "observation_mechanism.json",
        {
            "metric": "empirical_wasserstein_1",
            "unit": "observed_cycles",
            "weighting": "equal_engine",
            "distance": 1.25,
            "reservoir_engine_count": 40,
            "official_endpoint_count": 100,
            "evaluation_cap_saturation_fraction": 0.1,
            "reservoir_distinct_residuals": 40,
            "reservoir_residual_tied_rows": 0,
        },
    )
    for name in (
        "split_manifest.json",
        "reference_summary.json",
        "resolved_config.json",
    ):
        _write_json(run_dir / name, {"fixture": True})
    (run_dir / "run.log").write_text("fixture\n", encoding="utf-8")
    for relative in (
        "configs/cmapss/finite_reservoir.yaml",
        "data/registry.yaml",
        "uv.lock",
        "docs/decisions/ADR-0012-c11-finite-reservoir-design.md",
    ):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n", encoding="utf-8")

    artifact_names = sorted(path.name for path in run_dir.iterdir())
    producer = "1" * 40
    run_id = run_dir.name
    manifest = {
        "experiment_id": "C11",
        "run_id": run_id,
        "source": "FD001",
        "target": "FD001",
        "evaluation_unit": "engine_endpoint_finite_reservoir",
        "git": {"dirty": False, "sha": producer},
        "configuration": {
            "path": "configs/cmapss/finite_reservoir.yaml",
            "sha256": _digest(repository / "configs/cmapss/finite_reservoir.yaml"),
        },
        "data_registry": {
            "path": "data/registry.yaml",
            "sha256": _digest(repository / "data/registry.yaml"),
        },
        "lockfile": {"path": "uv.lock", "sha256": _digest(repository / "uv.lock")},
        "accepted_design_adr": {
            "path": "docs/decisions/ADR-0012-c11-finite-reservoir-design.md",
            "sha256": _digest(
                repository / "docs/decisions/ADR-0012-c11-finite-reservoir-design.md"
            ),
        },
        "split_manifest": {"fixture": True},
        "artifacts": {name: _digest(run_dir / name) for name in artifact_names},
    }
    manifest_path = run_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    index = {
        "schema_version": 1,
        "index_id": "fixture",
        "experiment_id": "C11",
        "artifact_root": "outputs/c11_official",
        "run_directory": run_id,
        "manifest_sha256": _digest(manifest_path),
        "producing_git_sha": producer,
        "expected_artifact_count": 11,
        "status": "VERIFIED_CANDIDATE",
    }
    index_path = repository / "docs" / "c11_artifact_index.yaml"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(yaml.safe_dump(index, sort_keys=False), encoding="utf-8")
    return repository, index_path, run_dir


def test_verified_candidate_build_is_deterministic_and_preserves_exclusions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, index_path, _ = _fixture_repository(tmp_path)
    monkeypatch.setattr(c11_results, "_git_state", lambda _: ("2" * 40, False))
    first = c11_results.build_c11_results(repository, index_path, repository / "first")
    second = c11_results.build_c11_results(repository, index_path, repository / "second")

    assert {path.name for path in first.iterdir()} == {
        "cells.csv",
        "discrepancies.csv",
        "observation_summary.csv",
        "provenance.json",
        "checksums.sha256",
    }
    assert all(
        (first / name).read_bytes() == (second / name).read_bytes()
        for name in {path.name for path in first.iterdir()}
    )
    reported = pd.read_csv(first / "discrepancies.csv", float_precision="round_trip")
    assert len(reported.loc[reported["reference"] == "not_evaluated"]) == 4
    provenance = json.loads((first / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["builder_git_sha"] == "2" * 40
    assert provenance["producing_git_sha"] == "1" * 40


def test_manifest_trust_anchor_and_exclusion_accounting_fail_closed(tmp_path: Path) -> None:
    repository, index_path, run_dir = _fixture_repository(tmp_path)
    index = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    index["manifest_sha256"] = "0" * 64
    index_path.write_text(yaml.safe_dump(index), encoding="utf-8")
    with pytest.raises(ValueError, match="trust anchor"):
        c11_results.verify_c11_artifact(repository, index_path)

    index["manifest_sha256"] = _digest(run_dir / "manifest.json")
    index_path.write_text(yaml.safe_dump(index), encoding="utf-8")
    summary_path = run_dir / "distribution_summary.csv"
    summary = pd.read_csv(summary_path).iloc[:-1]
    summary.to_csv(summary_path, index=False)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"][summary_path.name] = _digest(summary_path)
    _write_json(run_dir / "manifest.json", manifest)
    index["manifest_sha256"] = _digest(run_dir / "manifest.json")
    index_path.write_text(yaml.safe_dump(index), encoding="utf-8")
    with pytest.raises(ValueError, match="row-count contract"):
        c11_results.verify_c11_artifact(repository, index_path)


def test_declared_cell_contract_rejects_count_correct_substitution(tmp_path: Path) -> None:
    repository, index_path, run_dir = _fixture_repository(tmp_path)
    cells = pd.read_csv(run_dir / "enumeration_cells.csv")
    cells.loc[0, "cell_id"] = "substituted_cell"
    cells.to_csv(run_dir / "enumeration_cells.csv", index=False)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"]["enumeration_cells.csv"] = _digest(run_dir / "enumeration_cells.csv")
    _write_json(run_dir / "manifest.json", manifest)
    index = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    index["manifest_sha256"] = _digest(run_dir / "manifest.json")
    index_path.write_text(yaml.safe_dump(index), encoding="utf-8")
    with pytest.raises(ValueError, match="declared-cell contract"):
        c11_results.verify_c11_artifact(repository, index_path)


def test_overwrite_and_final_revalidation_fail_without_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, index_path, _ = _fixture_repository(tmp_path)
    monkeypatch.setattr(c11_results, "_git_state", lambda _: ("2" * 40, False))
    destination = repository / "report"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        c11_results.build_c11_results(repository, index_path, destination)

    destination.rmdir()
    original = c11_results.verify_c11_artifact
    calls = 0

    def fail_second(repository_arg: Path, index_arg: Path) -> tuple[dict[str, Any], Path]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("simulated final artifact change")
        return original(repository_arg, index_arg)

    monkeypatch.setattr(c11_results, "verify_c11_artifact", fail_second)
    with pytest.raises(ValueError, match="simulated final artifact change"):
        c11_results.build_c11_results(repository, index_path, destination)
    assert not destination.exists()
    assert not list(repository.glob(".report.tmp-*"))
