"""Gate D deterministic reporting, index, and provenance regression tests."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from calibrated_reliability.reporting import results

ROOT = Path(__file__).resolve().parents[2]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_index(repository: Path, *, duplicate_official: bool = False) -> Path:
    entries: list[dict[str, Any]] = []
    for number in range(1, 9):
        experiment_id = f"C{number:02d}"
        entries.append(
            {
                "path": f"outputs/{experiment_id.lower()}_official",
                "experiment_id": experiment_id,
                "status": "OFFICIAL",
                "expected_manifest_count": 1,
                "expected_git_shas": [str(number) * 40],
                "reason": "test_fixture",
            }
        )
    if duplicate_official:
        entries.append(
            {
                "path": "outputs/c01_duplicate",
                "experiment_id": "C01",
                "status": "OFFICIAL",
                "expected_manifest_count": 0,
                "expected_git_shas": ["f" * 40],
                "reason": "invalid_duplicate",
            }
        )
    payload = {
        "schema_version": 1,
        "index_id": "test_gate_d",
        "required_experiments": [f"C{number:02d}" for number in range(1, 9)],
        "mixed_sha_policy": {
            "mode": "allow_across_experiments_only",
            "provenance_field": "source_git_shas",
            "require_single_sha_per_official_tree": True,
        },
        "entries": entries,
    }
    path = repository / "artifact_index.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_runs(repository: Path) -> None:
    for number in range(1, 9):
        experiment_id = f"C{number:02d}"
        run = repository / "outputs" / f"{experiment_id.lower()}_official" / "run"
        run.mkdir(parents=True)
        (run / "predictions.csv").write_text(
            "engine_id,y_true_raw,y_true,mean_raw,mean,ridge_raw,ridge,"
            "hist_gradient_boosting_raw,hist_gradient_boosting\n"
            "1,0.10000000000000002,0.10000000000000002,1,1,1,1,1,1\n",
            encoding="utf-8",
        )
        (run / "metrics.json").write_text("{}\n", encoding="utf-8")
        (run / "resolved_config.json").write_text("{}\n", encoding="utf-8")
        artifacts = {
            filename: _hash(run / filename)
            for filename in ("predictions.csv", "metrics.json", "resolved_config.json")
        }
        manifest = {
            "run_id": f"{experiment_id}_run",
            "experiment_id": experiment_id,
            "git": {"sha": str(number) * 40, "dirty": False},
            "seed": 13,
            "source": "FD001",
            "target": "FD001",
            "evaluation_unit": "engine_endpoint",
            "artifacts": artifacts,
        }
        (run / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )


def _fixture_repository(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _write_runs(repository)
    return repository, _write_index(repository)


def test_index_requires_exactly_one_official_tree_per_experiment(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    index_path = _write_index(repository, duplicate_official=True)

    with pytest.raises(ValueError, match="exactly one OFFICIAL"):
        results.load_artifact_index(index_path)


def test_unindexed_artifact_tree_fails_closed(tmp_path: Path) -> None:
    repository, index_path = _fixture_repository(tmp_path)
    (repository / "outputs" / "undeclared").mkdir()

    with pytest.raises(ValueError, match="unindexed"):
        results.verify_indexed_artifacts(repository, results.load_artifact_index(index_path))


def test_missing_declared_artifact_fails_closed(tmp_path: Path) -> None:
    repository, index_path = _fixture_repository(tmp_path)
    missing = repository / "outputs" / "c01_official" / "run" / "predictions.csv"
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="Missing declared artifact"):
        results.verify_indexed_artifacts(repository, results.load_artifact_index(index_path))


def test_artifact_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    repository, index_path = _fixture_repository(tmp_path)
    changed = repository / "outputs" / "c01_official" / "run" / "predictions.csv"
    changed.write_text(changed.read_text(encoding="utf-8") + "2,1,1,1,1,1,1,1,1\n")

    with pytest.raises(ValueError, match="Artifact hash mismatch"):
        results.verify_indexed_artifacts(repository, results.load_artifact_index(index_path))


def test_round_trip_csv_parsing_preserves_float_bits(tmp_path: Path) -> None:
    repository, index_path = _fixture_repository(tmp_path)
    run = results.verify_indexed_artifacts(repository, results.load_artifact_index(index_path))[0]

    value = float(results._read_predictions(run).loc[0, "y_true_raw"])

    assert struct.pack("!d", value) == struct.pack("!d", 0.10000000000000002)


def test_summary_is_deterministic_and_pending_is_not_zero() -> None:
    provenance = {
        "experiment_id": "C01",
        "target": "FD001",
        "condition": "primary",
        "model": "mean",
        "prediction_variant": "raw",
        "metric": "nasa_score",
        "seed": 13,
        "source_git_sha": "a" * 40,
        "artifact_root": "outputs/c01",
        "run_id": "run",
        "manifest_sha256": "b" * 64,
        "value": results.PENDING,
    }

    first = results._summary_rows([provenance], [])
    second = results._summary_rows([dict(reversed(list(provenance.items())))], [])

    assert first == second
    assert first[0]["mean"] == results.PENDING
    assert first[0]["n_numeric_values"] == 0


def test_summary_never_drops_pending_seed_from_denominator() -> None:
    base = {
        "experiment_id": "C01",
        "target": "FD001",
        "condition": "primary",
        "model": "mean",
        "prediction_variant": "raw",
        "metric": "nasa_score",
        "source_git_sha": "a" * 40,
        "artifact_root": "outputs/c01",
        "manifest_sha256": "b" * 64,
    }
    rows = [
        {**base, "seed": 13, "run_id": "run_13", "value": 1.0},
        {**base, "seed": 37, "run_id": "run_37", "value": results.PENDING},
    ]

    summary = results._summary_rows(rows, [])[0]

    assert summary["mean"] == results.PENDING
    assert summary["minimum"] == results.PENDING
    assert summary["maximum"] == results.PENDING
    assert summary["n_seeds"] == 2
    assert summary["n_numeric_values"] == 1


def test_builder_is_deterministic_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, index_path = _fixture_repository(tmp_path)

    def point_rows(run: results.VerifiedRun, predictions: pd.DataFrame) -> list[dict[str, Any]]:
        return [
            {
                **results._base_provenance(run),
                "model": "mean",
                "prediction_variant": "clipped",
                "metric": "mae",
                "value": float(predictions.loc[0, "y_true"]),
                "n_endpoints": len(predictions),
                "stored_metric_match": results.PENDING,
            }
        ]

    monkeypatch.setattr(results, "_point_rows", point_rows)

    def interval_rows(
        run: results.VerifiedRun, _predictions: pd.DataFrame
    ) -> list[dict[str, Any]]:
        return [
            {
                **results._base_provenance(run),
                "model": "mean",
                "alpha": 0.1,
                "metric": "coverage",
                "value": 1.0,
            }
        ]

    monkeypatch.setattr(results, "_interval_rows", interval_rows)
    first = tmp_path / "first"
    second = tmp_path / "second"
    results.build_results(repository, index_path, first, builder_git_sha="f" * 40)
    results.build_results(repository, index_path, second, builder_git_sha="f" * 40)

    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
    with pytest.raises(FileExistsError):
        results.build_results(repository, index_path, first, builder_git_sha="f" * 40)


def test_failed_build_removes_temporary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, index_path = _fixture_repository(tmp_path)
    monkeypatch.setattr(results, "_point_rows", lambda _run, _predictions: [{"value": 1.0}])
    monkeypatch.setattr(results, "_interval_rows", lambda _run, _predictions: [])
    monkeypatch.setattr(results, "_run_rows", lambda _runs: [{"run": "x"}])
    monkeypatch.setattr(results, "_summary_rows", lambda _point, _interval: [{"mean": 1.0}])

    original_write = results._write_csv
    calls = 0

    def fail_after_first(path: Path, rows: list[dict[str, Any]]) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected write failure")
        original_write(path, rows)

    monkeypatch.setattr(results, "_write_csv", fail_after_first)
    destination = tmp_path / "failed"

    with pytest.raises(OSError, match="injected"):
        results.build_results(repository, index_path, destination, builder_git_sha="f" * 40)

    assert not destination.exists()
    assert not list(tmp_path.glob(".failed.*"))


def test_tracked_gate_d_reports_match_their_provenance_hashes() -> None:
    report_root = ROOT / "reports" / "results"
    provenance = json.loads((report_root / "provenance.json").read_text(encoding="utf-8"))

    assert provenance["schema_version"] == 1
    assert len(provenance["builder_git_sha"]) == 40
    assert provenance["official_run_count"] == 105
    assert len(provenance["source_git_shas"]) == 8
    for filename, expected_hash in provenance["report_files"].items():
        assert _hash(report_root / filename) == expected_hash


def test_tracked_tables_preserve_pending_and_rank_semantics() -> None:
    report_root = ROOT / "reports" / "results"
    point = pd.read_csv(
        report_root / "point_metrics_by_seed.csv",
        dtype=str,
        keep_default_na=False,
        float_precision="round_trip",
    )
    interval = pd.read_csv(
        report_root / "interval_metrics_by_seed.csv",
        dtype=str,
        keep_default_na=False,
        float_precision="round_trip",
    )

    pending_point = point.loc[point["value"] == results.PENDING]
    assert not pending_point.empty
    assert set(pending_point["metric_status"]) == {"nonfinite_on_raw_support"}
    weighted = interval.loc[interval["experiment_id"] == "C05"]
    assert set(weighted["requested_rank_min"]) == {results.PENDING}
    assert set(weighted["unattainable_rank_policy"]) == {"not_applicable_to_weighted_quantile"}
    adaptive = interval.loc[interval["experiment_id"] == "C08"]
    assert set(adaptive["unattainable_rank_policy"]) == {"legacy_max_clamp"}
