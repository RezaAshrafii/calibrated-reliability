"""Gate D deterministic reporting, index, and provenance regression tests."""

from __future__ import annotations

import hashlib
import inspect
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


def _manifest_set_digest(tree: Path) -> str:
    manifests = sorted(tree.rglob("manifest.json"))
    payload = "".join(
        f"{path.relative_to(tree).as_posix()}\0{_hash(path)}\n" for path in manifests
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _refresh_manifest_digest(index_path: Path, artifact_path: str) -> None:
    payload = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    repository = index_path.parent
    payload["manifest_set_sha256"][artifact_path] = _manifest_set_digest(
        repository / artifact_path
    )
    index_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _index_document(index_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_index_document(index_path: Path, payload: dict[str, Any]) -> None:
    index_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


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
        (repository / "outputs" / "c01_duplicate").mkdir(parents=True)
        entries.append(
            {
                "path": "outputs/c01_duplicate",
                "experiment_id": "C01",
                "status": "OFFICIAL",
                "expected_manifest_count": 1,
                "expected_git_shas": ["f" * 40],
                "reason": "invalid_duplicate",
            }
        )
    payload = {
        "schema_version": 2,
        "index_id": "test_gate_d",
        "required_experiments": [f"C{number:02d}" for number in range(1, 9)],
        "mixed_sha_policy": {
            "mode": "allow_across_experiments_only",
            "provenance_field": "source_git_shas",
            "require_single_sha_per_official_tree": True,
        },
        "official_run_contracts": {
            f"C{number:02d}": {
                "source": "FD001",
                "targets": ["FD001"],
                "conditions": ["primary"],
                "seeds": [13],
                "evaluation_unit": "engine_endpoint",
            }
            for number in range(1, 9)
        },
        "manifest_set_sha256": {
            entry["path"]: _manifest_set_digest(repository / entry["path"]) for entry in entries
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


def _allow_test_builder_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(results, "_builder_sha", lambda _repository: "f" * 40)


def test_index_requires_exactly_one_official_tree_per_experiment(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    index_path = _write_index(repository, duplicate_official=True)

    with pytest.raises(ValueError, match="exactly one OFFICIAL"):
        results.load_artifact_index(index_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("expected_manifest_count", True, "must be an integer"),
        ("expected_git_shas", ["not-a-sha"], "Invalid expected Git SHA"),
        ("status", "CURRENT", "Invalid artifact status"),
        ("experiment_id", "C09", "Unsupported experiment ID"),
        ("path", "outputs/../escape", "one top-level outputs directory"),
    ],
)
def test_artifact_index_entry_schema_fails_closed(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    repository, index_path = _fixture_repository(tmp_path)
    payload = _index_document(index_path)
    payload["entries"][0][field] = value
    _write_index_document(index_path, payload)

    with pytest.raises(ValueError, match=message):
        results.load_artifact_index(index_path)


def test_artifact_index_unknown_field_fails_closed(tmp_path: Path) -> None:
    repository, index_path = _fixture_repository(tmp_path)
    payload = _index_document(index_path)
    payload["unexpected"] = "not allowed"
    _write_index_document(index_path, payload)

    with pytest.raises(ValueError, match="unknown=.*unexpected"):
        results.load_artifact_index(index_path)


def test_artifact_index_manifest_digest_schema_fails_closed(tmp_path: Path) -> None:
    repository, index_path = _fixture_repository(tmp_path)
    payload = _index_document(index_path)
    payload["manifest_set_sha256"]["outputs/c01_official"] = "not-a-digest"
    _write_index_document(index_path, payload)

    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
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


def test_manifest_edit_fails_against_tracked_manifest_set_digest(tmp_path: Path) -> None:
    repository, index_path = _fixture_repository(tmp_path)
    manifest_path = repository / "outputs" / "c01_official" / "run" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["seed"] = 37
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Manifest-set hash mismatch"):
        results.verify_indexed_artifacts(repository, results.load_artifact_index(index_path))


def test_exact_official_run_matrix_rejects_missing_and_unexpected_cell(tmp_path: Path) -> None:
    repository, index_path = _fixture_repository(tmp_path)
    manifest_path = repository / "outputs" / "c01_official" / "run" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["seed"] = 37
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest_digest(index_path, "outputs/c01_official")

    with pytest.raises(ValueError, match="Official run matrix mismatch"):
        results.verify_indexed_artifacts(repository, results.load_artifact_index(index_path))


def test_duplicate_official_run_id_fails_closed(tmp_path: Path) -> None:
    repository, index_path = _fixture_repository(tmp_path)
    manifest_path = repository / "outputs" / "c02_official" / "run" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_id"] = "C01_run"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest_digest(index_path, "outputs/c02_official")

    with pytest.raises(ValueError, match="Duplicate official run ID"):
        results.verify_indexed_artifacts(repository, results.load_artifact_index(index_path))


def test_manifest_artifact_path_traversal_fails_closed(tmp_path: Path) -> None:
    repository, index_path = _fixture_repository(tmp_path)
    run = repository / "outputs" / "c01_official" / "run"
    outside = run.parent / "outside.csv"
    outside.write_text("untrusted\n", encoding="utf-8")
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["../outside.csv"] = _hash(outside)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest_digest(index_path, "outputs/c01_official")

    with pytest.raises(ValueError, match="direct filenames"):
        results.verify_indexed_artifacts(repository, results.load_artifact_index(index_path))


def test_manifest_experiment_id_mismatch_fails_closed(tmp_path: Path) -> None:
    repository, index_path = _fixture_repository(tmp_path)
    manifest_path = repository / "outputs" / "c01_official" / "run" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["experiment_id"] = "C02"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_manifest_digest(index_path, "outputs/c01_official")

    with pytest.raises(ValueError, match="Experiment mismatch"):
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
    _allow_test_builder_sha(monkeypatch)

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
    results.build_results(repository, index_path, first)
    results.build_results(repository, index_path, second)

    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
    with pytest.raises(FileExistsError):
        results.build_results(repository, index_path, first)


def test_failed_build_removes_temporary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, index_path = _fixture_repository(tmp_path)
    _allow_test_builder_sha(monkeypatch)
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
        results.build_results(repository, index_path, destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".failed.*"))


def test_tracked_gate_d_reports_match_their_provenance_hashes() -> None:
    report_root = ROOT / "reports" / "results"
    provenance = json.loads((report_root / "provenance.json").read_text(encoding="utf-8"))

    # Schema 1 remains readable only during the implementation-to-publication
    # migration commit. The publication commit tightens this assertion to v2.
    assert provenance["schema_version"] in {1, 2}
    assert len(provenance["builder_git_sha"]) == 40
    assert provenance["official_run_count"] == 105
    assert len(provenance["source_git_shas"]) == 8
    for filename, expected_hash in provenance["report_files"].items():
        assert _hash(report_root / filename) == expected_hash
    if provenance["schema_version"] == 2:
        assert provenance["builder_git_clean"] is True
        checksums = (report_root / provenance["detached_checksum_file"]).read_text(
            encoding="utf-8"
        )
        for line in checksums.splitlines():
            expected_hash, filename = line.split("  ", maxsplit=1)
            assert _hash(report_root / filename) == expected_hash


def test_public_builder_api_cannot_accept_spoofed_git_sha() -> None:
    assert "builder_git_sha" not in inspect.signature(results.build_results).parameters


def test_builder_rejects_dirty_worktree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Completed:
        stdout = "?? untracked.txt\n"

    monkeypatch.setattr(results.subprocess, "run", lambda *args, **kwargs: Completed())

    with pytest.raises(ValueError, match="clean Git worktree"):
        results._builder_sha(tmp_path)


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
