"""Fail-closed deterministic reporting for the single verified-candidate C11 run."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, cast

import pandas as pd
import yaml

INDEX_FIELDS = {
    "schema_version",
    "index_id",
    "experiment_id",
    "artifact_root",
    "run_directory",
    "manifest_sha256",
    "producing_git_sha",
    "expected_artifact_count",
    "status",
}
SHA_RE = re.compile(r"[0-9a-f]{40}")
DIGEST_RE = re.compile(r"[0-9a-f]{64}")
EXPECTED_ROWS = {
    "enumeration_cells.csv": 10,
    "distribution_summary.csv": 16,
    "quantile_distribution.csv": 121,
    "beta_binomial_distribution.csv": 606,
    "reservoir_scores.csv": 40,
    "evaluation_scores.csv": 100,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _git_state(repository: Path) -> tuple[str, bool]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if SHA_RE.fullmatch(sha) is None:
        raise RuntimeError("Could not resolve a full Git SHA")
    return sha, dirty


def _load_index(index_path: Path) -> dict[str, Any]:
    value = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != INDEX_FIELDS:
        raise ValueError("C11 artifact index fields differ from schema version 1")
    index = cast(dict[str, Any], value)
    if index["schema_version"] != 1 or index["experiment_id"] != "C11":
        raise ValueError("Unsupported C11 artifact index")
    if index["status"] != "VERIFIED_CANDIDATE":
        raise ValueError("C11 artifact must remain a verified candidate until report audit")
    if (
        isinstance(index["expected_artifact_count"], bool)
        or index["expected_artifact_count"] != 11
    ):
        raise ValueError("C11 expected artifact count must be the integer 11")
    if DIGEST_RE.fullmatch(str(index["manifest_sha256"])) is None:
        raise ValueError("C11 manifest SHA-256 is malformed")
    if SHA_RE.fullmatch(str(index["producing_git_sha"])) is None:
        raise ValueError("C11 producing Git SHA is malformed")
    root = PurePosixPath(str(index["artifact_root"]))
    run = PurePosixPath(str(index["run_directory"]))
    if root.is_absolute() or root.parts[:1] != ("outputs",) or len(root.parts) != 2:
        raise ValueError("C11 artifact root must be one top-level outputs directory")
    if run.is_absolute() or len(run.parts) != 1 or run.name in {".", ".."}:
        raise ValueError("C11 run directory must be a direct child name")
    return index


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def verify_c11_artifact(repository: Path, index_path: Path) -> tuple[dict[str, Any], Path]:
    """Verify the externally anchored C11 run and its reporting inputs."""
    index = _load_index(index_path)
    run_dir = repository / str(index["artifact_root"]) / str(index["run_directory"])
    manifest_path = run_dir / "manifest.json"
    if not run_dir.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError("Indexed C11 run is unavailable")
    if _sha256(manifest_path) != index["manifest_sha256"]:
        raise ValueError("C11 manifest trust anchor mismatch")
    manifest = _json(manifest_path)
    if manifest.get("experiment_id") != "C11" or manifest.get("run_id") != index["run_directory"]:
        raise ValueError("C11 manifest identity mismatch")
    git = manifest.get("git")
    if git != {"dirty": False, "sha": index["producing_git_sha"]}:
        raise ValueError("C11 producer provenance mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or len(artifacts) != index["expected_artifact_count"]:
        raise ValueError("C11 artifact declaration count mismatch")
    declared = cast(dict[str, Any], artifacts)
    expected_files = {"manifest.json", *declared}
    actual_files = {path.name for path in run_dir.iterdir() if path.is_file()}
    if actual_files != expected_files or any(path.is_dir() for path in run_dir.iterdir()):
        raise ValueError("C11 run file set differs from the immutable manifest")
    for name, digest in declared.items():
        pure = PurePosixPath(name)
        if (
            len(pure.parts) != 1
            or pure.name in {".", ".."}
            or DIGEST_RE.fullmatch(str(digest)) is None
        ):
            raise ValueError("C11 manifest contains an unsafe artifact declaration")
        path = run_dir / name
        if (
            path.is_symlink()
            or path.resolve().parent != run_dir.resolve()
            or _sha256(path) != digest
        ):
            raise ValueError(f"C11 artifact integrity failure: {name}")
    for name, count in EXPECTED_ROWS.items():
        if len(_read_csv(run_dir / name)) != count:
            raise ValueError(f"C11 row-count contract failed: {name}")
    cells = _read_csv(run_dir / "enumeration_cells.csv")
    summaries = _read_csv(run_dir / "distribution_summary.csv")
    if (
        cells["cell_id"].nunique() != 10
        or len(summaries.loc[summaries["status"] == "evaluated"]) != 12
    ):
        raise ValueError("C11 evaluated-cell accounting mismatch")
    excluded = summaries.loc[summaries["status"] != "evaluated"]
    if len(excluded) != 4 or set(excluded["reference"]) != {"not_evaluated"}:
        raise ValueError("C11 excluded cells are missing or misrepresented")
    observation = _json(run_dir / "observation_mechanism.json")
    if (
        observation.get("reservoir_engine_count") != 40
        or observation.get("official_endpoint_count") != 100
    ):
        raise ValueError("C11 observation-mechanism counts mismatch")
    return manifest, run_dir


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_c11_results(repository: Path, index_path: Path, destination: Path) -> Path:
    """Publish immutable C11 report tables from the verified-candidate artifact."""
    initial_sha, dirty = _git_state(repository)
    if dirty:
        raise RuntimeError("C11 reporting requires a clean Git worktree")
    manifest, run_dir = verify_c11_artifact(repository, index_path)
    if destination.exists():
        raise FileExistsError(f"C11 report destination already exists: {destination}")
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        cells = _read_csv(run_dir / "enumeration_cells.csv")
        summaries = _read_csv(run_dir / "distribution_summary.csv")
        observation = _json(run_dir / "observation_mechanism.json")
        cells.to_csv(temp_dir / "cells.csv", index=False, lineterminator="\n")
        summaries.to_csv(temp_dir / "discrepancies.csv", index=False, lineterminator="\n")
        observation_row = {
            "metric": observation["metric"],
            "unit": observation["unit"],
            "weighting": observation["weighting"],
            "distance": observation["distance"],
            "reservoir_engine_count": observation["reservoir_engine_count"],
            "official_endpoint_count": observation["official_endpoint_count"],
            "evaluation_cap_saturation_fraction": observation[
                "evaluation_cap_saturation_fraction"
            ],
            "reservoir_distinct_residuals": observation["reservoir_distinct_residuals"],
            "reservoir_residual_tied_rows": observation["reservoir_residual_tied_rows"],
        }
        _write_csv(temp_dir / "observation_summary.csv", [observation_row], list(observation_row))
        report_names = ["cells.csv", "discrepancies.csv", "observation_summary.csv"]
        provenance = {
            "schema_version": 1,
            "experiment_id": "C11",
            "builder_git_sha": initial_sha,
            "builder_git_clean": True,
            "artifact_index": {
                "path": index_path.relative_to(repository).as_posix(),
                "sha256": _sha256(index_path),
            },
            "artifact_root": run_dir.relative_to(repository).as_posix(),
            "run_id": manifest["run_id"],
            "producing_git_sha": manifest["git"]["sha"],
            "manifest_sha256": _sha256(run_dir / "manifest.json"),
            "source_artifact_hashes": manifest["artifacts"],
            "csv_float_parser": 'pandas.read_csv(..., float_precision="round_trip")',
            "claim_boundary": (
                "benchmark-specific descriptive audit; not a new theorem or conformal method"
            ),
            "report_hashes": {name: _sha256(temp_dir / name) for name in report_names},
        }
        (temp_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        checksum_names = [*report_names, "provenance.json"]
        (temp_dir / "checksums.sha256").write_text(
            "".join(f"{_sha256(temp_dir / name)}  {name}\n" for name in checksum_names),
            encoding="utf-8",
            newline="\n",
        )
        final_sha, final_dirty = _git_state(repository)
        verify_c11_artifact(repository, index_path)
        if final_dirty or final_sha != initial_sha:
            raise RuntimeError("Git state changed during C11 report construction")
        if destination.exists():
            raise FileExistsError(
                f"C11 report destination appeared during construction: {destination}"
            )
        temp_dir.rename(destination)
        return destination
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
