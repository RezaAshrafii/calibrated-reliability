"""Fail-closed Gate D builder for verified C01--C08 artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import yaml

from calibrated_reliability.evaluation.conformal import (
    bootstrap_interval_metric_cis,
    interval_metrics,
)
from calibrated_reliability.evaluation.metrics import rul_metrics

ArtifactStatus = Literal["OFFICIAL", "SUPERSEDED"]
PENDING = "PENDING"
POINT_MODELS = ("mean", "ridge", "hist_gradient_boosting")
ALPHAS = (0.1, 0.05)
REQUIRED_EXPERIMENTS = tuple(f"C{number:02d}" for number in range(1, 9))
INDEX_KEYS = {"schema_version", "index_id", "required_experiments", "mixed_sha_policy", "entries"}
ENTRY_KEYS = {
    "path",
    "experiment_id",
    "status",
    "expected_manifest_count",
    "expected_git_shas",
    "reason",
}
POLICY_KEYS = {"mode", "provenance_field", "require_single_sha_per_official_tree"}


@dataclass(frozen=True)
class ArtifactEntry:
    """One explicitly classified top-level artifact tree."""

    path: str
    experiment_id: str
    status: ArtifactStatus
    expected_manifest_count: int
    expected_git_shas: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class ArtifactIndex:
    """Validated artifact selection and mixed-revision policy."""

    index_id: str
    entries: tuple[ArtifactEntry, ...]
    mixed_sha_policy: dict[str, Any]


@dataclass(frozen=True)
class VerifiedRun:
    """One immutable run whose manifest and declared artifacts were verified."""

    entry: ArtifactEntry
    run_dir: Path
    manifest: dict[str, Any]
    manifest_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return cast(dict[str, Any], payload)


def _strict_keys(payload: dict[str, Any], expected: set[str], label: str) -> None:
    if set(payload) != expected:
        raise ValueError(
            f"{label} fields differ from the required schema; "
            f"missing={sorted(expected - set(payload))}, unknown={sorted(set(payload) - expected)}"
        )


def _plain_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _safe_output_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Artifact index paths must be non-empty strings")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 2
        or path.parts[0] != "outputs"
    ):
        raise ValueError(f"Artifact tree must be one top-level outputs directory: {value}")
    return value


def load_artifact_index(index_path: Path) -> ArtifactIndex:
    """Load and strictly validate the tracked official/superseded index."""
    payload = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Artifact index must be a mapping")
    document = cast(dict[str, Any], payload)
    _strict_keys(document, INDEX_KEYS, "Artifact index")
    if _plain_int(document["schema_version"], "schema_version") != 1:
        raise ValueError("Unsupported artifact-index schema version")
    if not isinstance(document["index_id"], str) or not document["index_id"]:
        raise ValueError("index_id must be a non-empty string")
    if document["required_experiments"] != list(REQUIRED_EXPERIMENTS):
        raise ValueError("Artifact index must require exactly C01 through C08")
    policy = document["mixed_sha_policy"]
    if not isinstance(policy, dict):
        raise ValueError("mixed_sha_policy must be a mapping")
    policy = cast(dict[str, Any], policy)
    _strict_keys(policy, POLICY_KEYS, "mixed_sha_policy")
    if policy != {
        "mode": "allow_across_experiments_only",
        "provenance_field": "source_git_shas",
        "require_single_sha_per_official_tree": True,
    }:
        raise ValueError("The Gate D mixed-SHA policy cannot be changed")
    raw_entries = document["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("Artifact index entries must be a non-empty list")
    entries: list[ArtifactEntry] = []
    seen_paths: set[str] = set()
    for number, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Artifact entry {number} must be a mapping")
        raw = cast(dict[str, Any], raw_entry)
        _strict_keys(raw, ENTRY_KEYS, f"Artifact entry {number}")
        path = _safe_output_path(raw["path"])
        if path in seen_paths:
            raise ValueError(f"Duplicate artifact tree in index: {path}")
        seen_paths.add(path)
        experiment_id = raw["experiment_id"]
        if experiment_id not in REQUIRED_EXPERIMENTS:
            raise ValueError(f"Unsupported experiment ID: {experiment_id}")
        status = raw["status"]
        if status not in {"OFFICIAL", "SUPERSEDED"}:
            raise ValueError(f"Invalid artifact status for {path}: {status}")
        count = _plain_int(raw["expected_manifest_count"], f"{path} manifest count")
        if count < 0:
            raise ValueError("Expected manifest counts cannot be negative")
        shas = raw["expected_git_shas"]
        if not isinstance(shas, list) or any(
            not isinstance(sha, str)
            or len(sha) != 40
            or any(character not in "0123456789abcdef" for character in sha)
            for sha in shas
        ):
            raise ValueError(f"Invalid expected Git SHA list for {path}")
        reason = raw["reason"]
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"Artifact entry {path} requires a reason")
        entries.append(
            ArtifactEntry(
                path=path,
                experiment_id=str(experiment_id),
                status=cast(ArtifactStatus, status),
                expected_manifest_count=count,
                expected_git_shas=tuple(shas),
                reason=reason,
            )
        )
    for experiment_id in REQUIRED_EXPERIMENTS:
        official = [
            entry
            for entry in entries
            if entry.experiment_id == experiment_id and entry.status == "OFFICIAL"
        ]
        if len(official) != 1:
            raise ValueError(f"{experiment_id} must have exactly one OFFICIAL artifact tree")
        if len(official[0].expected_git_shas) != 1:
            raise ValueError(f"OFFICIAL tree {official[0].path} must declare exactly one Git SHA")
    return ArtifactIndex(
        index_id=str(document["index_id"]),
        entries=tuple(entries),
        mixed_sha_policy=policy,
    )


def _validate_tree_inventory(repository_root: Path, index: ArtifactIndex) -> None:
    outputs = repository_root / "outputs"
    actual = {
        path.relative_to(repository_root).as_posix() for path in outputs.iterdir() if path.is_dir()
    }
    indexed = {entry.path for entry in index.entries}
    if actual != indexed:
        raise ValueError(
            "Artifact-tree inventory differs from the tracked index; "
            f"unindexed={sorted(actual - indexed)}, missing={sorted(indexed - actual)}"
        )


def verify_indexed_artifacts(
    repository_root: Path, index: ArtifactIndex
) -> tuple[VerifiedRun, ...]:
    """Verify all indexed trees and return only immutable OFFICIAL runs."""
    _validate_tree_inventory(repository_root, index)
    official_runs: list[VerifiedRun] = []
    for entry in index.entries:
        tree = repository_root / PurePosixPath(entry.path)
        manifests = sorted(tree.glob("*/manifest.json"), key=lambda path: path.as_posix())
        if len(manifests) != entry.expected_manifest_count:
            raise ValueError(
                f"Manifest count mismatch for {entry.path}: "
                f"expected {entry.expected_manifest_count}, found {len(manifests)}"
            )
        observed_shas: set[str] = set()
        for manifest_path in manifests:
            manifest = _json(manifest_path)
            run_dir = manifest_path.parent
            if manifest.get("experiment_id") != entry.experiment_id:
                raise ValueError(f"Experiment mismatch in {manifest_path}")
            git = manifest.get("git")
            if not isinstance(git, dict) or git.get("dirty") is not False:
                raise ValueError(f"Run is not recorded from a clean revision: {manifest_path}")
            sha = git.get("sha")
            if not isinstance(sha, str) or sha not in entry.expected_git_shas:
                raise ValueError(f"Unexpected producing Git SHA in {manifest_path}: {sha}")
            observed_shas.add(sha)
            artifacts = manifest.get("artifacts")
            if not isinstance(artifacts, dict) or not artifacts:
                raise ValueError(f"Missing artifact hash map in {manifest_path}")
            for filename, expected_hash in artifacts.items():
                if not isinstance(filename, str) or not isinstance(expected_hash, str):
                    raise ValueError(f"Malformed artifact hash map in {manifest_path}")
                artifact_path = run_dir / filename
                if not artifact_path.is_file():
                    raise FileNotFoundError(f"Missing declared artifact: {artifact_path}")
                if _sha256(artifact_path) != expected_hash:
                    raise ValueError(f"Artifact hash mismatch: {artifact_path}")
            for required in ("predictions.csv", "metrics.json", "resolved_config.json"):
                if required not in artifacts:
                    raise ValueError(
                        f"Required reporting artifact is undeclared: {run_dir / required}"
                    )
            if entry.status == "OFFICIAL":
                official_runs.append(
                    VerifiedRun(
                        entry=entry,
                        run_dir=run_dir,
                        manifest=manifest,
                        manifest_sha256=_sha256(manifest_path),
                    )
                )
        if observed_shas != set(entry.expected_git_shas):
            raise ValueError(
                f"Observed Git SHAs differ from the index for {entry.path}: "
                f"expected={sorted(entry.expected_git_shas)}, observed={sorted(observed_shas)}"
            )
        if entry.status == "OFFICIAL" and len(observed_shas) != 1:
            raise ValueError(f"OFFICIAL tree contains mixed Git SHAs: {entry.path}")
    return tuple(
        sorted(
            official_runs,
            key=lambda run: (
                run.entry.experiment_id,
                str(run.manifest.get("target", "")),
                str(_condition_id(run.manifest)),
                int(run.manifest["seed"]),
                str(run.manifest["run_id"]),
            ),
        )
    )


def _condition_id(manifest: dict[str, Any]) -> str:
    condition = manifest.get("condition")
    if condition is None:
        return "primary"
    if not isinstance(condition, dict) or not isinstance(condition.get("id"), str):
        raise ValueError(f"Malformed condition in {manifest.get('run_id')}")
    return str(condition["id"])


def _read_predictions(run: VerifiedRun) -> pd.DataFrame:
    frame = pd.read_csv(run.run_dir / "predictions.csv", float_precision="round_trip")
    if frame.empty or "engine_id" not in frame or frame["engine_id"].duplicated().any():
        raise ValueError(f"Predictions must contain one non-empty row per engine: {run.run_dir}")
    return frame


def _read_config(run: VerifiedRun) -> dict[str, Any]:
    return _json(run.run_dir / "resolved_config.json")


def _artifact_hash(run: VerifiedRun, filename: str) -> str:
    artifacts = cast(dict[str, Any], run.manifest["artifacts"])
    value = artifacts.get(filename)
    if not isinstance(value, str):
        raise ValueError(f"Missing artifact hash for {filename} in {run.run_dir}")
    return value


def _base_provenance(run: VerifiedRun) -> dict[str, Any]:
    return {
        "experiment_id": run.entry.experiment_id,
        "target": str(run.manifest.get("target", "")),
        "condition": _condition_id(run.manifest),
        "seed": int(run.manifest["seed"]),
        "run_id": str(run.manifest["run_id"]),
        "artifact_root": run.entry.path,
        "source_git_sha": str(cast(dict[str, Any], run.manifest["git"])["sha"]),
        "manifest_sha256": run.manifest_sha256,
        "predictions_sha256": _artifact_hash(run, "predictions.csv"),
        "metrics_sha256": _artifact_hash(run, "metrics.json"),
    }


def _assert_close(actual: float, expected: Any, label: str) -> None:
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        raise ValueError(f"Stored metric is not numeric: {label}")
    if not math.isclose(actual, float(expected), rel_tol=1e-11, abs_tol=1e-11):
        raise ValueError(f"Metric reconstruction mismatch for {label}: {actual} != {expected}")


def _point_rows(run: VerifiedRun, predictions: pd.DataFrame) -> list[dict[str, Any]]:
    if run.entry.experiment_id == "C03":
        return []
    required = {"y_true_raw", "y_true"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Missing raw/clipped truth columns in {run.run_dir}")
    rows: list[dict[str, Any]] = []
    stored = _json(run.run_dir / "metrics.json")
    for model in POINT_MODELS:
        for variant, truth_column, prediction_column in (
            ("raw", "y_true_raw", f"{model}_raw"),
            ("clipped", "y_true", model),
        ):
            if prediction_column not in predictions:
                raise ValueError(f"Missing prediction column {prediction_column} in {run.run_dir}")
            metrics: dict[str, float | str] = {}
            try:
                metrics.update(
                    rul_metrics(predictions[truth_column], predictions[prediction_column])
                )
            except OverflowError:
                truth = pd.to_numeric(predictions[truth_column], errors="raise").to_numpy(
                    dtype="float64"
                )
                estimate = pd.to_numeric(predictions[prediction_column], errors="raise").to_numpy(
                    dtype="float64"
                )
                error = estimate - truth
                metrics = {
                    "rmse": float(np.sqrt(np.mean(np.square(error)))),
                    "mae": float(np.mean(np.abs(error))),
                    "signed_error": float(np.mean(error)),
                    "nasa_score": PENDING,
                }
            for metric_name, value in metrics.items():
                stored_match: str | bool
                if value == PENDING:
                    stored_match = PENDING
                    metric_status = "nonfinite_on_raw_support"
                elif variant == "clipped" and run.entry.experiment_id in {"C01", "C07"}:
                    model_metrics = stored.get(model)
                    if not isinstance(model_metrics, dict):
                        raise ValueError(
                            f"Missing stored point metrics for {model} in {run.run_dir}"
                        )
                    _assert_close(
                        float(value),
                        model_metrics.get(metric_name),
                        f"{run.run_dir}:{model}:{metric_name}",
                    )
                    stored_match = True
                    metric_status = "reconstructed"
                else:
                    stored_match = PENDING
                    metric_status = "reconstructed"
                rows.append(
                    {
                        **_base_provenance(run),
                        "model": model,
                        "prediction_variant": variant,
                        "metric": metric_name,
                        "value": value,
                        "metric_status": metric_status,
                        "n_endpoints": len(predictions),
                        "stored_metric_match": stored_match,
                    }
                )
    return rows


def _interval_models(experiment_id: str) -> tuple[str, ...]:
    if experiment_id == "C03":
        return ("hist_gradient_boosting_cqr",)
    if experiment_id in {"C02", "C04", "C05", "C06", "C08"}:
        return POINT_MODELS
    return ()


def _interval_columns(experiment_id: str, model: str, alpha: float) -> tuple[str, str]:
    alpha_text = str(alpha)
    if experiment_id == "C03":
        return f"alpha_{alpha_text}_lower", f"alpha_{alpha_text}_upper"
    return f"{model}_alpha_{alpha_text}_lower", f"{model}_alpha_{alpha_text}_upper"


def _stored_interval_metrics(
    stored: dict[str, Any], experiment_id: str, model: str, alpha: float
) -> dict[str, Any]:
    key = f"alpha_{alpha}"
    if experiment_id == "C03":
        value = stored.get(key)
    else:
        model_value = stored.get(model)
        value = model_value.get(key) if isinstance(model_value, dict) else None
    if not isinstance(value, dict):
        raise ValueError(f"Missing stored interval metric cell: {experiment_id}/{model}/{key}")
    return cast(dict[str, Any], value)


def _n_cal(run: VerifiedRun) -> int:
    path = run.run_dir / "calibration_scores.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing calibration scores: {path}")
    scores = pd.read_csv(path, float_precision="round_trip")
    if scores.empty:
        raise ValueError(f"Calibration scores cannot be empty: {path}")
    return len(scores)


def _rank_diagnostics(
    run: VerifiedRun,
    predictions: pd.DataFrame,
    model: str,
    alpha: float,
    n_cal: int,
) -> dict[str, Any]:
    experiment_id = run.entry.experiment_id
    if experiment_id == "C05":
        quantile_column = f"{model}_alpha_{alpha}_quantile"
        return {
            "n_cal": n_cal,
            "requested_rank_min": PENDING,
            "requested_rank_max": PENDING,
            "effective_rank_min": PENDING,
            "effective_rank_max": PENDING,
            "finite_rank_attainable_fraction": PENDING,
            "unattainable_rank_fraction": PENDING,
            "quantile_regime": "weighted_pointwise_not_integer_ranked",
            "unattainable_rank_policy": "not_applicable_to_weighted_quantile",
            "n_distinct_quantiles": int(predictions[quantile_column].nunique()),
        }
    if experiment_id == "C08":
        prefix = f"{model}_alpha_{alpha}"
        alpha_used = pd.to_numeric(predictions[f"{prefix}_alpha_used"], errors="raise").to_numpy(
            dtype="float64"
        )
        ranks = np.ceil((n_cal + 1) * (1.0 - alpha_used)).astype(int)
        effective = np.minimum(ranks, n_cal)
        attainable = ranks <= n_cal
        regimes = np.where(
            ranks < n_cal,
            "interior",
            np.where(ranks == n_cal, "max_statistic", "finite_quantile_unattainable"),
        )
        return {
            "n_cal": n_cal,
            "requested_rank_min": int(ranks.min()),
            "requested_rank_max": int(ranks.max()),
            "effective_rank_min": int(effective.min()),
            "effective_rank_max": int(effective.max()),
            "finite_rank_attainable_fraction": float(attainable.mean()),
            "unattainable_rank_fraction": float((~attainable).mean()),
            "quantile_regime": "|".join(sorted(set(regimes.tolist()))),
            "unattainable_rank_policy": "legacy_max_clamp",
            "n_distinct_quantiles": int(predictions[f"{prefix}_quantile"].nunique()),
        }
    rank = int(math.ceil((n_cal + 1) * (1.0 - alpha)))
    if rank > n_cal:
        raise ValueError(
            f"Unexpected unattainable rank in {experiment_id}: n_cal={n_cal}, alpha={alpha}"
        )
    regime = "interior" if rank < n_cal else "max_statistic"
    return {
        "n_cal": n_cal,
        "requested_rank_min": rank,
        "requested_rank_max": rank,
        "effective_rank_min": rank,
        "effective_rank_max": rank,
        "finite_rank_attainable_fraction": 1.0,
        "unattainable_rank_fraction": 0.0,
        "quantile_regime": regime,
        "unattainable_rank_policy": "finite_observed_order_statistic",
        "n_distinct_quantiles": 1,
    }


def _interval_rows(run: VerifiedRun, predictions: pd.DataFrame) -> list[dict[str, Any]]:
    models = _interval_models(run.entry.experiment_id)
    if not models:
        return []
    config = _read_config(run)
    cap = config.get("rul_cap", run.manifest.get("rul_cap"))
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(float(cap)):
        raise ValueError(f"Invalid RUL cap in {run.run_dir}")
    bootstrap = config.get("bootstrap")
    if not isinstance(bootstrap, dict):
        raise ValueError(f"Missing bootstrap configuration in {run.run_dir}")
    n_resamples = int(bootstrap["n_resamples"])
    confidence_level = float(bootstrap["confidence_level"])
    seed = int(run.manifest["seed"])
    stored = _json(run.run_dir / "metrics.json")
    n_cal = _n_cal(run)
    rows: list[dict[str, Any]] = []
    for model in models:
        for alpha in ALPHAS:
            lower_column, upper_column = _interval_columns(run.entry.experiment_id, model, alpha)
            if not {"y_true", lower_column, upper_column}.issubset(predictions.columns):
                raise ValueError(f"Missing interval columns for {model}/{alpha} in {run.run_dir}")
            values = interval_metrics(
                predictions["y_true"],
                predictions[lower_column],
                predictions[upper_column],
                alpha,
                float(cap),
            )
            cis = bootstrap_interval_metric_cis(
                predictions["y_true"],
                predictions[lower_column],
                predictions[upper_column],
                alpha,
                float(cap),
                seed,
                n_resamples,
                confidence_level,
            )
            stored_cell = _stored_interval_metrics(stored, run.entry.experiment_id, model, alpha)
            for metric_name, value in values.items():
                _assert_close(
                    value,
                    stored_cell.get(metric_name),
                    f"{run.run_dir}:{model}:{alpha}:{metric_name}",
                )
                stored_ci = stored_cell.get("bootstrap_ci")
                if not isinstance(stored_ci, dict) or not isinstance(
                    stored_ci.get(metric_name), dict
                ):
                    raise ValueError(
                        f"Missing stored bootstrap CI: {run.run_dir}:{model}:{alpha}:{metric_name}"
                    )
                metric_ci = cast(dict[str, Any], stored_ci[metric_name])
                _assert_close(cis[metric_name]["lower"], metric_ci.get("lower"), "bootstrap lower")
                _assert_close(cis[metric_name]["upper"], metric_ci.get("upper"), "bootstrap upper")
                rows.append(
                    {
                        **_base_provenance(run),
                        "model": model,
                        "alpha": alpha,
                        "metric": metric_name,
                        "value": value,
                        "bootstrap_ci_lower": cis[metric_name]["lower"],
                        "bootstrap_ci_upper": cis[metric_name]["upper"],
                        "bootstrap_resamples": n_resamples,
                        "bootstrap_confidence_level": confidence_level,
                        "target_scale": float(cap),
                        "n_endpoints": len(predictions),
                        "stored_metric_match": True,
                        **_rank_diagnostics(run, predictions, model, alpha, n_cal),
                    }
                )
    return rows


def _run_rows(runs: tuple[VerifiedRun, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        manifest = run.manifest
        rows.append(
            {
                **_base_provenance(run),
                "source": str(manifest.get("source", "")),
                "evaluation_unit": str(manifest.get("evaluation_unit", "")),
                "status": run.entry.status,
                "endpoint_rows": len(_read_predictions(run)),
            }
        )
    return rows


def _summary_rows(
    point_rows: list[dict[str, Any]], interval_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_type, source_rows, grouping in (
        (
            "point",
            point_rows,
            ("experiment_id", "target", "condition", "model", "prediction_variant", "metric"),
        ),
        (
            "interval",
            interval_rows,
            ("experiment_id", "target", "condition", "model", "alpha", "metric"),
        ),
    ):
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for source in source_rows:
            groups.setdefault(tuple(source[key] for key in grouping), []).append(source)
        for key in sorted(groups, key=lambda values: tuple(str(value) for value in values)):
            group = groups[key]
            numeric_values = [
                float(item["value"])
                for item in group
                if isinstance(item["value"], (int, float)) and not isinstance(item["value"], bool)
            ]
            values = np.asarray(numeric_values, dtype="float64")
            source_shas = sorted({str(item["source_git_sha"]) for item in group})
            roots = sorted({str(item["artifact_root"]) for item in group})
            rows.append(
                {
                    "result_type": result_type,
                    **dict(zip(grouping, key, strict=True)),
                    "mean": float(values.mean()) if len(values) else PENDING,
                    "sample_std": float(values.std(ddof=1)) if len(values) > 1 else PENDING,
                    "minimum": float(values.min()) if len(values) else PENDING,
                    "maximum": float(values.max()) if len(values) else PENDING,
                    "n_seeds": len({int(item["seed"]) for item in group}),
                    "n_numeric_values": len(values),
                    "source_git_shas": "|".join(source_shas),
                    "artifact_roots": "|".join(roots),
                    "source_run_ids": "|".join(sorted(str(item["run_id"]) for item in group)),
                    "manifest_sha256s": "|".join(
                        sorted(str(item["manifest_sha256"]) for item in group)
                    ),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty results table: {path.name}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="raise"
        )
        writer.writeheader()
        writer.writerows(rows)


def _builder_sha(repository_root: Path, explicit_sha: str | None) -> str:
    if explicit_sha is not None:
        if len(explicit_sha) != 40 or any(
            character not in "0123456789abcdef" for character in explicit_sha
        ):
            raise ValueError(
                "Explicit builder Git SHA must be 40 lowercase hexadecimal characters"
            )
        return explicit_sha
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise ValueError("Results builder requires a clean Git worktree")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(revision) != 40:
        raise ValueError("Unable to resolve a full builder Git SHA")
    return revision


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def build_results(
    repository_root: Path,
    index_path: Path,
    output_root: Path,
    *,
    builder_git_sha: str | None = None,
) -> Path:
    """Validate official artifacts and atomically publish deterministic reports."""
    repository_root = repository_root.resolve()
    index_path = index_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    resolved_builder_sha = _builder_sha(repository_root, builder_git_sha)
    index = load_artifact_index(index_path)
    runs = verify_indexed_artifacts(repository_root, index)
    if {run.entry.experiment_id for run in runs} != set(REQUIRED_EXPERIMENTS):
        raise ValueError("Verified runs do not cover exactly C01 through C08")
    point_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    for run in runs:
        predictions = _read_predictions(run)
        point_rows.extend(_point_rows(run, predictions))
        interval_rows.extend(_interval_rows(run, predictions))
    summary_rows = _summary_rows(point_rows, interval_rows)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        _write_csv(temporary / "artifact_runs.csv", _run_rows(runs))
        _write_csv(temporary / "point_metrics_by_seed.csv", point_rows)
        _write_csv(temporary / "interval_metrics_by_seed.csv", interval_rows)
        _write_csv(temporary / "summary.csv", summary_rows)
        report_files = sorted(path for path in temporary.iterdir() if path.is_file())
        provenance = {
            "schema_version": 1,
            "builder_git_sha": resolved_builder_sha,
            "artifact_index": {
                "path": index_path.relative_to(repository_root).as_posix(),
                "sha256": _sha256(index_path),
                "index_id": index.index_id,
            },
            "mixed_sha_policy": index.mixed_sha_policy,
            "source_git_shas": sorted(
                {str(cast(dict[str, Any], run.manifest["git"])["sha"]) for run in runs}
            ),
            "official_artifact_roots": [
                entry.path for entry in index.entries if entry.status == "OFFICIAL"
            ],
            "official_run_count": len(runs),
            "report_files": {path.name: _sha256(path) for path in report_files},
            "csv_float_parser": 'pandas.read_csv(float_precision="round_trip")',
            "pending_semantics": "PENDING is not zero and is never excluded as a numeric value",
        }
        (temporary / "provenance.json").write_bytes(_json_bytes(provenance))
        if output_root.exists():
            raise FileExistsError(output_root)
        temporary.rename(output_root)
        return output_root
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
