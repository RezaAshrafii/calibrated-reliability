"""Immutable artifact writing and provenance capture for research runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

from calibrated_reliability.data.registry import compute_sha256
from calibrated_reliability.experiments.c01 import C01Config, C01Result

PACKAGE_NAMES = ("calibrated-reliability", "numpy", "pandas", "scikit-learn", "scipy")


def _json_bytes(payload: Any) -> bytes:
    """Serialize JSON deterministically for hashing and writing."""
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_bytes(path: Path, content: bytes) -> str:
    """Write one artifact and return its SHA-256."""
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _environment() -> dict[str, Any]:
    """Capture interpreter, platform, and runtime package versions."""
    versions = {name: importlib.metadata.version(name) for name in PACKAGE_NAMES}
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": versions,
    }


def write_c01_run(
    output_root: Path,
    seed: int,
    sha: str,
    config: C01Config,
    result: C01Result,
    config_path: Path,
    registry_path: Path,
    data_provenance: dict[str, dict[str, Any]],
) -> Path:
    """Write one immutable C01 run directory and provenance manifest."""
    run_id = f"C01_{sha[:12]}_seed_{seed}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    artifact_hashes: dict[str, str] = {}
    artifact_hashes["predictions.csv"] = _write_bytes(
        run_dir / "predictions.csv",
        result.predictions.to_csv(index=False).encode("utf-8"),
    )
    artifact_hashes["metrics.json"] = _write_bytes(
        run_dir / "metrics.json", _json_bytes(result.metrics)
    )
    artifact_hashes["split_manifest.json"] = _write_bytes(
        run_dir / "split_manifest.json",
        _json_bytes({"seed": seed, "partitions": result.partitions}),
    )
    artifact_hashes["resolved_config.json"] = _write_bytes(
        run_dir / "resolved_config.json", _json_bytes(config.as_dict())
    )
    log_content = (
        f"experiment_id=C01\nseed={seed}\ngit_sha={sha}\n"
        f"status=completed\nendpoint_rows={len(result.predictions)}\n"
    ).encode()
    artifact_hashes["run.log"] = _write_bytes(run_dir / "run.log", log_content)
    lock_path = Path("uv.lock")
    manifest = {
        "run_id": run_id,
        "experiment_id": config.experiment_id,
        "git": {"sha": sha, "dirty": False},
        "seed": seed,
        "source": config.source,
        "target": config.target,
        "evaluation_unit": config.evaluation_unit,
        "rul_cap": config.rul_cap,
        "prediction_clip": [config.clip_min, config.clip_max],
        "data": data_provenance,
        "data_registry": {
            "path": registry_path.as_posix(),
            "sha256": compute_sha256(registry_path),
        },
        "configuration": {
            "path": config_path.as_posix(),
            "sha256": compute_sha256(config_path),
        },
        "lockfile": {"path": lock_path.as_posix(), "sha256": compute_sha256(lock_path)},
        "environment": _environment(),
        "split_manifest": result.partitions,
        "preprocessing": {
            "regime_aware_scaling": False,
            "selected_sensors": result.selected_sensors,
            "feature_names": result.feature_names,
        },
        "models": config.as_dict()["models"],
        "artifacts": artifact_hashes,
    }
    _write_bytes(run_dir / "manifest.json", _json_bytes(manifest))
    return run_dir
