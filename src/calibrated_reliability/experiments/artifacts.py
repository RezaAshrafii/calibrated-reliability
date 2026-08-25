"""Immutable artifact writing and provenance capture for research runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any

from calibrated_reliability.data.registry import compute_sha256
from calibrated_reliability.experiments.c01 import C01Config, C01Result
from calibrated_reliability.experiments.c02 import C02Config, C02Result

PACKAGE_NAMES = ("calibrated-reliability", "numpy", "pandas", "scikit-learn", "scipy")


def _repository_root() -> Path:
    """Locate the checkout root from this source file, independent of cwd."""
    return Path(__file__).resolve().parents[3]


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
    output_root.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        raise FileExistsError(run_dir)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=output_root))
    try:
        artifact_hashes: dict[str, str] = {}
        artifact_hashes["predictions.csv"] = _write_bytes(
            temporary_dir / "predictions.csv",
            result.predictions.to_csv(index=False).encode("utf-8"),
        )
        artifact_hashes["metrics.json"] = _write_bytes(
            temporary_dir / "metrics.json", _json_bytes(result.metrics)
        )
        artifact_hashes["split_manifest.json"] = _write_bytes(
            temporary_dir / "split_manifest.json",
            _json_bytes({"seed": seed, "partitions": result.partitions}),
        )
        artifact_hashes["resolved_config.json"] = _write_bytes(
            temporary_dir / "resolved_config.json", _json_bytes(config.as_dict())
        )
        log_content = (
            f"experiment_id=C01\nseed={seed}\ngit_sha={sha}\n"
            f"status=completed\nendpoint_rows={len(result.predictions)}\n"
        ).encode()
        artifact_hashes["run.log"] = _write_bytes(temporary_dir / "run.log", log_content)
        lock_path = _repository_root() / "uv.lock"
        if not lock_path.is_file():
            raise FileNotFoundError(f"Repository lockfile not found: {lock_path}")
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
            "lockfile": {"path": "uv.lock", "sha256": compute_sha256(lock_path)},
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
        _write_bytes(temporary_dir / "manifest.json", _json_bytes(manifest))
        if run_dir.exists():
            raise FileExistsError(run_dir)
        temporary_dir.rename(run_dir)
        return run_dir
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def write_c02_run(
    output_root: Path,
    seed: int,
    sha: str,
    config: C02Config,
    result: C02Result,
    config_path: Path,
    registry_path: Path,
    data_provenance: dict[str, dict[str, Any]],
) -> Path:
    """Write one atomic, provenance-complete C02 artifact directory."""
    run_id = f"C02_{sha[:12]}_seed_{seed}"
    run_dir = output_root / run_id
    output_root.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        raise FileExistsError(run_dir)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=output_root))
    try:
        artifact_hashes: dict[str, str] = {}
        artifact_hashes["predictions.csv"] = _write_bytes(
            temporary_dir / "predictions.csv",
            result.predictions.to_csv(index=False).encode("utf-8"),
        )
        artifact_hashes["metrics.json"] = _write_bytes(
            temporary_dir / "metrics.json", _json_bytes(result.metrics)
        )
        artifact_hashes["split_manifest.json"] = _write_bytes(
            temporary_dir / "split_manifest.json",
            _json_bytes(
                {
                    "seed": seed,
                    "partitions": result.partitions,
                    "cut_points": result.cut_points,
                }
            ),
        )
        artifact_hashes["resolved_config.json"] = _write_bytes(
            temporary_dir / "resolved_config.json", _json_bytes(config.as_dict())
        )
        artifact_hashes["quantiles.json"] = _write_bytes(
            temporary_dir / "quantiles.json", _json_bytes(result.quantiles)
        )
        log_content = (
            f"experiment_id=C02\nseed={seed}\ngit_sha={sha}\n"
            f"status=completed\nendpoint_rows={len(result.predictions)}\n"
        ).encode()
        artifact_hashes["run.log"] = _write_bytes(temporary_dir / "run.log", log_content)
        lock_path = _repository_root() / "uv.lock"
        if not lock_path.is_file():
            raise FileNotFoundError(f"Repository lockfile not found: {lock_path}")
        manifest = {
            "run_id": run_id,
            "experiment_id": config.experiment_id,
            "git": {"sha": sha, "dirty": False},
            "seed": seed,
            "source": config.source,
            "target": config.target,
            "evaluation_unit": config.evaluation_unit,
            "rul_cap": config.rul_cap,
            "alphas": list(config.alphas),
            "data": data_provenance,
            "data_registry": {
                "path": registry_path.as_posix(),
                "sha256": compute_sha256(registry_path),
            },
            "configuration": {
                "path": config_path.as_posix(),
                "sha256": compute_sha256(config_path),
            },
            "lockfile": {"path": "uv.lock", "sha256": compute_sha256(lock_path)},
            "environment": _environment(),
            "split_manifest": result.partitions,
            "calibration_cut_points": result.cut_points,
            "preprocessing": {"feature_names": result.feature_names},
            "models": ["mean", "ridge", "hist_gradient_boosting"],
            "quantiles": result.quantiles,
            "artifacts": artifact_hashes,
        }
        _write_bytes(temporary_dir / "manifest.json", _json_bytes(manifest))
        if run_dir.exists():
            raise FileExistsError(run_dir)
        temporary_dir.rename(run_dir)
        return run_dir
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
