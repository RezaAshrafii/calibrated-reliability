"""Build a checksum-anchored public archive of verified artifact inputs."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from calibrated_reliability.reporting.c11_results import verify_c11_artifact
from calibrated_reliability.reporting.results import load_artifact_index, verify_indexed_artifacts

ARCHIVE_SCHEMA_VERSION = 1
ARCHIVE_METADATA_PATHS = (
    ".python-version",
    "uv.lock",
    "data/registry.yaml",
    "docs/artifact_index.yaml",
    "docs/c11_artifact_index.yaml",
    "reports/results",
    "reports/c11",
    "configs/cmapss",
    "docs/decisions",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_git_sha(repository: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise ValueError("Artifact archive requires a clean Git worktree")
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(sha) != 40:
        raise ValueError("Artifact archive requires a full Git SHA")
    return sha


def _regular_files(repository: Path, selected: tuple[Path, ...]) -> tuple[Path, ...]:
    files: list[Path] = []
    for path in selected:
        try:
            path.relative_to(repository)
        except ValueError as error:
            raise ValueError(f"Archive input escapes repository: {path}") from error
        if path.is_symlink():
            raise ValueError(f"Archive input must not be a symlink: {path}")
        if path.is_file():
            files.append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(f"Archive input is unavailable: {path}")
        for child in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
            if child.is_symlink():
                raise ValueError(f"Archive input must not contain a symlink: {child}")
            if child.is_file():
                files.append(child)
            elif not child.is_dir():
                raise ValueError(f"Archive input is not a regular file or directory: {child}")
    return tuple(sorted(set(files), key=lambda item: item.relative_to(repository).as_posix()))


def build_official_artifact_archive(repository: Path, destination: Path) -> Path:
    """Package verified official C01--C08 and C11 artifacts outside the checkout."""
    repository = repository.resolve()
    destination = destination.resolve()
    if not destination.is_absolute() or destination.suffix.lower() != ".zip":
        raise ValueError("Artifact archive destination must be an absolute .zip path")
    try:
        destination.relative_to(repository)
    except ValueError:
        pass
    else:
        raise ValueError("Artifact archive destination must be outside the repository")
    if destination.exists():
        raise FileExistsError(destination)
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"Archive destination parent is unavailable: {destination.parent}")

    builder_sha = _clean_git_sha(repository)
    gate_d_index = repository / "docs" / "artifact_index.yaml"
    c11_index = repository / "docs" / "c11_artifact_index.yaml"
    gate_d_runs = verify_indexed_artifacts(repository, load_artifact_index(gate_d_index))
    c11_manifest, c11_run = verify_c11_artifact(repository, c11_index)
    official_roots = tuple(sorted({repository / run.entry.path for run in gate_d_runs}))
    selected = (
        *official_roots,
        c11_run.parent,
        *(repository / relative for relative in ARCHIVE_METADATA_PATHS),
    )
    files = _regular_files(repository, selected)
    manifest_files = [
        {
            "path": path.relative_to(repository).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]
    archive_manifest: dict[str, Any] = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "builder_git_sha": builder_sha,
        "contains_raw_cmapss_data": False,
        "gate_d_official_roots": [
            root.relative_to(repository).as_posix() for root in official_roots
        ],
        "c11_artifact_root": c11_run.parent.relative_to(repository).as_posix(),
        "c11_manifest_sha256": _sha256(c11_run / "manifest.json"),
        "c11_producing_git_sha": c11_manifest["git"]["sha"],
        "files": manifest_files,
    }
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{destination.stem}.", dir=destination.parent))
    temporary_archive = temporary_dir / destination.name
    try:
        with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                relative = path.relative_to(repository).as_posix()
                entry = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                entry.external_attr = 0o100644 << 16
                archive.writestr(entry, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
            manifest_entry = zipfile.ZipInfo(
                "ARCHIVE_MANIFEST.json", date_time=(1980, 1, 1, 0, 0, 0)
            )
            manifest_entry.external_attr = 0o100644 << 16
            archive.writestr(
                manifest_entry,
                (json.dumps(archive_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
                compress_type=zipfile.ZIP_DEFLATED,
            )
        if _clean_git_sha(repository) != builder_sha:
            raise ValueError("Git state changed during artifact archive construction")
        if destination.exists():
            raise FileExistsError(destination)
        temporary_archive.rename(destination)
        return destination
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir, ignore_errors=True)
