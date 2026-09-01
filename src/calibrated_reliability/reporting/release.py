"""Build a checksum-anchored public archive of verified artifact inputs."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ReleaseInputs:
    """Verified source files and identity fields for one public archive."""

    files: tuple[Path, ...]
    gate_d_official_roots: tuple[Path, ...]
    c11_artifact_root: Path
    c11_manifest_sha256: str
    c11_producing_git_sha: str


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


def _tracked_metadata_files(repository: Path) -> tuple[Path, ...]:
    """Select release metadata from Git's tracked file set only."""
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", *ARCHIVE_METADATA_PATHS],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    relative_paths = tuple(
        Path(value.decode("utf-8")) for value in result.stdout.split(b"\0") if value
    )
    if not relative_paths:
        raise ValueError("Artifact archive metadata selection is empty")
    files = _regular_files(repository, tuple(repository / path for path in relative_paths))
    selected_names = {path.relative_to(repository).as_posix() for path in files}
    for declared in ARCHIVE_METADATA_PATHS:
        candidate = repository / declared
        if not candidate.exists() or candidate.is_symlink():
            raise FileNotFoundError(f"Tracked archive metadata is unavailable: {candidate}")
        prefix = f"{Path(declared).as_posix().rstrip('/')}/"
        if candidate.is_file() and Path(declared).as_posix() not in selected_names:
            raise ValueError(f"Archive metadata file is not tracked: {candidate}")
        if candidate.is_dir() and not any(name.startswith(prefix) for name in selected_names):
            raise ValueError(f"Archive metadata directory has no tracked files: {candidate}")
    return files


def _verified_release_inputs(repository: Path) -> ReleaseInputs:
    """Resolve only manifest-declared artifacts and tracked metadata."""
    gate_d_index = repository / "docs" / "artifact_index.yaml"
    c11_index = repository / "docs" / "c11_artifact_index.yaml"
    gate_d_runs = verify_indexed_artifacts(repository, load_artifact_index(gate_d_index))
    c11_manifest, c11_run = verify_c11_artifact(repository, c11_index)
    official_roots = tuple(sorted({repository / run.entry.path for run in gate_d_runs}))
    selected: list[Path] = []
    for run in gate_d_runs:
        selected.append(run.run_dir / "manifest.json")
        artifacts = run.manifest["artifacts"]
        if not isinstance(artifacts, dict):
            raise ValueError(f"Verified run has no artifact map: {run.run_dir}")
        selected.extend(run.run_dir / name for name in artifacts)
    selected.extend(path for path in c11_run.iterdir() if path.is_file())
    selected.extend(_tracked_metadata_files(repository))
    files = _regular_files(repository, tuple(selected))
    relative_names = {path.relative_to(repository).as_posix() for path in files}
    if any(name == "data/raw" or name.startswith("data/raw/") for name in relative_names):
        raise ValueError("Raw C-MAPSS data must never enter the artifact archive")
    return ReleaseInputs(
        files=files,
        gate_d_official_roots=official_roots,
        c11_artifact_root=c11_run.parent,
        c11_manifest_sha256=_sha256(c11_run / "manifest.json"),
        c11_producing_git_sha=c11_manifest["git"]["sha"],
    )


def _file_records(repository: Path, files: tuple[Path, ...]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(repository).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]


def _verify_completed_archive(archive_path: Path, archive_manifest: dict[str, Any]) -> None:
    """Fail closed unless the completed ZIP exactly matches its manifest."""
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        expected_names = [item["path"] for item in archive_manifest["files"]]
        expected_names.append("ARCHIVE_MANIFEST.json")
        if names != expected_names or len(names) != len(set(names)):
            raise ValueError("Completed artifact archive has an unexpected file set")
        retained_manifest = json.loads(archive.read("ARCHIVE_MANIFEST.json"))
        if retained_manifest != archive_manifest:
            raise ValueError("Completed artifact archive manifest differs from construction state")
        for item in archive_manifest["files"]:
            payload = archive.read(item["path"])
            if (
                len(payload) != item["bytes"]
                or hashlib.sha256(payload).hexdigest() != item["sha256"]
            ):
                raise ValueError(f"Completed artifact archive hash mismatch: {item['path']}")
        if any(name == "data/raw" or name.startswith("data/raw/") for name in names):
            raise ValueError("Completed artifact archive contains raw C-MAPSS data")


def build_official_artifact_archive(repository: Path, destination: Path) -> Path:
    """Package verified official C01--C08 and C11 artifacts outside the checkout."""
    if not destination.is_absolute() or destination.suffix.lower() != ".zip":
        raise ValueError("Artifact archive destination must be an absolute .zip path")
    repository = repository.resolve()
    destination = destination.resolve()
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
    inputs = _verified_release_inputs(repository)
    manifest_files = _file_records(repository, inputs.files)
    archive_manifest: dict[str, Any] = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "builder_git_sha": builder_sha,
        "contains_raw_cmapss_data": False,
        "gate_d_official_roots": [
            root.relative_to(repository).as_posix() for root in inputs.gate_d_official_roots
        ],
        "c11_artifact_root": inputs.c11_artifact_root.relative_to(repository).as_posix(),
        "c11_manifest_sha256": inputs.c11_manifest_sha256,
        "c11_producing_git_sha": inputs.c11_producing_git_sha,
        "files": manifest_files,
    }
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{destination.stem}.", dir=destination.parent))
    temporary_archive = temporary_dir / destination.name
    try:
        with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in inputs.files:
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
        final_inputs = _verified_release_inputs(repository)
        if (
            final_inputs != inputs
            or _file_records(repository, final_inputs.files) != manifest_files
        ):
            raise ValueError("Artifact archive inputs changed during construction")
        _verify_completed_archive(temporary_archive, archive_manifest)
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
