"""Fail-closed validation for registered research data files."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

EXPECTED_CMAPSS_COLUMNS = 26


@dataclass(frozen=True)
class FileRecord:
    """Expected metadata for one registered file."""

    filename: str
    sha256: str
    expected_bytes: int
    expected_rows: int
    expected_engines: int | None
    kind: str


@dataclass(frozen=True)
class ValidationResult:
    """Validation outcome for one file."""

    filename: str
    valid: bool
    errors: tuple[str, ...]


def load_registry(path: Path) -> tuple[FileRecord, ...]:
    """Load and validate a YAML registry with concrete metadata."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("files"), list):
        raise ValueError("Registry must contain a top-level 'files' list")
    records: list[FileRecord] = []
    for entry in raw["files"]:
        if not isinstance(entry, dict):
            raise ValueError("Each registry entry must be a mapping")
        required = ("filename", "sha256", "expected_bytes", "expected_rows", "kind")
        if any(key not in entry for key in required):
            raise ValueError(f"Registry entry is missing a required field: {entry}")
        records.append(
            FileRecord(
                filename=str(entry["filename"]),
                sha256=str(entry["sha256"]).lower(),
                expected_bytes=int(entry["expected_bytes"]),
                expected_rows=int(entry["expected_rows"]),
                expected_engines=(
                    None
                    if entry.get("expected_engines") is None
                    else int(entry["expected_engines"])
                ),
                kind=str(entry["kind"]),
            )
        )
    return tuple(records)


def compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a file's SHA-256 digest without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _data_shape(path: Path, kind: str) -> tuple[int, int | None]:
    """Return row and engine counts for a registered C-MAPSS file."""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if kind == "rul":
        return len(lines), None
    engine_ids: set[int] = set()
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != EXPECTED_CMAPSS_COLUMNS:
            raise ValueError(
                f"{path.name}: line {line_number} has {len(fields)} fields; "
                f"expected {EXPECTED_CMAPSS_COLUMNS}"
            )
        try:
            engine_ids.add(int(fields[0]))
        except ValueError as exc:
            raise ValueError(f"{path.name}: invalid engine_id on line {line_number}") from exc
    return len(lines), len(engine_ids)


def validate_file(data_root: Path, record: FileRecord) -> ValidationResult:
    """Validate presence, bytes, hash, row count, engine count, and schema."""
    path = data_root / record.filename
    errors: list[str] = []
    if not path.is_file():
        return ValidationResult(record.filename, False, (f"missing file: {path}",))
    if path.stat().st_size != record.expected_bytes:
        errors.append(f"byte size {path.stat().st_size} != {record.expected_bytes}")
    actual_hash = compute_sha256(path)
    if actual_hash != record.sha256:
        errors.append(f"sha256 {actual_hash} != {record.sha256}")
    try:
        rows, engines = _data_shape(path, record.kind)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        if rows != record.expected_rows:
            errors.append(f"row count {rows} != {record.expected_rows}")
        if record.expected_engines is not None and engines != record.expected_engines:
            errors.append(f"engine count {engines} != {record.expected_engines}")
    return ValidationResult(record.filename, not errors, tuple(errors))


def validate_registry(registry_path: Path, data_root: Path) -> tuple[ValidationResult, ...]:
    """Validate every file in a registry and return all results."""
    return tuple(validate_file(data_root, record) for record in load_registry(registry_path))
