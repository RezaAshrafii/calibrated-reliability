"""Fail-closed validation for registered research data files."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

EXPECTED_CMAPSS_COLUMNS = 26
VALID_KINDS = {"cmapss", "rul"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
TOP_LEVEL_FIELDS = {"version", "source", "license", "files"}
RECORD_FIELDS = {
    "filename",
    "sha256",
    "expected_bytes",
    "expected_rows",
    "expected_engines",
    "kind",
}


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
    if not isinstance(raw, dict) or set(raw) - TOP_LEVEL_FIELDS:
        raise ValueError("Registry top-level fields differ from the supported schema")
    optional_metadata = ("source", "license")
    if any(
        type(raw[field]) is not str or not raw[field]
        for field in optional_metadata
        if field in raw
    ):
        raise ValueError("Registry source and license must be non-empty strings when present")
    if type(raw.get("version")) is not int or raw["version"] != 1:
        raise ValueError("Registry version must be the integer 1")
    if not isinstance(raw.get("files"), list):
        raise ValueError("Registry must contain a top-level 'files' list")
    if not raw["files"]:
        raise ValueError("Registry must contain at least one file")
    records: list[FileRecord] = []
    filenames: set[str] = set()
    for entry in raw["files"]:
        if not isinstance(entry, dict) or set(entry) != RECORD_FIELDS:
            raise ValueError("Each registry entry must be a mapping")
        filename = entry["filename"]
        sha256 = entry["sha256"]
        kind = entry["kind"]
        if (
            type(filename) is not str
            or not filename
            or filename in {".", ".."}
            or "/" in filename
            or "\\" in filename
            or Path(filename).name != filename
            or Path(filename).is_absolute()
        ):
            raise ValueError(f"Registry filename must be a direct filename: {filename!r}")
        if type(sha256) is not str or SHA256_PATTERN.fullmatch(sha256) is None:
            raise ValueError(f"Invalid SHA-256 for {filename}")
        if type(kind) is not str:
            raise ValueError(f"Registry kind must be a string for {filename}")
        for field in ("expected_bytes", "expected_rows"):
            if type(entry[field]) is not int or entry[field] < 0:
                raise ValueError(f"Registry {field} must be a nonnegative integer for {filename}")
        engines = entry["expected_engines"]
        if engines is not None and (type(engines) is not int or engines < 0):
            raise ValueError(
                f"Registry expected_engines must be null or a nonnegative integer for {filename}"
            )
        if filename in filenames:
            raise ValueError(f"Duplicate registry filename: {filename}")
        if kind not in VALID_KINDS:
            raise ValueError(f"Unsupported registry kind: {kind}")
        filenames.add(filename)
        records.append(
            FileRecord(
                filename=filename,
                sha256=sha256,
                expected_bytes=entry["expected_bytes"],
                expected_rows=entry["expected_rows"],
                expected_engines=engines,
                kind=kind,
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
        if not lines:
            raise ValueError(f"{path.name}: RUL file is empty")
        for line_number, line in enumerate(lines, start=1):
            fields = line.split()
            if len(fields) != 1:
                raise ValueError(f"{path.name}: line {line_number} must contain one RUL value")
            try:
                value = float(fields[0])
            except ValueError as exc:
                raise ValueError(f"{path.name}: invalid RUL on line {line_number}") from exc
            if not value.is_integer() or value < 0:
                raise ValueError(f"{path.name}: RUL values must be nonnegative integers")
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
    root = data_root.resolve()
    path = root / record.filename
    errors: list[str] = []
    if path.is_symlink() or path.resolve().parent != root:
        return ValidationResult(
            record.filename,
            False,
            (f"registered file must be a direct regular file inside data root: {path}",),
        )
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
