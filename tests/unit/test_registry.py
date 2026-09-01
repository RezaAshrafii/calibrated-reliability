"""Tests for fail-closed data registry validation."""

from pathlib import Path

import pytest
import yaml

from calibrated_reliability.data.registry import (
    compute_sha256,
    load_registry,
    validate_registry,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic_fd001.txt"


def _registry(tmp_path: Path, filename: str = "synthetic_fd001.txt") -> Path:
    target = tmp_path / filename
    target.write_bytes(FIXTURE.read_bytes())
    payload = {
        "version": 1,
        "source": "https://example.invalid/cmapss",
        "license": "not specified",
        "files": [
            {
                "filename": filename,
                "sha256": compute_sha256(target),
                "expected_bytes": target.stat().st_size,
                "expected_rows": 4,
                "expected_engines": 2,
                "kind": "cmapss",
            }
        ],
    }
    registry = tmp_path / "registry.yaml"
    registry.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return registry


def test_valid_fixture_passes(tmp_path: Path) -> None:
    """A complete matching fixture is accepted."""
    results = validate_registry(_registry(tmp_path), tmp_path)
    assert results[0].valid


def test_checksum_mismatch_fails_closed(tmp_path: Path) -> None:
    """A changed file fails even when its shape remains valid."""
    registry = _registry(tmp_path)
    (tmp_path / "synthetic_fd001.txt").write_text(
        FIXTURE.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    result = validate_registry(registry, tmp_path)[0]
    assert not result.valid
    assert any("sha256" in error for error in result.errors)


def test_missing_file_is_actionable(tmp_path: Path) -> None:
    """A missing file reports the exact path and fails."""
    registry = _registry(tmp_path)
    (tmp_path / "synthetic_fd001.txt").unlink()
    result = validate_registry(registry, tmp_path)[0]
    assert not result.valid
    assert "missing file" in result.errors[0]


def test_schema_mismatch_fails(tmp_path: Path) -> None:
    """A malformed row fails schema validation."""
    registry = _registry(tmp_path)
    target = tmp_path / "synthetic_fd001.txt"
    target.write_text("1 1 2\n", encoding="utf-8")
    result = validate_registry(registry, tmp_path)[0]
    assert not result.valid
    assert any("fields" in error for error in result.errors)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("filename", "../escape.txt"),
        ("filename", "nested/file.txt"),
        ("sha256", "A" * 64),
        ("expected_bytes", True),
        ("expected_rows", "4"),
        ("expected_engines", False),
        ("kind", True),
    ],
)
def test_registry_rejects_coercion_and_unsafe_record_fields(
    tmp_path: Path, field: str, value: object
) -> None:
    registry = _registry(tmp_path)
    payload = yaml.safe_load(registry.read_text(encoding="utf-8"))
    payload["files"][0][field] = value
    registry.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_registry(registry)


def test_registry_rejects_unknown_top_level_and_record_fields(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    payload = yaml.safe_load(registry.read_text(encoding="utf-8"))
    payload["unexpected"] = "value"
    registry.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="top-level"):
        load_registry(registry)

    payload.pop("unexpected")
    payload["files"][0]["unexpected"] = "value"
    registry.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="entry"):
        load_registry(registry)


def test_validate_file_rejects_symlinked_registered_file(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    target = tmp_path / "synthetic_fd001.txt"
    linked = tmp_path / "linked.txt"
    try:
        linked.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable on this host")
    payload = yaml.safe_load(registry.read_text(encoding="utf-8"))
    payload["files"][0]["filename"] = linked.name
    payload["files"][0]["sha256"] = compute_sha256(linked)
    registry.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = validate_registry(registry, tmp_path)[0]
    assert not result.valid
    assert "direct regular file" in result.errors[0]
