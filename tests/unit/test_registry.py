"""Tests for fail-closed data registry validation."""

from pathlib import Path

import yaml

from calibrated_reliability.data.registry import compute_sha256, validate_registry

FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic_fd001.txt"


def _registry(tmp_path: Path, filename: str = "synthetic_fd001.txt") -> Path:
    target = tmp_path / filename
    target.write_bytes(FIXTURE.read_bytes())
    payload = {
        "files": [
            {
                "filename": filename,
                "sha256": compute_sha256(target),
                "expected_bytes": target.stat().st_size,
                "expected_rows": 4,
                "expected_engines": 2,
                "kind": "cmapss",
            }
        ]
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
