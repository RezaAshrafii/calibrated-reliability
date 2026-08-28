"""Cross-experiment artifact publication and environment-provenance tests."""

import inspect
from pathlib import Path

import pytest

from calibrated_reliability.experiments import artifacts


def test_environment_records_every_declared_direct_runtime_dependency() -> None:
    """Manifests cannot silently omit a direct runtime dependency version."""
    packages = artifacts._environment()["packages"]
    expected = {
        "calibrated-reliability",
        "click",
        "matplotlib",
        "numpy",
        "pandas",
        "pyyaml",
        "scikit-learn",
        "scipy",
    }
    assert set(packages) == expected
    assert all(isinstance(version, str) and version for version in packages.values())


def test_every_artifact_writer_uses_the_uniform_publish_guard() -> None:
    """Every experiment writer rechecks immutability at final publication."""
    for experiment in range(1, 9):
        writer = getattr(artifacts, f"write_c{experiment:02d}_run")
        assert "_publish_directory(temporary_dir, run_dir)" in inspect.getsource(writer)


def test_publish_guard_rejects_an_existing_destination(tmp_path: Path) -> None:
    """A destination appearing during a write cannot be replaced at publish time."""
    temporary = tmp_path / ".temporary"
    destination = tmp_path / "published"
    temporary.mkdir()
    destination.mkdir()
    (destination / "sentinel").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        artifacts._publish_directory(temporary, destination)
    assert (destination / "sentinel").read_text(encoding="utf-8") == "preserve"
    assert temporary.is_dir()


def test_c04_writer_does_not_claim_aci_bootstrap_semantics() -> None:
    """Static C04 manifests must not inherit C08 adaptive-bootstrap wording."""
    assert "bootstrap_interpretation" not in inspect.getsource(artifacts.write_c04_run)
