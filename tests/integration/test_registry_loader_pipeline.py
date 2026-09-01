"""Non-skipping synthetic registry-to-loader integration test."""

from pathlib import Path

import yaml

from calibrated_reliability.data.loader import load_train
from calibrated_reliability.data.registry import compute_sha256, validate_registry


def test_synthetic_registry_verification_then_loader(tmp_path: Path) -> None:
    """A registry-verified fixture can be loaded by the production loader."""
    fixture = Path(__file__).parents[1] / "fixtures" / "synthetic_fd001.txt"
    data_root = tmp_path / "raw"
    data_root.mkdir()
    target = data_root / fixture.name
    target.write_bytes(fixture.read_bytes())
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "files": [
                    {
                        "filename": target.name,
                        "kind": "cmapss",
                        "expected_bytes": target.stat().st_size,
                        "expected_rows": 4,
                        "expected_engines": 2,
                        "sha256": compute_sha256(target),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    results = validate_registry(registry, data_root)
    assert results[0].valid
    assert len(load_train(target)) == 4
