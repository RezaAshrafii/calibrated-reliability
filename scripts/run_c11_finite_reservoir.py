"""Generate the single immutable C11 artifact after readiness authorization."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import click

from calibrated_reliability.data.loader import load_rul, load_test, load_train
from calibrated_reliability.data.registry import load_registry, validate_file
from calibrated_reliability.experiments.artifacts import write_c11_run
from calibrated_reliability.experiments.c11 import C11Config, run_c11


@click.command()
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option(
    "--registry", "registry_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--data-root", type=click.Path(path_type=Path), default=Path("data/raw"))
@click.option("--output-root", type=click.Path(path_type=Path), default=Path("outputs/c11"))
def main(config_path: Path, registry_path: Path, data_root: Path, output_root: Path) -> None:
    """Execute C11 only after its separate implementation-readiness review passes."""
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise click.ClickException("C11 requires a clean Git worktree")
    config = C11Config.from_yaml(config_path.read_text(encoding="utf-8"))
    records = {record.filename: record for record in load_registry(registry_path)}
    names = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]
    provenance: dict[str, dict[str, Any]] = {}
    for name in names:
        checked = validate_file(data_root, records[name])
        if not checked.valid:
            raise click.ClickException(f"Data verification failed for {name}")
        record = records[name]
        provenance[name] = {
            "sha256": record.sha256,
            "bytes": record.expected_bytes,
            "rows": record.expected_rows,
            "engines": record.expected_engines,
        }
    result = run_c11(
        load_train(data_root / "train_FD001.txt"),
        load_test(data_root / "test_FD001.txt"),
        load_rul(data_root / "RUL_FD001.txt"),
        config,
    )
    run_dir = write_c11_run(
        output_root,
        sha,
        config,
        result,
        config_path,
        registry_path,
        provenance,
    )
    click.echo(f"PASS artifact={run_dir}")


if __name__ == "__main__":
    sys.exit(main())
