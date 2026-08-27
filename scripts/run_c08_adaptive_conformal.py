"""Generate immutable C08 adaptive-conformal artifacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import click

from calibrated_reliability.data.loader import load_rul, load_test, load_train
from calibrated_reliability.data.registry import load_registry, validate_file
from calibrated_reliability.experiments.artifacts import write_c08_run
from calibrated_reliability.experiments.c08 import C08Config, run_c08_seed


@click.command()
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option(
    "--registry", "registry_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--data-root", type=click.Path(path_type=Path), default=Path("data/raw"))
@click.option("--output-root", type=click.Path(path_type=Path), default=Path("outputs/c08"))
def main(config_path: Path, registry_path: Path, data_root: Path, output_root: Path) -> None:
    """Fit once per seed and generate prequential artifacts for all C08 targets."""
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise click.ClickException("C08 requires a clean Git worktree")
    config = C08Config.from_yaml(config_path.read_text(encoding="utf-8"))
    records = {record.filename: record for record in load_registry(registry_path)}
    names = [
        "train_FD001.txt",
        *[
            name
            for target in config.targets
            for name in (f"test_{target}.txt", f"RUL_{target}.txt")
        ],
    ]
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
    train = load_train(data_root / "train_FD001.txt")
    target_data = {
        target: (
            load_test(data_root / f"test_{target}.txt"),
            load_rul(data_root / f"RUL_{target}.txt"),
        )
        for target in config.targets
    }
    for seed in config.c02.seeds:
        for target, result in run_c08_seed(train, target_data, config, seed).items():
            run_dir = write_c08_run(
                output_root,
                target,
                seed,
                sha,
                config,
                result,
                config_path,
                registry_path,
                provenance,
            )
            click.echo(f"PASS target={target} seed={seed} artifact={run_dir}")


if __name__ == "__main__":
    sys.exit(main())
