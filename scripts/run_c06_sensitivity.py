"""Generate immutable C06 cap/calibration sensitivity artifacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

from calibrated_reliability.data.loader import load_rul, load_test, load_train
from calibrated_reliability.data.registry import load_registry, validate_file
from calibrated_reliability.experiments.artifacts import write_c06_run
from calibrated_reliability.experiments.c06 import C06Config, run_c06_condition


@click.command()
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option(
    "--registry", "registry_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--data-root", type=click.Path(path_type=Path), default=Path("data/raw"))
@click.option("--output-root", type=click.Path(path_type=Path), default=Path("outputs/c06"))
def main(config_path: Path, registry_path: Path, data_root: Path, output_root: Path) -> None:
    """Run every preregistered C06 condition from a clean commit."""
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise click.ClickException("C06 requires a clean Git worktree")
    config = C06Config.from_yaml(config_path.read_text(encoding="utf-8"))
    records = {record.filename: record for record in load_registry(registry_path)}
    names = ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")
    provenance = {}
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
    train = load_train(data_root / names[0])
    test, rul = load_test(data_root / names[1]), load_rul(data_root / names[2])
    for condition in config.conditions:
        for seed in config.seeds:
            result = run_c06_condition(train, test, rul, config, condition, seed)
            run_dir = write_c06_run(
                output_root,
                condition,
                seed,
                sha,
                config,
                result,
                config_path,
                registry_path,
                provenance,
            )
            click.echo(f"PASS condition={condition.id} seed={seed} artifact={run_dir}")


if __name__ == "__main__":
    sys.exit(main())
