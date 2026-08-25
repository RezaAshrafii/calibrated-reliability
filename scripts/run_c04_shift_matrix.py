"""Generate C04 FD001-to-FD001/2/3/4 shift-matrix artifacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import click

from calibrated_reliability.data.loader import load_rul, load_test, load_train
from calibrated_reliability.data.registry import load_registry, validate_file
from calibrated_reliability.experiments.artifacts import write_c04_run
from calibrated_reliability.experiments.c04 import C04Config, run_c04_target


@click.command()
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option(
    "--registry", "registry_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--data-root", type=click.Path(path_type=Path), default=Path("data/raw"))
@click.option("--output-root", type=click.Path(path_type=Path), default=Path("outputs/c04"))
def main(config_path: Path, registry_path: Path, data_root: Path, output_root: Path) -> None:
    """Run all declared C04 target domains and seeds."""
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise click.ClickException("C04 requires a clean Git worktree")
    config = C04Config.from_yaml(config_path.read_text(encoding="utf-8"))
    records = {record.filename: record for record in load_registry(registry_path)}
    provenance: dict[str, dict[str, Any]] = {}
    for target in config.targets:
        for name in (f"test_{target}.txt", f"RUL_{target}.txt"):
            checked = validate_file(data_root, records[name])
            if not checked.valid:
                raise click.ClickException(f"Data verification failed for {name}")
            provenance[name] = {
                "sha256": records[name].sha256,
                "bytes": records[name].expected_bytes,
            }
    train = load_train(data_root / "train_FD001.txt")
    for target in config.targets:
        test = load_test(data_root / f"test_{target}.txt")
        rul = load_rul(data_root / f"RUL_{target}.txt")
        for seed in config.c02.seeds:
            result = run_c04_target(train, test, rul, config, seed)
            run_dir = write_c04_run(
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
