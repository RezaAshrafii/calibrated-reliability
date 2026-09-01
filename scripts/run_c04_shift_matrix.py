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
from calibrated_reliability.experiments.c04 import C04Config, run_c04_seed


@click.command()
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option(
    "--registry", "registry_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--data-root", type=click.Path(path_type=Path), default=Path("data/raw"))
@click.option("--output-root", type=click.Path(path_type=Path), default=Path("outputs/c04"))
@click.option("--seed", "requested_seeds", multiple=True, type=int)
def main(
    config_path: Path,
    registry_path: Path,
    data_root: Path,
    output_root: Path,
    requested_seeds: tuple[int, ...],
) -> None:
    """Run all declared C04 target domains and seeds."""
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise click.ClickException("C04 requires a clean Git worktree")
    config = C04Config.from_yaml(config_path.read_text(encoding="utf-8"))
    targets = config.targets
    seeds = requested_seeds or config.c02.seeds
    if set(seeds) - set(config.c02.seeds):
        raise click.ClickException("Undeclared C04 seed requested")
    records = {record.filename: record for record in load_registry(registry_path)}
    provenance: dict[str, dict[str, Any]] = {}
    source_record = records["train_FD001.txt"]
    source_checked = validate_file(data_root, source_record)
    if not source_checked.valid:
        raise click.ClickException("Data verification failed for train_FD001.txt")
    provenance["train_FD001.txt"] = {
        "sha256": source_record.sha256,
        "bytes": source_record.expected_bytes,
        "rows": source_record.expected_rows,
        "engines": source_record.expected_engines,
    }
    for target in targets:
        for name in (f"test_{target}.txt", f"RUL_{target}.txt"):
            checked = validate_file(data_root, records[name])
            if not checked.valid:
                raise click.ClickException(f"Data verification failed for {name}")
            provenance[name] = {
                "sha256": records[name].sha256,
                "bytes": records[name].expected_bytes,
            }
    train = load_train(data_root / "train_FD001.txt")
    target_data = {
        target: (
            load_test(data_root / f"test_{target}.txt"),
            load_rul(data_root / f"RUL_{target}.txt"),
        )
        for target in targets
    }
    for seed in seeds:
        results = run_c04_seed(train, target_data, config, seed)
        for target, result in results.items():
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
