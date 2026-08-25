"""Generate immutable C03 conformalized quantile regression artifacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import click

from calibrated_reliability.data.loader import load_rul, load_test, load_train
from calibrated_reliability.data.registry import load_registry, validate_file
from calibrated_reliability.experiments.artifacts import write_c03_run
from calibrated_reliability.experiments.c03 import C03Config, run_c03

REQUIRED_DATA_FILES = ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")


def _clean_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
            raise RuntimeError("C03 refuses to run from a dirty Git worktree")
        return sha
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("C03 must run inside a Git repository") from exc


def _provenance(registry: Path, data_root: Path) -> dict[str, dict[str, Any]]:
    records = {record.filename: record for record in load_registry(registry)}
    missing = set(REQUIRED_DATA_FILES) - set(records)
    if missing:
        raise ValueError(f"Registry is missing C03 files: {sorted(missing)}")
    result: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_DATA_FILES:
        record = records[name]
        checked = validate_file(data_root, record)
        if not checked.valid:
            raise ValueError(f"Data verification failed for {name}: {list(checked.errors)}")
        result[name] = {
            "sha256": record.sha256,
            "bytes": record.expected_bytes,
            "rows": record.expected_rows,
            "engines": record.expected_engines,
        }
    return result


@click.command()
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option(
    "--registry", "registry_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--data-root", type=click.Path(path_type=Path), default=Path("data/raw"))
@click.option("--output-root", type=click.Path(path_type=Path), default=Path("outputs/c03"))
@click.option("--seed", "requested_seeds", type=int, multiple=True)
def main(
    config_path: Path,
    registry_path: Path,
    data_root: Path,
    output_root: Path,
    requested_seeds: tuple[int, ...],
) -> None:
    """Run declared C03 seeds from a clean committed worktree."""
    sha = _clean_sha()
    config = C03Config.from_yaml(config_path.read_text(encoding="utf-8"))
    seeds = requested_seeds or config.seeds
    if set(seeds) - set(config.seeds):
        raise click.ClickException("Undeclared C03 seed requested")
    provenance = _provenance(registry_path, data_root)
    train = load_train(data_root / "train_FD001.txt")
    test = load_test(data_root / "test_FD001.txt")
    test_rul = load_rul(data_root / "RUL_FD001.txt")
    for seed in seeds:
        result = run_c03(train, test, test_rul, config, seed)
        try:
            run_dir = write_c03_run(
                output_root, seed, sha, config, result, config_path, registry_path, provenance
            )
        except FileExistsError as exc:
            raise click.ClickException(
                f"Immutable C03 run directory already exists for seed {seed}"
            ) from exc
        click.echo(f"PASS seed={seed} artifact={run_dir}")


if __name__ == "__main__":
    sys.exit(main())
