"""Generate immutable C02 split-conformal artifacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import click

from calibrated_reliability.data.loader import load_rul, load_test, load_train
from calibrated_reliability.data.registry import load_registry, validate_file
from calibrated_reliability.experiments.artifacts import write_c02_run
from calibrated_reliability.experiments.c02 import C02Config, run_c02

REQUIRED_DATA_FILES = ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")


def _git_sha_clean() -> str:
    """Return the current commit and reject dirty or unknown Git state."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("C02 must run inside a Git repository") from exc
    if dirty:
        raise RuntimeError("C02 refuses to run from a dirty Git worktree")
    return sha


def _data_provenance(registry_path: Path, data_root: Path) -> dict[str, dict[str, Any]]:
    """Validate and record the exact C02 input files."""
    records = {record.filename: record for record in load_registry(registry_path)}
    missing = set(REQUIRED_DATA_FILES).difference(records)
    if missing:
        raise ValueError(f"Registry is missing C02 files: {sorted(missing)}")
    provenance: dict[str, dict[str, Any]] = {}
    for filename in REQUIRED_DATA_FILES:
        record = records[filename]
        result = validate_file(data_root, record)
        if not result.valid:
            raise ValueError(f"Data verification failed for {filename}: {list(result.errors)}")
        provenance[filename] = {
            "sha256": record.sha256,
            "bytes": record.expected_bytes,
            "rows": record.expected_rows,
            "engines": record.expected_engines,
        }
    return provenance


@click.command()
@click.option(
    "--config", "config_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option(
    "--registry", "registry_path", type=click.Path(path_type=Path, exists=True), required=True
)
@click.option("--data-root", type=click.Path(path_type=Path), default=Path("data/raw"))
@click.option("--output-root", type=click.Path(path_type=Path), default=Path("outputs/c02"))
@click.option("--seed", "requested_seeds", type=int, multiple=True)
def main(
    config_path: Path,
    registry_path: Path,
    data_root: Path,
    output_root: Path,
    requested_seeds: tuple[int, ...],
) -> None:
    """Run declared C02 seeds from a clean committed worktree."""
    sha = _git_sha_clean()
    config = C02Config.from_yaml(config_path.read_text(encoding="utf-8"))
    seeds = requested_seeds or config.seeds
    undeclared = set(seeds).difference(config.seeds)
    if undeclared:
        raise click.ClickException(f"Undeclared C02 seeds: {sorted(undeclared)}")
    provenance = _data_provenance(registry_path, data_root)
    train = load_train(data_root / "train_FD001.txt")
    test = load_test(data_root / "test_FD001.txt")
    test_rul = load_rul(data_root / "RUL_FD001.txt")
    for seed in seeds:
        result = run_c02(train, test, test_rul, config, seed)
        try:
            run_dir = write_c02_run(
                output_root, seed, sha, config, result, config_path, registry_path, provenance
            )
        except FileExistsError as exc:
            raise click.ClickException(
                f"Immutable C02 run directory already exists for seed {seed}"
            ) from exc
        click.echo(f"PASS seed={seed} artifact={run_dir}")


if __name__ == "__main__":
    sys.exit(main())
