"""Generate immutable, provenance-complete C01 artifacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import click

from calibrated_reliability.data.loader import load_rul, load_test, load_train
from calibrated_reliability.data.registry import load_registry, validate_file
from calibrated_reliability.experiments.artifacts import write_c01_run
from calibrated_reliability.experiments.c01 import C01Config, run_c01

REQUIRED_DATA_FILES = ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")


def _git_state() -> tuple[str, bool]:
    """Return revision and cleanliness, rejecting unverifiable Git state."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("C01 must run inside a Git repository") from exc
    if dirty:
        raise RuntimeError("C01 refuses to run from a dirty Git worktree")
    return sha, False


def _validated_data_records(registry_path: Path, data_root: Path) -> dict[str, dict[str, Any]]:
    """Verify and return provenance for the exact C01 inputs."""
    records = {record.filename: record for record in load_registry(registry_path)}
    missing = set(REQUIRED_DATA_FILES).difference(records)
    if missing:
        raise ValueError(f"Registry is missing C01 files: {sorted(missing)}")
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
@click.option("--output-root", type=click.Path(path_type=Path), default=Path("outputs/c01"))
@click.option("--seed", "requested_seeds", type=int, multiple=True)
def main(
    config_path: Path,
    registry_path: Path,
    data_root: Path,
    output_root: Path,
    requested_seeds: tuple[int, ...],
) -> None:
    """Run declared C01 seeds from a clean committed worktree."""
    sha, _ = _git_state()
    config = C01Config.from_yaml(config_path.read_text(encoding="utf-8"))
    seeds = requested_seeds or config.seeds
    undeclared = set(seeds).difference(config.seeds)
    if undeclared:
        raise click.ClickException(f"Undeclared C01 seeds: {sorted(undeclared)}")
    data_provenance = _validated_data_records(registry_path, data_root)
    train = load_train(data_root / "train_FD001.txt")
    test = load_test(data_root / "test_FD001.txt")
    test_rul = load_rul(data_root / "RUL_FD001.txt")
    for seed in seeds:
        result = run_c01(train, test, test_rul, config=config, seed=seed)
        try:
            run_dir = write_c01_run(
                output_root,
                seed,
                sha,
                config,
                result,
                config_path,
                registry_path,
                data_provenance,
            )
        except FileExistsError as exc:
            raise click.ClickException(
                f"Immutable run directory already exists for seed {seed}"
            ) from exc
        click.echo(f"PASS seed={seed} artifact={run_dir}")


if __name__ == "__main__":
    sys.exit(main())
