"""Package existing verified artifacts for a release without copying raw data."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from calibrated_reliability.reporting.release import build_official_artifact_archive


@click.command()
@click.option("--output", type=click.Path(path_type=Path), required=True)
def main(output: Path) -> None:
    """Write one immutable official-artifact ZIP outside the repository."""
    destination = build_official_artifact_archive(Path.cwd(), output)
    click.echo(f"PASS archive={destination}")


if __name__ == "__main__":
    sys.exit(main())
