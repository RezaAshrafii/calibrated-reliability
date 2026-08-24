"""Command-line entry point for fail-closed data verification."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from calibrated_reliability.data.registry import validate_registry


@click.command()
@click.option("--registry", type=click.Path(path_type=Path, exists=True), required=True)
@click.option(
    "--data-root", type=click.Path(path_type=Path), default=Path("data/raw"), show_default=True
)
def main(registry: Path, data_root: Path) -> None:
    """Verify all registered data files; fail closed on any mismatch."""
    results = validate_registry(registry, data_root)
    failed = False
    for result in results:
        if result.valid:
            click.echo(f"PASS {result.filename}")
        else:
            failed = True
            click.echo(f"FAIL {result.filename}")
            for error in result.errors:
                click.echo(f"  - {error}")
    if failed:
        raise click.exceptions.Exit(1)
    click.echo(f"Verified {len(results)} files.")


if __name__ == "__main__":
    sys.exit(main())
