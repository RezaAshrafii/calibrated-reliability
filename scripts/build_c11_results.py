"""Build deterministic C11 report tables without executing C11."""

from pathlib import Path

import click

from calibrated_reliability.reporting.c11_results import build_c11_results


@click.command()
@click.option(
    "--index",
    "index_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("docs/c11_artifact_index.yaml"),
    show_default=True,
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("reports/c11"),
    show_default=True,
)
def main(index_path: Path, output_root: Path) -> None:
    """Validate the frozen C11 artifact and publish an immutable report."""
    repository = Path(__file__).resolve().parents[1]
    destination = build_c11_results(repository, index_path, output_root)
    click.echo(f"Built deterministic C11 report at {destination}")


if __name__ == "__main__":
    main()
