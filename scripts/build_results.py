"""Build deterministic tracked C01--C08 result tables from verified artifacts."""

from pathlib import Path

import click

from calibrated_reliability.reporting.results import build_results


@click.command()
@click.option(
    "--index",
    "index_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("docs/artifact_index.yaml"),
    show_default=True,
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("reports/results"),
    show_default=True,
)
def main(index_path: Path, output_root: Path) -> None:
    """Validate the tracked index and publish immutable result tables."""
    repository_root = Path(__file__).resolve().parents[1]
    destination = build_results(repository_root, index_path, output_root)
    click.echo(f"Built deterministic reports at {destination}")


if __name__ == "__main__":
    main()
