"""Run one reproducible C01 C-MAPSS point-baseline evaluation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import click

from calibrated_reliability.data.labels import add_rul_targets
from calibrated_reliability.data.loader import load_rul, load_test, load_train
from calibrated_reliability.data.splitting import split_engine_ids
from calibrated_reliability.evaluation.metrics import evaluate_endpoint_predictions
from calibrated_reliability.features.regime import RegimeAwareScaler
from calibrated_reliability.features.temporal import TemporalFeatureTransformer
from calibrated_reliability.models.baselines import fit_baseline_models, predict_baselines


def _git_sha() -> str:
    """Return the current revision, or an explicit unknown marker."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@click.command()
@click.option("--data-root", type=click.Path(path_type=Path), default=Path("data/raw"))
@click.option("--output", type=click.Path(path_type=Path), default=Path("outputs/c01"))
@click.option("--seed", type=int, default=13, show_default=True)
@click.option("--rul-cap", type=int, default=125, show_default=True)
def main(data_root: Path, output: Path, seed: int, rul_cap: int) -> None:
    """Fit C01 on FD001 train engines and evaluate FD001 test endpoints."""
    train = load_train(data_root / "train_FD001.txt")
    test = load_test(data_root / "test_FD001.txt")
    test_rul = load_rul(data_root / "RUL_FD001.txt")
    labeled_train = add_rul_targets(train, cap=rul_cap)
    partitions = split_engine_ids(train["engine_id"], seed=seed)
    base_ids = partitions["base_train"]
    base = labeled_train[labeled_train["engine_id"].isin(base_ids)].copy()
    raw_base = base[train.columns]
    raw_test = test[train.columns]

    temporal = TemporalFeatureTransformer().fit(raw_base)
    base_features = temporal.transform(raw_base)
    test_features = temporal.transform(raw_test)
    scaler = RegimeAwareScaler(random_state=seed).fit(base_features)
    X_train = scaler.transform(base_features)
    X_test = scaler.transform(test_features)
    models = fit_baseline_models(X_train, base["rul_capped"], random_state=seed)
    predictions = predict_baselines(models, X_test)

    test_engine_ids = test["engine_id"].drop_duplicates().tolist()
    if len(test_engine_ids) != len(test_rul):
        raise ValueError("Test-engine count does not match the RUL file row count")
    endpoint_rows = test_features.loc[test_features.groupby("engine_id")["cycle"].idxmax()][
        ["engine_id"]
    ].reset_index(drop=True)
    endpoint_rows["y_true"] = test_rul["rul"].to_numpy()
    metrics: dict[str, dict[str, float]] = {}
    prediction_frame = endpoint_rows.copy()
    for name, values in predictions.items():
        endpoint_values = (
            test_features.groupby("engine_id")["cycle"]
            .idxmax()
            .map(dict(zip(test_features.index, values, strict=True)))
        )
        prediction_frame[name] = endpoint_values.to_numpy()
        metrics[name] = evaluate_endpoint_predictions(
            prediction_frame.rename(columns={name: "y_pred"})
        )

    output.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(output / f"predictions_seed_{seed}.csv", index=False)
    manifest = {
        "experiment_id": "C01",
        "git_sha": _git_sha(),
        "seed": seed,
        "rul_cap": rul_cap,
        "source": "FD001",
        "target": "FD001",
        "evaluation_unit": "engine_endpoint",
        "base_train_engine_count": len(base_ids),
        "test_engine_count": len(test_engine_ids),
        "models": list(metrics),
    }
    (output / f"manifest_seed_{seed}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / f"metrics_seed_{seed}.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    click.echo(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    sys.exit(main())
