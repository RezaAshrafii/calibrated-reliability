"""Build exploratory C-MAPSS sensor plots from training data only."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from calibrated_reliability.data.loader import load_train


def build_sensor_eda(data_root: Path, output: Path) -> None:
    """Plot selected sensors for the first FD001 engine."""
    frame = load_train(data_root / "train_FD001.txt")
    engine = frame[frame["engine_id"] == frame["engine_id"].min()]
    sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_7"]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, sensor in zip(axes.flat, sensors, strict=True):
        axis.plot(engine["cycle"], engine[sensor])
        axis.set(title=sensor, xlabel="cycle", ylabel="value")
    figure.suptitle("Exploratory only — not for inference")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    build_sensor_eda(Path("data/raw"), Path("reports/figures/sensor_eda.png"))
