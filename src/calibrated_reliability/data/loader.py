"""Schema-validated loaders for the NASA C-MAPSS data format."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

COLUMNS = [
    "engine_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    *[f"sensor_{index}" for index in range(1, 22)],
]
INTEGER_COLUMNS = ["engine_id", "cycle"]
FLOAT_COLUMNS = COLUMNS[2:]


def _read_trajectory(path: Path) -> pd.DataFrame:
    """Read and validate one space-separated C-MAPSS trajectory file."""
    if not path.is_file():
        raise FileNotFoundError(f"C-MAPSS file not found: {path}")
    try:
        frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    except Exception as exc:
        raise ValueError(f"Could not parse C-MAPSS file: {path}") from exc
    if frame.shape[1] != len(COLUMNS):
        raise ValueError(f"{path.name}: expected 26 columns, got {frame.shape[1]}")
    frame.columns = COLUMNS
    for column in INTEGER_COLUMNS:
        numeric = pd.to_numeric(frame[column], errors="raise")
        if not numeric.map(
            lambda value: math.isfinite(float(value)) and float(value).is_integer()
        ).all():
            raise ValueError(f"{path.name}: {column} values must be finite integers")
        frame[column] = numeric.astype("int64")
        if (frame[column] <= 0).any():
            raise ValueError(f"{path.name}: {column} values must be positive")
    for column in FLOAT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
        if not frame[column].map(math.isfinite).all():
            raise ValueError(f"{path.name}: {column} values must be finite")
    if frame.isna().any().any():
        raise ValueError(f"{path.name}: missing values are not allowed")
    if frame.duplicated(["engine_id", "cycle"]).any():
        raise ValueError(f"{path.name}: duplicate (engine_id, cycle) pair")
    if (
        not frame.groupby("engine_id", sort=False)["cycle"]
        .apply(lambda values: values.is_monotonic_increasing)
        .all()
    ):
        raise ValueError(f"{path.name}: cycles must increase within each engine")
    frame = frame.sort_values(["engine_id", "cycle"], kind="stable").reset_index(drop=True)
    return frame


def load_train(path: Path) -> pd.DataFrame:
    """Load a validated C-MAPSS training trajectory."""
    return _read_trajectory(path)


def load_test(path: Path) -> pd.DataFrame:
    """Load a validated C-MAPSS test trajectory."""
    return _read_trajectory(path)


def load_rul(path: Path) -> pd.DataFrame:
    """Load a one-column C-MAPSS RUL file as an integer DataFrame."""
    if not path.is_file():
        raise FileNotFoundError(f"RUL file not found: {path}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    parsed: list[int] = []
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 1:
            raise ValueError(f"{path.name}: line {line_number} must contain exactly one RUL value")
        try:
            value = float(fields[0])
        except ValueError as exc:
            raise ValueError(f"{path.name}: invalid RUL on line {line_number}") from exc
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{path.name}: RUL values must be finite integers")
        parsed.append(int(value))
    if not parsed:
        raise ValueError(f"{path.name}: RUL file is empty")
    values = pd.DataFrame({"rul": pd.Series(parsed, dtype="int64")})
    if (values["rul"] < 0).any():
        raise ValueError(f"{path.name}: RUL values cannot be negative")
    return values


def summarize_trajectory(frame: pd.DataFrame, filename: str) -> dict[str, object]:
    """Create a JSON-serializable summary for a loaded trajectory."""
    sensor_columns = [column for column in FLOAT_COLUMNS if column.startswith("sensor_")]
    constant_sensors = [
        column for column in sensor_columns if frame[column].nunique(dropna=False) <= 1
    ]
    return {
        "filename": filename,
        "rows": int(len(frame)),
        "engines": int(frame["engine_id"].nunique()),
        "cycle_range": [int(frame["cycle"].min()), int(frame["cycle"].max())],
        "constant_sensors": constant_sensors,
        "missing_count": int(frame.isna().sum().sum()),
    }


def write_summary(frame: pd.DataFrame, filename: str, output: Path) -> None:
    """Write a trajectory summary as indented JSON."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summarize_trajectory(frame, filename), indent=2) + "\n", encoding="utf-8"
    )
