"""Deterministic engine-level partitions and calibration cut points."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pandas as pd

SEEDS = (13, 37, 73, 101, 137)


def split_engine_ids(
    engine_ids: pd.Series | list[int],
    seed: int,
    train_fraction: float = 0.60,
    calibration_fraction: float = 0.20,
) -> dict[str, list[int]]:
    """Split unique engine IDs into disjoint deterministic partitions."""
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not 0 < train_fraction < 1 or not 0 < calibration_fraction < 1:
        raise ValueError("split fractions must be between 0 and 1")
    if train_fraction + calibration_fraction >= 1:
        raise ValueError("train and calibration fractions must leave validation data")
    unique = sorted({int(value) for value in engine_ids})
    if len(unique) < 5:
        raise ValueError("At least 5 engines are required for a three-way split")
    rng = random.Random(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    train_count = max(1, math.floor(len(unique) * train_fraction))
    calibration_count = max(1, math.floor(len(unique) * calibration_fraction))
    if train_count + calibration_count >= len(unique):
        raise ValueError("Fractions leave no validation engines")
    partitions = {
        "base_train": sorted(int(value) for value in shuffled[:train_count]),
        "calibration": sorted(
            int(value) for value in shuffled[train_count : train_count + calibration_count]
        ),
        "validation": sorted(int(value) for value in shuffled[train_count + calibration_count :]),
    }
    if set(partitions["base_train"]) & set(partitions["calibration"]):
        raise AssertionError("base_train and calibration engine IDs overlap")
    if set(partitions["base_train"]) & set(partitions["validation"]):
        raise AssertionError("base_train and validation engine IDs overlap")
    if set(partitions["calibration"]) & set(partitions["validation"]):
        raise AssertionError("calibration and validation engine IDs overlap")
    return partitions


def write_split_manifest(partitions: dict[str, list[int]], seed: int, output: Path) -> None:
    """Write a JSON split manifest with its seed and engine IDs."""
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": seed, "partitions": partitions}
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def generate_cut_points(
    frame: pd.DataFrame,
    calibration_engine_ids: list[int] | set[int],
    seed: int,
    min_observed_cycles: int = 30,
    lower_fraction: float = 0.40,
    upper_fraction: float = 0.90,
) -> dict[int, int]:
    """Generate one deterministic observed-cycle cut point per calibration engine."""
    required = {"engine_id", "cycle"}
    if not required.issubset(frame.columns):
        raise ValueError("Frame must contain engine_id and cycle")
    if min_observed_cycles < 1 or not 0 < lower_fraction <= upper_fraction <= 1:
        raise ValueError("Invalid cut-point constraints")
    rng = random.Random(seed)
    result: dict[int, int] = {}
    for engine_id in sorted(int(value) for value in calibration_engine_ids):
        cycles = frame.loc[frame["engine_id"] == engine_id, "cycle"]
        if cycles.empty:
            raise ValueError(f"Calibration engine {engine_id} is absent from frame")
        max_cycle = int(cycles.max())
        lower = max(min_observed_cycles, math.ceil(max_cycle * lower_fraction))
        upper = math.floor(max_cycle * upper_fraction)
        if lower > upper:
            raise ValueError(f"Engine {engine_id} has no valid cut-point range")
        result[engine_id] = rng.randint(lower, upper)
    return result


def restrict_to_cut_points(frame: pd.DataFrame, cut_points: dict[int, int]) -> pd.DataFrame:
    """Return only observations at or before each engine's cut point."""
    if not {"engine_id", "cycle"}.issubset(frame.columns):
        raise ValueError("Frame must contain engine_id and cycle")
    unknown = set(cut_points).difference(set(frame["engine_id"].unique()))
    if unknown:
        raise ValueError(f"Cut points reference unknown engines: {sorted(unknown)}")
    limits = frame["engine_id"].map(cut_points)
    return frame.loc[frame["cycle"] <= limits].copy()
