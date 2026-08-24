"""Tests for RUL targets, engine splits, and calibration cut points."""

import pandas as pd
import pytest

from calibrated_reliability.data.labels import add_rul_targets, validate_terminal_rul
from calibrated_reliability.data.splitting import (
    generate_cut_points,
    restrict_to_cut_points,
    split_engine_ids,
)


def trajectory() -> pd.DataFrame:
    """Create a small multi-engine trajectory for deterministic tests."""
    return pd.DataFrame(
        {"engine_id": [1, 1, 1, 2, 2, 2], "cycle": [1, 2, 3, 1, 2, 3], "sensor_1": 0.0}
    )


def test_rul_cap_and_terminal_zero() -> None:
    """Raw terminal RUL is zero and cap is applied exactly."""
    frame = add_rul_targets(trajectory(), cap=2)
    validate_terminal_rul(frame)
    assert list(frame["rul_raw"]) == [2, 1, 0, 2, 1, 0]
    assert list(frame["rul_capped"]) == [2, 1, 0, 2, 1, 0]


def test_cap_130_sensitivity() -> None:
    """The sensitivity cap is configuration-driven."""
    frame = pd.DataFrame({"engine_id": [1] * 140, "cycle": list(range(1, 141))})
    result = add_rul_targets(frame, cap=130)
    assert int(result["rul_capped"].max()) == 130


def test_split_is_disjoint_and_deterministic() -> None:
    """Same seed gives same partitions and no engine overlaps."""
    ids = list(range(1, 11))
    first = split_engine_ids(ids, seed=13)
    second = split_engine_ids(ids, seed=13)
    assert first == second
    assert set(first["base_train"]).isdisjoint(first["calibration"])
    assert set(first["base_train"]).isdisjoint(first["validation"])
    assert set(first["calibration"]).isdisjoint(first["validation"])


def test_split_requires_enough_engines() -> None:
    """Tiny inputs fail instead of creating an invalid partition."""
    with pytest.raises(ValueError, match="5 engines"):
        split_engine_ids([1, 2, 3, 4], seed=13)


def test_cut_points_are_deterministic_and_bounded() -> None:
    """Cut points obey minimum and fractional trajectory bounds."""
    frame = pd.DataFrame({"engine_id": [1] * 100 + [2] * 100, "cycle": list(range(1, 101)) * 2})
    first = generate_cut_points(frame, [1, 2], seed=13)
    second = generate_cut_points(frame, [1, 2], seed=13)
    assert first == second
    assert all(40 <= point <= 90 for point in first.values())


def test_cut_point_restricts_future_cycles() -> None:
    """No row after a cut point remains in the restricted frame."""
    frame = trajectory()
    restricted = restrict_to_cut_points(frame, {1: 2, 2: 2})
    assert restricted["cycle"].max() == 2
    assert len(restricted) == 4


def test_cut_point_minimum_cycles_guard() -> None:
    """Short trajectories fail when no valid minimum-observation cut exists."""
    with pytest.raises(ValueError, match="no valid cut-point"):
        generate_cut_points(trajectory(), [1], seed=13, min_observed_cycles=30)
