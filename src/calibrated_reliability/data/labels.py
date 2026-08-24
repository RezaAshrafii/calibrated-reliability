"""RUL target construction with explicit cap validation."""

from __future__ import annotations

import pandas as pd


def add_rul_targets(frame: pd.DataFrame, cap: int | None = 125) -> pd.DataFrame:
    """Add raw and optionally capped RUL targets to a trajectory frame.

    RUL is computed from each engine's final observed training cycle. The input
    frame is not modified in place.
    """
    required = {"engine_id", "cycle"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if cap is not None and cap <= 0:
        raise ValueError("RUL cap must be positive or None")
    result = frame.copy()
    last_cycle = result.groupby("engine_id")["cycle"].transform("max")
    result["rul_raw"] = (last_cycle - result["cycle"]).astype("int64")
    if (result["rul_raw"] < 0).any():
        raise ValueError("RUL cannot be negative")
    if cap is not None:
        result["rul_capped"] = result["rul_raw"].clip(upper=cap).astype("int64")
    return result


def validate_terminal_rul(frame: pd.DataFrame) -> None:
    """Raise if every engine's final observed cycle does not have raw RUL zero."""
    if "rul_raw" not in frame.columns:
        raise ValueError("Frame must contain rul_raw")
    terminal = frame.loc[frame.groupby("engine_id")["cycle"].idxmax(), "rul_raw"]
    if not (terminal == 0).all():
        raise ValueError("Final observed cycle for every engine must have rul_raw=0")
