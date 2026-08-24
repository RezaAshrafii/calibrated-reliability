"""Tests for the C-MAPSS loader and schema guards."""

from pathlib import Path

import pytest

from calibrated_reliability.data.loader import (
    COLUMNS,
    load_rul,
    load_test,
    load_train,
    summarize_trajectory,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic_fd001.txt"


def test_loader_has_26_columns_and_expected_dtypes() -> None:
    """The fixture loads with the official schema and explicit dtypes."""
    frame = load_train(FIXTURE)
    assert len(COLUMNS) == 26
    assert list(frame.columns) == COLUMNS
    assert frame["engine_id"].dtype == "int64"
    assert frame["sensor_1"].dtype == "float64"


def test_cycles_are_sorted_within_engine() -> None:
    """Rows are ordered by engine and cycle."""
    frame = load_train(FIXTURE)
    assert frame[["engine_id", "cycle"]].equals(
        frame[["engine_id", "cycle"]].sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    )


def test_duplicate_detection(tmp_path: Path) -> None:
    """Duplicate engine-cycle keys fail closed."""
    target = tmp_path / "duplicate.txt"
    target.write_text(
        FIXTURE.read_text(encoding="utf-8")
        + FIXTURE.read_text(encoding="utf-8").splitlines()[0]
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_train(target)


def test_malformed_line_detection(tmp_path: Path) -> None:
    """Rows with the wrong number of fields fail schema validation."""
    target = tmp_path / "malformed.txt"
    target.write_text("1 1 0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="26 columns"):
        load_train(target)


@pytest.mark.parametrize("replacement", ["1.5 1 ", "1 -1 "])
def test_invalid_integer_identifiers_fail_closed(tmp_path: Path, replacement: str) -> None:
    """Fractional and non-positive identifiers cannot be silently coerced."""
    target = tmp_path / "invalid_ids.txt"
    target.write_text(
        FIXTURE.read_text(encoding="utf-8").replace("1 1 ", replacement, 1), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="engine_id|cycle|positive|integers"):
        load_train(target)


def test_fractional_cycle_fails_closed(tmp_path: Path) -> None:
    """Fractional cycle values cannot be silently coerced."""
    target = tmp_path / "fractional_cycle.txt"
    target.write_text(
        FIXTURE.read_text(encoding="utf-8").replace("1 1 ", "1 1.5 ", 1), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="cycle.*integers"):
        load_train(target)


def test_malformed_rul_schema_fails_closed(tmp_path: Path) -> None:
    """RUL input must be exactly one finite integer per line."""
    target = tmp_path / "malformed_rul.txt"
    target.write_text("12 99\n12.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one RUL"):
        load_rul(target)


def test_fractional_rul_fails_closed(tmp_path: Path) -> None:
    """Fractional RUL values cannot be silently coerced."""
    target = tmp_path / "fractional_rul.txt"
    target.write_text("12.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="finite integers"):
        load_rul(target)


def test_empty_rul_fails_closed(tmp_path: Path) -> None:
    """An empty RUL file is invalid."""
    target = tmp_path / "empty_rul.txt"
    target.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_rul(target)


def test_infinite_sensor_fails_closed(tmp_path: Path) -> None:
    """Infinite sensor values are rejected."""
    target = tmp_path / "infinite_sensor.txt"
    target.write_text(
        FIXTURE.read_text(encoding="utf-8").replace(" 1 2 3 ", " inf 2 3 ", 1), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="finite"):
        load_train(target)


def test_train_test_and_rul_loaders_are_separate(tmp_path: Path) -> None:
    """The three public loaders expose their intended input contracts."""
    trajectory = tmp_path / "trajectory.txt"
    trajectory.write_bytes(FIXTURE.read_bytes())
    rul = tmp_path / "RUL.txt"
    rul.write_text("12\n7\n", encoding="utf-8")
    assert len(load_train(trajectory)) == 4
    assert len(load_test(trajectory)) == 4
    assert list(load_rul(rul)["rul"]) == [12, 7]


def test_summary_is_json_serializable() -> None:
    """Summary contains stable machine-readable fields."""
    frame = load_train(FIXTURE)
    summary = summarize_trajectory(frame, FIXTURE.name)
    assert summary["rows"] == 4
    assert summary["engines"] == 2
    assert summary["missing_count"] == 0
