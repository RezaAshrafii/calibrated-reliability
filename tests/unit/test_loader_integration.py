"""Integration checks against locally supplied C-MAPSS files."""

from pathlib import Path

import pytest

from calibrated_reliability.data.loader import load_rul, load_test, load_train

RAW = Path(__file__).parents[2] / "data" / "raw"


@pytest.mark.integration
def test_all_official_cmapss_files_load_when_present() -> None:
    """All registered local files satisfy loader contracts."""
    if not (RAW / "train_FD001.txt").exists():
        pytest.skip("raw C-MAPSS data is not present in this checkout")
    for subset, loader in (("train", load_train), ("test", load_test)):
        for index in range(1, 5):
            frame = loader(RAW / f"{subset}_FD{index:03d}.txt")
            assert len(frame.columns) == 26
            assert frame["engine_id"].nunique() > 0
    for index in range(1, 5):
        assert len(load_rul(RAW / f"RUL_FD{index:03d}.txt")) > 0
