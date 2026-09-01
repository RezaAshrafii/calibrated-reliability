"""Behavioral tests for the external official-artifact release archive."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from calibrated_reliability.reporting import release


def test_archive_rejects_a_destination_inside_the_repository(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the repository"):
        release.build_official_artifact_archive(tmp_path, tmp_path / "archive.zip")


def test_archive_uses_verified_inputs_and_has_deterministic_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "docs").mkdir()
    (repository / "outputs" / "c01").mkdir(parents=True)
    (repository / "outputs" / "c11" / "run").mkdir(parents=True)
    for relative, content in {
        ".python-version": "3.11.9\n",
        "uv.lock": "lock\n",
        "data/registry.yaml": "version: 1\nfiles: []\n",
        "docs/artifact_index.yaml": "index\n",
        "docs/c11_artifact_index.yaml": "c11\n",
        "reports/results/summary.csv": "metric,value\nx,1\n",
        "reports/c11/cells.csv": "cell,status\nx,evaluated\n",
        "configs/cmapss/example.yaml": "example: true\n",
        "docs/decisions/ADR-0001.md": "decision\n",
        "outputs/c01/manifest.json": "{}\n",
        "outputs/c11/run/manifest.json": "{}\n",
    }.items():
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")

    class Run:
        class Entry:
            path = "outputs/c01"

        entry = Entry()

    monkeypatch.setattr(release, "_clean_git_sha", lambda _: "a" * 40)
    monkeypatch.setattr(release, "load_artifact_index", lambda _: object())
    monkeypatch.setattr(release, "verify_indexed_artifacts", lambda *_: (Run(),))
    monkeypatch.setattr(
        release,
        "verify_c11_artifact",
        lambda *_: ({"git": {"sha": "b" * 40}}, repository / "outputs/c11/run"),
    )

    destination = tmp_path / "official.zip"
    archive_path = release.build_official_artifact_archive(repository, destination)
    assert archive_path == destination
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert "data/raw/train_FD001.txt" not in names
        assert names[:-1] == sorted(names[:-1])
        assert names[-1] == "ARCHIVE_MANIFEST.json"
        manifest = json.loads(archive.read("ARCHIVE_MANIFEST.json"))
        assert manifest["contains_raw_cmapss_data"] is False
        assert manifest["builder_git_sha"] == "a" * 40
        assert manifest["c11_producing_git_sha"] == "b" * 40
        assert any(item["path"] == "outputs/c11/run/manifest.json" for item in manifest["files"])

    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert len(digest) == 64
