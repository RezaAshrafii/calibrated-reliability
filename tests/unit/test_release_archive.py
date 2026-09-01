"""Behavioral tests for the external official-artifact release archive."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from calibrated_reliability.reporting import release


def test_archive_rejects_a_relative_destination(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        release.build_official_artifact_archive(tmp_path, Path("archive.zip"))


def test_archive_rejects_a_destination_inside_the_repository(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the repository"):
        release.build_official_artifact_archive(tmp_path, tmp_path / "archive.zip")


def test_archive_uses_verified_inputs_and_has_deterministic_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "docs").mkdir()
    (repository / "outputs" / "c01" / "run").mkdir(parents=True)
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
        "outputs/c01/run/manifest.json": "{}\n",
        "outputs/c01/run/predictions.csv": "prediction\n1\n",
        "outputs/c01/run/undeclared-secret.txt": "must not be released\n",
        "outputs/c11/run/manifest.json": "{}\n",
    }.items():
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")

    class Run:
        class Entry:
            path = "outputs/c01"

        entry = Entry()
        run_dir = repository / "outputs/c01/run"
        manifest = {"artifacts": {"predictions.csv": "unused-in-mocked-verifier"}}

    monkeypatch.setattr(release, "_clean_git_sha", lambda _: "a" * 40)
    monkeypatch.setattr(release, "load_artifact_index", lambda _: object())
    monkeypatch.setattr(release, "verify_indexed_artifacts", lambda *_: (Run(),))
    monkeypatch.setattr(
        release,
        "verify_c11_artifact",
        lambda *_: ({"git": {"sha": "b" * 40}}, repository / "outputs/c11/run"),
    )
    tracked_metadata = tuple(
        path
        for path in repository.rglob("*")
        if path.is_file() and not path.relative_to(repository).as_posix().startswith("outputs/")
    )
    monkeypatch.setattr(release, "_tracked_metadata_files", lambda _: tracked_metadata)

    destination = tmp_path / "official.zip"
    archive_path = release.build_official_artifact_archive(repository, destination)
    assert archive_path == destination
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert "data/raw/train_FD001.txt" not in names
        assert "outputs/c01/run/undeclared-secret.txt" not in names
        assert names[:-1] == sorted(names[:-1])
        assert names[-1] == "ARCHIVE_MANIFEST.json"
        manifest = json.loads(archive.read("ARCHIVE_MANIFEST.json"))
        assert manifest["contains_raw_cmapss_data"] is False
        assert manifest["builder_git_sha"] == "a" * 40
        assert manifest["c11_producing_git_sha"] == "b" * 40
        assert any(item["path"] == "outputs/c11/run/manifest.json" for item in manifest["files"])

    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert len(digest) == 64


def test_archive_revalidates_inputs_and_cleans_failed_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    source = repository / "manifest.json"
    source.write_text("{}\n", encoding="utf-8")
    inputs = release.ReleaseInputs(
        files=(source,),
        gate_d_official_roots=(repository / "outputs/c01",),
        c11_artifact_root=repository / "outputs/c11",
        c11_manifest_sha256="b" * 64,
        c11_producing_git_sha="c" * 40,
    )
    calls = 0

    def changed_inputs(_: Path) -> release.ReleaseInputs:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("Artifact archive inputs changed")
        return inputs

    monkeypatch.setattr(release, "_clean_git_sha", lambda _: "a" * 40)
    monkeypatch.setattr(release, "_verified_release_inputs", changed_inputs)
    destination = tmp_path / "official.zip"

    with pytest.raises(ValueError, match="inputs changed"):
        release.build_official_artifact_archive(repository, destination)

    assert calls == 2
    assert not destination.exists()
    assert not list(tmp_path.glob(".official.*"))


def test_archive_rejects_existing_destination(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    destination = tmp_path / "official.zip"
    destination.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        release.build_official_artifact_archive(repository, destination)


def test_archive_rechecks_git_immediately_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    source = repository / "manifest.json"
    source.write_text("{}\n", encoding="utf-8")
    inputs = release.ReleaseInputs(
        files=(source,),
        gate_d_official_roots=(repository / "outputs/c01",),
        c11_artifact_root=repository / "outputs/c11",
        c11_manifest_sha256="b" * 64,
        c11_producing_git_sha="c" * 40,
    )
    git_states = iter(("a" * 40, "d" * 40))
    monkeypatch.setattr(release, "_clean_git_sha", lambda _: next(git_states))
    monkeypatch.setattr(release, "_verified_release_inputs", lambda _: inputs)
    destination = tmp_path / "official.zip"

    with pytest.raises(ValueError, match="Git state changed"):
        release.build_official_artifact_archive(repository, destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".official.*"))


def test_completed_archive_rejects_tampered_payload(tmp_path: Path) -> None:
    archive_path = tmp_path / "tampered.zip"
    expected = hashlib.sha256(b"expected").hexdigest()
    manifest = {
        "files": [{"path": "artifact.bin", "bytes": 8, "sha256": expected}],
    }
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("artifact.bin", b"tampered")
        archive.writestr(
            "ARCHIVE_MANIFEST.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )

    with pytest.raises(ValueError, match="hash mismatch"):
        release._verify_completed_archive(archive_path, manifest)


def test_archive_input_rejects_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("content\n", encoding="utf-8")
    linked = tmp_path / "linked.txt"
    try:
        linked.symlink_to(source)
    except OSError:
        pytest.skip("symlink creation is unavailable on this host")

    with pytest.raises(ValueError, match="symlink"):
        release._regular_files(tmp_path, (linked,))
