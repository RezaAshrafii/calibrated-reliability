# Official artifact release contract

The repository intentionally excludes raw C-MAPSS data and generated artifact
trees from Git. A public companion archive is therefore required for a third
party to reproduce the complete C01--C08 and C11 artifact-level audits.

Build the archive only from a clean checkout with all ignored official artifact
roots available locally:

```bash
uv run python scripts/build_official_artifact_archive.py \
  --output C:/absolute/external/path/calibrated-reliability-official-artifacts.zip
```

The destination must be a new absolute ZIP path outside the repository. The
builder verifies the Gate D official roots and the separately indexed C11
artifact before writing anything. It includes only verified official artifact
roots, the two artifact indices, tracked reports, frozen configurations,
registry metadata, lockfile, and decision records. It rejects symlinks and
never includes `data/raw/`.

`ARCHIVE_MANIFEST.json` inside the ZIP records every included file's relative
path, byte count, SHA-256, current clean builder SHA, C11 producing SHA, and
C11 manifest SHA. Upload this exact immutable ZIP to a versioned GitHub Release
or Zenodo record together with its SHA-256. Do not alter its contents after
publication; create a new versioned archive instead.

The archive preserves artifact and report evidence, not a right to redistribute
NASA raw data. The NASA portal currently lists the C-MAPSS dataset license as
unspecified; recipients must obtain raw data from the NASA source recorded in
`data/registry.yaml` and verify its hashes independently.
