# Reproducibility

The repository is designed so data, configuration, code revision, environment,
split manifests, predictions, and metrics can be traced together.

## Status vocabulary

- `PLANNED`: a reviewed design exists, but its production implementation does
  not.
- `IMPLEMENTED`: production code, frozen configuration, and focused tests exist.
- `EXECUTED`: every declared target/seed/condition run exists as an immutable
  local artifact set.
- `VERIFIED`: artifact hashes and provenance match, metrics are independently
  reconstructed from stored predictions, and the producing Git revision was
  clean.
- `REPORTED`: a tracked deterministic builder has rendered approved values from
  verified artifacts. File presence or manual transcription never establishes
  this status.

C01--C08 are `IMPLEMENTED`, `EXECUTED`, `VERIFIED`, and `REPORTED`. Gate D uses
`docs/artifact_index.yaml` and `scripts/build_results.py` as the only allowed
numerical reporting path. C11 and C12 are not implemented and are not eligible
for reporting.

## Reproduction contract

1. Install Python 3.11+ and `uv`.
2. Run `uv sync --locked --extra dev`.
3. Place officially sourced C-MAPSS files under `data/raw/`; never commit them.
4. Verify registered files with `uv run python scripts/verify_data.py
   --registry data/registry.yaml --data-root data/raw`.
5. Run every quality command in `AGENTS.md`.
6. Commit source changes and require an empty `git status --short` before an
   experiment.
7. Use only the frozen C01--C08 commands in `docs/RUNBOOK.md`.
8. Preserve each run directory unchanged. Never replace, merge, or edit an
   existing artifact tree.
9. Verify manifest hashes, producing commit, configuration, data provenance,
   splits, predictions, and recomputed metrics before assigning `VERIFIED`.
10. Use only the Gate D tables under `reports/results/` for C01--C08 numerical
    reporting; do not transcribe values from notebooks, audit prose, or memory.
11. Treat the tracked manifest-set digests and exact official run contracts in
    `docs/artifact_index.yaml` as the Gate D artifact-identity trust anchor.
12. For byte-identical report reconstruction, use the clean builder revision
    recorded in `provenance.json` and verify `checksums.sha256`. A later builder
    revision must identify itself and therefore changes provenance even when all
    numerical CSV bytes are unchanged.

Raw data and generated outputs are intentionally excluded from Git. Their
absence from a clean clone means local data and artifact verification is not a
CI claim. Reproducibility depends on the registered hashes, immutable manifests,
and a durable external archive of the final official artifact trees.
