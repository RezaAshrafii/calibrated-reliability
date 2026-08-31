# C11 deterministic reporting contract

## Status

`IMPLEMENTED REPORTING PATH / NOT YET PUBLISHED / NOT YET REPORTED`

The C11 reporting path is deliberately separate from the historical C01--C08
Gate D builder. It consumes only the single immutable run identified by
`docs/c11_artifact_index.yaml`; it never fits a model, reads raw C-MAPSS data,
or executes C11.

Before publication the builder verifies the tracked manifest trust anchor, the
producer revision and clean state, the exact run-file set, every declared
artifact hash, the ten declared-cell rows, the twelve evaluated reference rows,
the four explicit exclusion rows, and the retained observation diagnostic. CSV
inputs use `float_precision="round_trip"`.

The immutable report contains:

- `cells.csv`, preserving all evaluated and excluded cells;
- `discrepancies.csv`, preserving both references separately;
- `observation_summary.csv`, preserving the label-free scalar diagnostic and
  its counts;
- `provenance.json`, retaining the source manifest, artifact hashes, producer
  revision, builder revision, and report hashes;
- `checksums.sha256`, externally anchoring the four report files above.

The builder requires a clean worktree, derives its own Git revision, validates
the artifact both before construction and immediately before atomic
publication, refuses overwrite, and cleans its temporary directory after any
failure. Publishing these files does not by itself authorize a scientific
claim. C11 becomes `REPORTED` only after an independent report-level audit.

The builder publishes the verified retained C11 tables; it is not an
independent statistical reconstruction engine. The mandatory report-level
audit must recompute the finite-reservoir summaries and all discrepancies from
the retained lower-level artifact tables before C11 can become `REPORTED`.
