# C11 deterministic reporting contract

## Status

`PUBLISHED / VERIFIED / REPORTED`

The immutable report was published in commit
`aedbb67dd223d102791ec36a8823c2cff6a0521b` from clean builder commit
`66180029e55a2b05b3b9495ed87a50318038d712`. The artifact index status
`VERIFIED_CANDIDATE` identifies the single artifact selected for this isolated
reporting path; it is not the experiment lifecycle status.

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
failure. Publication alone did not establish verification: the subsequent
independent report-level audit reconstructed the retained values before C11 was
assigned `VERIFIED` and `REPORTED`.

The builder publishes the verified retained C11 tables; it is not an
independent statistical reconstruction engine. The completed report-level audit
independently recomputed finite-reservoir endpoint metrics, both reference
distributions, all discrepancies, and the observation-mechanism distance from
the retained lower-level artifact tables within the frozen numerical
tolerances. It also reproduced all five report files byte-for-byte from clean
builder commit `66180029e55a2b05b3b9495ed87a50318038d712`.
