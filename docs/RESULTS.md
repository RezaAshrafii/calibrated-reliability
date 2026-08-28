# Results reporting state

## Publication rule

This file may contain numerical results only when a tracked deterministic
builder reconstructs them from immutable, indexed, independently verified
artifacts. Notebook output, memory, audit prose, and manual transcription are
not reporting paths.

Verified local artifacts exist for C01--C08, but generated outputs are ignored
by Git and multiple superseded trees remain in the workspace. Until Gate D adds
an artifact index, an allowed mixed-SHA policy, and `scripts/build_results.py`,
all numerical result cells remain `PENDING`. `PENDING` must never be rendered as
zero or omitted from a denominator.

## Lifecycle state

| Experiment | Implementation | Execution | Independent verification | Numerical reporting | Scientific role |
|---|---|---|---|---|---|
| C01 | IMPLEMENTED | EXECUTED | VERIFIED | PENDING | replication/context |
| C02 | IMPLEMENTED | EXECUTED | VERIFIED | PENDING | replication/context |
| C03 | IMPLEMENTED | EXECUTED | VERIFIED | PENDING | replication/context |
| C04 | IMPLEMENTED | EXECUTED | VERIFIED | PENDING | replication/context |
| C05 | IMPLEMENTED | EXECUTED | VERIFIED | PENDING | exploratory/appendix |
| C06 | IMPLEMENTED | EXECUTED | VERIFIED | PENDING | exploratory/appendix |
| C07 | IMPLEMENTED | EXECUTED | VERIFIED | PENDING | exploratory/appendix |
| C08 | IMPLEMENTED | EXECUTED | VERIFIED | PENDING | exploratory/appendix; anchor saturation diagnostic |
| C11 | NOT IMPLEMENTED | NOT EXECUTED | NOT VERIFIED | PENDING | candidate primary mechanism audit |
| C12 | NOT IMPLEMENTED | NOT EXECUTED | NOT VERIFIED | PENDING | optional oracle/supporting diagnostic |

The definitions of these terms are normative in `REPRODUCIBILITY.md`.

## Gate D requirements

Before any numerical table is added, the builder must:

1. consume only artifact trees listed as `OFFICIAL` in a tracked artifact index;
2. validate artifact and manifest hashes and reject missing files;
3. use `pandas.read_csv(..., float_precision="round_trip")` for lossless numeric
   reconstruction;
4. retain seed-level values and separate raw from clipped targets/predictions;
5. separate alpha `0.10` and `0.05`;
6. record and allow cross-experiment Git-SHA differences only through an
   explicit provenance field;
7. emit `n_cal`, requested rank, effective rank, attainability, quantile regime,
   policy, and distinct-quantile count for conformal rows where applicable;
8. reproduce metrics and confidence intervals from stored predictions;
9. fail closed on superseded, mixed-status, incomplete, or hash-mismatched
   inputs; and
10. generate deterministic tables without manual edits.

## Claim boundary

C01--C04 are replication and benchmark context. C05--C08 are exploratory or
appendix evidence. C08 motivates the rank-attainability question but does not
show that ACI generally fails under distribution shift. The candidate C11
contribution is an application-level comparison with the exact Beta reference,
not a new conformal theorem. Any C12 result must be labelled oracle-only and
must not be described as a deployable target-domain repair.

## Reproduction

Use `docs/RUNBOOK.md` for all C01--C08 commands. Raw data verification is local
because `data/raw/` is intentionally excluded from Git; CI cannot certify the
presence of official NASA files or local artifacts.

## Audit checklist

- [ ] Official and superseded artifact trees are indexed.
- [ ] Every manifest and artifact hash matches.
- [ ] Producing commits and clean-worktree status are verified.
- [ ] Metrics and confidence intervals are reconstructed from stored rows.
- [ ] Mixed Git SHAs are explicit rather than silently merged.
- [ ] Conformal rank and attainability diagnostics are retained.
- [ ] All rendered values originate from the tracked builder.
- [ ] Negative and null findings remain present.
- [ ] Raw data, secrets, outputs, and personal paths remain untracked.
