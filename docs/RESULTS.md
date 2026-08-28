# Deterministic results reporting

## Publication rule

The only reportable C01--C08 values are the tracked outputs of
`scripts/build_results.py`. The builder consumes only trees marked `OFFICIAL` in
`docs/artifact_index.yaml`, validates every declared artifact hash, parses CSV
files with `pandas.read_csv(..., float_precision="round_trip")`, reconstructs
stored metrics and bootstrap intervals, and publishes atomically without
overwriting an existing destination.

No research metric is manually transcribed into this Markdown file.

## Tracked result tables

- `reports/results/artifact_runs.csv`: one provenance row per official run.
- `reports/results/point_metrics_by_seed.csv`: raw and clipped point metrics,
  retained separately at seed level.
- `reports/results/interval_metrics_by_seed.csv`: seed-level interval metrics,
  bootstrap intervals, calibration size, rank attainability, quantile policy,
  regime, and realized quantile-resolution diagnostics.
- `reports/results/summary.csv`: deterministic across-seed aggregates with all
  source run IDs, producing Git SHAs, artifact roots, and manifest hashes.
- `reports/results/provenance.json`: artifact-index hash, builder revision,
  mixed-SHA policy, official roots, source revisions, and report-file hashes.

The classical integer order-statistic rank is not applicable to C05's weighted
pointwise threshold. Those rank fields are explicitly `PENDING` and labelled
`not_applicable_to_weighted_quantile`; they are never converted to zero. A raw
NASA score that is numerically non-finite outside the clipped target support is
also retained as `PENDING` with status `nonfinite_on_raw_support`. If any seed in
an aggregate is `PENDING`, the aggregate remains `PENDING`; no denominator is
silently reduced.

## Lifecycle state

| Experiment | Implementation | Execution | Independent verification | Numerical reporting | Scientific role |
|---|---|---|---|---|---|
| C01 | IMPLEMENTED | EXECUTED | VERIFIED | REPORTED | replication/context |
| C02 | IMPLEMENTED | EXECUTED | VERIFIED | REPORTED | replication/context |
| C03 | IMPLEMENTED | EXECUTED | VERIFIED | REPORTED | replication/context |
| C04 | IMPLEMENTED | EXECUTED | VERIFIED | REPORTED | replication/context |
| C05 | IMPLEMENTED | EXECUTED | VERIFIED | REPORTED | exploratory/appendix |
| C06 | IMPLEMENTED | EXECUTED | VERIFIED | REPORTED | exploratory/appendix |
| C07 | IMPLEMENTED | EXECUTED | VERIFIED | REPORTED | exploratory/appendix |
| C08 | IMPLEMENTED | EXECUTED | VERIFIED | REPORTED | exploratory/appendix; anchor saturation diagnostic |
| C11 | NOT IMPLEMENTED | NOT EXECUTED | NOT VERIFIED | NOT ELIGIBLE | candidate primary mechanism audit |
| C12 | NOT IMPLEMENTED | NOT EXECUTED | NOT VERIFIED | NOT ELIGIBLE | optional oracle/supporting diagnostic |

The lifecycle definitions are normative in `REPRODUCIBILITY.md`. `REPORTED`
means deterministic publication from indexed verified artifacts; it does not
promote exploratory C05--C08 evidence into a primary novelty claim.

## Claim boundary

C01--C04 remain replication and benchmark context. C05--C08 remain exploratory
or appendix evidence. C08 motivates the rank-attainability question but does not
show that ACI generally fails under distribution shift. The candidate C11
contribution is an application-level comparison with the exact Beta reference,
not a new conformal theorem. Any C12 result must be labelled oracle-only and
must not be described as a deployable target-domain repair.

## Independent reconstruction

Use `docs/RUNBOOK.md` to build into a fresh output directory, then compare every
generated file hash with `reports/results/provenance.json`. Raw data verification
and official artifact availability remain local because `data/raw/` and
`outputs/` are intentionally excluded from Git.
