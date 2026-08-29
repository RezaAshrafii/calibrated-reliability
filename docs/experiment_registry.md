# Experiment registry

## Status contract

The lifecycle terms are cumulative and use the operational definitions in
`REPRODUCIBILITY.md`. In particular, `VERIFIED` requires matching hashes and
provenance plus independent metric reconstruction. `REPORTED` requires the
tracked Gate D results builder; it cannot be inferred from local output files.

## Active completed experiments

| ID | Role | Method | Source -> target | Unit | Alpha | Seeds | Config | Lifecycle | Reporting class |
|---|---|---|---|---|---|---|---|---|---|
| C01 | RQ1 context | Point baselines | FD001 -> FD001 | engine endpoint | -- | 13,37,73,101,137 | `configs/cmapss/fd001_baseline.yaml` | IMPLEMENTED / EXECUTED / VERIFIED / REPORTED | replication/context |
| C02 | RQ1 context | Split conformal | FD001 -> FD001 | engine endpoint | .10,.05 | 13,37,73,101,137 | `configs/cmapss/conformal.yaml` | IMPLEMENTED / EXECUTED / VERIFIED / REPORTED | replication/context |
| C03 | RQ1 context | CQR | FD001 -> FD001 | engine endpoint | .10,.05 | 13,37,73,101,137 | `configs/cmapss/cqr.yaml` | IMPLEMENTED / EXECUTED / VERIFIED / REPORTED | replication/context |
| C04 | RQ1 context | Frozen shift matrix | FD001 -> FD001/2/3/4 | engine endpoint | .10,.05 | 13,37,73,101,137 | `configs/cmapss/shift_matrix.yaml` | IMPLEMENTED / EXECUTED / VERIFIED / REPORTED | replication/context |
| C05 | RQ2 motivation | Bounded weighted conformal | FD001 -> FD002/3/4 | engine endpoint | .10,.05 | 13,37,73,101,137 | `configs/cmapss/weighted_conformal.yaml` | IMPLEMENTED / EXECUTED / VERIFIED / REPORTED | exploratory/appendix |
| C06 | RQ2 sensitivity | Cap/cut-point sensitivity | FD001 -> FD001 | engine endpoint | .10,.05 | 13,37,73,101,137 | `configs/cmapss/sensitivity.yaml` | IMPLEMENTED / EXECUTED / VERIFIED / REPORTED | exploratory/appendix |
| C07 | RQ2 context | Regime-aware scaling | FD001 -> FD001/2/4 | engine endpoint | -- | 13,37,73,101,137 | `configs/cmapss/regime_scaling.yaml` | IMPLEMENTED / EXECUTED / VERIFIED / REPORTED | exploratory/appendix |
| C08 | RQ2 anchor diagnostic | Prequential ACI | FD001 -> FD001/2/3/4 | prequential engine endpoint | .10,.05 | 13,37,73,101,137 | `configs/cmapss/adaptive_conformal.yaml` | IMPLEMENTED / EXECUTED / VERIFIED / REPORTED | exploratory/appendix |

Every C01--C08 row is `REPORTED` through the tracked Gate D artifact index and
deterministic builder. This status does not extend to C11 or C12.

## Candidate future work

- C11 is not yet an experiment-registry entry. Proposed ADR-0012 specifies the
  narrow FD001 engine-level finite-reservoir rank-attainability/resolution audit.
  It is retrospectively motivated and prospectively frozen as a descriptive
  audit, not confirmatory preregistration.
  It must pass independent C11-A design review before configuration or production
  implementation; implementation tests and a separate readiness review are
  required before execution.
- C12 is optional and secondary. If retained after C11, only Conditions A/B may
  be designed, with an explicit `ORACLE / DIAGNOSTIC - NOT A DEPLOYMENT METHOD`
  label, disjoint target calibration/evaluation, observation-policy controls,
  and RUL-cap stratification.

## Removed or deferred scope

- Former C09 seed sensitivity is removed as a separate experiment: the declared
  C01--C08 designs already execute all five frozen seeds.
- Former C10 cap sensitivity is removed as a separate experiment: cap 125 versus
  130 is implemented by C06.
- MALT experiments M01--M08 are deferred and not implemented. They are outside
  the current core study, as are C12-D, N-CMAPSS expansion, new deep models, and
  dashboard/product work.
