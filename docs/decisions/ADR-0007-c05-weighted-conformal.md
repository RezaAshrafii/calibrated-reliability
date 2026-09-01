# ADR-0007: C05 weighted conformal calibration

## Status

Historical status at this decision checkpoint: accepted, implemented, executed,
and independently verified as bounded-weight exploratory/appendix evidence;
numerical reporting remained pending Gate D. Current lifecycle status is
maintained in `docs/experiment_registry.md`.

## Decision

C05 evaluates weighted split-conformal RUL intervals from FD001 to FD002, FD003 and FD004. The FD001 fitted temporal transformer, point models and calibration endpoints are reused unchanged from the frozen C02 pipeline.

Target test trajectories may be used without RUL labels only to estimate operating-condition density ratios. The density ratio is estimated with deterministic logistic regression using only the three operating settings. No target RUL, target labels or target-derived model refitting is permitted.

Calibration residuals are weighted by the estimated target/source density ratio. Each target endpoint retains its own density-ratio weight and receives its own weighted finite-sample quantile: the smallest sorted residual whose cumulative calibration weight reaches `(1 - alpha) * (sum calibration weights + that endpoint's target weight)`. Raw ratios are clipped to `[0.05, 1.0]`; calibration weights are normalized to mean one, and target weights are clipped again to the same range. This is a conservative bounded-weight sensitivity condition, not a claim of the untruncated covariate-shift guarantee. Logistic regression is standardized and frozen to `C=1.0`, `max_iter=1000`, `lbfgs`, and `random_state=0`. Prediction models and temporal features are never refit on target data.

Primary settings inherit C02: cap 125, alpha 0.10/0.05, five seeds, engine endpoints, and deterministic engine-level bootstrap with 2,000 resamples at 95% confidence.
