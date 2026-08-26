# ADR-0007: C05 weighted conformal calibration

## Decision

C05 evaluates weighted split-conformal RUL intervals from FD001 to FD002, FD003 and FD004. The FD001 fitted temporal transformer, point models and calibration endpoints are reused unchanged from the frozen C02 pipeline.

Target test trajectories may be used without RUL labels only to estimate operating-condition density ratios. The density ratio is estimated with deterministic logistic regression using only the three operating settings. No target RUL, target labels or target-derived model refitting is permitted.

Calibration residuals are weighted by the estimated target/source density ratio. A domain-level target weight is the mean ratio over target endpoints. The weighted finite-sample quantile is the smallest sorted residual whose cumulative calibration weight reaches `(1 - alpha) * (sum calibration weights + target weight)`. Prediction models and temporal features are never refit on target data.

Primary settings inherit C02: cap 125, alpha 0.10/0.05, five seeds, engine endpoints, and deterministic engine-level bootstrap with 2,000 resamples at 95% confidence.
