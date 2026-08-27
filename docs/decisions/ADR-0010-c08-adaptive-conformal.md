# ADR-0010: C08 adaptive conformal inference

## Status

Accepted.

## Decision

C08 evaluates adaptive conformal inference (ACI) with the frozen FD001 C02
pipeline on FD001, FD002, FD003, and FD004. The model, temporal transformer,
and calibration residuals are fit once per seed from FD001 base-train and
calibration engines exactly as in C02. No target-domain data is used for fit,
tuning, or initial calibration.

Official test-engine endpoints are processed in increasing engine-ID order.
For endpoint `t`, its interval uses the current adaptive alpha and fixed FD001
calibration residuals. Only after storing that interval may its RUL be
revealed. Its miss indicator updates the next alpha by
`clip(alpha_t + gamma * (alpha_nominal - miss_t), alpha_min, alpha_max)`.

The fixed C08 parameters are gamma `0.01` and bounds `[0.001, 0.999]`.
Intervals remain symmetric around the C02-clipped point center and unbounded.
This is a prequential simulation using NASA's official test-ID order; it is
not a claim that all RUL labels are available in deployment at once. It uses
the online adaptation framing of Gibbs and Candès, "Adaptive Conformal
Inference Under Distribution Shift" (2021), arXiv:2106.00170.

## Consequences

C08 retains the sequential alpha trajectory, interval quantile, and
post-outcome miss indicator per endpoint. It is an adaptive-monitoring
comparison, not retrospective target-domain recalibration or a universal
finite-sample coverage guarantee under arbitrary shifts.
