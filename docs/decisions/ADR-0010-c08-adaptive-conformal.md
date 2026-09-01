# ADR-0010: C08 adaptive conformal inference

## Status

Historical status at this decision checkpoint: accepted, implemented, executed,
and independently verified as exploratory/appendix evidence; numerical
reporting remained pending Gate D. Current lifecycle status is maintained in
`docs/experiment_registry.md`.

## Decision

C08 evaluates adaptive conformal inference (ACI) with the frozen FD001 C02
pipeline on FD001, FD002, FD003, and FD004. The model, temporal transformer,
and calibration residuals are fit once per seed from FD001 base-train and
calibration engines exactly as in C02. No target-domain data is used for fit,
tuning, or initial calibration.

Official test-engine endpoints are processed in increasing engine-ID order.
This is a deterministic benchmark order, not evidence that NASA engine IDs
represent deployment chronology.
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

The finite-rank attainability and resolution limits of this design are governed
by ADR-0011. With 20 calibration engines, adaptive alpha values below `1 / 21`
request a rank above the available calibration sample. The existing C08
implementation applies its legacy max-clamp policy in that case, so the
interval half-width saturates at the largest fixed FD001 residual.

## Consequences

C08 retains the sequential alpha trajectory, interval quantile, and
post-outcome miss indicator per endpoint. The standard engine-level bootstrap
resamples the realized interval rows and is therefore a conditional fixed-path
summary; it does not rerun the adaptive trajectory in each resample. It is an
adaptive-monitoring comparison, not retrospective target-domain recalibration
or a universal finite-sample coverage guarantee under arbitrary shifts.
The verified C08 artifacts therefore support a benchmark-specific saturation
diagnostic under the declared legacy policy; they do not establish that ACI in
general fails to adapt under distribution shift. Existing artifacts remain
immutable and are not regenerated for this interpretive correction.
