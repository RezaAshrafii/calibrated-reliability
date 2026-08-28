# ADR-0006: C04 C-MAPSS shift matrix

## Status

Accepted; implemented, executed, and independently verified. Numerical
reporting remains pending Gate D.

## Decision

C04 evaluates the frozen C02 split-conformal pipeline trained on FD001 and evaluated on FD001, FD002, FD003 and FD004 official test sets. The FD001 base-train, calibration and validation engine partitions, temporal feature fit, models and conformal quantiles are reused unchanged for each target dataset. Target datasets are used only for endpoint transformation, prediction and evaluation.

The shift labels are FD001→FD001 in-distribution, FD001→FD002 operating-condition shift, FD001→FD003 fault-mode/structural shift, and FD001→FD004 compound/structural shift. No target dataset is used for tuning or recalibration.
