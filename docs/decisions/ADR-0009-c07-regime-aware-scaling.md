# ADR-0009: C07 regime-aware scaling comparison

## Status

Historical status at this decision checkpoint: accepted, implemented, executed,
and independently verified as exploratory/appendix evidence; numerical
reporting remained pending Gate D. Current lifecycle status is maintained in
`docs/experiment_registry.md`.

## Decision

C07 evaluates point-prediction baselines trained on FD001 and evaluated at
official test-engine endpoints in FD001, FD002, and FD004.  FD001 is the
in-distribution reference, FD002 is operating-condition shift, and FD004 is
compound/structural shift.  FD003 is intentionally excluded from this
comparison because its fault-mode shift makes an operating-regime scaling
interpretation less direct.

Temporal features are fitted on FD001 base-train engines only.  The
`RegimeAwareScaler` then fits its setting standardizer, silhouette-selected
KMeans clustering (candidates 2--6), and per-regime scalers only on those
same rows.  Its fixed random state is 13.  If no valid clustering is
available, its documented deterministic global-scaling fallback is used.

The resulting frozen temporal transformer and scaler are reused without
fitting, tuning, or target-domain recalibration for every C07 target domain.
C07 uses its own preregistered point-model configuration: mean baseline, Ridge
alpha 1.0, and HistGradientBoosting with 200 iterations, learning rate 0.05,
15 maximum leaf nodes, and zero L2 regularization. These differ from C01's HGB
settings and must not be described as a pure preprocessing-only ablation of
C01. `engine_id` remains alignment metadata and never enters a model. The
target policy is the C01 policy:
RUL is capped at 125 and predictions used for metrics are clipped to
`[0, 125]`; raw target values and predictions are retained.

## Consequences

This is a train-only regime-scaling experiment with a separately fixed model
configuration, not a clean one-factor preprocessing ablation and not a claim
that regime assignment repairs arbitrary structural shift or provides
conformal coverage.
The fitted number of regimes, fallback reason, selected sensors, and feature
schema are retained in each immutable artifact.
