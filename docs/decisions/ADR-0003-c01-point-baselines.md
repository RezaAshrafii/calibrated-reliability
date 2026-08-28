# ADR-0003: C01 point-prediction baselines

## Status

Accepted; implemented, executed, and independently verified. Numerical
reporting remains pending Gate D.

## Decision

C01 uses three fixed point-prediction baselines: a training-mean predictor, standardized Ridge regression with `alpha=1.0`, and a deterministic `HistGradientBoostingRegressor` with `max_iter=50`, `learning_rate=0.05`, `max_leaf_nodes=31`, and `l2_regularization=1.0`. The primary target is the preregistered capped RUL at cap 125; cap 130 remains a later sensitivity condition.

All preprocessing is fit on base-train engines only. C01 uses temporal features without regime-aware scaling; regime-aware preprocessing remains reserved for C07. Ridge performs its declared train-only standardization internally. Evaluation is performed once per test engine at its final observed cycle. No hyperparameter search, official-test tuning, conformal calibration, or shift adaptation occurs in C01.

Training and evaluation targets are both capped to `[0, 125]`. Raw official test RUL and raw model predictions are retained in prediction artifacts, while primary metrics use values clipped to the declared target support `[0, 125]`. This policy is frozen before replacement artifacts are generated. Training remains cycle-weighted, so longer base-train trajectories contribute more supervised rows; primary evaluation remains engine-weighted with one endpoint per engine. Signed error is defined as `prediction - truth`.

## Rationale

The mean model provides a sanity lower bound, Ridge provides a transparent linear reference, and histogram gradient boosting provides a nonlinear reference without introducing a tuning loop into the first modeling phase.

## Consequences

C01 produces raw and evaluation-domain point predictions plus RMSE, MAE, signed error, and NASA asymmetric score artifacts. Interval prediction, coverage, calibration, and cross-shift claims remain outside this phase.
