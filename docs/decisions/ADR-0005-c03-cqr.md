# ADR-0005: C03 conformalized quantile regression

## Decision

C03 evaluates conformalized quantile regression (CQR) on FD001 endpoints. For each primary alpha, two fixed `HistGradientBoostingRegressor` models are fit on base-train rows only: quantiles `(0.05, 0.95)` for alpha `0.10` and `(0.025, 0.975)` for alpha `0.05`. The models use the same train-only temporal features and fixed hyperparameters as C02.

Calibration uses one deterministic endpoint per calibration engine. The conformity score is `max(lower_prediction - y, y - upper_prediction)`, and the calibrated interval is `[lower_prediction - q, upper_prediction + q]`. Calibration truth is derived from the complete trajectory before truncation and capped at 125. Intervals are not clipped after construction.

The official FD001 test set is used only for endpoint prediction and evaluation. Validation engines are not used for model fitting, quantile selection, calibration, or tuning. C03 is frozen to the same seeds, split, cut-point policy, feature windows, cap, and engine-level bootstrap policy as C02.

## Consequences

C03 reports coverage, mean width, normalized interval score, and deterministic engine-level bootstrap confidence intervals. Raw and capped targets, raw quantile predictions, conformity scores, calibrated intervals, model specifications, and complete provenance are retained in immutable artifacts.
