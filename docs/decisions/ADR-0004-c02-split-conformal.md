# ADR-0004: C02 split-conformal intervals

## Decision

C02 fits the fixed C01 point-prediction baselines on the 60% base-training engines only. It calibrates absolute residuals on exactly one deterministic cut-point endpoint from each 20% calibration engine. The official FD001 test endpoints are the primary evaluation set; the 20% validation partition remains held out for internal checks and is not used to select models, features, or conformal quantiles.

The primary cap is 125 and calibration truth is the capped RUL computed from each engine's complete training trajectory before restricting the calibration trajectory to its observed cut point. Calibration and test centers are the C01 evaluation-domain predictions clipped to `[0, 125]`. For each preregistered alpha (`0.10`, `0.05`), the finite-sample conformal quantile uses the order statistic with rank `ceil((n_cal + 1)(1 - alpha))`, capped at the largest available residual. Intervals are symmetric around the clipped center and are not clipped, preserving the stated split-conformal construction.

## Consequences

C02 reports endpoint coverage, mean interval width, and normalized interval score for every fixed baseline and alpha. Raw test RUL and raw point predictions remain in artifacts. No model, transformer, hyperparameter, or quantile is fit using official test endpoints.
