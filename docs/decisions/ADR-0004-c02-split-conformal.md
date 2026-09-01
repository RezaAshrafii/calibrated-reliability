# ADR-0004: C02 split-conformal intervals

## Status

Historical status at this decision checkpoint: accepted, implemented, executed,
and independently verified; numerical reporting remained pending Gate D.
Current lifecycle status is maintained in `docs/experiment_registry.md`.

## Decision

C02 fits the fixed C01 point-prediction baselines on the 60% base-training engines only. It calibrates absolute residuals on exactly one deterministic cut-point endpoint from each 20% calibration engine. The official FD001 test endpoints are the primary evaluation set; the 20% validation partition remains held out for internal checks and is not used to select models, features, or conformal quantiles.

The preregistered C02 configuration is frozen to cap 125, seeds `13,37,73,101,137`, alpha `0.10` and `0.05`, temporal windows `5,10,20`, a zero variance threshold, and cut-point constraints of at least 30 cycles and 40–90% of the complete trajectory. Bootstrap resampling is frozen to 2,000 engine-level resamples at 95% confidence, using the experiment seed as the bootstrap seed. A C02 runner rejects deviations; sensitivity conditions belong to their separately registered experiments.

Calibration truth is the capped RUL computed from each engine's complete training trajectory before restricting the calibration trajectory to its observed cut point. Calibration and test centers are the C01 evaluation-domain predictions clipped to `[0, 125]`. For each preregistered alpha, the finite-sample conformal quantile uses the order statistic with rank `ceil((n_cal + 1)(1 - alpha))`. Both fixed C02 ranks are attainable; no unattainable-rank fallback is invoked. Intervals are symmetric around the clipped center and are not clipped, preserving the stated split-conformal construction.

ADR-0011 distinguishes an interior rank, a valid sample-maximum rank, and a
finite rank that is unattainable from the available calibration sample. The
largest-residual cap is the repository's explicit legacy policy for the last
case; it does not retain the nominal finite-sample conformal guarantee when the
requested rank exceeds `n_cal`. C02 itself uses 20 calibration engines, for
which its fixed alpha values request ranks 19 and 20 and are therefore finite.

## Consequences

C02 reports endpoint coverage, mean interval width, and normalized interval score for every fixed baseline and alpha. The normalized interval score is the mean interval score divided by the cap 125. It also reports deterministic engine-level 95% percentile bootstrap confidence intervals based on 2,000 resamples for each metric, with the experiment seed used as the bootstrap seed. Raw test RUL, raw point predictions, calibration endpoint targets, clipped calibration centers, and absolute calibration residuals remain in artifacts. No model, transformer, hyperparameter, or quantile is fit using official test endpoints.
