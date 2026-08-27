# Research protocol

## Status and scope

This preregistration-style protocol defines evaluation before results are generated. The project studies calibrated predictive reliability under structured distribution shift. It does not claim a new method, solve AI safety, or provide universal guarantees.

## Questions and hypotheses

- **RQ1:** How do point-prediction baselines behave on C-MAPSS RUL?
- **RQ2:** How does split conformal calibration affect interval coverage and width?
- **RQ3:** How does CQR compare with split conformal at similar coverage?
- **RQ4:** How does performance change under operating-condition, fault-mode, and compound shifts?
- **RQ5:** Can weighting or adaptive calibration mitigate selected shifts?
- **RQ6:** How sensitive are conclusions to RUL cap, seed, and calibration size?
- **RQ7:** Does task-family grouping change MALT monitoring estimates?
- **RQ8:** Do marginal metrics conceal poor positive-class behavior?

These are hypotheses, not results: CQR may produce narrower intervals at comparable coverage; naive conformal may lose coverage under shift; weighting may help selectively; marginal MALT coverage may hide poor positive-class coverage; random transcript splits may be optimistic compared with task-family splits.

## Evaluation units and splits

C-MAPSS uses engine-level partitions and the last observed test-engine endpoint as the primary evaluation unit. MALT uses task-family-aware partitions and transcript-level evaluation. Transformers, feature selectors, scalers, thresholds, and tuning decisions must be fit only on relevant training data. Official test data is never used for tuning.

Phase 5 feature construction is fail-closed. Predictor input contains only `engine_id`, `cycle`, the three operating settings, and NASA C-MAPSS sensors `sensor_1` through `sensor_21`; targets are passed separately and are never accepted as feature columns. Temporal output may additionally contain `cycle_index`, one-cycle sensor deltas, and past-only rolling mean, population standard deviation, and least-squares slope features for configured positive windows. The fit-time column schema is enforced during transform. `engine_id` is retained only for trajectory alignment and is removed before model input. `cycle_ratio` and any feature derived from terminal trajectory length are prohibited.

Operating-regime clustering and all scalers are fit on base-training rows only. A requested clustering is valid only when it realizes exactly the requested number of non-empty regimes. Single-setting or invalid-clustering cases use a deterministic global scaler and record the fallback reason in the fitted transformer.

For C-MAPSS, each source training set is split by engine into 60% base-train, 20% calibration, and 20% validation using each fixed seed. Calibration uses one deterministic cut point per calibration engine, with at least 30 observed cycles and a cut point between 40% and 90% of that engine's observed trajectory. The primary RUL cap is 125; cap 130 is the preregistered sensitivity. Fixed seeds are `13, 37, 73, 101, 137`, and primary conformal levels are alpha `0.10` and `0.05`. C-MAPSS shift labels are FD001→FD001 in-distribution, FD001→FD002 operating-condition shift, FD001→FD003 fault-mode shift, and FD001→FD004 compound/structural shift. FD003/FD004 must not be described as pure covariate shift.

C01 is a cycle-weighted point-baseline experiment: every observed base-training cycle contributes one training row, while evaluation uses exactly one final observed endpoint per official FD001 test engine. It uses past-only temporal features without regime clustering or regime-aware scaling; those methods are reserved for C07. Both training targets and endpoint targets are capped at 125. Raw targets and raw predictions are retained for audit, but primary C01 metrics use targets and predictions clipped to the preregistered support `[0, 125]`. Signed error is defined as `prediction - target`, so positive values indicate overprediction. These choices are fixed in ADR-0003 and the executable C01 configuration.

C02 uses the same fixed C01 point models and train-only temporal preprocessing. It computes absolute residuals on exactly one endpoint at a deterministic cut point for each calibration engine, using the capped full-trajectory RUL truth; the observed calibration prefix is never treated as a run-to-failure trajectory. The official FD001 test endpoints are the primary evaluation set, while validation engines remain held out for internal checks. C02 is frozen to cap 125, seeds `13,37,73,101,137`, alpha `0.10` and `0.05`, temporal windows `5,10,20`, zero variance threshold, and calibration cut points with at least 30 observed cycles in the 40–90% range. Split-conformal quantiles use the finite-sample order statistic `ceil((n_cal + 1)(1 - alpha))`. Calibration and test centers use the C01 clipped prediction policy, and intervals remain symmetric and unbounded rather than being clipped after construction. Coverage, width, and normalized interval score use deterministic engine-level 95% percentile bootstrap confidence intervals with 2,000 resamples; normalized interval score is mean interval score divided by cap 125. Bootstrap resampling uses the experiment seed as its declared seed policy and this policy is recorded in the resolved configuration and manifest.

C03 uses conformalized quantile regression with fixed lower/upper HistGradientBoosting quantile models. Alpha `0.10` uses quantiles `0.05/0.95`; alpha `0.05` uses `0.025/0.975`. Quantile models and temporal preprocessing are fit only on base-train rows. Calibration uses one deterministic endpoint per calibration engine and the CQR conformity score `max(lower - truth, truth - upper)`. Intervals are expanded by the finite-sample conformal quantile, remain unbounded, and are evaluated on exactly one official test endpoint per engine. C03 inherits the C02 cap, split, cut-point, seed, feature, bootstrap and provenance rules.

C04 reuses the frozen C02 split-conformal pipeline trained and calibrated on FD001, then evaluates it without target-domain recalibration on FD001, FD002, FD003 and FD004 official test endpoints. FD001→FD002 is operating-condition shift; FD001→FD003 is fault-mode/structural shift; FD001→FD004 is compound/structural shift. Target data is never used for fitting, feature selection, quantile selection or tuning.

C05 is a transductive weighted-conformal sensitivity experiment from FD001 to FD002, FD003 and FD004. It reuses the frozen FD001 C02 transformer, point models and calibration residuals. Unlabeled target endpoint operating settings may be used only to estimate target/source density ratios; target RUL is never used for fitting or weighting. Each target endpoint uses its own ratio in the weighted conformal quantile. Raw ratios are clipped to `[0.05, 1.0]`, with calibration weights normalized to mean one; this is a bounded-weight sensitivity condition rather than an untruncated covariate-shift guarantee.

C06 is a preregistered FD001-to-FD001 sensitivity analysis. It reports four fixed, non-selected conditions: the primary C02 condition (cap 125, calibration cut-point range 40–90%), cap 130 with the same range, cap 125 with early cut points (40–65%), and cap 125 with late cut points (65–90%). All other C02 choices remain fixed. Conditions are reported separately; no validation or official-test result may select a condition.

C07 compares C01 point baselines after train-only regime-aware scaling. It fits the temporal transformer, setting standardizer, silhouette-selected operating-regime clustering (candidate counts 2–6, random state 13), per-regime scalers, and point models only on FD001 base-train engines. The full frozen state is then evaluated without refitting on FD001, FD002, and FD004 official test endpoints. FD001 is the in-distribution reference, FD002 is operating-condition shift, and FD004 is compound/structural shift. A deterministic global-scaling fallback is retained when no valid clustering can be realized. C07 uses the C01 target cap, clipping, baseline-model specifications, and endpoint metrics, but no conformal or adaptive method.

C08 is an adaptive conformal prequential evaluation. It fits and calibrates the C02 FD001 pipeline once per seed, then evaluates target endpoints in increasing official engine-ID order. This is a deterministic benchmark order, not an assertion that engine IDs encode deployment chronology. Each interval uses only the current alpha and fixed FD001 calibration residuals; after the endpoint outcome is revealed, its miss indicator updates alpha for the next endpoint. Gamma is fixed at 0.01 and alpha is projected to [0.001, 0.999]. C08 does not use target outcomes for fitting, tuning, initial calibration, or a current interval. Its standard bootstrap is a conditional fixed-path summary because it resamples realized interval rows without rerunning the adaptive trajectory. C08 reports a sequential simulation rather than a batch target-domain recalibration or a universal shift-coverage guarantee.

## Primary metrics

RUL: RMSE, MAE, signed error, NASA asymmetric score, interval coverage, interval width, normalized interval score, and engine-level bootstrap confidence intervals. MALT: AUPRC as primary, AUROC, precision, recall, F1, TPR at 5% FPR, Brier score, log loss, marginal and label-conditional conformal coverage, set size, singleton rate, and empty-set rate.

## Integrity rules

Runs must retain configuration, seed, data hashes, environment versions, split manifest, git SHA, metrics, predictions, and logs. Experiment runners must reject dirty worktrees and must not overwrite an existing run directory. No number is allowed into the paper unless it exists in a verified artifact. ACI is an adaptive/online calibration method, not a batch method. MALT is not assumed to be a clean binary dataset. Negative or inconvenient results must be retained.
