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

Fixed seeds are `13, 37, 73, 101, 137`. C-MAPSS shift labels are FD001→FD001 in-distribution, FD001→FD002 operating-condition shift, FD001→FD003 fault-mode shift, and FD001→FD004 compound/structural shift. FD003/FD004 must not be described as pure covariate shift.

## Primary metrics

RUL: RMSE, MAE, signed error, NASA asymmetric score, interval coverage, interval width, normalized interval score, and engine-level bootstrap confidence intervals. MALT: AUPRC as primary, AUROC, precision, recall, F1, TPR at 5% FPR, Brier score, log loss, marginal and label-conditional conformal coverage, set size, singleton rate, and empty-set rate.

## Integrity rules

Runs must retain configuration, seed, data hashes, environment versions, split manifest, git SHA, metrics, predictions, and logs. No number is allowed into the paper unless it exists in a verified artifact. ACI is an adaptive/online calibration method, not a batch method. MALT is not assumed to be a clean binary dataset. Negative or inconvenient results must be retained.

