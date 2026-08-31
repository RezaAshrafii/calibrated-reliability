# Research protocol

## Status and scope

This protocol records the frozen contracts used by completed C01--C08 and the
review gates that constrain future work. It contains no numerical research
results. The project studies calibrated predictive reliability under structured
distribution shift; it does not claim a new conformal method, state-of-the-art
performance, causal safety effects, or universal guarantees.

C01--C04 are replication and benchmark context. C05--C08 are exploratory or
appendix evidence. Gate N returned `REFRAME`: the candidate primary contribution
is a narrow application-level audit of engine-level conformal rank attainability
and resolution against the exact Beta reference and the C-MAPSS observation
mechanism. `docs/NOVELTY_MATRIX.md` and `docs/RELATED_WORK.md` govern the claim
boundary.

## Questions and hypotheses

- **RQ1 -- benchmark context:** What reproducible point and interval reliability
  behavior is observed under the declared C-MAPSS in-distribution and structured
  shift evaluations?
- **RQ2 -- candidate primary mechanism:** When whole engines are calibration
  units, how do finite-rank attainability and resolution constrain the frozen
  C-MAPSS conformal pipeline, and how does its calibration-draw distribution
  compare with the exact exchangeable Beta reference under the benchmark's
  observation mechanism?
- **RQ3 -- optional supporting diagnostic:** After controlling observation
  policy and RUL-cap saturation, how does strictly disjoint labelled
  target-domain recalibration compare with frozen source calibration when the
  predictor is held fixed?

RQ1 is addressed only as replication/context by C01--C04. C05--C08 are
exploratory evidence that motivates RQ2 but does not answer it. The C11-A and
implementation-readiness reviews passed, and the frozen C11 design was executed
once from clean commit `cba16d0`; its artifact and deterministic report were
subsequently independently reconstructed and verified. RQ3 is a
non-deployable oracle contrast and is
considered only after C11; it is not a novel repair method.

## Evaluation units and splits

C-MAPSS uses engine-level partitions and the last observed test-engine endpoint
as the primary evaluation unit. Transformers, feature selectors, scalers,
thresholds, and tuning decisions must be fit only on relevant training data.
Official test data is never used for tuning. MALT is deferred outside the core
study; ADR-0002 preserves the grouping rule that would apply if a separately
reviewed future study revives it.

Phase 5 feature construction is fail-closed. Predictor input contains only `engine_id`, `cycle`, the three operating settings, and NASA C-MAPSS sensors `sensor_1` through `sensor_21`; targets are passed separately and are never accepted as feature columns. Temporal output may additionally contain `cycle_index`, one-cycle sensor deltas, and past-only rolling mean, population standard deviation, and least-squares slope features for configured positive windows. The fit-time column schema is enforced during transform. `engine_id` is retained only for trajectory alignment and is removed before model input. `cycle_ratio` and any feature derived from terminal trajectory length are prohibited.

Operating-regime clustering and all scalers are fit on base-training rows only. A requested clustering is valid only when it realizes exactly the requested number of non-empty regimes. Single-setting or invalid-clustering cases use a deterministic global scaler and record the fallback reason in the fitted transformer.

For C-MAPSS, each source training set is split by engine into 60% base-train, 20% calibration, and 20% validation using each fixed seed. Calibration uses one deterministic cut point per calibration engine, with at least 30 observed cycles and a cut point between 40% and 90% of that engine's observed trajectory. The primary RUL cap is 125; cap 130 is the preregistered sensitivity. Fixed seeds are `13, 37, 73, 101, 137`, and primary conformal levels are alpha `0.10` and `0.05`. C-MAPSS shift labels are FD001→FD001 in-distribution, FD001→FD002 operating-condition shift, FD001→FD003 fault-mode shift, and FD001→FD004 compound/structural shift. FD003/FD004 must not be described as pure covariate shift.

The calibration and official-test observation mechanisms are not assumed to be
exchangeable. Calibration endpoints are policy-selected prefixes of complete
run-to-failure training trajectories, whereas official test endpoints are
provided by NASA with unobserved future trajectories and a separate RUL file.
Differences among domain, observed-cycle horizon, degradation stage, and
cap-saturation frequency can therefore be entangled. C04--C08 are benchmark
evaluations under these declared mechanisms, not ordinary-exchangeability
coverage guarantees. Any future C11/C12 interpretation must treat this mismatch
as a measured diagnostic or limitation rather than an established cause.

C01 is a cycle-weighted point-baseline experiment: every observed base-training cycle contributes one training row, while evaluation uses exactly one final observed endpoint per official FD001 test engine. It uses past-only temporal features without regime clustering or regime-aware scaling; those methods are reserved for C07. Both training targets and endpoint targets are capped at 125. Raw targets and raw predictions are retained for audit, but primary C01 metrics use targets and predictions clipped to the preregistered support `[0, 125]`. Signed error is defined as `prediction - target`, so positive values indicate overprediction. These choices are fixed in ADR-0003 and the executable C01 configuration.

C02 uses the same fixed C01 point models and train-only temporal preprocessing. It computes absolute residuals on exactly one endpoint at a deterministic cut point for each calibration engine, using the capped full-trajectory RUL truth; the observed calibration prefix is never treated as a run-to-failure trajectory. The official FD001 test endpoints are the primary evaluation set, while validation engines remain held out for internal checks. C02 is frozen to cap 125, seeds `13,37,73,101,137`, alpha `0.10` and `0.05`, temporal windows `5,10,20`, zero variance threshold, and calibration cut points with at least 30 observed cycles in the 40–90% range. Split-conformal quantiles request the finite-sample order statistic `r = ceil((n_cal + 1)(1 - alpha))`. A finite order statistic is attainable only when `r <= n_cal`. The regimes are `interior` for `r < n_cal`, `max_statistic` for `r == n_cal`, and `finite_quantile_unattainable` for `r > n_cal`. The generic `conformal_quantile` helper fails closed for the last regime unless a caller explicitly selects either the augmented-score `infinite` convention or the historical `legacy_max_clamp` policy. The finite legacy sensitivity value must not be described as retaining the nominal finite-sample guarantee. ADR-0011 governs this distinction and requires future outputs to retain `n_cal`, requested rank, effective rank, attainability, regime, and policy. Calibration and test centers use the C01 clipped prediction policy, and intervals remain symmetric and unbounded rather than being clipped after construction. Coverage, width, and normalized interval score use deterministic engine-level 95% percentile bootstrap confidence intervals with 2,000 resamples; normalized interval score is mean interval score divided by cap 125. Bootstrap resampling uses the experiment seed as its declared seed policy and this policy is recorded in the resolved configuration and manifest.

C03 uses conformalized quantile regression with fixed lower/upper HistGradientBoosting quantile models. Alpha `0.10` uses quantiles `0.05/0.95`; alpha `0.05` uses `0.025/0.975`. Quantile models and temporal preprocessing are fit only on base-train rows. Calibration uses one deterministic endpoint per calibration engine and the CQR conformity score `max(lower - truth, truth - upper)`. Intervals are expanded by the finite-sample conformal quantile, remain unbounded, and are evaluated on exactly one official test endpoint per engine. C03 inherits the C02 cap, split, cut-point, seed, feature, bootstrap and provenance rules.

C04 reuses the frozen C02 split-conformal pipeline trained and calibrated on FD001, then evaluates it without target-domain recalibration on FD001, FD002, FD003 and FD004 official test endpoints. FD001→FD002 is operating-condition shift; FD001→FD003 is fault-mode/structural shift; FD001→FD004 is compound/structural shift. Target data is never used for fitting, feature selection, quantile selection or tuning.

C05 is a transductive weighted-conformal sensitivity experiment from FD001 to FD002, FD003 and FD004. It reuses the frozen FD001 C02 transformer, point models and calibration residuals. Unlabeled target endpoint operating settings may be used only to estimate target/source density ratios; target RUL is never used for fitting or weighting. Each target endpoint uses its own ratio in the weighted conformal quantile. Raw ratios are clipped to `[0.05, 1.0]`, with calibration weights normalized to mean one; this is a bounded-weight sensitivity condition rather than an untruncated covariate-shift guarantee.

C06 is a preregistered FD001-to-FD001 sensitivity analysis. It reports four fixed, non-selected conditions: the primary C02 condition (cap 125, calibration cut-point range 40–90%), cap 130 with the same range, cap 125 with early cut points (40–65%), and cap 125 with late cut points (65–90%). All other C02 choices remain fixed. Conditions are reported separately; no validation or official-test result may select a condition.

C07 evaluates point baselines after train-only regime-aware scaling. It fits the temporal transformer, setting standardizer, silhouette-selected operating-regime clustering (candidate counts 2–6, random state 13), per-regime scalers, and point models only on FD001 base-train engines. The full frozen state is then evaluated without refitting on FD001, FD002, and FD004 official test endpoints. FD001 is the in-distribution reference, FD002 is operating-condition shift, and FD004 is compound/structural shift. A deterministic global-scaling fallback is retained when no valid clustering can be realized. C07 uses the C01 target cap, clipping, and endpoint metrics, but its preregistered HGB settings differ from C01 (`max_iter=200`, `max_leaf_nodes=15`, `l2_regularization=0.0`). It is therefore not interpreted as a clean one-factor preprocessing ablation. C07 introduces no conformal or adaptive method.

C08 is an adaptive conformal prequential evaluation. It fits and calibrates the C02 FD001 pipeline once per seed, then evaluates target endpoints in increasing official engine-ID order. This is a deterministic benchmark order, not an assertion that engine IDs encode deployment chronology. Each interval uses only the current alpha and fixed FD001 calibration residuals; after the endpoint outcome is revealed, its miss indicator updates alpha for the next endpoint. Gamma is fixed at 0.01 and alpha is projected to [0.001, 0.999]. C08 does not use target outcomes for fitting, tuning, initial calibration, or a current interval. Its standard bootstrap is a conditional fixed-path summary because it resamples realized interval rows without rerunning the adaptive trajectory. With 20 fixed FD001 calibration residuals, adaptive alpha below `1 / 21` requests an unattainable finite rank; C08 explicitly selects `legacy_max_clamp`, which saturates the interval half-width at the largest residual. Future C08 outputs retain endpoint-level rank and regime diagnostics plus run-level saturation summaries. C08 therefore supports a benchmark-specific saturation diagnostic under that declared policy, not a general claim that ACI fails to adapt under shift. The verified historical C08 artifacts remain immutable and are not regenerated for the ADR-0011 corrections. C08 reports a sequential simulation rather than a batch target-domain recalibration or a universal shift-coverage guarantee.

## Primary metrics

RUL: RMSE, MAE, signed error, NASA asymmetric score, interval coverage,
interval width, normalized interval score, and engine-level bootstrap confidence
intervals. Completed metric artifacts are reportable only through the tracked
Gate D builder, which reconstructs the tables under `reports/results/` from
indexed verified artifacts.

## Future design gates

C11-A independently accepted ADR-0012, and a separate implementation-readiness
review passed before the single authorized execution. One immutable local C11
artifact exists from clean commit `cba16d0`. Artifact-level and report-level
audits independently reconstructed its retained calculations, so C11 is now
`IMPLEMENTED`, `EXECUTED`, `VERIFIED`, and `REPORTED`. The frozen FD001-only design uses one seed-13 HGB
predictor, a disjoint 60-engine fit role and 40-engine finite
reservoir, the exact existing `generate_cut_points` algorithm, exact
combinatorial subset integration, six attainable size/alpha cells, continuous
Beta and finite-evaluation beta-binomial references, and an engine-weighted
observed-cycle Wasserstein diagnostic. The same five signed or unsigned
discrepancies are reported separately against both references under frozen CDF,
tail, tie and numerical-tolerance conventions. It replaces Monte Carlo
calibration draws with an exact conditional finite-reservoir distribution and
treats deviation from the references--not raw draw variance--as the scientific
quantity. C11 is retrospectively motivated by inspected C01--C08 outcomes and
prospectively frozen only with respect to future C11 computation; it is a
descriptive audit, not confirmatory preregistration. No binary adequacy or
material-departure classification is permitted. The completed C11-A review
preceded configuration and production implementation; the separate
implementation-readiness review passed before execution and official artifact
generation. Independent artifact-level and report-level reconstruction were
completed before assigning `VERIFIED` and `REPORTED`.

C12 is optional and must not precede the reviewed C11 mechanism result. If
retained, only Conditions A/B may be designed. Target calibration and evaluation
must be disjoint; target labels must not alter the frozen predictor; observation
policy and RUL-cap saturation require explicit controls; and every output must
state `ORACLE / DIAGNOSTIC - NOT A DEPLOYMENT METHOD`. C12-D is out of scope.

MALT, N-CMAPSS expansion, additional deep models, and product/dashboard work
remain deferred outside the current core study.

## Integrity rules

Runs must retain configuration, seed, data hashes, environment versions, split
manifest, git SHA, metrics, predictions, and logs. Experiment runners must
reject dirty worktrees and must not overwrite an existing run directory. No
number is allowed into a report or paper unless the tracked Gate D builder has
reconstructed it from an indexed verified artifact. ACI is an adaptive/online
calibration method, not a batch method. Negative, null, and inconvenient results
must be retained. Gate D report publication additionally requires the tracked
Python 3.11.9 interpreter and records the platform, direct runtime package
versions, `.python-version` hash, and `uv.lock` hash. Git state, environment,
and official artifacts are revalidated immediately before atomic publication.
