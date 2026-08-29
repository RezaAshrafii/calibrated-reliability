# ADR-0012: C11 engine-level finite-reservoir conformal audit

## Status

Proposed -- design only. This ADR must pass independent C11-A review before a
C11 configuration, production module, runner, or artifact writer is added. A
separate implementation-readiness review is required before execution. No C11
result or artifact exists at this checkpoint.

## Decision context

Gate N returned `REFRAME`. C11 is not a new conformal method, a new theorem, a
generic discovery of calibration-set variability, or evidence that calibration
size is universally more important than adaptation. The admissible question is
narrower:

> With whole engines as calibration units and the predictor, reservoir,
> calibration endpoint policy, and evaluation set held fixed, what exact
> distribution of interval behavior is induced by finite engine-level
> calibration subsets, which finite ranks are attainable, and how does the
> resulting coverage distribution differ from the idealized exchangeable Beta
> reference under the C-MAPSS observation mechanism?

The exact Beta law, calibration-grid discreteness, and generic small-sample
coverage variation are prior theory, not project contributions. C08 is only the
anchor diagnostic that motivated C11; C11 must not be framed as proving that
ACI generally fails under shift.

## Frozen scope and data roles

C11 is an FD001-to-FD001 mechanism audit. Target-shift and target-label
recalibration belong to the optional later C12 oracle diagnostic and are not
part of C11.

The design freezes:

- dataset: C-MAPSS FD001 only;
- split and predictor seed: `13`, selected as the first seed in the pre-existing
  frozen seed order rather than from performance;
- predictor-fit role: the 60 FD001 base-training engines from the deterministic
  engine split;
- calibration reservoir: the remaining 40 FD001 training engines, explicitly
  the union of the former 20 calibration and 20 validation roles for seed 13;
- evaluation: the 100 ordered official FD001 test-engine endpoints and their
  official RUL rows;
- model: only the C02/C01 `HistGradientBoostingRegressor` specification with
  `max_iter=50`, `learning_rate=0.05`, `max_leaf_nodes=31`,
  `l2_regularization=1.0`, and `random_state=13`;
- temporal preprocessing: windows `5, 10, 20`, variance threshold `0.0`, fit
  only on the 60 predictor-fit engines;
- RUL cap: `125` for fitting targets, reservoir truth, evaluation truth,
  prediction centers, and normalized interval score;
- calibration endpoint policy: one cut point per reservoir engine, generated
  once over the sorted 40-engine reservoir with seed `13`, minimum 30 observed
  cycles, and the 40--90% complete-trajectory range;
- interval form: a C02-style clipped point center with a symmetric, unbounded
  split-conformal interval.

The predictor and transformer are fit exactly once. Calibration truth is
computed from each complete source training trajectory before its observed
prefix is restricted. The prefix endpoint is never assigned terminal RUL zero
unless it is genuinely terminal. Reservoir engines cannot influence feature
selection or predictor fitting. Official test endpoints cannot influence
fitting, cut points, reservoir scores, cell selection, or quantiles.

Repurposing the former validation engines is a declared C11 role change. The
repository implementation of C01--C08 did not use validation-engine outcomes
for fitting, calibration, condition selection, or reported metrics. Because
those completed results were nevertheless inspected before C11 design, the
role change remains a retrospective-design limitation and must be disclosed.

## Sole calibration-subset random variable

Cut points, residual values, predictor state, and evaluation scores are frozen
before any calibration subset is considered. The only conceptual random
variable is a uniformly selected size-`n_cal` subset drawn without replacement
from the fixed 40-engine reservoir.

The primary analysis does not simulate this variable. It integrates it exactly.
Monte Carlo permutations, nested-prefix draws, or arbitrary resample counts are
not part of the scientific estimand. A fixed-seed Monte Carlo calculation may
exist only as a software-verification test against exact enumeration and must
not be published as C11 evidence.

## Preregistered cells and attainability

Let `N=40` be the reservoir size and

`r = ceil((n_cal + 1) * (1 - alpha))`.

C11 uses exactly these six finite observed-order-statistic cells:

| role | `n_cal` | `alpha` | requested `r` | regime | possible reservoir order indices | support size |
|---|---:|---:|---:|---|---|---:|
| primary | 10 | 0.10 | 10 | `max_statistic` | 10--40 | 31 |
| primary | 15 | 0.10 | 15 | `max_statistic` | 15--40 | 26 |
| primary | 20 | 0.10 | 19 | `interior` | 19--39 | 21 |
| primary | 30 | 0.10 | 28 | `interior` | 28--38 | 11 |
| sensitivity | 20 | 0.05 | 20 | `max_statistic` | 20--40 | 21 |
| sensitivity | 30 | 0.05 | 30 | `max_statistic` | 30--40 | 11 |

The cross-product cells `(10, 0.05)` and `(15, 0.05)` are excluded because
their requested ranks 11 and 16 exceed their calibration sizes. They must be
represented in design validation as
`not_evaluated_due_to_unattainable_finite_rank`; they must not use infinity,
`legacy_max_clamp`, a sentinel zero, or an imputed quantile. `n_cal=40` is also
excluded because a 40-of-40 subset has no calibration-subset variation.

The support interval follows from `r <= k <= N - n_cal + r`. The listed support
size is an upper bound on distinct numerical quantiles because tied reservoir
residuals are aggregated.

## Exact finite-reservoir distribution

Sort the 40 fixed reservoir residuals as
`v_(1) <= ... <= v_(N)`. For an attainable `(n, alpha)` cell, the exact
probability that the subset's requested order statistic occupies reservoir
position `k` is

`P(K=k) = C(k-1, r-1) * C(N-k, n-r) / C(N, n)`,

for `k=r,...,N-n+r`. Integer combination counts are retained exactly; floating
probabilities are derived only for reporting. Equal residual values are merged
by summing their position probabilities. Probability mass must sum to one
within a declared numerical tolerance, and the implementation must fail closed
otherwise.

For each supported quantile `q`, the fixed official evaluation scores determine
without resampling:

- endpoint coverage `mean(score_eval <= q)`;
- mean and median interval width (`2*q` before any aggregation, since intervals
  are symmetric and unbounded);
- normalized interval score at the cell's alpha and cap 125;
- severe-undercoverage indicator
  `coverage < (1 - alpha) - 0.10`.

Combining these deterministic values with the exact position probabilities
defines the complete conditional finite-reservoir distribution. No bootstrap
confidence interval is attached to this exact conditional distribution.

## Exchangeable references and primary estimand

Under idealized continuous exchangeable calibration and evaluation scores, the
latent calibration-conditional coverage is

`P ~ Beta(r, n_cal + 1 - r)`.

This is an analytic reference, not a claimed law for the fixed C-MAPSS
reservoir and official endpoint set. For each cell the ADR preregisters its
Beta mean, standard deviation, 5th and 10th percentiles, and probability below
the severe-undercoverage threshold.

The primary scientific discrepancy for each cell is the Kolmogorov--Smirnov
distance between the exact finite-reservoir empirical-coverage CDF and this
continuous Beta CDF. Mandatory secondary discrepancies are:

- signed mean-coverage difference;
- signed severe-undercoverage tail-probability difference;
- difference in coverage standard deviation;
- 1-Wasserstein distance, reported in coverage-probability units.

Because empirical coverage is observed on exactly 100 fixed endpoints, C11
must additionally report the Beta-binomial projection obtained by mixing
`Binomial(100, P)` over the same Beta law. Distances to this finite-evaluation
reference distinguish endpoint-grid effects from larger departures. Neither
reference licenses a causal attribution to engine dependence or observation
policy.

No p-value, model selection, cell selection, or pooled headline statistic is
used. All four alpha-0.10 cells are the primary family and are reported
separately; the two alpha-0.05 cells are sensitivity results.

## Analytic expectations fixed before execution

The Beta reference itself predicts non-monotone severe-undercoverage
probability across the primary sizes because the ceiling rank changes
discretely:

| `n_cal` | `alpha` | `r` | Beta mean | Beta SD | 5th percentile | 10th percentile | severe-undercoverage probability |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.10 | 10 | 0.9091 | 0.0830 | 0.7411 | 0.7943 | 0.1074 |
| 15 | 0.10 | 15 | 0.9375 | 0.0587 | 0.8190 | 0.8577 | 0.0352 |
| 20 | 0.10 | 19 | 0.9048 | 0.0626 | 0.7839 | 0.8190 | 0.0692 |
| 30 | 0.10 | 28 | 0.9032 | 0.0523 | 0.8047 | 0.8322 | 0.0442 |
| 20 | 0.05 | 20 | 0.9524 | 0.0454 | 0.8609 | 0.8913 | 0.0388 |
| 30 | 0.05 | 30 | 0.9677 | 0.0312 | 0.9050 | 0.9261 | 0.0076 |

These are analytic preregistration values, not project results. Reproducing the
non-monotone pattern is not a new empirical finding.

Sampling without replacement from `N=40` has the conventional finite-population
factor `(N-n)/(N-1)`: 0.7692, 0.6410, 0.5128, and 0.2564 for sizes 10, 15, 20,
and 30. This factor is retained as a reservoir-overlap diagnostic. It is not
asserted to be an exact multiplicative correction for the nonlinear quantile or
coverage variance. Raw cross-size dispersion is therefore not interpreted as a
causal calibration-size effect or compared as if draws were independent i.i.d.
samples from an infinite population.

## Observation-mechanism diagnostics

Calibration endpoints are policy-selected prefixes of complete FD001 training
trajectories; official test endpoints follow NASA's separate truncation
mechanism. C11 therefore does not assert exact exchangeability even for
FD001-to-FD001.

The future artifact must retain, without using these quantities for design
selection:

- reservoir cut-cycle and complete-lifetime distributions;
- official test observed-cycle distribution;
- a label-free distance between the reservoir cut-cycle and test endpoint-cycle
  distributions;
- the fraction of evaluation endpoints whose raw RUL exceeds cap 125;
- residual ties and the resulting reduction in distinct quantile support.

Departures from Beta may be described only as benchmark-specific descriptive
evidence consistent with engine dependence, a finite reservoir, endpoint-grid
effects, or observation-mechanism mismatch. C11 cannot identify which factor is
causal without an additional reviewed design.

## Required future configuration contract

After this ADR passes review, a separate implementation commit may add a
fail-closed configuration containing exactly the frozen values above. It must
reject unknown fields, extra targets or models, altered role sizes, changed
seeds, changed cut policy, changed cap, unsupported cells, numeric strings,
booleans, NaN, infinity, duplicate cells, and any unattainable cell marked for
evaluation.

The configuration must distinguish `predictor_seed`, `split_seed`, and
`cut_point_seed` even though all are frozen to 13. It must contain no
calibration-resample count because exact enumeration is primary. Any optional
Monte Carlo verification settings belong to tests, not the research config.

## Required future artifacts

An executed C11 run, if later authorized, must publish atomically and retain at
least:

- `manifest.json`: full Git and clean-state provenance, data/config/lock hashes,
  package environment, model and feature specification, data roles, artifact
  hashes, and the accepted ADR revision;
- `split_manifest.json`: all 60 predictor-fit and 40 reservoir engine IDs and
  their former calibration/validation roles;
- `reservoir_scores.csv`: one row per reservoir engine with cut point, complete
  lifetime, raw/capped truth, raw/clipped prediction, residual, and origin role;
- `evaluation_scores.csv`: one ordered official endpoint row per engine with
  observed cycles, raw/capped truth, raw/clipped prediction, residual, and cap
  saturation flag;
- `enumeration_cells.csv`: every declared and excluded cell, requested rank,
  attainability, regime, support-index range, combination count, and
  finite-population factor;
- `quantile_distribution.csv`: position, quantile, exact integer multiplicity,
  probability, exact induced coverage, width, and normalized interval score;
- `distribution_summary.csv`: finite-reservoir summaries, continuous Beta and
  beta-binomial references, preregistered discrepancy measures, tie diagnostics,
  and explicit status;
- `observation_mechanism.json`: the fixed descriptive diagnostics above.

No table may collapse an excluded cell into zero or omit it from a declared
denominator. CSV reconstruction must use `float_precision="round_trip"`.
Artifact publication must use the repository's immutable final publication
guard and remove temporary directories after failure.

## Required implementation tests before execution

The later implementation checkpoint must add behavioral tests for:

- exact 60/40 role assignment and complete/disjoint engine coverage;
- exclusion of reservoir engines from temporal and model fitting;
- one HGB/transformer fit and identity preservation throughout enumeration;
- fixed cut points and full-trajectory truth before prefix truncation;
- one residual per reservoir engine and one endpoint score per official engine;
- exact endpoint/RUL alignment and ordered IDs 1 through 100;
- prohibited feature and future-row isolation;
- the six-cell attainability and support table above;
- fail-closed handling of the two excluded alpha-0.05 cells and `n_cal=40`;
- exact PMF normalization, support bounds, integer multiplicities, and tie
  aggregation;
- exact-enumeration agreement with brute-force subset enumeration on small
  synthetic reservoirs;
- fixed-seed Monte Carlo convergence as a software check only;
- analytic Beta values and beta-binomial projection;
- expected non-monotone alpha-0.10 reference tail probabilities;
- all preregistered discrepancy calculations;
- strict adversarial configuration validation;
- deterministic repeated execution;
- complete artifact schema and hashes;
- immutable publication, overwrite rejection, and failed-write cleanup;
- independent summary reconstruction from retained score and distribution
  artifacts.

## Interpretation and stop rules

Allowed conclusions are conditional on the single frozen predictor, fixed
40-engine reservoir, fixed cut policy, and fixed official FD001 evaluation set.
Empirical subset probabilities are not deployment-failure probabilities.

If the finite-reservoir distribution is adequately explained by the continuous
Beta and finite-evaluation beta-binomial references, C11 reports that null
result and does not claim a new prognostics mechanism. If a material departure
remains, it is descriptive evidence requiring cautious interpretation against
the observation-mechanism and finite-reservoir diagnostics; it is not proof of
causation or universal conformal failure.

C12 remains blocked until C11 is implemented, executed, independently
reconstructed, and reviewed. C12-D, MALT, N-CMAPSS, new deep models, shift
repair methods, and dashboard/product work remain out of scope.

## Consequences

This design replaces an underpowered 50-draw fragility study with an exact
conditional finite-reservoir audit. It removes calibration-draw Monte Carlo
error and freezes cut points so engine-subset selection is the only conceptual
random variable. It also makes the known Beta/rank behavior a preregistered
reference rather than a result to be rediscovered.

The design does not yet authorize implementation or execution. A high-level
independent C11-A review must confirm the estimand, roles, attainable cells,
exact combinatorial distribution, references, observation diagnostics,
artifacts, tests, claim boundary, and stop rules.
