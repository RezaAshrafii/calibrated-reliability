# Related work and claim boundaries

## Purpose

This document defines how the repository must position its completed C01-C08
and C11 experiments and any future C12 work. It is a claim-control document, not a
systematic-review claim. The search protocol, evidence levels, and stop-rule
decisions are recorded in `docs/NOVELTY_MATRIX.md`.

## Conformal intervals for remaining useful life

Conformal RUL prediction on C-MAPSS is established prior work. Javanmardi and
Hullermeier applied several conformal algorithms to deep-convolutional and
gradient-boosting RUL predictors on C-MAPSS in 2023. More recent work includes
Onilede's split-conformal/CQR study on FD001, Hattar and Vincent's multi-model
conformal turbofan study, and Robinson's risk-aware asymmetric CQR framework
across all four C-MAPSS subsets. Consequently, C01-C03 in this repository are
replication, verification, and implementation context. They are not evidence
that this project introduced conformal RUL intervals or CQR for turbofan data.

Primary precedents:

- [Javanmardi and Hullermeier (2023)](https://doi.org/10.36001/ijphm.2023.v14i2.3417)
- [Onilede (2026)](https://doi.org/10.36227/techrxiv.177220374.46388084/v1)
- [Hattar and Vincent (2025)](https://doi.org/10.1109/ISPCE64260.2025.11044907)
- [Robinson (2026)](https://doi.org/10.36001/ijphm.2026.v17i1.4724)

## Conformal RUL under operating-condition shift

Diao et al. study same-condition and bidirectional FD001/FD002 transfer using
an LSTM quantile model, supervised target-domain fine-tuning, and target-domain
CQR calibration. Their calibration windows are randomly truncated from
target-domain training trajectories. The paper describes an independent
calibration set, but also says that the same small labelled target-domain set
is used for fine-tuning and calibration; the unit-level separation of those
roles is therefore not fully specified in the inspected text. Reported
variability is mean and standard deviation over three seeds intended to cover
random initialization and data shuffling, not a controlled distribution over
calibration draws.

This precedent means that target-domain conformal repair on C-MAPSS is not a
novel method claim. A future C12 comparison can be scientifically useful only
as a controlled oracle diagnostic that holds the predictor fixed, prevents
test-label leakage, aligns observation mechanisms, and stratifies RUL-cap
saturation. It must cite Diao et al. and must not be presented as deployable
unlabelled adaptation.

- [Diao et al. (2026)](https://doi.org/10.3390/s26072249)

## Adaptive and regime-aware conformal prognostics

The closest empirical competitor is Yan's 2026 Zenodo software and artifact
deposit. It covers C-MAPSS FD001-FD004 and N-CMAPSS DS02 with split conformal,
engine-stratified CV+, ACI, regime-conditional ACI, Mondrian variants, CQR,
multiple base predictors, calibration-allocation sweeps, regime-count sweeps,
and per-engine outputs. The downloaded archive matched the published MD5 and
was inspected at code-and-artifact level.

That deposit occupies broad claims about applying ACI, regime-aware conformal
methods, learned operating regimes, and calibration-allocation experiments to
turbofan RUL. Its inspected quantile helper computes the usual corrected rank
and then clamps it to the available score range. The repository documents
small calibration pools and fixed calibration budgets, but it does not state
the finite observed-rank attainability condition in engine-count terms or
report the frequency with which ACI requests an unattainable finite rank.

- [Yan (2026), Zenodo 10.5281/zenodo.21330745](https://doi.org/10.5281/zenodo.21330745)

The consequence for this repository is strict: C05-C08 are exploratory or
appendix material. They support the motivation for a mechanism audit, but they
are not the main novelty.

## Small calibration sets and calibration-conditional coverage

The exact finite-sample behavior of a realized split-conformal calibration set
is established theory. Bian and Barber distinguish marginal from
training-conditional coverage. Zwart uses the exact Beta/rank law to study and
correct small-sample calibration-conditional behavior, including attainable
grid points. Sanchez-Dominguez et al. likewise address the potentially large
coverage dispersion of conformal predictors built from small calibration
sets. Aadithiyan performs a 200-calibration-set ACI ablation outside
prognostics and finds that upper-tail support/diversity, rather than a
rare-event count alone, explains much of the coverage variability.

Relevant references:

- [Bian and Barber](https://arxiv.org/abs/2205.03647)
- [Zwart (2025)](https://arxiv.org/abs/2509.15349)
- [Zwart (2026)](https://arxiv.org/abs/2602.18045)
- [Sanchez-Dominguez et al. (2025)](https://arxiv.org/abs/2512.04566)
- [Aadithiyan (2026)](https://arxiv.org/abs/2608.21591)

Therefore this project must not claim to discover the Beta law, small-sample
coverage variability, calibration-grid discreteness, or the general
importance of calibration support. The admissible C11 question is narrower:

> When whole engines are the exchangeability and calibration units, how do
> finite conformal rank attainability and resolution constrain the intervals
> produced by the frozen C-MAPSS pipeline, and how does the empirical
> calibration-draw distribution compare with the exact exchangeable Beta
> reference under the benchmark's observation mechanism?

This is a prognostics application and mechanism audit. It is not a new
conformal theorem.

## Covariate shift and the limit of ordinary guarantees

Training-conditional guarantees under covariate shift and weighted conformal
methods have a substantial theory literature. Pournaderi and Xiang provide a
direct recent reference. The project's C04-C08 results must therefore be
reported as benchmark evidence under declared assumptions, not as universal
shift-coverage guarantees.

- [Pournaderi and Xiang](https://arxiv.org/abs/2405.16594)

The same broad experimental template--audit uncertainty under external or
structured shift and then assess a post-hoc repair--also appears in multiple
recent application domains. Domain transfer by itself is not the scientific
novelty.

## Remaining contribution boundary

The literature gate leaves three scoped contributions:

1. **Scientific/application:** an engine-level C-MAPSS audit of finite
   conformal rank attainability and resolution, benchmarked against the exact
   Beta reference.
2. **Scientific/application:** a label-free quantification of the mismatch
   between policy-truncated calibration prefixes and official C-MAPSS test
   observation endpoints, used as a possible explanation to test rather than
   a causal conclusion.
3. **Engineering:** a fail-closed, provenance-hashed, immutable-artifact
   implementation that makes calibration unit, rank, policy, and assumptions
   auditable.

No claim of universal novelty is made. The literature search is finite and has
declared access limitations, especially Semantic Scholar rate limiting and the
lack of a native Google Scholar export.

## Mandatory language for future reporting

Allowed:

- "In the searched corpus through 2026-08-28, we did not find a prognostics
  study that states and audits the finite observed-rank attainability boundary
  in engine-level calibration-count terms."
- "The result is specific to the frozen C-MAPSS design and declared conformal
  policy."
- "C12 uses target labels as an oracle diagnostic and is not a deployment
  method."
- "Observed departures from the Beta reference are descriptive evidence to be
  interpreted against engine dependence and observation-mechanism mismatch."

Not allowed:

- "First conformal RUL method" or "first CQR method for C-MAPSS."
- "First ACI/regime-aware conformal study for turbofan engines."
- "ACI fails under distribution shift."
- "Calibration budget is universally more important than adaptation."
- "Target-domain conformal recalibration is a novel repair."
- "The C-MAPSS result proves a universal limitation of conformal prediction."
- Any ordinary exchangeability guarantee for the structured-shift experiments.

## Execution position

- C11 design, implementation-readiness, artifact-level, and report-level
  reviews passed before the single authorized execution from clean commit
  `cba16d0`. C11 is now `IMPLEMENTED / EXECUTED / VERIFIED / REPORTED`; its
  narrow claim boundary remains unchanged.
- C12 is secondary and conditional. It cannot precede the C11 mechanism gate,
  and it requires observation-alignment and cap-saturation controls.
- C12-D, MALT, N-CMAPSS expansion, new deep models, and product/dashboard work
  remain outside the current core.

## Post-Gate-N adjacent-source update

The original Gate N corpus ended on 2026-08-28. A final adjacent-source check
on 2026-09-01 did not change the `REFRAME` decision, but it adds two useful
boundaries. They are not evidence of universal absence or a systematic-review
update.

- [Lospinoso (2026), *Conformal Prediction*](https://lospino.so/statistics/conformal-prediction/)
  is a teaching resource that uses FD001 to explain engine-level calibration,
  finite rank, and the augmented-infinity convention. It is not peer-reviewed
  prognostics evidence, but it further rules out presenting engine units or
  the finite-rank edge as a novel pedagogical observation.
- [Yang, Wang, and Wang (2026)](https://arxiv.org/abs/2607.08273) study
  empirical calibration and conditional-reliability diagnostics for bearing
  RUL under operating-regime shift. It is an adjacent benchmark/protocol
  precedent, not a duplicate C-MAPSS finite-reservoir study.
