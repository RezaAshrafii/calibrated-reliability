# Novelty gate: evidence matrix and execution decision

## Status

- Gate: N (novelty before C11/C12)
- Gate N search cut-off: 2026-08-28
- Final adjacent-source update: 2026-09-01 (not a systematic re-search)
- Decision: **REFRAME**
- Scope: C-MAPSS RUL interval reliability, conformal calibration under shift,
  calibration-size effects, and the proposed C11/C12 extensions
- Evidence standard: absence from a finite search is not proof of novelty. Every
  negative statement below means only "not found in the searched corpus by the
  cut-off date."

No C11 or C12 implementation or artifact was produced during Gate N. Later
C11 work followed its separately reviewed design, implementation-readiness,
execution, artifact, and reporting gates.

## Decision in one paragraph

The project must not claim novelty for conformal RUL prediction, CQR on
C-MAPSS, multi-subset evaluation, adaptive or regime-aware conformal RUL,
coverage degradation under shift, target-domain conformal repair, or
calibration-allocation sweeps. Those areas are occupied by peer-reviewed
papers, preprints, and a close 2026 code-and-artifact deposit. The narrow
surviving question is whether the standard prognostics practice of treating
engines as calibration units creates an operationally important conformal
rank-attainability and resolution constraint, and whether its empirical
behavior departs from the exact exchangeable `Beta(r, n_cal + 1 - r)`
reference because of engine dependence or the C-MAPSS observation mechanism.
That is a domain-specific mechanism audit, not a new conformal theorem.

## Stop-rule resolution

| Stop rule | Evidence | Decision |
|---|---|---|
| Diao et al. use disjoint target-domain calibration and report across-calibration-draw variability on C-MAPSS | The full text calls the calibration set "independent", but also says the same small labeled target-domain set is used for fine-tuning and building calibration data. Unit-level disjointness between those uses is not specified. Calibration windows are randomly truncated from target-domain training trajectories. Reported mean and standard deviation are over three seeds intended to cover random initialization and data shuffling, not an isolated repeated-calibration-draw estimand. | The literal stop condition is **not met**. Nevertheless, target-domain CQR repair under the same cross-condition directions is occupied. C12 may continue only as an explicitly labelled oracle/supporting diagnostic with observation and cap controls; it is not a novelty claim. |
| A prognostics work states the finite order-statistic attainability boundary in engine/calibration-count terms | The closest deposit, Yan (2026), sweeps calibration allocation and discusses fixed calibration budgets, but its helper silently clamps `ceil((n+1)(1-alpha))` to `n`. Its inspected code and documentation do not state `n_cal >= ceil(1/alpha)-1`, do not classify unattainable ranks, and do not report their frequency by engine budget. No other searched prognostics source did so. | The whole track does **not** stop. The surviving contribution is narrowed to this application-level diagnostic and its observation-mechanism interpretation. |

## Search protocol

All searches were run on 2026-08-28. Domain-constrained web search was used
when a database did not expose a usable anonymous API. This distinction is
recorded because a domain-constrained web result is weaker evidence than a
complete native-database export.

On 2026-09-01, a final targeted adjacent-source update inspected an FD001
engine-level conformal teaching resource and an arXiv bearing-RUL
conditional-reliability study. The update did not repeat every original query,
so all original negative statements remain bounded to the 2026-08-28 Gate N
corpus.

| System | Exact queries or endpoint | Access result | Evidence retained |
|---|---|---|---|
| Google Scholar | `conformal prediction remaining useful life C-MAPSS calibration size`; `conformal prediction prognostics calibration set size attainability`; `adaptive conformal inference turbofan RUL` | Domain-constrained web search of `scholar.google.com`; native result export was not available. Results were sparse and did not add an exact match. | Search limitation, not evidence of absence. |
| Semantic Scholar | Graph API paper search with the same three queries | HTTP 429 rate limiting for all calls. | No substantive evidence; failure is declared. |
| IEEE Xplore | `site:ieeexplore.ieee.org "conformal prediction" "remaining useful life"`; `site:ieeexplore.ieee.org "C-MAPSS" conformal prediction`; `site:ieeexplore.ieee.org prognostics conformal calibration` | Domain-constrained search. | Metadata for Hattar and Vincent (2025); no full text obtained. |
| PHM Society / IJPHM | `site:papers.phmsociety.org conformal prediction remaining useful life`; `site:papers.phmsociety.org C-MAPSS conformal`; `site:papers.phmsociety.org prognostics "calibration set" conformal` | Publisher pages and available full-text links inspected. | Javanmardi and Hullermeier (2023), Robinson (2026), and censored-RUL work. |
| OpenReview | `site:openreview.net "training-conditional coverage" "covariate shift"`; `site:openreview.net conformal "remaining useful life"`; `site:openreview.net conformal calibration set size coverage` | Primary OpenReview records/PDFs inspected where relevant. | Pournaderi and Xiang; no prognostics attainability match. |
| ScienceDirect / SpringerLink / Wiley | `site:sciencedirect.com conformal prediction remaining useful life C-MAPSS`; `site:link.springer.com conformal prediction remaining useful life prognostics`; `site:onlinelibrary.wiley.com conformal prediction remaining useful life`; `site:sciencedirect.com conformal prediction prognostics calibration set size` | Publisher/domain-constrained search. | Xu et al. (2026) and adjacent reliability work; no engine-level rank-boundary report found. |
| Zenodo | `site:zenodo.org conformal prediction remaining useful life C-MAPSS`; `site:zenodo.org conformal prognostics calibration set size` | Metadata plus the complete 12 MB software archive inspected. Archive MD5 matched `8a45f77d2166a25088833e288d6c68f6`. | Yan (2026), the closest code-and-artifact competitor. |
| GitHub | `site:github.com conformal prediction RUL C-MAPSS`; `site:github.com conformal calibration prognostics turbofan` | Domain-constrained search and relevant repository descriptions. | Implementation precedents only. GitHub absence is explicitly not treated as novelty evidence. |
| arXiv | Exact identifiers plus searches for conformal calibration scarcity, Beta coverage, RUL, shift, and attainability | Primary abstract/full-text records inspected. | Aadithiyan (2026), Zwart (2025, 2026), Sanchez-Dominguez et al. (2025), Bian and Barber, and related shift work. |
| PMC / publisher full text | DOI `10.3390/s26072249`; PMC `PMC13075270` | Full article read through PMC and the NCBI BioC record. | Diao et al. design, calibration source, truncation policy, and seed interpretation. |

## Closest-work matrix

Evidence levels are `metadata`, `abstract`, `full text`, and
`code + artifacts`. The matrix deliberately separates overlap from what was
not established.

| ID | Work and evidence | Confirmed overlap | Not established by the inspected evidence | Required treatment |
|---|---|---|---|---|
| W1 | Diao et al. (2026), *Sensors*, [DOI 10.3390/s26072249](https://doi.org/10.3390/s26072249), **full text** | C-MAPSS; FD001/FD002 same- and cross-condition transfer; target-domain supervised fine-tuning; target-domain CQR calibration from randomly truncated training trajectories; three-seed summaries | Isolated across-calibration-draw variability; clear unit-level separation of fine-tuning and calibration; engine-level rank attainability | Primary precedent for C03/C12. C12 cannot be framed as a novel repair method. |
| W2 | Aadithiyan (2026), [arXiv:2608.21591](https://arxiv.org/abs/2608.21591), **full abstract** | 200 random calibration sets; ACI; data scarcity; upper-tail support/diversity explains coverage variability | Prognostics, C-MAPSS, engine-level attainability boundary | Pre-empts a generic "calibration size drives ACI" claim. C11 must test a narrower prognostics mechanism. |
| W3 | Zwart (2025), [arXiv:2509.15349](https://arxiv.org/abs/2509.15349), and Zwart (2026), [arXiv:2602.18045](https://arxiv.org/abs/2602.18045), **full text/abstract** | Exact finite-sample Beta/rank law, small-sample calibration-conditional variability, attainable calibration-grid language and correction | Prognostics application; C-MAPSS observation mechanism; engine-level empirical diagnostic | Canonical recent theory precedent. C11 is not a new theorem or a discovery of the Beta law. |
| W4 | Sanchez-Dominguez et al. (2025), [arXiv:2512.04566](https://arxiv.org/abs/2512.04566), **full abstract** | Small calibration sets can have large coverage dispersion; alternative single-predictor guarantee for engineering surrogates | C-MAPSS; engine-level calibration; adaptive conformal | Cite as small-data engineering precedent. |
| W5 | Javanmardi and Hullermeier (2023), [DOI 10.36001/ijphm.2023.v14i2.3417](https://doi.org/10.36001/ijphm.2023.v14i2.3417), **publisher page/full text** | Conformal RUL intervals; deep CNN and gradient boosting; C-MAPSS | Structured source-to-target shift matrix; attainability mechanism | Establishes C01/C02 as replication/context, not novelty. |
| W6 | Onilede (2026), [TechRxiv DOI](https://doi.org/10.36227/techrxiv.177220374.46388084/v1), **full abstract** | Split conformal and CQR on C-MAPSS FD001 | Structured shift; repeated calibration draws; attainability | High overlap with C02/C03. |
| W7 | Hattar and Vincent (2025), [DOI 10.1109/ISPCE64260.2025.11044907](https://doi.org/10.1109/ISPCE64260.2025.11044907), **metadata/title** | Multi-model C-MAPSS RUL with conformal prediction | Detailed split, shift, calibration-size, and rank behavior were not verified without full text | Include conservatively; do not infer unobserved methods. |
| W8 | Pournaderi and Xiang, [arXiv:2405.16594](https://arxiv.org/abs/2405.16594), **full abstract/OpenReview record** | Training-conditional coverage under covariate shift | Prognostics-specific observation mechanism and artifacts | Theory/shift precedent for interpretation, not an experimental duplicate. |
| W9 | Bian and Barber, [arXiv:2205.03647](https://arxiv.org/abs/2205.03647), **full abstract** | Training-conditional coverage framing for distribution-free prediction; split conformal precedent | Prognostics application | Foundational reference for what a realized calibration set means. |
| W10 | Robinson (2026), [DOI 10.36001/ijphm.2026.v17i1.4724](https://doi.org/10.36001/ijphm.2026.v17i1.4724), **publisher page/abstract** | Risk-aware asymmetric CQR on all four C-MAPSS subsets | Structured frozen source-to-target shift; rank attainability | Further establishes C03 and broad four-subset conformal RUL as occupied. |
| W11 | Yan (2026), [Zenodo 10.5281/zenodo.21330745](https://doi.org/10.5281/zenodo.21330745), **code + artifacts** | C-MAPSS FD001-FD004 and N-CMAPSS; split, CV+, ACI, regime ACI, Mondrian, CQR; multiple predictors; calibration-fraction and regime-count sweeps; per-engine artifacts | Explicit finite-rank attainability boundary; unattainable-rank frequency; fail-closed or infinite-rank policy. The inspected helper clamps the requested index to the largest observed score | Closest empirical competitor. It pre-empts broad ACI/regime/calibration-budget claims but does not trigger the exact attainability stop rule. |
| W12 | Xu et al. (2026), [DOI 10.1016/j.ress.2026.112763](https://doi.org/10.1016/j.ress.2026.112763), **publisher abstract/highlights** | RUL uncertainty quantification plus split conformal on C-MAPSS and batteries | Engine-level rank boundary and C-MAPSS calibration mechanism | Adjacent recent RUL-calibration work; do not overstate details not available in the inspected abstract. |
| W13 | Same-template shift studies, [arXiv:2605.18008](https://arxiv.org/abs/2605.18008), [arXiv:2607.17405](https://arxiv.org/abs/2607.17405), [arXiv:2603.24475](https://arxiv.org/abs/2603.24475), and [arXiv:2510.05566](https://arxiv.org/abs/2510.05566), **abstract-level** | External/structured shift audits followed by post-hoc conformal assessment or repair in blood pressure, transcriptomics, batteries, and language models | Turbofan-specific observation and engine-budget mechanism | Establishes that the broad "audit under shift then repair" template is not novel. |
| W14 | [Lospinoso (2026)](https://lospino.so/statistics/conformal-prediction/), **teaching page** | FD001 as a teaching dataset; engines as calibration units; finite-rank augmented-infinity edge | Peer-reviewed prognostics mechanism evidence; C-MAPSS finite-reservoir audit | Pedagogical adjacency only; do not treat it as proof against the narrow C11 mechanism contribution. |
| W15 | Yang, Wang, and Wang (2026), [arXiv:2607.08273](https://arxiv.org/abs/2607.08273), **abstract-level** | RUL reliability under operating-regime shift; empirical calibration and conditional diagnostics on bearings | C-MAPSS finite-reservoir rank accounting; exact Beta comparison | Adjacent reliability-evaluation template; do not claim the broad protocol is unique. |

## Claim-by-claim decision

| Candidate claim | Decision | Reason |
|---|---|---|
| Conformal prediction for turbofan RUL | **DROP as novelty** | W5, W7, W10 and W11 already cover it. |
| CQR for C-MAPSS RUL | **DROP as novelty** | W1, W6 and W10 directly overlap. |
| C-MAPSS evaluation across FD001-FD004 | **DROP as novelty** | W5, W10 and W11 cover multiple/all subsets. |
| ACI or regime-aware ACI for turbofan RUL | **DROP as novelty** | W11 directly implements both. |
| Coverage degradation under operating-condition shift | **DROP as novelty** | W1 and W11 directly study cross-condition reliability. |
| Target-domain conformal recalibration repairs shift | **DROP as novelty** | W1 performs target-domain CQR calibration; the broader template is common. |
| Calibration allocation matters more than adaptation | **DROP** | Too broad and partly occupied by W2/W11; current evidence cannot support a causal ranking. |
| Exact Beta/rank law for a realized conformal calibration set | **DROP as novelty** | W3/W4/W9 are direct theory precedents. |
| Engine-level finite-rank attainability/resolution diagnostic on C-MAPSS, including observed unattainable-rank frequencies | **KEEP, narrowly** | Not found in the searched prognostics corpus. It is an application-level mechanism audit, not new theory. |
| Quantified mismatch between C-MAPSS calibration-prefix and official-test observation mechanisms | **KEEP, narrowly** | Not found in the inspected close works. It must be presented as a benchmark-specific diagnostic, not a universal explanation. |
| Provenance-hashed, fail-closed research implementation | **KEEP as engineering contribution** | Useful and differentiating, but not a statistical novelty claim. |

## Execution consequences

1. **C01-C04:** retain as replication and empirical context.
2. **C05-C08:** retain as exploratory/appendix evidence. C08 is the anchor
   diagnostic that motivated the attainability question, not evidence that ACI
   generally fails under shift.
3. **C11:** may proceed only after a reviewed design ADR. Its primary estimand
   must be deviation from the exact Beta reference plus rank
   attainability/resolution at engine-level budgets. Use one frozen HGB model
   for the primary analysis; do not headline raw calibration-draw variance.
4. **C12:** if retained, implement Conditions A/B only and label target-domain
   calibration as `ORACLE / DIAGNOSTIC - NOT A DEPLOYMENT METHOD`. It is a
   supporting contrast, not a method contribution. Observation-policy and RUL
   cap strata must be controlled before interpretation.
5. **Do not execute:** C12-D, MALT, N-CMAPSS expansion, additional deep models,
   dashboards, or any new shift-repair method during the core study.

## Gate N verdict

**REFRAME.** The exact combined implementation is not the novelty. The
defensible core is the application-level link among engine-level calibration
budgets, finite conformal rank attainability/resolution, the exact Beta
reference, and the C-MAPSS observation mechanism. This verdict permits design
work for the narrowed C11 and oracle-only C12, but it does not authorize their
execution without their own reviewed ADR gates.
