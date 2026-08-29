# Calibrated Predictive Reliability

Reproducible study of calibrated predictive reliability under structured
distribution shift using turbofan remaining-useful-life (RUL) prediction.

[![CI](https://github.com/RezaAshrafii/calibrated-reliability/actions/workflows/ci.yml/badge.svg)](https://github.com/RezaAshrafii/calibrated-reliability/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11.9-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> This is a reproducible research repository, not a claim of SOTA or universal safety guarantees. No result is reported until it is generated from a verified artifact.

## Status

**Implemented, executed, independently verified, and deterministically reported:** C01--C08. These runs
provide replication/context (C01--C04) and exploratory/appendix evidence
(C05--C08); they are not presented as new conformal methods.

**Current gate:** independent C11-A review of the proposed design in ADR-0012.
Gate D passed independent review; no C11 implementation or execution is yet
authorized. Gate N returned `REFRAME`: the surviving scientific scope is a narrow
application-level audit of engine-level conformal rank attainability and
resolution, compared with the exact Beta reference and interpreted alongside
the C-MAPSS observation mechanism.

**Not implemented:** C11 and C12. C11 now has a proposed design ADR that must
pass independent review before implementation. C12 is optional, oracle-only supporting
diagnostics and cannot precede C11. MALT, C12-D, N-CMAPSS expansion, new deep
models, and dashboard work are deferred outside the current core study.

## Research scope

The project evaluates point and interval predictions across declared C-MAPSS
operating-condition and fault-mode shifts. The completed experiments motivate
a narrower calibration-budget mechanism audit; they do not establish causal
effects, universal shift guarantees, or state-of-the-art performance. See
`docs/NOVELTY_MATRIX.md` and `docs/RELATED_WORK.md` for the claim boundary.

## Quickstart

```bash
uv sync --locked --python 3.11.9 --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy --python-version 3.11 src
uv run mypy --no-site-packages --disable-error-code=untyped-decorator scripts
uv run pytest -q
```

Raw data belongs in `data/raw/` and is intentionally ignored by Git. See
`docs/protocol.md`, `REPRODUCIBILITY.md`, and `docs/RUNBOOK.md` before running
experiments.

The repository contains code, protocols and test fixtures. Official C-MAPSS raw files are never committed; obtain them from the source recorded in `data/registry.yaml` and verify their hashes first.

The complete C01--C08 command set and artifact-handling rules are in
`docs/RUNBOOK.md`. Runners require a clean commit, validate frozen
configuration values and input hashes, and refuse to overwrite an existing
run directory.


## Design principles

- preregistered questions and explicit experiment registry;
- group-aware splits and train-only fitting to prevent leakage;
- immutable, provenance-tracked experiment artifacts;
- reported uncertainty and limitations, not only point estimates;
- no fabricated numbers, citations, or conclusions.

## Research deliverables

The protocol, active experiment registry, execution commands, claim boundary,
and reporting state live in [docs/protocol.md](docs/protocol.md),
[docs/experiment_registry.md](docs/experiment_registry.md),
[docs/RUNBOOK.md](docs/RUNBOOK.md),
[docs/NOVELTY_MATRIX.md](docs/NOVELTY_MATRIX.md), and
[docs/RESULTS.md](docs/RESULTS.md). The tracked Gate D builder validates the
official artifact index and reconstructs the C01--C08 tables without manual
transcription. C11 and C12 remain ineligible for numerical reporting.

## Citation

Citation metadata is provided in `CITATION.cff`. No DOI is claimed until an actual archive is created.
