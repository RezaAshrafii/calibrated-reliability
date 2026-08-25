# Calibrated Predictive Reliability

Reproducible study of calibrated predictive reliability under structured distribution shift, using turbofan remaining-useful-life (RUL) prediction and an agentic monitoring case study.

[![CI](https://github.com/RezaAshrafii/calibrated-reliability/actions/workflows/ci.yml/badge.svg)](https://github.com/RezaAshrafii/calibrated-reliability/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> This is a reproducible research repository, not a claim of SOTA or universal safety guarantees. No result is reported until it is generated from a verified artifact.

## Status

**Implemented:** C01 point-prediction baselines, leakage-safe temporal features, engine-aware splits, C02 split-conformal intervals, immutable provenance-tracked artifacts, and the CI quality gate.

**In progress:** C-MAPSS shift matrix, CQR comparison, sensitivity analysis, and the MALT study. These are intentionally not presented as implemented results.

## Research scope

The project evaluates point and interval predictions across C-MAPSS operating/fault shifts, then studies grouped evaluation and conformal prediction for MALT agent transcripts. Claims are limited to reproducible empirical evaluation; this is not a claim of SOTA, causal safety, or universal guarantees.

## Quickstart

```bash
uv sync --locked --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy --python-version 3.13 src
uv run mypy --no-site-packages --disable-error-code=untyped-decorator scripts
uv run pytest -q
```

Raw data belongs in `data/raw/` and is intentionally ignored by Git. See `docs/protocol.md` and `REPRODUCIBILITY.md` before running experiments.

The repository contains code, protocols and test fixtures. Official C-MAPSS raw files are never committed; obtain them from the source recorded in `data/registry.yaml` and verify their hashes first.

After data verification and from a clean commit, run the complete C01 seed set with:

```bash
uv run python scripts/run_c01_baseline.py --config configs/cmapss/fd001_baseline.yaml --registry data/registry.yaml
```

Run the complete C02 split-conformal seed set from a clean commit with:

```bash
uv run python scripts/run_c02_conformal.py --config configs/cmapss/conformal.yaml --registry data/registry.yaml
```

## Design principles

- preregistered questions and explicit experiment registry;
- group-aware splits and train-only fitting to prevent leakage;
- immutable, provenance-tracked experiment artifacts;
- reported uncertainty and limitations, not only point estimates;
- no fabricated numbers, citations, or conclusions.

## Research deliverables

The experiment protocol and planned output format live in [docs/protocol.md](docs/protocol.md), [docs/experiment_registry.md](docs/experiment_registry.md), and [docs/RESULTS.md](docs/RESULTS.md). The results document is deliberately a template until C01/C02 artifacts are generated and independently checked.

## Citation

Citation metadata is provided in `CITATION.cff`. No DOI is claimed until an actual archive is created.
