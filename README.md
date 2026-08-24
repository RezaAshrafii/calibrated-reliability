# Calibrated Predictive Reliability

Reproducible study of calibrated predictive reliability under structured distribution shift, using turbofan remaining-useful-life (RUL) prediction and an agentic monitoring case study.

## Status

**Phase 0–3: protocol, data registry, and validated C-MAPSS loaders.** Modeling and conformal phases are intentionally not implemented yet.

## Research scope

The project evaluates point and interval predictions across C-MAPSS operating/fault shifts, then studies grouped evaluation and conformal prediction for MALT agent transcripts. Claims are limited to reproducible empirical evaluation; this is not a claim of SOTA, causal safety, or universal guarantees.

## Quickstart

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
```

Raw data belongs in `data/raw/` and is intentionally ignored by Git. See `docs/protocol.md` and `REPRODUCIBILITY.md` before running experiments.

## Design principles

- preregistered questions and explicit experiment registry;
- group-aware splits and train-only fitting to prevent leakage;
- immutable, provenance-tracked experiment artifacts;
- reported uncertainty and limitations, not only point estimates;
- no fabricated numbers, citations, or conclusions.

## Citation

Citation metadata is provided in `CITATION.cff`. No DOI is claimed until an actual archive is created.
