# Calibrated Predictive Reliability

Reproducible study of calibrated predictive reliability under structured distribution shift, using turbofan remaining-useful-life (RUL) prediction and an agentic monitoring case study.

## Status

**Phase 0–6 complete; Phase 7 implements the C02 split-conformal runner.** Shift-matrix and MALT phases remain intentionally unimplemented.

## Research scope

The project evaluates point and interval predictions across C-MAPSS operating/fault shifts, then studies grouped evaluation and conformal prediction for MALT agent transcripts. Claims are limited to reproducible empirical evaluation; this is not a claim of SOTA, causal safety, or universal guarantees.

## Quickstart

```bash
uv sync --locked --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run mypy --no-site-packages --disable-error-code=untyped-decorator scripts
uv run pytest -q
```

Raw data belongs in `data/raw/` and is intentionally ignored by Git. See `docs/protocol.md` and `REPRODUCIBILITY.md` before running experiments.

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

## Citation

Citation metadata is provided in `CITATION.cff`. No DOI is claimed until an actual archive is created.
