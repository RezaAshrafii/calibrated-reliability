# Repository agent contract

This repository is a reproducible research project, not a demo application.

- Read `docs/protocol.md` before changing research logic.
- Keep raw data, secrets, tokens, and disposable generated outputs out of Git. Curated report figures may be tracked only when their generating script and train-only provenance are committed.
- Preserve engine-level C-MAPSS separation. MALT is deferred outside the core
  study; if a future reviewed study revives it, preserve task-family-level
  separation as required by ADR-0002.
- Fit preprocessing only on training data.
- Treat feature schemas as fail-closed contracts: labels and unknown columns must never be accepted by prefix matching.
- Never invent results, citations, or test output.
- Add a focused test for every new public behavior.
- Run `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy --python-version 3.11 src`, `uv run mypy --no-site-packages --disable-error-code=untyped-decorator scripts`, and `uv run pytest -q` before handoff.
- Do not weaken tests to make an implementation pass.
- Use the tracked Python 3.11.9 interpreter for Gate D reconstruction; do not
  publish reports from another supported or locally convenient interpreter.
- Do not implement or execute C11 before its dedicated ADR passes independent
  C11-A review. Treat any future C12 target calibration as an oracle diagnostic,
  never a deployment method.
