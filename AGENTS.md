# Repository agent contract

This repository is a reproducible research project, not a demo application.

- Read `docs/protocol.md` before changing research logic.
- Keep raw data, secrets, tokens, and generated outputs out of Git.
- Preserve engine-level (C-MAPSS) and task-family-level (MALT) separation.
- Fit preprocessing only on training data.
- Never invent results, citations, or test output.
- Add a focused test for every new public behavior.
- Run `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src`, and `uv run pytest -q` before handoff.
- Do not weaken tests to make an implementation pass.

