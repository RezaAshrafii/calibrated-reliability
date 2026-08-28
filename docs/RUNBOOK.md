# C-MAPSS execution runbook

## Preconditions

Run experiments only after all of the following succeed:

```bash
uv sync --locked --extra dev
uv run python scripts/verify_data.py --registry data/registry.yaml --data-root data/raw
uv run ruff check .
uv run ruff format --check .
uv run mypy --python-version 3.13 src
uv run mypy --no-site-packages --disable-error-code=untyped-decorator scripts
uv run pytest -q
git status --short
```

`git status --short` must produce no output. Raw NASA data remains under
`data/raw/` and must not be committed.

## Frozen experiment commands

```bash
uv run python scripts/run_c01_baseline.py --config configs/cmapss/fd001_baseline.yaml --registry data/registry.yaml
uv run python scripts/run_c02_conformal.py --config configs/cmapss/conformal.yaml --registry data/registry.yaml
uv run python scripts/run_c03_cqr.py --config configs/cmapss/cqr.yaml --registry data/registry.yaml
uv run python scripts/run_c04_shift_matrix.py --config configs/cmapss/shift_matrix.yaml --registry data/registry.yaml
uv run python scripts/run_c05_weighted_conformal.py --config configs/cmapss/weighted_conformal.yaml --registry data/registry.yaml
uv run python scripts/run_c06_sensitivity.py --config configs/cmapss/sensitivity.yaml --registry data/registry.yaml
uv run python scripts/run_c07_regime_scaling.py --config configs/cmapss/regime_scaling.yaml --registry data/registry.yaml
uv run python scripts/run_c08_adaptive_conformal.py --config configs/cmapss/adaptive_conformal.yaml --registry data/registry.yaml
```

Each runner validates its own frozen targets, seeds, alpha values, preprocessing
settings, and method-specific parameters. Do not edit a configuration to create
an undeclared sensitivity condition.

## Artifact handling

- Use a new output root for a deliberate rerun. Never replace an existing run.
- Preserve predictions, calibration scores, metrics, resolved configuration,
  split/cut-point manifests, logs, environment, Git SHA, and hashes together.
- Generated output trees are local and ignored by Git.
- Artifact existence means `EXECUTED`, not automatically `VERIFIED` or
  `REPORTED`.
- Historical C08 artifacts generated under `legacy_max_clamp` remain immutable.
  Future C08 runs must retain explicit rank, attainability, regime, and policy
  diagnostics.
- Do not run C11 or C12 from this runbook. Neither has passed its required design
  gate.

## Deterministic reporting

The canonical C01--C08 tables were built from a clean checkout with:

```bash
uv run python scripts/build_results.py --index docs/artifact_index.yaml --output-root reports/results
```

The destination must not already exist; the builder never overwrites a report.
For an independent byte-level reconstruction, use a new output root and compare
its file hashes with `reports/results/provenance.json`. The builder rejects an
unindexed output tree, a superseded or duplicate official selection, an
incomplete run, a dirty worktree, mixed Git SHAs within one official tree, and
every manifest or artifact hash mismatch.
