# C-MAPSS execution runbook

## Preconditions

Run experiments only after all of the following succeed:

```bash
uv sync --locked --python 3.11.9 --extra dev
uv run python scripts/verify_data.py --registry data/registry.yaml --data-root data/raw
uv run ruff check .
uv run ruff format --check .
uv run mypy --python-version 3.11 src
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
- The single authorized C11 execution is recorded in `docs/C11_EXECUTION.md`.
  Preserve that output unchanged and do not overwrite or silently rerun it.
  Its artifact and report passed independent reconstruction; C11 is now
  `VERIFIED` and `REPORTED`. C12 is not implemented and has no authorized
  command.

## Deterministic reporting

The canonical C01--C08 tables were built from a clean checkout with:

```bash
uv run python scripts/build_results.py --index docs/artifact_index.yaml --output-root reports/results
```

The destination must not already exist; the builder never overwrites a report.
For a full byte-level reconstruction, first reproduce the clean
`builder_git_sha` and exact `environment` recorded in
`reports/results/provenance.json`, including Python 3.11.9, platform, direct
package versions, `.python-version` hash, and `uv.lock` hash. Use a new output
root and compare every generated file with `reports/results/checksums.sha256`.
A later clean commit or different environment must record its own provenance;
last-bit floating-point serialization is not promised across environments. The
builder rejects an unindexed output tree, a superseded or duplicate official
selection, an incomplete or substituted run matrix, a dirty worktree, mixed Git
SHAs within one official tree, a manifest-set digest mismatch, nested manifest,
path traversal, and every manifest or artifact hash mismatch. Git state,
environment, and official artifacts are revalidated immediately before final
publication.

The canonical C11 publication is under `reports/c11/` and must not be
overwritten. Exact reconstruction of all C11 report files requires the recorded
clean builder commit `66180029e55a2b05b3b9495ed87a50318038d712` and its exact
environment. Because that historical commit predates the LF rules for three
C11 provenance inputs, create a Windows historical checkout with
`git -c core.autocrlf=false worktree add --detach <path> 66180029e55a2b05b3b9495ed87a50318038d712`.
Build into a fresh external destination, expose the existing immutable C11
artifact read-only, and compare every generated file with
`reports/c11/checksums.sha256`. Current commits force LF for the C11 config,
registry, ADR, index, and reports through `.gitattributes`.
