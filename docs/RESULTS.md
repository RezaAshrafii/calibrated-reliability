# Verified Results

## Publication rule

This file may contain numbers only when each number can be traced to an immutable experiment artifact generated from a clean commit and verified input registry. Do not fill this document from notebook output, memory, or an untracked local file.

At the current repository revision, the research runners and tests are implemented, but the official C-MAPSS raw data and generated experiment artifacts are intentionally excluded from Git. Therefore no scientific result is claimed here yet.

## Reproduction

~~~bash
uv sync --locked --extra dev
uv run python scripts/verify_data.py --registry data/registry.yaml --data-root data/raw
uv run python scripts/run_c01_baseline.py \
  --config configs/cmapss/fd001_baseline.yaml \
  --registry data/registry.yaml
uv run python scripts/run_c02_conformal.py \
  --config configs/cmapss/conformal.yaml \
  --registry data/registry.yaml
~~~

The runners refuse dirty worktrees, undeclared seeds, invalid input hashes and overwritten run directories.

## C01 — point baselines

| Model | Seed aggregation | RMSE | MAE | NASA score | 95% CI | Artifact |
|---|---:|---:|---:|---:|---|---|
| Mean | pending | pending | pending | pending | pending | pending |
| Ridge | pending | pending | pending | pending | pending | pending |
| HistGradientBoosting | pending | pending | pending | pending | pending | pending |

## C02 — split conformal

| Alpha | Coverage | Width | Normalized interval score | 95% CI | Artifact |
|---:|---:|---:|---:|---|---|
| 0.10 | pending | pending | pending | pending | pending |
| 0.05 | pending | pending | pending | pending | pending |

## Shift matrix

| Train → test | Shift interpretation | Point metrics | Coverage | Width | Status |
|---|---|---|---|---|---|
| FD001 → FD001 | In-distribution | pending | pending | pending | Planned |
| FD001 → FD002 | Operating-condition | pending | pending | pending | Planned |
| FD001 → FD003 | Fault-mode/structural | pending | pending | pending | Planned |
| FD001 → FD004 | Compound/structural | pending | pending | pending | Planned |

## Audit checklist

- [ ] `verify_data.py` passed against the registered hashes.
- [ ] All declared seeds completed.
- [ ] The worktree was clean at run time.
- [ ] Configuration, lockfile, git SHA and environment were retained.
- [ ] Metrics were independently regenerated from stored predictions.
- [ ] Negative findings and limitations were recorded.
- [ ] No raw data, secret or personal path was committed.
