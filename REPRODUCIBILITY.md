# Reproducibility

The repository is designed so data, configuration, code revision, environment, split manifests, and generated metrics can be traced together.

1. Install Python 3.11+ and `uv`.
2. Place officially sourced C-MAPSS files under `data/raw/`; never commit them.
3. Verify the registered files with `uv run python scripts/verify_data.py --registry data/registry.yaml --data-root data/raw`.
4. Run the locked environment and test suite before experiments.
5. Commit all source changes so the worktree is clean.
6. Run all preregistered C01 seeds with:

   ```bash
   uv run python scripts/run_c01_baseline.py --config configs/cmapss/fd001_baseline.yaml --registry data/registry.yaml
   ```

7. Retain the immutable seed-specific directories under `outputs/c01/`. Each directory includes raw and clipped endpoint predictions, metrics, the resolved configuration, split manifest, log, data/config/lock hashes, environment versions, and the exact clean Git revision.

8. Run C02 from the same clean commit with `uv run python scripts/run_c02_conformal.py --config configs/cmapss/conformal.yaml --registry data/registry.yaml`; retain its seed-specific immutable directories under `outputs/c02/`, including calibration cut points, quantiles, interval predictions, metrics and provenance.

C01 artifacts are generated locally because the official raw dataset is intentionally excluded from Git. Generated numbers remain research artifacts rather than conclusions until independently audited.
