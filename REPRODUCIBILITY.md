# Reproducibility

The repository is designed so data, configuration, code revision, environment, split manifests, and generated metrics can be traced together.

1. Install Python 3.11+ and `uv`.
2. Place officially sourced C-MAPSS files under `data/raw/`; never commit them.
3. Record file hashes in the data registry when the data phase is implemented.
4. Run the locked environment and test suite before experiments.
5. Use configuration-driven commands and retain each generated run manifest.

Until the experiment phases are implemented and verified, there are no research results to report.

