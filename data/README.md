# Data layout

Raw C-MAPSS files are excluded from Git and must be placed in `data/raw/`:

```text
train_FD001.txt ... train_FD004.txt
test_FD001.txt  ... test_FD004.txt
RUL_FD001.txt   ... RUL_FD004.txt
```

Source: [NASA C-MAPSS Open Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).

Run verification from the repository root:

```bash
uv run python scripts/verify_data.py --registry data/registry.yaml
```

The verifier is fail-closed: missing files, changed bytes, changed SHA-256, wrong row counts, wrong engine counts, and malformed 26-column files fail with exit code 1.

