# C11 controlled execution record

## Status

`IMPLEMENTED / EXECUTED / NOT VERIFIED / NOT REPORTED`

The C11-A design review and implementation-readiness review both passed before
execution. C11 was then executed exactly once from the clean synchronized
revision:

`cba16d0477ab19bf07ec02dd4be151e6b5fb670e`

The immutable local output root is:

`outputs/c11_final_cba16d0/C11_FD001_cba16d0477ab_seed_13`

The SHA-256 of its `manifest.json` is:

`5898f94018da19d54a376ecdaccafddb5092a8312b7633a9a2760f241014055b`

The authorized command was:

```bash
uv run python scripts/run_c11_finite_reservoir.py \
  --config configs/cmapss/finite_reservoir.yaml \
  --registry data/registry.yaml \
  --data-root data/raw \
  --output-root outputs/c11_final_cba16d0
```

The runner validated all three registered FD001 inputs before and after
computation and published one immutable run. This record does not assign
`VERIFIED` or `REPORTED`: an independent artifact-level audit must validate the
manifest, all declared hashes, data roles, exact multiplicities, retained
distributions, references, discrepancies, observation diagnostic, and
reconstruction determinism before either status may change.

C12 remains blocked and optional. No C11 numerical value may be promoted into a
research conclusion before the artifact-level review passes.
