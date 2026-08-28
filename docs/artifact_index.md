# Artifact index policy

The canonical machine-readable index is `docs/artifact_index.yaml`. It lists
every local top-level tree under `outputs/` and classifies it as `OFFICIAL` or
`SUPERSEDED`. An unindexed tree is an error, not an implicit candidate result.

## Official trees

| Experiment | Official tree | Producing Git SHA | Expected runs |
|---|---|---|---:|
| C01 | `outputs/c01_hardened` | `2be1b54b55316be988cc9a21c5c6acf6294882a5` | 5 |
| C02 | `outputs/c02_final_review_319975d` | `319975dc35d27f4edba0be75ccef69cb92077127` | 5 |
| C03 | `outputs/c03_final_review_b542ee0` | `b542ee03a95a4e43fc142a2ae8cafb02ef6ca7e3` | 5 |
| C04 | `outputs/c04_final_hardened_3bf9206` | `3bf92063fd99e23b7cbdf6a439a2fc4344807710` | 20 |
| C05 | `outputs/c05_final_bounded_0cfbe63` | `0cfbe6357863e5bffcf4b3223823d1323fd86b0a` | 15 |
| C06 | `outputs/c06_final_c86eff4` | `c86eff4203214cd7b64c5d908bb71cab69f7c555` | 20 |
| C07 | `outputs/c07_final_8b6c6bf` | `8b6c6bf96f05abb9a445fae98a7bba1bce888eb3` | 15 |
| C08 | `outputs/c08_final_8468e13` | `8468e13b7f2c7c3d937f0b4de6f5198e95e7b0b8` | 20 |

All remaining trees are explicitly `SUPERSEDED` in the YAML index, with the
reason, expected manifest count, and producing revision retained. Superseded
trees are never consumed by the results builder.

## Mixed-SHA policy

Cross-experiment reports necessarily contain different historical producing
commits. This is allowed only across experiments, never within an official
tree. Every derived row retains its experiment, run ID, artifact root,
manifest hash, prediction hash, metric hash, and producing Git SHA. Aggregate
outputs retain the complete `source_git_shas` and source-run fields. A duplicate
official tree, more than one SHA within an official tree, an unindexed output
tree, or a missing/mismatched artifact fails closed.

The index does not make an artifact `VERIFIED` by declaration. The Gate D
builder independently validates manifests and hashes and reconstructs stored
metrics from `predictions.csv` using `float_precision="round_trip"`.
