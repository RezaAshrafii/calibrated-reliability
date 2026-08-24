# Experiment registry

| ID | Question | Method | Source → target | Unit | Alpha | Seed | Status |
|---|---|---|---|---|---|---|---|
| C01 | RQ1 | Baselines | FD001 → FD001 | engine endpoint | — | 13,37,73,101,137 | planned |
| C02 | RQ2 | Split conformal | FD001 → FD001 | engine endpoint | .10,.05 | fixed seeds | planned |
| C03 | RQ3 | CQR | FD001 → FD001 | engine endpoint | .10,.05 | fixed seeds | planned |
| C04 | RQ4 | Shift matrix | FD001 → FD002/3/4 | engine endpoint | .10,.05 | fixed seeds | planned |
| C05 | RQ5 | Weighted conformal | FD001 → shifts | engine endpoint | .10,.05 | fixed seeds | planned |
| C06 | RQ6 | Sensitivity | cap/calibration | engine endpoint | .10,.05 | fixed seeds | planned |
| M01 | RQ7 | MALT baseline | grouped split | transcript | — | fixed seeds | planned |
| M02 | RQ8 | Conformal classification | grouped split | transcript | .20,.10,.05 | fixed seeds | planned |
| C07 | RQ4 | Regime-aware scaling | FD001 → FD002/4 | engine endpoint | — | fixed seeds | planned |
| C08 | RQ5 | ACI | FD001 → shifts | engine endpoint | .10,.05 | fixed seeds | planned |
| C09 | RQ6 | Seed sensitivity | all C-MAPSS | engine endpoint | .10,.05 | fixed seeds | planned |
| C10 | RQ6 | Cap sensitivity | 125 vs 130 | engine endpoint | .10,.05 | fixed seeds | planned |
| M03 | RQ7 | Text-only ablation | grouped split | transcript | — | fixed seeds | planned |
| M04 | RQ7 | Structural-only ablation | grouped split | transcript | — | fixed seeds | planned |
| M05 | RQ7 | Reasoning ablation | grouped split | transcript | — | fixed seeds | planned |
| M06 | RQ8 | LAC | grouped split | transcript | .20,.10,.05 | fixed seeds | planned |
| M07 | RQ8 | APS | grouped split | transcript | .20,.10,.05 | fixed seeds | planned |
| M08 | RQ8 | Optional LLM monitor | grouped split | transcript | — | fixed seeds | planned |
