# Experiment registry

| ID | Question | Method | Source → target | Unit | Alpha | Seed | Primary metric | Config path | Status |
|---|---|---|---|---|---|---|---|---|---|
| C01 | RQ1 | Baselines | FD001 → FD001 | engine endpoint | — | 13,37,73,101,137 | RMSE/MAE | configs/cmapss/fd001_baseline.yaml | planned |
| C02 | RQ2 | Split conformal | FD001 → FD001 | engine endpoint | .10,.05 | 13,37,73,101,137 | coverage/width | configs/cmapss/conformal.yaml | planned |
| C03 | RQ3 | CQR | FD001 → FD001 | engine endpoint | .10,.05 | 13,37,73,101,137 | coverage/width | configs/cmapss/conformal.yaml | planned |
| C04 | RQ4 | Shift matrix | FD001 → FD002/3/4 | engine endpoint | .10,.05 | 13,37,73,101,137 | coverage/width | configs/cmapss/shift_matrix.yaml | planned |
| C05 | RQ5 | Weighted conformal | FD001 → shifts | engine endpoint | .10,.05 | 13,37,73,101,137 | coverage/width | configs/cmapss/shift_matrix.yaml | planned |
| C06 | RQ6 | Sensitivity | cap/calibration | engine endpoint | .10,.05 | 13,37,73,101,137 | coverage/width | configs/cmapss/conformal.yaml | planned |
| C07 | RQ4 | Regime-aware scaling | FD001 → FD002/4 | engine endpoint | — | 13,37,73,101,137 | RMSE/MAE | configs/cmapss/shift_matrix.yaml | planned |
| C08 | RQ5 | ACI | FD001 → shifts | engine endpoint | .10,.05 | 13,37,73,101,137 | coverage/width | configs/cmapss/shift_matrix.yaml | planned |
| C09 | RQ6 | Seed sensitivity | all C-MAPSS | engine endpoint | .10,.05 | 13,37,73,101,137 | mean/std/CI | configs/cmapss/shift_matrix.yaml | planned |
| C10 | RQ6 | Cap sensitivity | 125 vs 130 | engine endpoint | .10,.05 | 13,37,73,101,137 | coverage/width | configs/cmapss/conformal.yaml | planned |
| M01 | RQ7 | MALT baseline | grouped split | transcript | — | 13,37,73,101,137 | AUPRC | configs/malt/baseline.yaml | planned |
| M02 | RQ8 | Conformal classification | grouped split | transcript | .20,.10,.05 | 13,37,73,101,137 | conditional coverage | configs/malt/baseline.yaml | planned |
| M03 | RQ7 | Text-only ablation | grouped split | transcript | — | 13,37,73,101,137 | AUPRC | configs/malt/baseline.yaml | planned |
| M04 | RQ7 | Structural-only ablation | grouped split | transcript | — | 13,37,73,101,137 | AUPRC | configs/malt/baseline.yaml | planned |
| M05 | RQ7 | Reasoning ablation | grouped split | transcript | — | 13,37,73,101,137 | AUPRC | configs/malt/baseline.yaml | planned |
| M06 | RQ8 | LAC | grouped split | transcript | .20,.10,.05 | 13,37,73,101,137 | conditional coverage | configs/malt/baseline.yaml | planned |
| M07 | RQ8 | APS | grouped split | transcript | .20,.10,.05 | 13,37,73,101,137 | conditional coverage | configs/malt/baseline.yaml | planned |
| M08 | RQ8 | Optional LLM monitor | grouped split | transcript | — | 13,37,73,101,137 | AUPRC | configs/malt/baseline.yaml | planned |
