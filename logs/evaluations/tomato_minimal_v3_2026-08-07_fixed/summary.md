# Tomato minimal-v3 fixed evaluation

Best checkpoint: `/home/descfly/linkagent/logs/rsl_rl/tomato_pick_minimal_v3/2026-08-07_05-13-16_2026-08-07_05-13_tomato_minimal_graspcredit_resume500b_8192/model_1450.pt`
Success: 62.57% (95% Wilson CI 61.95%–63.18%)
Drop: 19.03%
Episodes: 23842 across seeds [17, 42, 2026]

| iteration | success | drop | detach | retained | episodes |
|---:|---:|---:|---:|---:|---:|
| 1300 | 62.22% | 18.38% | 95.61% | 62.33% | 23488 |
| 1350 | 61.98% | 17.74% | 95.42% | 62.04% | 23944 |
| 1400 | 61.23% | 19.71% | 95.84% | 61.30% | 24182 |
| 1450 | 62.57% | 19.03% | 96.39% | 62.65% | 23842 |
| 1499 | 62.00% | 19.49% | 96.22% | 62.12% | 23484 |
