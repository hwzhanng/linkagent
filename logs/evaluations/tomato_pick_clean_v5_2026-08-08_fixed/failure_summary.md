# Clean-v5 fixed evaluation failure classification

Selected best: `model_1350.pt` (96.18% success).

| iteration | success | worst seed | spread | drop | seed gate |
|---:|---:|---:|---:|---:|:---:|
| 1300 | 96.11% | 95.83% | 0.44% | 2.03% | yes |
| 1350 | 96.18% | 96.06% | 0.21% | 2.10% | yes |
| 1400 | 96.08% | 95.79% | 0.74% | 2.15% | yes |
| 1450 | 96.06% | 95.87% | 0.31% | 2.22% | yes |
| 1500 | 96.14% | 95.97% | 0.38% | 2.33% | yes |
| 1550 | 95.81% | 95.47% | 0.52% | 2.42% | yes |
| 1600 | 95.64% | 95.41% | 0.38% | 2.54% | yes |

Best-checkpoint terminal distribution:

- success: 96.18%
- drop: 2.10%
- excess_force: 1.69%
- out_of_bounds: 0.03%
- timeout: 0.00%
- unclassified_failure: 0.00%
