# Step 6: maintenance policies (FD001, 5-fold engine-level cross-fitting, 100 engines)

Cost unit = one planned maintenance. Wasted life is charged as (cycles left) / (mean life). A failure before maintenance costs C_f. Thresholds are chosen inside each fold on 15 engines the evaluated engines never touch. Model: GBM + Mondrian conformal (selected for FD001 in step 4). Lead time = cycles between the decision and the maintenance.


## Lead time 0 cycles

| C_f | Policy | Mean cost / engine | vs Perfect | Failures | Median wasted cycles | Thresholds per fold |
|---|---|---|---|---|---|---|
| 5 | Fixed interval | 1.521 | 1.51x | 5/100 | 58 | 157, 134, 134, 127, 152 |
| 5 | A: point | 1.466 | 1.46x | 11/100 | 5 | 5, 4, 6, 8, 9 |
| 5 | B: lower bound | 1.061 | 1.06x | 0/100 | 12 | 0, 0, 0, 0, 0 |
| 5 | Perfect | 1.005 | 1.00x | 0/100 | 1 | - |
| 10 | Fixed interval | 1.771 | 1.76x | 5/100 | 58 | 157, 134, 134, 127, 152 |
| 10 | A: point | 2.016 | 2.01x | 11/100 | 5 | 5, 4, 6, 8, 9 |
| 10 | B: lower bound | 1.061 | 1.06x | 0/100 | 12 | 0, 0, 0, 0, 0 |
| 10 | Perfect | 1.005 | 1.00x | 0/100 | 1 | - |
| 50 | Fixed interval | 3.771 | 3.75x | 5/100 | 58 | 157, 134, 134, 127, 152 |
| 50 | A: point | 6.416 | 6.38x | 11/100 | 5 | 5, 4, 6, 8, 9 |
| 50 | B: lower bound | 1.061 | 1.06x | 0/100 | 12 | 0, 0, 0, 0, 0 |
| 50 | Perfect | 1.005 | 1.00x | 0/100 | 1 | - |
| 100 | Fixed interval | 6.271 | 6.24x | 5/100 | 58 | 157, 134, 134, 127, 152 |
| 100 | A: point | 11.916 | 11.86x | 11/100 | 5 | 5, 4, 6, 8, 9 |
| 100 | B: lower bound | 1.061 | 1.06x | 0/100 | 12 | 0, 0, 0, 0, 0 |
| 100 | Perfect | 1.005 | 1.00x | 0/100 | 1 | - |

### B vs A, lead time 0: paired engine bootstrap (negative = B cheaper)

| C_f | B − A | 95 % CI | B − fixed interval |
|---|---|---|---|
| 5 | -0.405 | -0.682 – -0.169 | -0.460 |
| 10 | -0.955 | -1.496 – -0.418 | -0.710 |
| 50 | -5.355 | -8.783 – -2.416 | -2.710 |
| 100 | -10.855 | -16.798 – -5.875 | -5.210 |

## Lead time 10 cycles

| C_f | Policy | Mean cost / engine | vs Perfect | Failures | Median wasted cycles | Thresholds per fold |
|---|---|---|---|---|---|---|
| 5 | Fixed interval | 1.521 | 1.51x | 5/100 | 58 | 157, 134, 134, 127, 152 |
| 5 | A: point | 1.084 | 1.08x | 1/100 | 8 | 16, 17, 22, 20, 20 |
| 5 | B: lower bound | 1.084 | 1.08x | 1/100 | 8 | 8, 4, 8, 7, 2 |
| 5 | Perfect | 1.005 | 1.00x | 0/100 | 1 | - |
| 10 | Fixed interval | 1.771 | 1.76x | 5/100 | 58 | 157, 134, 134, 127, 152 |
| 10 | A: point | 1.134 | 1.13x | 1/100 | 8 | 16, 17, 22, 20, 20 |
| 10 | B: lower bound | 1.134 | 1.13x | 1/100 | 8 | 8, 4, 8, 7, 2 |
| 10 | Perfect | 1.005 | 1.00x | 0/100 | 1 | - |
| 50 | Fixed interval | 3.771 | 3.75x | 5/100 | 58 | 157, 134, 134, 127, 152 |
| 50 | A: point | 1.534 | 1.53x | 1/100 | 8 | 16, 17, 22, 20, 20 |
| 50 | B: lower bound | 1.534 | 1.53x | 1/100 | 8 | 8, 4, 8, 7, 2 |
| 50 | Perfect | 1.005 | 1.00x | 0/100 | 1 | - |
| 100 | Fixed interval | 6.271 | 6.24x | 5/100 | 58 | 157, 134, 134, 127, 152 |
| 100 | A: point | 2.034 | 2.02x | 1/100 | 8 | 16, 17, 22, 20, 20 |
| 100 | B: lower bound | 2.034 | 2.02x | 1/100 | 8 | 8, 4, 8, 7, 2 |
| 100 | Perfect | 1.005 | 1.00x | 0/100 | 1 | - |

### B vs A, lead time 10: paired engine bootstrap (negative = B cheaper)

| C_f | B − A | 95 % CI | B − fixed interval |
|---|---|---|---|
| 5 | +0.000 | -0.001 – +0.001 | -0.437 |
| 10 | +0.000 | -0.001 – +0.001 | -0.637 |
| 50 | +0.000 | -0.001 – +0.001 | -2.237 |
| 100 | +0.000 | -0.001 – +0.001 | -4.237 |

## Diagnosis: hindsight-optimal thresholds on the 100 evaluated engines (not deployable)

| Lead | C_f | Policy | Best τ (hindsight) | Cost at best τ | Cost with fold-chosen τ | Failures at best τ |
|---|---|---|---|---|---|---|
| 0 | 10 | A: point | 9 | 1.040 | 2.016 | 0/100 |
| 0 | 10 | B: lower bound | 0 | 1.061 | 1.061 | 0/100 |
| 0 | 50 | A: point | 9 | 1.040 | 6.416 | 0/100 |
| 0 | 50 | B: lower bound | 0 | 1.061 | 1.061 | 0/100 |
| 10 | 10 | A: point | 23 | 1.065 | 1.134 | 0/100 |
| 10 | 10 | B: lower bound | 15 | 1.117 | 1.134 | 0/100 |
| 10 | 50 | A: point | 23 | 1.065 | 1.534 | 0/100 |
| 10 | 50 | B: lower bound | 15 | 1.117 | 1.534 | 0/100 |
