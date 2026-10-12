# Step 0 data report

## Subsets

| Subset | Train engines | Test engines | Train life min/median/max | Test length min/median/max | Test true RUL min/median/max | Operating conditions |
|---|---|---|---|---|---|---|
| FD001 | 100 | 100 | 128 / 199 / 362 | 31 / 133 / 303 | 7 / 86 / 145 | 1 |
| FD002 | 260 | 259 | 128 / 199 / 378 | 21 / 132 / 367 | 6 / 80 / 194 | 6 |
| FD003 | 100 | 100 | 145 / 220 / 525 | 38 / 148 / 475 | 6 / 77 / 145 | 1 |
| FD004 | 249 | 248 | 128 / 234 / 543 | 19 / 153 / 486 | 6 / 88 / 195 | 6 |

## Near-constant sensors (within each operating condition for FD002/FD004)

| Subset | Sensors |
|---|---|
| FD001 | s1 (T2), s5 (P2), s6 (P15), s10 (epr), s16 (farB), s18 (Nf_dmd), s19 (PCNfR_dmd) |
| FD002 | s1 (T2), s5 (P2), s10 (epr), s16 (farB), s18 (Nf_dmd), s19 (PCNfR_dmd) |
| FD003 | s1 (T2), s5 (P2), s16 (farB), s18 (Nf_dmd), s19 (PCNfR_dmd) |
| FD004 | s1 (T2), s5 (P2), s16 (farB), s18 (Nf_dmd), s19 (PCNfR_dmd) |

## Degradation trend strength: |Spearman| with RUL capped at 125 (top 8, condition-normalized for FD002/FD004)

| Subset | Sensors |
|---|---|
| FD001 | s11 0.72, s4 0.70, s12 0.70, s7 0.68, s15 0.67, s21 0.66, s20 0.66, s17 0.64 |
| FD002 | s11 0.72, s4 0.68, s15 0.67, s17 0.60, s2 0.59, s3 0.58, s12 0.50, s7 0.48 |
| FD003 | s11 0.74, s13 0.72, s8 0.71, s4 0.71, s17 0.68, s3 0.65, s2 0.64, s9 0.57 |
| FD004 | s11 0.71, s4 0.67, s17 0.63, s3 0.61, s2 0.54, s13 0.53, s8 0.53, s9 0.52 |

## Operating conditions (FD002/FD004)

| Subset | Distinct conditions (train) | Rows per condition | Distinct conditions (test) | Conditions only in test |
|---|---|---|---|---|
| FD002 | 6 | 8002–13458 | 6 | 없음 |
| FD004 | 6 | 9091–15395 | 6 | 없음 |

## Short test trajectories (limit the input window length)

| Subset | Test engines shorter than 30 cycles | Shortest test trajectory |
|---|---|---|
| FD001 | 0 | 31 |
| FD002 | 6 | 21 |
| FD003 | 0 | 38 |
| FD004 | 11 | 19 |
