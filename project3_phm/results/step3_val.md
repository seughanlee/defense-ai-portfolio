# Step 3: deep models on validation engines

Selection metric (EVALUATION.md sec. 5): RMSE against the capped true RUL. Predictions clipped to [0, 125].

| Subset | Model | Normalization | Seeds | RMSE capped (mean ± std) | Ensemble RMSE capped | 95 % CI (ensemble) | RMSE true RUL ≤ 150 (ensemble) | Best epoch (mean) |
|---|---|---|---|---|---|---|---|---|
| FD001 | cnn | condition | 5 | 16.91 ± 0.35 | 16.58 | 13.1–19.4 | 20.16 | 12 |
| FD001 | lstm | condition | 5 | 14.43 ± 0.35 | 13.97 | 11.6–16.1 | 17.61 | 19 |
| FD004 | cnn | condition | 5 | 18.68 ± 0.27 | 18.29 | 15.3–21.8 | 20.08 | 5 |
| FD004 | cnn | global | 5 | 22.60 ± 0.26 | 22.27 | 19.1–25.6 | 24.33 | 8 |
| FD004 | lstm | condition | 5 | 16.36 ± 0.17 | 15.80 | 12.5–19.7 | 16.87 | 9 |

## GBM baseline (same validation points)

| Subset | RMSE capped |
|---|---|
| FD001 | 14.65 |
| FD004 | 16.29 |
