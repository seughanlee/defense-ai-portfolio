# Step 2: baselines (validation engines, K=10 truncation points per engine)

Predictions are clipped to [0, 125]. RMSE uses the true (uncapped) RUL unless marked capped. 95 % CI: engine-level bootstrap.


## FD001  (train 60 engines / 12,661 windows, val 15 engines / 150 points, 22 of them with true RUL > 150)

| Model | RMSE (all) | RMSE (true RUL ≤ 150) | 95 % CI | RMSE capped | MAE | NASA mean |
|---|---|---|---|---|---|---|
| Constant (mean RUL) | 56.7 | 44.0 | 40.7–47.1 | 41.0 | 46.2 | 11488.69 |
| Linear (last cycle) | 36.4 | 22.4 | 19.1–25.6 | 18.9 | 26.6 | 1107.08 |
| GBM (window stats) | 32.7 | 18.5 | 15.2–22.7 | 14.7 | 21.4 | 847.87 |

## FD004  (train 149 engines / 37,458 windows, val 37 engines / 370 points, 109 of them with true RUL > 150)

| Model | RMSE (all) | RMSE (true RUL ≤ 150) | 95 % CI | RMSE capped | MAE | NASA mean |
|---|---|---|---|---|---|---|
| Constant (mean RUL) | 80.0 | 47.3 | 44.5–50.2 | 42.3 | 62.5 | 14866974.49 |
| Linear (last cycle) | 63.8 | 24.6 | 22.5–26.7 | 22.5 | 43.4 | 1506135.99 |
| GBM (window stats) | 59.9 | 17.0 | 15.1–19.3 | 16.3 | 36.3 | 1760606.84 |
