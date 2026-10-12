# Step 3 diagnosis: why the 1D-CNN does not beat GBM

All numbers on validation engines, target = RUL capped at 125. Deep models = 5-seed ensembles.

## 1. Paired comparison with GBM (positive difference = model is worse than GBM)

| Subset | Model | RMSE model | RMSE GBM | Difference | 95 % CI (engine bootstrap) | P(model worse) |
|---|---|---|---|---|---|---|
| FD001 | CNN | 16.58 | 14.65 | +1.93 | -0.95 – +4.33 | 0.91 |
| FD001 | LSTM | 13.97 | 14.65 | -0.68 | -2.48 – +1.23 | 0.21 |

### FD001: RMSE by true-RUL bin

| True RUL | Points | GBM | CNN | LSTM |
|---|---|---|---|---|
| 0–25 | 17 | 4.3 | 5.4 | 3.3 |
| 25–50 | 24 | 13.8 | 13.7 | 8.6 |
| 50–75 | 23 | 14.5 | 22.0 | 16.6 |
| 75–125 | 48 | 18.0 | 15.1 | 17.9 |
| 125–∞ | 38 | 13.5 | 19.3 | 12.3 |

| FD004 | CNN | 18.29 | 16.29 | +2.00 | -0.15 – +4.48 | 0.96 |
| FD004 | LSTM | 15.80 | 16.29 | -0.49 | -1.49 – +0.63 | 0.21 |

### FD004: RMSE by true-RUL bin

| True RUL | Points | GBM | CNN | LSTM |
|---|---|---|---|---|
| 0–25 | 45 | 5.9 | 5.5 | 5.4 |
| 25–50 | 53 | 12.4 | 18.4 | 11.8 |
| 50–75 | 46 | 22.8 | 20.7 | 21.2 |
| 75–125 | 86 | 16.8 | 19.2 | 17.5 |
| 125–∞ | 140 | 17.0 | 19.5 | 16.2 |


## 2. Learning curves (mean of 5 seeds)

| Subset | Model | Best epoch | Val at best | Val at epoch 40 | Train at best | Train at epoch 40 |
|---|---|---|---|---|---|---|
| FD001 | CNN | 17 | 17.43 | 18.72 | 15.20 | 12.85 |
| FD001 | LSTM | 15 | 15.11 | 15.59 | 13.31 | 11.64 |
| FD004 | CNN | 6 | 19.11 | 21.98 | 18.87 | 9.79 |
| FD004 | LSTM | 8 | 16.54 | 17.95 | 14.79 | 9.69 |
