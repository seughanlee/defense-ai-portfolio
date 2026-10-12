# Final evaluation on the official test sets

Run on 2026-10-13, once, after all model, interval and threshold choices were fixed on validation engines. One point per test engine (its last cycle). Point predictions clipped to [0, 125].


## FD001  (100 test engines; true RUL 7–145, 11 engines above 125)

### Point prediction

| Model | RMSE (true RUL) | 95 % CI | RMSE (capped) | MAE | NASA score (sum) | NASA score (mean) |
|---|---|---|---|---|---|---|
| GBM | 13.85 | 12.1–15.6 | 12.71 | 10.87 | 288 | 2.88 |
| LSTM | 16.71 | 13.9–19.4 | 15.81 | 12.24 | 707 | 7.07 |
| CNN | 18.71 | 15.9–21.4 | 17.38 | 13.96 | 750 | 7.50 |

### 90 % intervals (capped RUL scale; calibration = validation-stage calibration engines)

| Model + method | Selected on validation? | Coverage | Below lower (dangerous) | Above upper | Mean width | Coverage RUL<25 | Coverage RUL 50–75 |
|---|---|---|---|---|---|---|---|
| GBM + Conformal (Mondrian) | **yes** | 0.970 | 0.020 | 0.010 | 40.0 | 1.00 (n=19) | 1.00 (n=13) |
| LSTM + Conformal (Mondrian) |  | 0.920 | 0.040 | 0.040 | 39.5 | 1.00 (n=19) | 0.77 (n=13) |
| LSTM + Conformal (normalized) |  | 0.890 | 0.080 | 0.030 | 43.0 | 1.00 (n=19) | 0.69 (n=13) |
| LSTM + Conformal (fixed) |  | 0.910 | 0.050 | 0.040 | 43.5 | 1.00 (n=19) | 0.69 (n=13) |
| LSTM + Deep ensemble |  | 0.300 | 0.480 | 0.220 | 10.5 | 0.63 (n=19) | 0.08 (n=13) |

## FD004  (248 test engines; true RUL 6–195, 67 engines above 125)

### Point prediction

| Model | RMSE (true RUL) | 95 % CI | RMSE (capped) | MAE | NASA score (sum) | NASA score (mean) |
|---|---|---|---|---|---|---|
| GBM | 26.98 | 23.9–29.8 | 14.82 | 18.68 | 5132 | 20.69 |
| LSTM | 26.54 | 23.6–29.3 | 14.60 | 19.30 | 5104 | 20.58 |
| CNN | 31.21 | 28.1–33.9 | 19.25 | 23.42 | 8266 | 33.33 |

Padded engines (test trajectory shorter than 30 cycles): 11. LSTM RMSE (capped) padded 8.51 vs not padded 14.82.

### 90 % intervals (capped RUL scale; calibration = validation-stage calibration engines)

| Model + method | Selected on validation? | Coverage | Below lower (dangerous) | Above upper | Mean width | Coverage RUL<25 | Coverage RUL 50–75 |
|---|---|---|---|---|---|---|---|
| GBM + Conformal (Mondrian) |  | 0.940 | 0.032 | 0.028 | 44.8 | 0.98 (n=49) | 0.88 (n=26) |
| LSTM + Conformal (Mondrian) | **yes** | 0.903 | 0.060 | 0.036 | 39.1 | 0.98 (n=49) | 0.73 (n=26) |
| LSTM + Conformal (normalized) |  | 0.935 | 0.044 | 0.020 | 44.7 | 1.00 (n=49) | 0.81 (n=26) |
| LSTM + Conformal (fixed) |  | 0.927 | 0.056 | 0.016 | 45.0 | 1.00 (n=49) | 0.81 (n=26) |
| LSTM + Deep ensemble |  | 0.351 | 0.347 | 0.302 | 12.8 | 0.33 (n=49) | 0.35 (n=26) |
