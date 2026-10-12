# Step 4: 90 % intervals on validation engines

Target: RUL capped at 125 (the upper end 125 reads as "125 cycles or more"). Coverage = share of points whose true RUL lies inside the interval.

- **Below lower**: true RUL is shorter than the lower bound, so the engine would fail earlier than promised. This is the safety-relevant miss; with a symmetric 90 % interval it should be about 5 %.
- **Above upper**: the engine lasts longer than the upper bound; maintenance would come early (cost, not risk).
- Selection rule (EVALUATION.md sec. 5): coverage ≥ 88% on validation, then the narrowest mean width.


## FD001  (validation: 15 engines, 150 points)

| Model | Method | Calibration (points) | Coverage | Below lower (dangerous) | Above upper (early) | Mean width | Winkler | Coverage RUL<25 | Worst engine | Engines <80 % |
|---|---|---|---|---|---|---|---|---|---|---|
| LSTM | MC Dropout | K=10 (250) | 0.580 | 0.220 | 0.200 | 18.7 | 97.2 | 0.94 | 0.30 | 13/15 |
| LSTM | Deep ensemble | K=10 (250) | 0.273 | 0.367 | 0.360 | 9.5 | 131.4 | 0.41 | 0.10 | 15/15 |
| LSTM | Conformal (fixed) | K=10 (250) | 0.907 | 0.067 | 0.027 | 43.9 | 53.5 | 1.00 | 0.70 | 2/15 |
| LSTM | Conformal (asymmetric) | K=10 (250) | 0.893 | 0.013 | 0.093 | 41.6 | 59.4 | 1.00 | 0.50 | 2/15 |
| LSTM | Conformal (normalized) | K=10 (250) | 0.893 | 0.100 | 0.007 | 42.0 | 57.3 | 1.00 | 0.50 | 3/15 |
| LSTM | Conformal (Mondrian) | K=10 (250) | 0.913 | 0.073 | 0.013 | 42.1 | 47.7 | 1.00 | 0.70 | 1/15 |
| GBM | Conformal (fixed) | K=10 (250) | 0.940 | 0.033 | 0.027 | 46.5 | 55.6 | 1.00 | 0.70 | 1/15 |
| GBM | Conformal (asymmetric) | K=10 (250) | 0.847 | 0.020 | 0.133 | 40.1 | 68.8 | 1.00 | 0.40 | 4/15 |
| GBM | Conformal (Mondrian) | K=10 (250) | 0.880 | 0.080 | 0.040 | 41.6 | 54.4 | 0.94 | 0.50 | 3/15 |
| LSTM | MC Dropout | K=1 (25) | 0.580 | 0.220 | 0.200 | 18.7 | 97.2 | 0.94 | 0.30 | 13/15 |
| LSTM | Deep ensemble | K=1 (25) | 0.273 | 0.367 | 0.360 | 9.5 | 131.4 | 0.41 | 0.10 | 15/15 |
| LSTM | Conformal (fixed) | K=1 (25) | 0.933 | 0.053 | 0.013 | 48.4 | 52.4 | 1.00 | 0.70 | 1/15 |
| LSTM | Conformal (asymmetric) | K=1 (25) | 0.840 | 0.000 | 0.160 | 40.7 | 66.4 | 1.00 | 0.40 | 4/15 |
| LSTM | Conformal (normalized) | K=1 (25) | 0.933 | 0.067 | 0.000 | 49.2 | 55.8 | 1.00 | 0.70 | 2/15 |
| LSTM | Conformal (Mondrian) | K=1 (25) | 0.987 | 0.000 | 0.013 | 50.7 | 52.5 | 1.00 | 0.80 | 0/15 |
| GBM | Conformal (fixed) | K=1 (25) | 0.947 | 0.027 | 0.027 | 51.1 | 56.2 | 1.00 | 0.70 | 1/15 |
| GBM | Conformal (asymmetric) | K=1 (25) | 0.893 | 0.020 | 0.087 | 42.8 | 63.5 | 1.00 | 0.60 | 2/15 |
| GBM | Conformal (Mondrian) | K=1 (25) | 0.947 | 0.027 | 0.027 | 51.7 | 56.4 | 1.00 | 0.70 | 1/15 |

**Selected for FD001:** GBM + Conformal (Mondrian) (K=10): coverage 0.880, width 41.6 cycles.


## FD004  (validation: 37 engines, 370 points)

| Model | Method | Calibration (points) | Coverage | Below lower (dangerous) | Above upper (early) | Mean width | Winkler | Coverage RUL<25 | Worst engine | Engines <80 % |
|---|---|---|---|---|---|---|---|---|---|---|
| LSTM | MC Dropout | K=10 (630) | 0.614 | 0.203 | 0.184 | 17.9 | 107.2 | 0.78 | 0.00 | 25/37 |
| LSTM | Deep ensemble | K=10 (630) | 0.411 | 0.308 | 0.281 | 11.2 | 123.4 | 0.44 | 0.00 | 33/37 |
| LSTM | Conformal (fixed) | K=10 (630) | 0.930 | 0.046 | 0.024 | 43.4 | 62.4 | 1.00 | 0.60 | 2/37 |
| LSTM | Conformal (asymmetric) | K=10 (630) | 0.927 | 0.030 | 0.043 | 44.1 | 64.3 | 1.00 | 0.60 | 2/37 |
| LSTM | Conformal (normalized) | K=10 (630) | 0.924 | 0.049 | 0.027 | 41.5 | 63.1 | 1.00 | 0.70 | 4/37 |
| LSTM | Conformal (Mondrian) | K=10 (630) | 0.924 | 0.049 | 0.027 | 39.2 | 60.1 | 1.00 | 0.60 | 3/37 |
| GBM | Conformal (fixed) | K=10 (630) | 0.941 | 0.038 | 0.022 | 44.7 | 65.7 | 1.00 | 0.60 | 3/37 |
| GBM | Conformal (asymmetric) | K=10 (630) | 0.927 | 0.030 | 0.043 | 43.8 | 68.8 | 1.00 | 0.60 | 4/37 |
| GBM | Conformal (Mondrian) | K=10 (630) | 0.941 | 0.043 | 0.016 | 42.8 | 64.2 | 0.98 | 0.70 | 3/37 |
| LSTM | MC Dropout | K=1 (63) | 0.614 | 0.203 | 0.184 | 17.9 | 107.2 | 0.78 | 0.00 | 25/37 |
| LSTM | Deep ensemble | K=1 (63) | 0.411 | 0.308 | 0.281 | 11.2 | 123.4 | 0.44 | 0.00 | 33/37 |
| LSTM | Conformal (fixed) | K=1 (63) | 0.968 | 0.016 | 0.016 | 55.6 | 64.9 | 1.00 | 0.70 | 2/37 |
| LSTM | Conformal (asymmetric) | K=1 (63) | 0.968 | 0.016 | 0.016 | 55.7 | 64.9 | 1.00 | 0.70 | 2/37 |
| LSTM | Conformal (normalized) | K=1 (63) | 0.949 | 0.041 | 0.011 | 48.0 | 64.6 | 1.00 | 0.70 | 3/37 |
| LSTM | Conformal (Mondrian) | K=1 (63) | 0.976 | 0.016 | 0.008 | 54.3 | 66.0 | 1.00 | 0.80 | 0/37 |
| GBM | Conformal (fixed) | K=1 (63) | 0.965 | 0.022 | 0.014 | 52.4 | 67.6 | 1.00 | 0.70 | 2/37 |
| GBM | Conformal (asymmetric) | K=1 (63) | 0.965 | 0.022 | 0.014 | 52.4 | 67.6 | 1.00 | 0.70 | 2/37 |
| GBM | Conformal (Mondrian) | K=1 (63) | 0.941 | 0.046 | 0.014 | 48.0 | 67.7 | 0.98 | 0.70 | 2/37 |

**Selected for FD004:** LSTM + Conformal (Mondrian) (K=10): coverage 0.924, width 39.2 cycles.

