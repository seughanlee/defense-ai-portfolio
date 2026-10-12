# Step 5: sensor faults and an unseen fault mode

Everything (models, normalization, conformal calibration) comes from FD001. Target: RUL capped at 125.

| Condition | Model | Method | RMSE | Median ensemble std | Coverage | Below lower (dangerous) | Mean width | Coverage RUL<25 | Worst engine |
|---|---|---|---|---|---|---|---|---|---|
| FD001 val (clean) | LSTM | Conformal (fixed) | 14.0 | 2.32 | 0.907 | 0.067 | 43.9 | 1.00 | 0.70 |
| FD001 val (clean) | LSTM | Conformal (normalized) | 14.0 | 2.32 | 0.893 | 0.100 | 42.0 | 1.00 | 0.50 |
| FD001 val (clean) | LSTM | Conformal (Mondrian) | 14.0 | 2.32 | 0.913 | 0.073 | 42.1 | 1.00 | 0.70 |
| FD001 val (clean) | GBM | Conformal (Mondrian) | 14.7 | nan | 0.880 | 0.080 | 41.6 | 0.94 | 0.50 |
| FD001 val, stuck s11 (Ps30) | LSTM | Conformal (fixed) | 16.5 | 3.14 | 0.887 | 0.113 | 41.7 | 1.00 | 0.40 |
| FD001 val, stuck s11 (Ps30) | LSTM | Conformal (normalized) | 16.5 | 3.14 | 0.893 | 0.107 | 45.1 | 1.00 | 0.40 |
| FD001 val, stuck s11 (Ps30) | LSTM | Conformal (Mondrian) | 16.5 | 3.14 | 0.887 | 0.113 | 40.4 | 0.94 | 0.40 |
| FD001 val, stuck s11 (Ps30) | GBM | Conformal (Mondrian) | 15.6 | nan | 0.853 | 0.140 | 42.2 | 0.53 | 0.50 |
| FD001 val, noise x2 (all sensors) | LSTM | Conformal (fixed) | 16.4 | 2.90 | 0.860 | 0.073 | 44.2 | 1.00 | 0.60 |
| FD001 val, noise x2 (all sensors) | LSTM | Conformal (normalized) | 16.4 | 2.90 | 0.907 | 0.073 | 47.2 | 1.00 | 0.70 |
| FD001 val, noise x2 (all sensors) | LSTM | Conformal (Mondrian) | 16.4 | 2.90 | 0.880 | 0.060 | 41.5 | 1.00 | 0.70 |
| FD001 val, noise x2 (all sensors) | GBM | Conformal (Mondrian) | 21.6 | nan | 0.793 | 0.060 | 45.0 | 0.94 | 0.40 |
| FD001 val, drift s4 (T50) | LSTM | Conformal (fixed) | 25.4 | 5.43 | 0.700 | 0.000 | 48.6 | 1.00 | 0.20 |
| FD001 val, drift s4 (T50) | LSTM | Conformal (normalized) | 25.4 | 5.43 | 0.927 | 0.000 | 68.6 | 1.00 | 0.60 |
| FD001 val, drift s4 (T50) | LSTM | Conformal (Mondrian) | 25.4 | 5.43 | 0.627 | 0.000 | 43.7 | 1.00 | 0.10 |
| FD001 val, drift s4 (T50) | GBM | Conformal (Mondrian) | 17.2 | nan | 0.880 | 0.027 | 45.3 | 0.94 | 0.70 |
| FD003 run-to-failure engines (K=10) | LSTM | Conformal (fixed) | 25.9 | 1.98 | 0.801 | 0.191 | 38.2 | 0.74 | 0.20 |
| FD003 run-to-failure engines (K=10) | LSTM | Conformal (normalized) | 25.9 | 1.98 | 0.834 | 0.159 | 37.9 | 0.90 | 0.30 |
| FD003 run-to-failure engines (K=10) | LSTM | Conformal (Mondrian) | 25.9 | 1.98 | 0.798 | 0.191 | 37.5 | 0.70 | 0.20 |
| FD003 run-to-failure engines (K=10) | GBM | Conformal (Mondrian) | 18.6 | nan | 0.840 | 0.151 | 40.0 | 0.70 | 0.20 |
| FD003 official test | LSTM | Conformal (fixed) | 37.3 | 2.90 | 0.690 | 0.310 | 41.3 | 0.33 | 0.00 |
| FD003 official test | LSTM | Conformal (normalized) | 37.3 | 2.90 | 0.720 | 0.280 | 48.0 | 0.73 | 0.00 |
| FD003 official test | LSTM | Conformal (Mondrian) | 37.3 | 2.90 | 0.690 | 0.310 | 43.2 | 0.33 | 0.00 |
| FD003 official test | GBM | Conformal (Mondrian) | 21.6 | nan | 0.740 | 0.240 | 44.2 | 0.33 | 0.00 |

## FD003 fault-mode proxy (LSTM + Conformal Mondrian, run-to-failure engines)

FD001 training engines: s12 falls in 60 of 60 engines (HPC degradation only), so 's12 rises' marks behaviour FD001 never showed (presumably the fan-degradation mode; FD003 has no per-engine fault labels). The s12 direction agrees with the s7 direction in 100% of FD003 engines.

| Group | Engines | RMSE | Coverage | Below lower (dangerous) | Median ensemble std |
|---|---|---|---|---|---|
| FD001-like (s12 falls) | 56 | 14.5 | 0.911 | 0.070 | 2.40 |
| unseen (s12 rises) | 44 | 35.4 | 0.655 | 0.345 | 1.52 |

## Input-range guard (no model): share of windows flagged

A window is flagged when any sensor in its last 10 cycles exceeds |z| in FD001-normalized units.

| Threshold \|z\| | FD001 val (false alarms) | FD003 FD001-like | FD003 unseen | FD003 unseen, RUL ≤ 50 | FD003 FD001-like, RUL ≤ 50 |
|---|---|---|---|---|---|
| 4 | 0.000 | 0.002 | 0.584 | 1.000 | 0.006 |
| 6 | 0.000 | 0.000 | 0.375 | 0.957 | 0.000 |
| 8 | 0.000 | 0.000 | 0.273 | 0.870 | 0.000 |
