# v2 step 2: conformal prediction sets (v1 CNN ensemble, seeds [0, 1, 2], no retraining)

Test n = 2000, split 50/50 into calibration / evaluation 200 times (mean +- std over splits). Ensemble top-1 accuracy on test = 0.848. RAPS: k_reg = 1, lambda = 0.5 chosen on val (mean val set size per lambda: 0: 1.277, 0.002: 1.273, 0.01: 1.280, 0.05: 1.229, 0.1: 1.201, 0.2: 1.151, 0.5: 1.119).

Coverage = share of samples whose true class is in the set. Singleton accuracy = coverage among size-1 sets.

## Marginal results, alpha = 0.1 (target coverage 0.90)

| Method | Coverage | Mean size | Singleton | Empty | Singleton accuracy | Worst-class coverage |
|---|---|---|---|---|---|---|
| LAC | 0.899 +- 0.014 | 1.12 +- 0.02 | 0.847 | 0.018 | 0.905 | 0.716 (quad) |
| APS | 0.899 +- 0.016 | 1.29 +- 0.03 | 0.643 | 0.057 | 0.948 | 0.867 (quad) |
| RAPS | 0.899 +- 0.014 | 1.14 +- 0.02 | 0.856 | 0.000 | 0.902 | 0.698 (quad) |
| LAC-class | 0.901 +- 0.014 | 1.20 +- 0.03 | 0.723 | 0.040 | 0.934 | 0.898 (heli) |
| LAC-SNR | 0.902 +- 0.013 | 1.14 +- 0.02 | 0.838 | 0.014 | 0.912 | 0.727 (quad) |

## Marginal results, alpha = 0.05 (target coverage 0.95)

| Method | Coverage | Mean size | Singleton | Empty | Singleton accuracy | Worst-class coverage |
|---|---|---|---|---|---|---|
| LAC | 0.949 +- 0.010 | 1.26 +- 0.02 | 0.742 | 0.000 | 0.952 | 0.881 (quad) |
| APS | 0.949 +- 0.010 | 1.42 +- 0.03 | 0.603 | 0.028 | 0.971 | 0.933 (quad) |
| RAPS | 0.950 +- 0.010 | 1.31 +- 0.02 | 0.687 | 0.000 | 0.956 | 0.881 (quad) |
| LAC-class | 0.952 +- 0.010 | 1.33 +- 0.02 | 0.663 | 0.017 | 0.973 | 0.951 (quad) |
| LAC-SNR | 0.952 +- 0.009 | 1.29 +- 0.02 | 0.743 | 0.000 | 0.952 | 0.878 (quad) |

## Per class, alpha = 0.1

Coverage / mean set size / share of sets that are exactly {quad, hexa}.

| Method | quad | hexa | fixed | heli | bird |
|---|---|---|---|---|---|
| LAC | 0.72 / 1.28 / 0.32 | 0.88 / 1.32 / 0.34 | 0.91 / 0.99 / 0.01 | 0.99 / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 |
| APS | 0.87 / 1.70 / 0.54 | 0.94 / 1.67 / 0.55 | 0.89 / 1.26 / 0.04 | 0.89 / 0.91 / 0.00 | 0.90 / 0.90 / 0.00 |
| RAPS | 0.70 / 1.30 / 0.26 | 0.87 / 1.27 / 0.24 | 0.93 / 1.14 / 0.06 | 1.00 / 1.01 / 0.00 | 1.00 / 1.00 / 0.00 |
| LAC-class | 0.90 / 1.53 / 0.52 | 0.90 / 1.57 / 0.57 | 0.90 / 1.08 / 0.05 | 0.90 / 0.91 / 0.00 | 0.90 / 0.90 / 0.00 |
| LAC-SNR | 0.73 / 1.30 / 0.32 | 0.87 / 1.32 / 0.33 | 0.92 / 1.06 / 0.05 | 0.99 / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 |

## Per SNR bin (training range), alpha = 0.1

Coverage / mean set size.

| Method | [-5, 0) dB | [0, 5) dB | [5, 10) dB | [10, 20) dB |
|---|---|---|---|---|
| LAC | 0.812 / 1.18 | 0.893 / 1.20 | 0.918 / 1.11 | 0.938 / 1.05 |
| APS | 0.910 / 1.67 | 0.903 / 1.39 | 0.894 / 1.18 | 0.894 / 1.09 |
| RAPS | 0.865 / 1.36 | 0.877 / 1.20 | 0.899 / 1.08 | 0.928 / 1.03 |
| LAC-class | 0.779 / 1.28 | 0.916 / 1.34 | 0.927 / 1.19 | 0.941 / 1.09 |
| LAC-SNR | 0.901 / 1.42 | 0.905 / 1.24 | 0.903 / 1.07 | 0.900 / 0.97 |

## Shift: SNR sweep, thresholds calibrated on the whole test split (alpha = 0.1)

Training SNR range is -5 to 20 dB; -15 and -10 dB lie outside it. Coverage / mean set size.

| Method | -15 dB | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---|---|---|---|---|---|---|---|
| LAC | 0.36 / 1.46 | 0.54 / 1.18 | 0.75 / 1.10 | 0.92 / 1.24 | 0.91 / 1.16 | 0.95 / 1.08 | 0.96 / 1.07 | 0.94 / 1.04 |
| APS | 0.48 / 2.35 | 0.75 / 2.23 | 0.91 / 1.77 | 0.92 / 1.45 | 0.91 / 1.28 | 0.91 / 1.17 | 0.92 / 1.11 | 0.88 / 1.02 |
| RAPS | 0.35 / 1.69 | 0.62 / 1.71 | 0.86 / 1.48 | 0.92 / 1.27 | 0.90 / 1.14 | 0.94 / 1.07 | 0.95 / 1.05 | 0.92 / 1.03 |
| LAC-class | 0.38 / 1.72 | 0.39 / 1.27 | 0.61 / 1.14 | 0.93 / 1.34 | 0.95 / 1.26 | 0.95 / 1.15 | 0.96 / 1.10 | 0.95 / 1.08 |
| LAC-SNR | 0.42 / 1.83 | 0.65 / 1.63 | 0.88 / 1.46 | 0.93 / 1.27 | 0.88 / 1.08 | 0.89 / 0.96 | 0.92 / 0.96 | 0.91 / 0.98 |

LAC-SNR assigns sweep levels outside the training range to the nearest bin (clipped).

## Unknown target: octocopter (never seen), alpha = 0.1

No class is correct, so coverage is undefined. 'Contains quad or hexa' = the set at least says 'a multicopter'. 'Size >= 2' compares with the same rate on the known-class test set.

| Method | Mean size | Contains quad or hexa | Contains both | Singleton | Empty | Size >= 2 (octo) | Size >= 2 (test) |
|---|---|---|---|---|---|---|---|
| LAC | 1.22 | 0.940 | 0.253 | 0.713 | 0.035 | 0.253 | 0.137 |
| APS | 1.57 | 0.990 | 0.507 | 0.460 | 0.010 | 0.530 | 0.302 |
| RAPS | 1.25 | 1.000 | 0.212 | 0.750 | 0.000 | 0.250 | 0.132 |
| LAC-class | 1.46 | 0.988 | 0.445 | 0.542 | 0.000 | 0.458 | 0.236 |
| LAC-SNR | 1.29 | 0.960 | 0.292 | 0.682 | 0.015 | 0.302 | 0.143 |

## Post hoc: empty set replaced by the top-1 class (+fallback), alpha = 0.1

Added after the first run showed empty sets on easy classes. Coverage / mean set size per class.

| Method | Coverage | Mean size | Empty before | quad | hexa | fixed | heli | bird |
|---|---|---|---|---|---|---|---|---|
| LAC+fallback | 0.908 | 1.14 | 0.018 | 0.74 / 1.32 | 0.89 / 1.34 | 0.92 / 1.01 | 0.99 / 1.00 | 1.00 / 1.00 |
| APS+fallback | 0.956 | 1.34 | 0.057 | 0.87 / 1.70 | 0.95 / 1.67 | 0.96 / 1.33 | 1.00 / 1.02 | 1.00 / 1.00 |
| RAPS+fallback | 0.899 | 1.14 | 0.000 | 0.70 / 1.30 | 0.87 / 1.27 | 0.93 / 1.14 | 1.00 / 1.01 | 1.00 / 1.00 |
| LAC-class+fallback | 0.940 | 1.24 | 0.040 | 0.90 / 1.53 | 0.90 / 1.58 | 0.91 / 1.08 | 0.99 / 1.00 | 1.00 / 1.00 |
| LAC-SNR+fallback | 0.908 | 1.15 | 0.014 | 0.74 / 1.34 | 0.88 / 1.35 | 0.92 / 1.07 | 0.99 / 1.00 | 1.00 / 1.00 |
