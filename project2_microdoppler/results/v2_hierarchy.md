# v2 step 3: hierarchical output (v1 CNN ensemble, seeds [0, 1, 2], no retraining)

Answer = a node of the hierarchy {quad, hexa} -> multicopter; fixed-wing, helicopter, bird are leaves; root = 'unidentified'. Correct = true class lies under the answered node. Thresholds (tau, conformal q) are fit on val (n = 1000); test n = 2000, octocopter n = 400.

## Known classes (test)

| Method | Hierarchical accuracy | Leaf answers | 'Multicopter' | 'Unidentified' | Accuracy of leaf answers |
|---|---|---|---|---|---|
| Flat (v1) | 0.848 | 1.000 | 0.000 | 0.000 | 0.848 |
| Coarse (4 groups) | 0.969 | 0.578 | 0.422 | 0.000 | 0.990 |
| Hier-tau, target 0.95 (tau=0.640) | 0.940 | 0.766 | 0.203 | 0.030 | 0.944 |
| Hier-tau, target 0.98 (tau=0.750) | 0.971 | 0.677 | 0.265 | 0.058 | 0.974 |
| Conf-hier (LAC-class + fallback) | 0.929 | 0.800 | 0.197 | 0.003 | 0.924 |

## Per class (test): hierarchical accuracy / share answered at leaf level / share answered 'multicopter'

| Method | quad | hexa | fixed | heli | bird |
|---|---|---|---|---|---|
| Flat (v1) | 0.53 / 1.00 / 0.00 | 0.80 / 1.00 / 0.00 | 0.92 / 1.00 / 0.00 | 0.99 / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 |
| Coarse (4 groups) | 0.98 / 0.01 / 0.98 | 0.99 / 0.01 / 0.99 | 0.87 / 0.87 / 0.13 | 0.99 / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 |
| Hier-tau, target 0.95 (tau=0.640) | 0.84 / 0.49 / 0.47 | 0.94 / 0.51 / 0.46 | 0.92 / 0.83 / 0.09 | 1.00 / 0.99 / 0.00 | 1.00 / 1.00 / 0.00 |
| Hier-tau, target 0.98 (tau=0.750) | 0.93 / 0.29 / 0.63 | 0.98 / 0.31 / 0.64 | 0.94 / 0.79 / 0.06 | 1.00 / 0.99 / 0.00 | 1.00 / 1.00 / 0.00 |
| Conf-hier (LAC-class + fallback) | 0.83 / 0.52 / 0.47 | 0.90 / 0.54 / 0.46 | 0.92 / 0.94 / 0.05 | 0.99 / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 |

## Unknown target: octocopter (a multicopter never seen in training)

Correct informative answer = 'multicopter'. 'Wrong leaf' = called quad or hexa; 'wrong group' = fixed-wing, helicopter or bird. The last column compares the leaf-answer rate with real quad/hexa test samples: if they are equal, the method cannot tell octocopter from the classes it knows.

| Method | 'Multicopter' (correct) | 'Unidentified' | Wrong leaf (quad/hexa) | Wrong group | Leaf rate: octo vs real quad/hexa |
|---|---|---|---|---|---|
| Flat (v1) | 0.000 | 0.000 | 0.973 | 0.028 | 1.000 vs 1.000 |
| Coarse (4 groups) | 0.990 | 0.000 | 0.000 | 0.010 | 0.010 vs 0.010 |
| Hier-tau, target 0.95 (tau=0.640) | 0.393 | 0.022 | 0.583 | 0.003 | 0.585 vs 0.501 |
| Hier-tau, target 0.98 (tau=0.750) | 0.463 | 0.060 | 0.477 | 0.000 | 0.477 vs 0.304 |
| Conf-hier (LAC-class + fallback) | 0.367 | 0.000 | 0.605 | 0.028 | 0.632 vs 0.531 |

## SNR sweep: hierarchical accuracy / leaf-answer rate

Training range is -5 to 20 dB; -15 and -10 dB lie outside it.

| Method | -15 dB | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---|---|---|---|---|---|---|---|
| Flat (v1) | 0.22 / 1.00 | 0.45 / 1.00 | 0.69 / 1.00 | 0.81 / 1.00 | 0.84 / 1.00 | 0.92 / 1.00 | 0.93 / 1.00 | 0.91 / 1.00 |
| Coarse (4 groups) | 0.42 / 0.03 | 0.61 / 0.22 | 0.86 / 0.48 | 0.96 / 0.57 | 0.99 / 0.59 | 1.00 / 0.60 | 1.00 / 0.60 | 1.00 / 0.60 |
| Hier-tau, target 0.95 (tau=0.640) | 0.49 / 0.00 | 0.70 / 0.19 | 0.91 / 0.44 | 0.98 / 0.57 | 0.95 / 0.76 | 0.95 / 0.88 | 0.97 / 0.91 | 0.94 / 0.94 |
| Hier-tau, target 0.98 (tau=0.750) | 0.52 / 0.00 | 0.81 / 0.17 | 0.96 / 0.41 | 0.99 / 0.53 | 0.99 / 0.65 | 0.97 / 0.77 | 0.98 / 0.85 | 0.96 / 0.88 |
| Conf-hier (LAC-class + fallback) | 0.42 / 0.17 | 0.62 / 0.48 | 0.84 / 0.67 | 0.96 / 0.64 | 0.94 / 0.76 | 0.96 / 0.86 | 0.97 / 0.90 | 0.94 / 0.93 |

## Cross-group errors (test): answer lies in a different top-level group than the truth

| Method | Cross-group error rate | Drone (quad/hexa/fixed) answered 'bird' | Bird answered as a drone |
|---|---|---|---|
| Flat (v1) | 0.032 | 0.000 | 0.000 |
| Coarse (4 groups) | 0.032 | 0.000 | 0.000 |
| Hier-tau, target 0.95 (tau=0.640) | 0.018 | 0.000 | 0.000 |
| Hier-tau, target 0.98 (tau=0.750) | 0.012 | 0.000 | 0.000 |
| Conf-hier (LAC-class + fallback) | 0.032 | 0.000 | 0.000 |

## Post hoc: thresholds fit on clean data (random half of test, 200 splits), mean +- std

val is not clean (checkpoint selection and temperature both used it). Main tables above are unchanged.

| Method | Hierarchical accuracy | Quad accuracy | Leaf answers | Octo 'multicopter' | Octo wrong leaf |
|---|---|---|---|---|---|
| Hier-tau 0.95 | 0.951 +- 0.010 | 0.880 +- 0.027 | 0.743 +- 0.014 | 0.404 +- 0.009 | 0.561 +- 0.014 |
| Hier-tau 0.98 | 0.980 +- 0.006 | 0.949 +- 0.015 | 0.652 +- 0.014 | 0.515 +- 0.021 | 0.403 +- 0.030 |
| Conf-hier | 0.943 +- 0.011 | 0.904 +- 0.030 | 0.761 +- 0.019 | 0.443 +- 0.024 | 0.530 +- 0.025 |
