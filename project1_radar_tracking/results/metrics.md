Monte Carlo runs: 100, seed: 0

NEES 95% band (per step, averaged over runs): [3.46, 4.57]

| Filter | Pos RMSE [m] | Straight | Turn | Turn onset (5 s) | Peak RMSE | Vel RMSE [m/s] | Avg NEES (ideal 4) | Peak NEES |
|---|---|---|---|---|---|---|---|---|
| Raw measurement | 239.6 | 240.7 | 234.9 | - | - | - | - | - |
| CV-KF (high q) | 148.4 | 142.0 | 172.3 | 135.8 | 227.0 | 50.5 | 2.6 | 6.2 |
| IMM CV+CV (v1) | 137.1 | 118.9 | 194.6 | 114.6 | 293.2 | 45.9 | 3.1 | 13.3 |
| IMM CV+CT fixed | 108.5 | 105.8 | 118.7 | 88.5 | 174.6 | 30.9 | 2.8 | 11.8 |
| IMM CV+CT-EKF | 111.1 | 106.0 | 130.0 | 91.0 | 206.5 | 33.0 | 2.8 | 17.5 |
| IMM CV+CT-EKF+CVhigh | 115.6 | 109.5 | 138.1 | 102.6 | 215.5 | 34.1 | 2.4 | 13.3 |
