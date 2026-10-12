# Off-design robustness (30 runs, seed 0)

Both turns use the same load factor. Filter parameters are unchanged (tuned around 2-3g).

Cells: turn-segment position RMSE [m] / peak NEES

| Filter | 1g | 2g | 3g | 4g | 5g | 6g |
|---|---|---|---|---|---|---|
| CV-KF (high q) | 140 / 3.3 | 160 / 4.5 | 172 / 6.1 | 190 / 8.8 | 230 / 13.2 | 267 / 17.4 |
| IMM CV+CV (v1) | 158 / 7.5 | 176 / 13.8 | 191 / 22.6 | 217 / 35.2 | 285 / 72.2 | 298 / 77.3 |
| IMM CV+CT fixed | 122 / 4.7 | 103 / 7.9 | 121 / 13.5 | 354 / 181.2 | 700 / 798.3 | 1275 / 4037.3 |
| IMM CV+CT-EKF | 123 / 5.2 | 116 / 11.3 | 136 / 24.9 | 216 / 68.1 | 589 / 384.8 | 443 / 497.1 |
| IMM CV+CT-EKF+CVhigh | 133 / 6.0 | 129 / 10.1 | 141 / 16.9 | 173 / 26.2 | 265 / 66.3 | 247 / 69.5 |
