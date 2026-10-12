# Unknown-target detection (octocopter, n=400) vs in-distribution test (n=2000)

Seeds [0, 1, 2]. Features: 96-d penultimate layer. kNN k=50. Covariance condition number before shrinkage: 67909 (mean over seeds).

| Score | Space | AUROC (mean ± std) | FPR@95TPR (mean ± std) |
|---|---|---|---|
| MSP | output | 0.739 ± 0.020 | 0.906 ± 0.019 |
| Energy | output | 0.606 ± 0.040 | 0.939 ± 0.028 |
| Mahalanobis | feature | 0.478 ± 0.029 | 0.978 ± 0.001 |
| kNN | feature | 0.521 ± 0.028 | 0.990 ± 0.004 |
