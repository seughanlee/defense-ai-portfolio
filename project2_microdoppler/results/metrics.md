# Evaluation (seeds [0, 1, 2]; test n=2000, OOD n=400)

## Main metrics

| Model | Test acc | Macro F1 | Balanced acc, \|v\|<=15 m/s | ECE (raw → TS) | Unknown AUROC (MSP) | FPR@95TPR (MSP) |
|---|---|---|---|---|---|---|
| CNN (mean ± std over seeds) | 0.835 ± 0.010 | 0.832 ± 0.011 | 0.819 ± 0.016 | 0.020 ± 0.006 → 0.021 ± 0.005 | 0.739 ± 0.020 | 0.906 ± 0.019 |
| CNN 3-seed ensemble | 0.848 | 0.845 | 0.834 | 0.023 | 0.733 | 0.917 |
| Features + RF | 0.799 | 0.803 | 0.765 | 0.078 | 0.739 | 0.920 |

Energy score (CNN): AUROC 0.606 ± 0.040, FPR@95TPR 0.939 ± 0.028. Temperatures: 0.67, 0.64, 0.70

## Per-class recall

| Model | quadcopter | hexacopter | fixed_wing_uav | helicopter | bird |
|---|---|---|---|---|---|
| CNN ensemble | 0.530 | 0.797 | 0.917 | 0.993 | 1.000 |
| Features + RF | 0.565 | 0.595 | 0.850 | 0.995 | 0.993 |

## Unknown target (octocopter) — what the ensemble calls it

| quadcopter | hexacopter | fixed_wing_uav | helicopter | bird |
|---|---|---|---|---|
| 12.5% | 84.8% | 2.8% | 0.0% | 0.0% |

## Accuracy vs SNR

| Model | -15 dB | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---|---|---|---|---|---|---|---|
| CNN ensemble | 0.224 | 0.446 | 0.686 | 0.806 | 0.844 | 0.918 | 0.928 | 0.914 |
| Features + RF | 0.216 | 0.338 | 0.690 | 0.758 | 0.804 | 0.846 | 0.874 | 0.854 |

## Deployment (seed-0 model, single CPU thread, batch 1)

Parameters: 199,509. PyTorch latency: median 17.36 ms (p99 29.99 ms)

| Format | Size [KB] | Median latency [ms] | p99 [ms] | Test acc | Max \|Δlogit\| vs PyTorch | Top-1 agreement |
|---|---|---|---|---|---|---|
| onnx_fp32 | 782 | 2.64 | 3.41 | 0.833 | 0.0000 | 1.000 |
| onnx_int8 | 218 | 1.72 | 2.19 | 0.839 | 9.5076 | 0.967 |
