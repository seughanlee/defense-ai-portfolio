# v2 step 1: rotor-count information in the input (quad / hexa / octo)

300 targets per class, identical physical targets across settings. Feature: cadence spectrum (CVD summed over Doppler, body line removed) on a 1-600 Hz grid. Linear probe: standardized logistic regression, 5-fold CV. Chance = 0.333. Reference: the v1 CNN's own features gave 0.496 (seed 0).

Aligned = window of +-40 Hz centred on each sample's dominant flash line (shift-invariant).

| SNR | Setting | Cadence Nyquist | Cadence resolution | Probe (absolute position) | Probe (aligned on flash line) |
|---|---|---|---|---|---|
| 10 dB | v1: 0.5 s, 258 fps | 129 Hz | 2.0 Hz | 0.439 ± 0.035 | 0.477 ± 0.036 |
| 10 dB | 0.5 s, 625 fps | 312 Hz | 2.0 Hz | 0.442 ± 0.050 | 0.488 ± 0.028 |
| 10 dB | 0.5 s, 1250 fps | 625 Hz | 2.0 Hz | 0.480 ± 0.030 | 0.489 ± 0.022 |
| 10 dB | 1.0 s, 625 fps | 312 Hz | 1.0 Hz | 0.432 ± 0.024 | 0.517 ± 0.024 |
| 10 dB | 2.0 s, 625 fps | 312 Hz | 0.5 Hz | 0.418 ± 0.013 | 0.509 ± 0.021 |
| 0 dB | v1: 0.5 s, 258 fps | 129 Hz | 2.0 Hz | 0.326 ± 0.022 | 0.362 ± 0.032 |
| 0 dB | 0.5 s, 625 fps | 312 Hz | 2.0 Hz | 0.350 ± 0.014 | 0.336 ± 0.038 |
| 0 dB | 0.5 s, 1250 fps | 625 Hz | 2.0 Hz | 0.344 ± 0.028 | 0.334 ± 0.013 |
| 0 dB | 1.0 s, 625 fps | 312 Hz | 1.0 Hz | 0.326 ± 0.017 | 0.340 ± 0.034 |
| 0 dB | 2.0 s, 625 fps | 312 Hz | 0.5 Hz | 0.343 ± 0.037 | 0.378 ± 0.022 |

## Nonlinear check (gradient boosting on the aligned feature, 10 dB)
_The sections below were run separately and added by hand; re-running rotor_info.py rewrites only the table above._
| Setting | Probe accuracy |
|---|---|
| v1: 0.5 s, 258 fps | 0.428 +- 0.031 |
| 2.0 s, 625 fps | 0.508 +- 0.010 |

## Conclusion
Neither hypothesis explains the v1 failure. A higher frame rate adds at most 4 points, a longer dwell adds about
4 points only once the feature is aligned on the flash line, and a nonlinear probe does not beat the linear one.
Rotor count is weakly present in this simulator's signal even under ideal settings (about 0.5 against a 0.333
chance at 10 dB, chance level at 0 dB): identical rotors of the same size and speed superpose, and the count leaves
little trace. v2 therefore changes direction from "learn a better representation" to "answer with a guaranteed
set when the classes are not separable" (conformal prediction sets, the method carried over from project 3).
