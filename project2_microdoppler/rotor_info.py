"""v2 step 1: is rotor count present in the input at all, and which input settings carry it?

The same 900 multicopters (quad/hexa/octo, 300 each; the random draws do not depend on the settings) are observed
with several (dwell, STFT hop) settings. From each observation we compute a cadence spectrum (CVD collapsed over
Doppler, body line removed) and measure how well a linear classifier recovers the rotor count (5-fold CV).

Hypotheses
- H-alias: v1 frames at 258 Hz, so blade-flash rates of 120-260 Hz (2 blades x 60-130 rev/s) alias.
  A smaller hop (higher frame rate) should raise the probe accuracy.
- H-res  : rotors differ in speed by about 5 %, so their flash lines are a few Hz apart. Cadence resolution is
  1 / dwell; a longer dwell should raise the probe accuracy.

Usage: python rotor_info.py     # writes results/v2_rotor_info.md and results/v2_cadence.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import PRF, STFT_WIN
from simulate import simulate

RESULTS = Path(__file__).parent / "results"
CLASSES = ["quadcopter", "hexacopter", "octocopter"]
N_PER_CLASS = 300
SNR_DB = [10.0, 0.0]
SETTINGS = [  # (label, dwell [s], hop [samples])
    ("v1: 0.5 s, 258 fps", 0.5, 155),
    ("0.5 s, 625 fps", 0.5, 64),
    ("0.5 s, 1250 fps", 0.5, 32),
    ("1.0 s, 625 fps", 1.0, 64),
    ("2.0 s, 625 fps", 2.0, 64),
]
GRID = np.arange(1, 601)          # cadence grid [Hz], 1 Hz steps
BODY_GUARD = 2
_WIN = np.hanning(STFT_WIN)


def cadence_spectrum(s, hop):
    n_frames = 1 + (len(s) - STFT_WIN) // hop
    idx = np.arange(STFT_WIN)[None, :] + hop * np.arange(n_frames)[:, None]
    P = np.abs(np.fft.fft(s[idx] * _WIN, axis=1)) ** 2                  # (frames, Doppler bins), linear power
    body = int(P.mean(0).argmax())
    keep = np.abs((np.arange(STFT_WIN) - body + STFT_WIN // 2) % STFT_WIN - STFT_WIN // 2) > BODY_GUARD
    P = P[:, keep]
    C = np.abs(np.fft.rfft(P - P.mean(0), axis=0)).sum(1)              # cadence spectrum, summed over Doppler
    nu = np.fft.rfftfreq(n_frames, hop / PRF)
    c = np.interp(GRID, nu, C, right=0.0)                               # beyond Nyquist: no information
    return np.log(c / (c.sum() + 1e-12) + 1e-6)


ALIGN_HALF = 40                   # Hz kept on each side of the dominant flash line


def aligned(spec):
    """Shift-invariant view: cut a window centred on the dominant cadence line (above 20 Hz), so the probe sees
    the *shape* around the flash line (how many neighbouring lines) instead of where the line happens to sit."""
    lo = 20
    peak = lo + int(np.argmax(spec[lo:]))
    padded = np.pad(spec, ALIGN_HALF, constant_values=spec.min())
    return padded[peak:peak + 2 * ALIGN_HALF + 1]


def main():
    rows, spectra = [], {}
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for snr in SNR_DB:
        for label, dwell, hop in SETTINGS:
            rng = np.random.default_rng(42)            # same physical targets for every setting
            X, y = [], []
            for k, target in enumerate(CLASSES):
                for _ in range(N_PER_CLASS):
                    s, _ = simulate(target, snr, rng, dwell=dwell)
                    X.append(cadence_spectrum(s, hop))
                    y.append(k)
            X, y = np.array(X), np.array(y)
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000))
            acc = cross_val_score(clf, X, y, cv=cv)
            acc_al = cross_val_score(clf, np.array([aligned(x) for x in X]), y, cv=cv)
            nyq = PRF / hop / 2
            rows.append(f"| {snr:g} dB | {label} | {nyq:.0f} Hz | {1 / dwell:.1f} Hz | {acc.mean():.3f} ± {acc.std():.3f} | "
                        f"{acc_al.mean():.3f} ± {acc_al.std():.3f} |")
            print(rows[-1], flush=True)
            if snr == SNR_DB[0]:
                spectra[label] = (X, y)

    lines = ["# v2 step 1: rotor-count information in the input (quad / hexa / octo)\n",
             f"{N_PER_CLASS} targets per class, identical physical targets across settings. Feature: cadence spectrum "
             "(CVD summed over Doppler, body line removed) on a 1-600 Hz grid. Linear probe: standardized logistic "
             "regression, 5-fold CV. Chance = 0.333. Reference: the v1 CNN's own features gave 0.496 (seed 0).\n",
             "Aligned = window of +-40 Hz centred on each sample's dominant flash line (shift-invariant).\n",
             "| SNR | Setting | Cadence Nyquist | Cadence resolution | Probe (absolute position) | Probe (aligned on flash line) |",
             "|---|---|---|---|---|---|", *rows]
    (RESULTS / "v2_rotor_info.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    for ax, label in zip(axes, ["v1: 0.5 s, 258 fps", "0.5 s, 1250 fps", "2.0 s, 625 fps"]):
        X, y = spectra[label]
        for k, name in enumerate(CLASSES):
            ax.plot(GRID, X[y == k].mean(0), lw=1, label=name)
        ax.set(title=label, xlabel="Cadence frequency [Hz]", xlim=(0, 600))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean log cadence spectrum")
    axes[0].legend()
    fig.suptitle("Class-mean cadence spectra (10 dB): blade-flash lines appear only when the frame rate is high enough")
    fig.tight_layout()
    fig.savefig(RESULTS / "v2_cadence.png", dpi=110)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
