"""Spectrogram computation and cached dataset generation.

Usage: python dataset.py          # generate all splits into data/ and save a spectrogram gallery
"""
from pathlib import Path

import numpy as np

from config import CLASSES, DYN_RANGE_DB, N_FRAMES, N_FREQ, N_SAMPLES, OOD_CLASS, PRF, SPLITS, STFT_WIN
from simulate import simulate

DATA = Path(__file__).parent / "data"
RESULTS = Path(__file__).parent / "results"
HOP = (N_SAMPLES - STFT_WIN) // (N_FRAMES - 1)
_IDX = np.arange(STFT_WIN)[None, :] + HOP * np.arange(N_FRAMES)[:, None]
_WIN = np.hanning(STFT_WIN)
FREQ_KHZ = np.linspace(-PRF / 2, PRF / 2, N_FREQ, endpoint=False) / 1e3 + PRF / N_FREQ / 2e3


def spectrogram(s):
    """Log-power spectrogram (freq x time), normalized to [0, 1] over DYN_RANGE_DB below the peak."""
    spec = np.fft.fftshift(np.fft.fft(s[_IDX] * _WIN, axis=1), axes=1)
    P = (np.abs(spec) ** 2).reshape(N_FRAMES, N_FREQ, 2).sum(-1).T
    db = 10 * np.log10(P + 1e-20)
    db -= db.max()
    return np.clip(db, -DYN_RANGE_DB, 0) / DYN_RANGE_DB + 1


def generate(classes, n_per_class, seed, snr):
    """snr is a (low, high) range sampled uniformly, or a list of levels (n_per_class at each level)."""
    rng = np.random.default_rng(seed)
    X, y, snrs, v, tip = [], [], [], [], []
    for label, target in enumerate(classes):
        levels = [lvl for lvl in snr for _ in range(n_per_class)] if isinstance(snr, list) else None
        for i in range(len(levels) if levels else n_per_class):
            snr_db = levels[i] if levels else rng.uniform(*snr)
            s, meta = simulate(target, snr_db, rng)
            X.append(np.round(spectrogram(s) * 255).astype(np.uint8))   # 0.24 dB quantization
            y.append(label)
            snrs.append(snr_db)
            v.append(meta["v_radial"])
            tip.append(meta["tip_speed"])
    return dict(X=np.array(X), y=np.array(y), snr=np.array(snrs), v_radial=np.array(v),
                tip_speed=np.array(tip))


def load(split):
    """Load a cached split; X is float32 in [0, 1] with shape (N, freq, time)."""
    path = DATA / f"{split}.npz"
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        np.savez(path, **generate(*SPLITS[split]))
    d = dict(np.load(path))
    d["X"] = d["X"].astype(np.float32) / 255
    return d


def save_gallery(per_class=3, seed=123):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = CLASSES + [OOD_CLASS]
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(per_class, len(names), figsize=(2.6 * len(names), 2.3 * per_class),
                             sharex=True, sharey=True)
    extent = [0, N_SAMPLES / PRF, FREQ_KHZ[0], FREQ_KHZ[-1]]
    for j, name in enumerate(names):
        for i in range(per_class):
            s, meta = simulate(name, 15.0, rng)
            ax = axes[i, j]
            ax.imshow(spectrogram(s), origin="lower", aspect="auto", extent=extent, cmap="viridis")
            ax.set_title(f"{name}" if i == 0 else "", fontsize=9)
            ax.text(0.02, 0.95, f"tip {meta['tip_speed']:.0f} m/s", color="w", fontsize=7,
                    transform=ax.transAxes, va="top")
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Doppler [kHz]")
    fig.suptitle("Simulated micro-Doppler spectrograms (SNR 15 dB; octocopter = unseen class)")
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "gallery.png", dpi=110)


if __name__ == "__main__":
    import time
    for split in SPLITS:
        t0 = time.time()
        d = load(split)
        print(f"{split:10s} {d['X'].shape}  ({time.time() - t0:.1f} s)")
    save_gallery()
    print(f"Saved {RESULTS / 'gallery.png'}")
