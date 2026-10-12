"""Hand-crafted micro-Doppler features for the classical baseline."""
import numpy as np

from config import DYN_RANGE_DB, N_FREQ
from dataset import FREQ_KHZ

BIN_KHZ = FREQ_KHZ[1] - FREQ_KHZ[0]
BODY_GUARD = 2      # bins on each side of the body line excluded from micro-motion energy


def handcrafted(X):
    """X: (N, freq, time) normalized spectrograms -> (N, n_features)."""
    N = len(X)
    rows = np.arange(N)
    P = 10 ** ((X - 1) * DYN_RANGE_DB / 10)                 # linear power relative to peak
    prof = P.mean(2)
    prof_n = prof / prof.sum(1, keepdims=True)
    prof_db = 10 * np.log10(prof)

    # Doppler spread around the power centroid (50/90/99 % energy)
    centroid = (prof_n * FREQ_KHZ).sum(1)
    dev = np.abs(FREQ_KHZ[None, :] - centroid[:, None])
    order = np.argsort(dev, 1)
    cum = np.cumsum(np.take_along_axis(prof_n, order, 1), 1)
    dev_sorted = np.take_along_axis(dev, order, 1)
    spread = [dev_sorted[rows, (cum >= q).argmax(1)] for q in (0.5, 0.9, 0.99)]

    # Occupied bandwidth above the noise floor
    floor = np.median(prof_db, 1, keepdims=True)
    occupied = ((prof_db > floor + 6).sum(1) * BIN_KHZ, (prof_db > floor + 12).sum(1) * BIN_KHZ)

    # Micro-motion energy over time (body line removed): flashes and cadence
    peak = prof.argmax(1)
    mask = np.abs(np.arange(N_FREQ)[None, :] - peak[:, None]) > BODY_GUARD
    micro = (P * mask[:, :, None]).sum(1)                     # (N, time)
    micro_ratio = micro.mean(1) / P.sum(1).mean(1)
    micro_cv = micro.std(1) / (micro.mean(1) + 1e-12)
    cadence = np.abs(np.fft.rfft(micro - micro.mean(1, keepdims=True), axis=1))[:, 1:]
    cadence_n = cadence / (cadence.sum(1, keepdims=True) + 1e-12)
    cadence_peak = cadence.argmax(1)
    cadence_ratio = cadence.max(1) / (cadence.mean(1) + 1e-12)
    cadence_bands = cadence_n[:, :64].reshape(N, 16, 4).sum(2)

    # Spectral entropy per frame
    p_t = P / P.sum(1, keepdims=True)
    H = -(p_t * np.log(p_t + 1e-20)).sum(1)

    # Mean Doppler profile centered on the body line, coarse 32-bin shape
    shift = N_FREQ // 2 - peak
    idx = (np.arange(N_FREQ)[None, :] - shift[:, None]) % N_FREQ
    centered = np.take_along_axis(prof_db, idx, 1).reshape(N, 32, 4).mean(2)

    scalars = np.column_stack([centroid, *spread, *occupied, micro_ratio, micro_cv,
                               cadence_peak, cadence_ratio, H.mean(1), H.std(1)])
    return np.hstack([scalars, cadence_bands, centered]).astype(np.float32)
