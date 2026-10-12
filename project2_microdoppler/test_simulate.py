"""Physics sanity checks for the simulator. Run: python test_simulate.py"""
import numpy as np

from config import LAMBDA, N_FRAMES, N_SAMPLES, PRF
from dataset import FREQ_KHZ, spectrogram
from simulate import rotor_return, simulate


def test_max_doppler_matches_tip_speed():
    """Blade-flash Doppler extent should be 2 * v_tip / lambda."""
    for L, tip in [(0.15, 90.0), (6.0, 210.0)]:
        s = 1.0 + rotor_return(L, tip / L, 2, 0.3, 1.0)
        S = spectrogram(s)
        occupied = FREQ_KHZ[(S > 1 - 40 / 60).any(1)]
        expected_khz = 2 * tip / LAMBDA / 1e3
        measured = max(abs(occupied.min()), occupied.max())
        assert abs(measured - expected_khz) / expected_khz < 0.08, (L, measured, expected_khz)


def test_blade_flash_rate():
    """Even blade count N flashes N times per revolution (opposite blades flash together)."""
    rev_hz, n_blades = 5.0, 2
    L = 6.0
    s = rotor_return(L, 2 * np.pi * rev_hz, n_blades, 0.0, 1.0)
    S = spectrogram(s)
    frame_power = (10 ** ((S - 1) * 6)).sum(0)
    frame_rate = N_FRAMES / (N_SAMPLES / PRF)
    spec = np.abs(np.fft.rfft(frame_power - frame_power.mean()))
    freqs = np.fft.rfftfreq(N_FRAMES, 1 / frame_rate)
    assert abs(freqs[spec.argmax()] - n_blades * rev_hz) <= freqs[1], freqs[spec.argmax()]


def test_snr_definition():
    rng = np.random.default_rng(0)
    for snr_db in [-10.0, 0.0, 20.0]:
        p = np.mean([np.mean(np.abs(simulate("quadcopter", snr_db, rng)[0]) ** 2) for _ in range(20)])
        assert abs(p - (1 + 10 ** (-snr_db / 10))) / p < 0.05, (snr_db, p)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
