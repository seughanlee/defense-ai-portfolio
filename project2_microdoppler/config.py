"""Radar, spectrogram and dataset parameters."""
C = 3e8
FC = 10e9                       # X-band carrier [Hz]
LAMBDA = C / FC                 # 0.03 m
PRF = 40e3                      # complex slow-time sampling [Hz] -> unambiguous Doppler +-20 kHz
DWELL = 0.5                     # observation time [s]
N_SAMPLES = int(PRF * DWELL)    # 20000

STFT_WIN = 256                  # 6.4 ms Hann window
N_FRAMES = 128
N_FREQ = 128                    # 256 FFT bins pooled 2:1 -> 312.5 Hz per bin
DYN_RANGE_DB = 60.0

TIP_SPEED_MAX = 240.0           # m/s, keeps propeller tips subsonic

CLASSES = ["quadcopter", "hexacopter", "fixed_wing_uav", "helicopter", "bird"]
OOD_CLASS = "octocopter"        # never seen in training; used for unknown-target detection

SNR_TRAIN_DB = (-5.0, 20.0)     # per-sample SNR before STFT integration
SNR_SWEEP_DB = [-15, -10, -5, 0, 5, 10, 15, 20]

# split name -> (classes, samples per class, seed, snr: range tuple or list of fixed levels)
SPLITS = {
    "train": (CLASSES, 1000, 1, SNR_TRAIN_DB),
    "val": (CLASSES, 200, 2, SNR_TRAIN_DB),
    "test": (CLASSES, 400, 3, SNR_TRAIN_DB),
    "snr_sweep": (CLASSES, 100, 4, SNR_SWEEP_DB),
    "ood": ([OOD_CLASS], 400, 5, SNR_TRAIN_DB),
}
