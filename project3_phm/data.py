"""C-MAPSS loading, RUL labels and operating-condition handling."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data" / "cmapss"
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SETTINGS = ["setting1", "setting2", "setting3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle"] + SETTINGS + SENSORS

# Saxena et al. (2008), Table 2. The paper lists the 21 outputs in this order; mapping them to
# columns 6-26 in the same order is the standard convention (the paper does not state it explicitly).
SENSOR_INFO = {
    "s1": ("T2", "Total temperature at fan inlet", "°R"),
    "s2": ("T24", "Total temperature at LPC outlet", "°R"),
    "s3": ("T30", "Total temperature at HPC outlet", "°R"),
    "s4": ("T50", "Total temperature at LPT outlet", "°R"),
    "s5": ("P2", "Pressure at fan inlet", "psia"),
    "s6": ("P15", "Total pressure in bypass-duct", "psia"),
    "s7": ("P30", "Total pressure at HPC outlet", "psia"),
    "s8": ("Nf", "Physical fan speed", "rpm"),
    "s9": ("Nc", "Physical core speed", "rpm"),
    "s10": ("epr", "Engine pressure ratio (P50/P2)", "-"),
    "s11": ("Ps30", "Static pressure at HPC outlet", "psia"),
    "s12": ("phi", "Ratio of fuel flow to Ps30", "pps/psi"),
    "s13": ("NRf", "Corrected fan speed", "rpm"),
    "s14": ("NRc", "Corrected core speed", "rpm"),
    "s15": ("BPR", "Bypass ratio", "-"),
    "s16": ("farB", "Burner fuel-air ratio", "-"),
    "s17": ("htBleed", "Bleed enthalpy", "-"),
    "s18": ("Nf_dmd", "Demanded fan speed", "rpm"),
    "s19": ("PCNfR_dmd", "Demanded corrected fan speed", "rpm"),
    "s20": ("W31", "HPT coolant bleed", "lbm/s"),
    "s21": ("W32", "LPT coolant bleed", "lbm/s"),
}


def _read(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def load_subset(name):
    """Return (train, test, test_rul). train gets an uncapped 'rul' column (cycles to failure)."""
    train = _read(DATA / f"train_{name}.txt")
    test = _read(DATA / f"test_{name}.txt")
    rul = pd.read_csv(DATA / f"RUL_{name}.txt", sep=r"\s+", header=None)[0].to_numpy()
    life = train.groupby("unit")["cycle"].transform("max")
    train["rul"] = life - train["cycle"]
    return train, test, rul


def operating_condition(df):
    """Discrete operating-condition id from rounded settings (FD002/FD004 fly six regimes)."""
    # "+ 0.0" turns -0.0 into 0.0; without it, near-zero settings split one regime into two keys.
    s1 = df["setting1"].round(0).astype(int)
    s2 = (df["setting2"].round(2) + 0.0).map("{:.2f}".format)
    s3 = df["setting3"].round(0).astype(int)
    return s1.astype(str) + "_" + s2 + "_" + s3.astype(str)


def capped_rul(rul, cap):
    """Piecewise-linear target: constant at `cap` early in life, then linear to zero."""
    return np.minimum(rul, cap)
