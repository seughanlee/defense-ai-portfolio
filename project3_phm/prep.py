"""Engine-level splits, normalization, windowing and test-like evaluation points (EVALUATION.md sec. 2-3)."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data import SENSORS, load_subset, operating_condition

SPLIT_FRACS = (0.60, 0.15, 0.25)     # train / val / calib
SPLIT_SEED = 0
WINDOW = 30
CAP = 125
MIN_OBSERVED = 31                    # shortest FD001 test trajectory; truncation points start here
K_TRUNC = 10


def split_units(units, seed=SPLIT_SEED, fracs=SPLIT_FRACS):
    units = np.random.default_rng(seed).permutation(np.asarray(sorted(units)))
    n_tr = int(round(fracs[0] * len(units)))
    n_va = int(round(fracs[1] * len(units)))
    return {"train": np.sort(units[:n_tr]), "val": np.sort(units[n_tr:n_tr + n_va]),
            "calib": np.sort(units[n_tr + n_va:])}


def informative_sensors(df, max_levels=2):
    """Drop sensors that only show quantization (<= max_levels values) within every condition."""
    levels = df.groupby("cond")[SENSORS].nunique().max()
    return [s for s in SENSORS if levels[s] > max_levels]


@dataclass
class Normalizer:
    """z-score per operating condition (by_cond=True) or with one global mean/std. Fit on train engines only."""
    sensors: list
    by_cond: bool = True
    stats: dict = field(default_factory=dict)

    def _key(self, df):
        return df["cond"] if self.by_cond else pd.Series("all", index=df.index)

    def fit(self, df):
        for key, g in df.groupby(self._key(df)):
            self.stats[key] = (g[self.sensors].mean(), g[self.sensors].std().replace(0, 1.0))
        return self

    def transform(self, df):
        out = df.copy()
        out[self.sensors] = out[self.sensors].astype(np.float64)   # some sensors (e.g. s17) are stored as int
        keys = self._key(df)
        unknown = set(keys) - set(self.stats)
        if unknown:
            raise ValueError(f"operating conditions not seen in training: {sorted(unknown)}")
        for key, (mu, sd) in self.stats.items():
            m = keys == key
            out.loc[m, self.sensors] = (out.loc[m, self.sensors] - mu) / sd
        return out


@dataclass
class Prepared:
    subset: str
    sensors: list
    conds: list                     # condition ids used for one-hot channels ([] if single condition)
    splits: dict                    # name -> unit ids (train file)
    train_df: pd.DataFrame          # normalized train file (all splits) with 'rul'
    test_df: pd.DataFrame           # normalized official test file
    test_rul: np.ndarray
    norm: Normalizer

    @property
    def n_channels(self):
        return len(self.sensors) + len(self.conds)


def prepare(subset, seed=SPLIT_SEED, norm_source=None, per_condition=True):
    """Load, split and normalize a subset. norm_source: another Prepared whose sensors/normalizer to reuse
    (used for the FD001 -> FD003 shift experiment, where nothing may be fit on the target subset).
    per_condition=False uses one global mean/std (comparison for FD004)."""
    train, test, rul = load_subset(subset)
    for df in (train, test):
        df["cond"] = operating_condition(df)
    splits = split_units(train["unit"].unique(), seed)
    if norm_source is None:
        fit_rows = train[train["unit"].isin(splits["train"])]
        sensors = informative_sensors(fit_rows)
        norm = Normalizer(sensors, by_cond=per_condition).fit(fit_rows)
        conds = sorted(fit_rows["cond"].unique()) if fit_rows["cond"].nunique() > 1 else []
    else:
        sensors, norm, conds = norm_source.sensors, norm_source.norm, norm_source.conds
    return Prepared(subset, sensors, conds, splits, norm.transform(train), norm.transform(test), rul, norm)


def _channels(p, df):
    x = df[p.sensors].to_numpy(np.float32)
    if p.conds:
        onehot = (df["cond"].to_numpy()[:, None] == np.array(p.conds)[None, :]).astype(np.float32)
        x = np.hstack([x, onehot])
    return x


def _padded(x, W):
    """Repeat the first row so that a window can end at any cycle >= 1."""
    return np.vstack([np.repeat(x[:1], W - 1, axis=0), x])


def training_windows(p, split, W=WINDOW, cap=CAP):
    """Every window ending at every cycle of every engine in `split` (train file)."""
    X, y, units, t_end = [], [], [], []
    df = p.train_df[p.train_df["unit"].isin(p.splits[split])]
    for u, g in df.groupby("unit"):
        x = _padded(_channels(p, g), W)
        n = len(g)
        idx = np.arange(n)[:, None] + np.arange(W)[None, :]
        X.append(x[idx])
        y.append(g["rul"].to_numpy())
        units.append(np.full(n, u))
        t_end.append(g["cycle"].to_numpy())
    rul = np.concatenate(y)
    return dict(X=np.concatenate(X), y=np.minimum(rul, cap).astype(np.float32), rul=rul,
                unit=np.concatenate(units), cycle=np.concatenate(t_end))


def truncation_points(p, split, K=K_TRUNC, seed=SPLIT_SEED, W=WINDOW, min_obs=MIN_OBSERVED):
    """Test-like points from run-to-failure engines: cut each engine at K random cycles and predict
    from the last window. Returns true (uncapped) RUL = life - cut."""
    rng = np.random.default_rng(seed + 1000)
    X, rul, units, cuts = [], [], [], []
    df = p.train_df[p.train_df["unit"].isin(p.splits[split])]
    for u, g in df.groupby("unit"):
        x = _padded(_channels(p, g), W)
        life = len(g)
        candidates = np.arange(min_obs, life)               # cut after cycle c, c in [min_obs, life-1]
        for c in np.sort(rng.choice(candidates, size=min(K, len(candidates)), replace=False)):
            X.append(x[c - 1:c - 1 + W])                     # window ending at cycle c
            rul.append(life - c)
            units.append(u)
            cuts.append(c)
    return dict(X=np.stack(X), rul=np.array(rul), unit=np.array(units), cycle=np.array(cuts))


def official_test_points(p, W=WINDOW):
    """Official test set: last window of each engine, true RUL from the RUL file."""
    X, units, lengths = [], [], []
    for u, g in p.test_df.groupby("unit"):
        x = _padded(_channels(p, g), W)
        X.append(x[-W:])
        units.append(u)
        lengths.append(len(g))
    return dict(X=np.stack(X), rul=p.test_rul.astype(float), unit=np.array(units), length=np.array(lengths))
