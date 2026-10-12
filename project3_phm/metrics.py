"""Point and interval metrics (EVALUATION.md sec. 4)."""
import numpy as np

from prep import CAP

RUL_BINS = [(0, 25), (25, 50), (50, 75), (75, CAP), (CAP, np.inf)]


def rmse(pred, true):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def mae(pred, true):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


def nasa_score(pred, true):
    """Per-sample asymmetric score; late predictions (d > 0) are penalized more."""
    d = np.asarray(pred, float) - np.asarray(true, float)
    return np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)


def point_metrics(pred, true, cap=CAP):
    s = nasa_score(pred, true)
    return dict(rmse=rmse(pred, true), rmse_capped=rmse(pred, np.minimum(true, cap)), mae=mae(pred, true),
                nasa_sum=float(s.sum()), nasa_mean=float(s.mean()))


def interval_metrics(lo, hi, true, alpha=0.1):
    lo, hi, true = map(np.asarray, (lo, hi, true))
    inside = (true >= lo) & (true <= hi)
    width = hi - lo
    winkler = width + (2 / alpha) * ((lo - true) * (true < lo) + (true - hi) * (true > hi))
    cond = {}
    for a, b in RUL_BINS:
        m = (true >= a) & (true < b)
        cond[f"{a}-{b}"] = (float(inside[m].mean()) if m.any() else float("nan"), int(m.sum()))
    return dict(coverage=float(inside.mean()), width=float(width.mean()), winkler=float(winkler.mean()), conditional=cond)


def engine_bootstrap(stat_fn, units, n_boot=1000, seed=0):
    """95 % CI of stat_fn(index_array) resampling whole engines (points within an engine are correlated)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(units)
    groups = {u: np.flatnonzero(units == u) for u in uniq}
    vals = [stat_fn(np.concatenate([groups[u] for u in rng.choice(uniq, len(uniq))])) for _ in range(n_boot)]
    return tuple(np.percentile(vals, [2.5, 97.5]))
