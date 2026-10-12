"""Step 4: interval construction. All methods produce 90 % intervals on the capped-RUL scale (EVALUATION.md,
amended: the upper end 125 reads as "125 cycles or more").

Methods (deep = 5-seed ensemble; GBM supports the conformal variants only, it has no spread estimate)
- MC Dropout            : 5 % / 95 % quantiles of N_MC stochastic passes of one model (dropout on)
- Deep ensemble         : ensemble mean ± z * std across seeds (spread between models only)
- Conformal (fixed)     : mean ± q, q = conformal quantile of |residual| on calibration engines
- Conformal (asymmetric): separate conformal quantiles for under- and over-prediction (alpha/2 each)
- Conformal (normalized): mean ± q * (std + beta), q from |residual| / (std + beta)
- Conformal (Mondrian)  : fixed-width conformal with a separate q per *predicted*-RUL bin
"""
import numpy as np
import torch

from prep import CAP
from train import predict

ALPHA = 0.10
N_MC = 50
PRED_BINS = [0, 50, 100, np.inf]
MIN_PER_BIN = 10


def conformal_quantile(scores, alpha):
    """Finite-sample split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest score."""
    scores = np.sort(np.asarray(scores))
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(scores[min(k, n) - 1])


@torch.no_grad()
def mc_dropout(model, X, n=N_MC, seed=0):
    torch.manual_seed(seed)
    model.train()                    # dropout active (the models have no batch-norm layers)
    samples = np.stack([model(torch.from_numpy(X)).numpy() for _ in range(n)])
    model.eval()
    return samples


def deep_outputs(models, X, with_mc=False):
    P = np.stack([predict(m, X) for m in models])
    out = dict(mu=P.mean(0), sd=P.std(0))
    if with_mc:
        out["mc"] = mc_dropout(models[0], X)
    return out


def build_intervals(cal, ev, y_cal, alpha=ALPHA):
    """cal/ev: dicts with 'mu' and optionally 'sd' (ensemble spread) and 'mc' (MC samples, ev only).
    Returns {method: (lo, hi)} on the evaluation points, clipped to [0, CAP]."""
    z = {0.5: 0.6745, 0.6: 0.8416, 0.7: 1.0364, 0.8: 1.2816, 0.9: 1.6449, 0.95: 1.9600}[round(1 - alpha, 2)]
    mu_c, mu_e = cal["mu"], ev["mu"]
    r = y_cal - mu_c
    out = {}
    if "mc" in ev:
        out["MC Dropout"] = (np.quantile(ev["mc"], alpha / 2, 0), np.quantile(ev["mc"], 1 - alpha / 2, 0))
    if "sd" in ev:
        out["Deep ensemble"] = (mu_e - z * ev["sd"], mu_e + z * ev["sd"])

    q = conformal_quantile(np.abs(r), alpha)
    out["Conformal (fixed)"] = (mu_e - q, mu_e + q)

    q_under = conformal_quantile(-r, alpha / 2)          # how far above the truth predictions go
    q_over = conformal_quantile(r, alpha / 2)            # how far below the truth predictions go
    out["Conformal (asymmetric)"] = (mu_e - q_under, mu_e + q_over)

    if "sd" in cal:
        beta = float(np.median(cal["sd"]))
        qn = conformal_quantile(np.abs(r) / (cal["sd"] + beta), alpha)
        out["Conformal (normalized)"] = (mu_e - qn * (ev["sd"] + beta), mu_e + qn * (ev["sd"] + beta))

    lo, hi = mu_e - q, mu_e + q
    for a, b in zip(PRED_BINS[:-1], PRED_BINS[1:]):
        mc_, me_ = (mu_c >= a) & (mu_c < b), (mu_e >= a) & (mu_e < b)
        qb = conformal_quantile(np.abs(r[mc_]), alpha) if mc_.sum() >= MIN_PER_BIN else q
        lo[me_], hi[me_] = mu_e[me_] - qb, mu_e[me_] + qb
    out["Conformal (Mondrian)"] = (lo, hi)
    return {k: (np.clip(l, 0, CAP), np.clip(h, 0, CAP)) for k, (l, h) in out.items()}


def safety_metrics(lo, hi, y, units):
    """Split the misses by direction and look at per-engine coverage.
    below = true RUL under the lower bound: the engine fails earlier than promised (dangerous).
    above = true RUL over the upper bound: maintenance would come too early (costly, not dangerous)."""
    inside = (y >= lo) & (y <= hi)
    per_engine = np.array([inside[units == u].mean() for u in np.unique(units)])
    return dict(below=float((y < lo).mean()), above=float((y > hi).mean()),
                engine_min=float(per_engine.min()), engine_p10=float(np.percentile(per_engine, 10)),
                engines_below_80=int((per_engine < 0.8).sum()), n_engines=len(per_engine))
