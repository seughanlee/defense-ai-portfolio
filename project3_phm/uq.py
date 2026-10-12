"""Step 4: uncertainty quantification. Builds 90 % intervals with five methods from the same deep models
and evaluates them on validation points (or the official test set when --split test, used once at the end).

Methods
- MC Dropout            : quantiles of N_MC stochastic passes of one model (dropout kept on)
- Deep ensemble         : ensemble mean ± z * std across seeds (epistemic spread only)
- Conformal (fixed)     : ensemble mean ± q, q from calibration residuals
- Conformal (normalized): ensemble mean ± q * (std + beta), q from residuals / (std + beta)
- Conformal (Mondrian)  : fixed-width conformal with a separate q per predicted-RUL bin

Usage: python uq.py --subset FD001 --model cnn [--target capped|true] [--split val|test] [--k-calib 10]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from metrics import interval_metrics
from models import MODELS
from prep import CAP, prepare, official_test_points, truncation_points
from train import CKPT, predict

RESULTS = Path(__file__).parent / "results"
ALPHA = 0.10
N_MC = 50
Z90 = 1.6449
PRED_BINS = [0, 50, 100, np.inf]          # Mondrian bins on the *predicted* RUL (known at prediction time)


def conformal_quantile(scores, alpha=ALPHA):
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


@torch.no_grad()
def mc_dropout(model, X, n=N_MC, seed=0):
    torch.manual_seed(seed)
    model.train()                          # dropout active; the models have no batch-norm
    samples = np.stack([model(torch.from_numpy(X)).numpy() for _ in range(n)])
    model.eval()
    return samples


def load_models(subset, model, norm, n_channels):
    out = []
    for path in sorted(CKPT.glob(f"{subset}_{model}_{norm}_s*.pt")):
        m = MODELS[model](n_channels)
        m.load_state_dict(torch.load(path))
        out.append(m.eval())
    return out


def intervals(models, cal, ev, cal_target, alpha=ALPHA):
    """Return {method: (lo, hi, point)} on the evaluation points."""
    clip = lambda a: np.clip(a, 0, None)
    P_cal = np.stack([predict(m, cal["X"]) for m in models])
    P_ev = np.stack([predict(m, ev["X"]) for m in models])
    mu_cal, sd_cal = P_cal.mean(0), P_cal.std(0)
    mu_ev, sd_ev = P_ev.mean(0), P_ev.std(0)
    out = {}

    mc = mc_dropout(models[0], ev["X"])
    out["MC Dropout"] = (np.quantile(mc, alpha / 2, 0), np.quantile(mc, 1 - alpha / 2, 0), mc.mean(0))
    out["Deep ensemble"] = (mu_ev - Z90 * sd_ev, mu_ev + Z90 * sd_ev, mu_ev)

    q = conformal_quantile(np.abs(cal_target - mu_cal))
    out["Conformal (fixed)"] = (mu_ev - q, mu_ev + q, mu_ev)

    beta = float(np.median(sd_cal))
    qn = conformal_quantile(np.abs(cal_target - mu_cal) / (sd_cal + beta))
    out["Conformal (normalized)"] = (mu_ev - qn * (sd_ev + beta), mu_ev + qn * (sd_ev + beta), mu_ev)

    lo, hi = np.empty_like(mu_ev), np.empty_like(mu_ev)
    for a, b in zip(PRED_BINS[:-1], PRED_BINS[1:]):
        mc_, me_ = (mu_cal >= a) & (mu_cal < b), (mu_ev >= a) & (mu_ev < b)
        qb = conformal_quantile(np.abs(cal_target - mu_cal)[mc_]) if mc_.sum() >= 10 else q
        lo[me_], hi[me_] = mu_ev[me_] - qb, mu_ev[me_] + qb
    out["Conformal (Mondrian)"] = (lo, hi, mu_ev)
    return {k: (clip(l), h, pt) for k, (l, h, pt) in out.items()}


def coverage_curve(models, cal, ev, cal_target, ev_target, levels=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95)):
    """Empirical coverage at several nominal levels for each method."""
    curves = {}
    for lv in levels:
        for name, (lo, hi, _) in intervals(models, cal, ev, cal_target, alpha=1 - lv).items():
            curves.setdefault(name, []).append(float(((ev_target >= lo) & (ev_target <= hi)).mean()))
    return list(levels), curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--model", default="cnn")
    ap.add_argument("--norm", default="condition")
    ap.add_argument("--target", default="capped", choices=["capped", "true"])
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--k-calib", type=int, default=10)
    args = ap.parse_args()

    p = prepare(args.subset, per_condition=args.norm == "condition")
    models = load_models(args.subset, args.model, args.norm, p.n_channels)
    cal = truncation_points(p, "calib", K=args.k_calib)
    ev = truncation_points(p, "val") if args.split == "val" else official_test_points(p)
    to_target = (lambda r: np.minimum(r, CAP)) if args.target == "capped" else (lambda r: r)
    res = {}
    for name, (lo, hi, pt) in intervals(models, cal, ev, to_target(cal["rul"])).items():
        hi_eval = np.minimum(hi, CAP) if args.target == "capped" else hi
        m = interval_metrics(lo, hi_eval, to_target(ev["rul"]))
        res[name] = m
        cond = "  ".join(f"{k}:{v[0]:.2f}" for k, v in m["conditional"].items())
        print(f"{name:24s} cov {m['coverage']:.3f}  width {m['width']:6.1f}  winkler {m['winkler']:6.1f}  | {cond}")
    tag = f"{args.subset}_{args.model}_{args.norm}_{args.target}_{args.split}_k{args.k_calib}"
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"uq_{tag}.json").write_text(json.dumps(dict(args=vars(args), n_models=len(models),
                                                            n_calib=len(cal["rul"]), n_eval=len(ev["rul"]),
                                                            metrics=res), indent=1))


if __name__ == "__main__":
    main()
