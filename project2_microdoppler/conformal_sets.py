"""v2 step 2: conformal prediction sets on the v1 classifier (no retraining).

Instead of one label, the classifier returns a set that contains the true class with probability >= 1 - alpha.
Probabilities: the v1 3-seed CNN ensemble with per-seed temperature scaling (same as evaluate.py).

Data use
- RAPS hyperparameter (lambda) is chosen on val only (half fit / half check), never on test.
- Calibration and evaluation: the 2000 test samples are split 50/50 at random, R times; we report mean +- std.
- SNR sweep (includes -15/-10 dB, outside the training range) and the octocopter set are scored with
  thresholds calibrated on the whole test split: they show what happens when exchangeability breaks.

Methods
- LAC        : score 1 - p_y. Smallest average sets; coverage can be uneven across classes.
- APS        : cumulative probability up to the true class (randomized). Adapts set size to difficulty.
- RAPS       : APS + penalty lambda * (rank - k_reg)^+ to cut long tails of unlikely classes.
- LAC-class  : LAC calibrated per true class (Mondrian). Guarantees coverage for every class.
- LAC-SNR    : LAC calibrated per SNR bin (Mondrian). Assumes the radar's SNR estimate is available.

Hypotheses (written before running)
- C1: every method reaches about 90 % marginal coverage at alpha = 0.1 (the guarantee).
- C2: quad/hexa samples get two-element sets {quad, hexa} far more often than the other classes; fixed-wing,
      helicopter and bird stay mostly singletons.
- C3: marginal methods under-cover the hardest class and the lowest SNR bin; Mondrian variants fix it at the
      cost of larger sets.
- C4: below the training SNR range (-15/-10 dB) coverage drops below the target (no guarantee under shift).
- C5: octocopter (never seen) has no valid set; at best its sets contain quad or hexa, i.e. "some multicopter".

Added after seeing the first run (post hoc, reported separately)
- APS and LAC-class returned empty sets on easy classes (helicopter, bird): with an exact 90 % target the method
  spends the allowed 10 % misses there. Variant "+fallback": an empty set is replaced by the top-1 class. Sets only
  grow, so the coverage guarantee still holds (coverage can only rise).

Usage: python conformal_sets.py      # needs data/ and checkpoints/; writes results/v2_conformal.md and .png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import CLASSES, SNR_SWEEP_DB
from dataset import load
from evaluate import fit_temperature, predict_logits, softmax
from models import MicroDopplerCNN
from train import CKPT

RESULTS = Path(__file__).parent / "results"
ALPHAS = [0.1, 0.05]
R = 200                                        # random cal/eval splits of the test set
SNR_BINS = [-5, 0, 5, 10, 20.001]              # training SNR range split into 4 bins
K_REG = 1
LAMBDAS = [0.0, 0.002, 0.01, 0.05, 0.1, 0.2, 0.5]
SHORT = ["quad", "hexa", "fixed", "heli", "bird"]
QUAD, HEXA = 0, 1


def ensemble_probs(data):
    seeds = sorted(int(p.stem.split("seed")[1]) for p in CKPT.glob("cnn_seed*.pt"))
    probs = {s: [] for s in data}
    for seed in seeds:
        model = MicroDopplerCNN(len(CLASSES))
        model.load_state_dict(torch.load(CKPT / f"cnn_seed{seed}.pt"))
        model.eval()
        logits = {s: predict_logits(model, d["X"]).numpy() for s, d in data.items()}
        T = fit_temperature(logits["val"], data["val"]["y"])
        for s in data:
            probs[s].append(softmax(logits[s] / T))
    return {s: np.mean(p, 0) for s, p in probs.items()}, seeds


def cq(scores, alpha):
    """Conformal quantile: the ceil((n + 1)(1 - alpha))-th smallest score (inf if n is too small)."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(scores)[k - 1]


# ---- score functions: all return an (N, K) matrix S where class k is in the set iff S[:, k] <= q -------------
def lac_scores(p, u=None, lam=0.0):
    return 1 - p


def aps_scores(p, u, lam=0.0):
    """Randomized APS / RAPS score of every class: mass ranked above it + u * own mass + rank penalty."""
    order = np.argsort(-p, 1)
    ps = np.take_along_axis(p, order, 1)
    above = np.cumsum(ps, 1) - ps
    rank = np.arange(1, p.shape[1] + 1)[None]
    s_sorted = above + u[:, None] * ps + lam * np.maximum(0, rank - K_REG)
    S = np.empty_like(p)
    np.put_along_axis(S, order, s_sorted, 1)
    return S


def snr_bin(snr):
    return np.clip(np.digitize(snr, SNR_BINS) - 1, 0, len(SNR_BINS) - 2)


def calibrate(method, S_cal, y_cal, g_cal, alpha):
    """Thresholds: one global q, or one per group (true class for LAC-class, SNR bin for LAC-SNR)."""
    s = S_cal[np.arange(len(y_cal)), y_cal]
    if method == "LAC-class":
        return {c: cq(s[y_cal == c], alpha) for c in range(len(CLASSES))}
    if method == "LAC-SNR":
        return {b: cq(s[g_cal == b], alpha) for b in range(len(SNR_BINS) - 1)}
    return cq(s, alpha)


def predict_sets(method, S, g, q):
    if method == "LAC-class":
        return S <= np.array([q[c] for c in range(len(CLASSES))])[None]
    if method == "LAC-SNR":
        return S <= np.array([q[b] for b in g])[:, None]
    return S <= q


METHODS = {"LAC": lac_scores, "APS": aps_scores, "RAPS": aps_scores, "LAC-class": lac_scores, "LAC-SNR": lac_scores}


def scores_for(method, p, rng, lam):
    u = rng.random(len(p))
    return METHODS[method](p, u, lam if method == "RAPS" else 0.0)


def choose_lambda(p_val, y_val, rng, alpha=0.1, reps=50):
    """RAPS lambda with the smallest mean set size on val (half calibrates, half evaluates)."""
    sizes = {}
    for lam in LAMBDAS:
        z = []
        for _ in range(reps):
            idx = rng.permutation(len(y_val))
            a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
            S = aps_scores(p_val, rng.random(len(p_val)), lam)
            q = cq(S[a, y_val[a]], alpha)
            z.append((S[b] <= q).sum(1).mean())
        sizes[lam] = float(np.mean(z))
    return min(sizes, key=sizes.get), sizes


def with_fallback(sets, p):
    out = sets.copy()
    empty = ~out.any(1)
    out[empty, p[empty].argmax(1)] = True
    return out


def set_stats(sets, y, snr):
    covered = sets[np.arange(len(y)), y]
    size = sets.sum(1)
    qh = (sets[:, QUAD] & sets[:, HEXA] & (size == 2))
    return dict(
        cov=covered.mean(), size=size.mean(), single=(size == 1).mean(), empty=(size == 0).mean(),
        single_acc=covered[size == 1].mean() if (size == 1).any() else np.nan,
        cls_cov=[covered[y == c].mean() for c in range(len(CLASSES))],
        cls_size=[size[y == c].mean() for c in range(len(CLASSES))],
        cls_qh=[qh[y == c].mean() for c in range(len(CLASSES))],
        snr_cov=[covered[snr_bin(snr) == b].mean() for b in range(len(SNR_BINS) - 1)],
        snr_size=[size[snr_bin(snr) == b].mean() for b in range(len(SNR_BINS) - 1)],
    )


def main():
    data = {s: load(s) for s in ["val", "test", "snr_sweep", "ood"]}
    P, seeds = ensemble_probs(data)
    rng = np.random.default_rng(0)
    lam, lam_sizes = choose_lambda(P["val"], data["val"]["y"], rng)
    print("RAPS lambda", lam, lam_sizes, flush=True)

    p, y, snr = P["test"], data["test"]["y"], data["test"]["snr"]
    g = snr_bin(snr)
    acc = (p.argmax(1) == y).mean()
    res = {a: {m: [] for m in METHODS} for a in ALPHAS}
    res_fb = {m: [] for m in METHODS}
    for r in range(R):
        idx = rng.permutation(len(y))
        cal, ev = idx[: len(y) // 2], idx[len(y) // 2:]
        for m in METHODS:
            S = scores_for(m, p, rng, lam)
            for a in ALPHAS:
                q = calibrate(m, S[cal], y[cal], g[cal], a)
                sets = predict_sets(m, S[ev], g[ev], q)
                res[a][m].append(set_stats(sets, y[ev], snr[ev]))
                if a == 0.1:
                    res_fb[m].append(set_stats(with_fallback(sets, p[ev]), y[ev], snr[ev]))

    def agg(lst, key):
        v = np.array([d[key] for d in lst], dtype=float)
        return np.nanmean(v, 0), np.nanstd(v, 0)

    # Shift checks: calibrate on the whole test split, score the SNR sweep and the octocopter set
    shift = {}
    for m in METHODS:
        S_te = scores_for(m, p, rng, lam)
        q = calibrate(m, S_te, y, g, 0.1)
        sw, ys, ss = P["snr_sweep"], data["snr_sweep"]["y"], data["snr_sweep"]["snr"]
        S_sw = scores_for(m, sw, rng, lam)
        sets_sw = predict_sets(m, S_sw, snr_bin(ss), q)
        cov_sw = [sets_sw[ss == lv][np.arange((ss == lv).sum()), ys[ss == lv]].mean() for lv in SNR_SWEEP_DB]
        size_sw = [sets_sw[ss == lv].sum(1).mean() for lv in SNR_SWEEP_DB]
        po, so = P["ood"], data["ood"]["snr"]
        sets_o = predict_sets(m, scores_for(m, po, rng, lam), snr_bin(so), q)
        size_o = sets_o.sum(1)
        shift[m] = dict(cov_sw=cov_sw, size_sw=size_sw,
                        o_size=size_o.mean(), o_multi=(sets_o[:, QUAD] | sets_o[:, HEXA]).mean(),
                        o_qh=(sets_o[:, QUAD] & sets_o[:, HEXA]).mean(), o_single=(size_o == 1).mean(),
                        o_ge2=(size_o >= 2).mean(), o_empty=(size_o == 0).mean())
    # test-set reference for "set size >= 2" as an uncertainty flag
    for m in METHODS:
        S_te = scores_for(m, p, rng, lam)
        q = calibrate(m, S_te, y, g, 0.1)
        shift[m]["te_ge2"] = (predict_sets(m, S_te, g, q).sum(1) >= 2).mean()

    # ---- report ----------------------------------------------------------------------------------------------
    L = [f"# v2 step 2: conformal prediction sets (v1 CNN ensemble, seeds {seeds}, no retraining)\n",
         f"Test n = {len(y)}, split 50/50 into calibration / evaluation {R} times (mean +- std over splits). "
         f"Ensemble top-1 accuracy on test = {acc:.3f}. RAPS: k_reg = {K_REG}, lambda = {lam:g} chosen on val "
         f"(mean val set size per lambda: " + ", ".join(f"{k:g}: {v:.3f}" for k, v in lam_sizes.items()) + ").\n",
         "Coverage = share of samples whose true class is in the set. Singleton accuracy = coverage among size-1 sets.\n"]
    for a in ALPHAS:
        L += [f"## Marginal results, alpha = {a} (target coverage {1 - a:.2f})\n",
              "| Method | Coverage | Mean size | Singleton | Empty | Singleton accuracy | Worst-class coverage |",
              "|---|---|---|---|---|---|---|"]
        for m in METHODS:
            c, cs = agg(res[a][m], "cov")
            s, ss_ = agg(res[a][m], "size")
            si, _ = agg(res[a][m], "single")
            e, _ = agg(res[a][m], "empty")
            sa, _ = agg(res[a][m], "single_acc")
            cc, _ = agg(res[a][m], "cls_cov")
            L.append(f"| {m} | {c:.3f} +- {cs:.3f} | {s:.2f} +- {ss_:.2f} | {si:.3f} | {e:.3f} | {sa:.3f} | "
                     f"{cc.min():.3f} ({SHORT[int(cc.argmin())]}) |")
        L.append("")
    a = 0.1
    L += ["## Per class, alpha = 0.1\n",
          "Coverage / mean set size / share of sets that are exactly {quad, hexa}.\n",
          "| Method | " + " | ".join(SHORT) + " |", "|---|" + "---|" * len(SHORT)]
    for m in METHODS:
        cc, _ = agg(res[a][m], "cls_cov")
        cz, _ = agg(res[a][m], "cls_size")
        qh, _ = agg(res[a][m], "cls_qh")
        L.append(f"| {m} | " + " | ".join(f"{cc[k]:.2f} / {cz[k]:.2f} / {qh[k]:.2f}" for k in range(len(SHORT))) + " |")
    bins = [f"[{SNR_BINS[i]:g}, {round(SNR_BINS[i + 1]):g}) dB" for i in range(len(SNR_BINS) - 1)]
    L += ["", "## Per SNR bin (training range), alpha = 0.1\n", "Coverage / mean set size.\n",
          "| Method | " + " | ".join(bins) + " |", "|---|" + "---|" * len(bins)]
    for m in METHODS:
        sc, _ = agg(res[a][m], "snr_cov")
        sz, _ = agg(res[a][m], "snr_size")
        L.append(f"| {m} | " + " | ".join(f"{sc[k]:.3f} / {sz[k]:.2f}" for k in range(len(bins))) + " |")
    L += ["", "## Shift: SNR sweep, thresholds calibrated on the whole test split (alpha = 0.1)\n",
          "Training SNR range is -5 to 20 dB; -15 and -10 dB lie outside it. Coverage / mean set size.\n",
          "| Method | " + " | ".join(f"{lv} dB" for lv in SNR_SWEEP_DB) + " |", "|---|" + "---|" * len(SNR_SWEEP_DB)]
    for m in METHODS:
        L.append(f"| {m} | " + " | ".join(f"{c:.2f} / {s:.2f}" for c, s in zip(shift[m]["cov_sw"], shift[m]["size_sw"])) + " |")
    L += ["", "LAC-SNR assigns sweep levels outside the training range to the nearest bin (clipped).\n",
          "## Unknown target: octocopter (never seen), alpha = 0.1\n",
          "No class is correct, so coverage is undefined. 'Contains quad or hexa' = the set at least says "
          "'a multicopter'. 'Size >= 2' compares with the same rate on the known-class test set.\n",
          "| Method | Mean size | Contains quad or hexa | Contains both | Singleton | Empty | Size >= 2 (octo) | Size >= 2 (test) |",
          "|---|---|---|---|---|---|---|---|"]
    for m in METHODS:
        o = shift[m]
        L.append(f"| {m} | {o['o_size']:.2f} | {o['o_multi']:.3f} | {o['o_qh']:.3f} | {o['o_single']:.3f} | "
                 f"{o['o_empty']:.3f} | {o['o_ge2']:.3f} | {o['te_ge2']:.3f} |")
    L += ["", "## Post hoc: empty set replaced by the top-1 class (+fallback), alpha = 0.1\n",
          "Added after the first run showed empty sets on easy classes. Coverage / mean set size per class.\n",
          "| Method | Coverage | Mean size | Empty before | " + " | ".join(SHORT) + " |", "|---|---|---|---|" + "---|" * len(SHORT)]
    for m in METHODS:
        c, _ = agg(res_fb[m], "cov")
        s, _ = agg(res_fb[m], "size")
        e, _ = agg(res[0.1][m], "empty")
        cc, _ = agg(res_fb[m], "cls_cov")
        cz, _ = agg(res_fb[m], "cls_size")
        L.append(f"| {m}+fallback | {c:.3f} | {s:.2f} | {e:.3f} | " + " | ".join(f"{cc[k]:.2f} / {cz[k]:.2f}" for k in range(len(SHORT))) + " |")
    (RESULTS / "v2_conformal.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (RESULTS / "v2_conformal.json").write_text(json.dumps(dict(lam=lam, lam_sizes=lam_sizes, shift=shift), default=float, indent=1))
    print("\n".join(L))

    # ---- figure ----------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    x = np.arange(len(SHORT))
    for i, m in enumerate(["LAC", "APS", "LAC-class"]):
        cc, _ = agg(res[0.1][m], "cls_cov")
        axes[0].bar(x + (i - 1) * 0.27, cc, 0.27, label=m)
    axes[0].axhline(0.9, color="k", ls="--", lw=1)
    axes[0].set(xticks=x, xticklabels=SHORT, ylim=(0.6, 1.0), ylabel="Coverage", title="Per-class coverage (alpha = 0.1)")
    axes[0].legend(fontsize=8)
    for i, m in enumerate(["LAC", "APS", "LAC-class"]):
        cz, _ = agg(res[0.1][m], "cls_size")
        axes[1].bar(x + (i - 1) * 0.27, cz, 0.27, label=m)
    axes[1].set(xticks=x, xticklabels=SHORT, ylabel="Mean set size", title="Per-class set size: the model says 'not sure' where it should")
    for m in ["LAC", "APS", "LAC-SNR"]:
        axes[2].plot(SNR_SWEEP_DB, shift[m]["cov_sw"], "o-", label=m)
    axes[2].axhline(0.9, color="k", ls="--", lw=1)
    axes[2].axvspan(-16, -5, color="red", alpha=0.08, label="outside training SNR")
    axes[2].set(xlabel="SNR [dB]", ylabel="Coverage", title="Guarantee breaks under SNR shift", xlim=(-16, 21))
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "v2_conformal.png", dpi=110)


if __name__ == "__main__":
    main()
