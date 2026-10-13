"""v2 step 3: hierarchical output - answer "multicopter" when quad vs hexa is not separable.

Hierarchy (two levels):
    aerial target (root)
    |- multicopter      = {quadcopter, hexacopter}
    |- fixed-wing UAV
    |- helicopter
    |- bird
An answer is a node; it is *correct* if the true class lies under it. A leaf is specific, "multicopter" is less
specific, the root ("unidentified") is correct but carries no information.
Octocopter was never seen in training, but it *is* a multicopter: under the hierarchy it finally has a correct
answer ("multicopter"), which a flat 5-class classifier can never give.

Methods (v1 3-seed ensemble with temperature scaling; no retraining)
- Flat      : top-1 leaf (v1 behaviour).
- Coarse    : top-1 over the 4 groups (quad + hexa probabilities merged). Never answers quad or hexa.
- Hier-tau  : most specific node whose probability >= tau (leaf, then multicopter, then root).
              tau is chosen on val as the smallest value whose hierarchical accuracy reaches the target (0.95, 0.98).
- Conf-hier : LAC-class conformal set (alpha = 0.1, empty -> top-1 fallback; the best method of step 2),
              calibrated on val and mapped to the smallest node that contains the set.

Hypotheses (written before running)
- K1: coarse 4-group accuracy is far above flat 5-class accuracy (0.848), because most errors are quad <-> hexa.
- K2: Hier-tau reaches its accuracy target on test while keeping fixed-wing, helicopter and bird mostly at leaf level;
      the coarse "multicopter" answers fall on quad/hexa samples.
- K3: octocopter: correct "multicopter" answers rise from 0 % (flat) to a majority, but a sizeable share is still
      called a specific leaf (quad or hexa) with the same confidence as real quad/hexa - so this is a correct
      fallback answer, not unknown-target detection.
- K4: at low SNR, Hier-tau gives coarser answers and keeps hierarchical accuracy above flat accuracy.

Added after seeing the first run (post hoc, reported separately)
- Thresholds fit on val came out too tight (quad under-covered, tau targets missed). val is not clean: train.py
  keeps the checkpoint with the best val accuracy and evaluate.py fits the temperature on val. Check: re-fit tau and
  the conformal thresholds on a random half of test, evaluate on the other half, R = 200 splits.

Usage: python hierarchy.py      # needs data/ and checkpoints/; writes results/v2_hierarchy.md and .png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import CLASSES, SNR_SWEEP_DB
from conformal_sets import cq, ensemble_probs, with_fallback
from dataset import load

RESULTS = Path(__file__).parent / "results"
QUAD, HEXA = 0, 1
K = len(CLASSES)
MULTI, ROOT = K, K + 1                      # node ids: 0..4 leaves, 5 multicopter, 6 root
NODE_NAMES = ["quad", "hexa", "fixed", "heli", "bird", "multicopter", "unidentified"]
TARGETS = [0.95, 0.98]
TAUS = np.round(np.arange(0.30, 1.0001, 0.005), 3)
ALPHA = 0.1


def covers(node, y):
    """True if the true class y (array) lies under the answered node (array)."""
    return (node == y) | ((node == MULTI) & ((y == QUAD) | (y == HEXA))) | (node == ROOT)


def flat(p):
    return p.argmax(1)


def coarse(p):
    g = np.column_stack([p[:, QUAD] + p[:, HEXA], p[:, 2:]])      # multicopter, fixed, heli, bird
    a = g.argmax(1)
    return np.where(a == 0, MULTI, a + 1)


def hier_tau(p, tau):
    top = p.argmax(1)
    pm = p[:, QUAD] + p[:, HEXA]
    return np.where(p.max(1) >= tau, top, np.where(pm >= tau, MULTI, ROOT))


def choose_tau(p, y, target):
    for tau in TAUS:
        if covers(hier_tau(p, tau), y).mean() >= target:
            return float(tau)
    return float(TAUS[-1])


def conf_hier_fit(p_cal, y_cal):
    s = 1 - p_cal[np.arange(len(y_cal)), y_cal]
    return np.array([cq(s[y_cal == c], ALPHA) for c in range(K)])


def conf_hier(p, q):
    sets = with_fallback((1 - p) <= q[None], p)
    size = sets.sum(1)
    only_multi = ~sets[:, 2:].any(1)
    node = np.full(len(p), ROOT)
    node[size == 1] = sets[size == 1].argmax(1)
    node[(size == 2) & only_multi] = MULTI
    return node


def stats(node, y):
    ok = covers(node, y)
    leaf = node < K
    return dict(acc=ok.mean(), leaf=leaf.mean(), multi=(node == MULTI).mean(), root=(node == ROOT).mean(),
                leaf_acc=ok[leaf].mean() if leaf.any() else np.nan,
                cls_leaf=[leaf[y == c].mean() for c in range(K)],
                cls_multi=[(node[y == c] == MULTI).mean() for c in range(K)],
                cls_acc=[ok[y == c].mean() for c in range(K)])


def main():
    data = {s: load(s) for s in ["val", "test", "snr_sweep", "ood"]}
    P, seeds = ensemble_probs(data)
    yv, yt = data["val"]["y"], data["test"]["y"]
    taus = {t: choose_tau(P["val"], yv, t) for t in TARGETS}
    q = conf_hier_fit(P["val"], yv)
    methods = {"Flat (v1)": flat, "Coarse (4 groups)": coarse,
               **{f"Hier-tau, target {t:.2f} (tau={taus[t]:.3f})": (lambda p, tau=taus[t]: hier_tau(p, tau)) for t in TARGETS},
               "Conf-hier (LAC-class + fallback)": lambda p: conf_hier(p, q)}
    print("taus", taus, "conformal q", q, flush=True)

    L = [f"# v2 step 3: hierarchical output (v1 CNN ensemble, seeds {seeds}, no retraining)\n",
         "Answer = a node of the hierarchy {quad, hexa} -> multicopter; fixed-wing, helicopter, bird are leaves; root = "
         "'unidentified'. Correct = true class lies under the answered node. Thresholds (tau, conformal q) are fit on val "
         f"(n = {len(yv)}); test n = {len(yt)}, octocopter n = {len(data['ood']['y'])}.\n",
         "## Known classes (test)\n",
         "| Method | Hierarchical accuracy | Leaf answers | 'Multicopter' | 'Unidentified' | Accuracy of leaf answers |",
         "|---|---|---|---|---|---|"]
    res = {}
    for name, f in methods.items():
        res[name] = s = stats(f(P["test"]), yt)
        L.append(f"| {name} | {s['acc']:.3f} | {s['leaf']:.3f} | {s['multi']:.3f} | {s['root']:.3f} | {s['leaf_acc']:.3f} |")
    L += ["", "## Per class (test): hierarchical accuracy / share answered at leaf level / share answered 'multicopter'\n",
          "| Method | " + " | ".join(NODE_NAMES[:K]) + " |", "|---|" + "---|" * K]
    for name, s in res.items():
        L.append(f"| {name} | " + " | ".join(f"{s['cls_acc'][c]:.2f} / {s['cls_leaf'][c]:.2f} / {s['cls_multi'][c]:.2f}"
                                           for c in range(K)) + " |")

    # Octocopter: correct answer is "multicopter" (or the uninformative root)
    po = P["ood"]
    L += ["", "## Unknown target: octocopter (a multicopter never seen in training)\n",
          "Correct informative answer = 'multicopter'. 'Wrong leaf' = called quad or hexa; 'wrong group' = fixed-wing, "
          "helicopter or bird. The last column compares the leaf-answer rate with real quad/hexa test samples: if they are "
          "equal, the method cannot tell octocopter from the classes it knows.\n",
          "| Method | 'Multicopter' (correct) | 'Unidentified' | Wrong leaf (quad/hexa) | Wrong group | Leaf rate: octo vs real quad/hexa |",
          "|---|---|---|---|---|---|"]
    octo = {}
    mq = (yt == QUAD) | (yt == HEXA)
    for name, f in methods.items():
        n = f(po)
        nt = f(P["test"])
        octo[name] = dict(multi=(n == MULTI).mean(), root=(n == ROOT).mean(),
                          wleaf=np.isin(n, [QUAD, HEXA]).mean(), wgroup=np.isin(n, [2, 3, 4]).mean())
        o = octo[name]
        L.append(f"| {name} | {o['multi']:.3f} | {o['root']:.3f} | {o['wleaf']:.3f} | {o['wgroup']:.3f} | "
                 f"{(n < K).mean():.3f} vs {(nt[mq] < K).mean():.3f} |")

    # SNR sweep
    sw, ys, ss = P["snr_sweep"], data["snr_sweep"]["y"], data["snr_sweep"]["snr"]
    L += ["", "## SNR sweep: hierarchical accuracy / leaf-answer rate\n",
          "Training range is -5 to 20 dB; -15 and -10 dB lie outside it.\n",
          "| Method | " + " | ".join(f"{lv} dB" for lv in SNR_SWEEP_DB) + " |", "|---|" + "---|" * len(SNR_SWEEP_DB)]
    sweep = {}
    for name, f in methods.items():
        n = f(sw)
        sweep[name] = [(covers(n[ss == lv], ys[ss == lv]).mean(), (n[ss == lv] < K).mean()) for lv in SNR_SWEEP_DB]
        L.append(f"| {name} | " + " | ".join(f"{a:.2f} / {b:.2f}" for a, b in sweep[name]) + " |")

    # Severe errors: answer in a different top-level group (e.g. a drone called a bird)
    L += ["", "## Cross-group errors (test): answer lies in a different top-level group than the truth\n",
          "| Method | Cross-group error rate | Drone (quad/hexa/fixed) answered 'bird' | Bird answered as a drone |", "|---|---|---|---|"]
    drone = np.isin(yt, [QUAD, HEXA, 2])
    for name, f in methods.items():
        n = f(P["test"])
        wrong = ~covers(n, yt) & ~((n == QUAD) & (yt == HEXA)) & ~((n == HEXA) & (yt == QUAD))
        d2b = (n[drone] == 4).mean()
        b2d = np.isin(n[yt == 4], [QUAD, HEXA, 2, MULTI]).mean()
        L.append(f"| {name} | {wrong.mean():.3f} | {d2b:.3f} | {b2d:.3f} |")

    # Post hoc: calibrate on clean data (half of test), evaluate on the other half
    rng = np.random.default_rng(0)
    pt, po = P["test"], P["ood"]
    rows = {k: [] for k in ["Hier-tau 0.95", "Hier-tau 0.98", "Conf-hier"]}
    for _ in range(200):
        idx = rng.permutation(len(yt))
        a, b = idx[: len(yt) // 2], idx[len(yt) // 2:]
        fns = {f"Hier-tau {tg:.2f}": (lambda pp, tau=choose_tau(pt[a], yt[a], tg): hier_tau(pp, tau)) for tg in TARGETS}
        qa = conf_hier_fit(pt[a], yt[a])
        fns["Conf-hier"] = lambda pp, qa=qa: conf_hier(pp, qa)
        for k, f in fns.items():
            nb, no = f(pt[b]), f(po)
            ok = covers(nb, yt[b])
            rows[k].append([ok.mean(), ok[yt[b] == QUAD].mean(), (nb < K).mean(), (no == MULTI).mean(),
                            np.isin(no, [QUAD, HEXA]).mean()])
    L += ["", "## Post hoc: thresholds fit on clean data (random half of test, 200 splits), mean +- std\n",
          "val is not clean (checkpoint selection and temperature both used it). Main tables above are unchanged.\n",
          "| Method | Hierarchical accuracy | Quad accuracy | Leaf answers | Octo 'multicopter' | Octo wrong leaf |",
          "|---|---|---|---|---|---|"]
    for k, v in rows.items():
        m, s = np.mean(v, 0), np.std(v, 0)
        L.append(f"| {k} | " + " | ".join(f"{m[i]:.3f} +- {s[i]:.3f}" for i in range(5)) + " |")

    (RESULTS / "v2_hierarchy.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    names = list(methods)
    short = ["Flat", "Coarse", f"Hier {TARGETS[0]:.2f}", f"Hier {TARGETS[1]:.2f}", "Conf-hier"]
    parts = [("multi", "'Multicopter' (correct)", "C2"), ("root", "'Unidentified'", "C7"),
             ("wleaf", "Wrong leaf (quad/hexa)", "C3"), ("wgroup", "Wrong group", "C1")]
    bottom = np.zeros(len(names))
    for key, lab, col in parts:
        v = np.array([octo[n][key] for n in names])
        axes[0].bar(short, v, bottom=bottom, label=lab, color=col)
        bottom += v
    axes[0].set(ylabel="Share of octocopters", title="Octocopter (never seen): what each method answers")
    axes[0].legend(fontsize=8, loc="lower right")
    for n, s in zip(names, short):
        axes[1].plot(SNR_SWEEP_DB, [a for a, _ in sweep[n]], "o-", label=s)
    axes[1].axvspan(-16, -5, color="red", alpha=0.08)
    axes[1].set(xlabel="SNR [dB]", ylabel="Hierarchical accuracy", title="Accuracy vs SNR (shaded: outside training range)",
                xlim=(-16, 21))
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "v2_hierarchy.png", dpi=110)


if __name__ == "__main__":
    main()
