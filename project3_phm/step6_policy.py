"""Step 6: does uncertainty change maintenance cost? Engine-level 5-fold cross-fitting on FD001 run-to-failure
engines (EVALUATION.md sec. 7). In every outer fold the remaining 80 engines are split into model training (50),
conformal calibration (15) and threshold selection (15); the 20 held-out engines are only simulated.

Policies (decision made at every cycle, the first trigger wins)
- Fixed interval : maintain at age T cycles (no model; T chosen on the threshold-selection engines' lives)
- A: point       : maintain when the predicted RUL <= tau
- B: lower bound : maintain when the lower end of the 90 % Mondrian conformal interval <= tau
- Perfect        : maintain one cycle before failure (lower bound on cost)

Cost per engine (unit = one planned maintenance)
- planned maintenance with r cycles of life left : 1 + r / mean_life   (wasted life converted to cost)
- failure before maintenance                     : C_f
Lead time: the work happens `lead` cycles after the trigger (waiting for parts or a maintenance slot).

Usage: python step6_policy.py     # writes results/step6_policy.md and results/step6_policy.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from baselines import window_features
from data import load_subset, operating_condition
from prep import CAP, Normalizer, Prepared, informative_sensors, training_windows, truncation_points
from uq import build_intervals

RESULTS = Path(__file__).parent / "results"
COST_RATIOS = [5, 10, 50, 100]
LEAD_TIMES = [0, 10]
TAUS = np.arange(0, CAP + 1)
AGES = np.arange(50, 401)
N_FOLDS = 5
SEED = 0
POLICIES = ["Fixed interval", "A: point", "B: lower bound", "Perfect"]


def prepared_for(raw, units_by_split):
    fit_rows = raw[raw["unit"].isin(units_by_split["train"])]
    sensors = informative_sensors(fit_rows)
    norm = Normalizer(sensors).fit(fit_rows)
    return Prepared("FD001", sensors, [], units_by_split, norm.transform(raw), None, None, norm)


def trajectories(p, split, model, cal_mu, y_cal, mean_life):
    """Per engine: life, point prediction and Mondrian lower bound at every cycle."""
    w = training_windows(p, split)
    mu = np.clip(model.predict(window_features(w["X"])), 0, CAP)
    lo, _ = build_intervals({"mu": cal_mu}, {"mu": mu}, y_cal)["Conformal (Mondrian)"]
    out = {}
    for u in np.unique(w["unit"]):
        m = w["unit"] == u
        out[u] = dict(life=int(w["cycle"][m].max()), mu=mu[m], lo=lo[m], mean_life=mean_life)
    return out


def first_trigger(signal, tau):
    """Cycle (1-based) of the first cycle where signal <= tau, or None if it never triggers."""
    hit = np.flatnonzero(signal <= tau)
    return int(hit[0]) + 1 if len(hit) else None


def engine_cost(life, t_maint, c_f, mean_life):
    """t_maint = cycle at which maintenance happens (None = never). Failure if it does not happen before life."""
    if t_maint is None or t_maint >= life:
        return c_f, True, 0
    wasted = life - t_maint
    return 1 + wasted / mean_life, False, wasted


def evaluate(trajs, policy, thr, c_f, lead=0):
    """The fixed-interval policy is scheduled in advance, so its age already includes any lead time."""
    costs, fails, wasted = [], [], []
    for tr in trajs.values():
        if policy == "Fixed interval":
            t = int(thr)
        elif policy == "Perfect":
            t = tr["life"] - 1
        else:
            t = first_trigger(tr["mu"] if policy == "A: point" else tr["lo"], thr)
            t = None if t is None else t + lead
        c, f, w = engine_cost(tr["life"], t, c_f, tr["mean_life"])
        costs.append(c)
        fails.append(f)
        wasted.append(w)
    return np.array(costs), np.array(fails), np.array(wasted)


def main():
    raw, _, _ = load_subset("FD001")
    raw["cond"] = operating_condition(raw)
    units = np.random.default_rng(SEED).permutation(np.sort(raw["unit"].unique()))
    folds = np.array_split(units, N_FOLDS)
    keys = [(pol, cf, ld) for pol in POLICIES for cf in COST_RATIOS for ld in LEAD_TIMES]
    per_engine = {k: {} for k in keys}
    chosen = {k: [] for k in keys}
    pooled_test = {}

    for k, test_units in enumerate(folds):
        rest = np.random.default_rng(SEED + k).permutation(np.setdiff1d(units, test_units))
        splits = {"train": np.sort(rest[:50]), "calib": np.sort(rest[50:65]), "select": np.sort(rest[65:]),
                  "test": np.sort(test_units)}
        p = prepared_for(raw, splits)
        tr = training_windows(p, "train")
        model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                              l2_regularization=1.0, random_state=0)
        model.fit(window_features(tr["X"]), tr["y"])
        cal = truncation_points(p, "calib", K=10)
        cal_mu = np.clip(model.predict(window_features(cal["X"])), 0, CAP)
        y_cal = np.minimum(cal["rul"], CAP)
        mean_life = float(raw[raw["unit"].isin(splits["train"])].groupby("unit")["cycle"].max().mean())
        sel = trajectories(p, "select", model, cal_mu, y_cal, mean_life)
        test = trajectories(p, "test", model, cal_mu, y_cal, mean_life)
        pooled_test.update(test)

        for cf in COST_RATIOS:
            for ld in LEAD_TIMES:
                for pol in POLICIES:
                    if pol == "Perfect":
                        thr = None
                    else:
                        grid = AGES if pol == "Fixed interval" else TAUS
                        thr = grid[int(np.argmin([evaluate(sel, pol, g, cf, ld)[0].mean() for g in grid]))]
                    chosen[(pol, cf, ld)].append(thr)
                    c, f, w = evaluate(test, pol, thr, cf, ld)
                    for u, ci, fi, wi in zip(test, c, f, w):
                        per_engine[(pol, cf, ld)][u] = (ci, fi, wi)
        print(f"fold {k + 1}/{N_FOLDS} done", flush=True)

    rng = np.random.default_rng(1)
    lines = ["# Step 6: maintenance policies (FD001, 5-fold engine-level cross-fitting, 100 engines)\n",
             "Cost unit = one planned maintenance. Wasted life is charged as (cycles left) / (mean life). "
             "A failure before maintenance costs C_f. Thresholds are chosen inside each fold on 15 engines the "
             "evaluated engines never touch. Model: GBM + Mondrian conformal (selected for FD001 in step 4). "
             "Lead time = cycles between the decision and the maintenance.\n"]
    summary = {}
    for ld in LEAD_TIMES:
        lines += [f"\n## Lead time {ld} cycles\n",
                  "| C_f | Policy | Mean cost / engine | vs Perfect | Failures | Median wasted cycles | Thresholds per fold |",
                  "|---|---|---|---|---|---|---|"]
        for cf in COST_RATIOS:
            perfect = np.mean([v[0] for v in per_engine[("Perfect", cf, ld)].values()])
            for pol in POLICIES:
                vals = np.array(list(per_engine[(pol, cf, ld)].values()))
                summary[(pol, cf, ld)] = (vals[:, 0].mean(), vals[:, 1].mean())
                thr = "-" if pol == "Perfect" else ", ".join(str(int(t)) for t in chosen[(pol, cf, ld)])
                lines.append(f"| {cf} | {pol} | {vals[:, 0].mean():.3f} | {vals[:, 0].mean() / perfect:.2f}x | "
                             f"{int(vals[:, 1].sum())}/100 | {np.median(vals[:, 2]):.0f} | {thr} |")
        lines += [f"\n### B vs A, lead time {ld}: paired engine bootstrap (negative = B cheaper)\n",
                  "| C_f | B − A | 95 % CI | B − fixed interval |", "|---|---|---|---|"]
        for cf in COST_RATIOS:
            us = sorted(per_engine[("A: point", cf, ld)])
            a = np.array([per_engine[("A: point", cf, ld)][u][0] for u in us])
            b = np.array([per_engine[("B: lower bound", cf, ld)][u][0] for u in us])
            fx = np.array([per_engine[("Fixed interval", cf, ld)][u][0] for u in us])
            boots = [np.mean((b - a)[i]) for i in (rng.integers(0, len(us), len(us)) for _ in range(5000))]
            lines.append(f"| {cf} | {np.mean(b - a):+.3f} | {np.percentile(boots, 2.5):+.3f} – "
                         f"{np.percentile(boots, 97.5):+.3f} | {np.mean(b - fx):+.3f} |")

    # Diagnosis only: thresholds picked on the evaluated engines themselves (not deployable). Separates
    # "how good can each signal be" from "how stable is choosing its threshold on 15 engines".
    lines += ["\n## Diagnosis: hindsight-optimal thresholds on the 100 evaluated engines (not deployable)\n",
              "| Lead | C_f | Policy | Best τ (hindsight) | Cost at best τ | Cost with fold-chosen τ | Failures at best τ |",
              "|---|---|---|---|---|---|---|"]
    curves = {}
    for ld in LEAD_TIMES:
        for cf in [10, 50]:
            for pol in ["A: point", "B: lower bound"]:
                cs = [evaluate(pooled_test, pol, t, cf, ld)[0].mean() for t in TAUS]
                curves[(pol, cf, ld)] = cs
                i = int(np.argmin(cs))
                fails = int(evaluate(pooled_test, pol, TAUS[i], cf, ld)[1].sum())
                lines.append(f"| {ld} | {cf} | {pol} | {TAUS[i]} | {cs[i]:.3f} | {summary[(pol, cf, ld)][0]:.3f} | "
                             f"{fails}/100 |")
    (RESULTS / "step6_policy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.3))
    for pol, style in zip(POLICIES, ["s--", "o-", "D-", "k:"]):
        for ld, alpha in zip(LEAD_TIMES, [1.0, 0.45]):
            axes[0].plot(COST_RATIOS, [summary[(pol, cf, ld)][0] for cf in COST_RATIOS], style, alpha=alpha,
                         label=f"{pol} (lead {ld})")
            axes[1].plot(COST_RATIOS, [100 * summary[(pol, cf, ld)][1] for cf in COST_RATIOS], style, alpha=alpha,
                         label=f"{pol} (lead {ld})")
    axes[0].set(xscale="log", yscale="log", xlabel="Failure cost C_f (× planned maintenance)",
                ylabel="Mean cost per engine", title="Total cost (lower is better)")
    axes[1].set(xscale="log", xlabel="Failure cost C_f (× planned maintenance)",
                ylabel="Engines failing before maintenance [%]", title="Failure rate")
    for ax in axes[:2]:
        ax.set_xticks(COST_RATIOS)
        ax.set_xticklabels(COST_RATIOS)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6, ncol=2)
    for (pol, cf, ld), cs in curves.items():
        if cf == 50:
            axes[2].plot(TAUS, cs, "-" if pol == "A: point" else "--", alpha=1.0 if ld == 0 else 0.5,
                         label=f"{pol}, lead {ld}")
    axes[2].set(yscale="log", xlabel="Threshold τ [cycles]", ylabel="Mean cost per engine (C_f = 50)",
                title="Cost vs threshold on evaluated engines (diagnosis)")
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "step6_policy.png", dpi=110)


if __name__ == "__main__":
    main()
