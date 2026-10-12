"""Step 0: data inspection before any modelling.

Usage: python inspect_data.py    # writes results/data_report.md and results/inspect_*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import SENSOR_INFO, SENSORS, SETTINGS, SUBSETS, load_subset, operating_condition

RESULTS = Path(__file__).parent / "results"
MAX_LEVELS = 2         # a sensor taking at most this many values within a condition only shows quantization


def life_stats(train):
    life = train.groupby("unit")["cycle"].max()
    return life.min(), int(life.median()), life.max()


def near_constant(df):
    """Sensors that take at most MAX_LEVELS distinct values within every operating condition.

    A relative-variation (CV) rule is unsuitable here: Nf/NRf sit near 2388 rpm, so a real but small
    degradation drift gives a tiny CV and would be misclassified as constant.
    """
    levels = df.groupby("cond")[SENSORS].nunique().max()
    return [s for s in SENSORS if levels[s] <= MAX_LEVELS]


def trend_strength(train, by_condition):
    """|Spearman correlation| between each sensor (condition-normalized if needed) and capped RUL."""
    df = train.copy()
    if by_condition:
        g = df.groupby("cond")
        for s in SENSORS:
            df[s] = (df[s] - g[s].transform("mean")) / g[s].transform("std").replace(0, np.nan)
    target = np.minimum(df["rul"], 125)
    return {s: abs(df[s].corr(target, method="spearman")) for s in SENSORS}


def main():
    RESULTS.mkdir(exist_ok=True)
    rows, const_rows, trend_rows, test_rows, cond_rows = [], [], [], [], []
    data = {}
    for name in SUBSETS:
        train, test, rul = load_subset(name)
        for df in (train, test):
            df["cond"] = operating_condition(df)
        data[name] = (train, test, rul)
        multi = train["cond"].nunique() > 1

        # integrity checks
        assert not train.isna().any().any() and not test.isna().any().any(), name
        for df in (train, test):
            ok = df.groupby("unit")["cycle"].apply(lambda c: (np.diff(c.to_numpy()) == 1).all() and c.iloc[0] == 1)
            assert ok.all(), f"{name}: non-consecutive cycles"
        assert len(rul) == test["unit"].nunique(), name

        lo, med, hi = life_stats(train)
        tlen = test.groupby("unit")["cycle"].max()
        rows.append(f"| {name} | {train['unit'].nunique()} | {test['unit'].nunique()} | {lo} / {med} / {hi} | "
                    f"{tlen.min()} / {int(tlen.median())} / {tlen.max()} | {rul.min()} / {int(np.median(rul))} / {rul.max()} | "
                    f"{train['cond'].nunique()} |")
        const = near_constant(train)
        const_rows.append(f"| {name} | {', '.join(f'{s} ({SENSOR_INFO[s][0]})' for s in const) or '-'} |")
        ts = trend_strength(train, by_condition=multi)
        top = sorted((s for s in SENSORS if s not in const), key=lambda s: -ts[s])
        trend_rows.append(f"| {name} | " + ", ".join(f"{s} {ts[s]:.2f}" for s in top[:8]) + " |")
        if multi:
            counts = train["cond"].value_counts()
            cond_rows.append(f"| {name} | {len(counts)} | {counts.min()}–{counts.max()} | "
                             f"{test['cond'].nunique()} | {sorted(set(test['cond']) - set(train['cond'])) or '없음'} |")
        test_rows.append((name, int((tlen < 30).sum()), int(tlen.min())))

    report = ["# Step 0 data report\n",
              "## Subsets\n",
              "| Subset | Train engines | Test engines | Train life min/median/max | Test length min/median/max | Test true RUL min/median/max | Operating conditions |",
              "|---|---|---|---|---|---|---|", *rows,
              "\n## Near-constant sensors (within each operating condition for FD002/FD004)\n",
              "| Subset | Sensors |", "|---|---|", *const_rows,
              "\n## Degradation trend strength: |Spearman| with RUL capped at 125 (top 8, condition-normalized for FD002/FD004)\n",
              "| Subset | Sensors |", "|---|---|", *trend_rows,
              "\n## Operating conditions (FD002/FD004)\n",
              "| Subset | Distinct conditions (train) | Rows per condition | Distinct conditions (test) | Conditions only in test |",
              "|---|---|---|---|---|", *cond_rows,
              "\n## Short test trajectories (limit the input window length)\n",
              "| Subset | Test engines shorter than 30 cycles | Shortest test trajectory |", "|---|---|---|",
              *[f"| {n} | {k} | {m} |" for n, k, m in test_rows]]
    (RESULTS / "data_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))

    # Figure 1: FD001 sensors aligned at failure
    train = data["FD001"][0]
    show = ["s2", "s3", "s4", "s7", "s11", "s12", "s15", "s21"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
    for ax, s in zip(axes.flat, show):
        for u in range(1, 11):
            d = train[train["unit"] == u]
            ax.plot(-d["rul"], d[s], lw=0.8, alpha=0.8)
        ax.axvline(-125, ls=":", color="0.4", lw=0.8)
        sym, desc, unit = SENSOR_INFO[s]
        ax.set_title(f"{s} {sym} [{unit}]", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("Cycles before failure")
    fig.suptitle("FD001: 10 engines aligned at failure (dotted = RUL cap 125)")
    fig.tight_layout()
    fig.savefig(RESULTS / "inspect_fd001_sensors.png", dpi=110)

    # Figure 2: FD002 operating conditions and the effect of per-condition normalization
    train = data["FD002"][0].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    codes = {c: i for i, c in enumerate(sorted(train["cond"].unique()))}
    axes[0].scatter(train["setting1"], train["setting2"], c=train["cond"].map(codes), s=3, cmap="tab10")
    axes[0].set(xlabel="setting1 (altitude, kft)", ylabel="setting2 (Mach)", title="FD002 operating settings")
    d = train[train["unit"] <= 5]
    axes[1].scatter(-d["rul"], d["s3"], s=3, c=d["cond"].map(codes), cmap="tab10")
    axes[1].set(xlabel="Cycles before failure", ylabel="s3 T30 raw", title="Raw sensor: dominated by condition")
    g = train.groupby("cond")["s3"]
    train["s3n"] = (train["s3"] - g.transform("mean")) / g.transform("std")
    d = train[train["unit"] <= 5]
    for u in range(1, 6):
        e = d[d["unit"] == u]
        axes[2].plot(-e["rul"], e["s3n"].rolling(10, min_periods=1).mean(), lw=1)
    axes[2].set(xlabel="Cycles before failure", ylabel="s3 normalized per condition (10-cycle mean)",
                title="Per-condition normalization reveals degradation")
    fig.tight_layout()
    fig.savefig(RESULTS / "inspect_conditions.png", dpi=110)

    # Figure 3: life-length and test-RUL distributions
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for name in SUBSETS:
        tr, te, rul = data[name]
        axes[0].hist(tr.groupby("unit")["cycle"].max(), bins=25, histtype="step", lw=1.5, label=name)
        axes[1].hist(rul, bins=25, histtype="step", lw=1.5, label=name)
    axes[0].set(xlabel="Train engine life [cycles]", ylabel="Engines")
    axes[1].set(xlabel="Test true RUL at last cycle [cycles]")
    axes[1].axvline(125, ls=":", color="0.4")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "inspect_distributions.png", dpi=110)


if __name__ == "__main__":
    main()
