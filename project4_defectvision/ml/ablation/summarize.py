"""Collect ablation stage outputs into CSV files and Markdown tables.

Outputs (results/ablation/):
- runs.csv      one row per planned (experiment, category, seed), status OK / FAILED / NOT_RUN
- summary.csv   seed-averaged metrics per (experiment, category) plus a cross-category MEAN row
- selection.json  configuration chosen by the pre-registered rule, using validation metrics only
- rank_agreement.csv  Spearman correlation between validation-proxy and test rankings of configs
- tables.md     Markdown tables used in docs/ABLATION_STUDY.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ml.ablation.run_one import DEFAULT_PLAN, PROJECT_DIR, load_plan, run_name, run_overrides
from ml.config import load_config

METRICS = [
    "val_image_auroc", "val_pixel_auroc", "val_aupro",
    "test_image_auroc", "test_pixel_auroc", "test_aupro",
    "test_f1_at_val_threshold", "test_fpr_at_val_threshold",
    "latency_p95_ms", "peak_rss_serve_mb", "peak_rss_fit_mb", "memory_bank_mb", "fit_wall_seconds",
]


def _describe(plan: dict, exp_id: str) -> dict:
    """Fully resolved settings (base config + baseline + experiment overrides)."""
    cfg = load_config(PROJECT_DIR / plan["base_config"], run_overrides(plan, exp_id, 0))
    m = cfg.model
    return {"backbone": m.backbone, "layers": "+".join(m.layers), "sampling": m.sampling,
            "coreset_ratio": m.coreset_ratio, "image_size": cfg.preprocessing.image_size}


def collect_runs(plan: dict, out_root: Path) -> pd.DataFrame:
    rows = []
    for e in plan["experiments"]:
        for cat in plan["categories"]:
            for seed in e.get("seeds", [0]):
                row = {"exp_id": e["id"], "factor": e["factor"], "category": cat, "seed": seed,
                       **_describe(plan, e["id"])}
                stage = out_root / "stages" / run_name(e["id"], cat, seed)
                if e.get("not_run"):
                    row.update(status="NOT_RUN", reason=" ".join(e["not_run"].split()))
                elif (stage / "eval.json").exists():
                    ev = json.loads((stage / "eval.json").read_text(encoding="utf-8"))
                    fit = json.loads((stage / "fit.json").read_text(encoding="utf-8"))
                    row.update(status="OK", reason="", plan_sha256=ev["plan_sha256"][:12],
                               n_fit=ev["n_fit"], n_val_normal=ev["n_val_normal"], n_test=ev["n_test"],
                               memory_bank_rows=fit["fit_stats"]["memory_bank_shape"][0],
                               memory_bank_mb=fit["memory_bank_mb"], peak_rss_fit_mb=fit["peak_rss_fit_mb"],
                               fit_wall_seconds=fit["fit_wall_seconds"],
                               **{k: ev[k] for k in ev if k.startswith(("val_", "test_", "latency_", "peak_rss_"))
                                  or k in ("model_load_seconds", "rss_after_load_mb")})
                elif (stage / "failed.json").exists():
                    f = json.loads((stage / "failed.json").read_text(encoding="utf-8"))
                    row.update(status="FAILED", reason=f"{f['stage']} rc={f['returncode']}")
                else:
                    row.update(status="NOT_RUN", reason="not executed yet")
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_runs(runs: pd.DataFrame, plan: dict) -> pd.DataFrame:
    ok = runs[runs.status == "OK"]
    keys = ["exp_id", "factor", "backbone", "layers", "sampling", "coreset_ratio", "image_size"]
    if ok.empty:
        return pd.DataFrame(columns=[*keys, "category", "n_seeds"])
    metrics = [m for m in METRICS if m in ok.columns]
    g = ok.groupby([*keys, "category"], sort=False)
    mean = g[metrics].mean()
    std = g[metrics].std(ddof=1).add_suffix("_std")
    per_cat = pd.concat([mean, std, g.size().rename("n_seeds")], axis=1).reset_index()

    # Cross-category mean, only from seed-averaged category values; flags incomplete coverage.
    n_cat = len(plan["categories"])
    agg = per_cat.groupby(keys, sort=False).agg(
        **{m: (m, "mean") for m in metrics}, n_categories=("category", "nunique"),
    ).reset_index()
    agg["category"] = np.where(agg.n_categories == n_cat, "MEAN", "MEAN_PARTIAL")
    return pd.concat([per_cat, agg], ignore_index=True)


def select(summary: pd.DataFrame, plan: dict) -> dict:
    rule = plan["selection"]
    mean = summary[summary.category == "MEAN"].dropna(subset=[rule["metric"]])
    if mean.empty:
        return {"status": "NOT_RUN", "reason": "no configuration has results for every category"}
    best = mean[rule["metric"]].max()
    pool = mean[mean[rule["metric"]] >= best - rule["tolerance"]]
    chosen = pool.sort_values(rule["tie_break"]).iloc[0]
    return {
        "status": "OK",
        "rule": rule,
        "candidates_within_tolerance": pool.exp_id.tolist(),
        "selected": chosen.exp_id,
        "selected_val_metric": float(chosen[rule["metric"]]),
        "selected_test": {k: float(chosen[k]) for k in ("test_image_auroc", "test_pixel_auroc", "test_aupro")},
        "note": "Selection used validation metrics only; test values are reported after the fact.",
    }


def rank_agreement(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cat, df in summary[summary.category != "MEAN_PARTIAL"].groupby("category"):
        for val, test in (("val_image_auroc", "test_image_auroc"), ("val_pixel_auroc", "test_pixel_auroc"),
                          ("val_aupro", "test_aupro")):
            d = df[[val, test]].dropna()
            rho = spearmanr(d[val], d[test]).statistic if len(d) >= 3 else np.nan
            rows.append({"category": cat, "val_metric": val, "test_metric": test, "n_configs": len(d),
                         "spearman_rho": None if np.isnan(rho) else float(rho)})
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame, cols: list[str], digits: int = 3) -> str:
    def fmt(v):
        if isinstance(v, float):
            return "NOT_RUN" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(lines)


def summarize(plan: dict, out_root: Path, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(plan, out_root)
    runs.to_csv(results_dir / "runs.csv", index=False)
    summary = summarize_runs(runs, plan)
    summary.to_csv(results_dir / "summary.csv", index=False)
    selection = select(summary, plan) if not summary.empty else {"status": "NOT_RUN"}
    (results_dir / "selection.json").write_text(json.dumps(selection, indent=2, default=str), encoding="utf-8")
    agree = rank_agreement(summary) if not summary.empty else pd.DataFrame()
    agree.to_csv(results_dir / "rank_agreement.csv", index=False)

    # Markdown: one row per experiment for the MEAN (or partial) block, NOT_RUN rows included.
    exp_rows = []
    for e in plan["experiments"]:
        s = summary[(summary.exp_id == e["id"]) & summary.category.isin(["MEAN", "MEAN_PARTIAL"])]
        base = {"exp_id": e["id"], "factor": e["factor"], **_describe(plan, e["id"])}
        if s.empty:
            base["coverage"] = "NOT_RUN"
            exp_rows.append(base)
        else:
            r = s.iloc[0].to_dict()
            r["coverage"] = f"{int(r['n_categories'])}/{len(plan['categories'])} cat"
            exp_rows.append(r)
    table = pd.DataFrame(exp_rows)
    for c in METRICS:
        if c not in table.columns:
            table[c] = np.nan
    cols_q = ["exp_id", "backbone", "layers", "sampling", "coreset_ratio", "image_size", "coverage",
              "val_image_auroc", "test_image_auroc", "test_pixel_auroc", "test_aupro"]
    cols_c = ["exp_id", "latency_p95_ms", "peak_rss_serve_mb", "peak_rss_fit_mb", "memory_bank_mb",
              "fit_wall_seconds"]
    per_cat = summary[~summary.category.isin(["MEAN", "MEAN_PARTIAL"])]
    md = [
        "<!-- generated by ml/ablation/summarize.py — do not edit by hand -->",
        f"plan sha256 `{plan['_sha256'][:12]}`, runs OK {int((runs.status == 'OK').sum())} / "
        f"FAILED {int((runs.status == 'FAILED').sum())} / NOT_RUN {int((runs.status == 'NOT_RUN').sum())}",
        "", "### Quality (mean over seeds, then over categories)", "", _md_table(table, cols_q),
        "", "### Cost (mean over categories)", "", _md_table(table, cols_c, 1),
    ]
    if not per_cat.empty:
        md += ["", "### Per category", "", _md_table(per_cat, ["exp_id", "category", "n_seeds", "val_image_auroc",
               "test_image_auroc", "test_pixel_auroc", "test_aupro", "latency_p95_ms", "peak_rss_serve_mb"])]
        seeded = per_cat[per_cat.n_seeds > 1]
        if not seeded.empty:
            md += ["", "### Seed spread (std over seeds)", "", _md_table(seeded, [
                "exp_id", "category", "n_seeds", "val_image_auroc_std", "test_image_auroc_std",
                "test_pixel_auroc_std", "test_aupro_std"], 4)]
    if not agree.empty:
        md += ["", "### Validation-proxy vs test rank agreement (Spearman)", "",
               _md_table(agree, ["category", "val_metric", "n_configs", "spearman_rho"])]
    md += ["", "### Selection (validation only)", "", "```json", json.dumps(selection, indent=2, default=str), "```"]
    (results_dir / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    p.add_argument("--out", type=Path, default=PROJECT_DIR / "artifacts/ablation")
    p.add_argument("--results", type=Path, default=PROJECT_DIR / "results/ablation")
    a = p.parse_args()
    summarize(load_plan(a.plan), a.out, a.results)
