"""Final evaluation on the official test sets (run once, after every choice was fixed on validation engines).

Usage: python final_test.py     # writes results/final_test.md
"""
import datetime
import json
from pathlib import Path

import numpy as np

from baselines import fit_predict
from eval_step3 import load_runs
from metrics import engine_bootstrap, interval_metrics, point_metrics, rmse
from prep import CAP, WINDOW, official_test_points, prepare, training_windows, truncation_points
from uq import build_intervals, deep_outputs, safety_metrics

RESULTS = Path(__file__).parent / "results"


def main():
    selected = json.loads((RESULTS / "step4_selected.json").read_text())
    lines = [f"# Final evaluation on the official test sets\n",
             f"Run on {datetime.date.today().isoformat()}, once, after all model, interval and threshold choices were fixed "
             f"on validation engines. One point per test engine (its last cycle). Point predictions clipped to [0, 125].\n"]
    for subset in ["FD001", "FD004"]:
        p = prepare(subset)
        tr = training_windows(p, "train")
        te = official_test_points(p)
        cal = truncation_points(p, "calib", K=10)
        y_true, y_cap, y_cal = te["rul"], np.minimum(te["rul"], CAP), np.minimum(cal["rul"], CAP)

        point = {"GBM": (np.clip(fit_predict("GBM (window stats)", tr, te), 0, CAP),
                         np.clip(fit_predict("GBM (window stats)", tr, cal), 0, CAP))}
        deep = {}
        for model in ["lstm", "cnn"]:
            ms = [m for m, _ in load_runs(subset, model, "condition", p.n_channels)]
            deep[model.upper()] = (deep_outputs(ms, te["X"]), deep_outputs(ms, cal["X"]))
            point[model.upper()] = (np.clip(deep[model.upper()][0]["mu"], 0, CAP), None)

        lines += [f"\n## {subset}  ({len(y_true)} test engines; true RUL {int(y_true.min())}–{int(y_true.max())}, "
                  f"{int((y_true > CAP).sum())} engines above 125)\n",
                  "### Point prediction\n",
                  "| Model | RMSE (true RUL) | 95 % CI | RMSE (capped) | MAE | NASA score (sum) | NASA score (mean) |",
                  "|---|---|---|---|---|---|---|"]
        for name, (pred, _) in point.items():
            m = point_metrics(pred, y_true)
            ci = engine_bootstrap(lambda i: rmse(pred[i], y_true[i]), te["unit"])
            lines.append(f"| {name} | {m['rmse']:.2f} | {ci[0]:.1f}–{ci[1]:.1f} | {m['rmse_capped']:.2f} | {m['mae']:.2f} | "
                         f"{m['nasa_sum']:.0f} | {m['nasa_mean']:.2f} |")
        short = te["length"] < WINDOW
        if short.any():
            lines.append(f"\nPadded engines (test trajectory shorter than {WINDOW} cycles): {int(short.sum())}. "
                         f"LSTM RMSE (capped) padded {rmse(point['LSTM'][0][short], y_cap[short]):.2f} vs "
                         f"not padded {rmse(point['LSTM'][0][~short], y_cap[~short]):.2f}.")

        lines += ["\n### 90 % intervals (capped RUL scale; calibration = validation-stage calibration engines)\n",
                  "| Model + method | Selected on validation? | Coverage | Below lower (dangerous) | Above upper | Mean width | "
                  "Coverage RUL<25 | Coverage RUL 50–75 |", "|---|---|---|---|---|---|---|---|"]
        sel = selected[subset]
        candidates = [("GBM", "Conformal (Mondrian)"), ("LSTM", "Conformal (Mondrian)"), ("LSTM", "Conformal (normalized)"),
                      ("LSTM", "Conformal (fixed)"), ("LSTM", "Deep ensemble")]
        for model, method in candidates:
            if model == "GBM":
                iv = build_intervals({"mu": point["GBM"][1]}, {"mu": point["GBM"][0]}, y_cal)[method]
            else:
                ev, cl = deep["LSTM"]
                iv = build_intervals(cl, ev, y_cal)[method]
            m = interval_metrics(*iv, y_cap)
            s = safety_metrics(*iv, y_cap, te["unit"])
            flag = "**yes**" if (model == sel["model"] and method == sel["method"]) else ""
            c = m["conditional"]
            lines.append(f"| {model} + {method} | {flag} | {m['coverage']:.3f} | {s['below']:.3f} | {s['above']:.3f} | "
                         f"{m['width']:.1f} | {c['0-25'][0]:.2f} (n={c['0-25'][1]}) | {c['50-75'][0]:.2f} (n={c['50-75'][1]}) |")
    (RESULTS / "final_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
