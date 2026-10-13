"""Run the pre-registered ablation plan. Resumable: finished runs are skipped.

    python -m ml.ablation.run_ablation                       # everything in the plan
    python -m ml.ablation.run_ablation --categories metal_nut --experiments B0 BB-r18
    python -m ml.ablation.run_ablation --dry-run             # list what would run

Order: every experiment with seed 0 first (all categories), extra seeds afterwards, so a partial run
still covers every factor. Each run's fit and eval happen in separate processes (clean peak memory).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from ml.ablation.run_one import DEFAULT_PLAN, PROJECT_DIR, load_plan, run_name
from ml.ablation.summarize import summarize


def planned_runs(plan: dict, categories: list[str], experiments: list[str] | None) -> list[tuple[str, str, int]]:
    runs = []
    for pass_extra_seeds in (False, True):
        for e in plan["experiments"]:
            if experiments and e["id"] not in experiments or e.get("not_run"):
                continue
            seeds = e.get("seeds", [0])
            for seed in (seeds[1:] if pass_extra_seeds else seeds[:1]):
                runs += [(e["id"], c, seed) for c in categories]
    return runs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--experiments", nargs="*", default=None)
    p.add_argument("--data-root", type=Path, default=PROJECT_DIR / "data/mvtec_ad")
    p.add_argument("--out", type=Path, default=PROJECT_DIR / "artifacts/ablation")
    p.add_argument("--results", type=Path, default=PROJECT_DIR / "results/ablation")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    plan = load_plan(args.plan)
    categories = args.categories or plan["categories"]
    runs = planned_runs(plan, categories, args.experiments)
    print(f"{len(runs)} runs planned (plan sha256 {plan['_sha256'][:12]})", flush=True)
    logs = args.out / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    for i, (exp_id, cat, seed) in enumerate(runs, 1):
        name = run_name(exp_id, cat, seed)
        stage_dir = args.out / "stages" / name
        failed = stage_dir / "failed.json"
        if (stage_dir / "eval.json").exists() or (failed.exists() and not args.retry_failed):
            continue
        if args.dry_run:
            print("would run", name)
            continue
        failed.unlink(missing_ok=True)
        t0 = time.time()
        common = ["--plan", str(args.plan), "--exp", exp_id, "--category", cat, "--seed", str(seed),
                  "--data-root", str(args.data_root), "--out", str(args.out)]
        if exp_id == "B0" and seed == 0:
            common.append("--keep-bank")  # keep one reference bank per category
        for stage in ("fit", "eval"):
            with open(logs / f"{name}.{stage}.log", "w", encoding="utf-8") as log:
                rc = subprocess.run(
                    [sys.executable, "-W", "ignore", "-m", "ml.ablation.run_one", stage, *common],
                    cwd=PROJECT_DIR, stdout=log, stderr=subprocess.STDOUT, check=False,
                ).returncode
            if rc != 0:
                tail = (logs / f"{name}.{stage}.log").read_text(encoding="utf-8", errors="replace")[-2000:]
                stage_dir.mkdir(parents=True, exist_ok=True)
                failed.write_text(json.dumps({"status": "FAILED", "stage": stage, "returncode": rc,
                                              "log_tail": tail}, indent=2), encoding="utf-8")
                break
        status = "FAILED" if failed.exists() else "OK"
        print(f"[{i}/{len(runs)}] {name}: {status} in {time.time() - t0:.0f} s", flush=True)
        summarize(plan, args.out, args.results)  # refresh CSVs after every run

    if not args.dry_run:
        summarize(plan, args.out, args.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
