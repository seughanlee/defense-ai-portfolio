"""Validate MVTec AD categories and write a dataset report (counts, sizes, masks, duplicates, fingerprint).

    python scripts/inspect_dataset.py --data-root data/mvtec_ad --categories metal_nut screw capsule transistor
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.datasets.mvtec import DatasetError, validate_category  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--categories", nargs="+", required=True)
    p.add_argument("--out", type=Path, default=Path("artifacts/dataset_reports"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    exit_code = 0
    rows = []
    for cat in args.categories:
        try:
            r = validate_category(args.data_root, cat)
        except DatasetError as exc:
            print(f"[BLOCKED] {cat}: {exc}")
            rows.append({"category": cat, "status": "BLOCKED", "reason": str(exc)})
            exit_code = 2
            continue
        (args.out / f"{cat}.json").write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
        status = "PASS" if r.ok else "FAIL"
        exit_code = exit_code or (0 if r.ok else 1)
        train_n = sum(r.counts.get("train", {}).values())
        test = r.counts.get("test", {})
        print(f"[{status}] {cat}: train/good={train_n} test={test} sizes={r.image_sizes} "
              f"errors={len(r.errors)} warnings={len(r.warnings)} fingerprint={str(r.fingerprint)[:12]}")
        for e in r.errors[:10]:
            print("   ERROR", e)
        for w in r.warnings[:10]:
            print("   WARN ", w)
        rows.append({"category": cat, "status": status, "train_good": train_n, "test": test,
                     "image_sizes": r.image_sizes, "fingerprint": r.fingerprint})
    (args.out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
