"""Create the synthetic smoke-test dataset (MVTec layout). Not a benchmark.

    python scripts/make_synthetic.py --out data/synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.datasets.synthetic import make_synthetic_dataset  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/synthetic"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    print(make_synthetic_dataset(args.out, seed=args.seed))
