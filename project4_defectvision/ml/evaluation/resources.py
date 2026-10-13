"""Process memory measurement.

Peak memory is a per-process high-water mark that cannot be reset, so each measured phase (fit, serve)
runs in its own fresh process (see ml/ablation/run_one.py).
"""

from __future__ import annotations

import sys


def current_rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 2**20


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far, in MiB.

    Windows: peak working set (psutil ``peak_wset``). Linux/macOS: ``ru_maxrss``.
    """
    if sys.platform == "win32":
        import psutil

        return psutil.Process().memory_info().peak_wset / 2**20
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10  # bytes on macOS, KiB on Linux
