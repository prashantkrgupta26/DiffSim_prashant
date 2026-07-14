"""Profiling diagnostics — wall-clock timing and peak device memory.

Honest cost reporting (spec P2/C4): report REAL cost — wall time of labelled
stages and peak device memory — with CUDA synchronisation so timings are not
truncated by asynchronous kernel launches. Best-effort: works without torch
(memory fields become ``None``), never crashes a run.
"""
from __future__ import annotations

import time
from contextlib import contextmanager


def _cuda_sync():
    """Synchronise the CUDA device if torch is present, so a timer measures
    completed work rather than queued launches. No-op otherwise."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def peak_device_memory_bytes():
    """Peak CUDA memory (bytes) since the last reset, or ``None`` if
    unavailable. Uses ``torch.cuda.max_memory_allocated``."""
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except Exception:
        pass
    return None


def reset_peak_memory():
    """Reset the CUDA peak-memory high-water mark (best effort)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


@contextmanager
def timer(label="block", sync=True, sink=None):
    """Context manager timing a block of work (seconds).

    Parameters
    ----------
    label : str
        Stage name for the recorded entry.
    sync : bool
        Synchronise CUDA before start and before stop (measures completed GPU
        work). Set ``False`` for pure host timing.
    sink : dict, optional
        If given, ``sink[label]`` is set to the elapsed seconds.

    Yields
    ------
    dict
        A one-entry record ``{"label", "seconds"}`` populated on exit.
    """
    rec = {"label": label, "seconds": None}
    if sync:
        _cuda_sync()
    t0 = time.perf_counter()
    try:
        yield rec
    finally:
        if sync:
            _cuda_sync()
        rec["seconds"] = time.perf_counter() - t0
        if sink is not None:
            sink[label] = rec["seconds"]


class StageTimer:
    """Accumulate labelled stage timings across a run.

    >>> st = StageTimer()
    >>> with st.stage("assembly"):
    ...     pass
    >>> "assembly" in st.totals
    True
    """

    def __init__(self, sync=True):
        self.sync = sync
        self.totals = {}
        self.counts = {}

    @contextmanager
    def stage(self, label):
        if self.sync:
            _cuda_sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.sync:
                _cuda_sync()
            dt = time.perf_counter() - t0
            self.totals[label] = self.totals.get(label, 0.0) + dt
            self.counts[label] = self.counts.get(label, 0) + 1

    def summary(self):
        """Dict of ``label -> {"total_s", "calls", "mean_s"}`` (seconds)."""
        return {k: {"total_s": self.totals[k], "calls": self.counts[k],
                    "mean_s": self.totals[k] / self.counts[k]}
                for k in self.totals}


__all__ = [
    "timer", "StageTimer", "peak_device_memory_bytes", "reset_peak_memory",
]
