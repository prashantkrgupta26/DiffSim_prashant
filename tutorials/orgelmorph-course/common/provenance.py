"""Tutorial-facing provenance: write ``metadata.json`` for a run.

Thin wrapper over :mod:`diffsim.diagnostics.provenance` (the single source of
truth, which lives in the installed package so the research code shares it).
This module adds the tutorials' ergonomics: a :class:`RunProvenance` context
manager that times the run, captures peak device memory, records the exit
reason, and writes ``metadata.json`` into the output directory on exit — even
if the run raised (the exit reason then names the exception).
"""
from __future__ import annotations

import os

from diffsim.diagnostics.provenance import (collect_environment,
                                            collect_metadata, git_commit,
                                            git_dirty, write_metadata)
from diffsim.diagnostics.profiling import (peak_device_memory_bytes,
                                           reset_peak_memory)
import time


class RunProvenance:
    """Context manager that records and writes ``metadata.json``.

    Parameters
    ----------
    output_dir : str
        Directory to write ``metadata.json`` into.
    run_info : dict, optional
        Static run fields known up front (precision, device, solver, mesh,
        time_integrator, tolerances, seeds ...). More can be added later via
        :meth:`update`.

    Usage
    -----
    >>> with RunProvenance("outputs/run", {"solver": "cudss"}) as prov:  # doctest: +SKIP
    ...     prov.update(mesh={"level": 7})
    ...     ...                                   # run the simulation
    ...     prov.set_exit("completed")
    The ``metadata.json`` (with wall_time_seconds, peak_device_memory_bytes and
    exit_reason filled in) is written when the block exits.
    """

    def __init__(self, output_dir, run_info=None):
        self.output_dir = output_dir
        self.run_info = dict(run_info or {})
        self._t0 = None
        self._exit_set = False
        self.metadata = None

    def update(self, **fields):
        """Merge additional run fields (e.g. discovered mesh/tolerances)."""
        self.run_info.update(fields)
        return self

    def set_exit(self, reason):
        """Record the exit reason explicitly (else inferred on __exit__)."""
        self.run_info["exit_reason"] = reason
        self._exit_set = True
        return self

    def __enter__(self):
        self._t0 = time.perf_counter()
        reset_peak_memory()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.run_info.setdefault("wall_time_seconds",
                                 time.perf_counter() - self._t0)
        self.run_info["wall_time_seconds"] = time.perf_counter() - self._t0
        self.run_info["peak_device_memory_bytes"] = peak_device_memory_bytes()
        if not self._exit_set:
            self.run_info["exit_reason"] = (
                "completed" if exc_type is None
                else f"error:{exc_type.__name__}")
        os.makedirs(self.output_dir, exist_ok=True)
        self.metadata = write_metadata(
            os.path.join(self.output_dir, "metadata.json"),
            run_info=self.run_info)
        return False       # never suppress an exception


def write_run_metadata(output_dir, run_info=None):
    """One-shot: collect + write ``metadata.json``; return the payload."""
    os.makedirs(output_dir, exist_ok=True)
    return write_metadata(os.path.join(output_dir, "metadata.json"),
                          run_info=run_info)


__all__ = [
    "RunProvenance", "write_run_metadata", "collect_environment",
    "collect_metadata", "write_metadata", "git_commit", "git_dirty",
]
