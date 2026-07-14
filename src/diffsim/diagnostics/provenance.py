"""Provenance collection — the reproducibility fingerprint of a run.

Gathers the full environment + run fingerprint the course requires in every
``metadata.json`` (spec Phase 0): diffsim commit + dirty flag, timestamp,
host, GPU + memory, CUDA driver/runtime, python/warp/torch/nvmath versions,
precision, device, solver, mesh, time integrator, tolerances, seeds, wall
time, peak device memory, exit reason.

Every field is **best-effort**: if a probe fails (missing torch, no GPU, not a
git checkout) the field is ``None`` and collection never raises. This is the
single source of truth; ``tutorials/orgelmorph-course/common/provenance.py``
thins over it for the tutorial-facing API.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import socket
import subprocess
import sys

# Field set required by the spec; used to guarantee every key is present
# (value ``None`` when a probe is unavailable) so downstream consumers can
# rely on the schema.
_RUN_FIELDS = (
    "precision", "device", "solver", "mesh", "time_integrator",
    "nonlinear_tol", "linear_tol", "seeds",
    "wall_time_seconds", "peak_device_memory_bytes", "exit_reason",
)


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _diffsim_repo_root():
    # this file: src/diffsim/diagnostics/provenance.py -> repo root is 3 up
    return os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        os.pardir, os.pardir, os.pardir))


def git_commit(repo=None):
    """Short-and-long commit hash of the diffsim checkout, or ``None``."""
    repo = repo or _diffsim_repo_root()

    def _run():
        out = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    return _safe(_run)


def git_dirty(repo=None):
    """``True`` if the checkout has uncommitted changes, ``None`` if unknown."""
    repo = repo or _diffsim_repo_root()

    def _run():
        out = subprocess.run(["git", "-C", repo, "status", "--porcelain"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return None
        return bool(out.stdout.strip())
    return _safe(_run)


def _pkg_version(name):
    def _run():
        mod = __import__(name)
        return getattr(mod, "__version__", None)
    return _safe(_run)


def _torch_info():
    info = {"torch_version": None, "cuda_runtime": None, "cuda_driver": None,
            "gpu": None, "gpu_memory_bytes": None, "device_count": None}

    def _run():
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            info["device_count"] = torch.cuda.device_count()
            info["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_bytes"] = int(props.total_memory)
            drv = _safe(lambda: torch.cuda.driver_version()) \
                if hasattr(torch.cuda, "driver_version") else None
            info["cuda_driver"] = drv
    _safe(_run)
    return info


def _nvidia_smi_driver():
    def _run():
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0].strip()
        return None
    return _safe(_run)


def collect_environment():
    """Static environment fingerprint (host, versions, GPU, CUDA).

    Returns a dict; all values best-effort (``None`` if unavailable).
    """
    torch_info = _torch_info()
    env = {
        "diffsim_commit": git_commit(),
        "git_dirty": git_dirty(),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "hostname": _safe(socket.gethostname),
        "platform": _safe(platform.platform),
        "python_version": sys.version.split()[0],
        "warp_version": _pkg_version("warp"),
        "torch_version": torch_info["torch_version"],
        "nvmath_version": _pkg_version("nvmath"),
        "numpy_version": _pkg_version("numpy"),
        "scipy_version": _pkg_version("scipy"),
        "gpu": torch_info["gpu"],
        "gpu_memory_bytes": torch_info["gpu_memory_bytes"],
        "gpu_count": torch_info["device_count"],
        "cuda_runtime": torch_info["cuda_runtime"],
        "cuda_driver": torch_info["cuda_driver"] or _nvidia_smi_driver(),
    }
    return env


def collect_metadata(run_info=None):
    """Full ``metadata.json`` payload: environment + run-specific fields.

    Parameters
    ----------
    run_info : dict, optional
        Run-specific fields to record; recognised keys are the spec's run
        fields (``precision``, ``device``, ``solver``, ``mesh``,
        ``time_integrator``, ``nonlinear_tol``, ``linear_tol``, ``seeds``,
        ``wall_time_seconds``, ``peak_device_memory_bytes``, ``exit_reason``).
        Any extra keys are also stored (best-effort, never rejected).

    Returns
    -------
    dict
        Environment fields plus every spec run field (defaulting to ``None``)
        overlaid with ``run_info``.
    """
    meta = collect_environment()
    for k in _RUN_FIELDS:
        meta.setdefault(k, None)
    if run_info:
        for k, v in run_info.items():
            meta[k] = v
    return meta


def write_metadata(path, run_info=None, metadata=None):
    """Write ``metadata.json`` (pretty-printed) and return the payload dict.

    Pass either ``run_info`` (collected fresh) or a pre-built ``metadata``
    dict. Non-JSON-serialisable values are coerced to strings so a stray numpy
    scalar never aborts the write.
    """
    meta = metadata if metadata is not None else collect_metadata(run_info)
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)
    return meta


__all__ = [
    "git_commit", "git_dirty", "collect_environment", "collect_metadata",
    "write_metadata",
]
