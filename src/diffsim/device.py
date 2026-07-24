"""Central device resolver — pick the compute device once, for the package.

DiffSim historically hardcoded ``"cuda:0"`` throughout its scripts and library
defaults, which made it fail immediately on CPU-only machines (e.g. a laptop
without a GPU). This module resolves the device by a single, well-defined
precedence so DiffSim runs out-of-the-box on CPU and unchanged on GPU boxes.

Precedence (highest first)::

    1. explicit argument   default_device(prefer="cuda:1")   — always wins
    2. env override        DIFFSIM_DEVICE=cpu / cuda:0 / cuda:1
    3. auto-detect         "cuda:0" if a CUDA device is visible, else "cpu"

Auto-detection is defensive: if Warp is not importable, or Warp reports no
CUDA devices, it returns ``"cpu"`` rather than raising — so ``import`` and a
call to :func:`default_device` never crash on a machine without CUDA.

Typical use::

    from diffsim import default_device
    device = default_device()            # respects env + auto-detect
    device = default_device(user_arg)    # user_arg wins if not None
"""

from __future__ import annotations

import os

__all__ = ["default_device", "default_linsolver"]

_ENV_VAR = "DIFFSIM_DEVICE"
_LINSOLVER_ENV_VAR = "DIFFSIM_LINSOLVER"


def _cuda_is_available() -> bool:
    """True iff Warp is importable AND reports at least one CUDA device.

    Never raises: any import/probe failure is treated as "no CUDA".
    """
    try:
        import warp as wp

        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


def default_device(prefer: str | None = None) -> str:
    """Resolve the compute device string ("cpu", "cuda:0", "cuda:1", ...).

    Precedence: explicit ``prefer`` argument  >  ``DIFFSIM_DEVICE`` env var
    >  auto-detect ("cuda:0" if a CUDA device is visible, else "cpu").

    Args:
        prefer: an explicit device string; if not ``None`` it is returned
            verbatim (the caller has already decided).

    Returns:
        A device string suitable for Warp / DiffSim APIs.
    """
    if prefer is not None:
        return prefer

    env = os.environ.get(_ENV_VAR)
    if env:
        return env.strip()

    return "cuda:0" if _cuda_is_available() else "cpu"


def _is_cpu_device(device: str | None) -> bool:
    """True iff ``device`` denotes the host CPU (or is None -> auto-detect)."""
    if device is None:
        device = default_device()
    return device.strip().lower().startswith("cpu")


def default_linsolver(device: str | None = None,
                      prefer: str | None = None) -> str:
    """Resolve a working linear-solver backend for ``device``.

    Mirrors :func:`default_device`: the GPU-only direct solvers (``cudss``,
    ``amgx``, ``fused``) are the right default on a CUDA box but fail hard on
    a CPU-only machine (e.g. a laptop without a GPU). This picks the host
    direct solver (``splu`` — scipy SuperLU) on CPU so DiffSim's benchmarks
    run out-of-the-box, and keeps the GPU default (``cudss``) unchanged on a
    CUDA device.

    Precedence (highest first)::

        1. explicit argument   default_linsolver(prefer="blockch") — always wins
        2. env override        DIFFSIM_LINSOLVER=splu / cudss / blockch / ...
        3. auto by device      "splu" on CPU, "cudss" on CUDA

    Args:
        device: the resolved device string (``"cpu"``, ``"cuda:0"``, ...). If
            ``None``, :func:`default_device` is consulted (so env + auto-detect
            still drive the choice). Only used for the auto branch.
        prefer: an explicit solver name; if not ``None`` it is returned
            verbatim (the caller has already decided).

    Returns:
        A solver-name string suitable for ``solve_linear`` / the steppers.

    Never raises on a machine without a GPU: the CPU branch does not import
    Warp/CUDA.
    """
    if prefer is not None:
        return prefer

    env = os.environ.get(_LINSOLVER_ENV_VAR)
    if env:
        return env.strip()

    return "splu" if _is_cpu_device(device) else "cudss"
