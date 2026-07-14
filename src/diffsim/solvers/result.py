"""Structured solver-return types (critical-eval P2.1).

The linear/nonlinear solvers historically returned bare arrays or ``(x, info)``
tuples, so convergence/telemetry handling was inconsistent across backends.
These dataclasses give a uniform, self-describing return value.

They are introduced **non-invasively**: :func:`diffsim.solvers.solve_linear`
still returns a bare host array by default, and only returns a
:class:`LinearSolveResult` when called with ``return_result=True``.  The
dataclass also unpacks like the historical ``(x, info)`` tuple via
:meth:`LinearSolveResult.__iter__`, so ``x, info = ...`` keeps working if a
caller opts in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LinearSolveResult:
    """Result of a single linear solve ``A x = b``.

    Attributes
    ----------
    x:
        Solution vector (host array).
    converged:
        Whether the solve reached the requested tolerance.  Direct solves that
        succeed report ``True``.
    iterations:
        Krylov iteration count, or ``None`` for direct solves.
    residual_norm:
        Final residual norm if measured, else ``None``.
    backend:
        Solver string (``"splu"``, ``"cudss"``, ``"fused"``, ``"blockch"`` ...).
    reason:
        Short human-readable status (``"converged"``, ``"maxiter"`` ...).
    info:
        Backend-specific extra telemetry.
    """

    x: Any
    converged: bool = True
    iterations: Optional[int] = None
    residual_norm: Optional[float] = None
    backend: str = ""
    reason: str = "converged"
    info: dict = field(default_factory=dict)

    def __iter__(self):
        # tuple back-compat: ``x, info = result``
        yield self.x
        yield self.info

    def __getitem__(self, i):
        return (self.x, self.info)[i]


@dataclass
class NonlinearSolveResult:
    """Result of a Newton/SNES nonlinear solve."""

    u: Any
    converged: bool
    iterations: int
    residual_norm: Optional[float] = None
    fnorm_history: list = field(default_factory=list)
    reason: str = "converged"

    @property
    def _info(self) -> dict:
        return {
            "iters": self.iterations,
            "fnorm_history": self.fnorm_history,
            "converged": self.converged,
        }

    def __iter__(self):
        # tuple back-compat: ``u, info = solver.solve(u0)``
        yield self.u
        yield self._info

    def __getitem__(self, i):
        return (self.u, self._info)[i]


__all__ = ["LinearSolveResult", "NonlinearSolveResult"]
