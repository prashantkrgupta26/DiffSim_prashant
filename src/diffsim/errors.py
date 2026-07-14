"""DiffSim typed exception hierarchy.

Motivation (critical-eval P0.2 / P0.3): several runtime-correctness checks in
the public solver/assembly paths were guarded by bare ``assert`` statements,
which Python strips entirely under ``python -O``.  In the device-assembly and
device-solve paths an invalidated symbolic slot map or a 32-bit slot overflow
can then scatter *wrong* sparse entries instead of raising — a silent
correctness failure.  Load-bearing guards therefore raise these typed
exceptions, which survive ``-O``.

Back-compat is deliberate:

* :class:`ConfigError` subclasses :class:`ValueError` — existing
  ``except ValueError`` handlers keep working.
* :class:`SolverError` (and its :class:`ConvergenceError` subclass) subclass
  :class:`RuntimeError` — the film/multiphase adaptive-timestep reject ladders
  already catch ``RuntimeError`` to convert an *expected* numerical
  non-convergence into a NaN divergence signal, and continue to do so.

Reserve plain ``assert`` for genuinely-impossible developer invariants that are
independently guaranteed and not safety-critical.
"""

from __future__ import annotations


class DiffSimError(Exception):
    """Base class for all DiffSim-raised errors."""


class ConfigError(DiffSimError, ValueError):
    """Invalid user configuration or input (bad option string, unsupported
    mesh/order/backend combination, out-of-range parameter).

    Subclasses :class:`ValueError` for back-compat with callers that already
    catch ``ValueError`` on bad input.
    """


class BackendError(DiffSimError, RuntimeError):
    """A backend / device contract was violated: a symbolic slot map failed to
    validate, a 32-bit device slot index overflowed, or a requested path is not
    supported by the selected assembly/solver backend.

    These guard *silent-corruption* failure modes (wrong sparse entries under
    ``python -O``); they are always programming/configuration bugs, never an
    expected numerical outcome.
    """


class SolverError(DiffSimError, RuntimeError):
    """A linear/nonlinear solve failed for a runtime reason.

    Subclasses :class:`RuntimeError` so the adaptive-stepper reject ladders that
    already catch ``RuntimeError`` continue to treat it as a divergence signal.
    """


class ConvergenceError(SolverError):
    """An iterative or direct solve did not reach the requested tolerance."""


# --- programming / environment errors that must never be swallowed ---------
# Broad ``except Exception`` around a solve historically converted *every*
# failure into an all-NaN vector so the timestep-reject heuristic could react
# (critical-eval P0.3).  That also hides real defects — an API rename
# (AttributeError), a bad call (TypeError), a missing optional dependency
# (ImportError), or device/host OOM (MemoryError) would masquerade as a
# "physically hard timestep" and be retried forever.  ``reraise_if_bug`` is
# called at the top of those handlers to let genuine numerical failures
# (singular factor, non-convergence) fall through to the NaN signal while
# surfacing bugs with a real traceback.
_BUG_ERRORS: tuple[type[BaseException], ...] = (
    TypeError,
    AttributeError,
    NameError,
    ImportError,          # includes ModuleNotFoundError
    IndexError,
    KeyError,
    MemoryError,
    SyntaxError,
    RecursionError,
    KeyboardInterrupt,
    SystemExit,
)


def reraise_if_bug(exc: BaseException) -> None:
    """Re-raise *exc* if it is a programming/environment error that must not be
    swallowed as an expected numerical solver failure; otherwise return so the
    caller can handle it (e.g. emit a NaN divergence signal).
    """
    if isinstance(exc, _BUG_ERRORS):
        raise exc


__all__ = [
    "DiffSimError",
    "ConfigError",
    "BackendError",
    "SolverError",
    "ConvergenceError",
    "reraise_if_bug",
]
