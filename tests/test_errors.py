"""CPU-only gate for the typed-exception hierarchy and structured solver
results introduced for critical-eval P0.2 / P0.3 / P2.1.

Deliberately Warp/CUDA-free: it exercises only the exception hierarchy, the
``reraise_if_bug`` policy, and the scipy-SuperLU ``solve_linear`` path (no
device code).  This is the test the CPU CI job runs as a real gate.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from diffsim.errors import (
    BackendError,
    ConfigError,
    ConvergenceError,
    DiffSimError,
    SolverError,
    reraise_if_bug,
)
from diffsim.solvers.result import LinearSolveResult, NonlinearSolveResult


def test_hierarchy_and_backcompat_bases():
    # everything descends from the package base
    for exc in (ConfigError, BackendError, SolverError, ConvergenceError):
        assert issubclass(exc, DiffSimError)
    # ConfigError stays catchable as ValueError (back-compat)
    assert issubclass(ConfigError, ValueError)
    # SolverError/ConvergenceError stay catchable as RuntimeError so the
    # adaptive-stepper reject ladders (except RuntimeError) keep working
    assert issubclass(SolverError, RuntimeError)
    assert issubclass(ConvergenceError, SolverError)
    assert issubclass(ConvergenceError, RuntimeError)
    assert not issubclass(BackendError, ConfigError)


def test_config_error_catchable_as_valueerror():
    with pytest.raises(ValueError):
        raise ConfigError("bad option")


def test_convergence_error_catchable_as_runtimeerror():
    # the film/multiphase dt-reject ladders rely on this exact catch
    caught = False
    try:
        raise ConvergenceError("did not converge")
    except RuntimeError:
        caught = True
    assert caught


@pytest.mark.parametrize(
    "bug",
    [TypeError("x"), AttributeError("x"), NameError("x"),
     ImportError("x"), ModuleNotFoundError("x"), IndexError("x"),
     KeyError("x"), MemoryError()],
)
def test_reraise_if_bug_surfaces_programming_errors(bug):
    # a programming/environment error must NOT be swallowed as an expected
    # numerical solver failure (P0.3)
    with pytest.raises(type(bug)):
        reraise_if_bug(bug)


@pytest.mark.parametrize(
    "numerical",
    [ValueError("singular"), RuntimeError("cudss status"),
     ArithmeticError("overflow"), ZeroDivisionError()],
)
def test_reraise_if_bug_passes_numerical_failures(numerical):
    # expected numerical failures fall through so the caller can emit the
    # NaN divergence signal
    reraise_if_bug(numerical)  # returns None, does not raise


def _spd(n=6):
    d = np.arange(2, 2 + n, dtype=float)
    A = sp.diags([d, np.full(n - 1, 0.3), np.full(n - 1, 0.3)],
                 [0, 1, -1], format="csr")
    return A


def test_solve_linear_splu_bare_backcompat():
    A = _spd()
    b = np.ones(A.shape[0])
    x = solve_linear_import()(A, b, solver="splu")
    assert isinstance(x, np.ndarray)
    assert np.allclose(A @ x, b)


def test_solve_linear_return_result_optin():
    A = _spd()
    b = np.ones(A.shape[0])
    res = solve_linear_import()(A, b, solver="splu", return_result=True)
    assert isinstance(res, LinearSolveResult)
    assert res.converged is True
    assert res.backend == "splu"
    assert np.allclose(A @ res.x, b)
    # tuple back-compat: unpack and index
    x, info = res
    assert np.allclose(x, res.x)
    assert res[0] is res.x


def test_solve_linear_unknown_solver_raises_configerror():
    # a triggered error surfaces the typed exception through a real public
    # path — and stays catchable as ValueError (back-compat)
    A = _spd()
    b = np.ones(A.shape[0])
    sl = solve_linear_import()
    with pytest.raises(ConfigError):
        sl(A, b, solver="does-not-exist")
    with pytest.raises(ValueError):
        sl(A, b, solver="does-not-exist")


def test_solve_linear_cache_staleness_raises_configerror():
    sl = solve_linear_import()
    cache = {}
    A1 = _spd(6)
    sl(A1, np.ones(6), solver="splu", cache=cache, cache_key="k")
    A2 = _spd(7)          # different shape/pattern under the same key
    with pytest.raises(ConfigError):
        sl(A2, np.ones(7), solver="splu", cache=cache, cache_key="k")


def test_linearsolveresult_tuple_shims():
    r = LinearSolveResult(x=np.array([1.0, 2.0]), backend="fused")
    x, info = r
    assert list(x) == [1.0, 2.0]
    assert isinstance(info, dict)
    assert r[0] is r.x and r[1] is r.info


def test_nonlinearsolveresult_tuple_shims():
    r = NonlinearSolveResult(u=np.array([3.0]), converged=True,
                             iterations=4, fnorm_history=[1.0, 1e-9])
    u, info = r
    assert list(u) == [3.0]
    assert info["iters"] == 4
    assert info["converged"] is True
    assert info["fnorm_history"][-1] == 1e-9
    assert r[0] is r.u


def solve_linear_import():
    # imported lazily so this module stays importable even if the solver
    # subpackage grows Warp-dependent siblings
    from diffsim.solvers.linsolve import solve_linear
    return solve_linear


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
