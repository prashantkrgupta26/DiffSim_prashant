"""M1b GPU-residency gates: the steppers' solver backends agree.

Parity contract: 'fused' (device single-sync Krylov) must reproduce 'splu'
trajectories to solver tolerance on both steppers; 'amgx' joins the matrix
when pyamgx is importable (skipped otherwise, never failed)."""
import importlib.util
import os
import sys

import numpy as np
import pytest
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh

sys.path.insert(0, os.path.dirname(__file__))
from test_ns_stepper import u_ex, f_ex, NU  # noqa: E402

pytestmark = pytest.mark.tier5

HAS_AMGX = importlib.util.find_spec("pyamgx") is not None
HAS_CUDSS = importlib.util.find_spec("nvmath") is not None
SOLVERS = (["splu", "fused"] + (["amgx"] if HAS_AMGX else [])
           + (["cudss"] if HAS_CUDSS else []))


def _run_mono(solver, device, level=4, n=8, T=0.2):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    from diffsim.steppers.linearized import LinearizedMonolithicStepper
    st = LinearizedMonolithicStepper(
        dm, NU, T / n, f_fn=f_ex, g_fn=lambda x, t: np.zeros((len(x), 2)),
        order=2, timestab=False, solver=solver)
    st.set_initial(lambda x: u_ex(x, 0.0))
    for _ in range(n):
        x = st.step()
    return x[:, :2]


def _run_leray(solver, device, level=4, n=8, T=0.2):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    from diffsim.steppers.leray import LerayProjectionStepper
    st = LerayProjectionStepper(
        dm, NU, T / n, f_fn=f_ex, g_fn=lambda x, t: np.zeros((len(x), 2)),
        order=2, picard_iters=1, timestab=False, solver=solver)
    st.set_initial(lambda x: u_ex(x, 0.0))
    for _ in range(n):
        u, p = st.step()
    return u


def _parity(runner_name, solver, device, tol):
    """AMGX runs SUBPROCESS-ISOLATED: it initializes cleanly in a fresh
    process but can crash ('Internal error'/SIGABRT) when first initialized
    after heavy warp/torch CUDA activity in the same process — measured;
    m1b findings 8. Everything else runs in-process."""
    if solver != "amgx":
        run = globals()[runner_name]
        ref = run("splu", device)
        alt = run(solver, device)
        scale = np.abs(ref).max()
        assert np.abs(alt - ref).max() < tol * scale, (
            solver, np.abs(alt - ref).max(), scale)
        return
    import subprocess
    import sys as _sys
    code = (
        "import sys, numpy as np\n"
        f"sys.path.insert(0, {os.path.dirname(__file__)!r})\n"
        f"from test_stepper_solvers import {runner_name} as run\n"
        f"ref = run('splu', {device!r})\n"
        f"alt = run('amgx', {device!r})\n"
        "scale = np.abs(ref).max()\n"
        f"assert np.abs(alt - ref).max() < {tol} * scale\n"
        "print('PARITY_OK')\n")
    r = subprocess.run([_sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=600)
    assert "PARITY_OK" in r.stdout, (r.returncode, r.stdout[-400:],
                                     r.stderr[-400:])


@pytest.mark.parametrize("solver", SOLVERS[1:])
def test_monolithic_solver_parity(solver, device):
    _parity("_run_mono", solver, device, 1e-7)


@pytest.mark.parametrize("solver", SOLVERS[1:])
def test_leray_solver_parity(solver, device):
    _parity("_run_leray", solver, device, 1e-6)
