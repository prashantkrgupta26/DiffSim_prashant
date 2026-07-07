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


def test_blocktri_solver_sigma0(device):
    """Task-#6 production recipe as a solve_linear backend: steady NS
    system solved by FGMRES + exact-F block preconditioner."""
    import numpy as np
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from test_transient_adjoint import _lid
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh
    from diffsim.api.ns_bricks import assemble_linear_ns
    from diffsim.physics.poisson import gauss_points
    from diffsim.solvers.linsolve import solve_linear
    from scipy.sparse.linalg import splu

    tree = build_uniform(5, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    xq = gauss_points(mesh, dm.tables_by_p)
    rng = np.random.default_rng(0)
    pv = list(dm.bins)[0]
    ngp = len(xq[pv])
    aq = {pv: rng.standard_normal((ngp, 2)) * 0.3}
    dq = {pv: np.zeros(ngp)}
    fq = {pv: np.zeros((ngp, 2))}
    A, b = assemble_linear_ns(dm, aq, dq, fq, 0.01, sigma=0.0,
                              sig2tau=0.0)
    A = A.tolil()
    # velocity Dirichlet on the box boundary (without it the sigma=0
    # velocity block carries rigid modes -> singular F, FGMRES stalls)
    coords = mesh.node_coords[cons.free_nodes]
    bdry = np.zeros(len(coords), bool)
    for c in range(2):
        bdry |= (np.abs(coords[:, c]) < 1e-12) | \
                (np.abs(coords[:, c] - 1) < 1e-12)
    for i in np.where(bdry)[0]:
        for c in range(2):
            r = i * 3 + c
            A.rows[r] = [int(r)]
            A.data[r] = [1.0]
    A.rows[2] = [2]; A.data[2] = [1.0]
    A = A.tocsr()
    b = rng.standard_normal(A.shape[0])
    cache = {("blocktri_meta", "t"): {"ndof": 3}}
    x = solve_linear(A, b, solver="blocktri", tol=1e-10, cache=cache,
                     cache_key="t")
    x_ref = splu(A.tocsc()).solve(b)
    rel = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)
    assert rel < 1e-8, rel
