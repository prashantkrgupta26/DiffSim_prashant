"""M1c gate: transient adjoint chain — dJ/dnu over a 4-step BDF2 cavity
run vs central FD (full rerun). All cotangents taped; no FD inside the
adjoint."""
import numpy as np
import pytest

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.steppers.linearized import LinearizedMonolithicStepper
from diffsim.sbm.transient_adjoint import TransientAdjoint

pytestmark = pytest.mark.ad


def _lid(x, t):
    g = np.zeros((len(x), 2))
    g[np.abs(x[:, 1] - 1.0) < 1e-12, 0] = 1.0
    return g


def _run(nu, device, n_steps=4, record=False):
    tree = build_uniform(4, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    st = LinearizedMonolithicStepper(
        dm, nu, 0.05, f_fn=lambda x, t: np.zeros((len(x), 2)),
        g_fn=_lid, order=2)
    st.finescale_extrap = False
    st.set_initial(lambda x: np.zeros((len(x), 2)))
    if record:
        ta = TransientAdjoint(st)
        xs = ta.run(n_steps)
        return ta, xs
    xs = []
    for _ in range(n_steps):
        xs.append(st.step().reshape(-1).copy())
    return None, xs


def test_transient_nu_gradient(device):
    nu0 = 0.02
    ta, xs = _run(nu0, device, record=True)
    J0 = 0.5 * sum(float(x @ x) for x in xs)
    dJdx = [x.copy() for x in xs]
    g_adj = ta.nu_gradient(dJdx)

    eps = 1e-6
    Js = []
    for dnu in (+eps, -eps):
        _, xs2 = _run(nu0 + dnu, device)
        Js.append(0.5 * sum(float(x @ x) for x in xs2))
    fd = (Js[0] - Js[1]) / (2 * eps)
    print(f"transient dJ/dnu: adjoint = {g_adj:.8e}  FD = {fd:.8e}  "
          f"rel = {abs(g_adj - fd) / max(abs(fd), 1e-14):.2e}")
    assert abs(g_adj - fd) < 5e-5 * max(abs(fd), 1e-12), (g_adj, fd, J0)
