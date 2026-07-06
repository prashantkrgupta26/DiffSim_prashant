"""M1c gate: single-step Leray adjoint dJ/dnu vs FD (production stepper
#2). Frozen history; J = 0.5 |u_new|^2. FD reruns ONE step from the same
state at nu +- eps."""
import numpy as np
import pytest

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.steppers.leray import LerayProjectionStepper
from diffsim.sbm.leray_adjoint import LerayStepAdjoint

pytestmark = pytest.mark.ad


def _lid(x, t):
    g = np.zeros((len(x), 2))
    g[np.abs(x[:, 1] - 1.0) < 1e-12, 0] = 1.0
    return g


def _make(nu, device, steps=3, picard=1, finescale=False):
    tree = build_uniform(4, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    st = LerayProjectionStepper(
        dm, nu, 0.05, f_fn=lambda x, t: np.zeros((len(x), 2)),
        g_fn=_lid, order=2, picard_iters=picard)
    st.ppe_finescale = finescale
    st.set_initial(lambda x: np.zeros((len(x), 2)))
    for _ in range(steps):
        st.step()
    return st


import pytest as _pt


@_pt.mark.parametrize("picard,finescale",
                      [(1, False), (2, False), (1, True)])
def test_leray_step_nu_gradient(picard, finescale, device):
    nu0 = 0.02
    st = _make(nu0, device, picard=picard, finescale=finescale)
    adj = LerayStepAdjoint(st)             # capture frozen pre-step state
    u_new, _ = st.step()
    dJdu = u_new.copy()                    # J = 0.5 sum u^2
    g_adj = adj.nu_gradient(dJdu)

    # FD: rerun the SAME step from the same frozen history at nu +- eps
    eps = 1e-6
    Js = []
    for dnu in (+eps, -eps):
        st2 = _make(nu0, device, picard=picard,
                    finescale=finescale)  # identical history (seed
        #                                    path: deterministic stepper)
        st2.nu = nu0 + dnu
        u2, _ = st2.step()
        Js.append(0.5 * float((u2 ** 2).sum()))
    fd = (Js[0] - Js[1]) / (2 * eps)
    assert abs(fd - g_adj) < 5e-5 * max(abs(fd), 1e-12), (fd, g_adj)
