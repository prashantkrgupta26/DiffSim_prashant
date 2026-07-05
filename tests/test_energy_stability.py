"""M1b Task 9: the s = 1/2 energy-stability claim (Biswajit's draft Prop. 2)
measured on an under-resolved unforced decay problem — with s = 1/2 the
generalized convection is exactly skew (locked discretely in test_vms), so
the convective term cannot inject kinetic energy: KE must be non-increasing
(BDF + viscosity + stabilization are all dissipative). For s = 0 the
symmetric part -(1/2)(div a_h) u is sign-indefinite; its energy behavior is
RECORDED, not direction-asserted (the degenerate-observable lesson)."""
import os
import sys

import numpy as np
import pytest
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.steppers.linearized import LinearizedMonolithicStepper

sys.path.insert(0, os.path.dirname(__file__))
from test_ns_bricks import u_star  # noqa: E402

pytestmark = pytest.mark.tier5


def _ke(st):
    uq = st._gp_eval(st._node_field(st.hist.pre1))[0]
    dm = st.dm
    tot = 0.0
    for pv in uq:
        tb = dm.tables_by_p[pv]
        h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
        jac = (h / 2.0) ** dm.dim
        w = np.tile(tb.w, len(h)) * np.repeat(jac, tb.nqp)
        tot += ((uq[pv] ** 2).sum(1) * w).sum()
    return 0.5 * tot


@pytest.mark.parametrize("s_skew", [0.5, 0.0])
def test_unforced_decay_energy(s_skew, device):
    # level 4, nu = 1e-4 (convection-dominated, under-resolved), no forcing,
    # zero-Dirichlet box, strong initial vortex
    tree = build_uniform(4, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    st = LinearizedMonolithicStepper(
        dm, 1e-4, 0.02, f_fn=lambda x, t: np.zeros((len(x), 2)),
        g_fn=lambda x, t: np.zeros((len(x), 2)), order=2, s_skew=s_skew)
    st.set_initial(lambda x: 2.0 * u_star(x))
    ke = [_ke(st)]
    for _ in range(50):
        st.step()
        ke.append(_ke(st))
    ke = np.array(ke)
    if s_skew == 0.5:
        # THE claim: no convective energy injection — monotone decay
        # (tiny tolerance for the pressure/PSPG projection transients)
        growth = (ke[1:] - ke[:-1]).max()
        assert growth < 1e-10 * ke[0], (growth, ke[0])
        assert ke[-1] < ke[0]
    else:
        # recorded, not direction-asserted: must stay FINITE/bounded; the
        # measured trajectory is printed for the findings log
        assert np.isfinite(ke).all()
        assert ke.max() < 10.0 * ke[0], ke.max() / ke[0]
        print(f"s=0 KE trajectory: start {ke[0]:.4e} max {ke.max():.4e} "
              f"end {ke[-1]:.4e} (ratio max/start "
              f"{ke.max() / ke[0]:.3f})")
