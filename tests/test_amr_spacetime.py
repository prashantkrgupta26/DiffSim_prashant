"""A3 (C3 Phase-3): space-time interaction — a remesh DURING a variable-step
BDF2 march. Both BDF history levels are conservatively transferred across the
remesh; the temporal order is measured across it and must recover order 2.
Needs the CH kernel -> GPU (tier2).
"""
import numpy as np
import pytest

from diffsim.octree.build import build_uniform, refine_elements
from diffsim.octree.balance import balance2to1
from diffsim.adaptivity.amr_march import build_ch_stepper, remesh

pytestmark = pytest.mark.tier2

M, KAPPA = 1.0, 5e-4


def _ic(coords):
    return 0.6 + 0.2 * np.cos(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])


def _band(tree, r=0.3, band=1.5, cap=5):
    rr = np.sqrt(((tree.centers() - 0.5) ** 2).sum(1))
    return (np.abs(rr - r) < tree.h() * band) & (tree.levels < cap)


def _march(dt, N, device, remesh_frac=0.5):
    st, mesh, cons = build_ch_stepper(build_uniform(4, dim=2), M, KAPPA, dt,
                                      order=2, energy="poly", device=device)
    st.set_initial(_ic, mu_init="consistent")
    rstep = int(round(remesh_frac * N))
    for i in range(1, N + 1):
        st.dt = dt
        st.step()
        if i == rstep:
            new_tree = balance2to1(refine_elements(mesh.tree, _band(mesh.tree)))
            st, mesh, cons = remesh(st, mesh, cons, new_tree, M, KAPPA,
                                    "poly", device)   # transfers BOTH levels
    return st.x[0::2].copy()


def test_bdf2_order_recovered_across_remesh(device):
    T = 0.048
    ref = _march(T / 600, 600, device)
    Ns = (15, 30, 60)
    errs = [float(np.abs(_march(T / N, N, device) - ref).max()) for N in Ns]
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    # order 2 is preserved/recovered across the remesh (both history levels
    # transferred); >2x headroom over the BDF1-restart order-1 failure mode
    assert min(orders) > 1.7, (errs, orders)
