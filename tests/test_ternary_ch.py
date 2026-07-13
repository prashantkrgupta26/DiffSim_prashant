"""M4 P2.5 gates: ternary coupled CH — dual mass conservation, ternary
spinodal (both solutes separate), simplex bounds."""
import numpy as np
import pytest

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.ternary_ch import TernaryCHStepper

pytestmark = pytest.mark.tier2


def test_ternary_spinodal(device):
    tree = build_uniform(5, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    # chi_12 = 3 (strong solute-solute repulsion), mild chi_is
    # chi12=6: unstable eig -3.1 at (0.35,0.35) — chi12=3
    # was only -0.14 (marginal; no growth in test horizon)
    st = TernaryCHStepper(dm, chi=(6.0, 0.8, 0.8),
                          M=(1.0, -0.2, 1.0), kappa=(8e-4, 8e-4),
                          dt=0.005, order=1)
    rng = np.random.default_rng(4)
    st.set_initial(lambda x: 0.35 + 0.02 * rng.standard_normal(len(x)),
                   lambda x: 0.35 + 0.02 * rng.standard_normal(len(x)))

    def mass(vec):
        v, _ = st._gp(vec)
        m = 0.0
        for pv, b in dm.bins.items():
            h = mesh.tree.h()[mesh.bins[pv]]
            ne = len(mesh.conn_of[pv])
            wq = np.tile(dm.tables_by_p[pv].w, ne) \
                * np.repeat((h / 2) ** 2, b["nqp"])
            m += float((wq * v[pv]).sum())
        return m

    m1_0, m2_0 = mass(st.hist[0][0]), mass(st.hist[0][1])
    for n in range(40):
        p1, p2 = st.step()
        assert abs(mass(p1) - m1_0) < 1e-9, n
        assert abs(mass(p2) - m2_0) < 1e-9, n
    ps = 1.0 - p1 - p2
    # separation happened + simplex respected
    sep = (p1.max() - p1.min()) + (p2.max() - p2.min())
    print(f"ternary: p1 [{p1.min():.2f},{p1.max():.2f}] "
          f"p2 [{p2.min():.2f},{p2.max():.2f}] "
          f"ps [{ps.min():.2f},{ps.max():.2f}] sep={sep:.2f} "
          f"|dm|=({abs(mass(p1)-m1_0):.1e},{abs(mass(p2)-m2_0):.1e})")
    assert sep > 0.5, sep                      # phases forming
    assert p1.min() > -0.02 and p2.min() > -0.02
    assert ps.min() > -0.05                    # simplex ~respected


def test_tch_bdf2_variable_dt(device):
    """Retrofit G3b gate (directive 2026-07-13): variable-coefficient
    BDF2 under an ADAPTIVE-dt sequence (alternating dt0, dt0/2 — r = 2
    and 0.5 every step) on the 4-dof ternary system.  Measured at the
    retrofit: orders ~2 (values in the audit doc); fixed-dt
    trajectories bit-identical pre/post (r = 1 coefficients exact).
    NOTE (recorded): adaptive_march itself cannot drive this stepper —
    its rewind copies hist entries with .copy(), and the ternary hist
    holds (phi1, phi2) TUPLES (pre-existing; deferred in the audit)."""
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh

    T = 0.096
    tree = build_uniform(4, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)

    def run(dt0, var):
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2),
                                  device)
        st = TernaryCHStepper(dm, chi=(1.0, 0.3, 0.3),
                              M=(0.225, 0.0, 0.225),
                              kappa=(2e-4, 2e-4), dt=dt0, order=2)
        st.set_initial(
            lambda x: 0.2 + 0.02 * np.cos(np.pi * x[:, 1]),
            lambda x: 0.2 - 0.02 * np.cos(np.pi * x[:, 1]))
        if var:
            for _ in range(round(T / (1.5 * dt0))):
                st.dt = dt0
                st.step()
                st.dt = dt0 / 2
                st.step()
        else:
            for _ in range(round(T / dt0)):
                st.step()
        assert abs(st.t - T) < 1e-12
        return st.x.copy()

    ref = run(2e-4, False)
    ev = [np.abs(run(d, True) - ref).max() for d in (8e-3, 4e-3, 2e-3)]
    ov = [np.log2(ev[i] / ev[i + 1]) for i in range(2)]
    print(f"TCH var-dt errs {['%.2e' % e for e in ev]} orders "
          f"{['%.2f' % o for o in ov]}")
    assert min(ov) > 1.7, (ev, ov)
