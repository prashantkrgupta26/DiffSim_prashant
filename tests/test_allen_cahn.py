"""M4-b gates: AC brick — MMS orders (p1/p2), energy decay, shrinking
circle."""
import numpy as np
import pytest

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.allen_cahn import AllenCahnStepper
from diffsim.physics.poisson import l2_error

pytestmark = pytest.mark.tier2


def _dm(level, p, device):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    return DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2),
                                device), mesh, cons


@pytest.mark.parametrize("p,levels,order_lo", [(1, (4, 5), 1.8),
                                               (2, (3, 4), 2.6)])
def test_ac_mms_orders(p, levels, order_lo, device):
    M, kap, dt = 1.0, 0.02, 0.0025
    cs = lambda x, t: (np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
                       * np.exp(-t))

    def f_fn(x, t):
        c = cs(x, t)
        lap = -2 * np.pi ** 2 * c
        return -c + M * kap * (-lap) / 1.0 * 1.0 + M * (c ** 3 - c)

    errs = []
    for lv in levels:
        dm, mesh, cons = _dm(lv, p, device)
        coords = mesh.node_coords[cons.free_nodes]
        bdry = np.zeros(len(coords), bool)
        for cc in range(2):
            bdry |= (np.abs(coords[:, cc]) < 1e-12) | \
                    (np.abs(coords[:, cc] - 1) < 1e-12)
        st = AllenCahnStepper(dm, M, kap, dt, order=2, f_fn=f_fn,
                              dirichlet=np.where(bdry)[0],
                              g_fn=lambda x, t: cs(x, t))
        st.set_initial(lambda x: cs(x, 0.0))
        for _ in range(8):
            c = st.step()
        errs.append(l2_error(dm, np.asarray(cons.T @ c),
                             lambda x: cs(x, st.t)))
    order = np.log2(errs[0] / errs[1])
    print(f"AC p{p}: errs {[f'{e:.2e}' for e in errs]} order {order:.2f}")
    assert order > order_lo, (errs, order)


def test_ac_energy_decay_and_circle(device):
    # energy decay on a random IC (no source, no Dirichlet)
    dm, mesh, cons = _dm(5, 1, device)
    M, kap, dt = 1.0, 1e-3, 0.01
    st = AllenCahnStepper(dm, M, kap, dt, order=1)
    rng = np.random.default_rng(0)
    st.set_initial(lambda x: 0.2 * rng.standard_normal(len(x)))

    def energy(cf):
        cv, cg = st._gp(cf)
        E = 0.0
        for pv, b in dm.bins.items():
            h = mesh.tree.h()[mesh.bins[pv]]
            ne = len(mesh.conn_of[pv])
            nqp = b["nqp"]
            w = dm.tables_by_p[pv].w
            jac = (h / 2.0) ** 2
            wq = np.tile(w, ne) * np.repeat(jac, nqp)
            fbulk = 0.25 * (cv[pv] ** 2 - 1) ** 2
            gsq = (cg[pv] ** 2).sum(1)
            E += float((wq * (fbulk + 0.5 * kap * gsq)).sum())
        return E

    Es = [energy(st.hist[0])]
    for _ in range(10):
        c = st.step()
        Es.append(energy(c))
    assert all(Es[i + 1] <= Es[i] + 1e-10 for i in range(len(Es) - 1)), Es

    # shrinking circle: dR/dt = -M kap / R  => R^2(t) = R0^2 - 2 M kap t
    dm2, mesh2, cons2 = _dm(6, 1, device)
    st2 = AllenCahnStepper(dm2, 1.0, 2e-4, 0.02, order=2)
    R0 = 0.30
    eps_i = np.sqrt(2 * 2e-4)

    def circ(x):
        r = np.linalg.norm(x - 0.5, axis=1)
        return -np.tanh((r - R0) / (np.sqrt(2) * eps_i))

    st2.set_initial(circ)
    coords2 = mesh2.node_coords[cons2.free_nodes]
    for _ in range(25):
        c = st2.step()
    r_now = np.linalg.norm(coords2 - 0.5, axis=1)
    # measured radius: interpolate the c=0 crossing radially
    o = np.argsort(r_now)
    cz = c[o]
    iz = np.where(np.sign(cz[:-1]) != np.sign(cz[1:]))[0]
    Rm = float(r_now[o][iz[0]])
    Rth = np.sqrt(R0 ** 2 - 2 * 1.0 * 2e-4 * st2.t)
    rel = abs(Rm - Rth) / Rth
    print(f"circle: R={Rm:.4f} vs theory {Rth:.4f} rel={rel:.3f}")
    assert rel < 0.05, (Rm, Rth)


def test_ac_bdf2_variable_dt_order(device):
    """Retrofit G3 gate (directive 2026-07-13): variable-coefficient
    BDF2 under an ADAPTIVE-dt sequence (alternating dt0, dt0/2 — r = 2
    and 0.5 every step).  Measured at the retrofit: orders 2.01/2.01
    (constant-coefficient CH baseline pre-fix: 0.90/0.95); fixed-dt
    trajectories bit-identical (r = 1 coefficients exact)."""
    T = 0.096
    ic = lambda x: (0.8 + 0.05 * np.cos(np.pi * x[:, 0])
                    * np.cos(np.pi * x[:, 1]))

    def run(dt0, var):
        dm, _, _ = _dm(4, 1, device)
        st = AllenCahnStepper(dm, 1.0, 1e-3, dt0, order=2)
        st.set_initial(ic)
        if var:
            for _ in range(round(T / (1.5 * dt0))):
                st.dt = dt0
                st.step()
                st.dt = dt0 / 2
                c = st.step()
        else:
            for _ in range(round(T / dt0)):
                c = st.step()
        assert abs(st.t - T) < 1e-12
        return c.copy()

    ref = run(2e-4, False)
    ev = [np.abs(run(d, True) - ref).max() for d in (8e-3, 4e-3, 2e-3)]
    ov = [np.log2(ev[i] / ev[i + 1]) for i in range(2)]
    print(f"AC var-dt errs {['%.2e' % e for e in ev]} orders "
          f"{['%.2f' % o for o in ov]}")
    assert min(ov) > 1.7, (ev, ov)
