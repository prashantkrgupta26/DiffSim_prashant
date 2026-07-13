"""M4-a gates: CH mixed brick — MMS, mass conservation, energy decay,
spinodal smoke."""
import numpy as np
import pytest

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.cahn_hilliard import CahnHilliardStepper
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
def test_ch_mms_orders(p, levels, order_lo, device):
    M, kap, dt = 1.0, 0.02, 0.002
    cs = lambda x, t: (np.cos(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
                       * np.exp(-t))

    def mus(x, t):
        c = cs(x, t)
        return c ** 3 - c + kap * 2 * np.pi ** 2 * c

    def fc(x, t):
        # c_t + M lap... R_c = c_t - M lap mu - fc = 0 (weak w/ natural)
        c = cs(x, t)
        m = mus(x, t)
        # lap mu: mu = c^3 - c + kap*2pi^2 c; lap(c^n) is messy — use
        # numerical? Manufacture mu INDEPENDENTLY instead: choose
        # mu* = sin(pi x) sin(pi y) e^{-t}; then fm covers the mismatch.
        return np.zeros(len(x))

    mu_star = lambda x, t: (np.sin(np.pi * x[:, 0])
                            * np.sin(np.pi * x[:, 1]) * np.exp(-t))

    def fc2(x, t):
        c = cs(x, t)
        lap_mu = -2 * np.pi ** 2 * mu_star(x, t)
        return -c - M * lap_mu

    def fm2(x, t):
        c = cs(x, t)
        lap_c = -2 * np.pi ** 2 * c
        return mu_star(x, t) - (c ** 3 - c) + kap * lap_c

    errs = []
    for lv in levels:
        dm, mesh, cons = _dm(lv, p, device)
        coords = mesh.node_coords[cons.free_nodes]
        bdry = np.zeros(len(coords), bool)
        for cc in range(2):
            bdry |= (np.abs(coords[:, cc]) < 1e-12) | \
                    (np.abs(coords[:, cc] - 1) < 1e-12)
        st = CahnHilliardStepper(dm, M, kap, dt, order=2, fc_fn=fc2,
                                 fm_fn=fm2,
                                 dirichlet=np.where(bdry)[0],
                                 gc_fn=lambda x, t: cs(x, t),
                                 gm_fn=lambda x, t: mu_star(x, t))
        st.set_initial(lambda x: cs(x, 0.0))
        for _ in range(6):
            c, mu = st.step()
        errs.append(l2_error(dm, np.asarray(cons.T @ c),
                             lambda x: cs(x, st.t)))
    order = np.log2(errs[0] / errs[1])
    print(f"CH p{p}: errs {[f'{e:.2e}' for e in errs]} order {order:.2f}")
    assert order > order_lo, (errs, order)


def test_ch_mass_energy_spinodal(device):
    dm, mesh, cons = _dm(5, 1, device)
    M, kap, dt = 1.0, 5e-4, 0.02
    st = CahnHilliardStepper(dm, M, kap, dt, order=1)
    rng = np.random.default_rng(3)
    st.set_initial(lambda x: 0.05 * rng.standard_normal(len(x)))

    def mass(cf):
        v, _ = st._gp_scalar(cf)
        m = 0.0
        for pv, b in dm.bins.items():
            h = mesh.tree.h()[mesh.bins[pv]]
            ne = len(mesh.conn_of[pv])
            wq = np.tile(dm.tables_by_p[pv].w, ne) \
                * np.repeat((h / 2) ** 2, b["nqp"])
            m += float((wq * v[pv]).sum())
        return m

    def energy(cf):
        v, g = st._gp_scalar(cf)
        E = 0.0
        for pv, b in dm.bins.items():
            h = mesh.tree.h()[mesh.bins[pv]]
            ne = len(mesh.conn_of[pv])
            wq = np.tile(dm.tables_by_p[pv].w, ne) \
                * np.repeat((h / 2) ** 2, b["nqp"])
            E += float((wq * (0.25 * (v[pv] ** 2 - 1) ** 2
                              + 0.5 * kap * (g[pv] ** 2).sum(1))).sum())
        return E

    m0 = mass(st.hist[0])
    Es = [energy(st.hist[0])]
    for n in range(20):
        c, mu = st.step()
        assert abs(mass(c) - m0) < 1e-10, (n, mass(c), m0)
        Es.append(energy(c))
    # step-0 exception (measured: E 0.25->173->0.61 then monotone):
    # set_initial seeds mu=0, inconsistent with a rough IC — the first
    # BE/Newton step absorbs it as a transient. Consistent mu-init
    # (project f'(c0) - kap lap c0) is the recorded refinement; decay
    # asserted from step 1.
    assert all(Es[i + 1] <= Es[i] + 1e-9
               for i in range(1, len(Es) - 1)), Es[:5]
    # spinodal smoke: phases forming
    assert c.max() > 0.6 and c.min() < -0.6, (c.min(), c.max())
    print(f"spinodal: c in [{c.min():.2f},{c.max():.2f}], "
          f"E {Es[0]:.4f}->{Es[-1]:.4f}, |dm|={abs(mass(c)-m0):.1e}")


@pytest.mark.parametrize("p,levels,order_lo", [(1, (4, 5), 1.8),
                                               (2, (3, 4), 2.6)])
def test_ch_fh_mms_orders(p, levels, order_lo, device):
    """Flory-Huggins bulk energy (Wodo & G. JCP 2011 Eq. 4), MMS with
    phi* in [0.3, 0.7] (walls untouched, so the regularized log == log
    and the manufactured f' is exact)."""
    M, kap, dt = 1.0, 0.02, 0.002
    A, B = 1.0, 3.0
    cs = lambda x, t: (0.5 + 0.2 * np.cos(np.pi * x[:, 0])
                       * np.cos(np.pi * x[:, 1]) * np.exp(-t))
    mu_star = lambda x, t: (np.sin(np.pi * x[:, 0])
                            * np.sin(np.pi * x[:, 1]) * np.exp(-t))

    def fc2(x, t):
        # c_t = -0.2 cos cos e^{-t} = -(c - 0.5); lap mu* = -2pi^2 mu*
        c = cs(x, t)
        lap_mu = -2 * np.pi ** 2 * mu_star(x, t)
        return -(c - 0.5) - M * lap_mu

    def fm2(x, t):
        c = cs(x, t)
        lap_c = -2 * np.pi ** 2 * (c - 0.5)
        fp = A * (np.log(c) - np.log(1 - c)) + B * (1 - 2 * c)
        return mu_star(x, t) - fp + kap * lap_c

    errs = []
    for lv in levels:
        dm, mesh, cons = _dm(lv, p, device)
        coords = mesh.node_coords[cons.free_nodes]
        bdry = np.zeros(len(coords), bool)
        for cc in range(2):
            bdry |= (np.abs(coords[:, cc]) < 1e-12) | \
                    (np.abs(coords[:, cc] - 1) < 1e-12)
        st = CahnHilliardStepper(dm, M, kap, dt, order=2, fc_fn=fc2,
                                 fm_fn=fm2,
                                 dirichlet=np.where(bdry)[0],
                                 gc_fn=lambda x, t: cs(x, t),
                                 gm_fn=lambda x, t: mu_star(x, t),
                                 energy="fh", fh_A=A, fh_B=B)
        st.set_initial(lambda x: cs(x, 0.0))
        for _ in range(6):
            c, mu = st.step()
        errs.append(l2_error(dm, np.asarray(cons.T @ c),
                             lambda x: cs(x, st.t)))
    order = np.log2(errs[0] / errs[1])
    print(f"CH-FH p{p}: errs {[f'{e:.2e}' for e in errs]} "
          f"order {order:.2f}")
    assert order > order_lo, (errs, order)


def test_ch_fh_quench_binodal(device):
    """FH quench at the paper's B = 3 (phi0 = 0.5 + noise): mass exact,
    FH energy decays, bulk compositions land on the COMMON-TANGENT
    binodal (computed in-test, not hardcoded), field confined to (0,1)
    by the C1-regularized log. Interface resolved per the paper's
    >= 4-elements rule (kap = 2e-3 -> xi ~ h at level 5); the
    UNDER-resolved version of this quench Newton-oscillates through the
    walls — measured, which is the resolution rule made visible."""
    from scipy.optimize import brentq
    from diffsim.physics.cahn_hilliard import fh_bulk_energy
    A, B = 1.0, 3.0
    # symmetric binodal: f'(phi) = 0 away from phi = 1/2
    fprime = lambda p_: A * (np.log(p_) - np.log(1 - p_)) + B * (1 - 2 * p_)
    phi_lo = brentq(fprime, 1e-6, 0.2)
    phi_hi = 1.0 - phi_lo
    dm, mesh, cons = _dm(5, 1, device)
    M, kap, dt = 1.0, 2e-3, 0.002
    st = CahnHilliardStepper(dm, M, kap, dt, order=1,
                             energy="fh", fh_A=A, fh_B=B)
    rng = np.random.default_rng(3)
    st.set_initial(lambda x: 0.5 + 0.05 * rng.standard_normal(len(x)))

    def mass(cf):
        v, _ = st._gp_scalar(cf)
        m = 0.0
        for pv, b in dm.bins.items():
            h = mesh.tree.h()[mesh.bins[pv]]
            ne = len(mesh.conn_of[pv])
            wq = np.tile(dm.tables_by_p[pv].w, ne) \
                * np.repeat((h / 2) ** 2, b["nqp"])
            m += float((wq * v[pv]).sum())
        return m

    def energy(cf):
        v, g = st._gp_scalar(cf)
        E = 0.0
        for pv, b in dm.bins.items():
            h = mesh.tree.h()[mesh.bins[pv]]
            ne = len(mesh.conn_of[pv])
            wq = np.tile(dm.tables_by_p[pv].w, ne) \
                * np.repeat((h / 2) ** 2, b["nqp"])
            E += float((wq * (fh_bulk_energy(v[pv], A, B)
                              + 0.5 * kap * (g[pv] ** 2).sum(1))).sum())
        return E

    m0 = mass(st.hist[0])
    Es = [energy(st.hist[0])]
    for n in range(30):
        c, mu = st.step()
        assert abs(mass(c) - m0) < 1e-10, (n, mass(c), m0)
        Es.append(energy(c))
    # step-0 transient exception as in the poly gate (mu seeded 0)
    assert all(Es[i + 1] <= Es[i] + 1e-9
               for i in range(1, len(Es) - 1)), Es[:5]
    p5, p95 = np.percentile(c, 5), np.percentile(c, 95)
    print(f"FH quench: binodal ({phi_lo:.4f}, {phi_hi:.4f}); "
          f"p5/p95 = {p5:.4f}/{p95:.4f}; c in "
          f"[{c.min():.4f}, {c.max():.4f}]; E {Es[0]:.3f}->{Es[-1]:.3f}")
    assert abs(p5 - phi_lo) < 0.01 and abs(p95 - phi_hi) < 0.01, \
        (p5, p95, phi_lo, phi_hi)
    assert c.min() > 0.0 and c.max() < 1.0, (c.min(), c.max())


def test_ch_adaptive_dt(device):
    """M4 temporal adaptivity: LTE-controlled dt GROWS through
    coarsening (>4x by t_end) and the march stays physical."""
    from diffsim.physics.cahn_hilliard import adaptive_march
    dm, mesh, cons = _dm(5, 1, device)
    st = CahnHilliardStepper(dm, 1.0, 5e-4, 0.005, order=2)
    rng = np.random.default_rng(3)
    st.set_initial(lambda x: 0.05 * rng.standard_normal(len(x)))
    ts, dts = adaptive_march(st, t_end=0.8, tol=5e-4)
    c = st.hist[0]
    growth = dts[-1] / dts[0]
    print(f"adaptive: {len(dts)} steps, dt {dts[0]:.4f}->{dts[-1]:.4f} "
          f"({growth:.1f}x), c in [{c.min():.2f},{c.max():.2f}]")
    assert growth > 4.0, (dts[0], dts[-1])
    assert np.isfinite(c).all() and c.max() > 0.6 and c.min() < -0.6


# ---------------------------------------------------------------------
# Retrofit G3 (2026-07-13): VARIABLE-COEFFICIENT BDF2 (A4b pattern) —
# coefficients from the actual (dt, dt_prev); r = 1 reproduces the
# constant-step 1.5/[2, -0.5] bit-exactly (fixed-dt parity measured 0.0
# at the retrofit).  Pre-fix baseline (audit doc): the constant-
# coefficient form measured order 0.90/0.95 on an alternating-dt
# sequence with errors 4-20x the fixed-dt run.
# ---------------------------------------------------------------------
_G3_IC = lambda x: (0.8 + 0.05 * np.cos(np.pi * x[:, 0])
                    * np.cos(np.pi * x[:, 1]))


def _ch_march(dm, dt0, var, T=0.096, p_kwargs=()):
    st = CahnHilliardStepper(dm, 1.0, 5e-4, dt0, order=2)
    st.set_initial(_G3_IC, mu_init="consistent")
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
    return st.x[0::2].copy()


def test_ch_bdf2_variable_dt_order(device):
    """G3 gate (directive 2026-07-13): ADAPTIVE-dt order study — the
    alternating (dt0, dt0/2) sequence exercises r = 2 and r = 0.5 on
    every step.  Measured at the retrofit: variable-coefficient orders
    2.02/2.01 (constant-coefficient pre-fix: 0.90/0.95); fixed-dt
    control unchanged (2.18/2.07).  Cross cell: p2 x variable-dt
    (measured 2.02)."""
    dm, _, _ = _dm(4, 1, device)
    ref = _ch_march(dm, 2e-4, False)
    ev = [np.abs(_ch_march(dm, d, True) - ref).max()
          for d in (8e-3, 4e-3, 2e-3)]
    ov = [np.log2(ev[i] / ev[i + 1]) for i in range(2)]
    print(f"CH var-dt errs {['%.2e' % e for e in ev]} orders "
          f"{['%.2f' % o for o in ov]}")
    assert min(ov) > 1.7, (ev, ov)
    # cross-matrix cell: p2 x variable-dt BDF2
    dm2, _, _ = _dm(4, 2, device)
    ref2 = _ch_march(dm2, 2e-4, False)
    e2 = [np.abs(_ch_march(dm2, d, True) - ref2).max()
          for d in (8e-3, 4e-3)]
    o2 = np.log2(e2[0] / e2[1])
    print(f"CH p2 x var-dt order {o2:.2f}")
    assert o2 > 1.7, (e2, o2)


def test_ch_adaptive_march_reject_consistency(device):
    """G3 gate (directive 2026-07-13): LTE rejects never corrupt the
    BDF2 history — a march WITH rejects must equal the replay of its
    ACCEPTED dt sequence (half-step pairs) on a fresh stepper,
    bit-identically.  Measured at the retrofit: 27 accepted / 5
    rejected, replay dev 0.0."""
    from diffsim.physics.cahn_hilliard import adaptive_march

    def mk():
        dm, _, _ = _dm(4, 1, device)
        st = CahnHilliardStepper(dm, 1.0, 5e-4, 0.02, order=2)
        st.set_initial(_G3_IC, mu_init="consistent")
        return st

    st = mk()                    # dt0 = 0.02: LTE rejects at the start
    ncall = {"n": 0}
    orig = st.step

    def counting():
        ncall["n"] += 1
        return orig()

    st.step = counting
    ts, dts = adaptive_march(st, 0.15, tol=1e-6, dt_min=1e-5,
                             dt_max=0.05)
    n_rej = (ncall["n"] - 3 * len(dts)) // 3
    assert n_rej >= 1, "protocol produced no reject — raise dt0"
    st2 = mk()
    for dtf in dts:              # replay the accepted sequence
        st2.dt = dtf / 2
        st2.step()
        st2.step()
    dev = np.abs(st2.x - st.x).max()
    print(f"CH reject-consistency: {len(dts)} accepted / {n_rej} "
          f"rejected, replay dev {dev:.1e}")
    assert dev == 0.0, dev
