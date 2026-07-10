"""M5 S0/S1 gates: (M, K)-generic multi-CH x multi-AC brick
(diffsim.physics.multiphase) — amorphous regression, coupled MMS,
single-crystal sanity + impingement, 2310.11844 replication anchors,
grain identification.  Tolerances measured-then-locked (>= 2x headroom;
measured numbers in the assertion prints)."""
import numpy as np
import pytest

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.multiphase import (MultiPhaseStepper, grain_labels,
                                        np_potentials)
from diffsim.physics.ternary_ch import TernaryCHStepper

pytestmark = pytest.mark.tier3


def _dm(level, device, p=1):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    return (DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2),
                                 device), mesh, cons)


# ---------------------------------------------------------------------
# S0 — AMORPHOUS REGRESSION: (M=2, K=0) == TernaryCHStepper
# ---------------------------------------------------------------------
def _s0_chi_aa(chi):
    chi_aa = np.zeros((3, 3))
    chi_aa[0, 1] = chi_aa[1, 0] = chi[0]
    chi_aa[0, 2] = chi_aa[2, 0] = chi[1]
    chi_aa[1, 2] = chi_aa[2, 1] = chi[2]
    return chi_aa


def _s0_ics(nfree):
    rng = np.random.default_rng(4)
    ic1 = 0.35 + 0.02 * rng.standard_normal(nfree)
    ic2 = 0.35 + 0.02 * rng.standard_normal(nfree)
    return ic1, ic2


def test_s0_amorphous_regression_fixed_dt(device):
    """(M=2, K=0) discretizes EXACTLY the ternary brick's equations
    (chi_eff == chi_aa, no crystal terms).  Fixed-dt parity vs
    TernaryCHStepper on the spinodal config over the CONVERGED Newton
    segment.  MEASURED FINDING (dbg 2026-07-10): at dt = 5e-3 the
    reference stepper's OWN Newton stops converging at step 2 (the
    quench onset: 60 iterations, |dx| wandering 0.5-40; TernaryCHStepper
    commits iterate newton_max regardless) — the fixed-dt trajectory
    beyond step 1 is a non-converged-Newton artifact, the SAME
    ill-posedness recorded in test_wodo_film's spinodal parity gate.
    Steps 0-1 converge (5 and 7 iterations) and are the well-posed
    fixed-dt comparison; measured parity 2.1e-15 / 8.3e-15.  The
    6-step MARCH parity (ladder-rejected attempts, house template) is
    the companion gate below."""
    dm, mesh, cons = _dm(5, device)
    chi = (6.0, 0.8, 0.8)
    st_ref = TernaryCHStepper(dm, chi=chi, M=(1.0, -0.2, 1.0),
                              kappa=(8e-4, 8e-4), order=1, dt=0.005)
    ic1, ic2 = _s0_ics(len(st_ref.free_coords))
    st_ref.set_initial(lambda x: ic1, lambda x: ic2)
    st = MultiPhaseStepper(dm, M=2, K=0, chi_aa=_s0_chi_aa(chi),
                           onsager=[[1.0, -0.2], [-0.2, 1.0]],
                           kappa=(8e-4, 8e-4), dt=0.005, guards=False)
    st.set_initial([lambda x: ic1, lambda x: ic2])
    errs = []
    for n in range(2):
        p1r, p2r = st_ref.step()
        st.step()
        errs.append(max(np.abs(st.phi(0) - p1r).max(),
                        np.abs(st.phi(1) - p2r).max()))
    print(f"S0 fixed-dt parity (converged steps): "
          f"{[f'{e:.2e}' for e in errs]} max {max(errs):.3e}")
    assert max(errs) < 1e-13, errs      # measured 8.3e-15; 2x-class lock


def test_s0_amorphous_regression_march(device):
    """S0 companion: 6 ACCEPTED steps through the quench onset via a
    JOINT Appendix-A ladder vs WodoFilmStepper (k_e = 0, N = (1,1,1) —
    the gate-verified TernaryCHStepper equivalent with the reject
    machinery).  Independent ladders are ILL-POSED here (measured
    2026-07-10): the onset attempts are Newton-chaotic (|dx| wandering
    0.5-40 over 50 iterations at every fixed dt in [1e-3, 5e-3] from
    this IC), and a chaotic attempt's converge/not-converge fate is an
    FP knife edge between two differently-ordered assemblies (measured:
    dt = 6.1e-4 attempt converges at 37 iterations for multiphase, hits
    50 for wodo — same root class as the cuDSS coin flip documented in
    test_wodo_film).  The joint ladder commits only MUTUALLY converged
    attempts (well-posed: every committed state is a converged root of
    each discretization), rejects both otherwise, and requires any fate
    MISMATCH to occur only on chaotic attempts (min iters >= 30).
    Measured: 6 accepted steps, parity <= 2.9e-14, one chaotic-fate
    mismatch (50, 37)."""
    from diffsim.physics.wodo_film import WodoFilmStepper
    dm, mesh, cons = _dm(5, device)
    chi = (6.0, 0.8, 0.8)
    st_ref = WodoFilmStepper(dm, chi=chi, N=(1.0, 1.0, 1.0),
                             M=(1.0, -0.2, 1.0), kappa=(8e-4, 8e-4),
                             k_e=0.0, dt=5e-3)
    ic1, ic2 = _s0_ics(len(st_ref.free_coords))
    st_ref.set_initial(lambda x: ic1, lambda x: ic2)
    st = MultiPhaseStepper(dm, M=2, K=0, chi_aa=_s0_chi_aa(chi),
                           onsager=[[1.0, -0.2], [-0.2, 1.0]],
                           kappa=(8e-4, 8e-4), dt=5e-3, guards=False)
    st.set_initial([lambda x: ic1, lambda x: ic2])
    dt, acc, mism, errs, lad = 5e-3, 0, [], [], []
    for att in range(40):
        if acc >= 6:
            break
        xr, itr, okr = st_ref._attempt(dt, 0.0)
        xm, itm, okm = st._attempt(dt)
        if okr != okm:
            mism.append((dt, itr, itm))
            assert min(itr, itm) >= 30, \
                ("fate mismatch on a NON-chaotic attempt", dt, itr, itm)
        if okr and okm:
            st_ref.x = xr
            st_ref.hist = [(xr[0::4].copy(), xr[2::4].copy()),
                           st_ref.hist[0]]
            st_ref.t += dt
            st.x = xm
            st.hist = xm.copy()
            st.t += dt
            acc += 1
            errs.append(np.abs(xm - xr).max())
            lad.append((dt, itr, itm))
            if max(itr, itm) < 20:
                dt *= 1.25
        else:
            dt *= 0.25
    print(f"S0 march parity: {acc} accepted steps, ladder "
          f"{[(f'{d:.2e}', a, b) for d, a, b in lad]}, "
          f"fate mismatches {mism}; per-step max|dx| "
          f"{[f'{e:.2e}' for e in errs]}")
    assert acc == 6, acc
    assert max(errs) < 1e-9, errs       # measured 2.9e-14
    assert len(mism) <= 1, mism


# ---------------------------------------------------------------------
# S1a — MMS ORDERS: (M=1, K=1) coupled binary + crystal system
# ---------------------------------------------------------------------
@pytest.mark.parametrize("bulk", ["p1", "r14"])
def test_s1a_mms_orders(bulk, device):
    """Manufactured phi*, psi*, theta* (smooth, nonzero gradient),
    independent-mu* trick (cahn_hilliard MMS pattern); sources numpy
    pointwise from the analytic strong forms, incl. the KG-regularized
    theta flux divergence.  p1 elements: expect ~2 for phi and psi.
    theta rides the C1-class regularized |grad theta| coefficient —
    measured order asserted at what is robust (see print)."""
    Mons, kap, dt = 1.0, 0.02, 1e-3
    Ninv = np.array([1.0, 1.0])
    dsig, drive_v = 1.5, -0.8          # drive frozen via Tm=1, T=1-...
    # choose T, Tm so that stepper._drive reproduces drive_v exactly:
    # p1: dh (1 - T/Tm); r14: dh (T/Tm - 1). Use Tm=1, dh=+-1.6, T=0.5.
    Tm, Tq = 1.0, 0.5
    dh = drive_v / ((1 - Tq / Tm) if bulk == "p1" else (Tq / Tm - 1))
    eps2, Lpsi = 5e-3, 1.0
    alpha, beta, Lth = 0.4, 0.1, 1.0
    kgd, pfl = 1e-2, 1e-3
    chi_aa = np.array([[0.0, 1.2], [1.2, 0.0]])
    chi_ac = np.array([[0.0, 0.7], [0.4, 0.0]])   # chi_ac[i,j]: am-i/cr-j
    chi_ca = chi_ac.T.copy()
    chi_cc = np.array([[0.0, 0.9], [0.9, 0.0]])
    pars = dict(chi_aa=chi_aa, chi_ac=chi_ac, chi_ca=chi_ca,
                chi_cc=chi_cc, Ninv=Ninv, dsig=[dsig], drive=[drive_v],
                breg=0.0, bulk=bulk)

    pi = np.pi
    E = lambda t: np.exp(-t)
    phis = lambda x, t: 0.5 + 0.1 * np.cos(pi * x[:, 0]) \
        * np.cos(pi * x[:, 1]) * E(t)
    mus = lambda x, t: np.sin(pi * x[:, 0]) * np.sin(pi * x[:, 1]) * E(t)
    psis = lambda x, t: 0.5 + 0.25 * np.cos(pi * x[:, 0]) \
        * np.cos(2 * pi * x[:, 1]) * E(t)
    ths = lambda x, t: 0.5 + 0.2 * np.sin(pi * x[:, 0]) \
        * np.cos(pi * x[:, 1]) * E(t)

    def th_grad(x, t):
        gx = 0.2 * pi * np.cos(pi * x[:, 0]) * np.cos(pi * x[:, 1]) * E(t)
        gy = -0.2 * pi * np.sin(pi * x[:, 0]) * np.sin(pi * x[:, 1]) \
            * E(t)
        return gx, gy

    def psi_grad(x, t):
        gx = -0.25 * pi * np.sin(pi * x[:, 0]) * np.cos(2 * pi * x[:, 1]) \
            * E(t)
        gy = -0.5 * pi * np.cos(pi * x[:, 0]) * np.sin(2 * pi * x[:, 1]) \
            * E(t)
        return gx, gy

    pori = lambda s: s ** 2 * (3 - 2 * s)
    porip = lambda s: 6 * s * (1 - s)

    def f_phi(x, t):
        # phi_t + Mons lap mu* ... R_phi: phi_t - div(M grad mu) = f
        return -(phis(x, t) - 0.5) - Mons * (-2 * pi ** 2 * mus(x, t))

    def f_mu(x, t):
        mu_b, _ = np_potentials([phis(x, t)], [psis(x, t)], pars)
        lap_phi = -2 * pi ** 2 * (phis(x, t) - 0.5)
        return mus(x, t) - mu_b[0] + kap * lap_phi

    def f_psi(x, t):
        _, dfs = np_potentials([phis(x, t)], [psis(x, t)], pars)
        gx, gy = th_grad(x, t)
        g2 = gx ** 2 + gy ** 2
        S = np.sqrt(g2 + kgd ** 2)
        Eori = 0.5 * alpha * S + 0.5 * beta * g2
        lap_psi = -5 * pi ** 2 * (psis(x, t) - 0.5)
        return (-(psis(x, t) - 0.5)
                + Lpsi * (dfs[0] + porip(psis(x, t)) * Eori)
                - Lpsi * eps2 * lap_psi)

    def f_th(x, t):
        # (p(psi)+pf) th_t - div[(p+pf) Lth ((a/2)/S + b) grad th] = f
        s = psis(x, t)
        th = ths(x, t)
        gx, gy = th_grad(x, t)
        g2 = gx ** 2 + gy ** 2
        S = np.sqrt(g2 + kgd ** 2)
        c = (pori(s) + pfl) * Lth * (0.5 * alpha / S + beta)
        # grad c = Lth [ p'(s) grad s ((a/2)/S + b)
        #               + (p+pf)(a/2)(-1/S^3)(H g) ]
        sx, sy = psi_grad(x, t)
        Hxx = -pi ** 2 * (th - 0.5)
        Hyy = -pi ** 2 * (th - 0.5)
        Hxy = -0.2 * pi ** 2 * np.cos(pi * x[:, 0]) \
            * np.sin(pi * x[:, 1]) * E(t)
        Hgx = Hxx * gx + Hxy * gy
        Hgy = Hxy * gx + Hyy * gy
        cf = 0.5 * alpha / S + beta
        cx = Lth * (porip(s) * sx * cf
                    - (pori(s) + pfl) * 0.5 * alpha * Hgx / S ** 3)
        cy = Lth * (porip(s) * sy * cf
                    - (pori(s) + pfl) * 0.5 * alpha * Hgy / S ** 3)
        lap_th = -2 * pi ** 2 * (th - 0.5)
        return (pori(s) + pfl) * (-(th - 0.5)) \
            - (cx * gx + cy * gy + c * lap_th)

    errs = {f: [] for f in ("phi", "psi", "theta")}
    for lv in (4, 5):
        dm, mesh, cons = _dm(lv, device)
        coords = mesh.node_coords[cons.free_nodes]
        bdry = np.zeros(len(coords), bool)
        for cc in range(2):
            bdry |= (np.abs(coords[:, cc]) < 1e-12) | \
                    (np.abs(coords[:, cc] - 1) < 1e-12)
        st = MultiPhaseStepper(
            dm, M=1, K=1, chi_aa=chi_aa, chi_ac=chi_ac, chi_ca=chi_ca,
            chi_cc=chi_cc, N=1.0 / Ninv, onsager=[[Mons]], kappa=[kap],
            dsig=[dsig], dh=[dh], Tm=[Tm], eps2=[eps2], L_psi=[Lpsi],
            alpha_th=[alpha], beta_th=[beta], L_th=[Lth], T=Tq,
            dt=dt, bulk=bulk, kg_delta=kgd, p_floor=pfl,
            newton_tol=1e-11, newton_max=30,
            dirichlet=np.where(bdry)[0],
            g_fns=[phis, mus, psis, ths],
            src_fns=[f_phi, f_mu, f_psi, f_th])
        st.set_initial([lambda x: phis(x, 0.0)],
                       [lambda x: psis(x, 0.0)],
                       [lambda x: ths(x, 0.0)])
        st.x[1::st.ndof] = mus(st.free_coords, 0.0)
        for _ in range(4):
            st.step()
        from diffsim.physics.poisson import l2_error
        errs["phi"].append(l2_error(dm, np.asarray(cons.T @ st.phi(0)),
                                    lambda x: phis(x, st.t)))
        errs["psi"].append(l2_error(dm, np.asarray(cons.T @ st.psi(0)),
                                    lambda x: psis(x, st.t)))
        errs["theta"].append(l2_error(dm, np.asarray(cons.T @ st.theta(0)),
                                      lambda x: ths(x, st.t)))
    orders = {f: np.log2(errs[f][0] / errs[f][1]) for f in errs}
    print(f"S1a MMS [{bulk}]: " + "; ".join(
        f"{f} errs {[f'{e:.2e}' for e in errs[f]]} "
        f"order {orders[f]:.2f}" for f in errs))
    assert orders["phi"] > 1.8, (errs["phi"], orders["phi"])
    assert orders["psi"] > 1.8, (errs["psi"], orders["psi"])
    # theta: the KG-regularized |grad theta| coefficient is only ~C1 in
    # the fields; measured orders locked at > 1.5 (see milestone report)
    assert orders["theta"] > 1.5, (errs["theta"], orders["theta"])
