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
            # KG-Picard frozen 1/|g|_d gives a LINEAR Newton tail on the
            # theta rows (measured contraction 0.52/iterate at this
            # config); 1e-10 in <= 80 iterations, far below the spatial
            # error floor
            newton_tol=1e-10, newton_max=80,
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
    # theta subtlety, resolved: with the KG smoothing (kg_delta = 1e-2)
    # the |grad theta| coefficient is C-infinity and theta measures a
    # CLEAN order 2.00 in both bulk modes (p1: 6.2e-4 -> 1.6e-4); the
    # nonsmoothness worry only materializes as kg_delta -> 0.
    assert orders["theta"] > 1.8, (errs["theta"], orders["theta"])


# ---------------------------------------------------------------------
# S1b — SINGLE-CRYSTAL SANITY (r14 family = the S1c replication mode)
# ---------------------------------------------------------------------
def _s1b_stepper(dm, T, eps2, alpha=0.0, beta=0.0, L_th=5.0,
                 kg_delta=1e-2, L_psi=5.0, tol=1e-7, newton_max=60):
    """PCBM-class binary (M=1, K=1), r14 bulk (2310.11844 Table-1
    energetics nondimensionalized on a 64 nm box; W-bar 2.6355,
    L-bar 1.3072), constant small Onsager mobility (growth-limited
    regime), no noise."""
    chi_aa = np.array([[0.0, 0.7248], [0.7248, 0.0]])
    chi_ca = np.array([[0.0, 1.0836], [0.0, 0.0]])
    return MultiPhaseStepper(
        dm, M=1, K=1, chi_aa=chi_aa, chi_ac=chi_ca.T.copy(),
        chi_ca=chi_ca, N=[5.0298, 1.0], onsager=[[0.1]], kappa=[2e-4],
        dsig=[2.6355], dh=[1.3072], Tm=[558.0], eps2=[eps2],
        L_psi=[L_psi], alpha_th=[alpha], beta_th=[beta], L_th=[L_th],
        T=T, dt=2e-3, bulk="r14", kg_delta=kg_delta, p_floor=1e-6,
        newton_tol=tol, newton_max=newton_max)


def _disc(c, r0, w):
    return lambda x: 0.5 * (1.0 - np.tanh(
        (np.sqrt((x[:, 0] - c[0]) ** 2
                 + (x[:, 1] - c[1]) ** 2) - r0) / w))


def test_s1b_growth_melt(device):
    """Seeded psi disc in a uniform undercooled blend (phi0 = 0.6,
    no noise): crystalline area GROWS for T < Tm and MELTS OUT for
    T > Tm.  Measured (r14, PCBM energetics): T=333 area
    0.0693 -> 0.1120 (1.62x, with solvent expulsion phi -> 0.93 in the
    crystal); T=700 area -> 0.0002, psi_max 0.50.  Locked with
    >= 2x-class headroom."""
    dm, mesh, cons = _dm(6, device)
    phi0 = 0.6
    res = {}
    for T in (333.0, 700.0):
        st = _s1b_stepper(dm, T, 1e-3)
        st.set_initial([lambda x: np.full(len(x), phi0)],
                       [_disc((0.5, 0.5), 0.15, 0.02)],
                       [lambda x: np.zeros(len(x))])
        a0 = float(np.mean(st.psi(0) > 0.5))
        reason = st.march(t_end=0.5, dt_max=0.02, max_steps=300,
                          dt_min=1e-7)
        res[T] = (a0, float(np.mean(st.psi(0) > 0.5)), reason)
    print(f"S1b grow/melt: T=333 area {res[333.0][0]:.4f} -> "
          f"{res[333.0][1]:.4f}; T=700 area {res[700.0][0]:.4f} -> "
          f"{res[700.0][1]:.4f}")
    a0, a1, _ = res[333.0]
    assert a1 > a0 * 1.3, res[333.0]        # measured 1.62x
    b0, b1, _ = res[700.0]
    assert b1 < b0 * 0.1, res[700.0]        # measured 0.003x


def test_s1b_growth_melt_p1(device):
    """The RATIFIED p1 bulk form, same sanity: dh < 0 (crystallization
    enthalpy; the sign ruling in the module docstring), chi-neutral
    (chi_ca = chi_aa isolates the psi-bulk driving), N = 1.  Measured:
    T = 0.5 Tm area 0.0693 -> 0.0949 (+37%); T = 1.5 Tm -> 0.0324
    (-53%, psi_max 0.88 melting)."""
    dm, mesh, cons = _dm(6, device)
    chi_aa = np.array([[0.0, 0.7248], [0.7248, 0.0]])
    res = {}
    for T in (0.5, 1.5):
        st = MultiPhaseStepper(
            dm, M=1, K=1, chi_aa=chi_aa, chi_ac=chi_aa.copy(),
            chi_ca=chi_aa.copy(), N=[1.0, 1.0], onsager=[[0.1]],
            kappa=[2e-4], dsig=[1.0], dh=[-1.0], Tm=[1.0], eps2=[1e-3],
            L_psi=[5.0], alpha_th=[0.0], beta_th=[0.0], L_th=[5.0],
            T=T, dt=2e-3, bulk="p1", newton_tol=1e-7, newton_max=40)
        st.set_initial([lambda x: np.full(len(x), 0.6)],
                       [_disc((0.5, 0.5), 0.15, 0.02)],
                       [lambda x: np.zeros(len(x))])
        a0 = float(np.mean(st.psi(0) > 0.5))
        st.march(t_end=0.4, dt_max=0.02, max_steps=300, dt_min=1e-7)
        res[T] = (a0, float(np.mean(st.psi(0) > 0.5)))
    print(f"S1b p1 grow/melt: T=0.5Tm {res[0.5][0]:.4f} -> "
          f"{res[0.5][1]:.4f}; T=1.5Tm {res[1.5][0]:.4f} -> "
          f"{res[1.5][1]:.4f}")
    assert res[0.5][1] > res[0.5][0] * 1.15, res[0.5]   # measured 1.37x
    assert res[1.5][1] < res[1.5][0] * 0.7, res[1.5]    # measured 0.47x


def test_s1b_interface_width_scaling(device):
    """Interface width vs the eps/sqrt(W)-class scaling: at T = Tm
    (zero driving) the relaxed profile width must scale as sqrt(eps2)
    — quadrupling eps2 doubles it.  EXTRACTION NOTE (measured): the
    crystal-interior psi plateau sits at ~0.89, not 1 (T = Tm, chi_ca
    feedback), so the crossings are PLATEAU-NORMALIZED 25%/75% levels
    (absolute 10/90 levels alias the plateau and return disc-scale
    garbage — measured 0.26 for both eps2)."""
    dm, mesh, cons = _dm(6, device)
    phi0 = 0.6
    widths = {}
    for eps2 in (1e-3, 4e-3):
        st = _s1b_stepper(dm, 558.0, eps2)
        st.set_initial([lambda x: np.full(len(x), phi0)],
                       [_disc((0.5, 0.5), 0.25, 0.02)],
                       [lambda x: np.zeros(len(x))])
        st.march(t_end=0.1, dt_max=0.01, max_steps=200, dt_min=1e-7)
        coords = st.free_coords
        row = np.abs(coords[:, 1] - 0.5) < 1e-9
        xs, ps = coords[row, 0], st.psi(0)[row]
        o = np.argsort(xs)
        xs, ps = xs[o], ps[o]
        right = xs > 0.55
        xr, pr = xs[right], ps[right]
        plateau = pr[xr < 0.62].mean()
        x25 = np.interp(0.25 * plateau, pr[::-1], xr[::-1])
        x75 = np.interp(0.75 * plateau, pr[::-1], xr[::-1])
        widths[eps2] = x25 - x75
    ratio = widths[4e-3] / widths[1e-3]
    print(f"S1b width (25-75, plateau-normalized): eps2 1e-3 -> "
          f"{widths[1e-3]:.4f}, 4e-3 -> {widths[4e-3]:.4f}, ratio "
          f"{ratio:.2f} (sqrt-scaling expects 2.0); theory-class "
          f"delta = sqrt(eps2/(2 W phi)) = "
          f"{np.sqrt(1e-3 / (2 * 2.6355 * phi0)):.4f}")
    assert 1.5 < ratio < 2.5, (widths, ratio)


def test_s1b_impingement_and_grain_id(device):
    """Two near-contact seeds with DIFFERENT theta plateaus (0 and 1,
    smooth tanh transition): with the KWC term the crystals STOP at the
    boundary (measured: psi dip 0.0011 persists — the seam stays
    amorphous-class; theta-watershed finds exactly 2 grains, 210/203
    nodes); without it they merge (measured: dip heals to 0.916, one
    psi-connected component).  NUMERICS (measured ladder study): the
    paper-magnitude alpha = 3.334 with sharp theta collapses the
    Appendix-A dt to 3e-5 (GB force ~ alpha |grad theta| ~ 10^2 x bulk
    driving), and kg_delta <= 0.05 keeps dt pinned ~7e-4 via rejects;
    kg_delta = 0.2 + beta = 0.1 tame the singular coefficient with the
    GB energy (~ p alpha/2 |Delta theta|, delta-independent to leading
    order) intact — 0 rejects, ~40 steps.  grow_iters = 45 covers the
    KG-Picard linear Newton tail."""
    dm, mesh, cons = _dm(6, device)
    phi0 = 0.6
    dips, grains = {}, {}
    for alpha, beta in ((0.5, 0.1), (0.0, 0.0)):
        st = _s1b_stepper(dm, 333.0, 1e-3, alpha=alpha, beta=beta,
                          L_th=0.2, kg_delta=0.2, tol=1e-6,
                          newton_max=80)
        st.dt = 2e-4
        # seeds nearly touching (gap ~ interface width): the GB forms
        # within the horizon without a long growth phase
        st.set_initial(
            [lambda x: np.full(len(x), phi0)],
            [lambda x: np.maximum(_disc((0.365, 0.5), 0.13, 0.02)(x),
                                  _disc((0.635, 0.5), 0.13, 0.02)(x))],
            [lambda x: 0.5 * (1.0 + np.tanh((x[:, 0] - 0.5) / 0.05))])
        st.march(t_end=0.12, dt_max=0.02, max_steps=250, dt_min=1e-8,
                 grow_iters=45)
        coords = st.free_coords
        seg = (np.abs(coords[:, 1] - 0.5) < 1e-9) \
            & (np.abs(coords[:, 0] - 0.5) < 0.1)
        dips[alpha] = float(st.psi(0)[seg].min())
        labels, sizes = grain_labels(
            dm.mesh.node_coords, np.asarray(cons.T @ st.psi(0)),
            np.asarray(cons.T @ st.theta(0)), psi_th=0.5,
            theta_tol=0.3)
        grains[alpha] = sizes
    print(f"S1b impinge: theta-ON dip {dips[0.5]:.4f}, grains "
          f"{grains[0.5][:4]}; theta-OFF dip {dips[0.0]:.4f}, "
          f"psi-connected components {len(grains[0.0])}")
    # KWC ON: boundary persists, watershed finds exactly 2 crystals
    assert len(grains[0.5]) == 2, grains[0.5]
    assert dips[0.5] < 0.3, dips                # measured 0.0011
    # OFF: merged (seam heals to one crystal)
    assert dips[0.0] > 0.7, dips                # measured 0.916
    assert len(grains[0.0]) == 1, grains[0.0]
    assert dips[0.0] - dips[0.5] > 0.5, dips    # measured 0.915


def test_grain_labels_synthetic():
    """Pure-numpy watershed unit test: two psi blobs TOUCHING through a
    crystalline bridge but with distinct theta plateaus -> the theta
    tolerance splits them into exactly 2 grains; with theta_tol = inf
    the same field is 1 connected component; sizes sorted descending."""
    n = 33
    xs = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)
    r1 = np.sqrt((X - 0.35) ** 2 + (Y - 0.5) ** 2)
    r2 = np.sqrt((X - 0.68) ** 2 + (Y - 0.5) ** 2)
    psi = np.maximum(1.0 * (r1 < 0.20), 1.0 * (r2 < 0.16)).ravel()
    theta = np.where(X.ravel() < 0.52, 0.0, 1.0)
    labels, sizes = grain_labels(coords, psi, theta, psi_th=0.5,
                                 theta_tol=0.3)
    labels1, sizes1 = grain_labels(coords, psi, theta, psi_th=0.5,
                                   theta_tol=np.inf)
    print(f"grain_labels synthetic: split {len(sizes)} grains "
          f"{sizes}, merged {len(sizes1)} component {sizes1}")
    assert len(sizes) == 2 and sizes[0] >= sizes[1], sizes
    assert len(sizes1) == 1, sizes1
    assert labels[psi <= 0.5].max() == -1
    assert sizes.sum() == (psi > 0.5).sum() == sizes1.sum()


# ---------------------------------------------------------------------
# S1c — 2310.11844 REFERENCE-CASE REPLICATION (in-suite anchors)
# ---------------------------------------------------------------------
def _s1c_stepper(dm, cons_mesh, phi0, level, seed=11):
    """The paper's Table-1 PCBM/oDCB reference case, nondimensionalized
    on a (2^level) nm box at their Delta-x = 1 nm (length unit l0 =
    box edge, time unit 1 s, energy density RT/v0): W-bar 2.6355,
    L-bar 1.3072, drive L(T/Tm - 1) = -0.5271 at T = 333 K, chi_aa
    0.7248 + chi_ca 1.0836 psi^2 (their Eq. 7), fast-mode Onsager
    (Eqs. 15-16, log-mean D interpolation), M(phi) = M0 = 0.1/s
    constant, FDT noise std sqrt((2 v0/Na) N1 M0) with 2-D cell volume
    dx^2 * dx (depth = dx ASSUMPTION, recorded — the paper states no
    2-D noise normalization).  Thermal nucleation ONLY (uniform IC,
    no seeds, no IC noise — their Sec 3.2)."""
    R, Na = 8.314, 6.022e23
    rho, v0, N1, N2 = 1600.0, 1.131e-4, 5.0298, 1.0
    Tq, Tm = 333.0, 558.0
    u0 = R * Tq / v0
    l0 = float(1 << level) * 1e-9
    Dsc = 1.0 / l0 ** 2
    Lpsi = N1 * 0.1
    A_fdt = (2.0 * v0 / Na) * Lpsi
    noise_psi = np.sqrt(A_fdt / (l0 ** 2 * 1e-9 * 2.0 * Lpsi))
    chi_aa = np.array([[0.0, 0.7248], [0.7248, 0.0]])
    chi_ca = np.array([[0.0, 1.0836], [0.0, 0.0]])
    st = MultiPhaseStepper(
        dm, M=1, K=1, chi_aa=chi_aa, chi_ac=chi_ca.T.copy(),
        chi_ca=chi_ca, N=[N1, N2], mob="fastmode",
        D_lo=[5e-10 * Dsc, 2e-9 * Dsc], D_hi=[1e-13 * Dsc, 1e-12 * Dsc],
        kappa=[2e-10 / (u0 * l0 ** 2)],
        dsig=[40322.5806 * rho / u0], dh=[20000.0 * rho / u0],
        Tm=[Tm], eps2=[1e-10 / (u0 * l0 ** 2)], L_psi=[Lpsi],
        alpha_th=[8.1621e7 / u0], beta_th=[0.0], L_th=[Lpsi], T=Tq,
        dt=1e-3, bulk="r14", kg_delta=1e-2, p_floor=1e-6,
        newton_tol=1e-8, newton_max=50, linsolver="splu",
        noise_psi=noise_psi, noise_seed=seed, clip_psi=False)
    st.set_initial([lambda x: np.full(len(x), phi0)],
                   [lambda x: np.zeros(len(x))],
                   [lambda x: np.zeros(len(x))])
    return st


def test_s1c_fdt_statistics(device):
    """QUANTITATIVE noise gate: the stationary pre-nucleation psi
    fluctuation variance vs the Gaussian equipartition mode sum
    Var = (kT/L^2) SUM_q 1/(f''(0) + eps2 q^2) on the periodic grid
    (kT = noise_psi^2 by the wJ-normalized FDT construction; f''(0)
    includes the chi_ca psi^2-coupling curvature 2 phi(1-phi) chi_ca).
    Measured at the paper's parameters: ratio 0.822 at L5/32 nm
    (0.848 at L6/64 nm) — the deficit is the BDF1 high-q damping at
    dt * rate ~ O(1).  Locked ratio in [0.65, 1.10]."""
    level = 5
    tree = build_uniform(level, dim=2, periodic=(True, True))
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2),
                              device)
    st = _s1c_stepper(dm, (cons, mesh), 0.3, level, seed=5)
    st.dt = 0.25
    vs = []
    st.march(t_end=30.0, dt_max=0.25,
             callback=lambda s, dt, it:
             vs.append(np.var(s.psi(0))) if s.t > 8.0 else None)
    var_meas = float(np.mean(vs[::4]))
    phi0 = 0.3
    Wb, Lb, Tq, Tm = 2.6355, 1.3072, 333.0, 558.0
    drive = Lb * (Tq / Tm - 1.0)
    fpp = phi0 * (2 * Wb + 6 * drive) + 2 * phi0 * (1 - phi0) * 1.0836
    eps2 = st.eps2[0]
    kT = st.noise_psi ** 2
    ng = 1 << level
    k1 = 2 * np.pi * np.fft.fftfreq(ng, d=1.0 / ng)
    KX, KY = np.meshgrid(k1, k1)
    var_pred = kT * np.mean(1.0 / (fpp + eps2 * (KX ** 2 + KY ** 2))) \
        * ng ** 2
    ratio = var_meas / var_pred
    print(f"S1c FDT: measured psi-var {var_meas:.3e} (std "
          f"{np.sqrt(var_meas):.4f}) vs equipartition {var_pred:.3e} "
          f"(std {np.sqrt(var_pred):.4f}); ratio {ratio:.3f}")
    assert 0.65 < ratio < 1.10, (var_meas, var_pred, ratio)


def test_s1c_replication_anchors(device):
    """2310.11844 reference case on a 64 nm periodic box (their 1 nm
    resolution; domain reduced from 512 nm — DEVIATION recorded: fewer
    simultaneous nuclei, intensive kinetics unaffected to leading
    order).  Text-stated anchors exercised:
    (a) phi0 = 0.8: THERMAL nucleation from the FDT noise alone, then
        sigmoidal crystallinity (their Sec 4.1) — measured (seed 11):
        first nucleus t = 8.0 s, tau50 = 27 s, plateau X = 0.939;
    (b) phi0 = 0.3: NO crystallization on the same horizon (their
        'no crystallization below phi0 = 0.4' / one-step pathway
        kinetically impeded, Secs 4.1 + Fig 2-b);
    (c) orientation field inert in (a)/(b): the paper's theta
        ORIENTATION-ASSIGNMENT at nucleation is unspecified (Eq. 5 is a
        Kronecker penalty; kinetics defer to Ronsin-Harting 2022), so
        kinetics run orientation-uniform — the impingement mechanism is
        gated separately (S1b).  Full-horizon curves + the 128 nm
        campaign live in the milestone report."""
    level = 6
    tree = build_uniform(level, dim=2, periodic=(True, True))
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2),
                              device)
    full = lambda st, v: np.asarray(cons.T @ v)
    horizon = 45.0
    curves = {}
    for phi0 in (0.8, 0.3):
        st = _s1c_stepper(dm, (cons, mesh), phi0, level)
        rec = []

        def cb(s, dt, iters):
            phi = full(s, s.phi(0))
            psi = full(s, s.psi(0))
            X = float((phi * psi).mean() / phi.mean())
            _, sizes = grain_labels(mesh.node_coords, psi,
                                    np.zeros(len(psi)), psi_th=0.5,
                                    theta_tol=np.inf, periodic=True)
            rec.append((s.t, X, int((sizes >= 4).sum())))
        reason = st.march(t_end=horizon, dt_max=1.0, callback=cb)
        curves[phi0] = np.array(rec)
        assert reason == "t_end", reason
    a = curves[0.8]
    onset = a[np.argmax(a[:, 2] >= 1), 0] if (a[:, 2] >= 1).any() \
        else np.inf
    plateau_X = a[-1, 1]
    i50 = np.argmax(a[:, 1] >= 0.5)
    t50 = a[i50, 0] if (a[:, 1] >= 0.5).any() else np.inf
    b = curves[0.3]
    print(f"S1c anchors (64 nm, seed 11): phi0=0.8 onset {onset:.1f} s "
          f"tau50-class {t50:.1f} s X(45) {plateau_X:.3f} nuclei_max "
          f"{int(a[:, 2].max())}; phi0=0.3 X_max {b[:, 1].max():.4f} "
          f"nuclei_max {int(b[:, 2].max())}")
    # (a) thermal nucleation + sigmoid past 50% (measured onset 8 s,
    #     tau50 27 s, X(45) ~ 0.87; >= 2x-headroom locks)
    assert onset < 25.0, onset
    assert t50 < 45.0, t50
    assert plateau_X > 0.5, plateau_X
    # (b) the below-0.4 anchor: nothing nucleates at phi0 = 0.3
    assert int(b[:, 2].max()) == 0, b[:, 2].max()
    assert b[:, 1].max() < 0.05, b[:, 1].max()
