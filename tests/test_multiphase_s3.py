"""M5 S3 gates: EVAPORATION-INDUCED structure formation — the (M, K)
multiphase brick on the Landau-mapped moving film frame (S3a) and the
evaporation-quench crystallization pathway (S3b).

Contract: docs/theory/crystallization_formulation_p1.md Sec 4 stage S3
(psi/theta ride the mapped frame exactly as phi; the top-flux term is
phi-only — solvent evaporates AMORPHOUS); reference machinery
src/diffsim/physics/wodo_film.py (mapped gradients, frame advection,
enrichment flux, evaporation dt-cap, Biot parameterization).  Dev
ledger: docs/dev/2026-07-12-m5-s3-evaporation.md.  Tolerances
measured-then-locked (>= 2x headroom; measured numbers in comments)."""
import numpy as np
import pytest

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.multiphase import (MultiPhaseStepper, grain_labels,
                                        np_potentials)

pytestmark = pytest.mark.tier3


def _dm(level, device, p=1, periodic=None):
    tree = build_uniform(level, dim=2, periodic=periodic)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    return (DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2),
                                 device), mesh, cons)


def _integ(dm, mesh, st, *vecs):
    """Computational-domain quadrature integral of the PRODUCT of free
    vectors (GP-interpolated) — the mapped content bookkeeping is
    h_curr * this (physical content per unit lat_scale width)."""
    vs = [st._gp(v)[0] for v in vecs]
    m = 0.0
    for pv, b in dm.bins.items():
        h = mesh.tree.h()[mesh.bins[pv]]
        ne = len(mesh.conn_of[pv])
        wq = np.tile(dm.tables_by_p[pv].w, ne) \
            * np.repeat((h / 2.0) ** dm.dim, b["nqp"])
        prod = np.ones_like(wq)
        for v in vs:
            prod = prod * v[pv]
        m += float((wq * prod).sum())
    return m


def _disc(c, r0, w):
    return lambda x: 0.5 * (1.0 - np.tanh(
        (np.sqrt((x[:, 0] - c[0]) ** 2
                 + (x[:, 1] - c[1]) ** 2) - r0) / w))


# ---------------------------------------------------------------------
# S3a gate (iii) — FILM-FRAME MMS (manufactured h(t) via K_fn)
# ---------------------------------------------------------------------
def test_s3a_film_mms(device):
    """Manufactured phi*, mu*, psi*, theta* on the film frame with a
    MANUFACTURED frame velocity (K_fn = const, h(t) = h0 - K t exactly
    — the explicit h-update integrates a constant K without error).
    Sources from the mapped strong forms (computational coords,
    grad~ = (d/dx, (1/h) d/dy), advection K xi_y/h d/dy on phi, psi
    AND the KWC theta row).  This closes the (i)+(ii) coverage gap:
    the mapped eps2 psi flux and the mapped KG theta coefficient are
    order-verified here (gate (i) runs L_psi = 0; gate (ii) is K = 0
    crystal-free).  h0 = 0.6 exercises mvert = 1/h != 1 strongly;
    K = 0.05 keeps the frozen-h O(dt K) lag (h_curr at t_n vs source
    h(t_new)) below the L5 spatial floor.  p1 elements: expect ~2."""
    Mons, kap, dt = 1.0, 0.02, 1e-3
    Ninv = np.array([1.0, 1.0])
    dsig, drive_v = 1.5, -0.8
    Tm, Tq = 1.0, 0.5
    dh = drive_v / (Tq / Tm - 1.0)          # r14 drive law
    eps2, Lpsi = 5e-3, 1.0
    alpha, beta, Lth = 0.4, 0.1, 1.0
    kgd, pfl = 1e-2, 1e-3
    h0, Kf = 0.6, 0.05
    chi_aa = np.array([[0.0, 1.2], [1.2, 0.0]])
    chi_ac = np.array([[0.0, 0.7], [0.4, 0.0]])
    chi_ca = chi_ac.T.copy()
    chi_cc = np.array([[0.0, 0.9], [0.9, 0.0]])
    pars = dict(chi_aa=chi_aa, chi_ac=chi_ac, chi_ca=chi_ca,
                chi_cc=chi_cc, Ninv=Ninv, dsig=[dsig], drive=[drive_v],
                breg=0.0, bulk="r14")

    pi = np.pi
    E = lambda t: np.exp(-t)
    hf = lambda t: h0 - Kf * t
    phis = lambda x, t: 0.5 + 0.1 * np.cos(pi * x[:, 0]) \
        * np.cos(pi * x[:, 1]) * E(t)
    mus = lambda x, t: np.sin(pi * x[:, 0]) * np.sin(pi * x[:, 1]) * E(t)
    psis = lambda x, t: 0.5 + 0.25 * np.cos(pi * x[:, 0]) \
        * np.cos(2 * pi * x[:, 1]) * E(t)
    ths = lambda x, t: 0.5 + 0.2 * np.sin(pi * x[:, 0]) \
        * np.cos(pi * x[:, 1]) * E(t)

    def phi_y(x, t):
        return -0.1 * pi * np.cos(pi * x[:, 0]) * np.sin(pi * x[:, 1]) \
            * E(t)

    def psi_y(x, t):
        return -0.5 * pi * np.cos(pi * x[:, 0]) \
            * np.sin(2 * pi * x[:, 1]) * E(t)

    def psi_x(x, t):
        return -0.25 * pi * np.sin(pi * x[:, 0]) \
            * np.cos(2 * pi * x[:, 1]) * E(t)

    def th_derivs(x, t):
        s, c = np.sin(pi * x[:, 0]), np.cos(pi * x[:, 0])
        sy, cy = np.sin(pi * x[:, 1]), np.cos(pi * x[:, 1])
        tx = 0.2 * pi * c * cy * E(t)
        ty = -0.2 * pi * s * sy * E(t)
        txx = -pi ** 2 * (ths(x, t) - 0.5)
        tyy = -pi ** 2 * (ths(x, t) - 0.5)
        txy = -0.2 * pi ** 2 * c * sy * E(t)
        return tx, ty, txx, tyy, txy

    pori = lambda s: s ** 2 * (3 - 2 * s)
    porip = lambda s: 6 * s * (1 - s)

    def f_phi(x, t):
        h = hf(t)
        adv = Kf * x[:, 1] / h
        lap_mu = -pi ** 2 * (1.0 + 1.0 / h ** 2) * mus(x, t)
        return -(phis(x, t) - 0.5) + adv * phi_y(x, t) - Mons * lap_mu

    def f_mu(x, t):
        h = hf(t)
        mu_b, _ = np_potentials([phis(x, t)], [psis(x, t)], pars)
        lap_phi = -pi ** 2 * (1.0 + 1.0 / h ** 2) * (phis(x, t) - 0.5)
        return mus(x, t) - mu_b[0] + kap * lap_phi

    def f_psi(x, t):
        h = hf(t)
        adv = Kf * x[:, 1] / h
        _, dfs = np_potentials([phis(x, t)], [psis(x, t)], pars)
        tx, ty, _, _, _ = th_derivs(x, t)
        g2 = tx ** 2 + ty ** 2 / h ** 2          # mapped |grad~ th|^2
        S = np.sqrt(g2 + kgd ** 2)
        Eori = 0.5 * alpha * S + 0.5 * beta * g2
        lap_psi = -pi ** 2 * (1.0 + 4.0 / h ** 2) * (psis(x, t) - 0.5)
        return (-(psis(x, t) - 0.5) + adv * psi_y(x, t)
                + Lpsi * (dfs[0] + porip(psis(x, t)) * Eori)
                - Lpsi * eps2 * lap_psi)

    def f_th(x, t):
        # (p+pf)(th_t + adv th_y) - div~[(p+pf) Lth ((a/2)/S~ + b)
        #  grad~ th] = f (mapped div~/grad~; h const in space)
        h = hf(t)
        adv = Kf * x[:, 1] / h
        s = psis(x, t)
        tx, ty, txx, tyy, txy = th_derivs(x, t)
        g2 = tx ** 2 + ty ** 2 / h ** 2
        S = np.sqrt(g2 + kgd ** 2)
        cf = 0.5 * alpha / S + beta
        c = (pori(s) + pfl) * Lth * cf
        Sx = (tx * txx + ty * txy / h ** 2) / S
        Sy = (tx * txy + ty * tyy / h ** 2) / S
        sx, sy = psi_x(x, t), psi_y(x, t)
        cx = Lth * (porip(s) * sx * cf
                    - (pori(s) + pfl) * 0.5 * alpha * Sx / S ** 2)
        cy = Lth * (porip(s) * sy * cf
                    - (pori(s) + pfl) * 0.5 * alpha * Sy / S ** 2)
        divF = cx * tx + cy * ty / h ** 2 \
            + c * (txx + tyy / h ** 2)
        return (pori(s) + pfl) * (-(ths(x, t) - 0.5)
                                  + adv * ty) - divF

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
            dt=dt, bulk="r14", kg_delta=kgd, p_floor=pfl,
            newton_tol=1e-10, newton_max=80,
            dirichlet=np.where(bdry)[0],
            g_fns=[phis, mus, psis, ths],
            src_fns=[f_phi, f_mu, f_psi, f_th],
            film=dict(k_e=0.0, h0=h0, K_fn=lambda t: Kf))
        st.set_initial([lambda x: phis(x, 0.0)],
                       [lambda x: psis(x, 0.0)],
                       [lambda x: ths(x, 0.0)])
        st.x[1::st.ndof] = mus(st.free_coords, 0.0)
        for _ in range(4):
            st.step()
        assert abs(st.h_curr - hf(st.t)) < 1e-14, (st.h_curr, hf(st.t))
        from diffsim.physics.poisson import l2_error
        errs["phi"].append(l2_error(dm, np.asarray(cons.T @ st.phi(0)),
                                    lambda x: phis(x, st.t)))
        errs["psi"].append(l2_error(dm, np.asarray(cons.T @ st.psi(0)),
                                    lambda x: psis(x, st.t)))
        errs["theta"].append(l2_error(dm,
                                      np.asarray(cons.T @ st.theta(0)),
                                      lambda x: ths(x, st.t)))
    orders = {f: np.log2(errs[f][0] / errs[f][1]) for f in errs}
    print("S3a film MMS: " + "; ".join(
        f"{f} errs {[f'{e:.2e}' for e in errs[f]]} "
        f"order {orders[f]:.2f}" for f in errs))
    assert orders["phi"] > 1.8, (errs["phi"], orders["phi"])
    assert orders["psi"] > 1.8, (errs["psi"], orders["psi"])
    assert orders["theta"] > 1.8, (errs["theta"], orders["theta"])


# ---------------------------------------------------------------------
# S3a gate (i) — MOVING-FRAME BOOKKEEPING (solute + crystallinity)
# ---------------------------------------------------------------------
def _bookkeeping_march(device, tstep):
    """Ternary film (M=2, K=1), miscible chi (no spinodal chaos),
    INTERIOR psi blob (deep: the h Int psi invariant of pure advection
    holds while psi_top = 0; a shallower blob measures the REAL top-
    tail leakage — the tanh tail advected into the receding surface —
    at 1.76e-6 over the same march, recorded in the ledger),
    CRYSTALLIZATION DRIVING OFF: L_psi = 0 (the psi row reduces to
    pure frame advection) AND NEUTRAL crystal energetics dsig = dh =
    0 (MEASURED NECESSITY, 2026-07-12: with dsig = dh != 0 the W(psi)
    crystal well rides in mu_0 even at L_psi = 0 and sucks species 0
    into the blob — phi_in_blob 0.25 -> 0.585 in t = 0.06, cv drift
    +131% — REAL thermodynamic coupling, not a mapping artifact; the
    mapping-isolation gate removes it), frozen-theta markers,
    clip_psi=False.  March h 1.0 -> 0.85 on the Appendix-A ladder +
    the evaporation dt-cap."""
    dm, mesh, cons = _dm(5, device)
    chi_aa = 0.3 * np.ones((3, 3)) - 0.3 * np.eye(3)
    st = MultiPhaseStepper(
        dm, M=2, K=1, chi_aa=chi_aa, onsager=[[0.5, 0.0], [0.0, 0.5]],
        kappa=(2e-4, 2e-4), N=[5.0, 5.0, 1.0], dsig=[0.0], dh=[0.0],
        Tm=[1.0], eps2=[1e-3], L_psi=[0.0], T=0.5, bulk="r14",
        dt=1e-3, film=dict(k_e=1.0), clip_psi=False, tstep=tstep)
    st.set_initial(
        [lambda x: np.full(len(x), 0.25),
         lambda x: np.full(len(x), 0.25)],
        [_disc((0.5, 0.35), 0.12, 0.04)],
        [lambda x: np.where(x[:, 0] < 0.5, 0.25, 1.25)])
    nd = st.ndof
    c0 = [st.h_curr * _integ(dm, mesh, st, st.x[2 * i::nd])
          for i in range(2)]
    q0 = st.h_curr * _integ(dm, mesh, st, st.x[4::nd])
    cv0 = st.h_curr * _integ(dm, mesh, st, st.x[0::nd], st.x[4::nd])
    dhs = []
    hprev = [st.h_curr]

    def cb(s, dt_eff, iters):
        dhs.append(hprev[0] - s.h_curr)
        hprev[0] = s.h_curr
    reason = st.march(t_end=5.0, dt_max=5e-3, max_steps=800,
                      h_min=0.85, callback=cb)
    c1 = [st.h_curr * _integ(dm, mesh, st, st.x[2 * i::nd])
          for i in range(2)]
    q1 = st.h_curr * _integ(dm, mesh, st, st.x[4::nd])
    cv1 = st.h_curr * _integ(dm, mesh, st, st.x[0::nd], st.x[4::nd])
    dr_c = max(abs(c1[i] - c0[i]) / c0[i] for i in range(2))
    dr_q = abs(q1 - q0) / q0
    dr_cv = (cv1 - cv0) / cv0           # SIGNED (analytic bracket)
    return st, reason, dr_c, dr_q, dr_cv, max(dhs)


def test_s3a_film_bookkeeping_bdf1(device):
    """Gate (i) BDF1: physical solute content h Int phi_i dtheta of
    both NONVOLATILE species conserved through the moving frame; psi
    content h Int psi dtheta conserved (pure advection, psi_top = 0 —
    the mapping neither creates nor destroys crystallinity); the
    crystalline volume h Int phi_0 psi lives in the ANALYTIC PHYSICS
    BRACKET [0, 1/h_end - 1]: slow-diffusion limit 0 (phi and psi
    both frozen in physical z => the physical integral is invariant),
    fast-diffusion limit phi -> phi0/h uniformly => cv ~ 1/h — the
    enrichment-through-blob correlation, real model physics of the
    psi-fraction formulation; a mapping bug lands OUTSIDE the
    bracket.  The evaporation dt-cap bounds every accepted dh <=
    dh_cap.  Measured (2026-07-12, L5): solute drift 6.7e-16 (EXACT
    class — the wodo sign-pairing identity), psi content 9.2e-10
    (residual tanh-tail top leakage: the blob edge sits tanh(11.7) ~
    1e-10 from the receding surface; the shallow-blob variant
    measures 1.76e-6 — ledger Sec 1), crystalline volume +1.649e-1
    inside the bracket [0, 0.178] (the miscible high-mobility config
    sits near the fast-diffusion limit), max dh 2.37e-3 <= dh_cap
    4e-3, 0 rejects, h_min stop."""
    st, reason, dr_c, dr_q, dr_cv, dh_max = \
        _bookkeeping_march(device, "bdf1")
    bracket = 1.0 / st.h_curr - 1.0
    print(f"S3a bookkeeping BDF1: reason={reason} h={st.h_curr:.4f} "
          f"t={st.t:.4f} rejects={st.n_reject}; solute drift "
          f"{dr_c:.2e}, psi content {dr_q:.2e}, crystalline volume "
          f"{dr_cv:+.3e} (bracket [0, {bracket:.3f}]), "
          f"max dh {dh_max:.2e}")
    assert reason == "h_min", reason
    assert dr_c < 1e-12, dr_c           # measured 6.7e-16
    assert dr_q < 1e-8, dr_q            # measured 9.2e-10 (tail note)
    assert -1e-3 < dr_cv < bracket, (dr_cv, bracket)  # measured .1649
    assert dh_max <= st.dh_cap * (1.0 + 1e-9), (dh_max, st.dh_cap)


def test_s3a_film_bookkeeping_bdf2(device):
    """Gate (i) BDF2 (deterministic; the standing BDF1+BDF2 rule): the
    film terms ride the variable-coefficient scheme.  The BDF1
    content pairing is telescoping-EXACT; under BDF2 the content
    drift is the scheme's O(dt^2) global error on P' = (K/h) P — a
    discretization-order drift, not a leak (module docstring S3a).
    Measured (2026-07-12, L5): solute drift 8.43e-5, psi content
    8.43e-5 (the SAME advection identity error — both contents ride
    the identical pairing), crystalline volume +1.646e-1 (bracket as
    BDF1), 0 rejects."""
    st, reason, dr_c, dr_q, dr_cv, dh_max = \
        _bookkeeping_march(device, "bdf2")
    bracket = 1.0 / st.h_curr - 1.0
    print(f"S3a bookkeeping BDF2: reason={reason} h={st.h_curr:.4f} "
          f"t={st.t:.4f} rejects={st.n_reject}; solute drift "
          f"{dr_c:.2e}, psi content {dr_q:.2e}, crystalline volume "
          f"{dr_cv:+.3e} (bracket [0, {bracket:.3f}]), "
          f"max dh {dh_max:.2e}")
    assert reason == "h_min", reason
    assert dr_c < 5e-4, dr_c            # measured 8.43e-5 (O(dt^2))
    assert dr_q < 5e-4, dr_q            # measured 8.43e-5
    assert -1e-3 < dr_cv < bracket, (dr_cv, bracket)  # measured .1646
    assert dh_max <= st.dh_cap * (1.0 + 1e-9), (dh_max, st.dh_cap)


# ---------------------------------------------------------------------
# S3a gate (ii) — AMORPHOUS REGRESSION: K=0 film mode == wodo_film
# ---------------------------------------------------------------------
def test_s3a_film_amorphous_regression(device):
    """(M=2, K=0) film mode vs WodoFilmStepper on a matched ternary
    evaporating config (the wodo Fig-3-class blend: chi (1, .3, .3),
    N (5, 5, 1), constant M0 = 0.225, k_e = 1) — the S0-analogue for
    the film frame.  FIXED-dt march (the wodo device-parity template:
    removes the Appendix-A heuristic), 10 steps at dt = 1e-3, K
    recomputed per step from the committed state on both sides.
    Measured (2026-07-12, L5): per-step trajectory max |dx| = 2.2e-16
    (MACHINE-IDENTICAL class — the film kernel's arithmetic
    reproduces wodo's to the last bit at this config), h parity
    0.0 exactly, K parity < 1e-14 per step."""
    from diffsim.physics.wodo_film import WodoFilmStepper
    dm, mesh, cons = _dm(5, device)
    chi = (1.0, 0.3, 0.3)
    st_w = WodoFilmStepper(dm, chi=chi, N=(5.0, 5.0, 1.0),
                           M=(0.225, 0.0, 0.225), kappa=(2e-4, 2e-4),
                           k_e=1.0, dt=1e-3)
    rng = np.random.default_rng(7)
    ic1 = 0.2 + 0.01 * rng.standard_normal(len(st_w.free_coords))
    ic2 = 0.2 + 0.01 * rng.standard_normal(len(st_w.free_coords))
    st_w.set_initial(lambda x: ic1, lambda x: ic2)
    chi_aa = np.zeros((3, 3))
    chi_aa[0, 1] = chi_aa[1, 0] = chi[0]
    chi_aa[0, 2] = chi_aa[2, 0] = chi[1]
    chi_aa[1, 2] = chi_aa[2, 1] = chi[2]
    st_m = MultiPhaseStepper(
        dm, M=2, K=0, chi_aa=chi_aa, N=[5.0, 5.0, 1.0],
        onsager=[[0.225, 0.0], [0.0, 0.225]], kappa=(2e-4, 2e-4),
        dt=1e-3, guards=False, film=dict(k_e=1.0))
    st_m.set_initial([lambda x: ic1, lambda x: ic2])
    dt, errs = 1e-3, []
    for n in range(10):
        p1n, p2n = st_w.hist[0]
        Kw = max(st_w.k_e * st_w._top_phis_avg(p1n, p2n), 0.0)
        Km = st_m._film_K()
        assert abs(Kw - Km) < 1e-14, (n, Kw, Km)
        xw, itw, okw = st_w._attempt(dt, Kw)
        assert okw, (n, itw)
        st_w.x = xw
        st_w.hist = [(xw[0::4].copy(), xw[2::4].copy()), st_w.hist[0]]
        st_w.t += dt
        st_w.h_curr -= dt * Kw
        st_m.step()
        errs.append(np.abs(st_m.x - xw).max())
    dh = abs(st_m.h_curr - st_w.h_curr)
    print(f"S3a amorphous regression: per-step max|dx| "
          f"{[f'{e:.2e}' for e in errs]} max {max(errs):.3e}; "
          f"h {st_m.h_curr:.6f} vs {st_w.h_curr:.6f} (|dh|={dh:.1e})")
    assert max(errs) < 1e-12, errs      # measured 2.2e-16
    assert dh < 1e-13, dh               # measured 0.0


# ---------------------------------------------------------------------
# S3a — A1 T-field in the film frame + evaporative cooling (hook)
# ---------------------------------------------------------------------
def test_s3a_film_tfield_evap_cooling(device):
    """T-field in the film frame: substrate Dirichlet T = 350, top
    face natural with the s_evap = -L_vap J_evap latent-heat sink
    (parameterized; default OFF).  With L_vap = 2 the receding surface
    cools toward the flux balance T_top ~ T_sub - L_vap K h / k_th
    (K ~ 0.5 falling, h -> 0.9); with L_vap = 0 the field stays
    uniform at T_sub to solver roundoff.  Measured (2026-07-12, L5):
    L_vap=2 top-row deficit T_sub - T_top = 0.421 (end-state flux
    balance L_vap K h / k_th = 0.741 — the transient hasn't reached
    steady state, deficit < balance as expected); L_vap=0 max
    |T - 350| = 2.0e-11."""
    dm, mesh, cons = _dm(5, device)
    chi_aa = 0.3 * np.ones((3, 3)) - 0.3 * np.eye(3)
    coords = mesh.node_coords[cons.free_nodes]
    sub = np.where(np.abs(coords[:, 1]) < 1e-12)[0]
    res = {}
    for lvap in (2.0, 0.0):
        st = MultiPhaseStepper(
            dm, M=2, K=1, chi_aa=chi_aa,
            onsager=[[0.5, 0.0], [0.0, 0.5]], kappa=(2e-4, 2e-4),
            N=[5.0, 5.0, 1.0], dsig=[1.0], dh=[0.0], Tm=[400.0],
            eps2=[1e-3], L_psi=[0.0], bulk="r14", dt=1e-3,
            film=dict(k_e=1.0), clip_psi=False,
            T_mode="field",
            T_field=dict(T0=350.0, rho_cp=1.0, k_th=1.0,
                         dirichlet=sub, g=lambda x, t: np.full(
                             len(x), 350.0), L_vap=lvap))
        st.set_initial(
            [lambda x: np.full(len(x), 0.25),
             lambda x: np.full(len(x), 0.25)],
            [lambda x: np.zeros(len(x))], [lambda x: np.zeros(len(x))])
        st.march(t_end=5.0, dt_max=5e-3, max_steps=400, h_min=0.9)
        top = np.abs(coords[:, 1] - 1.0) < 1e-12
        res[lvap] = (float(np.mean(st.T_nodes[top])),
                     float(np.abs(st.T_nodes - 350.0).max()),
                     st._K_pend, st.h_curr)
    dT = 350.0 - res[2.0][0]
    bal = 2.0 * res[2.0][2] * res[2.0][3] / 1.0
    print(f"S3a evap cooling: L_vap=2 top T {res[2.0][0]:.3f} "
          f"(deficit {dT:.3f}, end flux-balance {bal:.3f}); "
          f"L_vap=0 max|T-350| {res[0.0][1]:.2e}")
    assert dT > 0.15, dT                        # measured 0.421
    assert dT < 2.0 * bal + 0.5, (dT, bal)      # flux-balance class
    assert res[0.0][1] < 1e-9, res[0.0][1]      # measured 2.0e-11


