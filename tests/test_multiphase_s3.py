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
import importlib.util

import numpy as np
import pytest

_has_nvmath = importlib.util.find_spec("nvmath") is not None
_skip_nvmath = pytest.mark.skipif(
    not _has_nvmath, reason="nvmath (cuDSS) not available on this machine"
)

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


# ---------------------------------------------------------------------
# S3b — EVAPORATION-QUENCH CRYSTALLIZATION (film units; ledger Sec 2)
# ---------------------------------------------------------------------
N_F, N_P = 5.0298, 87.0
DSIG_F, DH_F, TM_F = 2.6355, 1.3072, 558.0      # 2310 Table-1, RT/v0


def _s3b_stepper(dm, K_on=True, noise_psi=0.0, mpsi=1.0, seed=11,
                 eps2=4e-3, ic_seed=1011):
    """The S3b production config (ledger Sec 2.0-2.2), FILM UNITS
    (h0 = 1, D_s = 1, Bi = k_e = 0.1 from the Wodo validated set):
    ternary (M=2, K=1): 0 = fullerene-class SM (crystallizable, 2310
    dimensionless energetics at T = 333 K), 1 = polymer (N = 87),
    eliminated solvent.  chi: fp 1.0 (Negi), fs 0.7248 (2310), ps 0.3
    (Wodo); linsolver cudss (production, S2 Sec-3: splu on the
    fastmode_n L5 Jacobian is 6-10x slower; NOTE the WSL2
    clock-governor caveat in the ledger Sec 3 — cuDSS wall times
    require boosted clocks);
    r14 delta-chi chi_ca = 1.6 on both f-contacts — the
    SOLUBILITY physics (crystallization forbidden below local phi_f =
    1 - drive/chi_ca = 0.67; calibrated in the ledger: the 2310
    literal 1.0836 under-confines at phi* = 0.51 — measured runaway
    into p-rich domains — while the 16390-class 4.48/2.0 over-
    confines at L5, dissolving seeds at the phi_f ~ 0.75 domains).
    Vignes D_self at the SOFTENED dry-limit floors (2-decade
    max contrast — the SD-burst Newton-wall deviation D-S3.1,
    measured in the ledger); ls_drop (1e-6, .97, 35); b_reg 1e-3
    (wodo Fig-6, stabilizes the 85%-solvent start).  Blend phi_f0 =
    0.10, phi_p0 = 0.05, phi_s0 = 0.85 (Negi-dilute 2:1)."""
    chi_aa = np.zeros((3, 3))
    chi_aa[0, 1] = chi_aa[1, 0] = 1.0
    chi_aa[0, 2] = chi_aa[2, 0] = 0.7248
    chi_aa[1, 2] = chi_aa[2, 1] = 0.3
    chi_ca = np.zeros((3, 3))
    chi_ca[0, 1] = chi_ca[0, 2] = 1.6
    Dslf = np.array([[1e-2, 1e-3, 0.5],
                     [1e-4, 1e-4, 1e-2],
                     [1e-2, 1e-3, 1.0]])
    K = 1 if K_on else 0
    kw = dict(dsig=[DSIG_F], dh=[DH_F], Tm=[TM_F], eps2=[eps2],
              L_psi=[N_F * mpsi], noise_psi=noise_psi,
              noise_damp=(1e-2, 0.85, 15.0), clip_psi=False) \
        if K_on else {}
    st = MultiPhaseStepper(
        dm, M=2, K=K, chi_aa=chi_aa, chi_ac=chi_ca.T.copy(),
        chi_ca=chi_ca, N=[N_F, N_P, 1.0], mob="fastmode_n",
        D_self=Dslf, ls_drop=(1e-6, 0.97, 35.0), kappa=[2e-4] * 2,
        T=333.0, dt=1e-4, bulk="r14", b_reg=1e-3, newton_tol=1e-8,
        newton_max=50, linsolver="cudss", line_search=True,
        noise_seed=seed, film=dict(k_e=0.1), **kw)
    rng = np.random.default_rng(ic_seed)
    nf = st.nfree
    icf = 0.10 + 0.01 * rng.standard_normal(nf)
    icp = 0.05 + 0.01 * rng.standard_normal(nf)
    if K_on:
        st.set_initial([lambda x: icf, lambda x: icp],
                       [lambda x: np.zeros(len(x))],
                       [lambda x: np.zeros(len(x))])
    else:
        st.set_initial([lambda x: icf, lambda x: icp])
    return st


def _s3b_implant(st, ctrs, r0=0.2, w=0.03, psi_amp=0.95):
    """psi discs + theta markers at given centers; commits into
    x AND hist (the implanted state is the new committed state).
    r0 = 0.2 IS THE FATE-ROBUSTNESS MARGIN (measured scan,
    2026-07-12): r0 = 0.15 sits AT the Gibbs-Thomson r* in the
    developing domains and its fate is an assembly-atomics FP coin
    flip — the identical dry5b command grew (X = 0.961) in one run
    and dissolved (X = 0) in its replica; aggregate pre-implant
    metrics match to print precision while the field wobbles below
    it, reshuffling the phi_f-richest sites and the growth-leg
    fate.  Measured fate matrix (campaign script, correct physics):
    r0 0.15 dissolves 4/4 (t_implant 12.5 and 13); r0 0.2 GROWS 6/6
    with X = 0.9612 repeat-stable to 4-5 digits.  HISTORY (ledger
    Sec 2.3-2.4): all earlier "gate protocol dissolves" claims —
    restore-vs-continuous, wrapped-vs-non-wrapped discs, the margin
    variant chi_ca 1.4 / r0 0.2 — were measured on a MIS-BUILT
    stepper (the kw splat was missing from the MultiPhaseStepper
    call, so Tm defaulted to 1.0 and dh(1 - T/Tm) was a -332x
    MELTING drive at T = 333); with the parameters actually passed,
    this configuration grows exactly as the campaign measured.
    Continuous march through the implant, non-wrapped discs
    (campaign-verbatim geometry).  psi_amp = 0.95 (REQUIRED:
    0.5-amplitude embryos halve the bulk driving and double r* —
    measured subcritical everywhere)."""
    nd = st.ndof
    coords = st.free_coords
    psi = st.x[2 * st.M::nd]
    th = st.x[2 * st.M + 1::nd]
    for k, c in enumerate(ctrs):
        r = np.hypot(coords[:, 0] - c[0], coords[:, 1] - c[1])
        disc = psi_amp * 0.5 * (1.0 - np.tanh((r - r0) / w))
        m = disc > psi
        psi[m] = disc[m]
        th[r < r0 + 3 * w] = 0.3 + 0.4 * k
    st.hist = st.x.copy()
    st.hist2 = None


def _s3b_sites(st, n_seeds=3, min_sep=0.3):
    """phi_f-richest well-separated sites (off the moving face;
    campaign-verbatim NON-periodic separation)."""
    coords = st.free_coords
    order = np.argsort(st.phi(0))[::-1]
    ctrs = []
    for i in order:
        c = coords[i]
        if c[1] > 0.85:
            continue
        ok = True
        for cc in ctrs:
            if np.hypot(c[0] - cc[0], c[1] - cc[1]) < min_sep:
                ok = False
                break
        if ok:
            ctrs.append((float(c[0]), float(c[1])))
        if len(ctrs) >= n_seeds:
            break
    return ctrs


def _cry_area(st, cons):
    return float(np.mean(np.asarray(cons.T @ st.psi(0)) > 0.5))


def _phis_mean(st, cons):
    phis = 1.0 - sum(np.asarray(cons.T @ st.phi(i)) for i in range(2))
    return float(np.mean(phis))


# CAMPAIGN-VERBATIM growth leg (option (iii), ledger Sec 2.3): each
# growth leg pays its own continuous march through t_implant = 14 —
# NO shared save/restore cache.  Measured (2026-07-12): the restore
# protocol dissolves the same seeds that the continuous march grows
# (dry5b X = 0.961), and margin parameters do not rescue restore
# (chi_ca 1.4 / r0 0.2: area 0.244 -> 0.0).  The ~134 s premarch per
# leg (healthy clocks) is the price of a lockable gate.
def _s3b_grow_leg(dm, cons, t_implant=12.5, t_end=30.0,
                  noise_psi=0.0, seed=11):
    """Continuous march to t_implant, implant, march to t_end (the
    phis_stop = 0.02 dryness criterion terminates ~t = 21-23, so the
    locks sit on the TERMINAL X-plateau, not a mid-growth snapshot).
    t_implant = 12.5 (mid-burst): drying still deepens the quench
    after the implant — the physics margin that, with r0 = 0.2,
    makes the fate deterministic in practice (4/4 measured).
    Returns (stepper, phis_at_implant, area_at_implant)."""
    st = _s3b_stepper(dm)
    r = st.march(t_end=t_implant, dt_max=0.02, max_steps=40000,
                 dt_min=1e-11, grow_iters=45, h_min=0.14,
                 phis_stop=0.02)
    assert r == "t_end", r
    if noise_psi:
        st.noise_psi = float(noise_psi)
        st._nrng = np.random.default_rng(seed)
    phis_imp = _phis_mean(st, cons)
    ctrs = _s3b_sites(st)
    _s3b_implant(st, ctrs)
    a0 = _cry_area(st, cons)
    print(f"S3b grow-leg implant t={st.t:g} ctrs={ctrs} a0={a0:.4f}")
    st.march(t_end=t_end, dt_max=0.02, max_steps=40000,
             dt_min=1e-11, grow_iters=45, h_min=0.14,
             phis_stop=0.02)
    return st, phis_imp, a0


@_skip_nvmath
def test_s3b_mechanism_dissolve_vs_grow(device):
    """S3b gate (i) — THE EVAPORATION-QUENCH MECHANISM, deterministic
    (no noise): identical psi = 0.95 seeds (r0 = 0.2, the measured
    fate-robustness margin — see _s3b_implant) implanted in the WET
    film (t = 1, phi_s = 0.836 — below the r14 solubility: the
    chi_ca crystal-contact penalty beats the undercooling at low
    phi_f) DISSOLVE; the same seeds implanted mid-drying-burst
    (t = 12.5, f-rich domains above the 0.67 solubility crossing at
    chi_ca = 1.6, and the continuing solvent loss DEEPENS the quench
    after the implant) GROW to the terminal X ~ 0.96 plateau
    (measured 4/4 across implant times and repeats).
    Crystallization onset strictly AFTER significant solvent loss —
    thermodynamic ordering (2310's below-solubility anchor operating
    in the film frame).  LATERALLY-PERIODIC mesh (the film
    convention; measured: on a walled box the phi_f-richest sites hug
    the walls and the clipped half-discs are subcritical — ledger).
    Measured (2026-07-12, L5): see print."""
    dm, mesh, cons = _dm(5, device, periodic=(True, False))
    # wet leg
    st = _s3b_stepper(dm)
    r = st.march(t_end=1.0, dt_max=0.02, max_steps=20000,
                 dt_min=1e-11, grow_iters=45, h_min=0.14,
                 phis_stop=0.02)
    assert r == "t_end", r
    phis_wet = _phis_mean(st, cons)
    _s3b_implant(st, _s3b_sites(st))
    a0w = _cry_area(st, cons)
    st.march(t_end=5.0, dt_max=0.02, max_steps=20000, dt_min=1e-11,
             grow_iters=45, h_min=0.14, phis_stop=0.02)
    a1w = _cry_area(st, cons)
    pmw = float(np.asarray(cons.T @ st.psi(0)).max())
    print(f"S3b mechanism [wet]: implant at phi_s={phis_wet:.3f}, "
          f"area {a0w:.4f} -> {a1w:.4f}, psi_max {pmw:.3e}, "
          f"rejects {st.n_reject}")
    # dry leg (campaign-verbatim continuous march)
    st, phis_dry, a0d = _s3b_grow_leg(dm, cons)
    a1d = _cry_area(st, cons)
    pmd = float(np.asarray(cons.T @ st.psi(0)).max())
    print(f"S3b mechanism [dry]: implant at phi_s={phis_dry:.3f}, "
          f"area {a0d:.4f} -> {a1d:.4f}, psi_max {pmd:.3e}, "
          f"rejects {st.n_reject}")
    # wet: dissolution; dry: growth; ordering via phi_s
    assert a1w < 0.2 * a0w, (a0w, a1w)
    assert pmw < 0.3, pmw
    assert a1d > 1.5 * a0d, (a0d, a1d)
    assert pmd > 0.9, pmd
    assert phis_dry < 0.5 * phis_wet, (phis_dry, phis_wet)


@_skip_nvmath
def test_s3b_coupling_contrast_and_variants(device):
    """S3b gates (ii)+(iii) — coupling contrast + variants:
    (ii) the crystallization-ON dried film differs measurably from
    the K = 0 twin at the same config/horizon (crystals purify
    phi_f in cores and re-shape the amorphous pattern);
    (iii) the seeded variant is fate-and-observable reproducible
    (full independent repeat, continuous marches, no noise): the
    trajectory is FP-CHAOTIC at bit level (assembly-atomics noise
    reshuffles site selection and iterate tails — measured, see
    _s3b_implant), so the gate asserts the TERMINAL OBSERVABLES
    (crystalline area, phi_f max), which the fate-robust config
    reproduces to 4-5 digits (measured X spread 4.5e-5 over 4 runs);
    the noise variant switches FDT noise_psi = 1e-2 ON at the
    implant (growth-stage stochasticity; distribution lock from 3
    noise seeds).  Each leg pays its own continuous premarch.
    Measured (2026-07-12, L5): see print."""
    dm, mesh, cons = _dm(5, device, periodic=(True, False))

    def grow(noise_psi=0.0, seed=11):
        st, _, _ = _s3b_grow_leg(dm, cons, noise_psi=noise_psi,
                                 seed=seed)
        return st.x.copy()
    xa = grow()
    xb = grow()
    pf_a = np.asarray(cons.T @ xa[0::6])
    pf_b = np.asarray(cons.T @ xb[0::6])
    ar_a = float(np.mean(np.asarray(cons.T @ xa[4::6]) > 0.5))
    ar_b = float(np.mean(np.asarray(cons.T @ xb[4::6]) > 0.5))
    rep_dev = max(abs(ar_a - ar_b),
                  abs(float(pf_a.max()) - float(pf_b.max())))
    k1_phif = pf_a
    k1_area = ar_a
    # K0 twin at the same config, to the same dryness criterion
    st0 = _s3b_stepper(dm, K_on=False)
    r0 = st0.march(t_end=30.0, dt_max=0.02, max_steps=40000,
                   dt_min=1e-11, grow_iters=45, h_min=0.14,
                   phis_stop=0.02)
    k0_phif = np.asarray(cons.T @ st0.phi(0))
    dphi = float(np.abs(k1_phif - k0_phif).max())
    print(f"S3b coupling: K1 phi_f_max {k1_phif.max():.3f} area "
          f"{k1_area:.4f} vs K0 phi_f_max {k0_phif.max():.3f}; "
          f"max|dphi_f| {dphi:.3f}; repeat max|dx| {rep_dev:.2e}")
    areas = [k1_area]
    for ns in (21, 22, 23):
        xn = grow(noise_psi=1e-2, seed=ns)
        areas.append(float(np.mean(
            np.asarray(cons.T @ xn[4::6]) > 0.5)))
    print(f"S3b variants: noise-growth areas "
          f"{[f'{a:.4f}' for a in areas[1:]]} (deterministic "
          f"{areas[0]:.4f})")
    assert k1_area > 0.2, k1_area
    assert k1_phif.max() > k0_phif.max() + 0.02, \
        (k1_phif.max(), k0_phif.max())
    assert dphi > 0.2, dphi
    # observable-level repeat lock: measured spread 4.5e-5-class
    # over the 4-run fate scan (X and phif_max); 100x-class headroom
    # because two samples only sketch the tail
    assert rep_dev < 5e-3, rep_dev
    assert min(areas[1:]) > 0.5 * areas[0], areas
    assert max(areas[1:]) < 1.5 * areas[0] + 0.1, areas
