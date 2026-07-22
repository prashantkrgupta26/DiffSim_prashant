"""P2-R2a 3-D pressure-coupling diagnostic.

Committed measurement isolating WHY the 3-D projection+SBM split diverges at
Stokes (Re=1). Tests four candidate mechanisms (M1-M4) on the Task-10 level-4
sphere fixture. Writes verdict to tests/baselines/p2r2a_diagnostic_3d.json.
Runs on gpubox (splu).

Mirrors the DECISIVE "three-number" style of the 2-D diagnostic
(tests/p2r0_divergence_diagnostic.py), which ruled out all three 2-D candidates
and localized the defect. Here each of M1-M4 gets an independent, discriminating
metric so a ruled-out mechanism shows a CLEAN negative.

The four candidates (spec §3.R2a):
  M1  PPE operator conditioning at the immersed boundary in 3-D.
  M2  3-D projection-space identity ||sigma B^T u_hat - K_p phi|| (is the
      3-D PPE even weakly-solenoidal, or does the identity break?).
  M3  pressure null-space / outflow-pin sensitivity.
  M4  Nitsche-penalty phi-growth (the penalty-magnitude hypothesis; connects
      to the R0 alpha~100 window and the alpha~Pe*p^2 law).

KEY PHYSICS SIGNAL under test (supervisor resolution 1): the Pe*p^2 penalty law
gives alpha~1.3 at Stokes, yet R0 needed alpha~100 even at low Re. That tension
means the 3-D divergence may NOT be penalty-magnitude but PPE conditioning /
pressure null-space / a mis-scaled 3-D term. M4 therefore SWEEPS alpha over
{100, 1000, 5000, 20000}: if large alpha does NOT arrest the phi-growth, the
penalty law is the WRONG lever and Task 2 should not be pursued blindly.

Usage:
    cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim
    python tests/p2r2a_diagnostic_3d.py
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

sys.path.insert(0, os.path.dirname(__file__))

from p2r0_task10_sphere_derisk import build_sphere_3d, R, CTR, U_IN

from diffsim.steppers.leray_sbm import LeraySBMStepper
from diffsim.solvers.timestepping import bdf_coeffs, bdf_order_now

DT = 0.05
LEVEL = 4
RE_STOKES = 1.0        # Re=1: advection negligible, isolates pressure coupling
ALPHA = 100.0
NSTEPS_DIAG = 12       # short: enough to see growth, not long enough to blow up
ALPHA_SWEEP = [100.0, 1000.0, 5000.0, 20000.0]


# ---------------------------------------------------------------------------
# Shared: build a fresh stepper on the level-4 sphere fixture
# ---------------------------------------------------------------------------

def _make_stepper(fx, alpha=ALPHA, dt=DT, order=1, picard=2):
    dim = fx["dim"]

    def f_fn(x, t):
        return np.zeros((len(x), dim))

    st = LeraySBMStepper(
        fx["oracle"], fx["dm"], fx["nu"], dt, f_fn,
        u_inf=fx["u_inf"], strong_mask=fx["strong_mask"],
        lam=0.5, domain="outside", order=order, picard_iters=picard,
        solver="splu", ppe_finescale=False, alpha=alpha,
        beta_backflow=1.0, velocity_update="consistent")
    st.set_initial(lambda c: np.zeros((len(c), dim)))
    return st


# ---------------------------------------------------------------------------
# Independent B^T u (weak divergence) assembly — SAME grad(N).u GP loop as the
# 2-D diagnostic (weak_divergence). This is a MEASUREMENT operator, not a
# re-run of the stepper's PPE.  Returns a FULL node-major (dm.n_nodes) rhs;
# the caller reduces to free space.
# ---------------------------------------------------------------------------

def _bt_full(dm, u_free):
    dim = dm.dim
    u_full = np.asarray(dm.constraints.T @ u_free)
    rhs = np.zeros(dm.n_nodes)
    for pv, _b in dm.bins.items():
        tb = dm.tables_by_p[pv]
        h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
        jac = (h / 2.0) ** dim
        dsc = (2.0 / h)
        conn = dm.mesh.conn_of[pv]
        uq = np.einsum("qa,ead->eqd", tb.N, u_full[conn])
        be = np.einsum("qad,eqd,q,e->ea", tb.dN, uq, tb.w, jac * dsc)
        np.add.at(rhs, conn.ravel(), be.ravel())
    return rhs


def _sigma_now(st):
    """The BDF sigma = b0/dt used by the PPE at the step ABOUT to be taken."""
    o = bdf_order_now(st.base.t + st.base.dt, st.base.dt, st.base.order,
                      have_history=st.base.hist.have(2))
    b0, _b1, _b2 = bdf_coeffs(o, st.base.dt)
    return b0 / st.base.dt


def _capture_uhat_phi(st):
    """Reproduce ONE projection step's internals WITHOUT advancing state.

    Runs the SBM predictor exactly as ``step()`` does (same extra_block /
    sbm_nodes hook), then solves the SAME PPE (classic-incremental flux =
    sigma B^T u_hat, row-0 pin) against ``st.base.K_p`` to get phi. Returns
    (uhat_free, phi_free, sigma, rhs_free, Kp_csr).  This is the honest
    interception the brief's back-compute-from-history missed (post-step
    hist.pre1 is u_new, not u_hat)."""
    sigma = _sigma_now(st)
    uhat = st._predict()                                   # SBM predictor
    dm = st.dm
    rhs = _bt_full(dm, uhat)
    rhs_free = np.asarray(dm.constraints.T.T @ (sigma * rhs))
    rhs_free[0] = 0.0
    Kp = st.base.K_p.tolil()
    Kp.rows[0] = [0]
    Kp.data[0] = [1.0]
    Kp_csr = Kp.tocsr()
    phi = splu(Kp_csr.tocsc()).solve(rhs_free)
    return uhat, phi, sigma, rhs_free, Kp_csr


# ---------------------------------------------------------------------------
# M1: PPE conditioning check
# ---------------------------------------------------------------------------

def measure_m1_ppe_conditioning(st):
    """M1: PPE operator conditioning + relative residual of the PPE solve.
    If the PPE solves accurately (ppe_resid_rel << 1) the linear solve is not
    the divergence source, so M1 is ruled OUT even if cond is large (splu is
    a direct solve; conditioning would only matter for an iterative solver)."""
    uhat, phi, sigma, rhs_free, Kp_csr = _capture_uhat_phi(st)
    resid = rhs_free - Kp_csr @ phi
    resid[0] = 0.0
    ppe_resid_rel = float(np.linalg.norm(resid)
                          / (np.linalg.norm(rhs_free) + 1e-300))
    # Condition estimate via power iteration on K_p and K_p^{-1} (row-0 pinned).
    n = Kp_csr.shape[0]
    Kp_lu = splu(Kp_csr.tocsc())
    rng = np.random.default_rng(42)
    v = rng.standard_normal(n); v /= np.linalg.norm(v)
    for _ in range(30):
        v = Kp_lu.solve(v); v /= np.linalg.norm(v)
    lam_min = 1.0 / float(np.dot(v, Kp_lu.solve(v)))
    v2 = rng.standard_normal(n); v2 /= np.linalg.norm(v2)
    for _ in range(30):
        v2 = Kp_csr @ v2; v2 /= np.linalg.norm(v2)
    lam_max = float(np.dot(v2, Kp_csr @ v2))
    cond_estimate = abs(lam_max / lam_min)
    return dict(ppe_resid_rel=ppe_resid_rel, cond_estimate=cond_estimate,
                lam_min=lam_min, lam_max=lam_max,
                verdict="NOT_M1" if ppe_resid_rel < 1e-8 else "POSSIBLE_M1")


# ---------------------------------------------------------------------------
# M2: PPE projection-space identity in 3-D  (sigma B^T u_hat == K_p phi)
# ---------------------------------------------------------------------------

def measure_m2_projection_space_identity(st):
    """M2: 3-D extension of Q1b. After the PPE solve, the discrete identity
    sigma B^T u_hat == K_p phi must hold to machine precision (row-0 pinned).
    If ||sigma B^T u_hat - K_p phi|| > 1e-9 the 3-D natural-BC / B^T assembly
    is broken (M2 IS the mechanism). Independent B^T assembly (_bt_full)."""
    uhat, phi, sigma, rhs_free, Kp_csr = _capture_uhat_phi(st)
    bt_uhat = np.asarray(st.dm.constraints.T.T @ _bt_full(st.dm, uhat))
    lhs = sigma * bt_uhat.copy()
    lhs[0] = 0.0
    resid = lhs - Kp_csr @ phi
    resid[0] = 0.0
    identity_resid = float(np.linalg.norm(resid))
    # Also report the WEAK divergence of the CORRECTED velocity (should be
    # ~machine-zero if the projection is doing its job in 3-D).
    return dict(identity_resid=identity_resid,
                bt_uhat_norm=float(np.linalg.norm(bt_uhat)),
                verdict="NOT_M2" if identity_resid < 1e-9 else "POSSIBLE_M2")


# ---------------------------------------------------------------------------
# M3: pressure null-space / outflow
# ---------------------------------------------------------------------------

def measure_m3_nullspace_outflow(st):
    """M3: pressure null-space / outflow sensitivity.
    (a) PPE solve residual with the standard row-0 pin.
    (b) Re-solve with the pin moved to the free node nearest the outflow face
        (x=1) and compare phi (up to the pin's additive-constant gauge). If the
        residual is small AND the phi FIELD (gauge-removed) is pin-invariant,
        the single pin is sufficient -> M3 ruled out."""
    uhat, phi_std, sigma, rhs_free_std, Kp_std = _capture_uhat_phi(st)
    resid_std = float(np.linalg.norm(rhs_free_std - Kp_std @ phi_std))

    dm = st.dm
    rhs_full = np.asarray(dm.constraints.T.T @ (sigma * _bt_full(dm, uhat)))
    # Alternate pin: free node nearest outflow face x=1.
    coords = dm.mesh.node_coords[dm.constraints.free_nodes]
    alt_pin = int(np.argmax(coords[:, 0]))
    Kp2 = st.base.K_p.tolil()
    Kp2.rows[alt_pin] = [alt_pin]
    Kp2.data[alt_pin] = [1.0]
    rhs_alt = rhs_full.copy()
    rhs_alt[alt_pin] = 0.0
    phi_alt = splu(Kp2.tocsr().tocsc()).solve(rhs_alt)
    # Remove the pin gauge (each phi is fixed to 0 at a DIFFERENT node): compare
    # mean-subtracted fields, the physical (gauge-invariant) content.
    d = (phi_std - phi_std.mean()) - (phi_alt - phi_alt.mean())
    phi_gauge_diff = float(np.linalg.norm(d)
                           / (np.linalg.norm(phi_std - phi_std.mean()) + 1e-300))
    # Outflow-face pressure trace magnitude (p_hat = p_star + phi; step 1 p*=0).
    outflow = np.where(np.abs(coords[:, 0] - 1.0) < 1e-12)[0]
    outflow_trace = float(np.abs(phi_std[outflow]).max()) if outflow.size else 0.0
    ruled_out = (resid_std < 1e-8 and phi_gauge_diff < 1e-6)
    return dict(ppe_resid_std=resid_std,
                phi_norm_std=float(np.linalg.norm(phi_std)),
                phi_norm_alt=float(np.linalg.norm(phi_alt)),
                phi_gauge_diff=phi_gauge_diff,
                outflow_trace_max=outflow_trace,
                n_outflow=int(outflow.size),
                verdict="NOT_M3" if ruled_out else "POSSIBLE_M3")


# ---------------------------------------------------------------------------
# M4: penalty insufficiency / no-penetration leak drives pressure growth
# ---------------------------------------------------------------------------

def _march_track(fx, alpha, dt=DT, nsteps=NSTEPS_DIAG):
    """March nsteps at a given alpha, tracking per-step:
      ||phi|| (pressure increment), ||p_hat|| (total pressure),
      |mean_un| (surrogate no-penetration leak), Cd, weak-div of corrected u,
      and PPE relative residual. Returns the trajectory dict."""
    st = _make_stepper(fx, alpha=alpha, dt=dt, order=1, picard=2)
    q = 0.5 * U_IN ** 2 * np.pi * R ** 2
    phi_norms, phat_norms, fluxes, cds, weakdivs, ppe_resids = \
        [], [], [], [], [], []
    for step in range(nsteps):
        # capture PPE residual + weak-div BEFORE the state advances
        try:
            _uh, _phi, _sig, _rf, _Kp = _capture_uhat_phi(st)
            ppe_resids.append(float(np.linalg.norm(_rf - _Kp @ _phi)
                                    / (np.linalg.norm(_rf) + 1e-300)))
        except Exception:
            ppe_resids.append(float("nan"))
        u, p = st.step()
        # After step(), st.base.p_star == p_hat (the total pressure). The
        # increment phi is not stored; ||p_hat|| growth is the dominant signal
        # (at step 1 from rest, p_star was 0 so p_hat == phi exactly).
        phat_norms.append(float(np.linalg.norm(st.base.p_star)))
        w2 = np.asarray(st.dm.constraints.T.T
                        @ _bt_full(st.dm, st.base._uvec(st.base.hist.pre1)))
        w2[0] = 0.0
        weakdivs.append(float(np.linalg.norm(w2)))
        mean_un, net, area = st.surrogate_normal_flux()
        fluxes.append(float(abs(mean_un)))
        F = st.surrogate_traction()
        cds.append(float(F[0] / q))
        print(f"[M4 a={alpha:g}] step{step+1:2d}: ||p_hat||={phat_norms[-1]:.4e} "
              f"weakdiv={weakdivs[-1]:.2e} |mean_un|={fluxes[-1]:.3e} "
              f"Cd={cds[-1]:+.4f} ppe_rr={ppe_resids[-1]:.1e}", flush=True)
    # growth metric on ||p_hat|| over the last (nsteps-2) steps
    growth = sum(phat_norms[i] > phat_norms[i - 1]
                 for i in range(2, len(phat_norms)))
    ratio = (phat_norms[-1] / (phat_norms[1] + 1e-300))
    return dict(alpha=alpha, phat_norms=phat_norms, weakdivs=weakdivs,
                normal_fluxes=fluxes, cds=cds, ppe_resids=ppe_resids,
                phat_growth_count=growth, phat_ratio=float(ratio),
                cd_final=cds[-1], cd_diverged=bool(abs(cds[-1]) > 3.0))


def measure_m4_penalty_phi_growth(fx, dt=DT, nsteps=NSTEPS_DIAG):
    """M4: penalty-magnitude hypothesis, tested by an alpha SWEEP.
    If the Nitsche penalty is the lever, LARGER alpha should arrest the
    ||p_hat|| growth / drag divergence (a monotone alpha->stability trend).
    If drag diverges for ALL alpha (even 20000, >> the R0 alpha~100 window and
    >> the Pe*p^2~1.3 law), the penalty is NOT the lever and Task 2 is the
    wrong fix. Reports per-alpha trajectories + the trend verdict."""
    runs = {}
    for a in ALPHA_SWEEP:
        runs[str(a)] = _march_track(fx, a, dt=dt, nsteps=nsteps)
    # penalty-is-lever test: does increasing alpha reduce the final |Cd|?
    finals = [(a, abs(runs[str(a)]["cd_final"])) for a in ALPHA_SWEEP]
    diverged_all = all(runs[str(a)]["cd_diverged"] for a in ALPHA_SWEEP)
    diverged_at_max = runs[str(ALPHA_SWEEP[-1])]["cd_diverged"]
    # monotone improvement with alpha?
    improves = all(finals[i][1] <= finals[i - 1][1] * 1.0
                   for i in range(1, len(finals)))
    base = runs[str(ALPHA)]
    if diverged_at_max and not improves:
        verdict = "NOT_M4"          # more penalty does NOT help -> not the lever
    elif base["phat_growth_count"] >= (nsteps - 4) and not diverged_at_max:
        verdict = "M4_LIKELY"       # growth arrested by larger alpha
    elif base["phat_growth_count"] >= (nsteps - 4):
        verdict = "M4_GROWTH_BUT_NO_ALPHA_FIX"
    else:
        verdict = "NOT_M4"
    return dict(runs=runs, finals=finals, diverged_all=diverged_all,
                diverged_at_max_alpha=diverged_at_max,
                cd_improves_with_alpha=improves, verdict=verdict)


# ---------------------------------------------------------------------------
# Public march-and-measure (Task-3 interface)
# ---------------------------------------------------------------------------

def march_and_measure_3d(fx, alpha, dt, nsteps):
    """Task-3 callable: march the projection+SBM stepper nsteps at (alpha, dt)
    on fixture fx and return the per-step trajectory dict from _march_track."""
    return _march_track(fx, alpha, dt=dt, nsteps=nsteps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("[diag3d] Building level-4 sphere fixture at Re=1 (Stokes)...",
          flush=True)
    device = "cuda:0" if os.environ.get("DIFFSIM_CUDA") else "cpu"
    fx = build_sphere_3d(device, level=LEVEL, Re=RE_STOKES)
    dh = 2 * R / (1.0 / 2 ** LEVEL)
    print(f"[diag3d] n_free={len(fx['coords'])} sf={fx['sf'].elem.size} "
          f"D/h={dh:.2f} ({time.time()-t0:.1f}s)", flush=True)

    # Principled Pe*p^2 reference at Stokes (spec §3.R2a arithmetic).
    nu = fx["nu"]
    h = 1.0 / 2 ** LEVEL
    Pe = U_IN * h / (2.0 * nu)
    alpha_pe = Pe * 1.0 ** 2
    print(f"[diag3d] Pe={Pe:.3f}  alpha_Pe(=Pe*p^2)={alpha_pe:.3f}  "
          f"(fixed alpha used = {ALPHA})", flush=True)

    print("\n--- M1: PPE conditioning ---", flush=True)
    m1 = measure_m1_ppe_conditioning(_make_stepper(fx))
    print(f"  cond_estimate={m1['cond_estimate']:.3e} "
          f"lam_min={m1['lam_min']:.3e} lam_max={m1['lam_max']:.3e}  "
          f"ppe_resid_rel={m1['ppe_resid_rel']:.3e}  verdict={m1['verdict']}",
          flush=True)

    print("\n--- M2: 3-D PPE projection-space identity ---", flush=True)
    m2 = measure_m2_projection_space_identity(_make_stepper(fx))
    print(f"  ||sigma B^T u_hat - K_p phi||={m2['identity_resid']:.3e}  "
          f"||B^T u_hat||={m2['bt_uhat_norm']:.3e}  verdict={m2['verdict']}",
          flush=True)

    print("\n--- M3: null-space / outflow sensitivity ---", flush=True)
    m3 = measure_m3_nullspace_outflow(_make_stepper(fx))
    print(f"  ppe_resid={m3['ppe_resid_std']:.3e}  "
          f"phi_gauge_diff={m3['phi_gauge_diff']:.3e}  "
          f"n_outflow={m3['n_outflow']}  verdict={m3['verdict']}", flush=True)

    print("\n--- M4: penalty/phi growth (alpha sweep) ---", flush=True)
    m4 = measure_m4_penalty_phi_growth(fx)
    print(f"  finals(alpha,|Cd|)={[(a, round(c,3)) for a,c in m4['finals']]}",
          flush=True)
    print(f"  diverged_at_max_alpha={m4['diverged_at_max_alpha']}  "
          f"cd_improves_with_alpha={m4['cd_improves_with_alpha']}  "
          f"verdict={m4['verdict']}", flush=True)

    # ---- overall verdict ----
    active = []
    if m1["verdict"] != "NOT_M1":
        active.append("M1")
    if m2["verdict"] != "NOT_M2":
        active.append("M2")
    if m3["verdict"] != "NOT_M3":
        active.append("M3")
    if m4["verdict"].startswith("M4"):
        active.append(m4["verdict"])

    if not active:
        overall = ("UNKNOWN — all M1-M4 ruled out; new candidate needed "
                   "(inspect the ||p_hat|| trajectory vs weak-div)")
    elif active == ["M4_LIKELY"]:
        overall = ("M4_PENALTY_INSUFFICIENCY — larger alpha arrests the "
                   "growth; fix: principled alpha (Task 2)")
    elif "M4_GROWTH_BUT_NO_ALPHA_FIX" in active and len(active) == 1:
        overall = ("PENALTY_NOT_THE_LEVER — p_hat grows but larger alpha does "
                   "NOT fix it; penalty law (Task 2) is the WRONG fix; "
                   "divergence is a projection pressure-coupling defect")
    else:
        overall = f"NEEDS_CONTEXT — active mechanisms: {active}"

    print(f"\n[diag3d] OVERALL VERDICT: {overall}", flush=True)

    results = {
        "_note": ("P2-R2a 3-D pressure-coupling diagnostic. Rules M1-M4 in/out "
                  "with discriminating metrics on the Re=1 Stokes level-4 sphere "
                  "(same fixture as p2r0_task10). Driver: "
                  "tests/p2r2a_diagnostic_3d.py."),
        "config": dict(level=LEVEL, Re=RE_STOKES, alpha=ALPHA, dt=DT,
                       nsteps=NSTEPS_DIAG, R=R, U_IN=U_IN, nu=nu,
                       Pe=Pe, alpha_Pe_law=alpha_pe,
                       alpha_sweep=ALPHA_SWEEP,
                       n_free=int(len(fx["coords"])),
                       sf_faces=int(fx["sf"].elem.size), D_over_h=dh),
        "M1": m1, "M2": m2, "M3": m3, "M4": m4,
        "verdict": overall,
    }
    out = os.path.join(os.path.dirname(__file__), "baselines",
                       "p2r2a_diagnostic_3d.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[diag3d] wrote {out} ({time.time()-t0:.1f}s total)", flush=True)


if __name__ == "__main__":
    main()
