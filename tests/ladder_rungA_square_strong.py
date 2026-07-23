"""Projection Validation Ladder — Rung A driver: body-fitted square, STRONG
Dirichlet no-slip (the DECISIVE control).

Rung 0 (cavity) already PASSED — the base projection plumbing is sound. Rung A
asks the single decisive question the sphere defect raised: on an EXACT
body-fitted square obstacle (aligned `Box`, `dmax==0`, so NO SBM shift and NO
weak Nitsche), with STRONG no-slip on the carved obstacle faces, does the
single-pass pressure-projection split develop flow past the body from rest to
the SAME-MESH MONOLITHIC magnitude — or does it PIN at a weak fixed point (the
~2%/8%-of-monolithic sphere symptom, the lagged-p* from-rest under-development
mechanism of docs/dev/2026-07-23-projection-sbm-weak-fixed-point-verdict.md)?

Two solves march on the SAME mesh (the Task-1 `build_square_channel_2d` fixture
with an ALIGNED `half` so `dmax==0`):

  * PROJECTION — `LerayProjectionStepper` (base projection, non-SBM), single-pass
    (`picard_iters=1`), BDF1/2 incremental pressure. The obstacle is enforced
    STRONGLY: its carved-face nodes join the box walls + inflow in the strong
    Dirichlet set (u_inf=U_IN at inflow, 0 on walls AND obstacle), by overriding
    `base.dir_nodes` and supplying the combined trace via `g_fn` — the
    `LeraySBMStepper` override pattern, but with pure strong Dirichlet (no SBM
    face block). Outflow (x=1) is FREE; the PPE pins pressure at the outflow
    nodes (Taly physical outlet BC — this is an EXTERNAL flow, not enclosed).
  * MONOLITHIC — the same-mesh saddle NS solve (inline, mirrors
    tests/p2r0_task10_sphere_derisk.py::monolithic_cd) with the IDENTICAL strong
    set (walls + inflow + obstacle) and the SAME single pressure-DOF pin. NO SBM
    Nitsche block (rung A is strong Dirichlet). This is the primary, box-free bar.

Drag: `surrogate_traction(dm, sf, geo, x_full, nu, ndof)` integrates the traction
on the carved obstacle faces; with `dmax==0` (aligned) this is the EXACT
body-fitted traction (`geo.corr==1`, surrogate faces == true box faces).
`Cd = F_x / qref`, `qref = 0.5 U_IN^2 D`, `D = 2*half` (2-D, frontal length).
Lift `Cl = F_y / qref` for the Re=100 Strouhal signal.

Run (gpubox, CPU-splu — 2-D is small):
    PYTHONPATH=src:tests .venv/bin/python tests/ladder_rungA_square_strong.py
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from diffsim.steppers.leray import LerayProjectionStepper
from diffsim.api.ns_bricks import assemble_linear_ns
from diffsim.physics.poisson import gauss_points
from diffsim.sbm.vector import surrogate_traction

from ladder_fixtures import build_square_channel_2d, U_IN


# --------------------------------------------------------------------------
# strong BC: walls + inflow + obstacle (reviewer note: ZERO u_inf on obstacle)
# --------------------------------------------------------------------------
def build_strong_bc(fx):
    """Combine the box strong set (walls + inflow) with the obstacle nodes into
    one STRONG Dirichlet set. Returns (strong_nodes, g_strong) in FREE-node
    space. Inflow nodes carry U_IN (+x); channel walls AND obstacle nodes carry
    zero (no-slip) — the reviewer note: ZERO `u_inf` at the obstacle nodes.

    `fx["u_inf"]` already has U_IN at inflow and 0 elsewhere (walls). The
    obstacle nodes are added to the strong set with their `u_inf` rows (== 0),
    so the combined trace is inflow=U_IN, walls+obstacle=0 automatically."""
    dim = fx["dim"]
    combined = fx["strong_mask"] | fx["obstacle_node_mask"]
    strong_nodes = np.where(combined)[0]
    g_strong = fx["u_inf"][strong_nodes]           # U_IN@inflow, 0@walls+obstacle
    return strong_nodes, g_strong


def qref(fx):
    """Dynamic pressure reference for the 2-D square: 0.5 U_IN^2 D,
    D = 2*half = obstacle side (frontal length)."""
    D = 2.0 * fx["half"]
    return 0.5 * U_IN ** 2 * D


def mean_speed(u_node):
    """Domain mean |u| over the free nodes (the development metric)."""
    return float(np.sqrt((u_node ** 2).sum(1)).mean())


# --------------------------------------------------------------------------
# PROJECTION march (base LerayProjectionStepper, strong obstacle)
# --------------------------------------------------------------------------
def march_projection(fx, dt=0.02, nsteps=400, rate_tol=None, order=2,
                     log_every=0):
    """March the base projection stepper (single-pass) with STRONG no-slip on
    the obstacle. Returns dict(cd, cl, mean_u, div, steps, cd_hist, cl_hist).

    Obstacle strong Dirichlet is imposed by overriding `base.dir_nodes` to the
    combined strong set and returning the combined trace from `g_fn`. Outflow
    pressure is pinned (physical outlet BC) so the incremental p* does not drift
    on this external flow."""
    dim = fx["dim"]
    dm, mesh = fx["dm"], fx["mesh"]
    sf, geo = fx["sf"], fx["geo"]
    ndof, nu = fx["ndof"], fx["nu"]
    strong_nodes, g_strong = build_strong_bc(fx)

    def f_fn(x, t):
        return np.zeros((len(x), dim))

    # g_fn receives free_coords[dir_nodes] in dir_nodes order; we return the
    # matching combined trace (strong_nodes order == dir_nodes order below).
    def g_fn(coords_at_dir, t):
        return g_strong

    st = LerayProjectionStepper(
        dm, nu, dt, f_fn=f_fn, g_fn=g_fn, order=order, picard_iters=1,
        solver="splu", pressure_outflow_nodes=fx["outflow_nodes"])
    st.dir_nodes = strong_nodes                    # override: box + obstacle strong
    st.set_initial(lambda c: np.zeros((len(c), dim)))

    T = dm.constraints.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    q = qref(fx)
    prev = None
    steps = 0
    cd = cl = np.nan
    cd_hist, cl_hist, mu_hist = [], [], []
    u = None
    blew_up = False
    for steps in range(1, nsteps + 1):
        u, p = st.step()
        # blow-up guard: the base single-pass projection on an OPEN outflow is
        # pressure-unstable (‖p‖/div grow unbounded — the weak-fixed-point
        # verdict's experiment-4 mechanism); bail once it diverges so the driver
        # records the blow-up (an expected rung-A failure mode) without wasting
        # the long shedding march.
        if not np.isfinite(u).all() or not np.isfinite(p).all() \
                or np.abs(u).max() > 1e4:
            blew_up = True
            if log_every:
                print(f"[rungA proj] BLOW-UP at step{steps} "
                      f"(max|u|={np.abs(u).max():.3e}) — base single-pass "
                      f"projection is pressure-unstable on the open outflow",
                      flush=True)
            break
        # assemble full (u,p) node-major vector for the traction integral
        xfree = np.zeros(st.n_free * ndof)
        xv = xfree.reshape(st.n_free, ndof)
        xv[:, :dim] = u
        xv[:, dim] = p
        F = surrogate_traction(dm, sf, geo, np.asarray(T_vec @ xfree), nu, ndof)
        cd = float(F[0] / q)
        cl = float(F[1] / q)
        mu = mean_speed(u)
        cd_hist.append(cd)
        cl_hist.append(cl)
        mu_hist.append(mu)
        if log_every and (steps <= 3 or steps % log_every == 0):
            print(f"[rungA proj] step{steps:4d}  Cd={cd:+.4f}  Cl={cl:+.4f}  "
                  f"mean|u|={mu:.4f}", flush=True)
        if rate_tol is not None and prev is not None:
            if np.abs(u - prev).max() / dt < rate_tol:
                break
        prev = u.copy()
    div = float(st.divergence_l2())
    return dict(cd=cd, cl=cl, mean_u=mean_speed(u), div=div, steps=steps,
                cd_hist=cd_hist, cl_hist=cl_hist, mu_hist=mu_hist,
                pnorm=float(np.linalg.norm(st.p_star)), blew_up=blew_up)


# --------------------------------------------------------------------------
# MONOLITHIC march (inline saddle, same mesh, strong obstacle, NO SBM block)
# --------------------------------------------------------------------------
def march_monolithic(fx, dt=0.02, nsteps=400, rate_tol=2e-4, log_every=0):
    """March the same-mesh monolithic saddle NS (inline; mirrors
    p2r0_task10_sphere_derisk::monolithic_cd) with STRONG no-slip on the
    obstacle and NO SBM Nitsche block. Returns dict(cd, cl, mean_u, div, steps,
    cd_hist, cl_hist)."""
    dim = fx["dim"]
    dm, mesh, cons = fx["dm"], fx["mesh"], fx["cons"]
    sf, geo = fx["sf"], fx["geo"]
    ndof, nu = fx["ndof"], fx["nu"]
    coords = fx["coords"]
    strong_nodes, g_strong = build_strong_bc(fx)

    T = cons.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    nfree = T.shape[1]
    xq = gauss_points(mesh, dm.tables_by_p)

    def gp_field(node_vec):
        full = np.asarray(T @ node_vec)
        aq, dq = {}, {}
        for pv in dm.bins:
            tb = dm.tables_by_p[pv]
            vals = full[mesh.conn_of[pv]]
            aq[pv] = np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)
            h = mesh.tree.h()[mesh.bins[pv]]
            dq[pv] = (np.einsum("qad,ead->eq", tb.dN, vals)
                      * (2.0 / h)[:, None]).reshape(-1)
        return aq, dq

    # outflow pressure pin: pin the pressure DOF at the free node with max x
    # (an outflow node) so the monolithic uses the SAME external-flow gauge as
    # the projection (pressure defined by the free outflow, not an arbitrary
    # interior node). This mirrors monolithic_cd's argmax-coords pin.
    p_pin = int(np.argmax(coords[:, 0]))
    q = qref(fx)
    x = np.zeros(nfree * ndof)
    sigma = 1.0 / dt
    prev_u = None
    steps = 0
    cd = cl = np.nan
    cd_hist, cl_hist, mu_hist = [], [], []
    u_new = None
    for step in range(nsteps):
        u_node = x.reshape(nfree, ndof)[:, :dim]
        aq, dq = gp_field(u_node)
        fq = {pv: aq[pv] / dt for pv in xq}
        A, b = assemble_linear_ns(dm, aq, dq, fq, nu, sigma=sigma)
        A = A.tolil()
        for k, i in enumerate(strong_nodes):
            for c in range(dim):
                r = i * ndof + c
                A.rows[r] = [int(r)]
                A.data[r] = [1.0]
                b[r] = g_strong[k, c]
        pin = p_pin * ndof + dim
        A.rows[pin] = [pin]
        A.data[pin] = [1.0]
        b[pin] = 0.0
        x = splu(A.tocsr().tocsc()).solve(b)
        u_new = x.reshape(nfree, ndof)[:, :dim]
        F = surrogate_traction(dm, sf, geo, np.asarray(T_vec @ x), nu, ndof)
        cd = float(F[0] / q)
        cl = float(F[1] / q)
        mu = mean_speed(u_new)
        cd_hist.append(cd)
        cl_hist.append(cl)
        mu_hist.append(mu)
        steps = step + 1
        if log_every and (steps <= 3 or steps % log_every == 0):
            print(f"[rungA mono] step{steps:4d}  Cd={cd:+.4f}  Cl={cl:+.4f}  "
                  f"mean|u|={mu:.4f}", flush=True)
        if rate_tol is not None and prev_u is not None and step > 5:
            if np.abs(u_new - prev_u).max() / dt < rate_tol:
                break
        prev_u = u_new.copy()
    # divergence sentinel on the final field
    _, dq = gp_field(u_new)
    tot, vol = 0.0, 0.0
    for pv in dm.bins:
        tb = dm.tables_by_p[pv]
        h = mesh.tree.h()[mesh.bins[pv]]
        jac = (h / 2.0) ** dim
        w = np.tile(tb.w, len(h)) * np.repeat(jac, tb.nqp)
        tot += (dq[pv] ** 2 * w).sum()
        vol += w.sum()
    div = float(np.sqrt(tot / vol))
    return dict(cd=cd, cl=cl, mean_u=mean_speed(u_new), div=div, steps=steps,
                cd_hist=cd_hist, cl_hist=cl_hist, mu_hist=mu_hist,
                p_pin=p_pin)


# --------------------------------------------------------------------------
# Strouhal from a lift (Cl) signal
# --------------------------------------------------------------------------
def strouhal(cl_hist, dt, D, U=U_IN, skip_frac=0.4):
    """Dominant Strouhal St = f D / U from the Cl time series (drop the first
    `skip_frac` as transient; peak of the FFT of the detrended tail). Returns
    (St, f_peak) or (nan, nan) if the tail is too short / featureless."""
    cl = np.asarray(cl_hist, dtype=float)
    n = len(cl)
    i0 = int(skip_frac * n)
    tail = cl[i0:]
    if len(tail) < 16:
        return float("nan"), float("nan")
    tail = tail - tail.mean()
    if tail.std() < 1e-8:
        return float("nan"), float("nan")
    win = np.hanning(len(tail))
    sp_ = np.abs(np.fft.rfft(tail * win))
    freqs = np.fft.rfftfreq(len(tail), d=dt)
    sp_[0] = 0.0                                   # kill DC
    k = int(np.argmax(sp_))
    f_peak = float(freqs[k])
    return float(f_peak * D / U), f_peak


# --------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------
# Same-mesh oracle tolerances. Projection must MATCH the monolithic on the
# identical mesh (the primary box-free bar). The decisive failure mode is the
# weak plateau: mean|u| stuck at a few % of monolithic (the sphere symptom).
TOL_CD_REL = 0.15      # projection Cd vs monolithic Cd (relative)
TOL_MU_REL = 0.20      # projection mean|u| vs monolithic mean|u| (relative)
WEAK_PLATEAU_FRAC = 0.5  # mean|u| below this fraction of monolithic == weak pin


def run_rungA(level=5, half=0.125, res=(40, 100), device="cpu",
              dt40=0.02, nsteps40=600, dt100=0.01, nsteps100=2000,
              log_every=25):
    """Full rung-A driver. Re=40 steady (PRIMARY, decisive): projection vs
    monolithic steady Cd + mean|u| development. Re=100 shedding: mean Cd over a
    period + Strouhal from Cl, both solvers. Prints PASS/FAIL vs the same-mesh
    oracle and returns a results dict.

    THE decisive read (per Re): does strong-Dirichlet body-fitted projection
    match the monolithic Cd AND develop mean|u| to the monolithic magnitude, or
    pin weak (~few %, the sphere symptom)?"""
    results = {}
    all_pass = True
    D = 2.0 * half
    for Re in res:
        steady = (Re < 50)
        dt = dt40 if steady else dt100
        nsteps = nsteps40 if steady else nsteps100
        rate_tol_p = 5e-4 if steady else None      # shedding never "goes steady"
        rate_tol_m = 2e-4 if steady else None

        print(f"\n{'='*70}\n Rung A — body-fitted square STRONG Dirichlet  "
              f"Re={Re}  level={level}  half={half} (D={D})"
              f"\n{'='*70}", flush=True)

        fx = build_square_channel_2d(level, Re, half=half, offset=0,
                                     device=device)
        dmax = fx["dmax"]
        # BODY-FITTED GUARD (anti-vacuity): the rung claims strong Dirichlet on
        # an EXACT body-fitted mesh — assert dmax==0 (no SBM shift active).
        assert dmax == 0, (f"rung A requires an EXACT body-fitted carve "
                           f"(dmax==0); got dmax={dmax} — pick an aligned half "
                           f"(k/2^level).")
        print(f" body-fitted guard: dmax={dmax}  (OK, exact)  "
              f"n_fluid_cells={fx['n_fluid_cells']}  "
              f"n_obstacle_nodes={int(fx['obstacle_node_mask'].sum())}",
              flush=True)

        pr = march_projection(fx, dt=dt, nsteps=nsteps, rate_tol=rate_tol_p,
                              log_every=log_every)
        mo = march_monolithic(fx, dt=dt, nsteps=nsteps, rate_tol=rate_tol_m,
                              log_every=log_every)

        cd_rel = (abs(pr["cd"] - mo["cd"]) / abs(mo["cd"])
                  if mo["cd"] != 0 else float("inf"))
        mu_rel = (abs(pr["mean_u"] - mo["mean_u"]) / abs(mo["mean_u"])
                  if mo["mean_u"] != 0 else float("inf"))
        mu_frac = pr["mean_u"] / mo["mean_u"] if mo["mean_u"] != 0 else 0.0
        weak_pin = mu_frac < WEAK_PLATEAU_FRAC

        blew = pr.get("blew_up", False)
        if steady:
            # a projection blow-up is itself a decisive FAIL (base single-pass is
            # pressure-unstable on the open outflow), independent of Cd/mean|u|.
            cd_ok = (not blew) and cd_rel < TOL_CD_REL
            mu_ok = (not blew) and (mu_rel < TOL_MU_REL) and (not weak_pin)
            verdict = cd_ok and mu_ok
            print(f"\n --- Re={Re} STEADY verdict ---")
            if blew:
                print(f"  projection : BLEW UP at step {pr['steps']} "
                      f"(‖p‖={pr['pnorm']:.3e}, ‖div‖={pr['div']:.3e}) — base "
                      f"single-pass is pressure-unstable on the open outflow")
            print(f"  projection : Cd={pr['cd']:+.4f}  mean|u|={pr['mean_u']:.4f}"
                  f"  ‖div‖={pr['div']:.3e}  ‖p‖={pr['pnorm']:.3e}  "
                  f"steps={pr['steps']}")
            print(f"  monolithic : Cd={mo['cd']:+.4f}  mean|u|={mo['mean_u']:.4f}"
                  f"  ‖div‖={mo['div']:.3e}  steps={mo['steps']}")
            print(f"  Cd rel-diff = {cd_rel:.3%} (tol {TOL_CD_REL:.0%})  "
                  f"-> {'OK' if cd_ok else 'FAIL'}")
            print(f"  mean|u| proj/mono = {mu_frac:.3f}  rel-diff={mu_rel:.3%} "
                  f"(tol {TOL_MU_REL:.0%})  weak_pin={weak_pin}  "
                  f"-> {'OK' if mu_ok else 'FAIL'}")
        else:
            # shedding: mean Cd over the tail (>=1 period) + Strouhal from Cl.
            def tail_mean(h):
                a = np.asarray(h)
                return float(a[int(0.4 * len(a)):].mean())
            pr_cdm = tail_mean(pr["cd_hist"])
            mo_cdm = tail_mean(mo["cd_hist"])
            cd_rel = (abs(pr_cdm - mo_cdm) / abs(mo_cdm)
                      if mo_cdm != 0 else float("inf"))
            st_p, fp_p = strouhal(pr["cl_hist"], dt, D)
            st_m, fm_m = strouhal(mo["cl_hist"], dt, D)
            cd_ok = (not blew) and cd_rel < TOL_CD_REL
            mu_ok = (not blew) and (mu_rel < TOL_MU_REL) and (not weak_pin)
            verdict = cd_ok and mu_ok
            pr["cd_mean_period"] = pr_cdm
            mo["cd_mean_period"] = mo_cdm
            pr["St"], mo["St"] = st_p, st_m
            print(f"\n --- Re={Re} SHEDDING verdict ---")
            if blew:
                print(f"  projection : BLEW UP at step {pr['steps']} — base "
                      f"single-pass is pressure-unstable on the open outflow")
            print(f"  projection : mean Cd={pr_cdm:+.4f}  St={st_p:.4f}  "
                  f"mean|u|={pr['mean_u']:.4f}  steps={pr['steps']}")
            print(f"  monolithic : mean Cd={mo_cdm:+.4f}  St={st_m:.4f}  "
                  f"mean|u|={mo['mean_u']:.4f}  steps={mo['steps']}")
            print(f"  mean-Cd rel-diff = {cd_rel:.3%} (tol {TOL_CD_REL:.0%})  "
                  f"-> {'OK' if cd_ok else 'FAIL'}")
            print(f"  mean|u| proj/mono = {mu_frac:.3f}  weak_pin={weak_pin}  "
                  f"-> {'OK' if mu_ok else 'FAIL'}")

        all_pass = all_pass and verdict
        print(f" ===> Re={Re}  {'PASS' if verdict else 'FAIL'}", flush=True)
        results[Re] = dict(
            cd_proj=pr["cd"], cd_mono=mo["cd"], cd_rel=cd_rel,
            mean_u_proj=pr["mean_u"], mean_u_mono=mo["mean_u"],
            mu_frac=mu_frac, weak_pin=weak_pin,
            div_proj=pr["div"], div_mono=mo["div"],
            steps_proj=pr["steps"], steps_mono=mo["steps"],
            St_proj=pr.get("St"), St_mono=mo.get("St"),
            cd_mean_period_proj=pr.get("cd_mean_period"),
            cd_mean_period_mono=mo.get("cd_mean_period"),
            pnorm_proj=pr["pnorm"], blew_up=blew,
            dmax=dmax, verdict=verdict)

    print(f"\n{'#'*70}\n RUNG A VERDICT: "
          f"{'PASS — strong-Dirichlet body-fitted projection matches the same-mesh monolithic (projection is structurally sound)' if all_pass else 'FAIL — single-pass projection does NOT match the same-mesh monolithic on the open outflow (base: pressure-unstable ‖p‖ blow-up; PSPG-stabilized: pins ~few%% of monolithic = the lagged-p* from-rest weak fixed point). Fix = stabilized inner predictor<->PPE iteration (Task 4), which re-runs this rung.'}"
          f"\n{'#'*70}", flush=True)
    results["all_pass"] = all_pass
    return results


if __name__ == "__main__":
    import json
    import sys
    lvl = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    out = run_rungA(level=lvl)
    print("\n[json]", json.dumps(out, default=lambda o: None))
