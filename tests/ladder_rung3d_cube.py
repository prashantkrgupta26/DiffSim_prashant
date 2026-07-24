"""Projection Validation Ladder — Task 7: the 3-D CUBE confirmation (rungs A'/C').

The 2-D ladder is COMPLETE: the consistent-projection + SBM scheme is faithful to
the same-mesh monolithic across projection (rung A), weak Nitsche (rung B), and the
genuine SBM shift (rung C). Task 7 confirms the SAME working scheme lifts to 3-D —
the original sphere regime that started the investigation — using the exact 2-D
config UNCHANGED (this is a confirmation, not a re-tune):

    consistent_projection=True + rot_pin_wall=True (wall pressure) + grad-div
    gamma=50 + backflow beta=0.5 + rotational_pin_outflow (via the base stepper).

Two 3-D rungs, both proj-vs-mono on the SAME cube-in-channel mesh (cuDSS on GPU):

  * RUNG A' — body-fitted cube, STRONG no-slip.  `build_cube_channel_3d(offset=0)`
    => the aligned Box carve is EXACT (`dmax==0`).  The obstacle nodes join the
    STRONG Dirichlet set (no-slip); the SBM block is inert (d=0).  This is the
    3-D analogue of rung A: does the base consistent-projection split reproduce
    the same-mesh monolithic Cd + mean|u| in 3-D?  BODY-FITTED guard: dmax==0.

  * RUNG C' — OFFSET cube, GENUINE SBM SHIFT.  `build_cube_channel_3d(offset=d!=0)`
    => the grid-aligned surrogate no longer coincides with the true Box face:
    `0 < dmax < h`.  Weak Nitsche + the Taylor shift `(grad N).d` + area
    correction `geo.corr`, projection vs the same-shifted-mesh monolithic.  This
    is the 3-D analogue of rung C.  SHIFT-ACTIVE guard: dmax>0.  Anti-vacuity:
    zeroing the shift (`geo.d=0`, `geo.corr=1`) must move the projection off the
    true shifted oracle (the shift is load-bearing).

Drag (3-D): `surrogate_traction` gives the force vector F on the immersed cube;
`Cd = F_x / qref3d`, `qref3d = 0.5 U_IN^2 A`, A = D^2 = (2*half)^2 is the cube's
frontal AREA (the 3-D analogue of the 2-D frontal LENGTH D).

Solver: BOTH projection and monolithic run cuDSS on the GPU (`solver="cudss"`,
`device="cuda:0"`).  3-D at L4/L5 is where the direct-GPU solve pays off (splu
stalls near the L5 sphere regime, ~142k saddle DOF).

Reuse: the projection marcher is rung B's `march_projection` VERBATIM (dim-generic,
routes its solves through `solver=`), so A'/C' exercise the identical stepper as the
2-D ladder.  The monolithic here is a 3-D-capable, cuDSS-routed twin of rung B's
`march_monolithic` (which hardcodes host splu) — same SBM Nitsche block, same
grad-div / backflow / boundary-vorticity blocks, same strong-BC + single p-pin.

Run (gpubox GPU0):
    CUDA_VISIBLE_DEVICES=0 \
    LD_LIBRARY_PATH=/usr/lib/wsl/lib:$PWD/.venv/lib/python3.12/site-packages/nvidia/cu12/lib \
    PYTHONPATH=src:tests DEVICE=cuda:0 PROJ_SOLVER=cudss \
    .venv/bin/python tests/ladder_rung3d_cube.py 4          # level 4
"""
import dataclasses

import numpy as np
import scipy.sparse as sp

from diffsim.sbm.vector import sbm_vector_dirichlet, surrogate_traction
from diffsim.api.ns_bricks import (assemble_linear_ns, assemble_backflow_block,
                                   assemble_bvs_block, outflow_faces)
from diffsim.physics.poisson import gauss_points
from diffsim.solvers.linsolve import solve_linear
from diffsim.steppers.leray import LerayProjectionStepper

from ladder_fixtures import build_cube_channel_3d, U_IN
# The projection marcher is rung B's, VERBATIM — dim-generic, `solver=`-routed.
# A'/C' therefore exercise the identical consistent-projection + weak-Nitsche
# stepper as the 2-D ladder (the only 3-D input is the fixture's dim=3 geo).
from ladder_rungB_square_nitsche import (
    march_projection, build_box_strong_bc, ALPHA, FN1_GRADDIV_GAMMA,
    TOL_CD_REL, TOL_MU_REL, WEAK_PLATEAU_FRAC)
from ladder_rungA_square_strong import mean_speed


# --------------------------------------------------------------------------
# 3-D drag reference: frontal AREA (D^2), not length
# --------------------------------------------------------------------------
def qref3d(fx):
    """Dynamic-pressure reference for the 3-D cube: 0.5 U_IN^2 A, with the
    frontal AREA A = D^2 = (2*half)^2 (the 3-D analogue of the 2-D frontal
    length D).  Cd = F_x / qref3d."""
    D = 2.0 * fx["half"]
    return 0.5 * U_IN ** 2 * D ** 2


# --------------------------------------------------------------------------
# shift-zeroing (anti-vacuity for rung C': drop Taylor + area-correction)
# --------------------------------------------------------------------------
def zero_shift(fx):
    """COPY of the fixture with the SBM shift ZEROED (`geo.d=0`, `geo.corr=1`).
    Everything else identical; the ONLY difference is the shift.  Used for the
    rung-C' load-bearing check: projection with this geo vs the TRUE shifted
    oracle must FAIL (the shift is doing real work at d!=0)."""
    geo = fx["geo"]
    geo0 = dataclasses.replace(
        geo, d=np.zeros_like(geo.d), corr=np.ones_like(geo.corr))
    fx0 = dict(fx)
    fx0["geo"] = geo0
    fx0["dmax"] = 0.0
    return fx0


# --------------------------------------------------------------------------
# RUNG A' projection: base consistent-projection split, STRONG obstacle
# --------------------------------------------------------------------------
def march_projection_strong_3d(fx, dt, nsteps, rate_tol, log_every, solver,
                               device, alpha=ALPHA,
                               graddiv_gamma=FN1_GRADDIV_GAMMA,
                               rotational_pin_wall=False):
    """Rung A' projection: the base LerayProjectionStepper in consistent mode
    with the cube nodes in the STRONG Dirichlet set (strong no-slip, dmax==0 —
    the SBM block is inert).  Same consistent-projection config as the 2-D
    ladder (grad-div gamma=50, rotational + rotational_pin_outflow via the base
    stepper).  Routes its solves through `solver` (cuDSS on GPU)."""
    dim = fx["dim"]
    dm = fx["dm"]
    ndof, nu = fx["ndof"], fx["nu"]
    # STRONG set = box (inflow + walls) UNION the obstacle nodes (no-slip=0).
    box_nodes, box_g = build_box_strong_bc(fx)
    obst = np.where(fx["obstacle_node_mask"])[0]
    strong_nodes = np.unique(np.concatenate([box_nodes, obst]))
    g_map = {int(i): np.zeros(dim) for i in strong_nodes}
    for k, i in enumerate(box_nodes):
        g_map[int(i)] = box_g[k]
    g_strong = np.array([g_map[int(i)] for i in strong_nodes])

    q = qref3d(fx)

    def f_fn(x, t):
        return np.zeros((len(x), dim))

    def g_fn(coords_at_dir, t):
        return g_strong

    st = LerayProjectionStepper(
        dm, nu, dt, f_fn=f_fn, g_fn=g_fn, order=2, picard_iters=1,
        solver=solver, pressure_outflow_nodes=fx["outflow_nodes"],
        consistent_projection=True, velocity_update="consistent",
        graddiv_gamma=(graddiv_gamma if graddiv_gamma else None),
        rotational_pin_wall=rotational_pin_wall)   # P2harden: strong-wall pin
    st.dir_nodes = strong_nodes
    st.set_initial(lambda c: np.zeros((len(c), dim)))
    # STRONG-WALL rotational pin (P2harden): pin q=0 on the obstacle SUBSET only
    # (NOT box inflow/walls). None when off => step() bit-for-bit.
    wall_pin_nodes = obst if rotational_pin_wall else None

    T = dm.constraints.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    sf, geo = fx["sf"], fx["geo"]

    prev = None
    cd = cl = np.nan
    cd_hist, mu_hist = [], []
    u = None
    blew_up = False
    steps = 0
    for steps in range(1, nsteps + 1):
        u, p = st.step(wall_pin_nodes=wall_pin_nodes)
        if not np.isfinite(u).all() or not np.isfinite(p).all() \
                or np.abs(u).max() > 1e4:
            blew_up = True
            if log_every:
                print(f"[A' proj] BLOW-UP step{steps} "
                      f"max|u|={np.abs(u).max():.3e}", flush=True)
            break
        xfree = np.zeros(st.n_free * ndof)
        xv = xfree.reshape(st.n_free, ndof)
        xv[:, :dim] = u
        xv[:, dim] = p
        F = surrogate_traction(dm, sf, geo, np.asarray(T_vec @ xfree), nu, ndof)
        cd = float(F[0] / q)
        mu = mean_speed(u)
        cd_hist.append(cd)
        mu_hist.append(mu)
        if log_every and (steps <= 3 or steps % log_every == 0):
            print(f"[A' proj] step{steps:4d}  Cd={cd:+.5f}  mean|u|={mu:.4f}",
                  flush=True)
        if rate_tol is not None and prev is not None:
            if np.abs(u - prev).max() / dt < rate_tol:
                break
        prev = u.copy()
    div = float(st.divergence_l2())
    return dict(cd=cd, mean_u=mean_speed(u), div=div, steps=steps,
                cd_hist=cd_hist, mu_hist=mu_hist,
                pnorm=float(np.linalg.norm(st.p_star)), blew_up=blew_up)


# --------------------------------------------------------------------------
# 3-D monolithic (same-mesh saddle oracle) — cuDSS-routed twin of rung B's
# --------------------------------------------------------------------------
# boundary_vorticity defaults OFF in 3-D: the P1 boundary-vorticity outflow
# term (VMS change #5) is implemented 2-D-only (curl u scalar), so it is absent
# on BOTH the projection PPE source AND this monolithic C-block in 3-D — the
# same-mesh oracle stays exact because NEITHER side carries it (Task 7 finding).
def march_monolithic_3d(fx, dt, nsteps, rate_tol, log_every, solver, device,
                        strong_obstacle, alpha=ALPHA,
                        graddiv_gamma=FN1_GRADDIV_GAMMA,
                        backflow_beta=0.5, boundary_vorticity=False):
    """Same-mesh monolithic saddle NS in 3-D (the oracle).  Mirrors rung B's
    `march_monolithic` EXACTLY (same SBM Nitsche block, same grad-div /
    backflow / boundary-vorticity blocks, same strong-BC + single outflow
    p-pin) but routes the per-step saddle solve through `solve_linear(solver=)`
    so it can use cuDSS on GPU (rung B's twin hardcodes host splu).

    `strong_obstacle` (rung A'): the cube nodes join the STRONG Dirichlet set
    and the SBM Nitsche block is DROPPED (strong no-slip — matches the A'
    projection).  False (rung C'): weak Nitsche + shift on the surrogate faces
    (matches the C' projection)."""
    dim = fx["dim"]
    dm, mesh, cons = fx["dm"], fx["mesh"], fx["cons"]
    sf, geo = fx["sf"], fx["geo"]
    ndof, nu = fx["ndof"], fx["nu"]
    coords = fx["coords"]

    box_nodes, box_g = build_box_strong_bc(fx)
    if strong_obstacle:
        obst = np.where(fx["obstacle_node_mask"])[0]
        strong_nodes = np.unique(np.concatenate([box_nodes, obst]))
        g_map = {int(i): np.zeros(dim) for i in strong_nodes}
        for k, i in enumerate(box_nodes):
            g_map[int(i)] = box_g[k]
        g_strong = np.array([g_map[int(i)] for i in strong_nodes])
    else:
        strong_nodes, g_strong = box_nodes, box_g

    T = cons.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    nfree = T.shape[1]
    xq = gauss_points(mesh, dm.tables_by_p)

    # SBM Nitsche block — only for WEAK obstacle (rung C'); dropped when strong.
    if not strong_obstacle:
        Af, bf = sbm_vector_dirichlet(
            dm, sf, geo, lambda y: np.zeros((len(y), dim)), nu, ndof,
            alpha=alpha)
        Af_c = (T_vec.T @ Af @ T_vec).tocsr()
        bf_c = np.asarray(T_vec.T @ bf)
    else:
        Af_c = None
        bf_c = np.zeros(nfree * ndof)

    # same grad-div (LSIC) block as the projection predictor (via the stepper)
    gd_block = None
    if graddiv_gamma:
        _st = LerayProjectionStepper(
            dm, nu, dt, f_fn=lambda x, t: np.zeros((len(x), dim)),
            g_fn=lambda c, t: None, order=1, graddiv_gamma=graddiv_gamma)
        gd_block = _st._graddiv_gamma_block

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

    p_pin = int(np.argmax(coords[:, 0]))
    q = qref3d(fx)
    x = np.zeros(nfree * ndof)
    sigma = 1.0 / dt
    bf_faces = (outflow_faces(mesh)
                if (backflow_beta != 0.0 or boundary_vorticity) else None)
    prev_u = None
    steps = 0
    cd = np.nan
    cd_hist = []
    u_new = None
    cache = {}
    for step in range(nsteps):
        u_node = x.reshape(nfree, ndof)[:, :dim]
        aq, dq = gp_field(u_node)
        fq = {pv: aq[pv] / dt for pv in xq}
        A, b = assemble_linear_ns(dm, aq, dq, fq, nu, sigma=sigma)
        if Af_c is not None:
            A = A + Af_c
            b = b + bf_c
        if gd_block is not None:
            A = A + gd_block
        if backflow_beta != 0.0:
            A = A + assemble_backflow_block(dm, u_node, backflow_beta, ndof,
                                            faces=bf_faces)
        if boundary_vorticity:
            A = A + assemble_bvs_block(dm, u_node, nu, dt, ndof, faces=bf_faces)
        A = A.tolil()
        for k, i in enumerate(strong_nodes):
            for c in range(dim):
                r = int(i) * ndof + c
                A.rows[r] = [r]
                A.data[r] = [1.0]
                b[r] = g_strong[k, c]
        pin = p_pin * ndof + dim
        A.rows[pin] = [pin]
        A.data[pin] = [1.0]
        b[pin] = 0.0
        Acsr = A.tocsr()
        if solver == "splu":
            from scipy.sparse.linalg import splu
            x = splu(Acsr.tocsc()).solve(b)
        else:
            # NON-constant matrix (Picard-updated convection) => no cache_key.
            x = solve_linear(Acsr, b, solver=solver, sym=False, device=device)
        u_new = x.reshape(nfree, ndof)[:, :dim]
        F = surrogate_traction(dm, sf, geo, np.asarray(T_vec @ x), nu, ndof)
        cd = float(F[0] / q)
        cd_hist.append(cd)
        steps = step + 1
        if log_every and (steps <= 3 or steps % log_every == 0):
            print(f"[mono3d] step{steps:4d}  Cd={cd:+.5f}  "
                  f"mean|u|={mean_speed(u_new):.4f}", flush=True)
        if rate_tol is not None and prev_u is not None and step > 5:
            if np.abs(u_new - prev_u).max() / dt < rate_tol:
                break
        prev_u = u_new.copy()

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
    return dict(cd=cd, mean_u=mean_speed(u_new), div=div, steps=steps,
                cd_hist=cd_hist, p_pin=p_pin)


# --------------------------------------------------------------------------
# rung A' — body-fitted cube, strong no-slip
# --------------------------------------------------------------------------
def run_rungA3d(level=4, half=0.125, Re=40, device="cuda:0", solver="cudss",
                dt=0.02, nsteps=600, log_every=25, rate_tol=5e-4, alpha=ALPHA):
    """Rung A': body-fitted cube (offset=0, dmax==0), STRONG no-slip.
    Consistent-projection split vs same-mesh monolithic (both cuDSS/GPU)."""
    print(f"\n{'='*70}\n Rung A' — BODY-FITTED cube STRONG no-slip  Re={Re}  "
          f"level={level}  half={half} (D={2*half})  solver={solver}\n"
          f"{'='*70}", flush=True)
    fx = build_cube_channel_3d(level, Re, half=half, offset=0.0, device=device)
    dmax = fx["dmax"]
    # BODY-FITTED guard (anti-vacuity): dmax==0 — the SBM shift is inert.
    assert dmax == 0, (f"rung A' requires an EXACT body-fitted carve "
                       f"(dmax==0); got dmax={dmax}.")
    print(f" body-fitted guard: dmax={dmax}  saddle_dof="
          f"{fx['cons'].free_nodes.shape[0]*fx['ndof']}  "
          f"n_fluid_cells={fx['n_fluid_cells']}  "
          f"n_obstacle_nodes={int(fx['obstacle_node_mask'].sum())}", flush=True)

    pr = march_projection_strong_3d(fx, dt=dt, nsteps=nsteps, rate_tol=rate_tol,
                                    log_every=log_every, solver=solver,
                                    device=device, alpha=alpha)
    mo = march_monolithic_3d(fx, dt=dt, nsteps=nsteps, rate_tol=2e-4,
                             log_every=log_every, solver=solver, device=device,
                             strong_obstacle=True, alpha=alpha)
    return _verdict("A'", Re, level, dmax, pr, mo, half)


# --------------------------------------------------------------------------
# rung C' — offset cube, genuine SBM shift
# --------------------------------------------------------------------------
def run_rungC3d(level=4, half=0.125, offset=0.05, Re=40, device="cuda:0",
                solver="cudss", dt=0.02, nsteps=600, log_every=25,
                rate_tol=5e-4, alpha=ALPHA, check_load_bearing=False):
    """Rung C': OFFSET cube (dmax>0), weak Nitsche + genuine SBM shift.
    Consistent-projection split vs same-shifted-mesh monolithic (both cuDSS).
    `check_load_bearing`: also run the shift-zeroed projection and confirm it
    moves OFF the true shifted oracle (the shift is load-bearing)."""
    print(f"\n{'='*70}\n Rung C' — OFFSET cube SBM SHIFT  Re={Re}  level={level}"
          f"  half={half} (D={2*half})  offset={offset}  solver={solver}\n"
          f"{'='*70}", flush=True)
    fx = build_cube_channel_3d(level, Re, half=half, offset=offset,
                               device=device)
    dmax = fx["dmax"]
    h = 1.0 / 2 ** level
    # SHIFT-ACTIVE guard (anti-vacuity, the OPPOSITE of A'): 0 < dmax < h.
    assert dmax > 0, (f"rung C' requires a GENUINE SBM shift (dmax>0); got "
                      f"dmax={dmax} (offset={offset} did not break alignment).")
    assert dmax < h, (f"rung C' wants a SUB-CELL shift (dmax<h={h}); got "
                      f"dmax={dmax} — offset too large.")
    corr = np.asarray(fx["geo"].corr)
    n_corr = int((np.abs(corr - 1.0) > 1e-9).sum())
    print(f" shift-active guard: dmax={dmax:.5f} (0<dmax<h={h:.5f}, "
          f"dmax/h={dmax/h:.3f})  n_area_corrected_gp={n_corr}  saddle_dof="
          f"{fx['cons'].free_nodes.shape[0]*fx['ndof']}  "
          f"n_obstacle_nodes={int(fx['obstacle_node_mask'].sum())}", flush=True)

    # projection: rung B's marcher VERBATIM (weak Nitsche + rot_pin_wall).
    pr = march_projection(fx, dt=dt, nsteps=nsteps, rate_tol=rate_tol,
                          log_every=log_every, alpha=alpha, solver=solver,
                          rot_pin_wall=True)
    mo = march_monolithic_3d(fx, dt=dt, nsteps=nsteps, rate_tol=2e-4,
                             log_every=log_every, solver=solver, device=device,
                             strong_obstacle=False, alpha=alpha)
    # rung-B projection returns Cd from qref (2-D length); recompute on 3-D area.
    pr["cd"] = _recd(pr["cd"], fx, half)
    mo["cd"] = mo["cd"]        # 3-D monolithic already uses qref3d
    res = _verdict("C'", Re, level, dmax, pr, mo, half)

    if check_load_bearing:
        fx0 = zero_shift(fx)
        pr0 = march_projection(fx0, dt=dt, nsteps=nsteps, rate_tol=rate_tol,
                               log_every=0, alpha=alpha, solver=solver,
                               rot_pin_wall=True)
        cd0 = _recd(pr0["cd"], fx, half)
        drift = (abs(cd0 - mo["cd"]) / abs(mo["cd"])
                 if mo["cd"] != 0 else float("inf"))
        load_bearing = drift > TOL_CD_REL
        print(f"  [load-bearing] shift-zeroed proj Cd={cd0:+.5f} vs TRUE "
              f"shifted mono Cd={mo['cd']:+.5f}  drift={drift:.3%} "
              f"(> tol {TOL_CD_REL:.0%} => shift is load-bearing) "
              f"-> {'OK' if load_bearing else 'FAIL'}", flush=True)
        res["load_bearing"] = load_bearing
        res["cd_shift_zeroed"] = cd0
        res["shift_drift"] = drift
    return res


def _recd(cd_from_length_qref, fx, half):
    """Rung B's `march_projection` computes Cd with the 2-D length qref
    (0.5 U^2 D).  Rescale to the 3-D area qref (0.5 U^2 D^2): divide by D."""
    D = 2.0 * half
    return cd_from_length_qref / D


# --------------------------------------------------------------------------
# shared verdict
# --------------------------------------------------------------------------
def _verdict(tag, Re, level, dmax, pr, mo, half):
    cd_rel = (abs(pr["cd"] - mo["cd"]) / abs(mo["cd"])
              if mo["cd"] != 0 else float("inf"))
    mu_rel = (abs(pr["mean_u"] - mo["mean_u"]) / abs(mo["mean_u"])
              if mo["mean_u"] != 0 else float("inf"))
    mu_frac = pr["mean_u"] / mo["mean_u"] if mo["mean_u"] != 0 else 0.0
    weak_pin = mu_frac < WEAK_PLATEAU_FRAC
    blew = pr.get("blew_up", False)
    cd_ok = (not blew) and cd_rel < TOL_CD_REL
    mu_ok = (not blew) and (mu_rel < TOL_MU_REL) and (not weak_pin)
    verdict = cd_ok and mu_ok
    print(f"\n --- Rung {tag}  Re={Re}  STEADY verdict ---")
    print(f"  projection : Cd={pr['cd']:+.5f}  mean|u|={pr['mean_u']:.4f}  "
          f"‖div‖={pr['div']:.3e}  steps={pr['steps']}")
    print(f"  monolithic : Cd={mo['cd']:+.5f}  mean|u|={mo['mean_u']:.4f}  "
          f"‖div‖={mo['div']:.3e}  steps={mo['steps']}")
    print(f"  Cd rel-diff = {cd_rel:.3%} (tol {TOL_CD_REL:.0%})  "
          f"-> {'OK' if cd_ok else 'FAIL'}")
    print(f"  mean|u| proj/mono = {mu_frac:.3f}  rel-diff={mu_rel:.3%} "
          f"(tol {TOL_MU_REL:.0%})  weak_pin={weak_pin}  "
          f"-> {'OK' if mu_ok else 'FAIL'}")
    print(f" ===> Rung {tag}  Re={Re}  {'PASS' if verdict else 'FAIL'}",
          flush=True)
    return dict(tag=tag, Re=Re, level=level, dmax=dmax,
                cd_proj=pr["cd"], cd_mono=mo["cd"], cd_rel=cd_rel,
                mean_u_proj=pr["mean_u"], mean_u_mono=mo["mean_u"],
                mu_frac=mu_frac, weak_pin=weak_pin, mu_rel=mu_rel,
                div_proj=pr["div"], div_mono=mo["div"],
                steps_proj=pr["steps"], steps_mono=mo["steps"],
                blew_up=blew, verdict=verdict)


def run_task7(level=4, half=0.125, offset=0.05, Re=40, device="cuda:0",
              solver="cudss", nsteps=600, log_every=25,
              check_load_bearing=True):
    """Task 7: rung A' + rung C' — the 3-D cube confirmation."""
    a = run_rungA3d(level=level, half=half, Re=Re, device=device,
                    solver=solver, nsteps=nsteps, log_every=log_every)
    c = run_rungC3d(level=level, half=half, offset=offset, Re=Re,
                    device=device, solver=solver, nsteps=nsteps,
                    log_every=log_every, check_load_bearing=check_load_bearing)
    all_pass = a["verdict"] and c["verdict"]
    print(f"\n{'#'*70}\n TASK 7 (3-D CUBE) VERDICT: "
          f"{'PASS — the consistent-projection + SBM scheme is FAITHFUL in 3-D: body-fitted (A dmax=0) AND genuine SBM shift (C dmax>0) both match the same-mesh monolithic Cd. The 2-D conclusion holds in 3-D (the sphere regime).' if all_pass else 'FAIL — the 3-D confirmation does NOT hold; see per-rung numbers (report whether it is a 3-D-specific issue).'}"
          f"\n{'#'*70}", flush=True)
    return dict(rungA=a, rungC=c, all_pass=all_pass)


if __name__ == "__main__":
    import json
    import os
    import sys
    lvl = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    solver = os.environ.get("PROJ_SOLVER", "cudss")
    device = os.environ.get("DEVICE", "cuda:0" if solver == "cudss" else "cpu")
    nsteps = int(os.environ.get("NSTEPS", "600"))
    out = run_task7(level=lvl, device=device, solver=solver, nsteps=nsteps)
    print("\n[json]", json.dumps(out, default=lambda o: None))
