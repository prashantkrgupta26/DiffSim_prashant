"""Vortex-shedding characterization — consistent-projection + weak Nitsche.

GOAL (characterization, NOT a stepper change): drive a genuine, self-sustaining
von Karman street with the rung-B-validated consistent-projection scheme
(`consistent_projection=True` + `rotational_pin_wall=True` + gamma=50 grad-div +
backflow beta=0.5 + `rotational_pin_outflow`), then extract the Strouhal number.

Why a new driver (the ladder Re=100 goes STEADY): an earlier finding showed the
scheme preserves y-symmetry BIT-EXACTLY from symmetric rest, so on the coarse,
confined ladder mesh the wake never sheds. Two ingredients fix that, neither of
which touches the stepper numerics:

  1. A SHEDDING-CAPABLE mesh — thinner obstacle (`half=0.0625`, D=0.125 =>
     blockage 12.5% vs the ladder's 25%) at a finer level (L6/L7) so the wake
     Reynolds number is genuinely supercritical (square-cylinder onset ~Re
     50-60). Re=100 (and 150) is well past onset.

  2. A SYMMETRY-BREAKING TRIGGER — a transient TRANSVERSE INFLOW TILT applied
     ONLY through the box strong-Dirichlet `g_fn` for the first `trigger_steps`
     steps, then removed. The inflow carries a small `v = eps*U_IN` component
     during the window; after it, inflow reverts to pure axial `U_IN`. This is
     a boundary perturbation stamped by the existing `u_new[dir_nodes]=gvals`
     overwrite — the stepper's numerics are untouched. Once the wake instability
     grows on the trigger seed it self-sustains after the trigger is off.

This reuses the rung-B machinery verbatim (`build_sbm_block`,
`build_correction_penalty`, backflow increment, `surrogate_traction`,
`strouhal`, `qref`) — the ONLY delta from `march_projection` is the
time-dependent trigger `g_fn` and the long march. The monolithic comparison
(same mesh, same trigger) reuses `march_monolithic`'s assembly but with the
transient inflow tilt on its strong rows.

Run (gpubox GPU1):
    CUDA_VISIBLE_DEVICES=1 PROJ_SOLVER=splu PYTHONPATH=src:tests \
        .venv/bin/python tests/ladder_shedding_square.py L6 100
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from diffsim.steppers.leray import LerayProjectionStepper
from diffsim.sbm.vector import (sbm_vector_dirichlet, surrogate_traction)
from diffsim.api.ns_bricks import (assemble_linear_ns, assemble_backflow_block,
                                   assemble_bvs_block, outflow_faces)
from diffsim.physics.poisson import gauss_points

from ladder_fixtures import build_square_channel_2d, U_IN
from ladder_rungA_square_strong import qref, mean_speed, strouhal
from ladder_rungB_square_nitsche import (build_box_strong_bc, build_sbm_block,
                                         build_correction_penalty, _bf_conn,
                                         _a_face, ALPHA, FN1_GRADDIV_GAMMA)


# --------------------------------------------------------------------------
# transient transverse inflow tilt (the symmetry-breaking trigger)
# --------------------------------------------------------------------------
def make_trigger_gfn(fx, g_strong, dt, trigger_steps, eps, ramp_steps=0):
    """Build a time-dependent `g_fn(coords_at_dir, t)` that adds a small
    transverse `v = eps*U_IN` to the INFLOW strong nodes for t < trigger_steps*dt,
    then reverts to the baseline (`g_strong`). The inflow rows are identified by
    matching coords to `fx['inflow_mask']` in the strong-node ordering.

    `ramp_steps > 0` (2026-07-23) SMOOTHLY ramps the AXIAL inflow from 0 to U_IN
    over the first `ramp_steps` steps (a raised-cosine start-up), instead of the
    impulsive `U_IN` at t=0. The impulsive start on the elongated mesh drove a
    violent Cd swing (+10 -> -6 -> +6 in three steps); the ramp removes that
    startup shock (a pure boundary perturbation — stepper numerics untouched).
    The transverse trigger is also ramped in over the SAME window and out again
    at `trigger_steps` (a smooth raised-cosine pulse), a gentler seed.

    Returns (g_fn, trigger_t_end). The trigger is a pure boundary perturbation:
    it enters ONLY through the existing `u_new[dir_nodes]=gvals` overwrite."""
    strong_nodes, _ = build_box_strong_bc(fx)
    inflow_mask_free = fx["inflow_mask"]              # over ALL free nodes
    # which of the strong_nodes are inflow nodes (in strong-node ordering):
    is_inflow_strong = inflow_mask_free[strong_nodes]
    t_end = trigger_steps * dt
    t_ramp = ramp_steps * dt

    def g_fn(coords_at_dir, t):
        g = g_strong.copy()
        # smooth raised-cosine axial ramp 0 -> U_IN over [0, t_ramp]
        if t_ramp > 0 and t < t_ramp - 1e-12:
            s = 0.5 * (1.0 - np.cos(np.pi * t / t_ramp))    # 0 -> 1
            g[is_inflow_strong, 0] *= s
        if t < t_end - 1e-12:
            # transverse tilt: raised-cosine pulse over [0, t_end] (peak mid-way)
            if t_end > 0:
                w = np.sin(np.pi * t / t_end) ** 2          # 0 at ends, 1 mid
            else:
                w = 1.0
            g[is_inflow_strong, 1] = eps * U_IN * w
        return g

    return g_fn, t_end


# --------------------------------------------------------------------------
# PROJECTION shedding march (base march_projection + transient trigger)
# --------------------------------------------------------------------------
def shed_projection(fx, dt, nsteps, trigger_steps=200, eps=0.05,
                    alpha=ALPHA, solver="splu", beta_backflow=0.5,
                    graddiv_gamma=FN1_GRADDIV_GAMMA, log_every=200,
                    ramp_steps=0):
    """Consistent-projection + weak-Nitsche long march with the transient
    inflow-tilt trigger. Mirrors `ladder_rungB_square_nitsche.march_projection`
    exactly (rot_pin_wall=True, correction re-pin on) except `g_fn` is the
    time-dependent trigger. Records Cd(t), Cl(t), mean|u|(t)."""
    dim = fx["dim"]
    dm = fx["dm"]
    ndof, nu = fx["ndof"], fx["nu"]
    strong_nodes, g_strong = build_box_strong_bc(fx)
    Af_c, bf_c, T_vec, sbm_nodes = build_sbm_block(fx, alpha=alpha,
                                                   beta_backflow=0.0)
    bf_pv, bf_ftab, bf_conn = _bf_conn(fx)
    correction_penalty = build_correction_penalty(fx, alpha=alpha)
    g_fn, t_end = make_trigger_gfn(fx, g_strong, dt, trigger_steps, eps,
                                   ramp_steps=ramp_steps)

    st = LerayProjectionStepper(
        dm, nu, dt, f_fn=lambda x, t: np.zeros((len(x), dim)), g_fn=g_fn,
        order=2, picard_iters=1, solver=solver,
        pressure_outflow_nodes=fx["outflow_nodes"],
        consistent_projection=True, velocity_update="consistent",
        graddiv_gamma=(graddiv_gamma if graddiv_gamma else None),
        rotational_pin_wall=True)
    st.dir_nodes = strong_nodes
    st.set_initial(lambda c: np.zeros((len(c), dim)))

    def extra_block(u_free):
        A, b = Af_c, bf_c
        if beta_backflow != 0.0 and u_free is not None and np.any(u_free):
            a_face = _a_face(fx, u_free, bf_ftab, bf_conn)
            if np.any(a_face):
                Af_bf, _ = sbm_vector_dirichlet(
                    dm, fx["sf"], fx["geo"],
                    lambda y: np.zeros((len(y), dim)), nu, ndof,
                    alpha=alpha, a_face=a_face, beta_backflow=beta_backflow)
                Ab = (T_vec.T @ Af_bf @ T_vec).tocsr() - Af_c
                A = (Af_c + Ab).tocsr()
        return (A, b)

    def cur_u_free():
        if st.hist.pre1 is None:
            return np.zeros((st.n_free, dim))
        return st._uvec(st.hist.pre1)

    q = qref(fx)
    cd_hist, cl_hist, mu_hist = [], [], []
    blew_up = False
    for step in range(1, nsteps + 1):
        u, p = st.step(extra_block=extra_block(cur_u_free()),
                       sbm_nodes=sbm_nodes, ppe_surrogate_flux=None,
                       correction_penalty=correction_penalty)
        if not np.isfinite(u).all() or np.abs(u).max() > 1e4:
            blew_up = True
            print(f"[proj] BLOW-UP at step {step} max|u|={np.abs(u).max():.3e}",
                  flush=True)
            break
        xfree = np.zeros(st.n_free * ndof)
        xv = xfree.reshape(st.n_free, ndof)
        xv[:, :dim] = u
        xv[:, dim] = p
        F = surrogate_traction(dm, fx["sf"], fx["geo"],
                               np.asarray(T_vec @ xfree), nu, ndof)
        cd_hist.append(float(F[0] / q))
        cl_hist.append(float(F[1] / q))
        mu_hist.append(mean_speed(u))
        if log_every and (step <= 3 or step % log_every == 0):
            trig = "TRIG" if step <= trigger_steps else "    "
            print(f"[proj {trig}] step{step:5d}  Cd={cd_hist[-1]:+.4f}  "
                  f"Cl={cl_hist[-1]:+.4f}  mean|u|={mu_hist[-1]:.4f}", flush=True)
    return dict(cd_hist=cd_hist, cl_hist=cl_hist, mu_hist=mu_hist,
                div=float(st.divergence_l2()), blew_up=blew_up,
                steps=len(cd_hist))


# --------------------------------------------------------------------------
# MONOLITHIC shedding march (same mesh, same trigger)
# --------------------------------------------------------------------------
def shed_monolithic(fx, dt, nsteps, trigger_steps=200, eps=0.05,
                    alpha=ALPHA, graddiv_gamma=FN1_GRADDIV_GAMMA,
                    backflow_beta=0.5, boundary_vorticity=True, log_every=200,
                    ramp_steps=0):
    """Same-mesh saddle NS with the SBM Nitsche block and the SAME transient
    inflow tilt on the strong rows. The unsteady-wake oracle."""
    dim = fx["dim"]
    dm, mesh, cons = fx["dm"], fx["mesh"], fx["cons"]
    sf, geo = fx["sf"], fx["geo"]
    ndof, nu = fx["ndof"], fx["nu"]
    coords = fx["coords"]
    strong_nodes, g_strong = build_box_strong_bc(fx)
    is_inflow_strong = fx["inflow_mask"][strong_nodes]
    t_end = trigger_steps * dt
    t_ramp = ramp_steps * dt

    T = cons.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    nfree = T.shape[1]
    xq = gauss_points(mesh, dm.tables_by_p)

    Af, bf = sbm_vector_dirichlet(dm, sf, geo,
                                  lambda y: np.zeros((len(y), dim)),
                                  nu, ndof, alpha=alpha)
    Af_c = (T_vec.T @ Af @ T_vec).tocsr()
    bf_c = np.asarray(T_vec.T @ bf)

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
    q = qref(fx)
    x = np.zeros(nfree * ndof)
    sigma = 1.0 / dt
    bf_faces = outflow_faces(mesh)
    cd_hist, cl_hist, mu_hist = [], [], []
    blew_up = False
    for step in range(1, nsteps + 1):
        t_new = step * dt
        u_node = x.reshape(nfree, ndof)[:, :dim]
        aq, dq = gp_field(u_node)
        fq = {pv: aq[pv] / dt for pv in xq}
        A, b = assemble_linear_ns(dm, aq, dq, fq, nu, sigma=sigma)
        A = (A + Af_c)
        b = b + bf_c
        if gd_block is not None:
            A = A + gd_block
        if backflow_beta != 0.0:
            A = A + assemble_backflow_block(dm, u_node, backflow_beta, ndof,
                                            faces=bf_faces)
        if boundary_vorticity:
            A = A + assemble_bvs_block(dm, u_node, nu, dt, ndof, faces=bf_faces)
        A = A.tolil()
        g_now = g_strong.copy()
        if t_ramp > 0 and t_new < t_ramp - 1e-12:
            s = 0.5 * (1.0 - np.cos(np.pi * t_new / t_ramp))
            g_now[is_inflow_strong, 0] *= s
        if t_new < t_end - 1e-12:
            w = np.sin(np.pi * t_new / t_end) ** 2 if t_end > 0 else 1.0
            g_now[is_inflow_strong, 1] = eps * U_IN * w
        for k, i in enumerate(strong_nodes):
            for c in range(dim):
                r = i * ndof + c
                A.rows[r] = [int(r)]
                A.data[r] = [1.0]
                b[r] = g_now[k, c]
        pin = p_pin * ndof + dim
        A.rows[pin] = [pin]
        A.data[pin] = [1.0]
        b[pin] = 0.0
        x = splu(A.tocsr().tocsc()).solve(b)
        u_new = x.reshape(nfree, ndof)[:, :dim]
        if not np.isfinite(u_new).all() or np.abs(u_new).max() > 1e4:
            blew_up = True
            print(f"[mono] BLOW-UP at step {step}", flush=True)
            break
        F = surrogate_traction(dm, sf, geo, np.asarray(T_vec @ x), nu, ndof)
        cd_hist.append(float(F[0] / q))
        cl_hist.append(float(F[1] / q))
        mu_hist.append(mean_speed(u_new))
        if log_every and (step <= 3 or step % log_every == 0):
            trig = "TRIG" if step <= trigger_steps else "    "
            print(f"[mono {trig}] step{step:5d}  Cd={cd_hist[-1]:+.4f}  "
                  f"Cl={cl_hist[-1]:+.4f}  mean|u|={mu_hist[-1]:.4f}", flush=True)
    return dict(cd_hist=cd_hist, cl_hist=cl_hist, mu_hist=mu_hist,
                blew_up=blew_up, steps=len(cd_hist))


# --------------------------------------------------------------------------
# shedding diagnostics: is it a self-sustaining limit cycle?
# --------------------------------------------------------------------------
def analyze(cl_hist, cd_hist, dt, D, tail_frac=0.5, label=""):
    cl = np.asarray(cl_hist, float)
    cd = np.asarray(cd_hist, float)
    n = len(cl)
    if n < 32:
        return dict(shedding=False, reason="too short")
    i0 = int((1.0 - tail_frac) * n)
    cl_tail = cl[i0:]
    cd_tail = cd[i0:]
    cl_amp = 0.5 * (cl_tail.max() - cl_tail.min())
    cd_mean = float(cd_tail.mean())
    cl_std = float(cl_tail.std())
    # split the tail in two halves — a decaying transient shrinks the second
    # half's amplitude; a limit cycle keeps it (ratio ~1).
    h = len(cl_tail) // 2
    amp1 = 0.5 * (cl_tail[:h].max() - cl_tail[:h].min())
    amp2 = 0.5 * (cl_tail[h:].max() - cl_tail[h:].min())
    sustain_ratio = amp2 / amp1 if amp1 > 1e-9 else 0.0
    St, f_peak = strouhal(cl_hist, dt, D, skip_frac=(1.0 - tail_frac))
    shedding = (cl_amp > 1e-3) and (sustain_ratio > 0.5) and np.isfinite(St)
    res = dict(shedding=bool(shedding), St=float(St), f_peak=float(f_peak),
               cl_amp=float(cl_amp), cl_std=cl_std, cd_mean=cd_mean,
               sustain_ratio=float(sustain_ratio), n=n)
    print(f"  [{label}] shedding={res['shedding']}  St={St:.4f}  "
          f"Cl_amp={cl_amp:.4f}  mean_Cd={cd_mean:+.4f}  "
          f"sustain(amp2/amp1)={sustain_ratio:.3f}  f={f_peak:.4f}", flush=True)
    return res


LEVELS = {"L5": 5, "L6": 6, "L7": 7}


def main():
    import json
    import os
    import sys
    lvl_key = sys.argv[1] if len(sys.argv) > 1 else "L6"
    Re = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
    level = LEVELS[lvl_key] if lvl_key in LEVELS else int(lvl_key)
    half = float(os.environ.get("HALF", "0.0625"))
    dt = float(os.environ.get("DT", "0.01"))
    nsteps = int(os.environ.get("NSTEPS", "6000"))
    trigger_steps = int(os.environ.get("TRIGGER_STEPS", "300"))
    eps = float(os.environ.get("EPS", "0.1"))
    solver = os.environ.get("PROJ_SOLVER", "splu")
    device = os.environ.get("DEVICE", "cpu" if solver == "splu" else "cuda:0")
    do_mono = os.environ.get("MONO", "1") == "1"
    # y_offset: a PERMANENT transverse obstacle shift (fraction of a cell) that
    # seeds shedding the symmetry-preserving scheme cannot erase. 0 => the
    # transient inflow-tilt trigger only (which decays if the wake is stable).
    cell = 1.0 / (2 ** level)
    y_off_cells = float(os.environ.get("YOFF_CELLS", "0.0"))
    y_offset = y_off_cells * cell
    D = 2.0 * half

    print(f"{'='*72}\n SHEDDING STUDY  {lvl_key}(level={level})  Re={Re}  "
          f"half={half} D={D} blockage={D:.3f}  dt={dt} nsteps={nsteps}  "
          f"trigger={trigger_steps}steps eps={eps}  y_off={y_off_cells}cell="
          f"{y_offset:.5f}  solver={solver} dev={device}"
          f"\n{'='*72}", flush=True)

    fx = build_square_channel_2d(level, Re, half=half, offset=0, device=device,
                                 y_offset=y_offset)
    if y_offset == 0.0:
        assert fx["dmax"] == 0, f"expected body-fitted (dmax=0); got {fx['dmax']}"
    print(f" mesh: n_fluid_cells={fx['n_fluid_cells']}  "
          f"n_obstacle_nodes={int(fx['obstacle_node_mask'].sum())}  "
          f"nu={fx['nu']:.5f}  dmax={fx['dmax']}", flush=True)

    print("\n--- PROJECTION (consistent + weak Nitsche) ---", flush=True)
    pr = shed_projection(fx, dt, nsteps, trigger_steps=trigger_steps, eps=eps,
                         solver=solver)
    pra = analyze(pr["cl_hist"], pr["cd_hist"], dt, D, label="proj")

    moa = None
    if do_mono:
        print("\n--- MONOLITHIC (same mesh, same trigger) ---", flush=True)
        mo = shed_monolithic(fx, dt, nsteps, trigger_steps=trigger_steps,
                             eps=eps)
        moa = analyze(mo["cl_hist"], mo["cd_hist"], dt, D, label="mono")

    out = dict(level=level, Re=Re, half=half, D=D, dt=dt, nsteps=nsteps,
               trigger_steps=trigger_steps, eps=eps,
               proj=pra, proj_div=pr["div"], proj_blew=pr["blew_up"],
               mono=moa)
    # dump time series for offline FFT / plotting
    outdir = os.environ.get("OUTDIR", ".")
    tag = f"{lvl_key}_Re{int(Re)}_half{half}"
    np.savez(os.path.join(outdir, f"shed_{tag}.npz"),
             proj_cl=pr["cl_hist"], proj_cd=pr["cd_hist"],
             proj_mu=pr["mu_hist"],
             mono_cl=(mo["cl_hist"] if do_mono else []),
             mono_cd=(mo["cd_hist"] if do_mono else []), dt=dt, D=D)
    print("\n[json]", json.dumps(out, default=lambda o: None), flush=True)
    return out


if __name__ == "__main__":
    main()
