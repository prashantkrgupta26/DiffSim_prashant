"""Rung A — stabilized inner predictor<->PPE iteration experiment (Task 4).

Re-runs the rung-A body-fitted square STRONG-Dirichlet march with the base
projection stepper's NEW inner-iteration knobs (inner_iterate + inner_relax +
inner_accel), against the same-mesh monolithic oracle, to answer THE question:
does driving the within-step predictor<->PPE fixed point flip the projection to
MATCH the monolithic (Cd + mean|u|) — or does it still pin at the weak fixed
point (docs/dev/2026-07-23-projection-sbm-weak-fixed-point-verdict.md)?

Usage (box):
  PYTHONPATH=src:tests .venv/bin/python tests/rungA_inner_experiment.py \
      [omega] [accel] [ppe_fine_scale] [nsteps]
"""
import sys

import numpy as np
import scipy.sparse as sp

from diffsim.steppers.leray import LerayProjectionStepper
from diffsim.sbm.vector import surrogate_traction

from ladder_fixtures import build_square_channel_2d, U_IN
from ladder_rungA_square_strong import (build_strong_bc, qref, mean_speed,
                                        march_monolithic)


def march_projection_inner(fx, dt, nsteps, rate_tol, *, inner_iterate,
                           inner_relax=1.0, inner_accel="none",
                           inner_max=8, ppe_fine_scale=False,
                           consistent_ppe=False, log_every=25, order=2):
    dim = fx["dim"]
    dm = fx["dm"]
    sf, geo = fx["sf"], fx["geo"]
    ndof, nu = fx["ndof"], fx["nu"]
    strong_nodes, g_strong = build_strong_bc(fx)

    def f_fn(x, t):
        return np.zeros((len(x), dim))

    def g_fn(coords_at_dir, t):
        return g_strong

    st = LerayProjectionStepper(
        dm, nu, dt, f_fn=f_fn, g_fn=g_fn, order=order, picard_iters=1,
        solver="splu", pressure_outflow_nodes=fx["outflow_nodes"],
        ppe_fine_scale=ppe_fine_scale, consistent_ppe=consistent_ppe,
        inner_iterate=inner_iterate, inner_relax=inner_relax,
        inner_accel=inner_accel, inner_max=inner_max)
    st.dir_nodes = strong_nodes
    st.set_initial(lambda c: np.zeros((len(c), dim)))

    T = dm.constraints.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    q = qref(fx)
    prev = None
    cd = cl = np.nan
    u = None
    blew_up = False
    inner_iters_hist = []
    for steps in range(1, nsteps + 1):
        u, p = st.step()
        inner_iters_hist.append(st.inner_iters)
        if not np.isfinite(u).all() or not np.isfinite(p).all() \
                or np.abs(u).max() > 1e4:
            blew_up = True
            print(f"[inner proj] BLOW-UP at step{steps} "
                  f"(max|u|={np.abs(u).max():.3e})", flush=True)
            break
        xfree = np.zeros(st.n_free * ndof)
        xv = xfree.reshape(st.n_free, ndof)
        xv[:, :dim] = u
        xv[:, dim] = p
        F = surrogate_traction(dm, sf, geo, np.asarray(T_vec @ xfree), nu, ndof)
        cd = float(F[0] / q)
        cl = float(F[1] / q)
        mu = mean_speed(u)
        if log_every and (steps <= 3 or steps % log_every == 0):
            print(f"[inner proj] step{steps:4d}  Cd={cd:+.4f}  "
                  f"mean|u|={mu:.4f}  inner_iters={st.inner_iters}  "
                  f"res_last={st.inner_res_hist[-1] if st.inner_res_hist else 0:.2e}",
                  flush=True)
        if rate_tol is not None and prev is not None:
            if np.abs(u - prev).max() / dt < rate_tol:
                break
        prev = u.copy()
    div = float(st.divergence_l2())
    return dict(cd=cd, cl=cl, mean_u=mean_speed(u), div=div, steps=steps,
                pnorm=float(np.linalg.norm(st.p_star)), blew_up=blew_up,
                inner_mean=float(np.mean(inner_iters_hist)) if inner_iters_hist
                else 0.0)


if __name__ == "__main__":
    omega = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    accel = sys.argv[2] if len(sys.argv) > 2 else "none"
    pfs = (sys.argv[3].lower() in ("1", "true", "yes")) \
        if len(sys.argv) > 3 else False
    nsteps = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    inner_max = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    cppe = (sys.argv[6].lower() in ("1", "true", "yes")) \
        if len(sys.argv) > 6 else False
    ii = (sys.argv[7].lower() in ("1", "true", "yes")) \
        if len(sys.argv) > 7 else True

    LEVEL, HALF, RE, DT = 5, 0.125, 40, 0.02
    fx = build_square_channel_2d(LEVEL, RE, half=HALF, offset=0, device="cpu")
    assert fx["dmax"] == 0

    print(f"\n=== Rung A  omega={omega} accel={accel} ppe_fine_scale={pfs} "
          f"consistent_ppe={cppe} inner_iterate={ii} inner_max={inner_max} "
          f"nsteps={nsteps} ===", flush=True)
    mo = march_monolithic(fx, dt=DT, nsteps=nsteps, rate_tol=2e-4)
    pr = march_projection_inner(
        fx, DT, nsteps, rate_tol=5e-4, inner_iterate=ii,
        inner_relax=omega, inner_accel=accel, inner_max=inner_max,
        ppe_fine_scale=pfs, consistent_ppe=cppe)

    cd_rel = abs(pr["cd"] - mo["cd"]) / abs(mo["cd"])
    mu_frac = pr["mean_u"] / mo["mean_u"]
    print("\n--- VERDICT (Re=40) ---")
    print(f"  MONO  Cd={mo['cd']:+.4f}  mean|u|={mo['mean_u']:.4f}  "
          f"div={mo['div']:.3e}  steps={mo['steps']}")
    print(f"  PROJ  Cd={pr['cd']:+.4f}  mean|u|={pr['mean_u']:.4f}  "
          f"div={pr['div']:.3e}  ‖p‖={pr['pnorm']:.3e}  steps={pr['steps']}  "
          f"blew={pr['blew_up']}  inner_mean={pr['inner_mean']:.1f}")
    print(f"  Cd rel={cd_rel:.3%}   mean|u| proj/mono={mu_frac:.3f}")
