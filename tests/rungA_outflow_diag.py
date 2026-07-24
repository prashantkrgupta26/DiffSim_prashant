"""Rung A Part-2 diagnostic: is the monolithic steady state a FIXED POINT of the
projection inner map, and does the OUTFLOW pressure BC break it?

Seeds the projection stepper's history with the MONOLITHIC converged velocity
and p_star = monolithic pressure, then takes ONE inner-iterated projection step.
If the projection split were faithful, the monolithic steady state is a fixed
point: u_new ~= u_mono, Cd ~= +4.18. A large deviation (esp. Cd sign flip)
localizes the operator/BC inconsistency.

Also probes: single-node outflow pin vs whole-face outflow pin.
"""
import sys
import numpy as np
import scipy.sparse as sp

from diffsim.steppers.leray import LerayProjectionStepper
from diffsim.api.ns_bricks import assemble_linear_ns
from diffsim.physics.poisson import gauss_points
from diffsim.sbm.vector import surrogate_traction
from scipy.sparse.linalg import splu

from ladder_fixtures import build_square_channel_2d, U_IN
from ladder_rungA_square_strong import (build_strong_bc, qref, mean_speed)


def solve_monolithic_steady(fx, dt, nsteps=400, rate_tol=2e-4):
    """Return (u_node[nfree,dim], p_node[nfree]) monolithic steady + Cd."""
    dim = fx["dim"]
    dm, mesh = fx["dm"], fx["mesh"]
    sf, geo = fx["sf"], fx["geo"]
    ndof, nu = fx["ndof"], fx["nu"]
    coords = fx["coords"]
    strong_nodes, g_strong = build_strong_bc(fx)
    T = dm.constraints.T.tocsr()
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

    p_pin = int(np.argmax(coords[:, 0]))
    x = np.zeros(nfree * ndof)
    sigma = 1.0 / dt
    prev_u = None
    q = qref(fx)
    cd = np.nan
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
        if prev_u is not None and step > 5 and \
                np.abs(u_new - prev_u).max() / dt < rate_tol:
            break
        prev_u = u_new.copy()
    xv = x.reshape(nfree, ndof)
    return xv[:, :dim].copy(), xv[:, dim].copy(), cd


def one_projection_step_from_seed(fx, dt, u_seed, p_seed, *, inner_iterate,
                                  inner_relax, inner_max, outflow_pin="face",
                                  consistent_projection=False):
    dim = fx["dim"]
    dm = fx["dm"]
    sf, geo = fx["sf"], fx["geo"]
    ndof, nu = fx["ndof"], fx["nu"]
    coords = fx["coords"]
    strong_nodes, g_strong = build_strong_bc(fx)

    if outflow_pin == "face":
        pon = fx["outflow_nodes"]
    elif outflow_pin == "single":
        # single outflow node (max x), mirrors the monolithic pin
        on = fx["outflow_nodes"]
        pon = np.array([on[int(np.argmax(coords[on, 1]*0 + coords[on, 0]))]])
    else:
        pon = None

    def f_fn(x, t):
        return np.zeros((len(x), dim))

    def g_fn(c, t):
        return g_strong

    st = LerayProjectionStepper(
        dm, nu, dt, f_fn=f_fn, g_fn=g_fn, order=1, picard_iters=1,
        solver="splu", pressure_outflow_nodes=pon,
        inner_iterate=inner_iterate, inner_relax=inner_relax,
        inner_accel="none", inner_max=inner_max,
        consistent_projection=consistent_projection)
    st.dir_nodes = strong_nodes
    # seed history with u_seed (BDF1 needs one slot), and p_star = p_seed
    st.hist.rotate(u_seed.ravel())
    st.p_star = p_seed.copy()
    # weak divergence of the SEEDED field (before the step), the fixed-point bar.
    div_seed = float(st.divergence_l2())

    T = dm.constraints.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    q = qref(fx)
    u, p = st.step()
    div_after = float(st.divergence_l2())
    xfree = np.zeros(st.n_free * ndof)
    xv = xfree.reshape(st.n_free, ndof)
    xv[:, :dim] = u
    xv[:, dim] = p
    F = surrogate_traction(dm, sf, geo, np.asarray(T_vec @ xfree), nu, ndof)
    cd = float(F[0] / q)
    du = float(np.abs(u - u_seed).max())
    return dict(cd=cd, mean_u=mean_speed(u), du=du,
                pnorm=float(np.linalg.norm(p)), res=st.inner_res_hist,
                div_seed=div_seed, div_after=div_after)


if __name__ == "__main__":
    LEVEL, HALF, RE, DT = 5, 0.125, 40, 0.02
    fx = build_square_channel_2d(LEVEL, RE, half=HALF, offset=0, device="cpu")
    u_m, p_m, cd_m = solve_monolithic_steady(fx, DT)
    print(f"MONOLITHIC steady: Cd={cd_m:+.4f}  mean|u|={mean_speed(u_m):.4f}  "
          f"‖p‖={np.linalg.norm(p_m):.3e}", flush=True)

    print("\n--- BASE projection (pre-fix): seed corrupts the fixed point ---",
          flush=True)
    for pin in ("face", "single"):
        for ii, om, im in ((False, 1.0, 1), (True, 0.5, 20)):
            r = one_projection_step_from_seed(
                fx, DT, u_m, p_m, inner_iterate=ii, inner_relax=om,
                inner_max=im, outflow_pin=pin)
            tag = f"pin={pin:6s} inner={ii} om={om}"
            print(f"  [{tag}]  Cd={r['cd']:+.4f}  mean|u|={r['mean_u']:.4f}  "
                  f"du_from_seed={r['du']:.3e}  "
                  f"div {r['div_seed']:.3f}->{r['div_after']:.3f}  "
                  f"res[0..2]={[f'{x:.2e}' for x in r['res'][:3]]}"
                  f"{'...' if len(r['res'])>3 else ''} "
                  f"res_last={r['res'][-1] if r['res'] else 0:.2e}", flush=True)

    print("\n--- CONSISTENT projection (the fix, #1-#4): seed is a fixed "
          "point ---", flush=True)
    for pin in ("face", "single"):
        r = one_projection_step_from_seed(
            fx, DT, u_m, p_m, inner_iterate=False, inner_relax=1.0,
            inner_max=1, outflow_pin=pin, consistent_projection=True)
        tag = f"pin={pin:6s} consistent_projection=True"
        print(f"  [{tag}]  Cd={r['cd']:+.4f} (seed +{cd_m:.4f})  "
              f"mean|u|={r['mean_u']:.4f}  du_from_seed={r['du']:.3e}  "
              f"div {r['div_seed']:.3f}->{r['div_after']:.3f}", flush=True)
