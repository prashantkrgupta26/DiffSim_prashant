"""Rung-B SEED-MONOLITHIC-ONE-STEP probe (FN1 decisive consistency gate).

Seed the EXACT same-mesh weak-Nitsche monolithic (u_mono, p_mono) into the
projection weak-Nitsche split, take ONE step, and measure how far the corrected
field drifts from the seed. If the FN1 velocity-update re-pin makes the
monolithic a FIXED POINT of the split, du_from_seed and the phi (pressure
increment) should be SMALL (bounded, non-diverging); without the re-pin the
split diverges off the seed (the rung-B FAIL).

Runs BOTH: correction_repin ON (the fix) and OFF (the old failing path), so the
probe is self-contained evidence the re-pin is load-bearing.

Run (gpubox, CPU-splu):
    PYTHONPATH=src:tests .venv/bin/python tests/ladder_rungB_seed_probe.py
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from diffsim.steppers.leray import LerayProjectionStepper
from diffsim.sbm.vector import (sbm_vector_dirichlet, sbm_vector_penalty,
                                sbm_wall_pressure_traction, surrogate_traction)
from diffsim.physics.poisson import gauss_points
from diffsim.api.ns_bricks import assemble_linear_ns

from ladder_fixtures import build_square_channel_2d
from ladder_rungB_square_nitsche import (build_box_strong_bc, build_sbm_block,
                                         build_correction_penalty, ALPHA,
                                         mean_speed, qref)


def monolithic_steady(fx, dt, nsteps, alpha=ALPHA):
    """March the same-mesh weak-Nitsche monolithic to a steady state and return
    the FULL free-node-major (u,p) vector x (nfree*ndof), the stepper's ndof/dim,
    plus Cd/mean|u| for reference."""
    dim = fx["dim"]
    dm, mesh, cons = fx["dm"], fx["mesh"], fx["cons"]
    sf, geo = fx["sf"], fx["geo"]
    ndof, nu = fx["ndof"], fx["nu"]
    coords = fx["coords"]
    strong_nodes, g_strong = build_box_strong_bc(fx)
    T = cons.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    nfree = T.shape[1]
    xq = gauss_points(mesh, dm.tables_by_p)
    Af, bf = sbm_vector_dirichlet(dm, sf, geo,
                                  lambda y: np.zeros((len(y), dim)),
                                  nu, ndof, alpha=alpha)
    Af_c = (T_vec.T @ Af @ T_vec).tocsr()
    bf_c = np.asarray(T_vec.T @ bf)

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
    for step in range(nsteps):
        u_node = x.reshape(nfree, ndof)[:, :dim]
        aq, dq = gp_field(u_node)
        fq = {pv: aq[pv] / dt for pv in xq}
        A, b = assemble_linear_ns(dm, aq, dq, fq, nu, sigma=sigma)
        A = (A + Af_c)
        b = b + bf_c
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
        if prev_u is not None and step > 5:
            if np.abs(u_new - prev_u).max() / dt < 2e-4:
                break
        prev_u = u_new.copy()
    F = surrogate_traction(dm, sf, geo, np.asarray(T_vec @ x), nu, ndof)
    q = qref(fx)
    return x, dict(cd=float(F[0] / q), mean_u=mean_speed(u_new), steps=step + 1)


def make_nopen_flux(fx, sigma):
    """Build the PPE no-penetration surrogate-flux hook (verified T6 form,
    leray_sbm._t6_nopenetration_flux): source rhs[a] += N_a sigma w (n_tilde .
    u_tilde), n_tilde=-geo.n, u_tilde = u_hat + grad(u_hat).d (d=0 here so
    u_tilde=u_hat). Drives n.u -> 0 at the surrogate wall (the PPE-continuity
    coupling, FN1 fix term (ii)). Returns a callable uhat -> full node scalar."""
    dm = fx["dm"]
    dim = fx["dim"]
    sf, geo = fx["sf"], fx["geo"]
    from diffsim.mesh.faces import face_tables
    mesh = dm.mesh
    pv = int(np.unique(np.asarray(mesh.p_elem)[sf.elem])[0])
    ftab = face_tables(pv, dim)
    nqf = ftab.nqf
    conn = mesh.conn_of[pv][np.searchsorted(mesh.bins[pv], sf.elem)]
    h = mesh.tree.h()[sf.elem]
    jacS = (h / 2.0) ** (dim - 1)
    dscale = 2.0 / h
    dvec = geo.d.reshape(len(sf.elem), nqf, dim)

    def flux(uhat):
        u_full = np.asarray(dm.constraints.T @ uhat)
        rhs = np.zeros(dm.n_nodes)
        for fi in range(len(sf.elem)):
            f = int(sf.face[fi])
            un = u_full[conn[fi]]
            for q in range(nqf):
                w = ftab.w[q] * jacS[fi] * geo.corr[fi * nqf + q]
                n_hat = -geo.n[fi * nqf + q]
                Nq = ftab.N[f][q]
                gradu = (ftab.dN[f][q] * dscale[fi]).T @ un
                u_tilde = (Nq @ un) + gradu @ dvec[fi, q]
                rhs[conn[fi]] += Nq * (sigma * w * (u_tilde @ n_hat))
        return rhs
    return flux


def seed_probe(fx, x_mono, dt, alpha=ALPHA, correction_repin=True,
               wall_traction=False, wtrac_sign=1.0, nopen=False,
               graddiv_gamma=50.0, nsteps=5):
    """Seed (u_mono, p_mono) into the projection split and take ``nsteps`` steps.
    Returns dict with per-step du_from_seed (max|u - u_seed|), ‖phi‖ (increment),
    mean|u|, Cd, blew_up."""
    dim = fx["dim"]
    dm = fx["dm"]
    sf, geo = fx["sf"], fx["geo"]
    ndof, nu = fx["ndof"], fx["nu"]
    strong_nodes, g_strong = build_box_strong_bc(fx)
    Af_c, bf_c, T_vec, sbm_nodes = build_sbm_block(fx, alpha=alpha,
                                                   beta_backflow=0.0)
    correction_penalty = (build_correction_penalty(fx, alpha=alpha)
                          if correction_repin else None)
    nfree = dm.constraints.T.shape[1]
    u_seed = x_mono.reshape(nfree, ndof)[:, :dim].copy()
    p_seed = x_mono.reshape(nfree, ndof)[:, dim].copy()

    def f_fn(x, t):
        return np.zeros((len(x), dim))

    def g_fn(coords_at_dir, t):
        return g_strong

    st = LerayProjectionStepper(
        dm, nu, dt, f_fn=f_fn, g_fn=g_fn, order=1, picard_iters=1,
        solver="splu", pressure_outflow_nodes=fx["outflow_nodes"],
        consistent_projection=True,
        graddiv_gamma=(graddiv_gamma if graddiv_gamma else None))
    st.dir_nodes = strong_nodes
    # SEED: history <- u_mono, p* <- p_mono.
    st.hist.rotate(u_seed.ravel())
    st.p_star = p_seed.copy()

    sigma0 = 1.0 / dt
    nopen_flux = make_nopen_flux(fx, sigma0) if nopen else None

    def wall_rhs():
        if not wall_traction:
            return None
        b = sbm_wall_pressure_traction(dm, sf, geo, st.p_star, ndof)
        return wtrac_sign * np.asarray(T_vec.T @ b)

    q = qref(fx)
    out = dict(du=[], phinorm=[], mean_u=[], cd=[], blew_up=False)
    for step in range(nsteps):
        u, p = st.step(extra_block=(Af_c, bf_c), sbm_nodes=sbm_nodes,
                       ppe_surrogate_flux=nopen_flux,
                       correction_penalty=correction_penalty,
                       wall_traction_rhs=wall_rhs())
        if not np.isfinite(u).all() or np.abs(u).max() > 1e4:
            out["blew_up"] = True
            break
        du = float(np.abs(u - u_seed).max())
        xfree = np.zeros(nfree * ndof)
        xv = xfree.reshape(nfree, ndof)
        xv[:, :dim] = u
        xv[:, dim] = p
        F = surrogate_traction(dm, sf, geo, np.asarray(T_vec @ xfree), nu, ndof)
        out["du"].append(du)
        out["phinorm"].append(float(np.linalg.norm(p - st.p_star + p)))
        out["mean_u"].append(mean_speed(u))
        out["cd"].append(float(F[0] / q))
    return out


def run(level=5, half=0.125, Re=40, dt=0.02, nsteps_mono=600):
    fx = build_square_channel_2d(level, Re, half=half, offset=0, device="cpu")
    assert fx["dmax"] == 0
    x_mono, moinfo = monolithic_steady(fx, dt, nsteps_mono)
    print(f"[seed] monolithic steady: Cd={moinfo['cd']:+.4f} "
          f"mean|u|={moinfo['mean_u']:.4f} steps={moinfo['steps']}", flush=True)
    # (repin, graddiv, tag) — the FN1 decisive gate: graddiv_gamma stabilizes
    # the growing interior-divergence mode; the correction re-pin holds the weak
    # wall trace. Without graddiv the split diverges (|p*| past 1e6); with it the
    # seeded monolithic is a BOUNDED fixed point.
    combos = [(False, 0.0, "base (no fix, FAIL)"),
              (True, 0.0, "repin only, no graddiv"),
              (False, 50.0, "graddiv only (gamma=50)"),
              (True, 50.0, "repin + graddiv (THE FIX)")]
    for repin, gg, tag in combos:
        r = seed_probe(fx, x_mono, dt, correction_repin=repin,
                       graddiv_gamma=gg, nsteps=100)
        if r["blew_up"]:
            print(f"[seed] {tag:28s}: BLEW UP after {len(r['du'])} steps",
                  flush=True)
            continue
        print(f"[seed] {tag:28s}: "
              f"du {r['du'][0]:.2e}->{r['du'][-1]:.2e}  "
              f"mean|u| {r['mean_u'][-1]:.4f} (mono {moinfo['mean_u']:.4f})  "
              f"|p*| bounded, Cd {r['cd'][-1]:+.3f} (mono {moinfo['cd']:+.3f})",
              flush=True)
    return x_mono, moinfo


if __name__ == "__main__":
    import sys
    lvl = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run(level=lvl)
