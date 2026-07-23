"""F3b — instrument the secular divergence drift in the consistent-projection
stepper (Rung A, Re=100). Discriminates H1 (rotational-update feedback),
H2 (outflow divergence / mass imbalance), H3 (BDF/history), H4 (convection).

Per step it records:
  * global mass flux balance:  Qin = -∮_inflow u·n ,  Qout = ∮_outflow u·n
    (n outward), and net = Qout - Qin  (net outflow; >0 means draining, <0
    means filling). By Gauss ∫_Ω div u = ∮ u·n = Qout - Qin - (obstacle flux).
  * ‖div‖ localized: near-outflow band (x > 1-band) vs interior.
  * mean|u|, global ‖div‖.

Usage:
  PYTHONPATH=src:tests python tests/f3b_instrument.py MODE NSTEPS
    MODE in {consistent, standard, graddiv<gamma>}
"""
import sys
import numpy as np
import scipy.sparse as sp

from diffsim.steppers.leray import LerayProjectionStepper
from diffsim.api.ns_bricks import outflow_faces
from ladder_fixtures import build_square_channel_2d, U_IN
from ladder_rungA_square_strong import build_strong_bc


def face_flux(dm, u_free, axis, side, coord):
    """∮_{face} u·n dGamma over the boundary faces on plane x_axis=coord with
    outward normal ntilde (side=1 -> +axis, side=0 -> -axis). Returns the scalar
    signed flux (n outward)."""
    from diffsim.mesh.faces import face_tables
    mesh = dm.mesh
    dim = dm.dim
    elem, face, ntilde = outflow_faces(mesh, axis=axis, side=side, coord=coord)
    if len(elem) == 0:
        return 0.0
    pv = int(np.unique(np.asarray(mesh.p_elem)[elem])[0])
    ftab = face_tables(pv, dim)
    conn = mesh.conn_of[pv][np.searchsorted(mesh.bins[pv], elem)]
    u_full = np.asarray(dm.constraints.T @ u_free)
    h = mesh.tree.h()[elem]
    jacS = (h / 2.0) ** (dim - 1)
    tot = 0.0
    for fi in range(len(elem)):
        f = int(face[fi])
        N = ftab.N[f]                       # [nqf, nbf]
        un = u_full[conn[fi], :dim]         # [nbf, dim]
        uq = N @ un                         # [nqf, dim]
        un_dot_n = uq @ ntilde              # [nqf]
        tot += (un_dot_n * ftab.w).sum() * jacS[fi]
    return float(tot)


def div_localized(st, band=0.15):
    """(‖div‖_near_outflow, ‖div‖_interior) RMS over GPs, split by GP x-coord.
    near-outflow == GP x > 1-band."""
    dm = st.dm
    u = st._uvec(st.hist.pre1)
    xq = st.xq
    _, g = st._gp_vals(u, grad=True)
    tot_out = vol_out = tot_in = vol_in = 0.0
    for pv in g:
        tb = dm.tables_by_p[pv]
        h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
        jac = (h / 2.0) ** dm.dim
        dv = np.einsum("gdd->g", g[pv].reshape(-1, dm.dim, dm.dim))
        w = np.tile(tb.w, len(h)) * np.repeat(jac, tb.nqp)
        xg = xq[pv][:, 0]                    # GP x-coord
        mask = xg > (1.0 - band)
        tot_out += (dv[mask] ** 2 * w[mask]).sum()
        vol_out += w[mask].sum()
        tot_in += (dv[~mask] ** 2 * w[~mask]).sum()
        vol_in += w[~mask].sum()
    d_out = np.sqrt(tot_out / vol_out) if vol_out > 0 else 0.0
    d_in = np.sqrt(tot_in / vol_in) if vol_in > 0 else 0.0
    return float(d_out), float(d_in)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "consistent"
    nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 2600
    dt = 0.01
    fx = build_square_channel_2d(5, 100.0, half=0.125, offset=0, device="cpu")
    dim = fx["dim"]
    dm = fx["dm"]
    nu = fx["nu"]
    ndof = fx["ndof"]
    strong_nodes, g_strong = build_strong_bc(fx)

    def f_fn(x, t):
        return np.zeros((len(x), dim))

    def g_fn(coords_at_dir, t):
        return g_strong

    kw = dict(dm=dm, nu=nu, dt=dt, f_fn=f_fn, g_fn=g_fn, order=2,
              picard_iters=1, solver="splu",
              pressure_outflow_nodes=fx["outflow_nodes"],
              consistent_projection=True)
    gamma = None
    if mode == "standard":
        # override rotational -> standard update, keep everything else.
        # rotational_pin_outflow default-on is inert for standard update.
        st = LerayProjectionStepper(**kw)
        st.pressure_update = "standard"
    elif mode == "consistent":
        # BASELINE reproduce: consistent_projection but q-pin OFF (the drifter).
        st = LerayProjectionStepper(rotational_pin_outflow=False, **kw)
    elif mode == "qpin":
        # F3b cure: rotational update WITH the outflow q-pin (default-on).
        st = LerayProjectionStepper(rotational_pin_outflow=True, **kw)
    elif mode.startswith("graddiv"):
        gamma = float(mode[len("graddiv"):]) if len(mode) > len("graddiv") else 1.0
        # grad-div cure ON TOP OF the drifting rotational update (q-pin OFF), to
        # isolate whether grad-div alone arrests the drift.
        st = LerayProjectionStepper(graddiv_gamma=gamma,
                                    rotational_pin_outflow=False, **kw)
    else:
        st = LerayProjectionStepper(**kw)
    st.dir_nodes = strong_nodes
    st.set_initial(lambda c: np.zeros((len(c), dim)))

    print(f"[f3b {mode}] START nu={nu:.5f} dt={dt} nsteps={nsteps} "
          f"n_free={st.n_free} gamma={gamma}", flush=True)
    log_every = 25
    for step in range(1, nsteps + 1):
        u, p = st.step()
        if not np.isfinite(u).all() or np.abs(u).max() > 1e4:
            print(f"[f3b {mode}] BLOW-UP step{step} max|u|={np.abs(u).max():.3e}",
                  flush=True)
            break
        if step <= 3 or step % log_every == 0:
            div = float(st.divergence_l2())
            mu = float(np.sqrt((u ** 2).sum(1)).mean())
            qin = -face_flux(dm, u, axis=0, side=0, coord=0.0)   # inflow x=0, n=-x => -flux = inflow rate
            qout = face_flux(dm, u, axis=0, side=1, coord=1.0)   # outflow x=1, n=+x
            net = qout - qin
            d_out, d_in = div_localized(st)
            print(f"[f3b {mode}] step{step:4d} div={div:.4e} mean|u|={mu:.4f} "
                  f"Qin={qin:+.5f} Qout={qout:+.5f} net={net:+.5f} "
                  f"div_out={d_out:.4e} div_in={d_in:.4e} "
                  f"|p*|={np.linalg.norm(st.p_star):.3e}", flush=True)
    print(f"[f3b {mode}] DONE", flush=True)


if __name__ == "__main__":
    main()
