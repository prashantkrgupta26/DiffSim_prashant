"""Change #5 gate — LONG Re=100 secular-drift probe.

Marches rung-A Re=100 for a long horizon and records the ‖div‖ and mean|u|
trajectory (every `sample` steps), so we can see whether the P1 boundary-vorticity
term (#5) ARRESTS the secular drift the baseline shows (‖div‖ 1.6 -> 51.6,
mean|u| 1.09 -> 1.75 by step 2600, monotone).

Usage:
    PYTHONPATH=src:tests python tests/rungA_drift_probe.py [nsteps] [bvs|nobvs]

Prints a JSON line with the sampled trajectory + the min/max/final ‖div‖ and
mean|u|, and whether ‖div‖ is monotone-increasing over the tail (the drift
signature).
"""
import json
import sys

import numpy as np
import scipy.sparse as sp

from diffsim.steppers.leray import LerayProjectionStepper
from diffsim.sbm.vector import surrogate_traction
from ladder_fixtures import build_square_channel_2d, U_IN
from ladder_rungA_square_strong import build_strong_bc, qref, mean_speed


def divergence_l2_of(st):
    return float(st.divergence_l2())


def march(nsteps=2600, dt=0.01, sample=100, bvs=True, level=5, half=0.125):
    fx = build_square_channel_2d(level, 100, half=half, offset=0, device="cpu")
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
        dm, nu, dt, f_fn=f_fn, g_fn=g_fn, order=2, picard_iters=1,
        solver="splu", pressure_outflow_nodes=fx["outflow_nodes"],
        consistent_projection=True)
    # the drift probe toggles ONLY the boundary-vorticity term (#5), keeping the
    # rest of the consistent-projection set fixed, so the effect is isolated.
    st.boundary_vorticity = bool(bvs)
    st.dir_nodes = strong_nodes
    st.set_initial(lambda c: np.zeros((len(c), dim)))

    T = dm.constraints.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    q = qref(fx)
    traj = []
    for step in range(1, nsteps + 1):
        u, p = st.step()
        if not np.isfinite(u).all() or np.abs(u).max() > 1e4:
            traj.append((step, float("inf"), float("inf"), float("nan")))
            break
        if step % sample == 0 or step <= 3:
            xfree = np.zeros(st.n_free * ndof)
            xv = xfree.reshape(st.n_free, ndof)
            xv[:, :dim] = u
            xv[:, dim] = p
            F = surrogate_traction(dm, sf, geo, np.asarray(T_vec @ xfree),
                                   nu, ndof)
            traj.append((step, divergence_l2_of(st), mean_speed(u),
                         float(F[0] / q)))
    return traj


if __name__ == "__main__":
    nsteps = int(sys.argv[1]) if len(sys.argv) > 1 else 2600
    bvs = not (len(sys.argv) > 2 and sys.argv[2] == "nobvs")
    traj = march(nsteps=nsteps, bvs=bvs)
    divs = [d for (_, d, _, _) in traj if np.isfinite(d)]
    mus = [m for (_, _, m, _) in traj if np.isfinite(m)]
    tail = divs[len(divs) // 2:]
    monotone = all(b >= a - 1e-6 for a, b in zip(tail, tail[1:])) \
        if len(tail) > 2 else False
    out = dict(
        bvs=bvs, nsteps=nsteps,
        div_first=divs[0] if divs else None,
        div_final=divs[-1] if divs else None,
        div_max=max(divs) if divs else None,
        mu_first=mus[0] if mus else None,
        mu_final=mus[-1] if mus else None,
        mu_max=max(mus) if mus else None,
        tail_div_monotone_increasing=monotone,
        traj=traj)
    print("[drift-json]", json.dumps(out))
