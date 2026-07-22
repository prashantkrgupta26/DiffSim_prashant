"""P2-R2a outflow-Dirichlet PPE BC probe (Task 3b, Baskar outflow-BC route).

The bake-off (tests/p2r2a_bakeoff_3d.py, committed 556166d) found NO stable
config: the incremental pressure p* accumulates unboundedly and the weak-
divergence never decays; only non-incremental Chorin bounds it (wrong Cd).
Root cause (Baskar's call): the PPE pins a single arbitrary FREE node (node 0)
as "enclosed flow", but the sphere fixture is an EXTERNAL flow with a FREE
OUTFLOW at x=1. A single node-pin leaves the outflow pressure floating -> the
incremental p* drifts. The Taly ns_vms reference instead imposes a physical
Dirichlet pressure BC on the outlet nodes.

Result (15-step probe, controller run): on (standard, ppe_fine_scale=True) the
outflow-BC (Dirichlet p=0 on x=1) makes weak-div DECAY, ||p|| DECELERATE toward
a bound, and Cd go POSITIVE and climb — vs the baseline node-0 pin which
diverges (||p||->1816, Cd->-11). Baskar's outflow hypothesis validated.

This probe (follow-up scope) marches the level-4 Re=1 Stokes sphere for STEPS
(env var, default 60) steps and prints:

  1. The outflow-BC projection trajectory: per-step ||p||, weak-div, Cd, plus a
     TAIL SUMMARY (does ||p|| plateau — last-5 increments; does weak-div keep
     decaying; where does Cd settle).
  2. The MONOLITHIC same-mesh trajectory over the SAME STEPS, its Cd printed
     alongside, using the IDENTICAL q = 0.5*U_IN^2*pi*R^2 normalization (both
     divide surrogate_traction F_x by qref()).
  3. The baseline node-0 pin trajectory (short, for the divergence contrast).

R0 faithfulness bar: does the outflow-BC projection Cd converge to the SAME
steady value as monolithic on this mesh? (At Re=1 Stokes the sphere Cd is
genuinely large, ~24-40, unlike the Re=100 value ~0.4 — so the reference is
the CONVERGED monolithic trajectory, not a single possibly-unconverged number.)

Runs on gpubox CPU (splu). Standalone script, NOT a pytest gate.

Usage:
    cd /path/to/DiffSim
    STEPS=60 .venv/bin/python tests/p2r2a_outflow_probe.py 2>&1 | tee /tmp/outflow_probe.log

Modelled on tests/p2r2a_bakeoff_3d.py (Task 3).
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

# ---------------------------------------------------------------------------
# Path bootstrap (mirrors the bake-off)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from p2r0_task10_sphere_derisk import (build_sphere_3d, outflow_free_nodes,   # noqa: E402
                                       qref, R, U_IN)
from p2r2a_diagnostic_3d import _weak_divergence_3d                          # noqa: E402
from diffsim.steppers.leray_sbm import LeraySBMStepper                       # noqa: E402
from diffsim.sbm.vector import sbm_vector_dirichlet, surrogate_traction      # noqa: E402
from diffsim.api.ns_bricks import assemble_linear_ns                         # noqa: E402
from diffsim.physics.poisson import gauss_points                             # noqa: E402

# ---------------------------------------------------------------------------
# Constants (match the bake-off / Task-1 diagnostic)
# ---------------------------------------------------------------------------
DT = 0.05
LEVEL = 4
RE = 1.0        # Re=1: Stokes, isolates pressure coupling
ALPHA = 100.0
STEPS = int(os.environ.get("STEPS", "60"))   # march length (env-parameterized)


# ---------------------------------------------------------------------------
# Projection march (LeraySBMStepper) — returns per-step trajectories
# ---------------------------------------------------------------------------

def _march_projection(fx, pressure_outflow_nodes, label, nsteps):
    """March nsteps on the level-4 sphere fixture in the given PPE-pin mode.

    pressure_outflow_nodes=None      -> baseline node-0 pin.
    pressure_outflow_nodes=<indices> -> Taly outflow Dirichlet BC.

    Both use pressure_update="standard", ppe_fine_scale=True. Cd = F_x / qref()
    (the SAME q as the monolithic reference). Returns trajectory + stability
    dict.
    """
    dim = fx["dim"]

    def f_fn(x, t):
        return np.zeros((len(x), dim))

    st = LeraySBMStepper(
        fx["oracle"], fx["dm"], fx["nu"], DT, f_fn,
        u_inf=fx["u_inf"], strong_mask=fx["strong_mask"],
        lam=0.5, domain="outside", order=2, picard_iters=2,
        solver="splu", ppe_finescale=False, alpha=ALPHA,
        beta_backflow=1.0, velocity_update="consistent",
        pressure_update="standard", ppe_fine_scale=True,
        pressure_outflow_nodes=pressure_outflow_nodes,
    )
    st.set_initial(lambda c: np.zeros((len(c), dim)))

    q = qref()   # IDENTICAL normalization to the monolithic reference

    phat_norms, weakdiv, cds = [], [], []
    for k in range(nsteps):
        _u, _p = st.step()
        phat_norms.append(float(np.linalg.norm(st.base.p_star)))
        w2, _winf = _weak_divergence_3d(st)
        weakdiv.append(float(w2))
        F = st.surrogate_traction()
        cds.append(float(F[0] / q))
        print(
            f"[{label}] step{k+1:3d}: "
            f"||p||={phat_norms[-1]:.4e}  "
            f"wdiv={weakdiv[-1]:.4e}  "
            f"Cd={cds[-1]:+.4f}",
            flush=True,
        )

    bounded = bool(
        np.isfinite(phat_norms[-1])
        and phat_norms[-1] < 2.0 * (phat_norms[0] + 1e-30)
    )
    weakdiv_decays = bool(len(weakdiv) > 2 and weakdiv[-1] < 0.5 * weakdiv[2])
    stable = bool(bounded and weakdiv_decays and np.isfinite(cds[-1]))

    return dict(
        phat_norms=phat_norms, weakdiv=weakdiv, cds=cds,
        bounded=bounded, weakdiv_decays=weakdiv_decays, stable=stable,
    )


# ---------------------------------------------------------------------------
# Monolithic march (NO projection split) — per-step Cd trajectory
# ---------------------------------------------------------------------------
# Mirrors p2r0_task10_sphere_derisk.monolithic_cd EXACTLY (same assembly, same
# strong-row overwrite, same enclosed-flow far-corner pressure pin, same
# surrogate_traction / qref() Cd), but records the per-step Cd so we can compare
# converged-to-converged against the projection. Cd = F_x / qref() — the SAME q.

def _march_monolithic(fx, alpha, nsteps):
    dm, mesh, cons = fx["dm"], fx["mesh"], fx["cons"]
    sf, geo = fx["sf"], fx["geo"]
    nu, ndof, dim = fx["nu"], fx["ndof"], fx["dim"]
    coords = fx["coords"]
    T = cons.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    nfree = T.shape[1]
    xq = gauss_points(mesh, dm.tables_by_p)
    strong = np.where(fx["strong_mask"])[0]
    g_strong = fx["u_inf"][strong]
    Af, bf = sbm_vector_dirichlet(
        dm, sf, geo, lambda y: np.zeros((len(y), dim)), nu, ndof, alpha=alpha)
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

    q = qref()   # IDENTICAL normalization to the projection Cd
    x = np.zeros(nfree * ndof)
    sigma = 1.0 / DT
    cds = []
    for step in range(nsteps):
        u_node = x.reshape(nfree, ndof)[:, :dim]
        aq, dq = gp_field(u_node)
        fq = {pv: aq[pv] / DT for pv in xq}
        A, b = assemble_linear_ns(dm, aq, dq, fq, nu, sigma=sigma)
        A = (A + Af_c).tolil()
        b = b + bf_c
        for k, i in enumerate(strong):
            for c in range(dim):
                r = i * ndof + c
                A.rows[r] = [int(r)]
                A.data[r] = [1.0]
                b[r] = g_strong[k, c]
        pin = int(np.argmax(coords.sum(1))) * ndof + dim
        A.rows[pin] = [pin]
        A.data[pin] = [1.0]
        b[pin] = 0.0
        x = splu(A.tocsr().tocsc()).solve(b)
        F = surrogate_traction(dm, sf, geo, np.asarray(T_vec @ x), nu, ndof)
        cds.append(float(F[0] / q))
        print(f"[mono] step{step+1:3d}: Cd={cds[-1]:+.4f}", flush=True)
    return dict(cds=cds)


# ---------------------------------------------------------------------------
# Tail summaries
# ---------------------------------------------------------------------------

def _tail_summary(label, res):
    p = res["phat_norms"]
    w = res["weakdiv"]
    c = res["cds"]
    incs = [p[i] - p[i - 1] for i in range(max(1, len(p) - 5), len(p))]
    incs_str = ", ".join(f"{d:+.3e}" for d in incs)
    # weak-div still decaying over the last 5?
    w_tail_decays = len(w) >= 6 and w[-1] < w[-6]
    # Cd settling: std of last-5 vs mean
    tail_c = c[-5:] if len(c) >= 5 else c
    c_mean = float(np.mean(tail_c))
    c_std = float(np.std(tail_c))
    print(
        f"[{label}] TAIL: last-5 ||p|| increments = [{incs_str}]  "
        f"(||p||_final={p[-1]:.4e})",
        flush=True,
    )
    print(
        f"[{label}] TAIL: weakdiv {w[0]:.4e} -> {w[-1]:.4e}  "
        f"tail_still_decaying={w_tail_decays}",
        flush=True,
    )
    print(
        f"[{label}] TAIL: Cd last-5 mean={c_mean:+.4f} std={c_std:.4f}  "
        f"(Cd_final={c[-1]:+.4f})",
        flush=True,
    )
    return dict(p_incs=incs, w_tail_decays=w_tail_decays,
                cd_mean=c_mean, cd_std=c_std)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print(f"[probe] STEPS={STEPS}  Building level-4 sphere fixture at Re=1 "
          f"(Stokes)...", flush=True)
    device = "cuda:0" if os.environ.get("DIFFSIM_CUDA") else "cpu"
    fx = build_sphere_3d(device, level=LEVEL, Re=RE)
    outflow = outflow_free_nodes(fx)
    print(
        f"[probe] fixture ready  n_free={len(fx['coords'])}  "
        f"n_outflow_free_nodes={len(outflow)}  ({time.time()-t0:.1f}s)",
        flush=True,
    )

    # --- (1) outflow-BC projection: full STEPS march ---
    print(f"\n=== (1) outflow-BC projection: Dirichlet p=0 on x=1, "
          f"{STEPS} steps ===", flush=True)
    res_out = _march_projection(fx, outflow, "outflow-BC", STEPS)
    tail_out = _tail_summary("outflow-BC", res_out)

    # --- (2) monolithic same-mesh trajectory (SAME STEPS, SAME q) ---
    print(f"\n=== (2) monolithic same-mesh trajectory, {STEPS} steps "
          f"(SAME q normalization) ===", flush=True)
    res_mono = _march_monolithic(fx, ALPHA, STEPS)
    mono_tail = res_mono["cds"][-5:]
    mono_final = float(res_mono["cds"][-1])
    mono_mean = float(np.mean(mono_tail))
    mono_std = float(np.std(mono_tail))
    print(f"[mono] TAIL: Cd last-5 mean={mono_mean:+.4f} std={mono_std:.4f}  "
          f"(Cd_final={mono_final:+.4f})", flush=True)

    # --- (3) baseline node-0 pin (short contrast: min(STEPS,15)) ---
    n_base = min(STEPS, 15)
    print(f"\n=== (3) baseline node-0 pin (contrast), {n_base} steps ===",
          flush=True)
    res_base = _march_projection(fx, None, "baseline", n_base)

    # --- faithfulness verdict: outflow-BC proj Cd vs monolithic (same q) ---
    cd_out = float(res_out["cds"][-1])
    rel = (abs(cd_out - mono_final) / abs(mono_final)
           if np.isfinite(mono_final) and mono_final != 0 else float("nan"))
    print("", flush=True)
    print(
        f"[probe] FAITHFULNESS: outflow-BC proj Cd_final={cd_out:+.4f} "
        f"(last-5 mean={tail_out['cd_mean']:+.4f})  vs  monolithic "
        f"Cd_final={mono_final:+.4f} (last-5 mean={mono_mean:+.4f})  "
        f"|diff|/mono={rel:.2%}",
        flush=True,
    )
    print(
        f"[probe] STABILITY: outflow-BC bounded={res_out['bounded']} "
        f"weakdiv_decays={res_out['weakdiv_decays']} stable={res_out['stable']} "
        f"| baseline diverges to Cd={res_base['cds'][-1]:+.4f} "
        f"||p||={res_base['phat_norms'][-1]:.4e}",
        flush=True,
    )
    print(f"\n[probe] done ({time.time()-t0:.1f}s total)", flush=True)


if __name__ == "__main__":
    main()
