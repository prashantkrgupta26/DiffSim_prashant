"""P2-R2a 3-D VELOCITY-DEFICIT LOCALIZATION probe (predictor vs projection).

On the level-4 Re=1 Stokes sphere the outflow-BC projection
(``pressure_update="standard"``, ``ppe_fine_scale=True``,
``pressure_outflow_nodes=<x=1 face>``) produces a velocity field ~16x TOO WEAK
vs the monolithic saddle solver on the SAME mesh (field-compare, commit 3df9128:
``||u_proj||=4.47`` vs ``||u_mono||=72.8`` over free fluid nodes; drag ~0, no
fore-aft Δp). The flow simply never develops around the sphere. Monolithic on the
same fixture/mesh is correct (Cd~40, physical field).

This probe SPLITS the deficit across the projection step's two halves to name
which one carries it. Each projection step is:

    uhat  = momentum PREDICTOR  (leray.py _predict, ~L364): assembles the Oseen
            block, ADDS the SBM Nitsche extra_block, applies strong inflow/wall
            Dirichlet (skipping SBM-governed nodes), pins pressure DOFs to p*,
            solves. Does NOT advance history/p_star (safe to call for probing).
    u_new = post-projection corrected velocity (leray.py step, ~L604-657):
            PPE solve + L2 velocity update u = u_hat - (1/sigma) grad(phi).

THE KEY QUESTION: is ``uhat`` (predictor) ALREADY ~16x too weak vs monolithic
(=> the momentum PREDICTOR / strong-BC / SBM-block is the bug), OR is ``uhat``
O(1)-correct but ``u_new`` collapses (=> the PROJECTION / velocity-update is the
bug)?

HOW ``uhat`` IS OBTAINED PER STEP (load-bearing — the report sanity-checks this):
we call ``st._predict()`` — the ``LeraySBMStepper._predict`` wrapper (leray_sbm.py
L252) — IMMEDIATELY BEFORE the matching ``st.step()`` for that step. That wrapper
calls ``st.base._predict(extra_block=st._extra_block(st._current_a_free()),
sbm_nodes=st._sbm_nodes)`` with the IDENTICAL extra_block (cached geometry SBM
block ``Af_c`` + per-step backflow increment from ``u^n`` = ``hist.pre1``) and
IDENTICAL ``sbm_nodes`` that ``st.step()`` itself passes to ``base.step``
(leray_sbm.py L278-280). Because ``_predict`` "Does NOT advance the history or
p_star" (its docstring, leray.py L380-381) and because ``st.step()`` rebuilds the
same extra_block from the same ``_current_a_free()`` (still ``hist.pre1``, not yet
rotated) at that same ``base.t``, calling ``st._predict()`` right before
``st.step()`` reproduces EXACTLY the predictor output the step consumes. We do NOT
re-implement the SBM-block construction; we reuse the stepper's own wrapper.

``u_new`` is read AFTER the step as ``st.base._uvec(st.base.hist.pre1)`` (the
corrected velocity the step just rotated into history — the same vector
LeraySBMStepper.surrogate_traction assembles, leray_sbm.py L321).

Fast: STANDALONE diagnostic, NOT a pytest gate. Marches the 3-D sphere (splu on
gpubox CPU) — do NOT run locally.

Usage:
    cd /path/to/DiffSim
    STEPS=30 .venv/bin/python tests/p2r2a_velocity_deficit_probe.py 2>&1 | tee /tmp/velocity_deficit_probe.log

Reuses tests/p2r0_task10_sphere_derisk (build_sphere_3d, outflow_free_nodes,
qref, R, CTR, U_IN) and mirrors the monolithic march / outflow-BC projection
setup of tests/p2r2a_outflow_probe.py + tests/p2r2a_field_compare.py (Task 3b/3c)
with attribution.
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

# ---------------------------------------------------------------------------
# Path bootstrap (mirrors the outflow probe / field-compare)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from p2r0_task10_sphere_derisk import (build_sphere_3d, outflow_free_nodes,   # noqa: E402
                                       qref, R, CTR, U_IN)
from diffsim.steppers.leray_sbm import LeraySBMStepper                        # noqa: E402
from diffsim.sbm.vector import sbm_vector_dirichlet, surrogate_traction       # noqa: E402
from diffsim.api.ns_bricks import assemble_linear_ns                          # noqa: E402
from diffsim.physics.poisson import gauss_points                              # noqa: E402

# ---------------------------------------------------------------------------
# Constants (match the outflow probe / field-compare)
# ---------------------------------------------------------------------------
DT = 0.05
LEVEL = 4
RE = 1.0        # Re=1: Stokes, isolates pressure coupling
ALPHA = 100.0
STEPS = int(os.environ.get("STEPS", "30"))


# ---------------------------------------------------------------------------
# Monolithic march (mirrors p2r2a_field_compare._march_monolithic /
# p2r0_task10_sphere_derisk.monolithic_cd EXACTLY) — RETURNS the full and free
# node-major (u,p) fields at the end so we can sample interior probe points.
# The saddle solve vector `x` IS the node-major FREE (u,p) vector.
# ---------------------------------------------------------------------------

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

    q = qref()
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
        if step < 3 or (step + 1) % 10 == 0 or step == nsteps - 1:
            print(f"[mono] step{step+1:3d}: Cd={cds[-1]:+.4f}", flush=True)
    # x is node-major FREE (u,p); split velocity comps out.
    xv = x.reshape(nfree, ndof)
    return dict(cds=cds, u_free=xv[:, :dim].copy(), p_free=xv[:, dim].copy())


# ---------------------------------------------------------------------------
# Projection stepper builder (mirrors p2r2a_outflow_probe / field-compare
# outflow-BC branch EXACTLY: standard pressure, ppe_fine_scale=True, outflow
# Dirichlet pressure BC on the x=1 face).
# ---------------------------------------------------------------------------

def _build_projection(fx, pressure_outflow_nodes):
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
    return st


# ---------------------------------------------------------------------------
# Interior probe points (upstream / shoulder / wake of the sphere) -> nearest
# FREE FLUID node indices. Sampled from fx["coords"] (= free-node coords).
# ---------------------------------------------------------------------------

def _probe_nodes(coords, fluid_mask):
    cx, cy, cz = CTR
    pts = {
        "upstream (x=Cx-2R)": np.array([cx - 2.0 * R, 0.5, 0.5]),
        "shoulder (x=Cx, y=Cy+R+)": np.array([cx, cy + R + 0.02, cz]),
        "wake     (x=Cx+3R)": np.array([cx + 3.0 * R, 0.5, 0.5]),
    }
    idx_fluid = np.where(fluid_mask)[0]
    c = coords[idx_fluid]
    probes = {}
    for name, pt in pts.items():
        j = int(np.argmin(np.linalg.norm(c - pt, axis=1)))
        probes[name] = (int(idx_fluid[j]), coords[idx_fluid[j]])
    return probes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print(f"[vd] STEPS={STEPS}  Building level-4 sphere fixture at Re=1 "
          f"(Stokes)...", flush=True)
    device = "cuda:0" if os.environ.get("DIFFSIM_CUDA") else "cpu"
    fx = build_sphere_3d(device, level=LEVEL, Re=RE)
    outflow = outflow_free_nodes(fx)
    coords = fx["coords"]
    n_free, dim = len(coords), fx["dim"]
    strong_mask = fx["strong_mask"]
    fluid_mask = ~strong_mask
    print(f"[vd] fixture ready  n_free={n_free}  "
          f"n_strong={int(strong_mask.sum())}  n_fluid={int(fluid_mask.sum())}  "
          f"n_outflow_free={len(outflow)}  ({time.time()-t0:.1f}s)", flush=True)

    # --- identify strong INFLOW nodes (x~0, u_inf~U_IN in x) and lateral-WALL
    #     strong nodes (rest of the strong set), for the BC-enforcement check ---
    strong = np.where(strong_mask)[0]
    inflow_mask = np.zeros(n_free, dtype=bool)
    inflow_mask[strong[np.abs(coords[strong, 0]) < 1e-12]] = True
    wall_mask = strong_mask & (~inflow_mask)
    print(f"[vd] strong-BC split: n_inflow(x~0)={int(inflow_mask.sum())}  "
          f"n_wall={int(wall_mask.sum())}", flush=True)

    # --- build projection stepper (outflow-BC config) ---
    st = _build_projection(fx, outflow)

    # =====================================================================
    # (2) per-step PREDICTOR vs POST-PROJECTION magnitude.
    #     uhat = st._predict()  (LeraySBMStepper._predict wrapper, SAME
    #     extra_block/sbm_nodes as the step; does NOT advance history/p_star),
    #     called IMMEDIATELY BEFORE the matching st.step().
    #     u_new = st.base._uvec(st.base.hist.pre1) AFTER the step.
    # =====================================================================
    print(f"\n=== (2) PREDICTOR (uhat) vs POST-PROJECTION (u_new) per step ===",
          flush=True)
    print(f"[vd]   uhat obtained via st._predict() (leray_sbm.py L252) "
          f"immediately BEFORE st.step(): extra_block="
          f"st._extra_block(st._current_a_free()) [cached Af_c + backflow(u^n)], "
          f"sbm_nodes=st._sbm_nodes — the SAME args st.step() passes to "
          f"base.step (leray_sbm.py L278-280).", flush=True)
    q = qref()
    for k in range(STEPS):
        ts = time.time()
        # PREDICTOR for THIS step (state = hist.pre1 = u^n, base.t unchanged) —
        # reproduces exactly the uhat st.step() will consume.
        uhat = st._predict()
        # advance the real step (rebuilds the same extra_block from the same
        # _current_a_free(), consumes the same predictor internally).
        st.step()
        u_new = st.base._uvec(st.base.hist.pre1)     # corrected velocity u^{n+1}
        nu_hat = float(np.linalg.norm(uhat))
        nu_new = float(np.linalg.norm(u_new))
        ratio = (nu_new / nu_hat) if nu_hat > 0 else float("nan")
        # interior magnitude on free non-strong (fluid) nodes:
        int_hat = float(np.mean(np.linalg.norm(uhat[fluid_mask], axis=1)))
        int_new = float(np.mean(np.linalg.norm(u_new[fluid_mask], axis=1)))
        F = st.surrogate_traction()
        cd = float(F[0] / q)
        if k < 5 or (k + 1) % 10 == 0 or k == STEPS - 1:
            print(
                f"[vd] step{k+1:3d}: ||uhat||={nu_hat:.4e}  "
                f"||u_new||={nu_new:.4e}  u_new/uhat={ratio:.3f}  "
                f"mean|u|_fluid uhat={int_hat:.4e} u_new={int_new:.4e}  "
                f"Cd={cd:+.4f}  ({time.time()-ts:.1f}s)",
                flush=True,
            )

    # capture the FINAL predictor + corrected fields for the sample table
    uhat_final = st._predict()                         # predictor at final state
    u_new_final = st.base._uvec(st.base.hist.pre1)     # corrected u

    # =====================================================================
    # (1) STRONG-BC enforcement check on the corrected field u_new.
    # =====================================================================
    print(f"\n=== (1) STRONG-BC enforcement (on corrected u_new after "
          f"{STEPS} steps) ===", flush=True)
    ux_inflow = u_new_final[inflow_mask, 0]
    mag_inflow = np.linalg.norm(u_new_final[inflow_mask], axis=1)
    mag_wall = np.linalg.norm(u_new_final[wall_mask], axis=1)
    print(f"[vd]   INFLOW (x~0, expect ux~U_IN={U_IN}): "
          f"ux mean={float(np.mean(ux_inflow)):+.4e} max={float(np.max(ux_inflow)):+.4e} "
          f"min={float(np.min(ux_inflow)):+.4e}  |u| mean={float(np.mean(mag_inflow)):.4e}",
          flush=True)
    print(f"[vd]   WALL   (lateral, expect |u|~0): "
          f"|u| mean={float(np.mean(mag_wall)):.4e} max={float(np.max(mag_wall)):.4e}",
          flush=True)
    inflow_ok = bool(np.allclose(ux_inflow, U_IN, atol=1e-6))
    wall_ok = bool(np.max(mag_wall) < 1e-6)
    print(f"[vd]   strong inflow enforced (ux==U_IN)? {inflow_ok}   "
          f"walls enforced (|u|==0)? {wall_ok}", flush=True)

    # =====================================================================
    # (3) MONOLITHIC reference + interior probe-point comparison.
    # =====================================================================
    print(f"\n=== (3a) MONOLITHIC reference march, {STEPS} steps ===",
          flush=True)
    mono = _march_monolithic(fx, ALPHA, STEPS)
    u_mono = mono["u_free"]
    cd_mono = float(mono["cds"][-1])
    print(f"[vd]   monolithic Cd_final={cd_mono:+.4f}  "
          f"||u_mono||(all free)={np.linalg.norm(u_mono):.4e}  "
          f"||u_mono||(fluid)={np.linalg.norm(u_mono[fluid_mask]):.4e}",
          flush=True)
    print(f"[vd]   projection  ||u_new||(all free)={np.linalg.norm(u_new_final):.4e}  "
          f"||u_new||(fluid)={np.linalg.norm(u_new_final[fluid_mask]):.4e}  "
          f"||uhat||(fluid)={np.linalg.norm(uhat_final[fluid_mask]):.4e}",
          flush=True)

    print(f"\n=== (3b) INTERIOR PROBE POINTS: x-velocity ux "
          f"(mono vs u_new vs uhat) ===", flush=True)
    probes = _probe_nodes(coords, fluid_mask)
    for name, (idx, c) in probes.items():
        print(
            f"[vd]   {name:26s} node{idx:5d} @ {np.round(c,3)}:  "
            f"ux_mono={u_mono[idx,0]:+.4e}  "
            f"ux_u_new={u_new_final[idx,0]:+.4e}  "
            f"ux_uhat={uhat_final[idx,0]:+.4e}",
            flush=True,
        )

    # =====================================================================
    # (4) VERDICT — which half carries the ~16x velocity deficit?
    # =====================================================================
    print(f"\n=== (4) VERDICT ===", flush=True)
    norm_mono = float(np.linalg.norm(u_mono[fluid_mask]))
    norm_uhat = float(np.linalg.norm(uhat_final[fluid_mask]))
    norm_unew = float(np.linalg.norm(u_new_final[fluid_mask]))
    r_hat = (norm_mono / norm_uhat) if norm_uhat > 0 else float("inf")
    r_new = (norm_mono / norm_unew) if norm_unew > 0 else float("inf")
    print(f"[vd]   (a) strong inflow BC enforced (ux~U_IN=1)? {inflow_ok}  "
          f"(inflow ux mean={float(np.mean(ux_inflow)):+.4e}); "
          f"walls |u|~0? {wall_ok}", flush=True)
    print(f"[vd]   (b) mono/uhat magnitude ratio (fluid) = {r_hat:.2f}x   "
          f"mono/u_new ratio (fluid) = {r_new:.2f}x", flush=True)
    # Is the predictor ALREADY weak (>~5x), or O(1) (<~2x)?
    predictor_weak = r_hat > 3.0
    projection_collapse = (r_hat < 2.0) and (r_new > 3.0)
    if predictor_weak:
        verdict = ("PREDICTOR carries the deficit: uhat is already "
                   f"~{r_hat:.1f}x weaker than monolithic. Bug is in the "
                   "momentum PREDICTOR (candidates: SBM Nitsche extra_block "
                   "over-damping the whole field / tau-sigma mis-scale in 3-D, "
                   "or strong-BC not propagating).")
    elif projection_collapse:
        verdict = ("PROJECTION carries the deficit: uhat is O(1)-correct "
                   f"(mono/uhat={r_hat:.1f}x) but u_new collapses "
                   f"(mono/u_new={r_new:.1f}x). Bug is in the PPE / "
                   "velocity-update (Step 2/3).")
    else:
        verdict = (f"INCONCLUSIVE split: mono/uhat={r_hat:.1f}x, "
                   f"mono/u_new={r_new:.1f}x — inspect the per-step trace and "
                   "probe table above.")
    print(f"[vd]   (c) {verdict}", flush=True)
    print(f"\n[vd] done ({time.time()-t0:.1f}s total)", flush=True)


if __name__ == "__main__":
    main()
