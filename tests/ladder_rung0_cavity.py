"""Projection Validation Ladder — Rung 0 driver: lid-driven cavity.

The BASE-soundness control. It runs BEFORE any obstacle / Nitsche / shift and
asks a single question: does the base pressure-projection scheme reproduce a
KNOWN flow on a clean closed box? Rung 0 is a closed unit-square lid-driven
cavity (no immersed body, no SBM) — the Ghia, Ghia & Shin (1982) benchmark.

Two solves march to steady on the SAME mesh (the Task-1 `build_cavity_2d`
fixture):

  * PROJECTION — the base `LerayProjectionStepper` (the non-SBM path in
    `steppers/leray.py`), single-pass (picard_iters=1), incremental pressure.
    Enclosed flow => the PPE pins pressure at free-node 0 (the stepper's default
    ``pressure_outflow_nodes=None``): the cavity has no in/outflow, so pressure
    is only defined up to a constant and MUST be pinned.
  * MONOLITHIC — `LinearizedMonolithicStepper` on the identical mesh: the
    saddle Stokes/NS solve with the SAME strong lid/wall Dirichlet and the SAME
    single pressure-DOF pin. This is the same-mesh oracle (box-free reference).

Both extract the centerline profiles `u(y)` at x=0.5 and `v(x)` at y=0.5 via
`point_eval_weights` at the Ghia table abscissae. The verdict per Re:

  PASS iff  projection centerline ≈ monolithic centerline (same mesh) within
            ``TOL_PROJ_MONO`` AND both track Ghia within ``TOL_GHIA``,
  and the projection reaches steady (``‖div u‖`` small / bounded).

If the projection CANNOT reproduce Ghia (fails to match monolithic or Ghia),
that is a MAJOR finding: the base projection impl is broken, and any downstream
rung-A failure would be explained by it. The driver prints the numbers plainly.

Run (gpubox, CPU-splu — 2-D cavity is small):
    PYTHONPATH=src:tests .venv/bin/python tests/ladder_rung0_cavity.py
"""
import numpy as np

from diffsim.steppers.leray import LerayProjectionStepper
from diffsim.steppers.linearized import LinearizedMonolithicStepper
from diffsim.mesh.pointeval import point_eval_weights

from ladder_fixtures import build_cavity_2d

# ------------------------------------------------------------------ Ghia tables
# Ghia, Ghia & Shin (1982), Tables I & II. u(y) on x=0.5, v(x) on y=0.5.
#
# NOTE (Task-2 finding): the *u* table (Table I) in
# docs/dev/2026-07-23-ladder-benchmark-refs.md was transcribed WRONG — its
# upper y-stations 0.9531/0.8516/0.7344 were paired with the u-values that
# actually belong to Ghia's 0.9688/0.9609/0.9531 stations (two stations,
# 0.9688 and 0.9609, were dropped), and its Re=400 column below y≈0.6 was
# mis-shifted. The values below are the CANONICAL Ghia Table I (cross-checked
# against tests/test_cavity.py's long-standing Re=100 table). Table II (v) in
# the refs doc is correct and is reproduced faithfully here. The refs doc is
# corrected in the same commit.
GHIA_Y = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344,
                   0.6172, 0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703,
                   0.0625, 0.0547, 0.0000])
GHIA_U = {
    100: np.array([1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151,
                   0.00332, -0.13641, -0.20581, -0.21090, -0.15662, -0.10150,
                   -0.06434, -0.04775, -0.04192, -0.03717, 0.00000]),
    400: np.array([1.00000, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093,
                   0.16256, 0.02135, -0.11477, -0.17119, -0.32726, -0.24299,
                   -0.14612, -0.10338, -0.09266, -0.08186, 0.00000]),
}
GHIA_X = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594,
                   0.8047, 0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781,
                   0.0703, 0.0625, 0.0000])
GHIA_V = {
    100: np.array([0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914,
                   -0.22445, -0.24533, 0.05454, 0.17527, 0.17507, 0.16077,
                   0.12317, 0.10890, 0.10091, 0.09233, 0.00000]),
    400: np.array([0.00000, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827,
                   -0.44993, -0.38598, 0.05186, 0.30174, 0.30203, 0.28124,
                   0.22965, 0.20920, 0.19713, 0.18360, 0.00000]),
}


def _lid_g(x, t):
    """Cavity Dirichlet: lid (y=1) drives +x at U_IN=1; all else no-slip.

    Matches the Task-1 fixture (`u_inf[lid_mask, 0] = U_IN`, walls zero); the
    lid wins at the top corners (standard Ghia practice)."""
    g = np.zeros((len(x), 2))
    g[np.abs(x[:, 1] - 1.0) < 1e-12, 0] = 1.0
    return g


def _centerlines(mesh, dm, u_node):
    """Sample centerline u(y) at x=0.5 and v(x) at y=0.5 at the Ghia points."""
    W_u = point_eval_weights(mesh, np.stack(
        [np.full_like(GHIA_Y, 0.5), GHIA_Y], axis=1))
    W_v = point_eval_weights(mesh, np.stack(
        [GHIA_X, np.full_like(GHIA_X, 0.5)], axis=1))
    T = dm.constraints.T.tocsr()
    u_full = np.asarray(T @ u_node[:, 0])
    v_full = np.asarray(T @ u_node[:, 1])
    return np.asarray(W_u @ u_full), np.asarray(W_v @ v_full)


# ------------------------------------------------------------------ the marches
def march_projection(level, Re, device="cpu", dt=0.05, nsteps=400,
                     rate_tol=None):
    """March the BASE projection stepper (non-SBM) on the cavity to steady.

    Returns (u_node, p_hat, div, steps). Single-pass (picard_iters=1), order-1
    pseudo-time BDF, default enclosed pressure pin (free-node 0)."""
    fx = build_cavity_2d(level, Re, device)
    dm, mesh = fx["dm"], fx["mesh"]
    st = LerayProjectionStepper(
        dm, fx["nu"], dt, f_fn=lambda x, t: np.zeros((len(x), 2)),
        g_fn=_lid_g, order=1, picard_iters=1)
    st.set_initial(lambda x: np.zeros((len(x), 2)))
    prev = None
    steps = 0
    u = None
    for steps in range(1, nsteps + 1):
        u, p = st.step()
        if rate_tol is not None and prev is not None:
            if np.abs(u - prev).max() / dt < rate_tol:
                break
        prev = u.copy()
    div = float(st.divergence_l2())
    return u, p, div, steps, fx


def march_monolithic(level, Re, device="cpu", dt=0.05, nsteps=400,
                     rate_tol=2e-4):
    """March `LinearizedMonolithicStepper` on the SAME cavity mesh to steady.

    Returns (u_node, div, steps). Same strong lid/wall Dirichlet, same single
    pressure-DOF pin (node 0)."""
    fx = build_cavity_2d(level, Re, device)
    dm, mesh = fx["dm"], fx["mesh"]
    st = LinearizedMonolithicStepper(
        dm, fx["nu"], dt, f_fn=lambda x, t: np.zeros((len(x), 2)),
        g_fn=_lid_g, order=1)
    st.set_initial(lambda x: np.zeros((len(x), 2)))
    prev = None
    steps = 0
    x = None
    for steps in range(1, nsteps + 1):
        x = st.step()
        u = x[:, :2]
        if prev is not None and np.abs(u - prev).max() / dt < rate_tol:
            break
        prev = u.copy()
    div = float(st.divergence_l2())
    return x[:, :2], div, steps, fx


# ------------------------------------------------------------------ the verdict
TOL_PROJ_MONO = 0.03   # projection centerline vs monolithic on the SAME mesh
TOL_GHIA = 0.06        # coarse-mesh (level 5-6 p1) vs Ghia 129^2 table


def run_rung0(level=5, res=(100, 400), device="cpu", proj_steps=300,
              mono_steps=400):
    """Full rung-0 driver: projection + monolithic to steady at each Re,
    centerline comparison to each other and to Ghia. Prints PASS/FAIL per Re
    and returns a results dict."""
    results = {}
    all_pass = True
    for Re in res:
        print(f"\n{'='*66}\n Rung 0 — lid-driven cavity  Re={Re}  level={level}"
              f"\n{'='*66}", flush=True)

        u_p, p_p, div_p, steps_p, fx = march_projection(
            level, Re, device, nsteps=proj_steps)
        u_m, div_m, steps_m, _ = march_monolithic(
            level, Re, device, nsteps=mono_steps)
        mesh, dm = fx["mesh"], fx["dm"]

        up_c, vp_c = _centerlines(mesh, dm, u_p)      # projection centerlines
        um_c, vm_c = _centerlines(mesh, dm, u_m)      # monolithic centerlines
        gu, gv = GHIA_U[Re], GHIA_V[Re]

        du_pm = float(np.abs(up_c - um_c).max())      # proj vs mono
        dv_pm = float(np.abs(vp_c - vm_c).max())
        du_pg = float(np.abs(up_c - gu).max())        # proj vs Ghia
        dv_pg = float(np.abs(vp_c - gv).max())
        du_mg = float(np.abs(um_c - gu).max())        # mono vs Ghia
        dv_mg = float(np.abs(vm_c - gv).max())

        proj_mono_ok = (du_pm < TOL_PROJ_MONO) and (dv_pm < TOL_PROJ_MONO)
        proj_ghia_ok = (du_pg < TOL_GHIA) and (dv_pg < TOL_GHIA)
        mono_ghia_ok = (du_mg < TOL_GHIA) and (dv_mg < TOL_GHIA)
        # Steady/divergence gate: for the EQUAL-ORDER VMS projection the
        # pointwise ‖div u‖ is NOT driven to ~0 (the scheme controls the WEAK /
        # PPE-space divergence; the pointwise div_l2 is the wrong gate — the
        # standing R0 lesson, p2r0_divergence_diagnostic.py). We require it
        # FINITE and BOUNDED (steady, not blowing up); the value is reported
        # alongside the monolithic oracle's for context.
        div_ok = np.isfinite(div_p) and div_p < 5.0
        verdict = proj_mono_ok and proj_ghia_ok and mono_ghia_ok and div_ok
        all_pass = all_pass and verdict

        print(f" projection: steps={steps_p}  ‖div u‖={div_p:.3e}")
        print(f" monolithic: steps={steps_m}  ‖div u‖={div_m:.3e}")
        print(f" centerline u(y) @ x=0.5 (proj / mono / Ghia):")
        for i, y in enumerate(GHIA_Y):
            print(f"   y={y:.4f}  {up_c[i]:+.4f} / {um_c[i]:+.4f} / "
                  f"{gu[i]:+.4f}")
        print(f" max|Δ|  proj-mono: u={du_pm:.4f} v={dv_pm:.4f}  "
              f"(tol {TOL_PROJ_MONO})")
        print(f" max|Δ|  proj-Ghia: u={du_pg:.4f} v={dv_pg:.4f}  "
              f"(tol {TOL_GHIA})")
        print(f" max|Δ|  mono-Ghia: u={du_mg:.4f} v={dv_mg:.4f}  "
              f"(tol {TOL_GHIA})")
        print(f" ---> Re={Re}: proj≈mono {proj_mono_ok} | proj≈Ghia "
              f"{proj_ghia_ok} | mono≈Ghia {mono_ghia_ok} | div {div_ok}")
        print(f" ===> Re={Re}  {'PASS' if verdict else 'FAIL'}", flush=True)

        results[Re] = dict(
            up_c=up_c.tolist(), vp_c=vp_c.tolist(),
            um_c=um_c.tolist(), vm_c=vm_c.tolist(),
            du_pm=du_pm, dv_pm=dv_pm, du_pg=du_pg, dv_pg=dv_pg,
            du_mg=du_mg, dv_mg=dv_mg, div_proj=div_p, div_mono=div_m,
            steps_proj=steps_p, steps_mono=steps_m, verdict=verdict)

    print(f"\n{'#'*66}\n RUNG 0 VERDICT: "
          f"{'PASS — base projection reproduces Ghia + matches monolithic' if all_pass else 'FAIL — see per-Re numbers above'}"
          f"\n{'#'*66}", flush=True)
    results["all_pass"] = all_pass
    return results


if __name__ == "__main__":
    import json
    import sys
    lvl = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    out = run_rung0(level=lvl)
    print("\n[json]", json.dumps({k: (v if k == "all_pass" else
          {kk: vv for kk, vv in v.items() if kk not in
           ("up_c", "vp_c", "um_c", "vm_c")}) for k, v in out.items()}))
