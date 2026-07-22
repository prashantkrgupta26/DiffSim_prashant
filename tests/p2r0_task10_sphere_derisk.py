"""P2-R0 Task 10 — sphere 3-D de-risk probe (gpubox helper, NOT an in-CI gate).

LIGHT 3-D confirmation that the composed `LeraySBMStepper`
(projection + volumetric SBM) WORKS in 3-D: it marches to a physical state on a
small feasible mesh and its drag matches the 3-D MONOLITHIC (no-split) SBM-NS
solver on the SAME mesh. R0 is accepted at the DE-RISK bar — literature Cd
convergence is DEFERRED to R2 (Baskar's decision), so this is NOT a mesh
convergence study to the literature sphere Cd (~0.6-0.7 at Re300). We expect
leak-drag / blockage on a small confined domain at feasible mesh (Task-9
finding); the FAITHFULNESS bar is the projection Cd matching the monolithic Cd
on the matched mesh, not the literature value.

This module is the gpubox probe that (a) sweeps the Nitsche penalty alpha to
find the stable regime where the projection march gives a physical (positive,
finite) Cd, and (b) computes the 3-D monolithic reference on the same mesh at
the chosen alpha. It writes tests/baselines/p2r0_task10_sphere.json. The results
seed the in-CI gate test_g5_sphere_3d_derisk in test_p2r0_projection_sbm.py.

Geometry mirrors tests/test_sphere.py (the M1b immersed-sphere smoke):
Sphere(CTR=(0.35,0.5,0.5), R=0.12), unit-box octree, ndof=4, strong inflow/walls,
free outflow (x=1), weak no-slip SBM sphere.
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

sys.path.insert(0, os.path.dirname(__file__))

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.geometry.csg import Sphere
from diffsim.sbm.surrogate import (classify_lambda, extract_surrogate,
                                    GeometryData)
from diffsim.sbm.vector import sbm_vector_dirichlet, surrogate_traction
from diffsim.api.ns_bricks import assemble_linear_ns
from diffsim.solvers.linsolve import solve_linear
from diffsim.physics.poisson import gauss_points
from diffsim.steppers.leray_sbm import LeraySBMStepper

R, CTR, U_IN = 0.12, (0.35, 0.5, 0.5), 1.0


def build_sphere_3d(device, level, Re):
    """3-D immersed-sphere fixture (mirrors tests/test_sphere.py wiring)."""
    ndof, dim = 4, 3
    nu = 2 * U_IN * R / Re
    oracle = Sphere(CTR, R)
    tree = build_uniform(level, dim=3)
    ret, _ = classify_lambda(tree, oracle, 0.5, domain="outside")
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3), device)
    geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, 3),
                                domain="outside")
    coords = mesh.node_coords[cons.free_nodes]
    on = lambda v, c: np.abs(coords[:, c] - v) < 1e-12
    # strong inflow (x=0) + lateral walls (y,z faces); outflow (x=1) free.
    strong = np.where(on(0.0, 0) | on(0.0, 1) | on(1.0, 1)
                      | on(0.0, 2) | on(1.0, 2))[0]
    strong_mask = np.zeros(len(coords), dtype=bool)
    strong_mask[strong] = True
    u_inf = np.zeros((len(coords), dim))
    inflow = strong[np.abs(coords[strong, 0]) < 1e-12]
    u_inf[inflow, 0] = U_IN
    return dict(oracle=oracle, dm=dm, sf=sf, geo=geo, mesh=mesh, cons=cons,
                strong_mask=strong_mask, u_inf=u_inf, coords=coords,
                nu=nu, ndof=ndof, dim=dim)


def outflow_free_nodes(fx, tol=1e-12):
    """FREE-node indices on the outflow face (x = x_max = 1.0) of the unit box.

    These are the nodes where the Taly-style physical Dirichlet pressure BC is
    imposed on the PPE (``pressure_outflow_nodes``). The fixture is an EXTERNAL
    flow with strong inflow (x=0) + lateral walls and a FREE OUTFLOW at x=1
    (see build_sphere_3d). Indexing is in the stepper's FREE-node space
    (``coords`` = mesh.node_coords[cons.free_nodes]), consistent with
    ``LerayProjectionStepper.pressure_outflow_nodes``.
    """
    coords = fx["coords"]
    return np.where(np.abs(coords[:, 0] - 1.0) < tol)[0]


def qref():
    return 0.5 * U_IN ** 2 * np.pi * R ** 2      # frontal area (as test_sphere)


def march_projection(fx, alpha, dt, max_steps, rate_tol, order=2,
                     beta_backflow=1.0, picard_iters=2,
                     velocity_update="consistent", graddiv_scale=1.0):
    """March LeraySBMStepper (projection + SBM) to steady; return dict."""
    dim = fx["dim"]

    def f_fn(x, t):
        return np.zeros((len(x), dim))

    st = LeraySBMStepper(
        fx["oracle"], fx["dm"], fx["nu"], dt, f_fn,
        u_inf=fx["u_inf"], strong_mask=fx["strong_mask"],
        lam=0.5, domain="outside", order=order, picard_iters=picard_iters,
        solver="splu", ppe_finescale=False, alpha=alpha,
        beta_backflow=beta_backflow, velocity_update=velocity_update,
        graddiv_scale=graddiv_scale)
    st.set_initial(lambda c: np.zeros((len(c), dim)))
    q = qref()
    cd_prev = None
    steps = 0
    orders = []
    for step in range(max_steps):
        ts = time.time()
        u, p = st.step()
        F = st.surrogate_traction()
        cd = float(F[0] / q)
        orders.append(int(st.base.order))
        steps = step + 1
        if step < 3 or step % 10 == 0:
            print(f"[task10]   proj step{step+1} Cd={cd:+.4f} "
                  f"ord={st.base.order} ({time.time()-ts:.1f}s)", flush=True)
        if cd_prev is not None and step > 10 and abs(cd - cd_prev) / dt < rate_tol:
            break
        cd_prev = cd
    clat = float(np.hypot(F[1], F[2]) / q)
    finite = bool(np.isfinite(u).all() and np.isfinite(p).all())
    bdf2_engaged = 2 in orders
    return dict(cd=cd, clat=clat, steps=steps, finite=finite,
                bdf2_engaged=bdf2_engaged, st=st, u=u, p=p)


def monolithic_cd(fx, alpha, dt, max_steps, rate_tol, solver="splu"):
    """3-D monolithic (NO projection split) SBM-NS steady Cd on the SAME mesh —
    the apples-to-apples de-risk reference. Mirrors tests/test_sphere.py.

    ``solver`` routes the per-step saddle solve through ``solve_linear`` — use
    ``"cudss"`` (GPU direct) on a cuda-device fixture to reach meshes past the
    host-``splu`` wall (splu stalls ~level-5/143k DOF)."""
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

    x = np.zeros(nfree * ndof)
    sigma = 1.0 / dt
    prev_u = None
    cd = None
    steps = 0

    # ---- blockamgx meta (geometry-static; assembled ONCE) ----------------
    # For solver="blockamgx" the monolithic saddle is solved by the
    # host-orchestrated block-preconditioned FGMRES (Cahouet-Chabard Schur +
    # AMG-on-F, block_precond.BlockAMGPreconditioner). It needs the pressure
    # stiffness Kp and mass-diagonal Mp_diag on the SAME free-node pressure
    # space, pinned at the SAME node the monolithic march pins the pressure
    # DOF, plus the strong-Dirichlet velocity row ids. All are geometry-
    # static (sigma=1/dt is constant here), so we build them ONCE before the
    # Picard loop and reuse the cache every step. This mirrors the R2b.1
    # de-risk build_saddle contract exactly (tests/p2r2b1_blockamgx_derisk.py).
    solve_cache = None
    cache_key = None
    if solver == "blockamgx":
        from diffsim.steppers.leray import LerayProjectionStepper
        # reuse the Leray scalar pressure stiffness K_p = T^T K T and
        # consistent mass M = T^T M T (both free-node scalar pressure space).
        stp = LerayProjectionStepper(
            dm, nu, dt,
            lambda xx, t: np.zeros((len(xx), dim)),   # f_fn (body force)
            lambda xx, t: np.zeros((len(xx), dim)),   # g_fn (Dirichlet data)
            solver="splu")
        p_pin = int(np.argmax(coords.sum(1)))         # scalar pressure node
        Kp = stp.K_p.tolil()
        Kp.rows[p_pin] = [p_pin]                       # pin consistently with
        Kp.data[p_pin] = [1.0]                         # the monolithic p pin
        Kp = Kp.tocsr()
        Mp_diag = np.asarray(stp.M.diagonal()).copy()
        Mp_diag[p_pin] = 1.0
        # strong-Dirichlet velocity row ids (the SAME identity rows the march
        # overwrites below): free-node i x ndof + component c, c in [0, dim).
        dir_rows = np.asarray(
            [int(i) * ndof + c for i in strong for c in range(dim)],
            dtype=np.int64)
        cache_key = "monolithic_cd"
        _meta = dict(
            n_nodes=nfree, ndof=ndof, Kp=Kp, Mp_diag=Mp_diag,
            sigma=sigma, nu=nu, dir_rows=dir_rows)
        # Optional inner-solve tuning knobs from the ENVIRONMENT so the
        # controller can sweep toward mesh-independent convergence without
        # code edits. Only set a meta key when the env var is present; else
        # omit it so the preconditioner default (== current behavior) holds.
        for _env, _key, _cast in (
                ("F_ITERS", "f_iters", int),
                ("F_TOL", "f_tol", float),
                ("KP_ITERS", "kp_iters", int),
                ("KP_TOL", "kp_tol", float),
                ("F_CYCLES", "f_cycles", int),
                ("KP_CYCLES", "kp_cycles", int),
                ("GMRES_RESTART", "gmres_restart", int),
                ("GMRES_MAXITER", "gmres_maxiter", int)):
            if os.environ.get(_env):
                _meta[_key] = _cast(os.environ[_env])
        # Schur approximation mode (string, no cast). Only set when present;
        # absent -> preconditioner default "cahouet_chabard" (== current).
        if os.environ.get("SCHUR_MODE"):
            _meta["schur_mode"] = os.environ["SCHUR_MODE"]
        solve_cache = {("blockamgx_meta", cache_key): _meta}

    for step in range(max_steps):
        u_node = x.reshape(nfree, ndof)[:, :dim]
        aq, dq = gp_field(u_node)
        fq = {pv: aq[pv] / dt for pv in xq}
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
        if solver == "splu":
            x = splu(A.tocsr().tocsc()).solve(b)
        elif solver == "blockamgx":
            x = solve_linear(A.tocsr(), b, solver=solver, sym=False,
                             device=dm.device, cache=solve_cache,
                             cache_key=cache_key)
        else:
            x = solve_linear(A.tocsr(), b, solver=solver, sym=False,
                             device=dm.device)
        u_new = x.reshape(nfree, ndof)[:, :dim]
        F = surrogate_traction(dm, sf, geo, np.asarray(T_vec @ x), nu, ndof)
        cd = float(F[0] / qref())
        steps = step + 1
        if step < 3 or step % 10 == 0:
            print(f"[task10]   mono step{step+1} Cd={cd:+.4f}", flush=True)
        if prev_u is not None and step > 10:
            if np.abs(u_new - prev_u).max() / dt < rate_tol:
                break
        prev_u = u_new.copy()
    return dict(cd=cd, steps=steps)


def main():
    device = "cuda:0" if os.environ.get("DIFFSIM_CUDA") else "cpu"
    level = int(os.environ.get("LEVEL", "4"))
    Re = float(os.environ.get("RE", "100"))
    dt = float(os.environ.get("DT", "0.05"))
    max_steps = int(os.environ.get("MAXSTEPS", "120"))
    rate_tol = float(os.environ.get("RATETOL", "5e-3"))
    order = int(os.environ.get("ORDER", "2"))
    beta = float(os.environ.get("BETA", "1.0"))
    picard = int(os.environ.get("PICARD", "2"))
    vupd = os.environ.get("VUPD", "consistent")
    gds = float(os.environ.get("GDS", "1.0"))
    alphas = [float(a) for a in
              os.environ.get("ALPHAS", "10,20,50,100").split(",")]

    print(f"[task10] sphere 3-D de-risk: level={level} Re={Re} dt={dt} "
          f"max_steps={max_steps} rate_tol={rate_tol}", flush=True)
    t0 = time.time()
    fx = build_sphere_3d(device, level, Re)
    dh = 2 * R / (1.0 / 2 ** level)
    print(f"[task10] built: n_free={len(fx['coords'])} sf.elem={fx['sf'].elem.size} "
          f"D/h={dh:.2f} ({time.time()-t0:.1f}s)", flush=True)

    results = {"config": dict(level=level, Re=Re, dt=dt, R=R, CTR=list(CTR),
                              U_IN=U_IN, max_steps=max_steps, rate_tol=rate_tol,
                              n_free=int(len(fx["coords"])),
                              sf_faces=int(fx["sf"].elem.size), D_over_h=dh),
               "alpha_sweep": []}

    if os.environ.get("MONO_ONLY"):
        for alpha in alphas:
            t0 = time.time()
            mono = monolithic_cd(fx, alpha, dt, max_steps, rate_tol)
            print(f"[task10] MONO alpha={alpha}: Cd={mono['cd']:+.4f} "
                  f"steps={mono['steps']} ({time.time()-t0:.1f}s)", flush=True)
        print("[task10] verdict = mono_only", flush=True)
        return

    best = None
    for alpha in alphas:
        t0 = time.time()
        pr = march_projection(fx, alpha, dt, max_steps, rate_tol, order=order,
                              beta_backflow=beta, picard_iters=picard,
                              velocity_update=vupd, graddiv_scale=gds)
        dt_wall = time.time() - t0
        row = dict(alpha=alpha, cd=pr["cd"], clat=pr["clat"], steps=pr["steps"],
                   finite=pr["finite"], bdf2=pr["bdf2_engaged"],
                   wall_s=round(dt_wall, 1))
        results["alpha_sweep"].append(row)
        print(f"[task10] proj alpha={alpha:6.1f}: Cd={pr['cd']:+.4f} "
              f"Clat={pr['clat']:.4f} steps={pr['steps']} "
              f"bdf2={pr['bdf2_engaged']} finite={pr['finite']} "
              f"({dt_wall:.1f}s)", flush=True)
        # "stable & physical" := finite, positive drag, small lateral (axisym).
        if pr["finite"] and pr["cd"] > 0 and pr["clat"] < 0.1 * abs(pr["cd"]):
            if best is None or abs(pr["cd"]) > 0:
                best = (alpha, pr)

    if best is None:
        print("[task10] NO stable positive-Cd alpha found in sweep", flush=True)
        results["verdict"] = "no_stable_alpha"
    else:
        alpha, pr = best
        t0 = time.time()
        mono = monolithic_cd(fx, alpha, dt, max_steps, rate_tol)
        rel = abs(pr["cd"] - mono["cd"]) / abs(mono["cd"])
        print(f"[task10] BEST alpha={alpha}: proj Cd={pr['cd']:.4f}  "
              f"MONOLITHIC Cd={mono['cd']:.4f}  rel_diff={rel:.3%}  "
              f"({time.time()-t0:.1f}s)", flush=True)
        results["best"] = dict(
            alpha=alpha, proj_cd=pr["cd"], proj_clat=pr["clat"],
            proj_steps=pr["steps"], mono_cd=mono["cd"], mono_steps=mono["steps"],
            rel_diff=rel, bdf2_engaged=pr["bdf2_engaged"])
        results["verdict"] = "ok" if rel < 0.20 else "cd_mismatch"

    out = os.path.join(os.path.dirname(__file__), "baselines",
                       "p2r0_task10_sphere.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[task10] wrote {out}", flush=True)
    print(f"[task10] verdict = {results.get('verdict')}", flush=True)


if __name__ == "__main__":
    main()
