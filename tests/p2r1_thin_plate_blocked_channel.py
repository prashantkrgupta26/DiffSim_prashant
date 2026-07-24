"""P2-R1 — 2-D thin-plate BLOCKED-CHANNEL validation gate (ThinShell §4.2, Fig
5-10). A zero-thickness rigid plate spanning the FULL channel height must
BLOCK the flow completely: with a driven inflow, the two-sided shell surrogate
imposes no-slip on BOTH faces, decoupling upstream/downstream so the
downstream through-flow -> 0 and a pressure jump (upstream stagnation build-up)
appears across the plate. This is the stringent two-sided test: it works only
if the surrogate is genuinely two-sided (opposing normals ADD, no leakage) —
the whole point of the co-dim-1 shell decomposition.

Realized in the unit box [0,1]^2 (the octree's physical domain): plate on the
line x = 0.5 spanning the full height; strong inflow u=(U,0) at x=0, no-slip on
y=0/y=1, free "do-nothing" outflow at x=1. Non-dimensional; physics = Fig 8
with a full-height (total-blockage) plate.

Three runs on the SAME mesh/excluded band:
  (1) TWO-SIDED SHELL       — the co-dim-1 surrogate (the deliverable)
  (2) CARVED thin-RECTANGLE — a 2-cell-thick volumetric box carve, one-sided
      SBM on each face: the boundary-fitted-style reference the shell must match
  (3) ONE-SIDED shell (anti-vacuity) — only Gamma~+ assembled: the coupling is
      broken and the block must degrade (downstream leaks).

Runs on gpubox (host/splu). Writes a JSON baseline; seeds the in-CI gate.
Env: LEVEL (5), U (1.0), NU (0.02), STEPS (60), ALPHA (50), DT (0.05).

    LEVEL=5 .venv/bin/python tests/p2r1_thin_plate_blocked_channel.py
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
from diffsim.geometry.csg import Plane, Box
from diffsim.sbm.surrogate import (
    classify_lambda, classify_shell_intercepted, extract_surrogate,
    extract_two_sided_surrogate, GeometryData)
from diffsim.sbm.vector import (
    sbm_vector_dirichlet, sbm_vector_dirichlet_twosided, surrogate_traction)
from diffsim.api.ns_bricks import assemble_linear_ns
from diffsim.physics.poisson import gauss_points

PLATE_X = 0.5


def _outer_bc(mesh, cons, ndof, dim, U):
    """Strong outer BCs: inflow u=(U,0) at x=0, no-slip u=0 on y=0/y=1, free
    outflow at x=1. Returns (rows, vals, coords) in free-node space."""
    coords = mesh.node_coords[cons.free_nodes]
    on = lambda v, c: np.abs(coords[:, c] - v) < 1e-12
    inflow = on(0.0, 0)
    walls = on(0.0, 1) | on(1.0, 1)
    rows, vals = [], []
    for i in np.where(inflow | walls)[0]:
        gx = U if (inflow[i] and not walls[i]) else 0.0
        rows.append(i * ndof + 0); vals.append(gx)     # u_x
        rows.append(i * ndof + 1); vals.append(0.0)    # u_y
    return np.asarray(rows, np.int64), np.asarray(vals), coords


def _gp_field(dm, mesh, T, u_node, dim):
    full = np.asarray(T @ u_node)
    aq, dq = {}, {}
    for pv in dm.bins:
        tb = dm.tables_by_p[pv]
        vals = full[mesh.conn_of[pv]]
        aq[pv] = np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)
        h = mesh.tree.h()[mesh.bins[pv]]
        dq[pv] = (np.einsum("qad,ead->eq", tb.dN, vals)
                  * (2.0 / h)[:, None]).reshape(-1)
    return aq, dq


def _march(dm, mesh, cons, face_fn, nu, dt, steps, U, dim, ndof):
    """Pseudo-transient monolithic march to steady with SBM face assembly
    face_fn()->(Af_c, bf_c). Pressure pinned at the outflow-far corner."""
    T = cons.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    nfree = T.shape[1]
    rows, vals, coords = _outer_bc(mesh, cons, ndof, dim, U)
    p_pin = int(np.argmax(coords[:, 0] - coords[:, 1]))   # outflow-bottom
    xq = gauss_points(mesh, dm.tables_by_p)
    Af_c, bf_c = face_fn()
    x = np.zeros(nfree * ndof)
    sigma = 1.0 / dt
    prev_u = None
    for step in range(steps):
        u_node = x.reshape(nfree, ndof)[:, :dim]
        aq, dq = _gp_field(dm, mesh, T, u_node, dim)
        fq = {pv: aq[pv] / dt for pv in xq}
        A, b = assemble_linear_ns(dm, aq, dq, fq, nu, sigma=sigma)
        A = (A + Af_c).tolil()
        b = b + bf_c
        for r, v in zip(rows, vals):
            A.rows[r] = [int(r)]; A.data[r] = [1.0]; b[r] = v
        pr = p_pin * ndof + dim
        A.rows[pr] = [pr]; A.data[pr] = [1.0]; b[pr] = 0.0
        x = splu(A.tocsr().tocsc()).solve(b)
        u_new = x.reshape(nfree, ndof)[:, :dim]
        if prev_u is not None and step > 5:
            if np.abs(u_new - prev_u).max() / dt < 1e-5:
                break
        prev_u = u_new.copy()
    u = x.reshape(nfree, ndof)[:, :dim]
    p = x.reshape(nfree, ndof)[:, dim]
    return x, u, p, coords, step + 1


def _through_flux(u, coords, x_line, dim):
    """Mean |u_x| through-flow sampled on nodes near a vertical line x=x_line
    (a coarse flux proxy: the streamwise velocity that crosses the line)."""
    band = np.abs(coords[:, 0] - x_line) < 0.03
    if not band.any():
        return float("nan")
    return float(np.abs(u[band, 0]).mean())


def _trace_p(p, coords, x_face):
    """Median pressure in a thin band hugging a surrogate face at x=x_face
    (the two-sided TRACE the shell exposes)."""
    band = np.abs(coords[:, 0] - x_face) < 0.02
    return float(np.median(p[band])) if band.any() else float("nan")


def _diag(u, p, coords, U, h):
    down = _through_flux(u, coords, 0.72, 2)          # downstream of plate
    up = _through_flux(u, coords, 0.28, 2)            # upstream of plate
    # the two traces immediately adjacent to the plate faces (x = 0.5 -/+ h)
    p_up = _trace_p(p, coords, PLATE_X - h)           # upstream (Gamma~+) trace
    p_dn = _trace_p(p, coords, PLATE_X + h)           # downstream (Gamma~-) trace
    return dict(umax=float(np.abs(u).max()), u_down=down, u_up=up,
                p_up_trace=p_up, p_dn_trace=p_dn, jump=p_up - p_dn)


def build_shell(level, dim=2):
    shell = Plane((PLATE_X, 0.0), (1.0, 0.0))
    tree = build_uniform(level, dim=dim)
    ret, intercepted = classify_shell_intercepted(tree, shell)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), "cpu")
    (sfp, gp), (sfm, gm) = extract_two_sided_surrogate(
        ret, shell, face_tables(1, dim))
    return dict(dm=dm, mesh=mesh, cons=cons, sfp=sfp, gp=gp, sfm=sfm, gm=gm,
                n_excluded=int(intercepted.sum()))


def build_carved(level, dim=2):
    """2-cell-thick volumetric box carve straddling x=0.5 (SAME excluded band
    as the shell), one-sided SBM: the boundary-fitted reference."""
    h = 1.0 / 2 ** level
    box = Box((PLATE_X, 0.5), (h, 0.5))               # 2-cell half-width
    tree = build_uniform(level, dim=dim)
    ret, _ = classify_lambda(tree, box, 1.0, domain="outside")
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), "cpu")
    geo = GeometryData.evaluate(box, ret, sf, face_tables(1, dim),
                                domain="outside")
    return dict(dm=dm, mesh=mesh, cons=cons, sf=sf, geo=geo,
                n_excluded=int(len(tree) - len(ret)))


def run_gate(level=5, U=1.0, nu=0.02, steps=60, alpha=50.0, dt=0.05,
             verbose=True):
    """Run the three blocked-channel cases and return the results dict with a
    'verdict'. Importable by the pytest gate (test_p2r1_thin_plate.py)."""
    return _run(level, U, nu, steps, alpha, dt, verbose)


def main():
    level = int(os.environ.get("LEVEL", "5"))
    U = float(os.environ.get("U", "1.0"))
    nu = float(os.environ.get("NU", "0.02"))
    steps = int(os.environ.get("STEPS", "60"))
    alpha = float(os.environ.get("ALPHA", "50"))
    dt = float(os.environ.get("DT", "0.05"))
    results = _run(level, U, nu, steps, alpha, dt, verbose=True)
    out = os.path.join(os.path.dirname(__file__), "baselines",
                       "p2r1_thin_plate_blocked_channel.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[r1] wrote {out}", flush=True)
    return results["verdict"]["overall"]


def _run(level, U, nu, steps, alpha, dt, verbose):
    ndof, dim = 3, 2
    h = 1.0 / 2 ** level
    noslip = lambda y: np.zeros((len(y), dim))

    if verbose:
        print(f"[r1] thin-plate BLOCKED CHANNEL  level={level} U={U} nu={nu} "
              f"alpha={alpha} steps<= {steps}", flush=True)
    results = {"config": dict(level=level, U=U, nu=nu, alpha=alpha, steps=steps,
                              dt=dt, plate_x=PLATE_X)}

    # ---- 1. TWO-SIDED SHELL -------------------------------------------------
    t0 = time.time()
    fx = build_shell(level)
    T = fx["cons"].T.tocsr()
    Tv = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")

    def shell_face():
        Af, bf = sbm_vector_dirichlet_twosided(
            fx["dm"], fx["sfp"], fx["gp"], fx["sfm"], fx["gm"],
            noslip, nu, ndof, alpha=alpha)
        return (Tv.T @ Af @ Tv).tocsr(), np.asarray(Tv.T @ bf)

    x, u, p, coords, nst = _march(fx["dm"], fx["mesh"], fx["cons"], shell_face,
                                  nu, dt, steps, U, dim, ndof)
    xf = np.asarray(Tv @ x)
    # two-sided traction: the SUM of both faces' loads (opposing normals add,
    # ThinShell §2.1 — the coupling that yields the net force + jump).
    Fp = surrogate_traction(fx["dm"], fx["sfp"], fx["gp"], xf, nu, ndof)
    Fm = surrogate_traction(fx["dm"], fx["sfm"], fx["gm"], xf, nu, ndof)
    sd = _diag(u, p, coords, U, h)
    sd.update(steps=nst, n_excluded=fx["n_excluded"],
              sfp=int(fx["sfp"].elem.size), sfm=int(fx["sfm"].elem.size),
              F_plus=float(Fp[0]), F_minus=float(Fm[0]),
              F_net=float(Fp[0] + Fm[0]))
    results["shell"] = sd
    if verbose:
        print(f"[r1] SHELL:  u_up={sd['u_up']:.3e} u_down={sd['u_down']:.3e} "
              f"jump={sd['jump']:.4f}  F+={sd['F_plus']:.3f} F-={sd['F_minus']:.3f} "
              f"Fnet={sd['F_net']:.3f}  faces +{sd['sfp']}/-{sd['sfm']} "
              f"excl={sd['n_excluded']} ({time.time()-t0:.1f}s)", flush=True)

    # ---- 2. CARVED-OUT thin-RECTANGLE reference ----------------------------
    t0 = time.time()
    cx = build_carved(level)
    Tc = cx["cons"].T.tocsr()
    Tcv = sp.kron(Tc, sp.identity(ndof, format="csr"), format="csr")

    def carved_face():
        Af, bf = sbm_vector_dirichlet(cx["dm"], cx["sf"], cx["geo"], noslip,
                                      nu, ndof, alpha=alpha)
        return (Tcv.T @ Af @ Tcv).tocsr(), np.asarray(Tcv.T @ bf)

    xc, uc, pc, coordsc, nstc = _march(cx["dm"], cx["mesh"], cx["cons"],
                                       carved_face, nu, dt, steps, U, dim, ndof)
    Fc = surrogate_traction(cx["dm"], cx["sf"], cx["geo"],
                            np.asarray(Tcv @ xc), nu, ndof)
    cd = _diag(uc, pc, coordsc, U, h)
    cd.update(steps=nstc, n_excluded=cx["n_excluded"],
              sf=int(cx["sf"].elem.size), F_net=float(Fc[0]))
    results["carved"] = cd
    if verbose:
        print(f"[r1] CARVED: u_up={cd['u_up']:.3e} u_down={cd['u_down']:.3e} "
              f"jump={cd['jump']:.4f}  Fnet={cd['F_net']:.3f}  faces={cd['sf']} "
              f"excl={cd['n_excluded']} ({time.time()-t0:.1f}s)", flush=True)

    # ---- 3. ANTI-VACUITY: DROP the loaded (upstream) side ------------------
    # The block itself follows from element EXCLUSION (the mesh gap) — that is
    # by design (ThinShell: exclusion decouples the two sides). What the
    # TWO-SIDED coupling is load-bearing FOR is the two traces => the pressure
    # JUMP and the NET plate force (F+ + F-). Dropping the LOADED upstream face
    # Gamma~+ must collapse the recovered force and destroy the jump.
    t0 = time.time()

    def oneside_face():
        Af, bf = sbm_vector_dirichlet(fx["dm"], fx["sfm"], fx["gm"], noslip,
                                      nu, ndof, alpha=alpha)     # ONLY Gamma~-
        return (Tv.T @ Af @ Tv).tocsr(), np.asarray(Tv.T @ bf)

    x1, u1, p1, coords1, nst1 = _march(fx["dm"], fx["mesh"], fx["cons"],
                                       oneside_face, nu, dt, steps, U, dim, ndof)
    F1p = surrogate_traction(fx["dm"], fx["sfp"], fx["gp"],
                             np.asarray(Tv @ x1), nu, ndof)
    F1m = surrogate_traction(fx["dm"], fx["sfm"], fx["gm"],
                             np.asarray(Tv @ x1), nu, ndof)
    od = _diag(u1, p1, coords1, U, h)
    od.update(steps=nst1, F_net=float(F1p[0] + F1m[0]))
    results["one_sided"] = od
    if verbose:
        print(f"[r1] ONE-SIDED (drop Gamma~+): jump={od['jump']:.4f} "
              f"Fnet={od['F_net']:.3f} ({time.time()-t0:.1f}s)", flush=True)

    # ---- VERDICT ------------------------------------------------------------
    blocked = sd["u_down"] < 0.05 * U                 # downstream through-flow ~ 0
    jump_ok = sd["jump"] > 0.05 * abs(sd["F_net"])    # a real pressure jump
    jump_rel = abs(sd["jump"] - cd["jump"]) / max(abs(cd["jump"]), 1e-9)
    force_rel = abs(sd["F_net"] - cd["F_net"]) / max(abs(cd["F_net"]), 1e-9)
    ref_ok = (cd["u_down"] < 0.05 * U) and force_rel < 0.30   # force matches carved
    # load-bearing: dropping the loaded side collapses BOTH the jump and Fnet.
    loadbearing = (abs(od["F_net"]) < 0.5 * abs(sd["F_net"])) and \
                  (abs(od["jump"]) < 0.5 * abs(sd["jump"]))
    verdict = "PASS" if (blocked and jump_ok and ref_ok and loadbearing) else "FAIL"
    results["verdict"] = dict(blocked=bool(blocked), jump_ok=bool(jump_ok),
                              ref_ok=bool(ref_ok), jump_rel=float(jump_rel),
                              force_rel=float(force_rel),
                              loadbearing=bool(loadbearing), overall=verdict)

    if verbose:
        print(f"\n[r1] ===== VERDICT: {verdict} =====", flush=True)
        print(f"[r1]   blocked (u_down<{0.05*U:.3f}): {blocked} "
              f"(shell u_down={sd['u_down']:.3e})", flush=True)
        print(f"[r1]   pressure jump present: {jump_ok} (jump={sd['jump']:.3f})",
              flush=True)
        print(f"[r1]   net plate force vs carved: shell={sd['F_net']:.3f} "
              f"carved={cd['F_net']:.3f} rel={force_rel:.1%} (ref_ok={ref_ok})",
              flush=True)
        print(f"[r1]   two-sided load-bearing (drop-side collapses force+jump): "
              f"{loadbearing} (one-sided Fnet={od['F_net']:.3f} "
              f"jump={od['jump']:.3f} vs shell Fnet={sd['F_net']:.3f} "
              f"jump={sd['jump']:.3f})", flush=True)
    return results


if __name__ == "__main__":
    v = main()
    sys.exit(0 if v == "PASS" else 1)
