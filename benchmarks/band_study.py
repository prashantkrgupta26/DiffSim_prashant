"""Comprehensive p2_band study: SBM Neumann with node-adjacent band(3).

Usage:  python benchmarks/band_study.py <dim> <level> [<level> ...]
        e.g.  python benchmarks/band_study.py 2 5 6 7 8
              python benchmarks/band_study.py 3 4 5 6
Measured reference results: m1a-deferred-findings.md, finding 4b(e).
CAUTION: 3-D level >= 6 is bounded by the build_constraints host stage
(hours; findings 6) until its vectorization lands.

2D circle  (r=0.25 @ (0.5,0.5)),  levels 5-8, p1-no-band contrast included.
3D sphere  (r=0.25 @ center),     levels 4-7.

Exterior domain (box minus ball), lambda=1.0 keep-all, strong outer
Dirichlet from u*, SBM Neumann on the ball. MMS:
  2D: u* = sin(pi x) cos(pi y)
  3D: u* = sin(pi x) cos(pi y) sin(pi z)
Results appended to band_study_results.txt as they land.
"""
import sys
import time

import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh, CSROperator
from diffsim.geometry.csg import Sphere
from diffsim.sbm.surrogate import (classify_lambda, extract_surrogate,
                                   GeometryData, p2_band)
from diffsim.sbm.poisson import SBMPoisson
from diffsim.physics.poisson import l2_error_masked

OUT = "band_study_results.txt"      # appended in the cwd
DEV = "cuda:0"
R = 0.25


def log(msg):
    print(msg, flush=True)
    with open(OUT, "a") as fh:
        fh.write(msg + "\n")


def u_star(dim):
    if dim == 2:
        return lambda x: np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
    return lambda x: (np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
                      * np.sin(np.pi * x[:, 2]))


def grad_u(dim):
    if dim == 2:
        def g(x):
            return np.stack([
                np.pi * np.cos(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1]),
                -np.pi * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]),
            ], axis=1)
        return g

    def g(x):
        sx, cx = np.sin(np.pi * x[:, 0]), np.cos(np.pi * x[:, 0])
        sy, cy = np.sin(np.pi * x[:, 1]), np.cos(np.pi * x[:, 1])
        sz, cz = np.sin(np.pi * x[:, 2]), np.cos(np.pi * x[:, 2])
        return np.stack([np.pi * cx * cy * sz, -np.pi * sx * sy * sz,
                         np.pi * sx * cy * cz], axis=1)
    return g


def f_star(dim):
    u = u_star(dim)
    return lambda x: dim * np.pi ** 2 * u(x)


def run_case(dim, level, band_layers, solver):
    t0 = time.time()
    ctr = (0.5,) * dim
    oracle = Sphere(ctr, R)
    tree = build_uniform(level, dim=dim)
    ret, _ = classify_lambda(tree, oracle, 1.0, domain="outside")
    sf = extract_surrogate(ret)
    if band_layers > 0:
        p_elem = p2_band(ret, sf, n_layers=band_layers)
        p_face = int(np.asarray(p_elem)[sf.elem].max())
    else:
        p_elem, p_face = 1, 1
    mesh = build_mesh(ret, p=p_elem)
    cons = build_constraints(mesh)
    if isinstance(p_elem, int):
        tables = basis_tables(p_elem, dim=dim)
    else:
        tables = {pv: basis_tables(pv, dim=dim) for pv in mesh.bins}
    dm = DeviceMesh.from_mesh(mesh, cons, tables, DEV)
    ftab = face_tables(p_face, dim)
    geo = GeometryData.evaluate(oracle, ret, sf, ftab, domain="outside")
    gu = grad_u(dim)

    def q_fn(y):
        n = -(y - np.asarray(ctr)) / np.linalg.norm(
            y - np.asarray(ctr), axis=1, keepdims=True)
        return np.einsum("id,id->i", gu(y), n)

    prob = SBMPoisson(dm, geo=None, sf=None, neumann=(sf, geo, q_fn))
    us = u_star(dim)
    if solver == "direct":
        u = prob.solve(f_fn=f_star(dim), g_outer_fn=us)
    else:
        # 3D large levels: AMGX (scalar elliptic = its wheelhouse; the
        # fused Jacobi-BiCGStab diverges on the nonsym SBM system at 356k
        # DOFs — measured), cuDSS fallback (GPU direct, 48 GB budget)
        A, b, meta = prob.assemble(f_star(dim), g_outer_fn=us)
        try:
            from diffsim.solvers.amgx import amgx_solve
            x = amgx_solve(A, b, sym=False, tol=1e-11)
        except Exception as e:
            log(f"    amgx failed ({str(e)[:60]}); falling back to cuDSS")
            from diffsim.solvers.linsolve import solve_linear
            x = solve_linear(A, b, solver="cudss")
        u = np.asarray(dm.constraints.T @ x)
    err = l2_error_masked(dm, u, us, lambda x: oracle.classify(x) > 0)
    n_free = dm.constraints.T.shape[1]
    n_band = (0 if isinstance(p_elem, int)
              else int((np.asarray(p_elem) == 2).sum()))
    return err, time.time() - t0, n_free, n_band, len(sf.elem)


if __name__ == "__main__":
    dim = int(sys.argv[1])
    levels = [int(v) for v in sys.argv[2:]]
    log(f"\n===== dim={dim}  (r={R}, lambda=1.0 keep-all, exterior, "
        f"node-band(3) vs p1) =====")
    errs_band, errs_p1 = {}, {}
    for lv in levels:
        solver = "direct" if (dim == 2 or lv <= 5) else "krylov"
        e, dt, nf, nb, nsf = run_case(dim, lv, 3, solver)
        errs_band[lv] = e
        log(f"  band(3) L{lv}: err={e:.4e}  dofs={nf}  band_elems={nb}  "
            f"surrfaces={nsf}  [{dt:.0f}s, {solver}]")
        if dim == 2 or lv <= 5:
            e1, dt1, _, _, _ = run_case(dim, lv, 0, solver)
            errs_p1[lv] = e1
            log(f"  p1-only L{lv}: err={e1:.4e}  [{dt1:.0f}s]")
    log(f"  orders band(3): " + "  ".join(
        f"L{a}->L{b}: {np.log2(errs_band[a] / errs_band[b]):.2f}"
        for a, b in zip(levels, levels[1:]) if b in errs_band))
    if len(errs_p1) > 1:
        lp = sorted(errs_p1)
        log(f"  orders p1-only: " + "  ".join(
            f"L{a}->L{b}: {np.log2(errs_p1[a] / errs_p1[b]):.2f}"
            for a, b in zip(lp, lp[1:])))
