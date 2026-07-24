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

import os as _bos, sys as _bsys  # noqa: E402  (benchmark import bootstrap)
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import _bench_bootstrap  # noqa: E402,F401
import sys
import time

import numpy as np

from diffsim import default_device
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
DEV = default_device()
R = 0.19   # canonical (findings 4b(j) Nova verdict): small AND far from dyadics at every level; 0.25 catastrophically dyadic, 0.27 sits in the L7 halo


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
    # solver in {direct, cudss, krylov}; 'direct'=host splu (exact),
    # 'cudss'=GPU direct (exact; the branch HISTORICALLY mislabeled
    # 'krylov' in the driver), 'krylov'=fused bicgstab_dev (iterative,
    # reports iters/res). Diagnostics are OBSERVATIONAL ONLY.
    t0 = time.time()
    ctr = (0.5,) * dim
    oracle = Sphere(ctr, R)
    tree = build_uniform(level, dim=dim)
    ret, _ = classify_lambda(tree, oracle, 1.0, domain="outside")
    sf = extract_surrogate(ret)
    if band_layers == -1:              # p2 EVERYWHERE (comparison mode)
        p_elem, p_face = 2, 2
    elif band_layers > 0:
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
    A, b, meta = prob.assemble(f_star(dim), g_outer_fn=us)
    sinfo = {"solver": solver, "tol": None, "iters": None,
             "cap": None}
    if solver == "direct":
        from scipy.sparse.linalg import splu
        x = splu(A.tocsc()).solve(b)
    elif solver == "cudss":
        from diffsim.solvers.linsolve import solve_linear
        x = solve_linear(A, b, solver="cudss")
    elif solver == "krylov":
        from diffsim.assembly.operators import CSROperator
        from diffsim.solvers.krylov_dev import bicgstab_dev
        KTOL, KCAP = 1e-10, 40000
        op = CSROperator(A.tocsr(), dm.device)
        x, kinfo = bicgstab_dev(op, b, tol=KTOL, atol=1e-14,
                                maxiter=KCAP,
                                diag=np.asarray(A.diagonal()),
                                check_every=200)
        sinfo.update(tol=KTOL, cap=KCAP,
                     iters=int(kinfo.get("iters", -1)))
    else:
        raise ValueError(f"unknown solver {solver!r}")
    # TRUE relative residual — observational, every solver
    sinfo["res"] = float(np.linalg.norm(A @ x - b)
                         / max(np.linalg.norm(b), 1e-300))
    u = np.asarray(dm.constraints.T @ x)
    err = l2_error_masked(dm, u, us, lambda x: oracle.classify(x) > 0)
    n_free = dm.constraints.T.shape[1]
    n_band = (0 if isinstance(p_elem, int)
              else int((np.asarray(p_elem) == 2).sum()))
    return err, time.time() - t0, n_free, n_band, len(sf.elem), sinfo


def _fmt(sinfo, dt):
    parts = [f"{dt:.0f}s", sinfo["solver"]]
    if sinfo["iters"] is not None:
        parts.append(f"iters={sinfo['iters']}/{sinfo['cap']}")
    parts.append(f"res={sinfo['res']:.1e}")
    if sinfo["tol"] is not None:
        parts.append(f"tol={sinfo['tol']:.0e}")
    return ", ".join(parts)


def _parse_flag(argv, name):
    """Extract --name <val> or --name=<val>; returns (value|None,
    remaining argv)."""
    out, val, skip = [], None, False
    for i, a in enumerate(argv):
        if skip:
            skip = False
            continue
        if a == f"--{name}":
            val = argv[i + 1]
            skip = True
        elif a.startswith(f"--{name}="):
            val = a.split("=", 1)[1]
        else:
            out.append(a)
    return val, out


if __name__ == "__main__":
    forced, args = _parse_flag(sys.argv[1:], "solver")
    radius, args = _parse_flag(args, "radius")
    if radius is not None:
        # reassign the module global BEFORE any use: Sphere(ctr, R) in
        # run_case and the header f-string both read it
        R = float(radius)
    dim = int(args[0])
    levels = [int(v) for v in args[1:]]
    log(f"\n===== dim={dim}  (r={R}, lambda=1.0 keep-all, exterior, "
        f"node-band(3) vs p1) =====")
    errs_band, errs_p1 = {}, {}
    for lv in levels:
        auto = "direct" if (dim == 2 or lv <= 5) else "cudss"
        solver = forced if forced not in (None, "auto") else auto
        e, dt, nf, nb, nsf, si = run_case(dim, lv, 3, solver)
        errs_band[lv] = e
        log(f"  band(3) L{lv}: r={R}  err={e:.4e}  dofs={nf}  "
            f"band_elems={nb}  surrfaces={nsf}  [{_fmt(si, dt)}]")
        if dim == 2 or lv <= 5:
            e1, dt1, _, _, _, si1 = run_case(dim, lv, 0, solver)
            errs_p1[lv] = e1
            log(f"  p1-only L{lv}: err={e1:.4e}  [{_fmt(si1, dt1)}]")
    log(f"  orders band(3): " + "  ".join(
        f"L{a}->L{b}: {np.log2(errs_band[a] / errs_band[b]):.2f}"
        for a, b in zip(levels, levels[1:]) if b in errs_band))
    if len(errs_p1) > 1:
        lp = sorted(errs_p1)
        log(f"  orders p1-only: " + "  ".join(
            f"L{a}->L{b}: {np.log2(errs_p1[a] / errs_p1[b]):.2f}"
            for a, b in zip(lp, lp[1:])))
