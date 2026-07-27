"""A1 — The MMS discipline: manufactured solutions and convergence orders.

LEARNING OUTCOME. You can verify a solver the way this group verifies
everything: pick the answer first (u*), derive the data (f, g) that make it
exact, then MEASURE how fast the discrete solution approaches it. You know
what orders to expect (p+1 in L2) and you distrust any solver claim that
isn't a measured order.

BACKGROUND. Solve -lap(u) = f on the unit square with u = g on the boundary
(strong Dirichlet: the boundary rows of the linear system are replaced by
identity rows). The Method of Manufactured Solutions (MMS): choose
u*(x,y) = sin(pi x) sin(pi y), then f = 2 pi^2 u* and g = u* = 0 on the
boundary. The discrete error should shrink as O(h^{p+1}) in L2 — order 2
for linear (p1) elements, order 3 for quadratic (p2).

EXPECTED RESULTS (RTX-class GPU, FP64):
    p=1: errors ~ 1.61e-3 / 4.02e-4 / 1.00e-4, orders 2.00 / 2.00
    p=2: errors ~ 2.05e-4 / 2.57e-5 / 3.22e-6, orders 2.99 / 3.00
Your exact numbers may differ in the third digit; the ORDERS must not.

Run:  python tutorials/A_foundations/A1_mms_convergence.py
"""
import os
import sys
import time

import numpy as np
from scipy.sparse.linalg import splu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _viz as _viz  # guarded viz helper (no-ops when [viz] not installed)

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh, assemble_csr
from diffsim.physics.poisson import (gauss_points, make_load_kernel,
                                     l2_error_masked)
from diffsim import default_device
import warp as wp

DEVICE = default_device()
u_star = lambda x: x[:, 0] ** 2 + x[:, 1] ** 2
f_star = lambda x: -4.0 * np.ones(len(x))


def solve(level: int, p: int) -> float:
    # mesh + constraints + device tables — the standard preamble
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2), DEVICE)

    # stiffness (GPU element kernels -> constrained CSR on host)
    A = assemble_csr(dm)

    # load vector: f evaluated at Gauss points, integrated by a GPU kernel
    F_full = wp.zeros(dm.n_nodes, dtype=wp.float64, device=DEVICE)
    xq = gauss_points(mesh, dm.tables_by_p)
    for pv, b in dm.bins.items():
        fq = wp.array(f_star(xq[pv]), dtype=wp.float64, device=DEVICE)
        lk = make_load_kernel(b["nbf"], b["nqp"], dm.dim)
        wp.launch(lk, dim=len(b["eids"]),
                  inputs=[b["conn"], b["h"], b["N"], b["w"], fq, F_full],
                  device=DEVICE)
    b_vec = np.asarray(cons.T.T @ F_full.numpy())

    # strong Dirichlet: replace boundary rows with identity, set g values
    coords = mesh.node_coords[cons.free_nodes]
    bdry = np.where(mesh.boundary_nodes[cons.free_nodes])[0]
    A = A.tolil()
    for i in bdry:
        A.rows[i] = [int(i)]
        A.data[i] = [1.0]
        b_vec[i] = u_star(coords[i:i + 1])[0]
    u_free = splu(A.tocsr().tocsc()).solve(b_vec)
    u_all = np.asarray(cons.T @ u_free)
    return l2_error_masked(dm, u_all, u_star, lambda x: np.ones(len(x), bool))


def profile_stages(level: int, p: int = 1):
    """Time mesh/constraints, assembly, and sparse solve separately."""
    t0 = time.perf_counter()

    # Stage 1: mesh + constraints
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2), DEVICE)
    t_mesh = time.perf_counter() - t0

    # Stage 2: matrix and load-vector assembly, including boundary rows
    t0 = time.perf_counter()
    A = assemble_csr(dm)
    F_full = wp.zeros(dm.n_nodes, dtype=wp.float64, device=DEVICE)
    xq = gauss_points(mesh, dm.tables_by_p)
    for pv, b in dm.bins.items():
        fq = wp.array(f_star(xq[pv]), dtype=wp.float64, device=DEVICE)
        lk = make_load_kernel(b["nbf"], b["nqp"], dm.dim)
        wp.launch(lk, dim=len(b["eids"]),
                  inputs=[b["conn"], b["h"], b["N"], b["w"], fq, F_full],
                  device=DEVICE)
    b_vec = np.asarray(cons.T.T @ F_full.numpy())

    coords = mesh.node_coords[cons.free_nodes]
    bdry = np.where(mesh.boundary_nodes[cons.free_nodes])[0]
    A = A.tolil()
    for i in bdry:
        A.rows[i] = [int(i)]
        A.data[i] = [1.0]
        b_vec[i] = u_star(coords[i:i + 1])[0]
    A = A.tocsr().tocsc()
    t_assembly = time.perf_counter() - t0

    # Stage 3: sparse factorization and solve
    t0 = time.perf_counter()
    splu(A).solve(b_vec)
    t_solve = time.perf_counter() - t0

    return A.shape[0], t_mesh, t_assembly, t_solve


def performance_study(levels=range(4, 9)):
    """Print timings and measured exponents for Explore (c)."""
    print("\nPerformance study (p=1)")
    print("level        n       mesh  assembly     solve")

    # The first run may compile Warp kernels; exclude it from measurements.
    profile_stages(4)
    rows = [profile_stages(level) for level in levels]

    for level, (n, tm, ta, ts) in zip(levels, rows):
        print(f"{level:5d} {n:8d} {tm:10.4f} {ta:10.4f} {ts:10.4f}")

    print("measured exponents (log time ratio / log DOF ratio):")
    for name, column in (("mesh", 1), ("assembly", 2), ("solve", 3)):
        exponents = []
        for old, new in zip(rows, rows[1:]):
            n0, t0 = old[0], old[column]
            n1, t1 = new[0], new[column]
            exponents.append(np.log(t1 / t0) / np.log(n1 / n0))
        print(f"  {name:8s}: " + "  ".join(f"{e:.2f}" for e in exponents))


def main(levels_p1=(4, 5, 6), levels_p2=(3, 4, 5)):
    """Run the MMS convergence study and emit a convergence figure."""
    results = {}
    for p, levels in ((1, levels_p1), (2, levels_p2)):
        errs = [solve(lv, p) for lv in levels]
        orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
        print(f"p={p}: errors " + "  ".join(f"{e:.3e}" for e in errs)
              + "   orders " + "  ".join(f"{o:.2f}" for o in orders))
        results[p] = (levels, errs)
    # --- viz (additive; no-ops on base venv) ---
    # p=1 convergence plot
    lv1, e1 = results[1]
    _viz.convergence(__file__, lv1, e1, "convergence_p1",
                     slope=2, xlabel="refinement level", ylabel="L2 error",
                     label="p=1")
    # p=2 convergence plot
    lv2, e2 = results[2]
    _viz.convergence(__file__, lv2, e2, "convergence_p2",
                     slope=3, xlabel="refinement level", ylabel="L2 error",
                     label="p=2")
    return results


if __name__ == "__main__":
    main()
    performance_study()
    print("""
EXPLORE
  (a) Use u* = x^2 + y^2 (f = -4). At p=2 the error should be MACHINE ZERO
      at every level — why? (This is a patch test: the exact solution lies
      in the discrete space.) Confirm it. What does p=1 give, and why is
      that still second order rather than exact?
  (b) Break the MMS on purpose: keep f but set g = 0 everywhere. What order
      do you measure now, and what does that tell you about how boundary
      errors pollute interior accuracy?
  (c) PERFORMANCE CORNER (cf. FEM_computational_cost.pdf): time the three
      stages (mesh+constraints, assemble, solve) separately at levels
      4..8 for p=1. Which stage grows fastest? Fit the exponent of each
      (log2 of consecutive time ratios). Predict before you measure:
      splu on a 2-D mesh should scale ~ O(n^1.5); assembly O(n).
""")
