"""A2 — Boundary conditions: strong Dirichlet, natural Neumann, and what
"do nothing" really imposes.

LEARNING OUTCOME. You can state precisely where each boundary condition
lives in the discrete system: Dirichlet as ROW REPLACEMENT (or a lifted
right-hand side), Neumann as a SURFACE TERM in the weak form — and you know
that omitting that surface term is not "no condition" but the CHOICE
grad(u).n = 0. You verify both with MMS.

BACKGROUND. Multiply -lap(u) = f by a test function w and integrate by
parts:
    int grad(w).grad(u) dV  -  oint w (grad(u).n) dS  =  int w f dV.
Whatever you do with the surface integral IS your boundary condition:
  * Dirichlet faces: w = 0 there (test space), and u is pinned — in code,
    identity rows.
  * Neumann faces: substitute the known flux q = grad(u).n. If q = 0 the
    term VANISHES — the famous "do-nothing" natural condition. Any code
    that assembles only the volume term is silently imposing zero flux on
    every non-Dirichlet face.
This chapter manufactures u* = cos(pi x) cos(pi y), whose normal derivative
is ZERO on all four faces of the unit square. That lets us impose:
  case 1: Dirichlet on all faces (as A1);
  case 2: Dirichlet on x-faces only, NOTHING on y-faces — legitimate,
          because the true flux there is zero.
Both must converge at order 2. (Inhomogeneous Neumann data q != 0 needs an
assembled surface integral; you meet it in A3 on immersed boundaries, where
this library implements it in shifted form.)

EXPECTED RESULTS: both cases order ~2.0; case-2 errors slightly larger
(the Neumann sides are only weakly pinned).

Run:  python tutorials/A_foundations/A2_boundary_conditions.py
"""
import time

import numpy as np
from scipy.sparse.linalg import splu
import warp as wp

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh, assemble_csr
from diffsim.physics.poisson import (gauss_points, make_load_kernel,
                                     l2_error_masked)
from diffsim import default_device

DEVICE = default_device()
u_star = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
f_star = lambda x: 2 * np.pi ** 2 * u_star(x)


def solve(level, dirichlet_faces):
    """dirichlet_faces: 'all', 'x-only', or 'pin'. Non-Dirichlet faces get NOTHING
    — which the weak form interprets as grad(u).n = 0 (true for our u*)."""
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), DEVICE)
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
    on = lambda v, c: np.abs(coords[:, c] - v) < 1e-12
    if dirichlet_faces == "all":
        mask = on(0.0, 0) | on(1.0, 0) | on(0.0, 1) | on(1.0, 1)
    elif dirichlet_faces == "x-only":     # x-faces only; y-faces natural
        mask = on(0.0, 0) | on(1.0, 0)
    else:                                  # pure Neumann: no Dirichlet faces
        mask = np.zeros(len(coords), dtype=bool)

    A = A.tolil()
    if dirichlet_faces == "pin":
        # Fix the additive constant by pinning the node closest to (0, 0).
        pin = np.argmin(np.sum(coords**2, axis=1))
        A.rows[pin] = [int(pin)]
        A.data[pin] = [1.0]
        b_vec[pin] = u_star(coords[pin:pin + 1])[0]
    else:
        for i in np.where(mask)[0]:
            A.rows[i] = [int(i)]
            A.data[i] = [1.0]
            b_vec[i] = u_star(coords[i:i + 1])[0]
    u_free = splu(A.tocsr().tocsc()).solve(b_vec)
    u_all = np.asarray(cons.T @ u_free)
    return l2_error_masked(dm, u_all, u_star, lambda x: np.ones(len(x), bool))


def time_dirichlet_replacement(level):
    """Time the host-side boundary-row replacement loop for Explore (c)."""
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), DEVICE)
    A = assemble_csr(dm).tolil()

    coords = mesh.node_coords[cons.free_nodes]
    mask = (
        (np.abs(coords[:, 0] - 0.0) < 1e-12)
        | (np.abs(coords[:, 0] - 1.0) < 1e-12)
        | (np.abs(coords[:, 1] - 0.0) < 1e-12)
        | (np.abs(coords[:, 1] - 1.0) < 1e-12)
    )
    boundary_ids = np.where(mask)[0]
    b_vec = np.zeros(len(coords))

    t0 = time.perf_counter()
    for i in boundary_ids:
        A.rows[i] = [int(i)]
        A.data[i] = [1.0]
        b_vec[i] = u_star(coords[i:i + 1])[0]
    elapsed = time.perf_counter() - t0

    return len(coords), len(boundary_ids), elapsed


def performance_study(levels=(5, 6, 7, 8)):
    """Print row-replacement timings and scaling exponents."""
    print("\nExplore (c): Dirichlet row-replacement timing")
    print("level        n  boundary_nodes    loop_time")

    rows = [time_dirichlet_replacement(level) for level in levels]
    for level, (n, nb, elapsed) in zip(levels, rows):
        print(f"{level:5d} {n:9d} {nb:16d} {elapsed:12.6f} s")

    print("measured exponents (log time ratio / log DOF ratio):")
    exponents = []
    for old, new in zip(rows, rows[1:]):
        n0, t0 = old[0], old[2]
        n1, t1 = new[0], new[2]
        exponents.append(np.log(t1 / t0) / np.log(n1 / n0))
    print("  row replacement: " + "  ".join(f"{e:.2f}" for e in exponents))


if __name__ == "__main__":
    for case in ("all", "x-only", "pin"):
        errs = [solve(lv, case) for lv in (4, 5, 6)]
        orders = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
        print(f"Dirichlet {case:>6}: errors "
              + "  ".join(f"{e:.3e}" for e in errs)
              + "   orders " + "  ".join(f"{o:.2f}" for o in orders))
    performance_study()
    print("""
EXPLORE
  (a) Change u* to sin(pi x) sin(pi y) and rerun case 2 WITHOUT changing
      anything else. The order collapses — measure it. Explain: what flux
      is the do-nothing side now silently asserting, and what is the true
      flux of this u*?
  (b) Pure-Neumann trap: make ALL faces natural (legal for our cosine u*,
      whose flux is zero everywhere). The solve fails or drifts — the
      matrix is singular (u + constant is also a solution). Fix it by
      pinning one node, as every pressure solve in track D does.
  (c) PERFORMANCE CORNER: the row-replacement loop is host-side Python over
      boundary nodes. Time it at levels 5..8 — when does it stop being
      free? (This exact cost is m1b findings item 4: production replaces
      the loop with a precomputed masked assembly.)
""")
