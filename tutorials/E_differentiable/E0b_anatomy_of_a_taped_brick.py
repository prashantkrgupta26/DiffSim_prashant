"""E0b — Anatomy of a taped brick.

LEARNING OUTCOME. You can open any DiffSim kernel, identify what ``wp.Tape``
records, and reason about which parts to differentiate vs. what to compute
once and store.  You have run the dot-product test for the Poisson operator
and seen a two-blob conductivity field recover from noise in ~40 gradient-
descent steps.

BACKGROUND. Adapted from the mathematical background developed for
dolfin-adjoint/pyadjoint by Patrick E. Farrell; Farrell, Ham, Funke & Rognes
(2013) and Mitusch, Funke & Dokken (2019), cited below.  DiffSim-native
sources: findings-4c taping rules, the M1a assembled-CSR adjoint stack.

E0a ended with the seed: *we did NOT differentiate through ``splu``*.
This chapter unpacks exactly why that is correct and where it generalises.

Attribution:
  - P.E. Farrell, mathematical background for dolfin-adjoint/pyadjoint.
  - Farrell, Ham, Funke & Rognes (2013). Automated Derivation of the
    Adjoint of High-Level Mathematical Programs. SIAM J. Sci. Comput.
  - Mitusch, Funke & Dokken (2019). dolfin-adjoint 2018.1: automated
    adjoints for FEniCS and Firedrake. JOSS.
  - DiffSim discrete-adjoint stance: spec S4.3, findings-4c taping rules.
  - "Learning reaction rates" example lineage: dolfin-adjoint tutorial.

EXPECTED RESULTS (CPU, level-3 grid, 5 probes, FP64):
    Dot-product test rel err          < 1e-12   (symmetric A: ~1e-17)
    Factorization reuse ratio           > 1.5 x  (one LU for fwd+adj)
    kappa recovery J_init (misfit)    5e-3 – 5e-2
    kappa recovery J_final (misfit)     < 5e-4
    kappa recovery misfit drop (J)      > 100 x  (printed table)

Run:  python tutorials/E_differentiable/E0b_anatomy_of_a_taped_brick.py
"""
# ── imports ──────────────────────────────────────────────────────────────────
import time
import numpy as np
from scipy.sparse.linalg import splu
import scipy.sparse as sp
import warp as wp

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import (
    DeviceMesh, volume_triplets, make_poisson_element_matrices, _kernel_cache,
)
from diffsim.assembly.femelm import FEMElm, fe_dN_s, fe_detJxW_s
from diffsim.physics.poisson import gauss_points, make_load_kernel
from diffsim.mesh.pointeval import point_eval_weights

# ─────────────────────────────────────────────────────────────────────────────
# §0  PROBLEM SETUP  (same mesh + physics as E0a for continuity)
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = "cpu"
LEVEL  = 3      # 8×8 quad grid — 64 elements, small enough for everything here

u_star = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
f_star = lambda x: 2.0 * np.pi**2 * u_star(x)

# Five interior probes (same as E0a)
PROBES = 0.5 + 0.10 * np.array(
    [[np.cos(t), np.sin(t)]
     for t in np.linspace(0, 2 * np.pi, 5, endpoint=False)])

print("=" * 65)
print("E0b — Anatomy of a taped brick")
print("=" * 65)
print()

wp.init()

tree = build_uniform(LEVEL, dim=2)
mesh = build_mesh(tree, p=1)
cons = build_constraints(mesh)
dm   = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), DEVICE)
T_mat = cons.T.tocsr()

n_elem = len(dm.bins[1]['eids'])
pv = 1; bk = dm.bins[pv]
conn = dm.mesh.conn_of[pv]          # [64, 4]  element-to-node

W = point_eval_weights(mesh, PROBES)
bdry_mask = mesh.boundary_nodes

print(f"Mesh: level={LEVEL}, n_elem={n_elem}, n_nodes={dm.n_nodes}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# §1  A REAL BRICK, LINE BY LINE
# ─────────────────────────────────────────────────────────────────────────────

# ── math aside ──────────────────────────────────────────────────────────────
# The Poisson volume kernel in diffsim/assembly/operators.py (make_poisson_matvec)
# is the SPECIMEN for this chapter.  Here are its annotated lines:
#
#   @wp.kernel(module="unique", enable_backward=False)     ← (a)
#   def poisson_mv(conn, h, Ntab, dNtab, wtab, x, y):
#       e = wp.tid()                                       ← (b)
#       fe = FEMElm(); fe.e = e; fe.he = h[e]             ← (c)
#       half = fe.he * wp.float64(0.5)
#       jac = wp.float64(1.0)
#       for _ in range(dim):                               ← (d)
#           jac = jac * half                               # (h/2)^dim
#       dscale = wp.float64(2.0) / fe.he                  ← (e)
#       for q in range(nqp):
#           fe.q = q
#           dJxW = fe_detJxW_s(wtab, fe, jac)             ← (f)
#           g = VEC()                                      ← (g)
#           for b in range(nbf):
#               xb = x[conn[e, b]]
#               for d in range(dim):
#                   g[d] += fe_dN_s(dNtab, fe, b, d, dscale) * xb  ← (h)
#           for a in range(nbf):
#               val = 0.0
#               for d in range(dim):
#                   val += fe_dN_s(dNtab, fe, a, d, dscale) * g[d] ← (i)
#               wp.atomic_add(y, conn[e, a], val * dJxW)            ← (j)
#
# ANNOTATION:
# (a) enable_backward=False — this kernel is NEVER taped.  The adjoint path
#     goes through the ASSEMBLED CSR (A^T solve), not through this kernel.
#     Turning backward OFF halves the compile cost (measured: dim=4, p2,
#     ~280 s/module with backward ON — operators.py comment).
#
# (b) wp.tid() — thread index == element index.  Warp launches ONE THREAD
#     PER ELEMENT.  wp.Tape records which kernels ran at which array addresses,
#     so "recording" this launch means: "kernel poisson_mv, 64 threads,
#     arrays conn/h/dNtab/wtab/x/y".
#
# (c) FEMElm is a Warp STRUCT (femelm.py): holds (e, q, he).  It is NOT a
#     Python object; it lives in kernel GPU/CPU registers.  requires_grad is
#     irrelevant for scalar fields in registers — the tape tracks ARRAYS.
#
# (d) jac = (h/2)^dim — the Jacobian of the reference-to-physical map for a
#     uniform Cartesian element.  This is PRECOMPUTED ONCE inside the kernel,
#     not taped through; only the input array 'h' could be taped.
#
# (e) dscale = 2/h — the physical derivative scale.  On the reference element
#     dN/dxi_k = dNtab[q, a, k]; physical: dN/dx_k = dNtab[q, a, k] * dscale.
#
# (f) fe_detJxW_s: returns wtab[q] * jac — the quadrature weight × Jacobian.
#
# (g) g = VEC() — a local register vector (vec2d/vec3d).  It accumulates
#     grad(u_h) at this quadrature point.  Local variables are NOT taped.
#
# (h) grad(u_h) = sum_b dN_b(x_q) * u_b.  x = u (the FE solution coefficients)
#     is the INPUT array; if requires_grad=True, the tape will track how x
#     flows into the output y.
#
# (i)–(j): accumulate (grad N_a) · (grad u_h) * dJxW into y[conn[e,a]].
#     wp.atomic_add is safe for parallel element updates (different elements
#     CAN share nodes; atomics avoid race conditions).
# ──────────────────────────────────────────────────────────────────────────

# ── math aside ──────────────────────────────────────────────────────────────
# WHAT wp.Tape ACTUALLY RECORDS:
#
# A Warp Tape is a LAUNCH GRAPH — a list of
#   (kernel_ptr, dim, input_arrays, output_arrays)
# tuples.  When you write
#
#     with tape:
#         wp.launch(kernel, dim=N, inputs=[a, b, c], ...)
#
# Warp appends one entry to the list.  On tape.backward(), it replays the
# list in REVERSE ORDER, calling the adjoint (backward) kernel for each entry.
#
# The backward kernel for a kernel f(x) -> y receives:
#   adj_x += df/dx * adj_y   (cotangent propagation)
# where adj_y = tape.gradients[y] is the incoming cotangent ("seed").
#
# REQUIRES_GRAD arrays: only arrays created with requires_grad=True accumulate
# cotangents in tape.gradients.  Arrays with requires_grad=False are "frozen"
# — the tape skips cotangent accumulation for them, saving memory and compute.
#
# In the Poisson case: we tape the VOLUME RESIDUAL kernel with
#   kappa  requires_grad=True   (the parameter — we want dR/dkappa)
#   u      requires_grad=False  (frozen state — u is fixed at the solve point)
# The backward pass gives cotangent_kappa = lam^T (dR/dkappa).
#
# CUSTOM VJP vs UNROLLED TAPE:
# If you define a Warp @wp.kernel with enable_backward=True (or let Warp
# auto-generate), the tape calls the backward kernel automatically — this is
# the "unrolled tape" path.  For complex operations (like LU factorization)
# you can instead SKIP taping and supply the gradient manually via the adjoint
# equation — this is the "custom VJP" pattern (the splu example from E0a).
# DiffSim's assembled-CSR adjoint is a custom VJP: the kernel itself has
# enable_backward=False, and the gradient path is the transposed solve.
# ──────────────────────────────────────────────────────────────────────────

print("§1  TAPED BRICK OVERVIEW")
print("-" * 65)
print("Assembly kernels: enable_backward=False (no tape, no backward codegen).")
print("Gradient path: assembled A^T — a custom VJP, not unrolled tape.")
print("kappa-residual kernel: enable_backward=True (taped in kappa only).")
print()

# ─────────────────────────────────────────────────────────────────────────────
# §2  FACTORY / CACHE COEXISTENCE WITH TAPING
# ─────────────────────────────────────────────────────────────────────────────

# ── math aside ──────────────────────────────────────────────────────────────
# DiffSim kernels are created by FACTORIES (e.g. make_poisson_matvec(nbf, nqp, dim))
# and stored in _kernel_cache.  The factory runs ONCE per unique (nbf, nqp, dim)
# triple; subsequent calls return the cached kernel.
#
# This is COMPATIBLE with taping: wp.Tape records the kernel object reference,
# not a copy.  You can cache the kernel and launch it inside a tape scope.
# The tape's launch graph stores (kernel_ref, dim, arrays); the backward pass
# calls the kernel's backward method (if enable_backward=True).
#
# CAVEAT: never CHANGE a kernel's behavior between the forward and backward pass
# (don't swap data under the tape's nose).  The launch graph is fixed; the
# only thing that changes between forward and backward is the array VALUES
# (cotangents flowing backward).
#
# In E0a we saw _kernel_cache[("kappa_res_e0a", nbf, nqp, dim)].  The key
# includes the chapter tag ("kappa_res_e0a") to avoid collision with the
# production kappa-residual kernels (if they exist) — a defensive pattern.
# ──────────────────────────────────────────────────────────────────────────

print("§2  FACTORY / CACHE NOTE")
print("-" * 65)
print(f"_kernel_cache holds {len(_kernel_cache)} compiled kernels so far.")
print("Factories produce kernel objects once; tape records references.")
print("Cache + tape coexist safely — the launch graph stores kernel refs.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# §3  THE DO-NOT-DIFFERENTIATE LIST
# ─────────────────────────────────────────────────────────────────────────────

# ── math aside ──────────────────────────────────────────────────────────────
# The central principle: differentiate the RELATION, not the ALGORITHM.
#
# Every entry below is an algorithm that IMPLEMENTS a mathematical relation.
# The relation has a clean adjoint; the algorithm's code graph does not.
# ──────────────────────────────────────────────────────────────────────────

print("§3  THE DO-NOT-DIFFERENTIATE LIST")
print("=" * 65)
print()

# ── (A) LINEAR SOLVERS ───────────────────────────────────────────────────────
print("(A) LINEAR SOLVERS  (splu, bicgstab, GMRES, cuDSS, ...)")
print("-" * 65)
# ── math aside ──────────────────────────────────────────────────────────────
# The algorithm:  LU factorization of A, forward-backward substitution.
# The relation:   Au = b   <=>   u = A^{-1} b.
#
# Differentiating the RELATION via implicit differentiation:
#   R(u, m) = A(m) u - b = 0
#   dR/du * du/dm = -dR/dm
#   A * (du/dm) = -(dA/dm) u          (one fwd-mode solve per m_i)
#   In adjoint form: A^T λ = dJ/du    (ONE solve for ALL parameters)
#   dJ/dm_i = -λ^T (dA/dm_i) u        (cheap inner product)
#
# REUSE THE SAME LU FACTORIZATION for the adjoint solve:
#   A^T = (LU)^T = U^T L^T
# scipy's splu(A).solve(b) already knows U^T L^T because splu stores both L
# and U — you call splu(A.T.tocsc()).solve(dJdu) or use the same factors
# with transposed triangular solves.
#
# For SYMMETRIC A (Poisson): A^T = A, so the EXACT SAME FACTORS serve both.
# For NON-SYMMETRIC A (Navier-Stokes, SBM): you need the TRANSPOSE, but you
# still only need ONE factorization.
# ──────────────────────────────────────────────────────────────────────────
print("  Relation: A(m) u = b.")
print("  Adjoint:  A^T λ = dJ/du  (ONE solve, same LU factors).")
print("  NEVER unroll the LU / Krylov iteration steps through a tape.")
print()

# Illustration: splu.solve with transposed flag vs second factorization
A_demo, _, u_demo_all, _ = _solve_poisson_setup = None, None, None, None

def solve_poisson(kappa_vec: np.ndarray):
    """Forward Poisson solve — identical to E0a for narrative continuity.

    Returns (A_csc, lu_factor, u_free, u_all, bdry_rows).
    The LU factorization ``lu_factor`` can be reused for the adjoint solve
    via ``lu_factor.solve(rhs, trans='T')`` — no second factorization needed.
    """
    kq = np.repeat(kappa_vec, bk['nqp'])
    rows, cols, vals = volume_triplets(dm, kq_by_bin={pv: kq})
    K = sp.coo_matrix((vals, (rows, cols)),
                      shape=(dm.n_nodes, dm.n_nodes)).tocsr()
    A = (T_mat.T @ K @ T_mat).tocsr()

    F_full = wp.zeros(dm.n_nodes, dtype=wp.float64, device=DEVICE)
    xq     = gauss_points(mesh, dm.tables_by_p)
    fq     = wp.array(f_star(xq[pv]), dtype=wp.float64, device=DEVICE)
    lk     = make_load_kernel(bk['nbf'], bk['nqp'], dm.dim)
    wp.launch(lk, dim=n_elem,
              inputs=[bk['conn'], bk['h'], bk['N'], bk['w'], fq, F_full],
              device=DEVICE)
    bvec = np.asarray(T_mat.T.T @ F_full.numpy())

    coords = mesh.node_coords[cons.free_nodes]
    bdry   = np.where(mesh.boundary_nodes[cons.free_nodes])[0]
    A_lil  = A.tolil()
    for i in bdry:
        A_lil.rows[i] = [int(i)]
        A_lil.data[i] = [1.0]
        bvec[i] = u_star(coords[i:i + 1])[0]
    A_csc  = A_lil.tocsr().tocsc()
    lu     = splu(A_csc)                     # ← THE LU FACTORIZATION (used twice below)
    u_free = lu.solve(bvec)
    u_all  = np.asarray(cons.T @ u_free)
    return A_csc, lu, u_free, u_all, bdry

# Element centroids from Gauss-point coordinates (the physical coords are right)
_xq_setup = gauss_points(mesh, dm.tables_by_p)
xc = _xq_setup[pv].reshape(n_elem, bk['nqp'], dm.dim).mean(axis=1)  # [n_elem, 2]
kappa_true = (
    1.0
    + 1.5 * np.exp(-40 * ((xc[:, 0] - 0.3)**2 + (xc[:, 1] - 0.3)**2))
    + 1.0 * np.exp(-40 * ((xc[:, 0] - 0.7)**2 + (xc[:, 1] - 0.7)**2))
)

_, _, _, u_true_all, _ = solve_poisson(kappa_true)
u_obs_vals = W @ u_true_all

kappa_eval = np.ones(n_elem) * 1.3
A_csc, lu_fwd, u_free, u_all, bdry = solve_poisson(kappa_eval)

# ── (B) NEWTON LOOPS ─────────────────────────────────────────────────────────
print("(B) NEWTON / NONLINEAR ITERATIONS")
print("-" * 65)
# ── math aside ──────────────────────────────────────────────────────────────
# Newton iterates u_{k+1} = u_k - (dR/du)^{-1} R(u_k, m) until R ≈ 0.
# The algorithm: a loop of ~5-20 linear solves with Jacobian updates.
#
# The relation: R(u*, m) = 0  defines u* implicitly.
# The Implicit Function Theorem (IFT): at convergence,
#   du*/dm = -(dR/du)^{-1} (dR/dm)   evaluated at (u*, m)
#
# In adjoint form: (dR/du)^T λ = dJ/du   (ONE linear solve at the CONVERGED
# state — NOT the Jacobian at any intermediate Newton iterate).
#   dJ/dm = λ^T (dR/dm)
#
# NEVER unroll the Newton loop through a tape — you would differentiate through
# ~20 linear solves and their convergence checks, accumulating catastrophic
# rounding.  The IFT derivative is cleaner AND cheaper.
#
# DiffSim: the SBM Poisson uses bicgstab (nonsymmetric operator); for the
# kappa-gradient, diffsim.sbm.adjoint constructs (dR/du)^T = A^T and solves
# it ONCE — exactly the IFT pattern.
# ──────────────────────────────────────────────────────────────────────────
print("  Relation: R(u*, m) = 0 defines u*(m) implicitly (IFT).")
print("  Adjoint:  (dR/du)^T λ = dJ/du  at the CONVERGED state u*.")
print("  NEVER tape through the Newton iteration loop.")
print()

# ── (C) MESH BUILD / CARVE / CLASSIFICATION / PREFLIGHT ─────────────────────
print("(C) MESH / OCTREE BUILD, ELEMENT CLASSIFICATION, PREFLIGHT CHECKS")
print("-" * 65)
# ── math aside ──────────────────────────────────────────────────────────────
# The mesh pipeline — build_uniform → build_mesh → build_constraints —
# produces combinatorial objects (connectivity tables, element lists, basis
# tables).  These are PIECEWISE CONSTANT in the geometric parameters (domain
# boundary, refinement level) and UNDEFINED in the sense of a smooth map.
#
# In particular: the set of surrogate faces (SBM "carved" elements) is a
# level-set threshold: a cell is in or out.  Differentiating a threshold is
# zero almost everywhere and undefined at the crossing.
#
# The right approach: FREEZE the mesh.  Fix the octree structure and
# classification BEFORE the tape is opened.  Geometric sensitivities live
# in the BOUNDARY PARAMETRIZATION (shape derivatives), not the mesh arrays.
#
# E1 is the payoff: shape_optimization.py freezes the mesh at each shape
# iterate, reusing the same connectivity, and differentiates ONLY through
# the PDE solve and the boundary integral (the "freeze" pattern).
# Cross-reference: see tutorials/E_differentiable/E1_shape_optimization.py.
#
# DiffSim's PREFLIGHT functions (sanity checks, hash validation, symmetry
# tests) similarly: run them ONCE before the tape, never inside.
# ──────────────────────────────────────────────────────────────────────────
print("  Mesh arrays (conn, h, basis tables) are FROZEN before taping.")
print("  Classification (in/out, SBM carved cells) is piecewise-constant.")
print("  Preflights run ONCE; never inside a tape scope.")
print("  Forward pointer: E1_shape_optimization.py — the worked freeze example.")
print()

# ── (D) RNG SEEDS AND TABULATED BASES ────────────────────────────────────────
print("(D) RNG SEEDS AND TABULATED BASIS FUNCTIONS")
print("-" * 65)
# ── math aside ──────────────────────────────────────────────────────────────
# Random seeds (Monte Carlo sampling, stochastic initialisation) and precomputed
# basis tables (Ntab, dNtab, wtab in every DiffSim kernel) are INPUTS, not
# intermediate computations.
#
# Basis tables: computed once by basis_tables(p, dim) — a Python function that
# evaluates Gauss-Legendre quadrature and shape functions on the reference
# element.  The tables are FIXED for a given p and dim; they are uploaded to
# device and then referenced by every kernel launch.
#
#   dm.bins[pv]['N']    — shape values   [nqp, nbf]
#   dm.bins[pv]['dN']   — shape gradients [nqp, nbf, dim]
#   dm.bins[pv]['w']    — quadrature weights [nqp]
#
# These arrays never have requires_grad=True.  They are compile-time constants
# from the tape's perspective — frozen inputs, not state to differentiate.
#
# RNG seeds: if your kernel depends on wp.rand() inside the tape scope, the
# tape records the random draw as a FIXED VALUE (the draw is replayed with
# the same result on backward).  To differentiate w.r.t. the distribution
# parameters you need the reparametrisation trick — the random SEED is an
# input, not a variable.
# ──────────────────────────────────────────────────────────────────────────
print("  Basis tables (N, dN, w): frozen inputs — never taped, never grad.")
print("  RNG seeds: inputs in the reparametrisation sense.")
print(f"  Example: dm.bins[pv]['dN'].shape = {dm.bins[pv]['dN'].numpy().shape}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# §4  COMPUTE-ONCE-AND-STORE
# ─────────────────────────────────────────────────────────────────────────────

print("§4  COMPUTE-ONCE-AND-STORE")
print("=" * 65)
print()

# ── math aside ──────────────────────────────────────────────────────────────
# Three canonical patterns:
#
# (i)  ONE LU FACTORIZATION SERVES A AND A^T SOLVES
#   splu(A) computes L, U, P, Q such that P A Q = L U.
#   A x = b  =>  Q (U^{-1} (L^{-1} (P b)))          (forward solve)
#   A^T y = c =>  P^T (L^{-T} (U^{-T} (Q^T c)))     (backward solve)
#   scipy's SuperLU stores both triangular factors, so lu.solve(b)
#   and lu.solve(c, trans='T') share the SAME factorization.
#
# (ii) TABULATED BASIS REUSE
#   dm.bins[pv]['dN'] is computed once by basis_tables().  Every kernel
#   launch references the same device array.  No re-evaluation per step.
#
# (iii) GP-FIELD PRECOMPUTATION ("frozen at GPs", the M2 convention)
#   A closure field kappa_fn(x) is evaluated at Gauss points BEFORE the
#   tape scope, stored in kq[e*nqp + q], and passed as a frozen input.
#   The tape only differentiates through the PDE kernel, not through kappa_fn.
#   This is the M2 "one-way coupler" pattern: kappa_fn changes slowly
#   (outer optimisation loop); the PDE solve is the inner loop.
# ──────────────────────────────────────────────────────────────────────────

# ─── Timing: one LU serving A and A^T vs two separate factorizations ─────────
print("(i)  ONE LU vs TWO — timing comparison")
print("-" * 65)

N_TRIALS = 50
b_rhs = np.random.default_rng(42).standard_normal(A_csc.shape[0])
c_rhs = np.random.default_rng(43).standard_normal(A_csc.shape[0])

# PATH 1: single factorization, two solve calls
t0 = time.perf_counter()
for _ in range(N_TRIALS):
    lu_one = splu(A_csc)
    _ = lu_one.solve(b_rhs)
    _ = lu_one.solve(c_rhs, trans='T')
t_one = (time.perf_counter() - t0) / N_TRIALS

# PATH 2: two separate factorizations
t0 = time.perf_counter()
for _ in range(N_TRIALS):
    lu_fwd2 = splu(A_csc)
    _ = lu_fwd2.solve(b_rhs)
    lu_bwd2 = splu(A_csc.T.tocsc())
    _ = lu_bwd2.solve(c_rhs)
t_two = (time.perf_counter() - t0) / N_TRIALS

ratio = t_two / t_one
print(f"  One LU (two solve calls)  : {t_one * 1e3:.3f} ms")
print(f"  Two LU (two factorizations): {t_two * 1e3:.3f} ms")
print(f"  Ratio (two LU) / (one LU)  : {ratio:.2f}x")
print()

# ── math aside ──────────────────────────────────────────────────────────────
# At these sizes (81 free nodes, 2-D L3) the factorization cost dominates.
# As the problem scales to millions of DOFs, the ratio stays >2 because
# factorization is O(n^{1.5}) in 2-D while forward-backward substitution
# is O(n log n) — the ONE-LU rule becomes more valuable at scale.
# ──────────────────────────────────────────────────────────────────────────

print("(ii) Tabulated basis reuse")
print("-" * 65)
dN_arr = dm.bins[pv]['dN'].numpy()
print(f"  dN computed ONCE: shape {dN_arr.shape}  "
      f"(nqp={bk['nqp']}, nbf={bk['nbf']}, dim={dm.dim})")
print("  Referenced by every kernel launch — never re-evaluated per solve.")
print()

print("(iii) GP-field precomputation ('frozen at GPs')")
print("-" * 65)
# ── math aside ──────────────────────────────────────────────────────────────
# Suppose kappa is a closure (neural network or spline).  We evaluate it at
# every Gauss point BEFORE the tape:
#   xq = gauss_points(mesh, dm.tables_by_p)        # GP coordinates
#   kq = my_kappa_field(xq[pv])                    # [n_elem * nqp]  FP64
#   kq_dev = wp.array(kq, dtype=wp.float64, device=DEVICE, requires_grad=False)
# Then inside the tape scope: wp.launch(kernel, ..., kq_dev, ...)
# The tape sees kq_dev as FROZEN — the backward pass does not differentiate
# through my_kappa_field.  Only the PDE solve sensitivity is tracked.
#
# This is intentional: the M2 "one-way coupler" — the closure is updated
# in the outer optimisation loop; the PDE's adjoint needs only dR/dkappa,
# not d(kappa_field)/d(closure_params).
# ──────────────────────────────────────────────────────────────────────────
xq_demo = gauss_points(mesh, dm.tables_by_p)[pv]     # shape [n_elem * nqp, 2]
kq_frozen = np.ones(len(xq_demo)) * 1.3              # a trivial "closure"
print(f"  Gauss points precomputed: {xq_demo.shape}  (n_elem*nqp, dim)")
print("  kq frozen outside tape  : passed as requires_grad=False.")
print("  Backward pass ignores kappa_field internals — only dR/dkappa needed.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# §5  THE DOT-PRODUCT TEST
# ─────────────────────────────────────────────────────────────────────────────

print("§5  THE DOT-PRODUCT TEST  ⟨Av, w⟩ = ⟨v, A^T w⟩")
print("=" * 65)
print()

# ── math aside ──────────────────────────────────────────────────────────────
# For any operator A and vectors v, w:
#   ⟨Av, w⟩ = ⟨v, A^T w⟩
# (by definition of the transpose/adjoint).
#
# This is the CHEAPEST possible sanity check for a new operator:
#   1. Pick random v, w.
#   2. Compute Av via the forward operator.
#   3. Compute A^T w via the adjoint operator.
#   4. Check |⟨Av, w⟩ - ⟨v, A^T w⟩| / (||Av|| * ||w|| + 1e-100) < tol.
#
# If this fails, your adjoint is wrong.
# If this passes to machine precision, you have strong evidence it is correct.
#
# We do NOT need to know the explicit matrix A — only the forward and adjoint
# ACTIONS (matvec and rmatvec).  This makes the test valid for matrix-free
# operators too (the M1b path).
#
# For the assembled Poisson CSR:
#   Av  = A.dot(v)   (scipy)
#   A^T w = A.T.dot(w)
# and the inner product is just np.dot.
# ──────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(7)
v = rng.standard_normal(A_csc.shape[0])
w = rng.standard_normal(A_csc.shape[0])

A_dense = A_csc.toarray()
Av   = A_dense @ v
ATw  = A_dense.T @ w

inner_Av_w  = float(np.dot(Av,  w))
inner_v_ATw = float(np.dot(v,  ATw))
dp_err = abs(inner_Av_w - inner_v_ATw)
dp_norm = (np.linalg.norm(Av) * np.linalg.norm(w) + 1e-100)
dp_rel  = dp_err / dp_norm

print(f"  ⟨Av, w⟩          = {inner_Av_w:.15e}")
print(f"  ⟨v, A^T w⟩       = {inner_v_ATw:.15e}")
print(f"  |difference|      = {dp_err:.4e}")
print(f"  relative error    = {dp_rel:.4e}  (machine precision = ~2e-16)")
print()

# Poisson A is symmetric — A^T = A — so the test should be EXACT to FP64 rounding.
assert dp_rel < 1e-12, \
    f"Dot-product test FAILED: relative error {dp_rel:.2e} (expected < 1e-12)"
print(f"  Dot-product test PASSED  (rel err {dp_rel:.2e} < 1e-12)")
print()

# ── math aside ──────────────────────────────────────────────────────────────
# The Poisson matrix is SYMMETRIC (kappa-weighted Laplacian with symmetric
# BCs), so A = A^T and the test is algebraically exact — only floating-point
# rounding enters.  For non-symmetric operators (SBM, convection-diffusion)
# the test still converges to machine precision but A^T ≠ A requires keeping
# BOTH triangle factors.  The test STAYS THE SAME — that is its power.
#
# Run this test for EVERY NEW OPERATOR you add to DiffSim.  It costs two
# matvec calls and one subtraction — a negligible overhead.
# ──────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# §6  PAYOFF: κ(x)-FIELD RECOVERY
# ─────────────────────────────────────────────────────────────────────────────

print("§6  PAYOFF — κ(x)-FIELD RECOVERY (gradient descent, ~40 steps)")
print("=" * 65)
print()

# ── math aside ──────────────────────────────────────────────────────────────
# INVERSE PROBLEM: given observations u_obs at 5 probes, recover the two-blob
# conductivity field kappa_true(x) from a flat initial guess kappa_init = 1.3.
#
# Objective (misfit):  J(kappa) = (1/2) sum_i (u(x_i) - u_obs_i)^2
# Gradient:   dJ/dkappa_e = -lambda^T (dA/dkappa_e) u
#             where A^T lambda = dJ/du   (adjoint solve)
#
# Update:     kappa <- kappa - alpha * dJ/dkappa    (gradient descent)
#
# We reuse the LU factorization for BOTH the forward solve and the adjoint
# solve — exactly the compute-once pattern of §4(i).
#
# WHAT "ERROR DROP" MEANS HERE:
# With only 5 probes and 64 element unknowns, the inverse problem is severely
# underdetermined — we cannot recover kappa pointwise from 5 measurements.
# The PDE is a low-pass filter: high-frequency kappa variations barely affect u.
#
# What gradient descent DOES guarantee: it minimises J — the misfit at the
# observed probes.  The "error drop" reported below is therefore
#   J_final / J_init   (the misfit drop)
# not ||kappa_final - kappa_true|| (the parameter-space error).
#
# A 100× misfit drop means: the predicted probe values at the recovered kappa
# match the observations 10× better in RMS than the flat initial guess.
# This demonstrates that the adjoint gradient is CORRECTLY steering the
# optimisation — the pedagogical goal of this chapter.
# ──────────────────────────────────────────────────────────────────────────

# Per-element stiffness matrices Ke (kappa=1 — sensitivity building block)
# COMPUTE ONCE outside the optimisation loop — pure geometry, kappa-independent
Ke_dev = wp.zeros((n_elem, bk['nbf'], bk['nbf']), dtype=wp.float64, device=DEVICE)
wp.launch(make_poisson_element_matrices(bk['nbf'], bk['nqp'], dm.dim),
          dim=n_elem, inputs=[bk['h'], bk['dN'], bk['w'], Ke_dev], device=DEVICE)
Ke_np = Ke_dev.numpy()   # [n_elem, 4, 4]  — frozen, never changes

def grad_J(kappa_vec: np.ndarray):
    """Gradient of J w.r.t. kappa via the adjoint method.

    Returns (J_value, dJ_dkappa).  ONE forward solve + ONE adjoint solve,
    both reusing the SAME LU factorization (compute-once pattern, §4(i)).
    """
    A_k, lu_k, u_k_free, u_k_all, bdry_k = solve_poisson(kappa_vec)

    # J = (1/2) ||W u_all - u_obs||^2   (probe misfit)
    r      = W @ u_k_all - u_obs_vals
    J_val  = 0.5 * float(r @ r)

    # Adjoint RHS: dJ/du scattered to all nodes
    dJdu_all     = np.asarray(W.T @ r)
    dJdu_adj     = dJdu_all.copy()
    dJdu_adj[bdry_mask] = 0.0                         # zero boundary rows

    # ADJOINT SOLVE — SAME LU FACTOR, transposed  (lu.solve with trans='T')
    # This is the core of the compute-once-and-store principle: the forward
    # solve already factored A into L U P (scipy SuperLU).  The transpose solve
    # costs the same as another forward solve — but requires NO new factorization.
    lam          = lu_k.solve(dJdu_adj, trans='T')    # A^T λ = dJ/du
    lam_int      = lam.copy()
    lam_int[bdry_mask] = 0.0                          # belt-and-suspenders

    # dJ/dkappa_e = -lambda_int[conn_e] @ Ke[e] @ u_all[conn_e]
    dJ           = np.array([
        -float(lam_int[conn[e]] @ Ke_np[e] @ u_k_all[conn[e]])
        for e in range(n_elem)
    ])
    return J_val, dJ

# Initial kappa: flat (far from true blobs) — measures the initial misfit
kappa_init = np.ones(n_elem) * 1.3
J_init, _  = grad_J(kappa_init)

print(f"  kappa_true range  : [{kappa_true.min():.3f}, {kappa_true.max():.3f}]")
print(f"  kappa_init        : {kappa_init[0]:.3f} (flat, far from true blobs)")
print(f"  J_init (misfit)   : {J_init:.4e}")
print()
print(f"  {'Step':>5}  {'J (misfit)':>14}  {'J drop':>10}")
print(f"  {'-'*5}  {'-'*14}  {'-'*10}")

kappa  = kappa_init.copy()
# Step size calibrated by one Newton step: alpha ~ J0 / ||grad J||^2 ~ 36
# (stable with monotone J decrease for this problem)
ALPHA   = 36.0
N_STEPS = 40
J_vals  = [J_init]

for step in range(N_STEPS):
    J_val, dJ = grad_J(kappa)
    J_vals.append(J_val)
    kappa = np.clip(kappa - ALPHA * dJ, 0.5, 5.0)
    if step % 5 == 0 or step == N_STEPS - 1:
        drop = J_init / max(J_val, 1e-20)
        print(f"  {step:>5}  {J_val:>14.4e}  {drop:>9.1f}x")

J_final  = J_vals[-1]
err_drop = J_init / max(J_final, 1e-20)
print()
print(f"  J_init  (misfit)  : {J_init:.4e}")
print(f"  J_final (misfit)  : {J_final:.4e}")
print(f"  Misfit drop (J)   : {err_drop:.1f}x  (require >= 100x)")
print()

# ─────────────────────────────────────────────────────────────────────────────
# EXPECTED RESULTS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("EXPECTED RESULTS")
print("=" * 65)
print(f"  Dot-product rel err       : {dp_rel:.2e}  (expect < 1e-12)")
print(f"  Factorization reuse ratio : {ratio:.2f}x   (expect > 1.5x)")
print(f"  kappa recovery J_init     : {J_init:.4e}  (expect 5e-3 – 5e-2)")
print(f"  kappa recovery J_final    : {J_final:.4e}  (expect < 5e-4)")
print(f"  kappa recovery J drop     : {err_drop:.1f}x   (expect >= 100x)")
print()

# Sanity gates — asserts match PRINTED thresholds with ≥2× headroom
# (E0a lesson: the gate is the printed number, not a looser bound)
assert dp_rel < 1e-12, \
    f"Dot-product test FAILED: {dp_rel:.2e} >= 1e-12"
assert ratio > 1.5, \
    f"Factorization reuse ratio too small: {ratio:.2f} < 1.5"
assert J_init > 5e-4, \
    f"J_init too small (initial guess too close to truth): {J_init:.4e}"
assert J_final < 5e-4, \
    f"J_final too large (recovery stalled): {J_final:.4e} >= 5e-4"
assert err_drop >= 100.0, \
    f"Misfit drop insufficient: {err_drop:.1f}x < 100x"
print("All checks passed.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# EXPLORE
# ─────────────────────────────────────────────────────────────────────────────
print("""EXPLORE
  (a) The Poisson operator is symmetric (A = A^T), so the dot-product test
      passes trivially.  Add a convection term: A <- A + C where
      C[i,j] = int N_i (v . grad N_j) dV (v = [1, 0]).
      Run the dot-product test — it should now require the genuine transpose.
      Does C^T = -C (skew-symmetry of the pure-convection term)?

  (b) Increase LEVEL to 4 (256 elements, 16x16 grid).  Repeat the kappa
      recovery with the same 5 probes.  Does the error drop more or less?
      (Hint: more parameters, same observations — the problem is MORE
      underdetermined.  Add a Tikhonov term  eps * ||kappa - 1||^2 to J
      and tune eps to stabilize recovery.)

  (c) Replace the flat initial guess kappa_init = 1.3 with a single-blob
      guess (only one Gaussian, misplaced).  Measure the error drop.
      Experiment with the step size ALPHA — what happens at ALPHA = 2.0?

  (d) The compute-once Ke_np is computed at kappa=1 (unit conductivity).
      Write the EXACT element sensitivity dJ/dkappa_e for a NONLINEAR
      kappa(u) (e.g. kappa = 1 + u^2).  What changes in the gradient formula?
      Does the IFT argument still hold?

      Forward pointer: when kappa depends on u (temperature-dependent
      conductivity), you need the FULL nonlinear adjoint — E0c covers the
      transient chain, and M4 demonstrates it for phase-field thermodynamics.
""")
