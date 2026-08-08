"""E0a — Thinking differentiable: from residual to gradient.

LEARNING OUTCOME. You understand three distinct paths to the gradient of a
simulation output J(u(m)) with respect to parameters m — finite differences,
the tangent-linear method, and the adjoint — and you can see exactly WHY the
adjoint wins for large parameter counts. You have run a naked ``wp.Tape``
example to see what automatic differentiation records, and you have verified
the adjoint for a Poisson conductivity-field problem at three-way agreement
at the 1e-6/1e-9 class (adjoint-vs-tape at 1e-9; vs FD at 1e-6).

BACKGROUND. Adapted from the mathematical background developed for
dolfin-adjoint/pyadjoint by Patrick E. Farrell; examples and the three-way
comparison device follow Farrell, Ham, Funke & Rognes (2013) and Mitusch,
Funke & Dokken (2019), cited below.

The mother problem: -div(kappa(x) grad u) = f on the unit square with
u = g on the boundary.  The conductivity field kappa(x) is our parameter.
Someone hands us probe measurements u_obs(x_i); we want dJ/dkappa where
    J = (1/2) sum_i (u(x_i) - u_obs_i)^2.

This quantity of interest is the reduced functional
    J_hat(kappa) = J(u(kappa), kappa)
where u(kappa) is implicitly defined by the PDE.  The adjoint gives
dJ_hat/dkappa in O(1) solves regardless of how many parameters kappa has.

Information-flow view (Farrell): forward solves propagate data FROM the
parameter field kappa THROUGH the PDE TO the output functional J.  The
adjoint reverses this flow — it carries sensitivity FROM the scalar output
J BACKWARD through the PDE TO every parameter at once, achieving O(1)
gradient cost independent of parameter count N.

Attribution:
  - P.E. Farrell, mathematical background for dolfin-adjoint/pyadjoint.
  - Farrell, Ham, Funke & Rognes (2013). Automated Derivation of the
    Adjoint of High-Level Mathematical Programs. SIAM J. Sci. Comput.
  - Mitusch, Funke & Dokken (2019). dolfin-adjoint 2018.1: automated
    adjoints for FEniCS and Firedrake. JOSS.
  - DiffSim discrete-adjoint stance: spec S4.3, findings-4c taping rules.
  - "Learning reaction rates" example lineage: dolfin-adjoint tutorial.

EXPECTED RESULTS (CPU, level-3 grid, 5 probes, FP64):
    J at eval kappa (kappa_true two-blob, kappa_eval=1.3)  ~ 8.35e-03
    Adjoint time                                           < 0.05 s
    FD time (64 params, central differences)               < 1.5  s
    adj  vs FD  max rel err                                < 1e-6
    tape vs FD  max rel err                                < 1e-6
    adj  vs tape max rel err                               < 1e-9
    FD cost / adjoint cost                                 ~ 30-80 x
      (n_params=64 elements; first-run tape includes kernel compilation;
       subsequent runs show the steady-state ratio of ~35x)
    kappa_true range                                       [1.0, ~2.5]
      (two-blob field; centroids from anchors()/2^31 + 0.5*h, NOT Morton integers)

Run:  python tutorials/E_differentiable/E0a_thinking_differentiable.py
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
# §0  PROBLEM SETUP
# ─────────────────────────────────────────────────────────────────────────────

# ── math aside ──────────────────────────────────────────────────────────────
# The REDUCED FUNCTIONAL  J_hat(kappa) = J(u(kappa), kappa)
#
# For a scalar-parameter problem J(m) is a 1-D function; optimising it is
# trivial.  For a field parameter kappa : Omega -> R we have O(N) unknowns
# (one per element) and the question is how cheaply we can evaluate the
# gradient dJ/dkappa_e for ALL e simultaneously.
#
# Key insight: u(kappa) satisfies  R(u, kappa) = A(kappa) u - b = 0.
# Differentiating this constraint is where the three methods diverge.
# ──────────────────────────────────────────────────────────────────────────

DEVICE = "cpu"
LEVEL = 3       # 8x8 quadrilateral grid (64 elements) — small enough for FD

# Manufactured solution used for boundary data and load
u_star   = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
f_star   = lambda x: 2.0 * np.pi**2 * u_star(x)  # -lap(u_star) = f_star

# Five interior probe locations arranged in a small ring
PROBES = 0.5 + 0.10 * np.array(
    [[np.cos(t), np.sin(t)]
     for t in np.linspace(0, 2 * np.pi, 5, endpoint=False)])

# ─────────────────────────────────────────────────────────────────────────────
# §1  SECTION 1 — THREE WAYS TO A GRADIENT
# ─────────────────────────────────────────────────────────────────────────────

# ── math aside ──────────────────────────────────────────────────────────────
# Let m = (m_1,...,m_N) be N parameters (e.g. kappa per element).
# We want grad_m J.
#
# METHOD 1 — Finite Differences (FD)
#   dJ/dm_i ≈ [J(m + eps e_i) - J(m - eps e_i)] / (2 eps)
#   Cost: 2N FORWARD SOLVES.  Exact only in the limit eps -> 0 (limited by
#   floating-point cancellation around eps ~ sqrt(machine_eps)).
#
# METHOD 2 — Tangent-Linear (TLM / "forward-mode")
#   Differentiate R(u,m) = 0 w.r.t. m_i:
#     A du_i/dm_i = -dA/dm_i * u        (one forward-mode solve per i)
#     dJ/dm_i = (dJ/du) * du_i/dm_i
#   Cost: N FORWARD SOLVES — same order O(N), but 2× cheaper than central
#   differences (which need 2N).
#
#   Note: even when R(u, κ) = 0 is nonlinear in u, the adjoint equation
#   (dR/du)^T λ = (dJ/du)^T is always LINEAR in λ — one linear solve, always,
#   at the converged state.
#
# METHOD 3 — Adjoint
#   Differentiate the Lagrangian L = J - lambda^T R:
#     dL/du = 0  =>  A^T lambda = (dJ/du)^T   (ONE solve for ALL i)
#     dJ/dm_i = -lambda^T dR/dm_i             (cheap, no extra solves)
#   Cost: 1 FORWARD + 1 ADJOINT SOLVE.  Exact.  Cost INDEPENDENT of N.
#
# The adjoint wins when N >> 1.  For N=64 elements the FD cost is 64x higher.
# ──────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("E0a — Thinking differentiable: from residual to gradient")
print("=" * 65)
print()
print("THREE WAYS TO A GRADIENT")
print("-" * 40)
print(f"{'Method':<20}  {'Solve cost':<20}  {'Exact?'}")
print(f"{'Finite differences':<20}  {'2N forward':<20}  No (eps-limited)")
print(f"{'Tangent-linear':<20}  {'N forward':<20}  Yes")
print(f"{'Adjoint':<20}  {'1 fwd + 1 adj':<20}  Yes")
print()

# ─────────────────────────────────────────────────────────────────────────────
# §2  NAKED wp.Tape TOY  (no FEM — just mechanics)
# ─────────────────────────────────────────────────────────────────────────────

# ── math aside ──────────────────────────────────────────────────────────────
# A Warp Tape records a LAUNCH GRAPH: which kernels ran with which arrays.
# On backward(), it replays that graph in reverse, accumulating cotangents
# (adjoints of array values) into tape.gradients[x].
#
# For J = sum_i kappa_i * u_i^2:
#   dJ/dkappa_i = u_i^2       (the cotangent we expect below)
#
# The tape computes this automatically without you writing the derivative.
# ──────────────────────────────────────────────────────────────────────────

# The kernel: J = sum_i kappa[i] * u[i]^2  — five lines including decorator
@wp.kernel
def toy_energy(kappa: wp.array(dtype=wp.float64),
               u:     wp.array(dtype=wp.float64),
               J:     wp.array(dtype=wp.float64)):
    """Accumulate J += kappa[i] * u[i]^2 — the minimal differentiable kernel."""
    i = wp.tid()
    wp.atomic_add(J, 0, kappa[i] * u[i] * u[i])

wp.init()

kappa_toy = wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float64, device=DEVICE,
                     requires_grad=True)   # differentiable input
u_toy     = wp.array([0.5, 0.3, 0.7, 0.2], dtype=wp.float64, device=DEVICE,
                     requires_grad=False)  # frozen state; NOT differentiated
J_toy     = wp.zeros(1, dtype=wp.float64, device=DEVICE, requires_grad=True)

tape = wp.Tape()
with tape:                          # ← scope records every wp.launch inside
    wp.launch(toy_energy, dim=4, inputs=[kappa_toy, u_toy, J_toy], device=DEVICE)

# Seed dJ/dJ = 1 (scalar output), propagate backward
tape.backward(grads={J_toy: wp.array([1.0], dtype=wp.float64, device=DEVICE)})

grad_kappa_tape = tape.gradients[kappa_toy].numpy()
grad_kappa_exact = u_toy.numpy() ** 2   # dJ/dkappa_i = u_i^2

print("NAKED wp.Tape TOY  (J = sum_i kappa_i * u_i^2)")
print("-" * 40)
print(f"J value          : {J_toy.numpy()[0]:.6f}")
print(f"tape grad kappa  : {np.array2string(grad_kappa_tape, precision=4)}")
print(f"exact dJ/dkappa  : {np.array2string(grad_kappa_exact, precision=4)}")
print(f"max error        : {np.abs(grad_kappa_tape - grad_kappa_exact).max():.2e}  "
      f"(expect 0 — purely algebraic kernel)")
print()

# ── math aside ──────────────────────────────────────────────────────────────
# Note: u_toy was NOT added to tape.gradients because requires_grad=False.
# The tape ONLY accumulates cotangents for arrays with requires_grad=True.
# This is the DiffSim "frozen state" pattern: the solution u is frozen while
# we differentiate through the parameter-dependent part of the residual.
# ──────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# §3  BUILD MESH AND ASSEMBLE POISSON SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

# ── math aside ──────────────────────────────────────────────────────────────
# We discretize  -div(kappa grad u) = f  on [0,1]^2.
# With bilinear Q1 elements (uniform Cartesian grid):
#   A(kappa) u = b
# where  A(kappa) = T^T K(kappa) T,  K(kappa) = sum_e kappa_e Ke.
#
# Ke is the per-element stiffness matrix (kappa=1); the full stiffness
# is just kappa_e * Ke summed and assembled.  T is the constraint matrix
# mapping free nodes to all nodes (absorbing strong Dirichlet BCs).
#
# DISCRETIZE-THEN-DIFFERENTIATE (DiffSim's stance, spec S4.3):
# We differentiate the discrete system, not the continuous adjoint PDE.
# The discrete adjoint is EXACT for the discrete problem — no extra
# spatial discretization error from the adjoint equation.
#
# BOUNDARY ROW NOTE: Strong Dirichlet BCs are enforced by replacing the
# rows of A for boundary nodes with identity (A[i,:] = e_i^T, b[i] = g_i).
# Those rows are INDEPENDENT of kappa_e.  Therefore dA_eff/dkappa_e has
# nonzero entries ONLY in rows corresponding to interior nodes of element e.
# The adjoint seed vector lam must be ZEROED at boundary nodes before the
# gradient inner product.
# ──────────────────────────────────────────────────────────────────────────

tree = build_uniform(LEVEL, dim=2)
mesh = build_mesh(tree, p=1)           # bilinear Q1 on uniform Cartesian grid
cons = build_constraints(mesh)
dm   = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), DEVICE)
T_mat = cons.T.tocsr()                 # constraint prolongation (identity here)

n_elem = len(dm.bins[1]['eids'])       # 64 for level-3 2-D grid
pv = 1; bk = dm.bins[pv]              # single polynomial-degree bin
conn = dm.mesh.conn_of[pv]            # element-to-node connectivity [64, 4]

# Probe interpolation weights W: (n_probes, n_nodes) sparse matrix
W = point_eval_weights(mesh, PROBES)   # reuse existing utility

# Boundary node mask: True for nodes with Dirichlet BCs
bdry_mask = mesh.boundary_nodes        # bool [n_nodes], True for boundary

print(f"MESH: level={LEVEL}, n_elem={n_elem}, n_nodes={dm.n_nodes}, "
      f"n_bdry_nodes={bdry_mask.sum()}")
print()


def solve_poisson(kappa_vec: np.ndarray):
    """Forward Poisson solve with a per-element kappa field.

    Parameters
    ----------
    kappa_vec : (n_elem,) array
        Per-element conductivity.

    Returns
    -------
    A_csc     : assembled constrained stiffness (scipy CSC, for adjoint reuse)
    lu        : SuperLU factorization — reuse for adjoint via lu.solve(rhs, trans='T')
                (one factorization serves both — see E0b §4)
    u_free    : free-node solution (= u_all since T=I for this mesh)
    u_all     : full-node solution
    bdry_dofs : indices of Dirichlet-constrained dofs
    """
    # ── Assemble kappa-weighted stiffness ──────────────────────────────────
    # Expand per-element kappa to all Gauss points in the bin (uniform grid
    # => nqp quadrature points per element).
    kq = np.repeat(kappa_vec, bk['nqp'])                     # [n_elem * nqp]
    rows, cols, vals = volume_triplets(dm, kq_by_bin={pv: kq})
    K = sp.coo_matrix((vals, (rows, cols)),
                      shape=(dm.n_nodes, dm.n_nodes)).tocsr()
    A = (T_mat.T @ K @ T_mat).tocsr()                        # constrained

    # ── Load vector (kappa-independent for this formulation) ──────────────
    F_full = wp.zeros(dm.n_nodes, dtype=wp.float64, device=DEVICE)
    xq = gauss_points(mesh, dm.tables_by_p)
    fq = wp.array(f_star(xq[pv]), dtype=wp.float64, device=DEVICE)
    lk = make_load_kernel(bk['nbf'], bk['nqp'], dm.dim)
    wp.launch(lk, dim=n_elem,
              inputs=[bk['conn'], bk['h'], bk['N'], bk['w'], fq, F_full],
              device=DEVICE)
    bvec = np.asarray(T_mat.T.T @ F_full.numpy())

    # ── Strong Dirichlet BCs: replace boundary rows with identity ──────────
    # A[i, :] <- e_i^T, bvec[i] <- g(x_i) for boundary nodes.
    # These rows are INDEPENDENT of kappa — key for the gradient formula below.
    coords = mesh.node_coords[cons.free_nodes]
    bdry   = np.where(mesh.boundary_nodes[cons.free_nodes])[0]
    A_lil  = A.tolil()
    for i in bdry:
        A_lil.rows[i] = [int(i)]
        A_lil.data[i] = [1.0]
        bvec[i] = u_star(coords[i:i + 1])[0]

    A_csc  = A_lil.tocsr().tocsc()                           # for splu
    lu     = splu(A_csc)                                     # ← THE LU FACTORIZATION (used twice)
    u_free = lu.solve(bvec)                                  # forward solve
    u_all  = np.asarray(cons.T @ u_free)                     # scatter to all nodes
    return A_csc, lu, u_free, u_all, bdry

# ─────────────────────────────────────────────────────────────────────────────
# §4  POISSON dJ/dkappa — THREE-WAY CHECK
# ─────────────────────────────────────────────────────────────────────────────

# ── Setup: fixed observation data from a "true" kappa field ───────────────
# Two-blob kappa field as the ground truth (range ~ 1.0 to 2.5).
# Element centroids from Gauss-point coordinates (same approach as E0b):
#   anchors() returns Morton integers in [0, 2^31); dividing by 2^31 maps
#   them to [0,1).  Adding half the cell width gives the centroid in [0,1]^2.
# NOTE: the old formula  anchors() * 2^{-LEVEL}  produced Morton-integer
# garbage (coordinates ~1e9), making kappa_true essentially flat.  Fixed here.
xc    = dm.mesh.tree.anchors() / 2**31 + 0.5 * dm.mesh.tree.h()[:, None]
print(f"  kappa_true centroid range check: x in [{xc[:,0].min():.3f},{xc[:,0].max():.3f}]"
      f", y in [{xc[:,1].min():.3f},{xc[:,1].max():.3f}]  (expect [0,1])")
kappa_true = (1.0
              + 1.5 * np.exp(-40 * ((xc[:, 0] - 0.3)**2 + (xc[:, 1] - 0.3)**2))
              + 1.0 * np.exp(-40 * ((xc[:, 0] - 0.7)**2 + (xc[:, 1] - 0.7)**2)))
print(f"  kappa_true range: [{kappa_true.min():.3f}, {kappa_true.max():.3f}]  (expect [1.0, ~2.5])")

_, _, _, u_true_all, _ = solve_poisson(kappa_true)
u_obs_vals = W @ u_true_all           # "measurements" at five probes

# Evaluation point: uniform kappa away from the truth
kappa_eval = np.ones(n_elem) * 1.3   # where we compute the gradient

# Warm the constant-kappa Ke kernel used ONLY by PATH A's sensitivity assembly
# below (solve_poisson above only ever touches the *_var kappa-weighted
# variant). Without this, PATH A's timed block pays a one-time JIT compile for
# a kernel variant that's never been launched yet, inflating "adjoint time" by
# ~180x and corrupting the FD/adjoint cost-ratio measurement below (P1's own
# launch-floor/JIT-tax lesson, applied to this file).
_Ke_warm = wp.zeros((n_elem, bk['nbf'], bk['nbf']), dtype=wp.float64, device=DEVICE)
wp.launch(make_poisson_element_matrices(bk['nbf'], bk['nqp'], dm.dim),
          dim=n_elem, inputs=[bk['h'], bk['dN'], bk['w'], _Ke_warm], device=DEVICE)

def J_func(kv: np.ndarray) -> float:
    """Scalar loss: (1/2) sum_i (u(x_i) - u_obs_i)^2."""
    _, _, _, ua, _ = solve_poisson(kv)
    r = W @ ua - u_obs_vals
    return 0.5 * float(r @ r)


# ══════════════════════════════════════════════════════════════════════════════
# PATH A: ADJOINT
# ══════════════════════════════════════════════════════════════════════════════

# ── math aside ──────────────────────────────────────────────────────────────
# Adjoint equation:  A^T lambda = (dJ/du)^T
#
# dJ/du_all = W^T (W u_all - u_obs)   (gradient of probe least-squares)
#
# GRADIENT FORMULA:
#   A(kappa) = sum_e kappa_e Ae  where Ae is the element-level contribution
#   to the assembled system (including BCs via row replacement).
#
#   The boundary rows of A_eff are IDENTITY and do NOT depend on kappa_e.
#   Therefore dA_eff/dkappa_e is nonzero ONLY in the rows corresponding to
#   INTERIOR nodes of element e.
#
#   dJ/dkappa_e = -lambda^T (dA_eff/dkappa_e) u
#               = -sum_{a: interior} lam[a] * sum_b Ke[e,a,b] * u[b]
#
#   Equivalently: zero out lam at boundary nodes, then
#   dJ/dkappa_e = -(lam_int)[conn_e] . Ke[e] . u[conn_e]
#
# KEY INSIGHT: this is the TRANSPOSED SOLVE you already know.  The forward
# solve was A u = b.  The adjoint solve is A^T lam = dJ/du.  For symmetric A
# (which Poisson gives), A = A^T, so you can REUSE the same LU factorization.
# In DiffSim, diffsim.sbm.adjoint.solve_adjoint does exactly this.
# ──────────────────────────────────────────────────────────────────────────

t_a0 = time.time()
A_csc, lu_fwd, u_free, u_all, bdry = solve_poisson(kappa_eval)

# Gradient of J w.r.t. solution at probe points, scattered to all nodes
r_probe   = W @ u_all - u_obs_vals                           # residual at probes
dJdu_all  = np.asarray(W.T @ r_probe)                       # [n_nodes]

# Adjoint solve (reuse the SAME factorization as the forward solve via A.T)
# STEP 1 — seed zeroing: boundary rows of A_eff are identity (Dirichlet
# elimination), so (dR/du)^T at boundary dofs is already identity; zeroing
# dJdu_adj[bdry] forces the adjoint solve to return lam[bdry]=0 directly
# from those rows — no extra work needed.
dJdu_adj  = dJdu_all.copy()
dJdu_adj[bdry_mask] = 0.0                                    # zero boundary rows
# Reuse the forward LU factor via trans='T' — one factorization serves both.
# (one factorization serves both — see E0b §4)
lam       = lu_fwd.solve(dJdu_adj, trans='T')               # A^T lam = dJ/du

# STEP 2 — belt-and-suspenders: lam[bdry] should already be 0 from STEP 1,
# but we zero explicitly as a safeguard before the gradient contraction below.
lam_int   = lam.copy()
lam_int[bdry_mask] = 0.0

# Per-element stiffness matrices Ke (kappa=1 — the sensitivity building block)
Ke_dev = wp.zeros((n_elem, bk['nbf'], bk['nbf']), dtype=wp.float64, device=DEVICE)
wp.launch(make_poisson_element_matrices(bk['nbf'], bk['nqp'], dm.dim),
          dim=n_elem, inputs=[bk['h'], bk['dN'], bk['w'], Ke_dev], device=DEVICE)
Ke_np = Ke_dev.numpy()                                       # [n_elem, 4, 4]

# dJ/dkappa_e = -lam_int[conn_e] @ Ke[e] @ u_all[conn_e]
grad_adj = np.array([
    -float(lam_int[conn[e]] @ Ke_np[e] @ u_all[conn[e]])
    for e in range(n_elem)
])
t_adj = time.time() - t_a0

# ══════════════════════════════════════════════════════════════════════════════
# PATH B: wp.Tape THROUGH THE RESIDUAL
# ══════════════════════════════════════════════════════════════════════════════

# ── math aside ──────────────────────────────────────────────────────────────
# The tape path differentiates the RELATION  R(kappa, u_frozen) = A(kappa) u
# with u frozen (EXACTLY the principle of §2: we differentiate the RELATION,
# not the algorithm for computing u).
#
# Kernel: for each element e, accumulate kappa[e] * (grad Na) . (grad u)
# into R[conn[e,a]].  With kappa as a differentiable warp array:
#   dR[a]/dkappa[e] = (grad Na at e) . (grad u at e)
#
# The tape backward propagates cotangents from R to kappa via:
#   dJ/dkappa_e = (lam_int)^T (dR/dkappa_e)
#
# CRITICAL: we seed the tape with lam_int (lam zeroed at boundary nodes),
# not raw lam — same reason as in the adjoint formula above.
#
# Notice: we do NOT differentiate through splu — we differentiate the
# RESIDUAL that splu implicitly inverts.  The derivative of the implicit
# function u(kappa) exists even though splu has no known gradient.
# This "relation, not algorithm" principle is named and flagged for E0b.
# ──────────────────────────────────────────────────────────────────────────

def _make_kappa_residual_kernel(nbf: int, nqp: int, dim: int):
    """Poisson volume residual R += kappa[e] * (grad Na) . (grad u), taped in kappa.

    u is a frozen parameter (requires_grad=False).
    kappa is a differentiable input (requires_grad=True).
    The backward pass gives dR[a]/dkappa[e] for all a, e simultaneously.
    """
    key = ("kappa_res_e0a", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel(module="unique", enable_backward=True)   # ← backward codegen ON
    def kappa_residual(conn:   wp.array2d(dtype=wp.int32),
                       h:      wp.array(dtype=wp.float64),
                       dNtab:  wp.array3d(dtype=wp.float64),
                       wtab:   wp.array(dtype=wp.float64),
                       kappa:  wp.array(dtype=wp.float64),  # DIFFERENTIABLE
                       u:      wp.array(dtype=wp.float64),  # frozen
                       r:      wp.array(dtype=wp.float64)):
        e = wp.tid()
        fe  = FEMElm(); fe.e = e; fe.he = h[e]
        half    = fe.he * wp.float64(0.5)
        jac     = wp.float64(1.0)
        for _d in range(dim):
            jac = jac * half                              # (h/2)^dim
        dscale  = wp.float64(2.0) / fe.he                # ref-to-phys gradient scale

        for q in range(nqp):
            fe.q = q
            dJxW = fe_detJxW_s(wtab, fe, jac)

            for a in range(nbf):
                # (grad Na) . (grad u_h): sum over spatial dimensions
                gradNa_dot_gradu = wp.float64(0.0)
                for d in range(dim):
                    dNa_d  = fe_dN_s(dNtab, fe, a, d, dscale)
                    gradu_d = wp.float64(0.0)
                    for bv in range(nbf):
                        gradu_d = (gradu_d
                                   + fe_dN_s(dNtab, fe, bv, d, dscale)
                                   * u[conn[e, bv]])
                    gradNa_dot_gradu = gradNa_dot_gradu + dNa_d * gradu_d

                # kappa[e] multiplies the entire element's contribution
                wp.atomic_add(r, conn[e, a],
                              kappa[e] * gradNa_dot_gradu * dJxW)

    _kernel_cache[key] = kappa_residual
    return kappa_residual

kern_kappa_res = _make_kappa_residual_kernel(bk['nbf'], bk['nqp'], dm.dim)

# Warp arrays: kappa is differentiable; u is frozen
kappa_wp = wp.array(kappa_eval.astype(np.float64), dtype=wp.float64,
                    device=DEVICE, requires_grad=True)
u_all_wp = wp.array(u_all.astype(np.float64), dtype=wp.float64,
                    device=DEVICE, requires_grad=False)    # FROZEN
R_wp     = wp.zeros(dm.n_nodes, dtype=wp.float64, device=DEVICE,
                    requires_grad=True)

t_b0 = time.time()
tape_res = wp.Tape()
with tape_res:
    wp.launch(kern_kappa_res, dim=n_elem,
              inputs=[bk['conn'], bk['h'], bk['dN'], bk['w'],
                      kappa_wp, u_all_wp, R_wp],
              device=DEVICE)

# Seed with lam_int: lam zeroed at boundary nodes.
# This zeros out the contribution of boundary rows (which are kappa-independent
# in A_eff) from the gradient accumulation — same principle as in PATH A.
lam_seed = wp.array(lam_int.astype(np.float64), dtype=wp.float64, device=DEVICE)
tape_res.backward(grads={R_wp: lam_seed})
grad_tape = tape_res.gradients[kappa_wp].numpy()
t_tape = time.time() - t_b0

# ── math aside ──────────────────────────────────────────────────────────────
# Sign convention check:
# The tape computed  grad_tape[e] = lam_int^T (dR/dkappa_e).
# Our adjoint formula: dJ/dkappa_e = -lam^T (dA_eff/dkappa_e) u.
# Since R = A(kappa) u - b, we have dR/dkappa_e = dA/dkappa_e u.
# So grad_tape[e] = lam_int^T (dA/dkappa_e u) = -grad_adj[e].
# Therefore grad_adj should equal -grad_tape (one extra minus sign because
# the adjoint formula sign comes from J_hat chain rule direction).
#
# We negate grad_tape below to match the adjoint sign convention.
# ──────────────────────────────────────────────────────────────────────────
grad_tape_signed = -grad_tape   # the adjoint sign: dJ/dk = -lam^T dR/dk

# ══════════════════════════════════════════════════════════════════════════════
# PATH C: CENTRAL FINITE DIFFERENCES
# ══════════════════════════════════════════════════════════════════════════════

eps = 1e-5
t_c0 = time.time()
J0 = J_func(kappa_eval)
grad_fd = np.zeros(n_elem)
for e in range(n_elem):
    kp = kappa_eval.copy(); kp[e] += eps
    km = kappa_eval.copy(); km[e] -= eps
    grad_fd[e] = (J_func(kp) - J_func(km)) / (2.0 * eps)
t_fd = time.time() - t_c0

# ═════════════════════════════════════════════════════════════════════════════
# THREE-WAY COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

# ── math aside ──────────────────────────────────────────────────────────────
# When the three paths agree to ~1e-9 relative error, you have strong
# evidence that all three are correct.  If any two disagree, the third
# acts as the tiebreaker.
#
# The "house verification contract" (also used in E1, M2, SP-1): before
# trusting any gradient in an optimization loop, run this check once on
# a small problem.  It costs 2N extra solves — worth it once, not every iter.
# ──────────────────────────────────────────────────────────────────────────

# Relative errors (vs FD as reference)
denom        = np.abs(grad_fd) + np.abs(grad_fd).max() * 1e-10
rel_adj_fd   = np.abs(grad_adj          - grad_fd) / denom
rel_tape_fd  = np.abs(grad_tape_signed  - grad_fd) / denom
rel_adj_tape = np.abs(grad_adj - grad_tape_signed) / (np.abs(grad_adj) + 1e-30)

print("POISSON dJ/dkappa  (level-3, 64 elements, 5 probes)")
print("-" * 65)
print(f"J at eval kappa          : {J0:.4e}")
print(f"||grad_adj||_inf         : {np.abs(grad_adj).max():.4e}")
print(f"||grad_tape||_inf        : {np.abs(grad_tape_signed).max():.4e}")
print(f"||grad_fd||_inf          : {np.abs(grad_fd).max():.4e}")
print()
print(f"THREE-WAY CHECK (relative errors):")
print(f"  adj   vs FD   max rel err : {rel_adj_fd.max():.2e}")
print(f"  tape  vs FD   max rel err : {rel_tape_fd.max():.2e}")
print(f"  adj   vs tape max rel err : {rel_adj_tape.max():.2e}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# §5  COST TABLE: FD vs ADJOINT
# ─────────────────────────────────────────────────────────────────────────────

# ── math aside ──────────────────────────────────────────────────────────────
# The cost ratio is empirical here, but the argument is clean:
#   FD:      2 * N_params forward solves (2 * 64 = 128 here)
#   Adjoint: 1 forward solve + 1 adjoint solve ≈ 2 solves total
#   Ratio:   N_params = 64
#
# At N_params = 10^4 (a realistic field kappa): FD would need 20,000 solves;
# adjoint needs 2.  For large problems the adjoint is not just faster — it
# is the ONLY feasible method.
# ──────────────────────────────────────────────────────────────────────────

print("COST TABLE (measured wall time on CPU)")
print("-" * 65)
print(f"{'Method':<22}  {'Time (s)':>10}  {'# solves':>10}  {'Ratio vs adj':>14}")
print(f"{'Adjoint':<22}  {t_adj:>10.3f}  {'2 (fwd+adj)':>10}  {'1.0x':>14}")
print(f"{'Tape backward':<22}  {t_tape:>10.3f}  {'same run':>10}  {'—':>14}")
print(f"{'Central FD':<22}  {t_fd:>10.3f}  {2*n_elem:>10}  {t_fd/t_adj:>13.1f}x")
print(f"\nFD / adjoint time ratio : {t_fd/t_adj:.1f}x")
print(f"n_params                : {n_elem}  (FD cost ≈ n_params × adj cost)")
print()

# ─────────────────────────────────────────────────────────────────────────────
# §6  THE DEEP PRINCIPLE  (seed for E0b)
# ─────────────────────────────────────────────────────────────────────────────

# ── math aside ──────────────────────────────────────────────────────────────
# SEED OF E0b: we did NOT differentiate through splu.
#
# The splu call factorizes A into L * U.  Differentiating through the full
# LU-factorization algorithm step by step would be (a) extremely expensive
# and (b) unnecessary.  Instead, we used the IMPLICIT FUNCTION THEOREM:
#
#   R(u, kappa) = A(kappa) u - b = 0  =>  dR/du * du/dkappa = -dR/dkappa
#   =>  A * du/dkappa = -dA/dkappa * u
#   =>  du/dkappa = A^{-1} (-dA/dkappa u)   (one solve per kappa direction)
#
# In adjoint form: A^T lam = dJ/du (one solve total), then
#   dJ/dkappa = -lam^T dA/dkappa u (cheap inner product, no extra solves).
#
# This is the "relation, not algorithm" principle:  we differentiate
# the RELATION Au = b, not the ALGORITHM that solves it.
# E0b walks through why this matters for every solver type: Krylov methods,
# Newton iterations, tabulated basis functions, and random seeds.
# ──────────────────────────────────────────────────────────────────────────

print("DEEP PRINCIPLE (seed for E0b)")
print("-" * 65)
print("We did NOT differentiate through splu.")
print("We differentiated the RELATION  A(kappa) u = b,")
print("obtaining the adjoint equation  A^T lambda = dJ/du.")
print("The algorithm that solves Au=b (splu) is opaque — its derivative")
print("is NOT needed. Only the matrix A and its transpose enter.")
print("See E0b for the full 'do-not-differentiate list'.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# EXPECTED RESULTS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("EXPECTED RESULTS")
print("=" * 65)
print(f"  J at eval kappa        : {J0:.3e}  (expect ~8.35e-03)")
print(f"  adj  vs FD  max rel err: {rel_adj_fd.max():.2e}  (expect < 1e-6)")
print(f"  tape vs FD  max rel err: {rel_tape_fd.max():.2e}  (expect < 1e-6)")
print(f"  adj  vs tape rel err   : {rel_adj_tape.max():.2e}  (expect < 1e-9)")
print(f"  adjoint time (s)       : {t_adj:.3f}  (expect < 0.05)")
print(f"  FD time (s)            : {t_fd:.3f}  (expect < 1.5)")
print(f"  FD / adj cost ratio    : {t_fd/t_adj:.1f}x  (expect 30-80x)")
print()

# Sanity gates (fail loudly; CI smoke will catch regressions)
assert np.abs(grad_kappa_tape - grad_kappa_exact).max() < 1e-14, \
    "toy tape check failed"
assert rel_adj_fd.max()   < 1e-6, \
    f"adj vs FD rel err too large: {rel_adj_fd.max():.2e}"
assert rel_tape_fd.max()  < 1e-6, \
    f"tape vs FD rel err too large: {rel_tape_fd.max():.2e}"
assert rel_adj_tape.max() < 1e-9, \
    f"adj vs tape rel err too large: {rel_adj_tape.max():.2e}"
print("All checks passed.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# EXPLORE
# ─────────────────────────────────────────────────────────────────────────────
print("""EXPLORE
  (a) Change LEVEL to 4 (16x16 = 256 elements). The FD cost grows with
      n_params while the adjoint cost stays constant. Measure the new ratio.
      What would happen at level 6 (64x64 = 4096 elements)?

  (b) Replace the probe J with a volume-integrated J:  J = int u dV.
      dJ/du_all is then the mass-weighted sum of all node values.  You can
      compute this with diffsim.sbm.adjoint.volume_qoi.  Does the three-way
      check still hold?

  (c) The Poisson operator here is SYMMETRIC (A = A^T).  Find a problem
      where A is not symmetric (e.g. add convection: -div(kappa grad u) +
      v . grad u = f) and explain why the adjoint solve STILL needs A^T,
      not A.  What does this mean for the LU factorization reuse argument?

  (d) Replace the toy_energy kernel with  J = sum_i exp(kappa[i] * u[i]).
      Derive the analytic dJ/dkappa_i by hand, then verify that the tape
      still recovers it exactly.  Try making kappa[i] negative — what
      happens numerically, and why?
""")
