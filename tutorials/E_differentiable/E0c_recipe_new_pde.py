r"""E0c — The recipe: differentiating any new PDE.

LEARNING OUTCOME.  You have a checklist you can apply to ANY new PDE to make
it differentiable: what to tape, what to freeze, what to store across steps,
how to build the transposed-solve infrastructure, how to verify (dot-product
-> three-way -> transient-chain), which nondifferentiability hazards to relax,
and the cost contract to hold yourself to (backward <= 2.5x forward).  You have
run a taped transient-heat chain and swept its adjoint backward through time;
you have implemented BOTH full-store and recompute-from-checkpoint for the same
chain and watched them agree to machine zero; you have seen four ways a badly
chosen J sabotages an otherwise-correct gradient; and you have differentiated a
mini phase-field (Allen-Cahn) problem to three-way agreement of 1e-6 or better.

BACKGROUND.  Adapted from the mathematical background developed for
dolfin-adjoint/pyadjoint by Patrick E. Farrell; the "learning reaction rates"
example (our phase-field payoff, below) follows the dolfin-adjoint tutorial of
that name.  Cite Farrell, Ham, Funke & Rognes (2013) and Mitusch, Funke &
Dokken (2019).  DiffSim-native sources: the findings-4c taping rules, the
M1a/M2/M4 adjoint stack, the H4 signal-scale protocol, and the beyond-FH gauge
story (docs/dev/2026-07-14-beyond-fh-learned-thermo.md).

E0a taught the three ways to a gradient and the "differentiate the relation,
not the algorithm" seed.  E0b opened a real brick and gave the do-not-
differentiate list, the compute-once patterns, and the dot-product test.  This
chapter is the RECIPE that ties them together and extends them to TIME: a
transient march has to store (or recompute) its primal trajectory, and the
adjoint sweeps backward through every stored step.  Everything you need to make
a new PDE differentiable is on the checklist in section 1 -- whose normative
twin lives on the reference page:

    NORMATIVE TWIN: docs/theory/adjoint_readiness_checklist.md (T4).
    That page owns the citable list; this chapter derives and demonstrates it.

Attribution:
  - P.E. Farrell, mathematical background for dolfin-adjoint/pyadjoint.
  - Farrell, Ham, Funke & Rognes (2013). Automated Derivation of the
    Adjoint of High-Level Mathematical Programs. SIAM J. Sci. Comput.
  - Mitusch, Funke & Dokken (2019). dolfin-adjoint 2018.1: automated
    adjoints for FEniCS and Firedrake. JOSS.
  - "Learning reaction rates" example lineage: dolfin-adjoint tutorial.
  - DiffSim discrete-adjoint stance: spec S4.3, findings-4c taping rules.
  - Beyond-FH gauge story: docs/dev/2026-07-14-beyond-fh-learned-thermo.md.

EXPECTED RESULTS (CPU, level-3 grid, FP64):
    Heat chain: adjoint dJ/dkappa vs central FD, max rel err   < 1e-5
    Store vs recompute-from-checkpoint, dJ agreement           < 1e-10
      (both compute the SAME discrete adjoint => machine zero, ~0.0)
    Backward-march / forward-march wall ratio                  < 2.5x
      (measured; contract is backward <= 2.5x forward)
    Primal storage inventory printed (arrays x steps x bytes)
    J-design (i)  logsumexp gradient-jump / max gradient-jump   < 0.5
      (the smooth relaxation has a markedly smaller max jump)
    J-design (ii) signal / FD-noise-floor ratio                > 1e3
      (H4 protocol: |dJ| clears the finite-difference floor)
    J-design (iii) informative / flat-J gradient-magnitude      > 10x
    Allen-Cahn mini phase-field, three-way (adj / cstep / FD)   < 1e-6
      dJ/dM and dJ/dkappa both checked

Run:  python tutorials/E_differentiable/E0c_recipe_new_pde.py
"""
# ── imports ──────────────────────────────────────────────────────────────────
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, spsolve
import warp as wp

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import (
    DeviceMesh, make_poisson_element_matrices,
)
from diffsim.physics.poisson import gauss_points
from diffsim.mesh.pointeval import point_eval_weights

DEVICE = "cpu"
LEVEL  = 3      # 8x8 quad grid — 64 elements, 81 nodes: small enough for FD

wp.init()

print("=" * 70)
print("E0c — The recipe: differentiating any new PDE")
print("=" * 70)
print()

# ═════════════════════════════════════════════════════════════════════════════
# §1  THE CHECKLIST  (the chapter's spine)
# ═════════════════════════════════════════════════════════════════════════════

# ── math aside ──────────────────────────────────────────────────────────────
# THE ADJOINT-READINESS CHECKLIST.  Apply this to ANY new PDE before you write
# a line of gradient code.  Each item is developed in E0a/E0b or below; the
# normative, citable statement lives on the reference page (T4):
#     docs/theory/adjoint_readiness_checklist.md
#
#  1. WHAT TO TAPE.  Only the residual kernels that depend on the parameters
#     you differentiate w.r.t.  Tape R(u, m) in m (and in u if J or a later
#     step needs du).  Everything algebraic-and-parameter-dependent is taped;
#     the solver is NOT (item 4).
#
#  2. WHAT TO FREEZE.  Mesh/octree/classification/preflight (piecewise-constant
#     in parameters — E0b §3C), basis & quadrature tables (inputs — E0b §3D),
#     RNG seeds (reparametrise — E0b §3D), and any GP-field precomputed outside
#     the tape (the M2 "frozen at GPs" convention — E0b §4iii).
#
#  3. WHAT TO STORE ACROSS STEPS.  A transient adjoint sweeps backward and needs
#     the PRIMAL state u^n at each step (the gradient contraction below uses it).
#     Store the whole trajectory, OR store checkpoints and recompute segments
#     (§3).  This is the compute-vs-memory trade-off.
#
#  4. TRANSPOSED-SOLVE INFRASTRUCTURE.  The gradient path through any solver is
#     the TRANSPOSED solve at the converged state (E0a, E0b §3A/§3B): one LU
#     serves A and A^T (E0b §4i).  Build A^T (or the rmatvec) once; reuse it.
#
#  5. THE VERIFICATION LADDER.  In increasing strength:
#       (a) dot-product test  <Av,w> = <v,A^T w>  per operator  (E0b §5),
#       (b) THREE-WAY check per gradient: adjoint vs a tape/complex-step vs FD
#           (E0a; §4 here uses complex-step as the exact third leg),
#       (c) TRANSIENT-CHAIN check: the full backward sweep vs FD end-to-end (§2).
#     Climb the ladder before trusting a gradient in an optimiser.
#
#  6. NONDIFFERENTIABILITY HAZARDS -> RELAXED FORMS.  abs/min/max/thresholds/
#     masks are nonsmooth: their gradient is jumpy or zero-a.e.  Replace with
#     relaxed forms (softabs, logsumexp, sigmoid gates) BEFORE optimising (§4i).
#
#  7. THE COST CONTRACT.  A correct reverse-mode adjoint costs O(1) forward
#     solves: the backward march should be <= ~2.5x the forward march (one
#     transposed solve + one gradient contraction per step).  MEASURE it (§3);
#     if backward >> forward you are unrolling something you should not be.
# ──────────────────────────────────────────────────────────────────────────

print("§1  THE ADJOINT-READINESS CHECKLIST")
print("-" * 70)
print("  1. Tape       : residual kernels vs their parameters (only).")
print("  2. Freeze     : mesh/classification/preflight, basis tables, RNG, GP fields.")
print("  3. Store      : primal u^n per step (full-store) OR checkpoints (recompute).")
print("  4. Transpose  : one LU serves A and A^T; build the adjoint solve once.")
print("  5. Verify     : dot-product -> three-way -> transient-chain (climb the ladder).")
print("  6. Relax      : abs/min/max/threshold/mask -> softabs/logsumexp/sigmoid.")
print("  7. Cost       : backward march <= ~2.5x forward march (MEASURE it).")
print("  Normative twin: docs/theory/adjoint_readiness_checklist.md")
print()

# ═════════════════════════════════════════════════════════════════════════════
# §2  TRANSIENT HEAT CHAIN  (taped BDF1 march + backward sweep)
# ═════════════════════════════════════════════════════════════════════════════

# ── math aside ──────────────────────────────────────────────────────────────
# THE PDE.  Transient heat with a conductivity FIELD kappa(x):
#     u_t = div(kappa(x) grad u)  on [0,1]^2,  u = 0 on the boundary.
#
# BDF1 (implicit Euler) in time, Q1 in space.  With the mass matrix M and the
# kappa-weighted stiffness K(kappa) = sum_e kappa_e Ke:
#     (M/dt + K(kappa)) u^n = (M/dt) u^{n-1}          n = 1..N
# i.e. a SEQUENCE of linear solves, each reusing the SAME factorization
# (kappa is constant in time, so M/dt + K is assembled and factored ONCE —
# item 4 on the checklist, applied across the whole march).
#
# THE FUNCTIONAL.  Probe misfit at the FINAL time:
#     J(kappa) = (1/2) ||W u^N - u_obs||^2
# so dJ/du only seeds the LAST step; the adjoint carries it backward.
#
# THE DISCRETE ADJOINT (the chain).  Write step n as
#     F^n := A u^n - P u^{n-1} = 0,   A = M/dt + K,   P = M/dt.
# The adjoint recurrence (Lagrangian with one multiplier lam^n per step):
#     A^T lam^n = seed^n,     seed^N = dJ/du^N = W^T (W u^N - u_obs),
#     grad += -lam^n^T (dA/dkappa) u^n         (contract with STORED primal u^n),
#     seed^{n-1} = P^T lam^n                    (feed the previous step).
# One TRANSPOSED solve per step (reusing the one factorization), one sparse
# matvec to propagate the seed, one element contraction for the gradient.
# ──────────────────────────────────────────────────────────────────────────

print("§2  TRANSIENT HEAT CHAIN  (BDF1 march, backward sweep, dJ/dkappa)")
print("=" * 70)
print()

tree = build_uniform(LEVEL, dim=2)
mesh = build_mesh(tree, p=1)
cons = build_constraints(mesh)
dm   = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), DEVICE)
pv   = 1; bk = dm.bins[pv]
conn = dm.mesh.conn_of[pv]                       # [64, 4]
n_elem  = len(bk['eids'])
n_nodes = dm.n_nodes
bdry = np.where(mesh.boundary_nodes)[0]          # Dirichlet node indices

# ── FROZEN building blocks (checklist item 2) — computed ONCE, outside any tape ──
# Element stiffness Ke (kappa=1) — pure geometry, reused every step & the adjoint.
Ke_dev = wp.zeros((n_elem, bk['nbf'], bk['nbf']), dtype=wp.float64, device=DEVICE)
wp.launch(make_poisson_element_matrices(bk['nbf'], bk['nqp'], dm.dim),
          dim=n_elem, inputs=[bk['h'], bk['dN'], bk['w'], Ke_dev], device=DEVICE)
Ke_np = Ke_dev.numpy()                           # [64, 4, 4]

# Element mass Me = int Na Nb dV — also frozen geometry.
Ntab = bk['N'].numpy(); wtab = bk['w'].numpy(); h_arr = bk['h'].numpy()
jac  = (h_arr * 0.5) ** dm.dim
Me_np = np.zeros((n_elem, bk['nbf'], bk['nbf']))
for e in range(n_elem):
    for q in range(bk['nqp']):
        Me_np[e] += np.outer(Ntab[q], Ntab[q]) * wtab[q] * jac[e]

def _assemble(elem_mats, kappa=None):
    """Assemble a global CSC from per-element blocks (optionally kappa-scaled)."""
    rows = []; cols = []; vals = []
    for e in range(n_elem):
        ce = conn[e]; s = 1.0 if kappa is None else kappa[e]
        for a in range(bk['nbf']):
            for b in range(bk['nbf']):
                rows.append(ce[a]); cols.append(ce[b])
                vals.append(s * elem_mats[e, a, b])
    return sp.coo_matrix((vals, (rows, cols)),
                         shape=(n_nodes, n_nodes)).tocsc()

M_glob = _assemble(Me_np)                         # frozen mass matrix
DT     = 0.01
NSTEPS = 30

# Initial condition: a localized bump; homogeneous Dirichlet on the boundary.
xn = mesh.node_coords
u0 = np.exp(-30 * ((xn[:, 0] - 0.5) ** 2 + (xn[:, 1] - 0.5) ** 2))
u0[bdry] = 0.0

# Four interior probes.
PROBES = 0.5 + 0.15 * np.array(
    [[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 4, endpoint=False)])
W = point_eval_weights(mesh, PROBES)

def _apply_bc(A):
    """Replace Dirichlet rows of A with identity (row-elimination BC)."""
    A = A.tolil()
    for i in bdry:
        A.rows[i] = [int(i)]; A.data[i] = [1.0]
    return A.tocsc()

def heat_forward(kappa, store=True):
    """BDF1 march.  Returns (u^N, trajectory-or-None, factorization).

    The factorization is built ONCE (kappa constant in time) and reused for
    every step — and later for the adjoint's transposed solves (checklist 4).
    """
    K   = _assemble(Ke_np, kappa)
    Abc = _apply_bc((M_glob / DT + K).copy())
    lu  = splu(Abc)                                  # ONE factorization
    u   = u0.copy()
    U   = [u.copy()] if store else None
    for _n in range(NSTEPS):
        rhs = (M_glob / DT) @ u
        rhs[bdry] = 0.0                              # homogeneous Dirichlet RHS
        u = lu.solve(rhs)
        if store:
            U.append(u.copy())
    return u, U, lu

# Observations from the "true" (flat) kappa; evaluate the gradient at a bumpy kappa.
kappa_true = np.ones(n_elem) * 1.0
uT, _, _   = heat_forward(kappa_true)
u_obs      = W @ uT
xc = gauss_points(mesh, dm.tables_by_p)[pv].reshape(
    n_elem, bk['nqp'], dm.dim).mean(axis=1)          # element centroids
kappa0 = 1.0 + 3.0 * np.exp(-25 * ((xc[:, 0] - 0.35) ** 2 + (xc[:, 1] - 0.35) ** 2))

def heat_J(kappa):
    uf, _, _ = heat_forward(kappa, store=False)
    r = W @ uf - u_obs
    return 0.5 * float(r @ r)

P_csc = (M_glob / DT).tocsc()

def _contract(lam, u_n):
    """-sum_e lam[conn_e] Ke[e] u_n[conn_e], vectorized over elements.

    This is  -lam^T (dK/dkappa_e) u_n  for every element e at once
    (dA/dkappa_e = Ke since M does not depend on kappa).
    """
    le = lam[conn]; ue = u_n[conn]
    return -np.einsum('ea,eab,eb->e', le, Ke_np, ue)

def heat_grad_store(U, lu):
    """Full-store backward sweep: dJ/dkappa from the stored trajectory U."""
    r    = W @ U[-1] - u_obs
    seed = np.asarray(W.T @ r)                        # dJ/du^N
    grad = np.zeros(n_elem)
    for n in range(NSTEPS, 0, -1):
        s = seed.copy(); s[bdry] = 0.0               # zero Dirichlet rows
        lam = lu.solve(s, trans='T')                 # A^T lam = seed (SAME LU)
        lam[bdry] = 0.0
        grad += _contract(lam, U[n])                 # uses STORED primal u^n
        seed = np.asarray(P_csc.T @ lam)             # seed^{n-1} = P^T lam^n
    return grad

# ── run the chain + verify vs central FD (transient-chain check, ladder rung c) ──
u_final, U, lu = heat_forward(kappa0)
grad_store = heat_grad_store(U, lu)

eps = 1e-6
grad_fd = np.zeros(n_elem)
for e in range(n_elem):
    kp = kappa0.copy(); kp[e] += eps
    km = kappa0.copy(); km[e] -= eps
    grad_fd[e] = (heat_J(kp) - heat_J(km)) / (2 * eps)
denom       = np.abs(grad_fd) + np.abs(grad_fd).max() * 1e-10
rel_adj_fd  = (np.abs(grad_store - grad_fd) / denom).max()

J0 = heat_J(kappa0)
print(f"  steps={NSTEPS}, dt={DT}, n_elem={n_elem}, n_nodes={n_nodes}")
print(f"  J(kappa0) misfit           : {J0:.4e}")
print(f"  ||dJ/dkappa||_inf (adjoint): {np.abs(grad_store).max():.4e}")
print(f"  transient-chain check      : adjoint vs central FD max rel err = "
      f"{rel_adj_fd:.2e}")
print()

# ── PRIMAL-STORAGE INVENTORY (checklist item 3) ──────────────────────────────
# ── math aside ──────────────────────────────────────────────────────────────
# The backward sweep needs u^n for EVERY n (the _contract call).  Full-store
# keeps the whole trajectory.  The memory is (steps+1) x n_nodes x 8 bytes for
# FP64, plus the factorization (shared).  This is the quantity that outgrows
# the GPU at scale — hence the recompute trade-off in §3.
# ──────────────────────────────────────────────────────────────────────────
bytes_per_state = n_nodes * 8
n_stored        = NSTEPS + 1                         # u^0 .. u^N
total_bytes     = n_stored * bytes_per_state
print("  PRIMAL-STORAGE INVENTORY (full-store):")
print(f"    array          : u^n   (nodal state)")
print(f"    dtype          : FP64  ({8} bytes/dof)")
print(f"    dofs/state     : {n_nodes}")
print(f"    states stored  : {n_stored}   (u^0 .. u^{NSTEPS})")
print(f"    total          : {n_stored} x {n_nodes} x 8 = {total_bytes} bytes "
      f"({total_bytes/1024:.1f} KiB)")
print(f"    scaling note   : at 1e6 dofs x 1e3 steps this is 8 GB — the tape "
      f"outgrows the GPU (see §3).")
print()

# ═════════════════════════════════════════════════════════════════════════════
# §3  COMPUTE-VS-MEMORY  (full-store vs recompute-from-checkpoint)
# ═════════════════════════════════════════════════════════════════════════════

# ── math aside ──────────────────────────────────────────────────────────────
# The backward sweep needs each u^n.  Two ways to supply it:
#
#   FULL-STORE   : keep every u^n.  Memory O(steps x dofs); backward is pure
#                  solves.  (§2 above.)
#   RECOMPUTE    : keep only a few CHECKPOINTS (here u^0 and every seg-th step).
#                  When the sweep needs u^n, RE-MARCH forward from the nearest
#                  earlier checkpoint.  Memory O(n_checkpoints x dofs); backward
#                  pays extra FORWARD solves.
#
# The extreme of recompute (one checkpoint = the IC) trades O(steps^2) recompute
# solves for O(1) memory.  The sweet spot is logarithmic: REVOLVE (Griewank &
# Walther) schedules O(log steps) checkpoints for O(steps log steps) recompute,
# provably optimal.  We NAME revolve as the production scaling path — we do NOT
# build it here (spec §7 YAGNI); binomial-checkpointing libraries exist
# (pyrevolve, checkpoint_schedules) and wire into exactly this loop.
#
# THE INVARIANT WE VERIFY: both modes compute the SAME discrete adjoint.  They
# must agree to MACHINE ZERO on dJ/dkappa (not merely "close") — the recompute
# reproduces the identical u^n bit-for-bit (same LU, same solves).
# ──────────────────────────────────────────────────────────────────────────

print("§3  COMPUTE-VS-MEMORY  (full-store vs recompute-from-checkpoint)")
print("=" * 70)
print()

N_CHECK = 6                                          # checkpoints for recompute

def heat_grad_recompute(kappa, lu, ncheck=N_CHECK):
    """Backward sweep storing only ncheck checkpoints; recompute u^n as needed."""
    seg   = NSTEPS // ncheck
    ckpts = {0: u0.copy()}
    u = u0.copy()
    for n in range(1, NSTEPS + 1):                   # forward, keep checkpoints only
        rhs = (M_glob / DT) @ u; rhs[bdry] = 0.0
        u = lu.solve(rhs)
        if n % seg == 0 or n == NSTEPS:
            ckpts[n] = u.copy()
    r    = W @ u - u_obs
    seed = np.asarray(W.T @ r)
    grad = np.zeros(n_elem)

    def recompute_upto(n):
        start = max(k for k in ckpts if k <= n)      # nearest earlier checkpoint
        uu = ckpts[start].copy()
        for _m in range(start + 1, n + 1):           # re-march the segment
            rhs = (M_glob / DT) @ uu; rhs[bdry] = 0.0
            uu = lu.solve(rhs)
        return uu

    for n in range(NSTEPS, 0, -1):
        s = seed.copy(); s[bdry] = 0.0
        lam = lu.solve(s, trans='T'); lam[bdry] = 0.0
        u_n = ckpts[n] if n in ckpts else recompute_upto(n)
        grad += _contract(lam, u_n)
        seed = np.asarray(P_csc.T @ lam)
    return grad

grad_recompute = heat_grad_recompute(kappa0, lu)
mode_agreement = np.abs(grad_store - grad_recompute).max()

# Memory footprints
store_states    = NSTEPS + 1
recomp_states   = len({0, NSTEPS} | {seg for seg in
                      range(NSTEPS // N_CHECK, NSTEPS + 1, NSTEPS // N_CHECK)})
store_bytes     = store_states  * bytes_per_state
recomp_bytes    = recomp_states * bytes_per_state

# Wall timing: forward march vs backward march (post-factorization; the LU is
# shared, so the fair comparison is the MARCH cost either way).
NREP = 20
K_c   = _assemble(Ke_np, kappa0)
lu_c  = splu(_apply_bc((M_glob / DT + K_c).copy()))

def _fwd_march():
    u = u0.copy(); traj = [u.copy()]
    for _n in range(NSTEPS):
        rhs = (M_glob / DT) @ u; rhs[bdry] = 0.0
        u = lu_c.solve(rhs); traj.append(u.copy())
    return traj

traj_ref = _fwd_march()
t0 = time.perf_counter()
for _ in range(NREP): _fwd_march()
t_fwd = (time.perf_counter() - t0) / NREP
t0 = time.perf_counter()
for _ in range(NREP): heat_grad_store(traj_ref, lu_c)
t_bwd = (time.perf_counter() - t0) / NREP
t0 = time.perf_counter()
for _ in range(NREP): heat_grad_recompute(kappa0, lu_c)
t_bwd_recomp = (time.perf_counter() - t0) / NREP
bwd_fwd_ratio = t_bwd / t_fwd

print("  MODE COMPARISON:")
print(f"    {'mode':<26}{'states kept':>12}{'memory':>12}{'march wall':>13}")
print(f"    {'full-store':<26}{store_states:>12}{store_bytes/1024:>10.1f}K"
      f"{t_bwd*1e3:>11.3f}ms")
print(f"    {'recompute (%d ckpts)' % recomp_states:<26}{recomp_states:>12}"
      f"{recomp_bytes/1024:>10.1f}K{t_bwd_recomp*1e3:>11.3f}ms")
print()
print(f"  dJ AGREEMENT (store vs recompute)  : {mode_agreement:.2e}  "
      f"(same discrete adjoint => ~0.0)")
print(f"  memory saved by recompute          : "
      f"{store_bytes/recomp_bytes:.1f}x fewer stored states")
print()
print("  THE 2.5x COST CONTRACT (checklist item 7):")
print(f"    forward march  wall : {t_fwd*1e3:.3f} ms")
print(f"    backward march wall : {t_bwd*1e3:.3f} ms  (one A^T solve + "
      f"contraction / step)")
print(f"    backward / forward  : {bwd_fwd_ratio:.2f}x   (contract: <= 2.5x)")
print(f"    scaling path        : revolve / binomial checkpointing (NAMED, "
      f"not built — spec §7).")
print()

# ═════════════════════════════════════════════════════════════════════════════
# §4  CHOOSING YOUR J  (runnable mini-demos)
# ═════════════════════════════════════════════════════════════════════════════

print("§4  CHOOSING YOUR J")
print("=" * 70)
print()

# ── (i) NONSMOOTH J vs SMOOTH RELAXATION (checklist item 6) ──────────────────
# ── math aside ──────────────────────────────────────────────────────────────
# A functional built on max/min/abs/threshold is NONSMOOTH: its gradient jumps
# where the active branch switches, and is zero-a.e. for a hard threshold.  An
# optimiser fed a jumpy gradient stalls or oscillates.
#
# The fix is a RELAXED replacement.  For J = max_i g_i(p), the smooth surrogate
# is the LOG-SUM-EXP (softmax) upper bound:
#     lse_beta(g) = (1/beta) log sum_i exp(beta g_i)  ->  max_i g_i  as beta->inf.
# Its gradient is a smooth softmax-weighted blend of the branch gradients — no
# jump at the switch.  We sweep a scalar p and compare the discrete jumps in the
# gradient of the hard max vs the logsumexp surrogate.
# ──────────────────────────────────────────────────────────────────────────
def _field(p):
    return np.array([np.sin(p), np.cos(1.7 * p), np.sin(0.3 * p + 1.0),
                     -np.cos(2.1 * p)])
def J_max(p):
    return _field(p).max()
def J_lse(p, beta=40.0):
    v = _field(p); m = v.max()
    return m + np.log(np.exp(beta * (v - m)).sum()) / beta

ps        = np.linspace(0, 3, 400)
g_max     = np.gradient([J_max(p) for p in ps], ps)
g_lse     = np.gradient([J_lse(p) for p in ps], ps)
jump_max  = np.abs(np.diff(g_max)).max()
jump_lse  = np.abs(np.diff(g_lse)).max()
jump_ratio = jump_lse / jump_max

print("  (i) NONSMOOTH max-J vs SMOOTH logsumexp relaxation")
print(f"      max-J      gradient max jump : {jump_max:.3f}  (branch switches)")
print(f"      logsumexp  gradient max jump : {jump_lse:.3f}  (smooth blend)")
print(f"      ratio (lse / max)            : {jump_ratio:.3f}  (smaller = "
      f"smoother; require < 0.5)")
print()

# ── (ii) SIGNAL-SCALE BEFORE OPTIMIZING (the H4 protocol) ────────────────────
# ── math aside ──────────────────────────────────────────────────────────────
# THE H4 PROTOCOL (DiffSim-native): before you launch an optimiser, check that
# |dJ/dp| CLEARS THE FINITE-DIFFERENCE NOISE FLOOR.  A gradient buried under the
# FD floor means J barely responds to p — the optimiser will chase rounding
# noise.  We estimate the floor from the SPREAD of central FD across step sizes
# (a well-scaled problem has a wide plateau; the min spread bounds the floor).
# LESSON: run this ONCE, up front.  If |dJ| does not clear the floor by orders
# of magnitude, rescale p or J (or pick a more informative observable) BEFORE
# spending a single optimiser iteration.
# ──────────────────────────────────────────────────────────────────────────
# Reuse the heat problem: signal = the largest element sensitivity we computed.
signal = np.abs(grad_store).max()
fd_vals = []
for e_step in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
    e0 = int(np.argmax(np.abs(grad_store)))
    kp = kappa0.copy(); kp[e0] += e_step
    km = kappa0.copy(); km[e0] -= e_step
    fd_vals.append((heat_J(kp) - heat_J(km)) / (2 * e_step))
fd_floor    = np.abs(np.diff(fd_vals)).min() + 1e-300   # narrowest FD spread
signal_ratio = signal / fd_floor

print("  (ii) SIGNAL-SCALE CHECK (H4 protocol) — run BEFORE optimizing")
print(f"       |dJ/dkappa| signal          : {signal:.3e}")
print(f"       FD noise floor (min spread) : {fd_floor:.3e}")
print(f"       signal / floor              : {signal_ratio:.2e}  (require > 1e3)")
print()

# ── (iii) FLAT-J GEOMETRY (probe placement) ──────────────────────────────────
# ── math aside ──────────────────────────────────────────────────────────────
# Where you MEASURE decides whether J even sees your parameter.  In the heat
# problem the initial bump decays and diffuses; a probe placed in a far corner
# where the field never has amplitude sees du/dkappa ~ 0 — a FLAT-J / vanishing-
# gradient geometry.  A probe ON the active region sees a large sensitivity.
# We place a sharp bump at (0.25,0.25), read a near probe vs a far-corner probe,
# and compare ||dJ/dkappa||.  Same physics, same adjoint — only the observable
# moved, and the gradient magnitude changes by orders of magnitude.
# ──────────────────────────────────────────────────────────────────────────
u0_sharp = np.exp(-80 * ((xn[:, 0] - 0.25) ** 2 + (xn[:, 1] - 0.25) ** 2))
u0_sharp[bdry] = 0.0
DT_f, NS_f = 0.005, 15
kappa0_f = 1.0 + 2.0 * np.exp(-25 * ((xc[:, 0] - 0.25) ** 2 + (xc[:, 1] - 0.25) ** 2))

def _flatJ_grad_mag(probe_xy):
    Wp   = point_eval_weights(mesh, np.array([probe_xy]))
    K    = _assemble(Ke_np, np.ones(n_elem))
    lu_f = splu(_apply_bc((M_glob / DT_f + K).copy()))
    # observations at flat kappa
    u = u0_sharp.copy(); traj = [u.copy()]
    for _n in range(NS_f):
        rhs = (M_glob / DT_f) @ u; rhs[bdry] = 0.0
        u = lu_f.solve(rhs); traj.append(u.copy())
    obs = Wp @ u
    # adjoint at bumpy kappa
    K2   = _assemble(Ke_np, kappa0_f)
    lu2  = splu(_apply_bc((M_glob / DT_f + K2).copy()))
    u = u0_sharp.copy(); traj = [u.copy()]
    for _n in range(NS_f):
        rhs = (M_glob / DT_f) @ u; rhs[bdry] = 0.0
        u = lu2.solve(rhs); traj.append(u.copy())
    seed = np.asarray(Wp.T @ (Wp @ u - obs)); grad = np.zeros(n_elem)
    for n in range(NS_f, 0, -1):
        s = seed.copy(); s[bdry] = 0.0
        lam = lu2.solve(s, trans='T'); lam[bdry] = 0.0
        grad += _contract(lam, traj[n])
        seed = np.asarray(((M_glob / DT_f).tocsc().T) @ lam)
    return np.abs(grad).max()

g_near = _flatJ_grad_mag([0.28, 0.28])               # ON the active bump
g_far  = _flatJ_grad_mag([0.84, 0.84])               # far corner: field ~ 0
flatJ_contrast = g_near / max(g_far, 1e-300)

print("  (iii) FLAT-J GEOMETRY (probe placement)")
print(f"        informative probe (on bump) ||dJ/dkappa|| : {g_near:.3e}")
print(f"        flat-J probe (far corner)    ||dJ/dkappa|| : {g_far:.3e}")
print(f"        contrast (informative / flat)              : {flatJ_contrast:.1f}x"
      f"  (require > 10x)")
print()

# ── (iv) IDENTIFIABILITY & CONDITIONING — the beyond-FH gauge story ──────────
# ── math aside ──────────────────────────────────────────────────────────────
# A gradient can be CORRECT, LARGE, and SMOOTH and the inverse problem STILL
# ill-posed — because the parameterisation has UNIDENTIFIABLE directions.  The
# diagnostic is the conditioning of the observation Jacobian, not the gradient.
#
# CASE STUDY (beyond-FH learned thermodynamics, M4 rung 2; numbers QUOTED from
# docs/dev/2026-07-14-beyond-fh-learned-thermo.md — not rerun here):
#
#   Learning a free-energy correction f'(c) on top of Flory-Huggins has two
#   structurally invisible ("gauge") modes: a CONSTANT in f' (shifts mu by a
#   constant -> invisible to conserved c-dynamics) and a LINEAR mode
#   ((1-2c) is proportional to P1), which is EXACTLY ALIASED with the FH chi
#   term (the trajectory difference along it was measured at 5e-15).
#
#   The observation-Jacobian conditioning made this quantitative:
#       un-anchored {chi, P1, P2, P3} :  cond = 1.4e16  (chi & P1 aliased -> singular)
#       anchored    {chi, P2, P3}     :  cond = 1.8e1   (well posed)
#   a 7.6e14x conditioning gap.  ANCHORING the basis orthogonal to {1, c}
#   (Legendre P_k, k>=2) removes the gauge modes BY CONSTRUCTION; on the
#   anchored parameterisation the coefficients recover from a single interior
#   trajectory to rel-err 5.6e-8.
#
#   Even gauge-anchored, a SINGLE narrow trajectory only weakly identifies the
#   high modes (visited composition 0.46-0.56 -> cond ~2e2-5e2); a COMPOSITION-
#   DIVERSE protocol (0.21-0.85) drops cond to ~14-18 (a ~15-30x improvement),
#   and end-to-end the diverse ensemble RECOVERS {P2..P5} to coeff-err ~4e-5
#   while the single narrow fit FAILS (~0.9-1.0) — a >2e4x recovery gap.
#
# TIKHONOV AS SCIENCE, not hack: a penalty eps*||theta||^2 is a PRIOR encoding
# that the correction is small/smooth.  In the beyond-FH fit it does not merely
# "stabilise"; it PULLS the ill-conditioned single fit to a definite (wrong)
# answer rather than letting it wander to NaN — making the failure legible and
# the well-posed (diverse) fit trustworthy.  Regularisation states a hypothesis;
# the conditioning number tells you whether the data can test it.
# ──────────────────────────────────────────────────────────────────────────
print("  (iv) IDENTIFIABILITY & CONDITIONING (beyond-FH gauge story, QUOTED)")
print("       un-anchored {chi,P1,P2,P3} obs-Jacobian cond : 1.4e16 (aliased)")
print("       anchored    {chi,P2,P3}    obs-Jacobian cond : 1.8e1  (well posed)")
print("       => 7.6e14x gap; anchoring removes the gauge mode by construction.")
print("       composition-diverse protocol: cond 2e2-5e2 -> 14-18 (~15-30x).")
print("       Tikhonov = a PRIOR (science), not a numerical hack.")
print("       Source: docs/dev/2026-07-14-beyond-fh-learned-thermo.md (not rerun).")
print()

# ── (v) MULTI-OBSERVABLE J — the bridge to SP-1 ──────────────────────────────
# ── math aside ──────────────────────────────────────────────────────────────
# A single scalar probe misfit is often flat or aliased (items iii, iv).  The
# cure is a RICHER, INSTRUMENT-SPACE observable — and often SEVERAL of them:
#   J = sum_k w_k ||O_k(u) - d_k||^2
# where O_k are diverse operators (probe values, a structure factor S(q), a film
# height h(t), a PSF-convolved image).  Diverse protocols illuminate different
# directions of parameter space, turning an underdetermined inverse (E0b's 5
# probes for 64 unknowns) into a well-posed one.  The beyond-FH result above is
# exactly this: a differentiable S(q) misfit backprops to the closure
# coefficients (matching FD to 1.8e-11).  SP-1 (R3) builds the multi-observable,
# multi-protocol inversion on the production stack; this is the bridge to it.
# ──────────────────────────────────────────────────────────────────────────
print("  (v) MULTI-OBSERVABLE J: J = sum_k w_k ||O_k(u) - d_k||^2.")
print("      Diverse instrument-space observables (probes, S(q), h(t), PSF)")
print("      make underdetermined inverses well-posed.  Bridge -> SP-1 (R3).")
print()

# ═════════════════════════════════════════════════════════════════════════════
# §5  MINI PHASE-FIELD  (Allen-Cahn: dJ/dM, dJ/dkappa, three-way checked)
# ═════════════════════════════════════════════════════════════════════════════

# ── math aside ──────────────────────────────────────────────────────────────
# THE PDE (Allen-Cahn, the group's own physics — physics/allen_cahn.py):
#     c_t = -M ( f'(c) - kappa lap c ),   f(c) = (1/4)(c^2-1)^2,  f'(c)=c^3-c.
# NON-conserved gradient flow of F[c] = int [ f(c) + (kappa/2)|grad c|^2 ].
#
# A DIFFERENTIABLE MARCH.  We use a semi-implicit BDF1 split: the interface
# (stiffness) term implicit, the reaction f'(c) explicit — linear in c^{n+1},
# hence a single solve per step and a clean adjoint.  (The production
# AllenCahnStepper is fully implicit Newton with enable_backward=False; per the
# do-not-differentiate list, we would differentiate its CONVERGED residual via
# the IFT, not unroll the Newton loop.  Here the semi-implicit split keeps the
# demo self-contained and every step exactly differentiable.)  With lumped mass
# m = diag(M):
#     (m/dt + M_mob kappa K) c^n = (m/dt) c^{n-1} - M_mob m f'(c^{n-1}).
#
# PARAMETERS.  We differentiate J = (1/2)||W c^N - c_obs||^2 w.r.t. the mobility
# M_mob and the gradient-energy coefficient kappa (both scalars here).
#
# THREE-WAY CHECK (ladder rung b) with an EXACT third leg — COMPLEX-STEP:
#   J(p + i*hh) with hh ~ 1e-30 gives dJ/dp = Im J / hh with NO subtraction error
#   (the complex step is independent of both the hand-adjoint and central FD).
# ATTRIBUTION: this "learning a rate/coefficient of a reaction-diffusion PDE"
# is the dolfin-adjoint "learning reaction rates" example, adapted to Warp.
# ──────────────────────────────────────────────────────────────────────────

print("§5  MINI PHASE-FIELD  (Allen-Cahn: dJ/dM, dJ/dkappa, three-way)")
print("=" * 70)
print()

# Consistent element mass & stiffness for AC (lumped mass, same Ke as heat).
Me_ac = Me_np                                        # reuse
Ke_ac = np.zeros((n_elem, bk['nbf'], bk['nbf']))
dNtab = bk['dN'].numpy(); dscale = 2.0 / h_arr
for e in range(n_elem):
    for q in range(bk['nqp']):
        for a in range(bk['nbf']):
            for b in range(bk['nbf']):
                g = sum(dNtab[q, a, d] * dscale[e] * dNtab[q, b, d] * dscale[e]
                        for d in range(dm.dim))
                Ke_ac[e, a, b] += g * wtab[q] * jac[e]
M_ac = _assemble(Me_ac); K_ac = _assemble(Ke_ac)
m_lump = np.asarray(M_ac.sum(axis=1)).ravel()        # lumped mass diagonal

DT_ac, NS_ac = 0.005, 25
c_init = 0.2 * np.cos(2 * np.pi * xn[:, 0]) * np.cos(2 * np.pi * xn[:, 1])
PROBES_ac = 0.5 + 0.2 * np.array(
    [[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 4, endpoint=False)])
W_ac = point_eval_weights(mesh, PROBES_ac)

def ac_forward(M_mob, kap, store=True):
    A  = (sp.diags(m_lump / DT_ac) + M_mob * kap * K_ac).tocsc()
    lu = splu(A)
    c  = c_init.copy(); C = [c.copy()] if store else None
    for _n in range(NS_ac):
        fp  = c ** 3 - c
        rhs = (m_lump / DT_ac) * c - M_mob * m_lump * fp
        c   = lu.solve(rhs)
        if store:
            C.append(c.copy())
    return c, C, lu

def ac_J(M_mob, kap):
    c, _, _ = ac_forward(M_mob, kap, store=False)
    r = W_ac @ c - c_obs_ac
    return 0.5 * float(r @ r)

def ac_J_cstep(M_mob, kap):
    """Complex-arithmetic forward + holomorphic J (no conjugate) for complex-step."""
    A = (sp.diags((m_lump / DT_ac).astype(complex))
         + M_mob * kap * K_ac.astype(complex)).tocsc()
    c = c_init.astype(complex)
    for _n in range(NS_ac):
        fp  = c ** 3 - c
        rhs = (m_lump / DT_ac) * c - M_mob * m_lump * fp
        c   = spsolve(A, rhs)
    r = W_ac @ c - c_obs_ac
    return 0.5 * np.sum(r ** 2)                        # holomorphic: r^2, not |r|^2

# Truth -> observations; evaluate the gradient away from truth.
M_true, kap_true = 1.0, 0.01
cT_ac, _, _ = ac_forward(M_true, kap_true)
c_obs_ac    = W_ac @ cT_ac
M_eval, kap_eval = 1.3, 0.02

def ac_grad_adjoint(M_mob, kap):
    """Hand-adjoint through the semi-implicit AC chain -> (dJ/dM, dJ/dkappa)."""
    cN, C, lu = ac_forward(M_mob, kap)
    seed = np.asarray(W_ac.T @ (W_ac @ cN - c_obs_ac))   # dJ/dc^N
    gM = 0.0; gk = 0.0
    for n in range(NS_ac, 0, -1):
        lam    = lu.solve(seed, trans='T')               # A^T lam = seed
        cprev  = C[n - 1]
        # F^n = A c^n - (m/dt) cprev + M_mob m f'(cprev) = 0
        dF_dM   = kap * (K_ac @ C[n]) + m_lump * (cprev ** 3 - cprev)
        dF_dk   = M_mob * (K_ac @ C[n])
        gM     += -float(lam @ dF_dM)
        gk     += -float(lam @ dF_dk)
        # dF^n/dcprev = -(m/dt) + M_mob m f''(cprev),  f''=3c^2-1
        dF_dcprev = -(m_lump / DT_ac) + M_mob * m_lump * (3 * cprev ** 2 - 1)
        seed = -(dF_dcprev * lam)                         # seed^{n-1}
    return gM, gk

gM_adj, gk_adj = ac_grad_adjoint(M_eval, kap_eval)

# Leg 2: complex-step (exact).
hh = 1e-30
gM_cs = np.imag(ac_J_cstep(M_eval + 1j * hh, kap_eval)) / hh
gk_cs = np.imag(ac_J_cstep(M_eval, kap_eval + 1j * hh)) / hh

# Leg 3: central FD.
epsc = 1e-6
gM_fd = (ac_J(M_eval + epsc, kap_eval) - ac_J(M_eval - epsc, kap_eval)) / (2 * epsc)
gk_fd = (ac_J(M_eval, kap_eval + epsc) - ac_J(M_eval, kap_eval - epsc)) / (2 * epsc)

rel_M = max(abs(gM_adj - gM_cs) / abs(gM_cs), abs(gM_adj - gM_fd) / abs(gM_fd))
rel_k = max(abs(gk_adj - gk_cs) / abs(gk_cs), abs(gk_adj - gk_fd) / abs(gk_fd))
ac_three_way = max(rel_M, rel_k)

print(f"  steps={NS_ac}, dt={DT_ac}, semi-implicit BDF1 (reaction explicit)")
print(f"  J(M_eval, kappa_eval)      : {ac_J(M_eval, kap_eval):.4e}")
print(f"  dJ/dM      adjoint={gM_adj:.6e}  cstep={gM_cs:.6e}  FD={gM_fd:.6e}")
print(f"  dJ/dkappa  adjoint={gk_adj:.6e}  cstep={gk_cs:.6e}  FD={gk_fd:.6e}")
print(f"  three-way max rel err (dJ/dM)     : {rel_M:.2e}")
print(f"  three-way max rel err (dJ/dkappa) : {rel_k:.2e}")
print(f"  AC three-way (worst)              : {ac_three_way:.2e}  (require < 1e-6)")
print()
print("  ATTRIBUTION: 'learning reaction rates' — dolfin-adjoint tutorial.")
print("  FORWARD POINTERS: F-track (phase-field course), M4 (learned")
print("  thermodynamics), SP-1 R3 (multi-observable production inversion).")
print()

# ═════════════════════════════════════════════════════════════════════════════
# EXPECTED RESULTS SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("EXPECTED RESULTS")
print("=" * 70)
print(f"  Heat chain adj vs FD          : {rel_adj_fd:.2e}   (expect < 1e-5)")
print(f"  Store vs recompute dJ         : {mode_agreement:.2e}   (expect < 1e-10)")
print(f"  Backward/forward wall ratio   : {bwd_fwd_ratio:.2f}x    (expect < 2.5x)")
print(f"  logsumexp/max grad-jump ratio : {jump_ratio:.3f}   (expect < 0.5)")
print(f"  signal / FD-floor ratio       : {signal_ratio:.2e}   (expect > 1e3)")
print(f"  flat-J contrast               : {flatJ_contrast:.1f}x   (expect > 10x)")
print(f"  AC three-way (adj/cstep/FD)   : {ac_three_way:.2e}   (expect < 1e-6)")
print()

# ─── Sanity gates: asserts match the PRINTED thresholds exactly (E0a/E0b lesson) ──
assert rel_adj_fd < 1e-5, \
    f"heat transient-chain adj vs FD too large: {rel_adj_fd:.2e} >= 1e-5"
assert mode_agreement < 1e-10, \
    f"store vs recompute disagree: {mode_agreement:.2e} >= 1e-10"
assert bwd_fwd_ratio < 2.5, \
    f"cost contract violated: backward/forward = {bwd_fwd_ratio:.2f}x >= 2.5x"
assert jump_ratio < 0.5, \
    f"logsumexp not smoother than max: jump ratio {jump_ratio:.3f} >= 0.5"
assert signal_ratio > 1e3, \
    f"signal below FD floor: signal/floor {signal_ratio:.2e} <= 1e3"
assert flatJ_contrast > 10.0, \
    f"flat-J contrast too weak: {flatJ_contrast:.1f}x <= 10x"
assert ac_three_way < 1e-6, \
    f"AC three-way check failed: {ac_three_way:.2e} >= 1e-6"
print("All checks passed.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# EXPLORE
# ─────────────────────────────────────────────────────────────────────────────
print(r"""EXPLORE
  (a) Push the heat chain to LEVEL 4 (256 elements) and 60 steps.  Recompute the
      primal-storage inventory and the backward/forward ratio.  At what
      (steps x dofs) does full-store become uncomfortable?  Add a SECOND
      checkpoint budget and watch the recompute wall grow as you keep fewer.

  (b) Replace the final-time J with a TIME-INTEGRATED J = (dt/2) sum_n
      ||W u^n - u_obs^n||^2.  Now dJ/du seeds EVERY step, not just the last.
      Re-derive the seed injection in heat_grad_store and re-run the FD check.

  (c) In §4(i) sweep beta in the logsumexp from 5 to 200.  Plot (mentally) the
      trade-off: small beta is smooth but biased (lse >> max); large beta
      approaches max but the gradient jump returns.  Where is the knee?

  (d) The AC demo uses a semi-implicit split.  Make it FULLY implicit (Newton)
      and differentiate the CONVERGED residual via the IFT (do NOT unroll
      Newton — E0b §3B).  Confirm the three-way check still holds.  This is the
      exact pattern M4 uses for learned phase-field thermodynamics.

  (e) Turn §5 into a real fit: Adam on (M, kappa) from a flat guess.  Add a
      small Tikhonov prior (§4iv) and watch it change WHICH minimum you reach.
      Then add composition-diverse initial conditions and measure the
      conditioning improvement — the beyond-FH rung-2 lesson, hands-on.
""")
