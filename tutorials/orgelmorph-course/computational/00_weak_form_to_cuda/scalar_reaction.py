"""OrgElMorph course - Computational C0: weak form to CUDA, the transparent core.

A ONE-unknown scalar nonlinear problem, assembled and Newton-solved in readable
NumPy/SciPy so you can see EVERY stage the production solver hides inside Warp
kernels.  Each stage below is annotated with the exact
``src/diffsim/...`` file + function that does the same thing on the device --
this file is the map, ``cahn_hilliard.py`` / ``example_bricks.py`` are the
territory.

The model (nonlinear reaction-diffusion, unit square, Dirichlet):

    -div(grad u) + alpha * u**3 = f      in Omega = (0,1)^2
                              u = g        on the boundary

Strong -> weak (test with w, w=0 on the Dirichlet boundary, integrate the
Laplacian by parts):

    R(w; u) = INT grad w . grad u dV        (diffusion / stiffness)
            + alpha INT w u**3 dV           (the REACTION term)
            - INT w f dV                    (the load)          =  0   for all w.

Newton linearizes R about the current iterate u_k.  The directional derivative
of R in a trial direction N_b gives the Jacobian (tangent) J = dR/du:

    J_ab = INT grad N_a . grad N_b dV               (stiffness, u-independent)
         + alpha INT N_a (3 u_k**2) N_b dV          (the REACTION JACOBIAN)

Only the SECOND Jacobian term is new when you add the reaction coefficient: the
derivative of alpha*u**3 is 3*alpha*u**2, a u-weighted mass matrix.  Getting
that analytic contribution right is the hands-on task of this chapter
(``scalar_reaction_starter.py`` is this file with that one term removed, so its
Jacobian is wrong and ``test_reaction_jacobian.py`` fails against it).

Method of manufactured solutions (MMS): pick u* = sin(pi x) sin(pi y) (which is
zero on the boundary, so g = 0), and feed the source that makes it exact:

    f = 2 pi^2 u*  +  alpha u*^3 .
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

# Real DiffSim infrastructure -- the SAME mesh, basis tables and Gauss points
# the production solver uses (nothing here is a re-implementation of those).
from diffsim.octree.build import build_uniform            # octree/build.py
from diffsim.mesh.nodes import build_mesh                 # mesh/nodes.py
from diffsim.mesh.constraints import build_constraints    # mesh/constraints.py
from diffsim.mesh.basis import basis_tables               # mesh/basis.py
from diffsim.assembly.operators import DeviceMesh         # assembly/operators.py
from diffsim.physics.poisson import gauss_points          # physics/poisson.py
from diffsim.solvers.linsolve import solve_linear         # solvers/linsolve.py

ALPHA_DEFAULT = 12.0     # reaction strength; large enough that a WRONG Jacobian
                         # (frozen stiffness) visibly degrades Newton convergence


# ---------------------------------------------------------------------------
# manufactured solution + its source
# ---------------------------------------------------------------------------
def manufactured(x):
    """u*(x) = sin(pi x) sin(pi y): smooth, and zero on the box boundary."""
    return np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])


def source(x, alpha):
    """f = -lap(u*) + alpha u*^3 = 2 pi^2 u* + alpha u*^3 (2-D)."""
    us = manufactured(x)
    return 2.0 * np.pi ** 2 * us + alpha * us ** 3


# ---------------------------------------------------------------------------
# STAGE 1 -- mesh + basis (build_uniform -> build_mesh -> DeviceMesh)
# ---------------------------------------------------------------------------
class Problem:
    """Everything the assembly needs, built from the real DiffSim mesh path.

    Mirrors ``spinodal.build_mesh_dm`` / ``tests/test_cahn_hilliard.py``:
        build_uniform(level, dim) -> build_mesh(tree, p) -> build_constraints
        -> basis_tables(p, dim) -> DeviceMesh.from_mesh(...).
    For a UNIFORM box the constraint matrix T is the identity (no hanging
    nodes), so the free-node system is just the node system and we impose
    Dirichlet data by row replacement -- exactly as
    ``CahnHilliardStepper.step`` does for its constrained rows.
    """

    def __init__(self, level=5, p=1, alpha=ALPHA_DEFAULT, device="cpu"):
        self.level, self.p, self.alpha, self.dim = level, p, alpha, 2
        tree = build_uniform(level, dim=2)
        self.mesh = build_mesh(tree, p)
        self.cons = build_constraints(self.mesh)
        self.tables = basis_tables(p, dim=2)
        self.dm = DeviceMesh.from_mesh(self.mesh, self.cons, self.tables, device)

        # per-degree connectivity [ne, nbf] and the (uniform) element size
        self.conn = self.mesh.conn_of[p].astype(np.int64)
        self.ne, self.nbf = self.conn.shape
        self.nqp = self.tables.nqp
        self.h = float(self.mesh.tree.h()[self.mesh.bins[p][0]])
        self.n = len(self.mesh.node_coords)
        self.bnd = self.mesh.boundary_nodes                    # bool [n]

        # quadrature: reference tables + the octree chain-rule factors
        #   physical grad N = (2/h) * reference grad N ;  |J| = (h/2)^dim
        self.N = self.tables.N                                 # [nqp, nbf]
        self.dscale = 2.0 / self.h
        self.detJxW = self.tables.w * (self.h / 2.0) ** self.dim   # [nqp]
        # physical-space gradient of every basis fn at every qp: [nqp, nbf, dim]
        self.gN = self.tables.dN * self.dscale

        # physical Gauss-point coordinates (production helper, (e,q) flat order)
        self.xq = gauss_points(self.mesh, self.dm.tables_by_p)[p]   # [ne*nqp, dim]

        # STAGE 3a -- the diffusion (stiffness) element matrices are CONSTANT
        # (u-independent), so assemble them once.  Ke[e,a,b] = INT gN_a . gN_b.
        # einsum over (q, dim): sum_q sum_k gN[q,a,k] gN[q,b,k] detJxW[q].
        Ke = np.einsum("qak,qbk,q->ab", self.gN, self.gN, self.detJxW)
        self._Ke = np.broadcast_to(Ke, (self.ne, self.nbf, self.nbf))
        # the boundary Dirichlet values g(x) at boundary nodes (here g = 0)
        self.gvals = manufactured(self.mesh.node_coords)
        # the load (right-hand side) integral, also constant
        self._Fload = self._assemble_load()

    # -- STAGE 3c: sample the MMS source at Gauss points, integrate w f -------
    def _assemble_load(self):
        fq = source(self.xq, self.alpha).reshape(self.ne, self.nqp)
        # be[e,a] = sum_q N[q,a] fq[e,q] detJxW[q]
        be = np.einsum("qa,eq,q->ea", self.N, fq, self.detJxW)
        F = np.zeros(self.n)
        np.add.at(F, self.conn.ravel(), be.ravel())            # local -> global
        return F


# ---------------------------------------------------------------------------
# STAGE 3 -- element residual + Jacobian (the physics), quadrature by einsum
# ---------------------------------------------------------------------------
def gp_value(prob, u):
    """u at every Gauss point: uq[e,q] = sum_b N[q,b] u[conn[e,b]]."""
    ue = u[prob.conn]                                          # [ne, nbf]
    return np.einsum("qb,eb->eq", prob.N, ue)                  # [ne, nqp]


def reaction_residual_elem(prob, uq):
    """alpha INT N_a u^3 dV, per element: [ne, nbf]."""
    rq = prob.alpha * uq ** 3                                  # [ne, nqp]
    return np.einsum("qa,eq,q->ea", prob.N, rq, prob.detJxW)


def reaction_jacobian_elem(prob, uq):
    """The analytic REACTION JACOBIAN contribution -- the hands-on term.

    d/du ( alpha u^3 ) = 3 alpha u^2, so the tangent gains a u-weighted mass
    matrix:  Jr[e,a,b] = INT N_a (3 alpha u_k^2) N_b dV.
    """
    wq = 3.0 * prob.alpha * uq ** 2                            # [ne, nqp]
    # sum_q N[q,a] wq[e,q] N[q,b] detJxW[q]
    return np.einsum("qa,eq,qb,q->eab", prob.N, wq, prob.N, prob.detJxW)


# ---------------------------------------------------------------------------
# STAGE 4 -- local-to-global assembly into a global CSR (+ Dirichlet rows)
# ---------------------------------------------------------------------------
def residual(prob, u):
    """Global residual vector R(u) with Dirichlet rows replaced by u - g."""
    # diffusion:  (K u)_e[a] = sum_b Ke[e,a,b] u[conn[e,b]]
    ue = u[prob.conn]
    r_diff = np.einsum("eab,eb->ea", prob._Ke, ue)
    r_reac = reaction_residual_elem(prob, gp_value(prob, u))
    be = r_diff + r_reac                                       # [ne, nbf]
    R = np.zeros(prob.n)
    np.add.at(R, prob.conn.ravel(), be.ravel())               # scatter
    R -= prob._Fload
    # STAGE 5b -- strong Dirichlet by row replacement (cahn_hilliard.step style)
    R[prob.bnd] = u[prob.bnd] - prob.gvals[prob.bnd]
    return R


def jacobian(prob, u):
    """Global tangent matrix J = dR/du as a SciPy CSR (Dirichlet rows = I)."""
    Ke = prob._Ke + reaction_jacobian_elem(prob, gp_value(prob, u))   # [ne,nbf,nbf]
    # COO triplets: every (a,b) node pair of every element (local -> global)
    rows = np.repeat(prob.conn, prob.nbf, axis=1).ravel()
    cols = np.tile(prob.conn, (1, prob.nbf)).ravel()
    J = sp.coo_matrix((Ke.ravel(), (rows, cols)),
                      shape=(prob.n, prob.n)).tocsr()
    # STAGE 5b -- Dirichlet rows -> identity (clear the row, put 1 on diagonal)
    J = J.tolil()
    for i in np.where(prob.bnd)[0]:
        J.rows[i] = [i]
        J.data[i] = [1.0]
    return J.tocsr()


# ---------------------------------------------------------------------------
# STAGE 6/7 -- Newton loop around the real diffsim linear solve
# ---------------------------------------------------------------------------
def newton_solve(prob, u0=None, tol=1e-10, max_it=25, solver="splu",
                 device="cuda:0"):
    """Monolithic Newton: assemble J,R; solve J du = -R; update; repeat.

    The linear solve is the production ``diffsim.solvers.linsolve.solve_linear``
    (``splu`` host direct here -- exact to round-off on this small system, so
    Newton is the only algebraic knob).  Returns ``(u, info)`` where info
    carries the residual-norm history and the iteration count.
    """
    u = np.zeros(prob.n) if u0 is None else np.array(u0, float)
    u[prob.bnd] = prob.gvals[prob.bnd]          # start on the Dirichlet data
    R = residual(prob, u)
    hist = [float(np.linalg.norm(R))]
    it = 0
    for it in range(1, max_it + 1):
        if hist[-1] < tol:
            break
        J = jacobian(prob, u)
        du = solve_linear(J, -R, solver=solver, device=device)
        u = u + du
        R = residual(prob, u)
        hist.append(float(np.linalg.norm(R)))
    converged = hist[-1] < tol
    return u, {"iters": it, "converged": converged,
               "res_norm": hist[-1], "res_history": hist}


# ---------------------------------------------------------------------------
# verification: L2 error on the solver's own quadrature (not a nodal proxy)
# ---------------------------------------------------------------------------
def l2_error(prob, u):
    """||u_h - u*||_L2, integrated at the assembly Gauss points."""
    uq = gp_value(prob, u)                                     # [ne, nqp]
    us = manufactured(prob.xq).reshape(prob.ne, prob.nqp)
    diff2 = (uq - us) ** 2
    return float(np.sqrt(np.einsum("eq,q->", diff2, prob.detJxW)))


def stiffness_csr(prob):
    """The transparent diffusion stiffness as a CSR (no BCs) -- used by the
    cross-check test against the production ``assemble_brick_csr`` kernel."""
    rows = np.repeat(prob.conn, prob.nbf, axis=1).ravel()
    cols = np.tile(prob.conn, (1, prob.nbf)).ravel()
    return sp.coo_matrix((prob._Ke.ravel(), (rows, cols)),
                         shape=(prob.n, prob.n)).tocsr()
