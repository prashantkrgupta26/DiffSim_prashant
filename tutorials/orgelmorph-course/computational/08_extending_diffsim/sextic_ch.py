"""OrgElMorph course - Computational C8: extending DiffSim safely.

A WORKED REFERENCE for the contributor workflow: add a small new free-energy
term to a Cahn-Hilliard model and carry it through every gate a real
contribution must pass -- equations + units, residual, ANALYTIC Jacobian, a
finite-difference derivative check, a limiting-case check, an MMS convergence
study, conservation + energy invariants -- before it is allowed near production.

The model is a transparent, readable-NumPy mixed Cahn-Hilliard solver built on
the SAME diffsim mesh / basis / quadrature infrastructure the production stepper
uses (exactly as Chapter C0's ``scalar_reaction.py`` does for a scalar problem),
so every stage is visible.  Backward-Euler in time, no-flux (natural) boundaries
unless Dirichlet data is supplied (for MMS):

    c_t = M div(grad mu),               weak:  INT w (c-c_old)/dt + M INT grad w . grad mu = 0
    mu  = f'(c) - kappa lap c,           weak:  INT q mu - INT q f'(c) - kappa INT grad q . grad c = 0

THE NEW TERM.  We extend the standard quartic double well with a sixth-order
Landau term (a higher-order stabiliser that steepens the well and bounds deep
quenches):

    f(c) = 1/4 (c^2 - 1)^2  +  (beta/6) c^6         [energy density, energy/volume]
    f'(c) = c^3 - c         +  beta c^5             (enters the mu residual)
    f''(c) = 3 c^2 - 1      +  5 beta c^4           (enters the mu-c Jacobian block)

``beta`` [dimensionless, energy-scaled] is the new coefficient; ``beta = 0``
recovers the base model exactly (the limiting case).  Getting ``f''`` -- the
analytic Jacobian contribution ``5 beta c^4`` -- right is the crux of the
contribution, and ``test_sextic_ch.py`` checks it against a finite difference.

Units: fields ``c, mu`` dimensionless order parameters; ``M`` mobility
[length^2/time], ``kappa`` [energy*length^2/volume] sets the interface width
ell ~ sqrt(kappa / W); quadrature weights carry length^dim.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.poisson import gauss_points

M_DEFAULT, KAPPA_DEFAULT = 1.0, 0.02


# ---------------------------------------------------------------------------
# the free energy: base double well + the NEW sextic term
# ---------------------------------------------------------------------------
def f_bulk(c, beta):
    """Bulk free-energy density f(c) = 1/4 (c^2-1)^2 + (beta/6) c^6."""
    return 0.25 * (c ** 2 - 1.0) ** 2 + (beta / 6.0) * c ** 6


def fprime(c, beta):
    """f'(c) = c^3 - c + beta c^5 (enters the mu residual)."""
    return c ** 3 - c + beta * c ** 5


def fdoubleprime(c, beta):
    """f''(c) = 3 c^2 - 1 + 5 beta c^4 (enters the mu-c Jacobian block)."""
    return 3.0 * c ** 2 - 1.0 + 5.0 * beta * c ** 4


def sextic_term(c, beta):
    """The NEW term in isolation: value, first, second derivative -- the object
    of the focused derivative unit test."""
    return dict(f=(beta / 6.0) * c ** 6, fp=beta * c ** 5, fpp=5.0 * beta * c ** 4)


# ---------------------------------------------------------------------------
# mesh + precomputed (constant, uniform-mesh) element operators
# ---------------------------------------------------------------------------
class Problem:
    """Transparent mixed-CH problem on a uniform box, degree-p elements.

    State ``x`` is interleaved ``[c0, mu0, c1, mu1, ...]`` (length ``2n``), the
    same layout as the production stepper.  For a uniform mesh the mass ``Me``
    and stiffness ``Ke`` element matrices and the physical basis gradients are
    constant, so we build them once; only the nonlinear ``f'``/``f''`` terms are
    re-evaluated at the Gauss points each Newton iteration.
    """

    def __init__(self, level=5, p=1, M=M_DEFAULT, kappa=KAPPA_DEFAULT,
                 device="cpu"):
        self.level, self.p, self.dim = level, p, 2
        self.M, self.kappa = float(M), float(kappa)
        tree = build_uniform(level, dim=2)
        self.mesh = build_mesh(tree, p)
        self.cons = build_constraints(self.mesh)
        self.tables = basis_tables(p, dim=2)
        self.dm = DeviceMesh.from_mesh(self.mesh, self.cons, self.tables, device)

        self.conn = self.mesh.conn_of[p].astype(np.int64)
        self.ne, self.nbf = self.conn.shape
        self.nqp = self.tables.nqp
        self.h = float(self.mesh.tree.h()[self.mesh.bins[p][0]])
        self.n = len(self.mesh.node_coords)
        self.bnd = self.mesh.boundary_nodes

        self.N = self.tables.N                              # [nqp, nbf]
        self.detJxW = self.tables.w * (self.h / 2.0) ** self.dim
        self.gN = self.tables.dN * (2.0 / self.h)           # [nqp, nbf, dim]
        # constant element operators
        self.Me = np.einsum("qa,qb,q->ab", self.N, self.N, self.detJxW)
        self.Ke = np.einsum("qak,qbk,q->ab", self.gN, self.gN, self.detJxW)
        # physical Gauss-point coordinates, per degree (flat (e,q) order)
        self.xq = gauss_points(self.mesh, self.dm.tables_by_p)[p]

    # -- helpers ----------------------------------------------------------
    def gp_value(self, u):
        """u at every Gauss point: [ne, nqp]."""
        return np.einsum("qb,eb->eq", self.N, u[self.conn])

    def load(self, fn):
        """Assemble INT N_a fn(x) for a source sampled at Gauss points."""
        fq = fn(self.xq).reshape(self.ne, self.nqp)
        be = np.einsum("qa,eq,q->ea", self.N, fq, self.detJxW)
        F = np.zeros(self.n)
        np.add.at(F, self.conn.ravel(), be.ravel())
        return F


# ---------------------------------------------------------------------------
# residual + analytic Jacobian (the contribution)
# ---------------------------------------------------------------------------
def residual(prob, x, c_old, dt, beta, src_c=None, src_m=None,
             dirichlet=None, gc=None, gm=None):
    """Global residual R(x) of the backward-Euler mixed CH step (length 2n).

    R_c = Me (c - c_old)/dt + M Ke mu           [- INT N f_c]
    R_mu = Me mu - b_fp(c) - kappa Ke c         [- INT N f_m]
    where b_fp = INT N_a f'(c) is the ONLY nonlinear assembly and carries the
    new term through f'(c) = c^3 - c + beta c^5.
    """
    c, mu = x[0::2], x[1::2]
    ce, mue, colde = c[prob.conn], mu[prob.conn], c_old[prob.conn]
    # linear (constant-operator) contributions, per element
    Rc = np.einsum("ab,eb->ea", prob.Me, (ce - colde) / dt) \
        + prob.M * np.einsum("ab,eb->ea", prob.Ke, mue)
    # nonlinear f'(c) integrated at Gauss points
    cq = prob.gp_value(c)                                    # [ne, nqp]
    b_fp = np.einsum("qa,eq,q->ea", prob.N, fprime(cq, beta), prob.detJxW)
    Rm = np.einsum("ab,eb->ea", prob.Me, mue) - b_fp \
        - prob.kappa * np.einsum("ab,eb->ea", prob.Ke, ce)

    R = np.zeros(2 * prob.n)
    np.add.at(R, prob.conn.ravel() * 2, Rc.ravel())          # c rows
    np.add.at(R, prob.conn.ravel() * 2 + 1, Rm.ravel())      # mu rows
    if src_c is not None:
        R[0::2] -= src_c
    if src_m is not None:
        R[1::2] -= src_m
    if dirichlet is not None:                                # strong BC (MMS)
        R[dirichlet * 2] = x[dirichlet * 2] - gc
        R[dirichlet * 2 + 1] = x[dirichlet * 2 + 1] - gm
    return R


def jacobian(prob, x, dt, beta, dirichlet=None):
    """Global analytic tangent J = dR/dx (2n x 2n CSR).

    Element blocks (interleaved c, mu per node):
        J_cc = Me/dt              J_cmu = M Ke
        J_muc = -INT N f''(c) N - kappa Ke        J_mumu = Me
    The NEW term enters J_muc through f''(c) = 3c^2 - 1 + 5 beta c^4.
    """
    c = x[0::2]
    cq = prob.gp_value(c)
    # c-dependent mass-like block  INT N_a f''(c) N_b
    fpp = fdoubleprime(cq, beta)                             # [ne, nqp]
    Jfpp = np.einsum("qa,eq,qb,q->eab", prob.N, fpp, prob.N, prob.detJxW)

    ne, nbf = prob.ne, prob.nbf
    Ae = np.zeros((ne, 2 * nbf, 2 * nbf))
    # local dof ordering: [c0,mu0,c1,mu1,...]; c at 2a, mu at 2a+1
    cc, mm = slice(0, 2 * nbf, 2), slice(1, 2 * nbf, 2)
    Ae[:, cc, cc] = prob.Me / dt                             # J_cc
    Ae[:, cc, mm] = prob.M * prob.Ke                         # J_cmu
    Ae[:, mm, cc] = -Jfpp - prob.kappa * prob.Ke             # J_muc
    Ae[:, mm, mm] = prob.Me                                  # J_mumu

    gdof = (prob.conn[:, :, None] * 2 + np.arange(2)[None, None, :]) \
        .reshape(ne, 2 * nbf)
    rows = np.repeat(gdof, 2 * nbf, axis=1).ravel()
    cols = np.tile(gdof, (1, 2 * nbf)).ravel()
    J = sp.coo_matrix((Ae.ravel(), (rows, cols)),
                      shape=(2 * prob.n, 2 * prob.n)).tolil()
    if dirichlet is not None:                                # Dirichlet rows -> I
        for i in dirichlet:
            for comp in (0, 1):
                r = int(i * 2 + comp)
                J.rows[r] = [r]
                J.data[r] = [1.0]
    return J.tocsr()


def newton_solve(prob, x0, c_old, dt, beta, tol=1e-10, max_it=30,
                 src_c=None, src_m=None, dirichlet=None, gc=None, gm=None):
    """Monolithic Newton for one backward-Euler step.  Returns (x, info)."""
    from scipy.sparse.linalg import splu
    x = np.array(x0, float)
    R = residual(prob, x, c_old, dt, beta, src_c, src_m, dirichlet, gc, gm)
    hist = [float(np.linalg.norm(R))]
    it = 0
    for it in range(1, max_it + 1):
        if hist[-1] < tol:
            break
        J = jacobian(prob, x, dt, beta, dirichlet)
        dx = splu(J.tocsc()).solve(-R)
        x = x + dx
        R = residual(prob, x, c_old, dt, beta, src_c, src_m, dirichlet, gc, gm)
        hist.append(float(np.linalg.norm(R)))
    return x, {"iters": it, "converged": hist[-1] < tol, "res_norm": hist[-1],
               "res_history": hist}


# ---------------------------------------------------------------------------
# diagnostics: mass, energy, MMS error (quadrature, not nodal proxies)
# ---------------------------------------------------------------------------
def total_mass(prob, c):
    """INT c dV by Gauss quadrature (the conserved quantity)."""
    cq = prob.gp_value(c)
    w = np.broadcast_to(prob.detJxW, (prob.ne, prob.nqp))
    return float(np.sum(w * cq))


def total_energy(prob, c, beta):
    """F = INT [ f(c) + (kappa/2)|grad c|^2 ] dV by quadrature (Lyapunov)."""
    cq = prob.gp_value(c)
    gcq = np.einsum("qbk,eb->eqk", prob.gN, c[prob.conn])    # [ne, nqp, dim]
    w = np.broadcast_to(prob.detJxW, (prob.ne, prob.nqp))
    dens = f_bulk(cq, beta) + 0.5 * prob.kappa * np.sum(gcq ** 2, axis=2)
    return float(np.sum(w * dens))


def l2_error(prob, c, c_star_fn):
    """||c_h - c*||_L2 on the solver's own quadrature (MMS)."""
    cq = prob.gp_value(c)
    cs = c_star_fn(prob.xq).reshape(prob.ne, prob.nqp)
    w = np.broadcast_to(prob.detJxW, (prob.ne, prob.nqp))
    return float(np.sqrt(np.sum(w * (cq - cs) ** 2)))


def boundary_free_nodes(prob):
    """Boundary node indices (into the global node numbering)."""
    return np.where(prob.bnd)[0]


# ---------------------------------------------------------------------------
# method of manufactured solutions (steady) -- shared by run.py and the tests
# ---------------------------------------------------------------------------
def c_star(x):
    """Manufactured composition c*(x) = cos(pi x) cos(pi y) (steady, smooth)."""
    return np.cos(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])


def mu_star(x):
    """Manufactured potential mu*(x) = sin(pi x) sin(pi y)."""
    return np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])


def mms_sources(prob, beta):
    """The steady MMS load vectors ``(src_c, src_m)`` that make (c*, mu*) an
    exact solution.  Both fields have Laplacian ``-2 pi^2 (field)``:

        s1 = -M lap(mu*) = 2 pi^2 M mu*
        s2 = mu* - f'(c*) + kappa lap(c*)      (carries the new term via f')
    """
    lap = -2.0 * np.pi ** 2

    def s1(x):
        return -prob.M * lap * mu_star(x)

    def s2(x):
        c = c_star(x)
        return mu_star(x) - fprime(c, beta) + prob.kappa * lap * c

    return prob.load(s1), prob.load(s2)


def solve_mms(prob, beta, nsteps=3, dt=1e3, tol=1e-11):
    """March the STEADY manufactured problem to its discrete steady state
    (pinned c*, mu* boundaries).  Returns (c, info); the L2 error of c vs c*
    is pure spatial discretization error (~h^{p+1}), independent of dt."""
    src_c, src_m = mms_sources(prob, beta)
    coords = prob.mesh.node_coords
    dnodes = boundary_free_nodes(prob)
    gc, gm = c_star(coords[dnodes]), mu_star(coords[dnodes])
    x = np.zeros(2 * prob.n)
    x[0::2] = c_star(coords)
    x[1::2] = mu_star(coords)
    c_old = c_star(coords)
    info = {"converged": False}
    for _ in range(nsteps):
        x, info = newton_solve(prob, x, c_old, dt, beta, tol=tol,
                               src_c=src_c, src_m=src_m, dirichlet=dnodes,
                               gc=gc, gm=gm)
        c_old = x[0::2].copy()
    return x[0::2].copy(), info


def march(prob, c0, beta, dt, nsteps, tol=1e-10):
    """No-flux backward-Euler march from ``c0``; returns per-step mass and
    energy series + the final state.  Mass is conserved (Model-B, no flux);
    energy decreases (Lyapunov)."""
    x = np.zeros(2 * prob.n)
    x[0::2] = c0
    c_old = c0.copy()
    masses = [total_mass(prob, x[0::2])]
    energies = [total_energy(prob, x[0::2], beta)]
    iters = []
    for _ in range(nsteps):
        x, info = newton_solve(prob, x, c_old, dt, beta, tol=tol)
        c_old = x[0::2].copy()
        masses.append(total_mass(prob, x[0::2]))
        energies.append(total_energy(prob, x[0::2], beta))
        iters.append(info["iters"])
    return dict(c=x[0::2].copy(), mass=masses, energy=energies, iters=iters)
