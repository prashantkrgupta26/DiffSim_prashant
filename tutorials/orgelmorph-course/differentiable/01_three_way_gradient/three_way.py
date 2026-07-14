"""OrgElMorph course - Differentiable D1: what a differentiable solver is.

Importable core for the first differentiable concept.  A classical solver
answers "what happens?"; a differentiable solver also answers "what
should I change?" - it returns the gradient dJ/dp of any scalar output J
with respect to any input parameter p.  This concept computes that
gradient THREE independent ways through a short Cahn-Hilliard march and
shows they agree, because a gradient you cannot trust is worse than none.

The three ways (the milestone's non-negotiable verification pattern):

  1. ADJOINT       - a hand-derived reverse sweep (implicit-function
                     theorem through the converged Newton relation).  This
                     is the production capability: O(1) cost regardless of
                     how many parameters, exact to machine precision.
  2. AUTOGRAD TWIN - a faithful PyTorch reimplementation of the SAME
                     discrete solver, differentiated by torch.autograd.
                     An INDEPENDENT reference: different code, different
                     differentiation engine, same math.
  3. FINITE DIFF   - central differences (J(p+e)-J(p-e))/2e.  No calculus
                     at all; the ground truth every gradient must match
                     (to the O(e^2) truncation + roundoff floor of FD).

Nothing here is a toy: the adjoint (src/diffsim/adjoint/phasefield.py) is
bit-parity with the production CahnHilliardStepper, so the gradient is
taken through the REAL brick.  We use the Flory-Huggins bulk energy with
interior initial data so the entropic logarithms stay smooth.

Objective:  J = 1/2 |c_N - 1/2|^2  (a simple scalar of the final field).
Parameters: M (mobility), kappa (gradient penalty), B = chi (Flory
interaction), A (entropic scale).
"""
import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.adjoint.phasefield import CHForward, CHAdjoint, FHEnergy


def build_dm(level=3, device="cuda:0"):
    """Small uniform 2-D box (level 3 = 8x8 cells, 81 nodes).  Tiny on
    purpose: reverse-mode differentiation stores the whole forward
    trajectory, so tutorial rollouts stay short and coarse."""
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    return dm, mesh


def _ic(coords):
    """Interior, smooth initial blend (stays inside (0,1) so the FH logs
    never hit the walls - no non-differentiable clamp is ever needed)."""
    return 0.5 + 0.1 * np.cos(np.pi * coords[:, 0]) \
        * np.cos(np.pi * coords[:, 1])


# --- the objective and its seed-cotangent ------------------------------
# J = 1/2 |c_N - target|^2, so dJ/dc_N = (c_N - target) and every earlier
# step contributes nothing directly (the objective only reads the final
# field).  The adjoint takes that one seed and propagates it backward.
def objective(cN, target):
    return 0.5 * float(((cN - target) ** 2).sum())


def run_forward(dm, coords, params, n_steps, dt=0.01, order=1):
    """March the discrete CH forward and return the recorder + final c."""
    fwd = CHForward(dm, FHEnergy(params["A"], params["B"]),
                    M=params["M"], kappa=params["kappa"], dt=dt, order=order)
    fwd.set_initial(_ic(coords))
    fwd.run(n_steps)
    return fwd, fwd.steps[-1]["c"]


def grad_adjoint(dm, coords, params, n_steps, names, dt=0.01, order=1):
    """Way 1 - the hand adjoint.  One reverse sweep returns dJ/dp for ALL
    parameters at once (that O(1)-in-parameters cost is the whole point
    of the adjoint method)."""
    nn = dm.n_nodes
    target = np.full(nn, 0.5)
    fwd, cN = run_forward(dm, coords, params, n_steps, dt, order)
    dJdx = [np.zeros(2 * nn) for _ in range(n_steps)]
    dJdx[-1][0::2] = cN - target          # dJ/dc on the final step's c-rows
    return CHAdjoint(fwd).gradient(dJdx, names), objective(cN, target)


def grad_finite_diff(dm, coords, params, n_steps, names, eps_rel=1e-6,
                     dt=0.01, order=1):
    """Way 3 - central finite differences.  Perturb one parameter at a
    time; no derivatives, just two extra forward solves per parameter."""
    nn = dm.n_nodes
    target = np.full(nn, 0.5)

    def J_of(p):
        _, cN = run_forward(dm, coords, p, n_steps, dt, order)
        return objective(cN, target)

    g = {}
    for name in names:
        eps = eps_rel * max(1.0, abs(params[name]))
        hi = dict(params); hi[name] += eps
        lo = dict(params); lo[name] -= eps
        g[name] = (J_of(hi) - J_of(lo)) / (2 * eps)
    return g


def grad_autograd_twin(dm, coords, params, n_steps, names, dt=0.01,
                       order=1):
    """Way 2 - the PyTorch autograd twin (CPU).  Same weak form and dof
    layout as the adjoint, but differentiated by torch.autograd through
    the converged Newton iterations - an independent gradient."""
    import torch
    from diffsim.adjoint.torch_twin import CHTwin
    nn = dm.n_nodes
    target = np.full(nn, 0.5)
    twin = CHTwin(dm, energy="fh", dt=dt, order=order, device="cpu")
    c0 = torch.tensor(_ic(coords))
    leaves = {p: torch.tensor(float(params[p]), requires_grad=True)
              for p in ("M", "kappa", "A", "B")}
    out = twin.march(c0, None, leaves["M"], leaves["kappa"],
                     dict(A=leaves["A"], B=leaves["B"]), n_steps)
    loss = 0.5 * ((out[-1][0] - torch.tensor(target)) ** 2).sum()
    loss.backward()
    return {p: float(leaves[p].grad) for p in names}


def three_way(level=3, n_steps=3, dt=0.01, order=1, device="cuda:0",
              params=None, names=("M", "kappa", "B", "A")):
    """Compute dJ/dp three ways and return everything the tutorial cites.

    Returns a dict with per-parameter (adjoint, twin, fd) triples and the
    worst-case relative agreements adj-vs-twin and adj-vs-FD.  The
    agreements ARE the gate: adjoint == twin to ~1e-16 (same math, two
    engines) and adjoint == FD to ~1e-9 (the FD truncation/roundoff
    floor)."""
    if params is None:
        params = dict(M=1.0, kappa=0.01, A=1.0, B=2.5)
    names = list(names)
    dm, mesh = build_dm(level, device)
    coords = mesh.node_coords

    g_adj, J = grad_adjoint(dm, coords, params, n_steps, names, dt, order)
    g_tw = grad_autograd_twin(dm, coords, params, n_steps, names, dt, order)
    g_fd = grad_finite_diff(dm, coords, params, n_steps, names, 1e-6, dt,
                            order)

    grads = {p: (g_adj[p], g_tw[p], g_fd[p]) for p in names}
    rel_tw = {p: abs(g_adj[p] - g_tw[p]) / max(abs(g_tw[p]), 1e-14)
              for p in names}
    rel_fd = {p: abs(g_adj[p] - g_fd[p]) / max(abs(g_fd[p]), 1e-14)
              for p in names}
    return dict(grads=grads, rel_tw=rel_tw, rel_fd=rel_fd, J=J,
                params=params, names=names, n_steps=n_steps, order=order,
                nn=dm.n_nodes, side=int(round(np.sqrt(dm.n_nodes))))


def fd_stepsize_study(level=3, n_steps=3, dt=0.01, order=1,
                      device="cuda:0", name="B",
                      exps=range(-1, -12, -1), params=None):
    """The classic gradient-check picture: sweep the FD step e and plot
    |FD(e) - adjoint| against e.  It forms a V: for large e the O(e^2)
    truncation error dominates (falls as e shrinks); for tiny e
    catastrophic cancellation (roundoff) takes over (rises).  The valley
    floor is the best FD can do - and the adjoint sits BELOW it, at
    machine precision, for FREE (no step to tune).  Returns (eps, err,
    g_adj)."""
    if params is None:
        params = dict(M=1.0, kappa=0.01, A=1.0, B=2.5)
    dm, mesh = build_dm(level, device)
    coords = mesh.node_coords
    nn = dm.n_nodes
    target = np.full(nn, 0.5)
    g_adj = grad_adjoint(dm, coords, params, n_steps, [name], dt, order)[0][name]

    def J_of(p):
        _, cN = run_forward(dm, coords, p, n_steps, dt, order)
        return objective(cN, target)

    eps = np.array([10.0 ** e for e in exps])
    err = np.zeros_like(eps)
    for i, e in enumerate(eps):
        hi = dict(params); hi[name] += e
        lo = dict(params); lo[name] -= e
        g_fd = (J_of(hi) - J_of(lo)) / (2 * e)
        err[i] = abs(g_fd - g_adj)
    return eps, err, g_adj
