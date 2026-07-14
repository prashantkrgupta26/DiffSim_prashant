"""OrgElMorph course - Differentiable D5: crystallinity sensitivity.

Importable core for the fifth differentiable concept.  Organic-electronic
performance often hinges on CRYSTALLINITY - how much of the film has
ordered into the crystalline phase, and how fast.  This concept
differentiates a crystallinity metric through the coupled
Cahn-Hilliard x Allen-Cahn crystallisation rollout with respect to the
thermodynamic crystallisation parameters:

    dh    latent heat (Turnbull driving strength),
    Tm    melting temperature (sets the drive sign via T/Tm - 1),
    dsig  the double-well barrier height q(psi) dsig,
    eps2  the crystalline gradient penalty (interface stiffness), and
    L     the Allen-Cahn kinetic coefficient (how fast psi relaxes).

The metric is J = 1/2 || psi_N - psi_target ||^2 on the final
crystallinity field psi; the adjoint returns all five sensitivities in
one reverse sweep, each verified against central finite differences.  The
headline pair is (dh, Tm): raising the latent heat dh deepens the
crystallisation drive, so more psi forms.

A HONEST NOTE ON "NOISE".  Thermal nucleation is driven by a stochastic
noise term (Physics P8).  The adjoint here differentiates the
DETERMINISTIC crystallisation drift; a sensitivity to the NOISE AMPLITUDE
is a different object (a pathwise / score-function derivative of an
expectation) and is a documented frontier, not part of this deterministic
gate.  We differentiate the drift parameters that the merged adjoint
covers, and say so.
"""
import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.adjoint.phasefield import FHEnergy
from diffsim.adjoint.crystallization import CACHForward, CACHAdjoint

NF = 3   # (phi, mu, psi) per node

BASE = dict(M=1.0, kappa=0.02, eps2=0.02, L=1.0, dsig=1.0, dh=1.0, Tm=2.0,
            T=0.5, A=1.0, B=2.5)
# the crystallisation parameters this concept differentiates (headline first)
CRYST_PARAMS = ["dh", "Tm", "dsig", "eps2", "L"]


def build_dm(level=3, device="cuda:0"):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    return dm, mesh


def _ic(coords):
    cc = np.cos(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])
    return 0.5 + 0.1 * cc, 0.3 + 0.1 * cc          # phi0, psi0


def march(dm, coords, P, n_steps, dt=0.01, order=1):
    phi0, psi0 = _ic(coords)
    fwd = CACHForward(dm, FHEnergy(P["A"], P["B"]), M=P["M"],
                      kappa=P["kappa"], eps2=P["eps2"], L=P["L"],
                      dsig=P["dsig"], dh=P["dh"], Tm=P["Tm"], T=P["T"],
                      dt=dt, order=order)
    fwd.set_initial(phi0, psi0)
    fwd.run(n_steps)
    return fwd


def crystallinity_metric(fwd, tgt_psi):
    xN = fwd.steps[-1]["x"]
    return 0.5 * float(((xN[2::NF] - tgt_psi) ** 2).sum())


def mean_crystallinity(fwd):
    return float(np.mean(fwd.steps[-1]["x"][2::NF]))


def sensitivity(level=3, n_steps=3, dt=0.01, order=1, device="cuda:0",
                tgt_psi_val=0.4, names=None):
    """Compute the crystallinity metric and its adjoint sensitivities to
    the crystallisation parameters, each finite-difference verified.
    Returns everything the tutorial cites."""
    if names is None:
        names = list(CRYST_PARAMS)
    dm, mesh = build_dm(level, device)
    coords = mesh.node_coords
    nn = dm.n_nodes
    tgt_psi = np.full(nn, tgt_psi_val)

    fwd = march(dm, coords, BASE, n_steps, dt, order)
    J = crystallinity_metric(fwd, tgt_psi)
    frac = mean_crystallinity(fwd)

    # adjoint: one reverse sweep -> all five sensitivities.  Seed on psi.
    dJdx = [np.zeros(NF * nn) for _ in range(n_steps)]
    dJdx[-1][2::NF] = fwd.steps[-1]["x"][2::NF] - tgt_psi
    g_adj = CACHAdjoint(fwd).gradient(dJdx, names)

    # finite-difference verification
    def J_of(P):
        f = march(dm, coords, P, n_steps, dt, order)
        return crystallinity_metric(f, tgt_psi)

    g_fd, rel = {}, {}
    for nm in names:
        e = 1e-6 * max(1.0, abs(BASE[nm]))
        hi = dict(BASE); hi[nm] += e
        lo = dict(BASE); lo[nm] -= e
        g_fd[nm] = (J_of(hi) - J_of(lo)) / (2 * e)
        rel[nm] = abs(g_adj[nm] - g_fd[nm]) / max(abs(g_fd[nm]), 1e-14)

    side = int(round(np.sqrt(nn)))
    psi_field = np.asarray(fwd.steps[-1]["x"][2::NF]).reshape(side, side)
    return dict(J=J, frac=frac, g_adj=g_adj, g_fd=g_fd, rel=rel,
                names=names, n_steps=n_steps, nn=nn, side=side,
                tgt_psi=tgt_psi_val, psi_field=psi_field)
