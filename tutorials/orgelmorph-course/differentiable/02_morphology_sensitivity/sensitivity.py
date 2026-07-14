"""OrgElMorph course - Differentiable D2: morphology sensitivity.

Importable core for the second differentiable concept.  Now that we trust
the gradient (D1), we ask a physical question: how sensitive is the
MORPHOLOGY to the Flory interaction parameter chi?  That is
d(observable)/dchi through a short Cahn-Hilliard rollout.

The morphology observable is the DEMIXING AMPLITUDE

    J(c_N) = 1/2 sum_i (c_N,i - c_bar)^2 ,   c_bar = mean(c_0),

the spatial variance of the final composition field.  It is small when
the blend is still nearly uniform and grows as the two phases separate,
so it is a scalar proxy for "how demixed is the film".  Because the
dynamics conserve mass, c_bar is fixed by the initial condition, so
dJ/dc_N = (c_N - c_bar) is exact and the adjoint carries it backward to

    dJ/dchi   (does raising the interaction sharpen the morphology?)
    dJ/dkappa (does a stiffer gradient penalty blur it?)

Both are verified against central finite differences.  The sign is the
physics: above the spinodal a larger chi drives demixing, so dJ/dchi>0;
a larger kappa (thicker interfaces, more gradient penalty) opposes it, so
dJ/dkappa<0.
"""
import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.adjoint.phasefield import CHForward, CHAdjoint, FHEnergy


def build_dm(level=3, device="cuda:0"):
    """Small uniform 2-D box (level 3 = 81 nodes) - short reverse-mode
    rollouts stay coarse."""
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    return dm, mesh


def _ic(coords):
    """Interior initial blend with a single cosine mode that the spinodal
    instability amplifies over the rollout (kept inside (0,1))."""
    return 0.5 + 0.1 * np.cos(np.pi * coords[:, 0]) \
        * np.cos(np.pi * coords[:, 1])


def demixing_amplitude(c, c_bar):
    """The morphology observable: spatial variance of the composition
    (grows as the blend separates)."""
    return 0.5 * float(((c - c_bar) ** 2).sum())


def march(dm, coords, chi, kappa, n_steps, dt=0.01, M=1.0, A=1.0,
          order=1):
    """Forward CH rollout; returns the recorder and the final field."""
    fwd = CHForward(dm, FHEnergy(A, chi), M=M, kappa=kappa, dt=dt,
                    order=order)
    fwd.set_initial(_ic(coords))
    fwd.run(n_steps)
    return fwd, fwd.steps[-1]["c"]


def sensitivity(level=3, n_steps=6, dt=0.005, chi=2.5, kappa=1e-2, M=1.0,
                A=1.0, order=1, device="cuda:0"):
    """Compute the morphology observable and its adjoint sensitivities
    dJ/dchi, dJ/dkappa, each finite-difference verified.  Returns
    everything the tutorial cites."""
    dm, mesh = build_dm(level, device)
    coords = mesh.node_coords
    nn = dm.n_nodes
    c_bar = float(_ic(coords).mean())

    fwd, cN = march(dm, coords, chi, kappa, n_steps, dt, M, A, order)
    J = demixing_amplitude(cN, c_bar)

    # adjoint: one reverse sweep -> both sensitivities.  Seed dj/dc_N.
    dJdx = [np.zeros(2 * nn) for _ in range(n_steps)]
    dJdx[-1][0::2] = cN - c_bar
    g_adj = CHAdjoint(fwd).gradient(dJdx, ["B", "kappa"])
    dJdchi_adj, dJdkap_adj = g_adj["B"], g_adj["kappa"]

    # finite-difference verification
    def J_of(chi_, kap_):
        _, c = march(dm, coords, chi_, kap_, n_steps, dt, M, A, order)
        return demixing_amplitude(c, c_bar)

    e_chi = 1e-6 * max(1.0, abs(chi))
    e_kap = 1e-6 * max(1.0, abs(kappa))
    dJdchi_fd = (J_of(chi + e_chi, kappa) - J_of(chi - e_chi, kappa)) \
        / (2 * e_chi)
    dJdkap_fd = (J_of(chi, kappa + e_kap) - J_of(chi, kappa - e_kap)) \
        / (2 * e_kap)

    rel_chi = abs(dJdchi_adj - dJdchi_fd) / max(abs(dJdchi_fd), 1e-14)
    rel_kap = abs(dJdkap_adj - dJdkap_fd) / max(abs(dJdkap_fd), 1e-14)

    side = int(round(np.sqrt(nn)))
    return dict(J=J, chi=chi, kappa=kappa, n_steps=n_steps, c_bar=c_bar,
                dJdchi_adj=dJdchi_adj, dJdchi_fd=dJdchi_fd, rel_chi=rel_chi,
                dJdkap_adj=dJdkap_adj, dJdkap_fd=dJdkap_fd, rel_kap=rel_kap,
                cN=cN, side=side, nn=nn,
                morph=np.asarray(cN).reshape(side, side))


def chi_sweep(level=3, n_steps=6, dt=0.005, kappa=1e-2, M=1.0, A=1.0,
              order=1, device="cuda:0",
              chis=(2.1, 2.3, 2.5, 2.7, 2.9)):
    """Trace the morphology observable J(chi) across a range of
    interaction strengths, and the adjoint slope dJ/dchi at each point.
    The slope is the tangent to J(chi): the figure overlays the adjoint
    tangents on the sampled curve so the student SEES that the gradient
    is the local slope of the response."""
    dm, mesh = build_dm(level, device)
    coords = mesh.node_coords
    nn = dm.n_nodes
    c_bar = float(_ic(coords).mean())
    chis = np.asarray(chis, float)
    J = np.zeros_like(chis)
    slope = np.zeros_like(chis)
    for i, chi in enumerate(chis):
        fwd, cN = march(dm, coords, chi, kappa, n_steps, dt, M, A, order)
        J[i] = demixing_amplitude(cN, c_bar)
        dJdx = [np.zeros(2 * nn) for _ in range(n_steps)]
        dJdx[-1][0::2] = cN - c_bar
        slope[i] = CHAdjoint(fwd).gradient(dJdx, ["B"])["B"]
    return chis, J, slope
