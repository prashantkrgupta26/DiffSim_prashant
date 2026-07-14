"""OrgElMorph course - Differentiable D6: learn the free energy from snapshots.

Importable core for the sixth differentiable concept.  So far we have
recovered scalar PARAMETERS (D3) of a KNOWN free energy.  Here we recover
the free-energy FUNCTIONAL itself: given a time series of morphology
snapshots, learn the bulk energy f(c) that produced them.  This is the
Wodo-Ganapathysubramanian "learn the thermodynamics from the morphology"
programme, at tutorial scale.

We parametrise the derivative of the bulk energy on a monomial basis

    f'(c) = sum_i a_i c^{p_i} ,     powers p = [3, 1],

so df'/da_i = c^{p_i} feeds the adjoint directly.  The ground truth is the
canonical double well f'(c) = c^3 - c, i.e. a = [1, -1].  A constant term
in f' is intentionally excluded: it only shifts the chemical potential by
a constant and leaves the conserved c-dynamics invariant, so it is
UNIDENTIFIABLE from composition data - a lesson about what a morphology
can and cannot tell you.

Objective: L2 misfit between the simulated and the recorded snapshots
over a chosen subset of steps.  The gradient dL/da_i comes from the same
CHAdjoint sweep as every other concept; the fit is by L-BFGS.  Two
results: (i) with clean data the truth is recovered to ~1e-6; (ii) with
noisy data, using MORE snapshots sharply improves the recovery.
"""
import numpy as np
from scipy.optimize import minimize

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.adjoint.phasefield import CHForward, CHAdjoint, PolyBasisEnergy

POWERS = [3, 1]                 # f'(c) = a0 c^3 + a1 c
TRUTH = np.array([1.0, -1.0])   # the double well f'(c) = c^3 - c


def build_dm(level=3, device="cuda:0"):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    return dm, mesh


def _ic(coords):
    """Symmetric small-amplitude field (poly double well: c in [-1,1], no
    log walls, so the whole optimisation stays robust)."""
    return 0.4 * np.cos(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])


def gen(dm, coords, a, M, kappa, dt, nst, order):
    """A recorded CH march with bulk energy f'(c) = sum a_i c^{p_i}."""
    f = CHForward(dm, PolyBasisEnergy(POWERS, a), M=M, kappa=kappa, dt=dt,
                  order=order)
    f.set_initial(_ic(coords))
    f.run(nst)
    return f


def loss_and_grad(dm, coords, a, snaps, data, M, kappa, dt, nst, order):
    """Trajectory misfit L = sum_{n in snaps} 1/2||c_n - data_n||^2 and its
    adjoint gradient dL/da (one march + one reverse sweep)."""
    nn = dm.n_nodes
    f = gen(dm, coords, a, M, kappa, dt, nst, order)
    L = 0.0
    dJ = [np.zeros(2 * nn) for _ in range(nst)]
    for n in snaps:
        r = f.steps[n]["c"] - data[n]
        L += 0.5 * float(r @ r)
        dJ[n][0::2] = r
    g = CHAdjoint(f).gradient(dJ, [f"a{i}" for i in range(len(POWERS))])
    return L, np.array([g[f"a{i}"] for i in range(len(POWERS))])


def learn(level=3, M=1.0, kappa=0.005, dt=0.005, nst=10, order=2,
          a0=(0.5, -0.5), noise=0.01, Ks=(1, 2, 3, 4, 6, 10), seed=0,
          nseeds=1, device="cuda:0"):
    """Generate a truth trajectory, recover the energy coefficients from
    clean data, and measure how the noisy recovery improves with the
    number of snapshots K.  With nseeds>1 the noisy-recovery error at each
    K is averaged over that many noise realisations (a smooth statistical
    trend for the figure; run.py uses nseeds=1 for a deterministic gate).
    Returns everything the tutorial cites."""
    dm, mesh = build_dm(level, device)
    coords = mesh.node_coords
    truth_fwd = gen(dm, coords, TRUTH, M, kappa, dt, nst, order)
    data = [truth_fwd.steps[n]["c"].copy() for n in range(nst)]
    a0 = np.asarray(a0, float)

    def fit(snaps, dat):
        res = minimize(
            lambda a: loss_and_grad(dm, coords, a, snaps, dat, M, kappa,
                                    dt, nst, order),
            a0, jac=True, method="L-BFGS-B",
            options=dict(maxiter=200, ftol=1e-16, gtol=1e-14))
        return res

    # (i) clean recovery on the full trajectory
    all_snaps = list(range(nst))
    clean = fit(all_snaps, data)
    a_clean = clean.x
    err_clean = float(np.abs(a_clean - TRUTH).max())

    # (ii) noisy recovery improves with more snapshots (averaged over seeds)
    errsK, errsK_std = {}, {}
    for K in Ks:
        s = sorted(set(np.linspace(0, nst - 1, K).astype(int)))
        errs = []
        for sd in range(seed, seed + nseeds):
            rng = np.random.default_rng(sd)
            noisy = [data[n] + noise * rng.standard_normal(dm.n_nodes)
                     for n in range(nst)]
            r = fit(s, noisy)
            errs.append(float(np.abs(r.x - TRUTH).max()))
        errsK[K] = float(np.mean(errs))
        errsK_std[K] = float(np.std(errs))

    # the recovered f'(c) curve for the figure
    cc = np.linspace(-0.6, 0.6, 121)
    fp_truth = TRUTH[0] * cc ** 3 + TRUTH[1] * cc
    fp_rec = a_clean[0] * cc ** 3 + a_clean[1] * cc
    return dict(truth=TRUTH, a_clean=a_clean, err_clean=err_clean,
                loss_clean=float(clean.fun), nst=nst, noise=noise,
                Ks=list(Ks), errsK=errsK, errsK_std=errsK_std, cc=cc,
                fp_truth=fp_truth, fp_rec=fp_rec, nseeds=nseeds,
                side=int(round(np.sqrt(dm.n_nodes))))


def fd_check(level=3, M=1.0, kappa=0.005, dt=0.005, nst=10, order=2,
             a_at=(0.7, -1.3), device="cuda:0"):
    """FD-verify the trajectory-loss gradient at an off-truth point (the
    self-check the fit relies on)."""
    dm, mesh = build_dm(level, device)
    coords = mesh.node_coords
    truth_fwd = gen(dm, coords, TRUTH, M, kappa, dt, nst, order)
    data = [truth_fwd.steps[n]["c"].copy() for n in range(nst)]
    snaps = list(range(nst))
    a_at = np.asarray(a_at, float)
    _, g_adj = loss_and_grad(dm, coords, a_at, snaps, data, M, kappa, dt,
                             nst, order)
    g_fd = np.zeros_like(a_at)
    for i in range(len(a_at)):
        e = 1e-6
        hi = a_at.copy(); hi[i] += e
        lo = a_at.copy(); lo[i] -= e
        Lh, _ = loss_and_grad(dm, coords, hi, snaps, data, M, kappa, dt,
                              nst, order)
        Ll, _ = loss_and_grad(dm, coords, lo, snaps, data, M, kappa, dt,
                              nst, order)
        g_fd[i] = (Lh - Ll) / (2 * e)
    rel = np.abs(g_adj - g_fd) / np.maximum(np.abs(g_fd), 1e-14)
    return g_adj, g_fd, rel
