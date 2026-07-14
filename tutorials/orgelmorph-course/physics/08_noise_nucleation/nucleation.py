"""OrgElMorph course - Physics P8: thermal noise and NUCLEATION.

Importable core.  In P6 a crystal grew from a SEED we placed by hand.
Where do the seeds come from?  From thermal FLUCTUATIONS.  An
undercooled melt (T < Tm) is METASTABLE: psi = 0 sits behind a free-
energy barrier, and the crystal cannot form until a fluctuation pushes a
small region OVER that barrier -- classical nucleation.  This concept
turns on the physical noise (the fluctuation-dissipation theorem, FDT)
and shows nucleation happening, plus a noise-amplitude sweep.

THE PHYSICAL SEED.  In P1 the random seed was a NUMERICAL convenience
(a fixed RNG seed for reproducibility, statistics seed-independent).
HERE the noise is PHYSICS: the FDT noise amplitude is kB*T, and its
strength sets the nucleation RATE.  The stochastic Allen-Cahn equation

    dpsi/dt = -L_psi [ df/dpsi - eps2 lap psi ] + xi,
    <xi(x,t) xi(x',t')> = 2 L_psi kB*T delta(x-x') delta(t-t'),

adds a Gaussian, FDT-normalized, load-only forcing xi (noise_psi^2 =
kB*T in the nondimensional units; multiphase docstring).  The p1 bulk
free energy has a genuine nucleation barrier at psi = 0 (df/dpsi =
phi^2/N * dsig > 0 there), so with NO noise psi stays 0 forever; with
noise, rare fluctuations climb the barrier, form supercritical nuclei,
and those grow.

WHAT THE SWEEP SHOWS.  Nucleation is a rare-event, barrier-crossing
process: the rate scales like exp(-DeltaF*/kB*T), so a small increase in
noise amplitude produces a large increase in how much crystal has formed
by a fixed time.  We sweep noise_psi and report the final crystalline
fraction X and the number of distinct grains -- a threshold-like onset.
NOTE: BDF1 only (the brick asserts noise off under BDF2 -- the weak
order of the stochastic step is a recorded scope limit).
"""
import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.multiphase import MultiPhaseStepper, grain_labels


def build_mesh_dm(level=5, p=1, device="cuda:0"):
    # level 5 (32x32) keeps the 4-amplitude noise sweep to a few minutes
    # (the FDT noise re-plans the direct solver each iterate); level 6 is
    # available (slower).
    tree = build_uniform(level, dim=2, periodic=(True, True))
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2), device)
    return dm, mesh, cons


def _grid_map(mesh):
    fc = mesh.node_coords
    xs = np.unique(np.round(fc[:, 0], 12))
    ys = np.unique(np.round(fc[:, 1], 12))
    ix = np.searchsorted(xs, np.round(fc[:, 0], 12))
    iy = np.searchsorted(ys, np.round(fc[:, 1], 12))
    return len(xs), len(ys), ix, iy


def _stepper(dm, noise_psi, seed, T=0.5, dh=-1.5, dsig=1.0, Tm=1.0,
             eps2=6e-4, L_psi=4.0, dt=1.5e-3, phi_chi=0.5):
    chi_aa = np.array([[0.0, phi_chi], [phi_chi, 0.0]])
    return MultiPhaseStepper(
        dm, M=1, K=1, chi_aa=chi_aa, chi_ac=chi_aa.copy(),
        chi_ca=chi_aa.copy(), N=[1.0, 1.0], onsager=[[0.05]],
        kappa=[2e-4], dsig=[dsig], dh=[dh], Tm=[Tm], eps2=[eps2],
        L_psi=[L_psi], alpha_th=[0.0], beta_th=[0.0], L_th=[5.0],
        T=T, dt=dt, bulk="p1", tstep="bdf1",
        noise_psi=noise_psi, noise_seed=seed,
        newton_tol=1e-7, newton_max=40, clip_psi=True,
        linsolver="cudss")


def run_one(dm, mesh, cons, noise_psi, seed=17, phi0=0.7, t_end=1.2,
            dt_max=6e-3, device="cuda:0", nchecks=30):
    """One undercooled melt with FDT noise amplitude noise_psi (psi
    starts at 0 -- NO seed).  Returns crystalline fraction X(t), the
    final psi grid, and the grain count."""
    st = _stepper(dm, noise_psi, seed)
    st.set_initial([lambda x: np.full(len(x), phi0)],
                   [lambda x: np.zeros(len(x))],
                   [lambda x: np.zeros(len(x))])
    nx, ny, ix, iy = _grid_map(mesh)

    def frac():
        return float(np.mean(np.asarray(cons.T @ st.psi(0)) > 0.5))

    ts, X = [0.0], [frac()]
    checks = np.linspace(t_end / nchecks, t_end, nchecks)
    for tc in checks:
        st.march(t_end=tc, dt_max=dt_max, max_steps=3000, dt_min=1e-8)
        ts.append(st.t); X.append(frac())
    psi_full = np.asarray(cons.T @ st.psi(0))
    th_full = np.asarray(cons.T @ st.theta(0))
    psf = np.zeros((nx, ny)); psf[ix, iy] = psi_full
    labels, sizes = grain_labels(mesh.node_coords, psi_full, th_full,
                                 periodic=True)
    # induction time: first checkpoint where X exceeds 1% of the box
    ind = next((t for t, x in zip(ts, X) if x > 0.01), float("nan"))
    return dict(t=np.array(ts), X=np.array(X), psi=psf,
                noise_psi=noise_psi, X_end=float(X[-1]),
                n_grains=int((sizes > 5).sum()), induction=ind,
                psi_max=float(psi_full.max()), nx=nx, ny=ny)


def sweep(dm, mesh, cons, noise_levels=(0.0, 0.03, 0.06, 0.12),
          seed=17, device="cuda:0", **kw):
    """Noise-amplitude sweep on the SAME undercooled melt.  Returns the
    per-level results and the (noise -> X_end) onset curve."""
    res = [run_one(dm, mesh, cons, nl, seed=seed, device=device, **kw)
           for nl in noise_levels]
    return dict(levels=list(noise_levels), runs=res,
                X_end=[r["X_end"] for r in res],
                n_grains=[r["n_grains"] for r in res])
