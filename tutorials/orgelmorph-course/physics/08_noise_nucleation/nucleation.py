"""OrgElMorph course - Physics P8: thermal noise and NUCLEATION.

Importable core.  In P6 a crystal grew from a SEED we placed by hand.
Where do the seeds come from?  From thermal FLUCTUATIONS.  An
undercooled melt (T < Tm) is METASTABLE: psi = 0 sits behind a free-
energy barrier, and the crystal cannot form until a fluctuation pushes a
small region OVER that barrier -- classical nucleation.

THE DISCRETE FDT NORMALIZATION (derived, then VERIFIED).  The stochastic
Allen-Cahn equation is

    dpsi/dt = -L_psi [ df/dpsi - eps2 lap psi ] + xi,
    <xi(x,t) xi(x',t')> = 2 L_psi kB*T delta(x-x') delta(t-t').

On the finite-element mesh the load-only forcing is drawn per Gauss point
with standard deviation

    std(xi_gp) = noise_psi * sqrt( 2 L_psi / (dt * wJ_gp) ),     (*)

where wJ_gp is the Gauss-point quadrature weight (volume element) and
noise_psi^2 = kB*T in the nondimensional units.  The two factors are NOT
cosmetic:

  * the 1/dt makes the TIME white-noise increment consistent (a Wiener
    increment over dt has variance proportional to dt, and the implicit
    load xi*dt then has the right O(dt) variance);
  * the 1/wJ_gp is the SPATIAL white-noise quadrature normalization: the
    assembled nodal force b_a = INT N_a xi dV has variance
        Var(b_a) = 2 noise_psi^2 L_psi INT N_a^2 dV / dt              (**)
    EXACTLY -- independent of how finely the mesh samples space.  A naive
    (mesh-blind) noise would make (**) scale with the element size and the
    equilibrium fluctuations would NOT converge under refinement.

VERIFICATION (central, not a footnote).  fdt_well fixes the physical
noise kB*T, puts psi in a STABLE quadratic well (a superheated r14 well,
clip-free, so fluctuations are two-sided and un-rectified), and measures
the equilibrium spatial variance <Var(psi)>.  Equipartition predicts a
value set by kB*T and the well/mesh, INDEPENDENT of dt.  Refining dt the
measured variance converges to a dt-independent plateau -- the physical
statistic converges -- confirming the (*)/(**) normalization.  (Refining
the mesh, the nodal variance scales as 1/V_cell as equipartition says.)

ENSEMBLES.  Nucleation is a random, rare-event process, so a single seed
is one sample, not a result.  run_ensemble runs many noise realizations
per amplitude and reports, via diffsim.diagnostics.stochastic, the
nucleation PROBABILITY, the induction-time distribution (first-passage
detection), the nuclei DENSITY, and the crystalline-fraction distribution,
each as a mean with a 95% interval.

NOISE-AMPLITUDE (kB*T calibration) SWEEP -- NOT a temperature sweep.  We
vary ONLY noise_psi (= sqrt(kB*T) in the FDT forcing) at FIXED undercooling
T/Tm.  A genuine temperature sweep would ALSO move the driving force and
the barrier (drive = dh(1 - T/Tm), the entropy, the mobility); this sweep
deliberately isolates the fluctuation amplitude, so read it as a
noise-amplitude / FDT-calibration sensitivity, not "raising the
temperature".

CLIPPING.  The undercooled (p1) runs clip psi to [0, 1]; we QUANTIFY the
clipping (the saturated psi~1 fraction, and note the psi=0 floor rectifies
sub-barrier fluctuations -- exactly why the quantitative FDT test uses the
clip-free r14 well).  BDF1 only (the brick asserts noise off under BDF2 --
the stochastic weak order under BDF2 is a recorded scope limit).
"""
import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.multiphase import MultiPhaseStepper, grain_labels
from diffsim.diagnostics import stochastic as _st, admissibility as _adm


def build_mesh_dm(level=5, p=1, device="cuda:0"):
    # level 5 (32x32) keeps the ensemble sweep to a few minutes (the FDT
    # noise re-plans the direct solver each iterate); level 6 is available
    # (slower).
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


def _cell_volume(mesh):
    """Uniform-mesh cell volume h^dim (for the equipartition scaling)."""
    h = float(mesh.tree.h()[0])
    return h ** mesh.dim


# ---------------------------------------------------------------------
# nucleation stepper (undercooled p1, clip on)
# ---------------------------------------------------------------------
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


def run_one(dm, mesh, cons, noise_psi, seed=17, phi0=0.7, t_end=1.0,
            dt_max=6e-3, x_detect=0.02, device="cuda:0", nchecks=30):
    """One undercooled melt with FDT noise amplitude noise_psi (psi starts
    at 0 -- NO seed).  Returns X(t), the final psi grid, the grain count,
    a first-passage induction time (interpolated X = x_detect crossing),
    and the clipping (saturated psi~1 fraction)."""
    st = _stepper(dm, noise_psi, seed)
    st.set_initial([lambda x: np.full(len(x), phi0)],
                   [lambda x: np.zeros(len(x))],
                   [lambda x: np.zeros(len(x))])
    nx, ny, ix, iy = _grid_map(mesh)

    def frac():
        return float(np.mean(np.asarray(cons.T @ st.psi(0)) > 0.5))

    ts, X, sat = [0.0], [frac()], [0.0]
    checks = np.linspace(t_end / nchecks, t_end, nchecks)
    for tc in checks:
        st.march(t_end=tc, dt_max=dt_max, max_steps=3000, dt_min=1e-8)
        ts.append(st.t); X.append(frac())
        p = np.asarray(cons.T @ st.psi(0))
        sat.append(float(np.mean(p > 0.99)))   # upper-clip saturation
    ts = np.array(ts); X = np.array(X)
    psi_full = np.asarray(cons.T @ st.psi(0))
    th_full = np.asarray(cons.T @ st.theta(0))
    psf = np.zeros((nx, ny)); psf[ix, iy] = psi_full
    labels, sizes = grain_labels(mesh.node_coords, psi_full, th_full,
                                 periodic=True)
    # induction time by first-passage (interpolated crossing of x_detect)
    ind = _st.first_passage(X, x_detect, times=ts, direction="up")
    return dict(t=ts, X=X, psi=psf, noise_psi=noise_psi,
                X_end=float(X[-1]), n_grains=int((sizes > 5).sum()),
                induction=(float("nan") if ind is None else float(ind)),
                nucleated=bool(X[-1] > x_detect),
                psi_max=float(psi_full.max()),
                sat_frac=float(sat[-1]), nx=nx, ny=ny)


# ---------------------------------------------------------------------
# ensembles (many noise realizations per amplitude)
# ---------------------------------------------------------------------
def run_ensemble(dm, mesh, cons, noise_psi, seeds, device="cuda:0", **kw):
    """Ensemble of noise realizations at one amplitude.  Aggregates, via
    diffsim.diagnostics.stochastic, the nucleation probability, the
    induction-time distribution, the nuclei density and the crystalline-
    fraction distribution (mean + 95% interval)."""
    runs = [run_one(dm, mesh, cons, noise_psi, seed=s, device=device, **kw)
            for s in seeds]
    X_end = np.array([r["X_end"] for r in runs])
    grains = np.array([r["n_grains"] for r in runs])
    nucleated = np.array([r["nucleated"] for r in runs])
    inds = np.array([r["induction"] for r in runs], float)
    area = 1.0                                   # unit square
    prob = _st.event_probability(nucleated)
    Xagg = _st.ensemble_aggregate(X_end)
    gagg = _st.ensemble_aggregate(grains / area)
    valid_ind = inds[np.isfinite(inds)]
    ind_ci = (_st.bootstrap_ci(valid_ind) if valid_ind.size >= 2
              else {"estimate": float(valid_ind[0]) if valid_ind.size
                    else float("nan"), "low": float("nan"),
                    "high": float("nan")})
    return dict(noise_psi=noise_psi, runs=runs, n=len(seeds),
                nucleation_prob=prob,
                X_mean=float(Xagg["mean"]), X_sd=float(Xagg["sd"]),
                X_sem=float(Xagg["sem"]),
                nuclei_density_mean=float(gagg["mean"]),
                nuclei_density_sd=float(gagg["sd"]),
                induction=ind_ci,
                sat_frac_mean=float(np.mean([r["sat_frac"] for r in runs])))


def sweep(dm, mesh, cons, noise_levels=(0.0, 0.03, 0.06),
          seeds=(11, 17, 23, 29), device="cuda:0", **kw):
    """Noise-amplitude (kB*T calibration) ENSEMBLE sweep on the SAME
    undercooled melt.  Zero noise is deterministic (one run); each nonzero
    amplitude runs the full seed ensemble.  Returns the per-amplitude
    aggregates and the onset curve."""
    ens = []
    for nl in noise_levels:
        if nl == 0.0:
            r0 = run_one(dm, mesh, cons, 0.0, seed=seeds[0], device=device,
                         **kw)
            ens.append(dict(noise_psi=0.0, runs=[r0], n=1,
                            nucleation_prob={"p": float(r0["nucleated"]),
                                             "low": 0.0, "high": 0.0},
                            X_mean=r0["X_end"], X_sd=0.0, X_sem=0.0,
                            nuclei_density_mean=float(r0["n_grains"]),
                            nuclei_density_sd=0.0,
                            induction={"estimate": r0["induction"],
                                       "low": float("nan"),
                                       "high": float("nan")},
                            sat_frac_mean=r0["sat_frac"]))
        else:
            ens.append(run_ensemble(dm, mesh, cons, nl, seeds,
                                    device=device, **kw))
    return dict(levels=list(noise_levels), ensembles=ens,
                X_mean=[e["X_mean"] for e in ens],
                X_sd=[e["X_sd"] for e in ens],
                prob=[e["nucleation_prob"]["p"] for e in ens],
                density=[e["nuclei_density_mean"] for e in ens])


# ---------------------------------------------------------------------
# the central FDT verification: equilibrium variance in a quadratic well
# ---------------------------------------------------------------------
def _well_mesh(level, device="cpu"):
    """A small periodic mesh for the FDT well.  The verification is cheap
    and mesh-independent by design, so it runs on a COARSE mesh with the
    CPU direct solver (splu) -- unbeatable at this size and, unlike the
    GPU cuDSS path, it does not re-plan the factorization when the noise
    grows the sparsity pattern, keeping the dt sweep fast."""
    return build_mesh_dm(level, device=device)


def _well_stepper(dm, noise_psi, seed, dt, T=3.0, dh=1.5, dsig=1.0,
                  Tm=1.0, eps2=6e-4, L_psi=4.0, phi_chi=0.5,
                  solver="splu"):
    """A STABLE quadratic well for psi: superheated r14 (drive > 0 makes
    psi = 0 a genuine minimum, no constant offset) with clip_psi=False so
    fluctuations are two-sided and UN-rectified -- the clean setting for an
    equipartition/FDT measurement."""
    chi_aa = np.array([[0.0, phi_chi], [phi_chi, 0.0]])
    return MultiPhaseStepper(
        dm, M=1, K=1, chi_aa=chi_aa, chi_ac=chi_aa.copy(),
        chi_ca=chi_aa.copy(), N=[1.0, 1.0], onsager=[[0.05]],
        kappa=[2e-4], dsig=[dsig], dh=[dh], Tm=[Tm], eps2=[eps2],
        L_psi=[L_psi], alpha_th=[0.0], beta_th=[0.0], L_th=[5.0],
        T=T, dt=dt, bulk="r14", tstep="bdf1",
        noise_psi=noise_psi, noise_seed=seed,
        newton_tol=1e-7, newton_max=40, clip_psi=False,
        linsolver=solver)


def fdt_well(dm, mesh, cons, noise_psi, dt, seed=3, t_burn=0.12,
             t_win=0.02, n_sample=8, phi0=0.7, solver="splu"):
    """Fix the physical noise, relax psi to its stationary state in a
    stable quadratic well, then measure the equilibrium spatial variance
    <Var(psi)> over several sampling windows.  The measured variance is the
    physical statistic that must converge (dt-independently) if the (*)
    FDT normalization is correct."""
    st = _well_stepper(dm, noise_psi, seed, dt, solver=solver)
    st.set_initial([lambda x: np.full(len(x), phi0)],
                   [lambda x: np.zeros(len(x))],
                   [lambda x: np.zeros(len(x))])
    st.march(t_end=t_burn, dt_max=dt, max_steps=40000, dt_min=1e-9)
    vs, ms = [], []
    for k in range(n_sample):
        st.march(t_end=t_burn + t_win * (k + 1), dt_max=dt,
                 max_steps=40000, dt_min=1e-9)
        p = np.asarray(cons.T @ st.psi(0))
        vs.append(float(p.var())); ms.append(float(p.mean()))
    return dict(noise_psi=noise_psi, dt=dt, var=float(np.mean(vs)),
                var_sd=float(np.std(vs, ddof=1)), mean=float(np.mean(ms)),
                cell_volume=_cell_volume(mesh))


def verify_fdt(noise_psi=0.03, dts=(3e-3, 1.5e-3, 7.5e-4), level=4,
               mesh_levels=(3, 4), device="cpu"):
    """The CENTRAL FDT verification.

    (1) dt-independence: at fixed physical noise, sweep dt; the equilibrium
        variance must converge to a dt-INDEPENDENT plateau (a wrong 1/dt
        normalization would make it scale with dt).  Reported as the
        coefficient of variation of Var(psi) across dt (small == verified).
    (2) mesh scaling: equipartition predicts Var(psi_node) ~ kB*T /
        (a * V_cell), i.e. Var * V_cell is mesh-INDEPENDENT; we check the
        product across two mesh levels.

    Runs on a coarse CPU/splu mesh (see _well_mesh)."""
    dm, mesh, cons = _well_mesh(level, device=device)
    dt_runs = [fdt_well(dm, mesh, cons, noise_psi, dt) for dt in dts]
    var = np.array([r["var"] for r in dt_runs])
    dt_cov = float(np.std(var, ddof=1) / np.mean(var)) if var.mean() else \
        float("nan")
    mesh_runs = []
    for lv in mesh_levels:
        dmk, meshk, consk = _well_mesh(lv, device=device)
        r = fdt_well(dmk, meshk, consk, noise_psi, dts[0])
        r["level"] = lv
        r["var_times_vcell"] = r["var"] * r["cell_volume"]
        mesh_runs.append(r)
    vv = np.array([r["var_times_vcell"] for r in mesh_runs])
    mesh_cov = float(np.std(vv, ddof=1) / np.mean(vv)) if vv.mean() else \
        float("nan")
    return dict(noise_psi=noise_psi, dts=list(dts), dt_runs=dt_runs,
                var=var.tolist(), var_mean=float(var.mean()),
                dt_cov=dt_cov, mesh_runs=mesh_runs, mesh_levels=list(mesh_levels),
                var_times_vcell=vv.tolist(), mesh_cov=mesh_cov,
                cell_volume=_cell_volume(mesh))
