"""OrgElMorph course - Physics P1: binary Cahn-Hilliard, two free energies.

Importable core for the tutorial: run a binary spinodal-decomposition
march with either the polynomial double-well or the Flory-Huggins
(logarithmic) bulk free energy, and measure the Ginzburg-Landau energy
budget (total = bulk + interfacial) at every step.

This is the object the student reads FIRST, then run.py drives it and
gen_figures.py renders the figures the LaTeX shows.  Nothing here is
tutorial-only: `CahnHilliardStepper` is the same production brick the
research code uses (src/diffsim/physics/cahn_hilliard.py) - the reason
the interface below is small is that the physics lives in the brick,
not in a toy re-implementation.

Free energy (Ginzburg-Landau):  F[c] = INT [ f(c) + (kap/2)|grad c|^2 ] dV
  bulk        f(c)                       - the homogeneous mixing energy
  interfacial (kap/2)|grad c|^2          - the gradient penalty on
                                           composition variation (sets
                                           the interface width ~ sqrt(kap))

  poly:  f = (1/4)(c^2 - 1)^2,  c in [-1, 1], symmetric double well.
  fh:    f = A[c ln c + (1-c) ln(1-c)] + B c(1-c),  c in (0, 1);
         A = entropic scale (~1/N), B = enthalpic (~chi).  Spinodal
         where the entropic curvature A(1/c + 1/(1-c)) drops below 2B.

Conserved (Model-B) dynamics: c_t = div(M grad mu), mu = dF/dc, so F is
a Lyapunov functional - it can only DECREASE (the numbers prove it).
"""
import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.cahn_hilliard import CahnHilliardStepper
from diffsim.diagnostics import morphology as dmorph


# --- bulk-energy curvature f''(c), and the two length scales it sets ----
def fpp(c_bar, energy, A, B):
    """Homogeneous free-energy curvature f''(c_bar).

    poly: f''=3c^2-1.  fh: f''=A(1/c+1/(1-c))-2B.  A negative value marks a
    spinodally unstable mean composition (fluctuations grow)."""
    if energy == "fh":
        return A * (1.0 / c_bar + 1.0 / (1.0 - c_bar)) - 2.0 * B
    return 3.0 * c_bar ** 2 - 1.0


def interface_width(kappa, energy, c_bar, A, B, well=1.0):
    """Equilibrium interface width ell ~ sqrt(kappa / W), NOT sqrt(kappa).

    ``W`` is the bulk-energy scale (well depth / barrier), so the width has the
    right units and shrinks when the quench is deeper. For the symmetric
    polynomial f=(W/4)(c^2-1)^2 the exact tanh profile gives ell=sqrt(2 kappa/W)
    with W=1 here. For a general f we report the spinodal length
    sqrt(kappa/|f''|) (the linear-instability length balancing gradient vs bulk
    curvature). Units: same as the domain (length), NOT cells."""
    if energy == "poly":
        return float(np.sqrt(2.0 * kappa / well))
    return float(np.sqrt(kappa / abs(fpp(c_bar, energy, A, B))))


def linear_dispersion(c_bar, kappa, M, energy, A, B, k=None):
    """Cahn-Hilliard linear dispersion sigma(k) about the mean c_bar.

    Linearizing c_t = M div grad(f'(c) - kappa lap c) about a uniform c_bar
    gives, for a Fourier mode ~exp(i k.x + sigma t),

        sigma(k) = -M k^2 ( f''(c_bar) + kappa k^2 ).

    When f''<0 a band 0<k<k_c=sqrt(-f''/kappa) is unstable; the FASTEST mode is
    k* = sqrt(-f''/(2 kappa)), wavelength lambda* = 2 pi / k*. This sets the
    initial modulation the spinodal picks out (compared below against S(q))."""
    fp2 = fpp(c_bar, energy, A, B)
    if k is None:
        kc = np.sqrt(max(-fp2, 0.0) / kappa) if fp2 < 0 else np.sqrt(1.0 / kappa)
        k = np.linspace(1e-6, 1.6 * max(kc, 1.0), 400)
    sigma = -M * k ** 2 * (fp2 + kappa * k ** 2)
    k_star = float(np.sqrt(-fp2 / (2.0 * kappa))) if fp2 < 0 else float("nan")
    lam_star = float(2.0 * np.pi / k_star) if k_star == k_star and k_star > 0 \
        else float("nan")
    k_c = float(np.sqrt(-fp2 / kappa)) if fp2 < 0 else float("nan")
    return {"k": k, "sigma": sigma, "fpp": float(fp2),
            "k_star": k_star, "lambda_star": lam_star, "k_c": k_c}


def fit_coarsening_exponent(t, L, t_min=None):
    """Fit L(t) ~ t^n on the coarsening window; return (n, sigma_n, R2).

    Least-squares slope of log L vs log t with its standard error (a real
    uncertainty, spec P1), restricted to t>=t_min (skip the early linear-growth
    transient where no domains exist yet). Diffusive coarsening of a conserved
    order parameter is Lifshitz-Slyozov n=1/3; short horizons often measure less
    and we report the honest fitted value + interval, not the textbook number."""
    t = np.asarray(t, dtype=float)
    L = np.asarray(L, dtype=float)
    ok = np.isfinite(t) & np.isfinite(L) & (t > 0) & (L > 0)
    if t_min is not None:
        ok &= t >= t_min
    t, L = t[ok], L[ok]
    if t.size < 3:
        return float("nan"), float("nan"), float("nan")
    x, y = np.log(t), np.log(L)
    n, b = np.polyfit(x, y, 1)
    yhat = n * x + b
    resid = y - yhat
    dof = max(t.size - 2, 1)
    s2 = float(np.sum(resid ** 2) / dof)
    sxx = float(np.sum((x - x.mean()) ** 2))
    sigma_n = float(np.sqrt(s2 / sxx)) if sxx > 0 else float("nan")
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    return float(n), sigma_n, r2


def _lengthscales(snap, dx):
    """Three length definitions of a nodal field snapshot from the shared
    morphology diagnostics (spec P1: >=2 length defs). Drops the duplicated
    no-flux boundary row/col so the FFT sees the 2^level unique cells at
    spacing dx (units: length):
      L_peak  = 2pi/q_peak         (S(q) peak; quantization-limited on coarse grids)
      L_fm    = 2pi/<q>            (first moment; biased by the sharp-interface tail)
      L_area  = A_domain / P_int   (total area / interfacial length; the robust,
                                    monotone coarsening measure -- interface
                                    shrinks as ~1/L, so this grows cleanly)."""
    f = np.asarray(snap)[:-1, :-1]
    area = f.shape[0] * f.shape[1] * dx * dx
    perim = dmorph.interfacial_area(f, dx=dx)      # total interface length (2-D)
    L_area = float(area / perim) if perim > 0 else float("nan")
    return (dmorph.peak_wavelength(f, dx=dx),
            dmorph.first_moment_wavelength(f, dx=dx),
            L_area)


def linear_probe(energy="poly", level=7, dt=2e-4, n_steps=10,
                 M=1.0, kappa=5e-4, fh_A=1.0, fh_B=2.5, c_avg=None,
                 amp=0.02, seed=3, device="cuda:0", linsolver="splu"):
    """Verify the linear dispersion relation against a MEASURED growth
    spectrum (spec P1: dispersion -> fastest k -> vs measured S(q)).

    The quench is extremely fast (sigma* ~ M f''^2/(4 kappa) ~ 500), so at the
    production dt the linear window is under one step. Here we march a FEW tiny
    steps from the small-amplitude IC and read the growth of each Fourier shell,

        sigma_meas(k) = (1/2) ln[ S(k, tau) / S(k, 0) ] / tau,

    (S ~ amplitude^2, hence the 1/2). While the perturbation stays small this
    must track the analytic sigma(k) = -M k^2 (f'' + kappa k^2), and its peak
    must land on the predicted fastest mode k*. Returns the paired spectra and
    the measured vs predicted fastest wavelength."""
    if c_avg is None:
        c_avg = 0.0 if energy == "poly" else 0.5
    dm, mesh, cons = build_mesh_dm(level, p=1, device=device)
    st = CahnHilliardStepper(dm, M, kappa, dt, order=1, energy=energy,
                             fh_A=fh_A, fh_B=fh_B, linsolver=linsolver)
    rng = np.random.default_rng(seed)
    st.set_initial(lambda x: c_avg + amp * rng.standard_normal(len(x)),
                   mu_init="consistent")
    side = int(round(np.sqrt(len(mesh.node_coords))))
    dx = 1.0 / (side - 1)
    full = lambda: np.asarray(cons.T @ st.x[0::2]).reshape(side, side)[:-1, :-1]
    q0, S0 = dmorph.structure_factor(full() - c_avg, dx=dx)
    for _ in range(n_steps):
        st.step()
    q1, S1 = dmorph.structure_factor(full() - c_avg, dx=dx)
    tau = st.t
    # measured growth rate per shell where both spectra have signal
    ok = (S0 > 1e-14 * S0.max()) & (S1 > 0)
    sigma_meas = np.full_like(S0, np.nan)
    sigma_meas[ok] = 0.5 * np.log(S1[ok] / S0[ok]) / tau
    disp = linear_dispersion(c_avg, kappa, M, energy, fh_A, fh_B, k=q0)
    # measured fastest mode = argmax of the measured growth spectrum
    valid = ok & (q0 < disp["k_c"] if disp["k_c"] == disp["k_c"] else ok)
    if valid.any():
        k_meas = float(q0[valid][int(np.argmax(sigma_meas[valid]))])
        lam_meas = float(2 * np.pi / k_meas) if k_meas > 0 else float("nan")
    else:
        k_meas = lam_meas = float("nan")
    return {"q": q0, "sigma_meas": sigma_meas, "sigma_analytic": disp["sigma"],
            "tau": tau, "k_star": disp["k_star"], "lambda_star": disp["lambda_star"],
            "k_c": disp["k_c"], "k_meas": k_meas, "lambda_meas": lam_meas,
            "level": level, "dx": dx}


def build_mesh_dm(level=7, p=1, device="cuda:0"):
    """Uniform 2-D box, 2^level cells per side (level 7 = 128x128).
    Natural (no-flux) boundaries: grad c . n = grad mu . n = 0, so
    total mass is conserved and no composition leaves the box."""
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2), device)
    return dm, mesh, cons


# --- the Ginzburg-Landau energy budget, computed by quadrature --------
# We evaluate c and grad c at the Gauss points (the stepper exposes the
# same helper it uses internally) and integrate the two energy densities
# separately, so the student sees bulk and interfacial energy on the
# same axes.

def _bulk_density(c, energy, A, B):
    if energy == "fh":
        eps = 1e-4
        rl = lambda x: np.where(x < eps, np.log(eps) + (x - eps) / eps,
                                np.log(np.maximum(x, eps)))
        return A * (c * rl(c) + (1 - c) * rl(1 - c)) + B * c * (1 - c)
    return 0.25 * (c ** 2 - 1.0) ** 2


def energy_budget(st, dm, mesh, cf, energy, A, B):
    """Return (F_total, F_bulk, F_interface) for the free-field vector
    cf, by Gauss-point quadrature of the two energy densities."""
    v, g = st._gp_scalar(cf)               # c and grad c at Gauss pts
    F_bulk = F_int = 0.0
    for pv, b in dm.bins.items():
        h = mesh.tree.h()[mesh.bins[pv]]
        ne = len(mesh.conn_of[pv])
        wq = np.tile(dm.tables_by_p[pv].w, ne) \
            * np.repeat((h / 2.0) ** dm.dim, b["nqp"])
        F_bulk += float((wq * _bulk_density(v[pv], energy, A, B)).sum())
        F_int += float((wq * 0.5 * st.kappa * (g[pv] ** 2).sum(1)).sum())
    return F_bulk + F_int, F_bulk, F_int


def total_mass(st, dm, mesh, cf):
    v, _ = st._gp_scalar(cf)
    m = 0.0
    for pv, b in dm.bins.items():
        h = mesh.tree.h()[mesh.bins[pv]]
        ne = len(mesh.conn_of[pv])
        wq = np.tile(dm.tables_by_p[pv].w, ne) \
            * np.repeat((h / 2.0) ** dm.dim, b["nqp"])
        m += float((wq * v[pv]).sum())
    return m


def run_spinodal(energy="poly", level=6, steps=300, dt=0.02,
                 M=1.0, kappa=5e-4, fh_A=1.0, fh_B=2.5,
                 c_avg=None, amp=0.05, seed=3, device="cuda:0",
                 snap_steps=None, linsolver="splu", measure_every=0):
    """March a binary spinodal decomposition and record the energy
    budget each step.  Returns a dict with the time series + snapshots.

    c_avg default: 0.0 for poly (symmetric wells), 0.5 for fh (the
    symmetric point of the log entropy).  amp is the initial random
    perturbation that seeds the instability (small: the linear
    spinodal analysis governs the early growth)."""
    if c_avg is None:
        c_avg = 0.0 if energy == "poly" else 0.5
    if snap_steps is None:
        snap_steps = [0, steps // 10, steps // 3, steps]
    dm, mesh, cons = build_mesh_dm(level, p=1, device=device)
    st = CahnHilliardStepper(dm, M, kappa, dt, order=1, energy=energy,
                             fh_A=fh_A, fh_B=fh_B, linsolver=linsolver)
    rng = np.random.default_rng(seed)
    st.set_initial(lambda x: c_avg + amp * rng.standard_normal(len(x)),
                   mu_init="consistent")

    def budget(cf):
        return energy_budget(st, dm, mesh, cf, energy, fh_A, fh_B)

    full = lambda cf: np.asarray(cons.T @ cf)
    side = int(round(np.sqrt(len(mesh.node_coords))))
    dx = 1.0 / (side - 1)                     # unit box, 2^level cells
    m0 = total_mass(st, dm, mesh, st.hist[0])
    Ft, Fb, Fi = budget(st.hist[0])
    field0 = full(st.x[0::2]).reshape(side, side)
    rec = dict(t=[0.0], F_total=[Ft], F_bulk=[Fb], F_interface=[Fi],
               mass=[m0], snaps={}, side=side, dx=dx, energy=energy,
               kappa=kappa, c_avg=c_avg, seed=seed,
               cmin=[float(field0.min())], cmax=[float(field0.max())],
               Lt_t=[], Lt_peak=[], Lt_fm=[], Lt_area=[],
               params=dict(M=M, dt=dt, fh_A=fh_A, fh_B=fh_B, level=level))
    if 0 in snap_steps:
        rec["snaps"][0] = field0.copy()
    for n in range(1, steps + 1):
        c, mu = st.step()
        Ft, Fb, Fi = budget(c)
        rec["t"].append(st.t)
        rec["F_total"].append(Ft)
        rec["F_bulk"].append(Fb)
        rec["F_interface"].append(Fi)
        rec["mass"].append(total_mass(st, dm, mesh, c))
        field = full(c).reshape(side, side)
        rec["cmin"].append(float(field.min()))
        rec["cmax"].append(float(field.max()))
        if n in snap_steps:
            rec["snaps"][n] = field.copy()
        # length-scale time series (coarsening L(t)) — measured live so we
        # need not store every field. Two definitions bracket the scale.
        if measure_every and (n % measure_every == 0 or n == steps):
            lp, lf, la = _lengthscales(field, dx)
            rec["Lt_t"].append(st.t)
            rec["Lt_peak"].append(lp)
            rec["Lt_fm"].append(lf)
            rec["Lt_area"].append(la)
    for k in ("t", "F_total", "F_bulk", "F_interface", "mass",
              "cmin", "cmax", "Lt_t", "Lt_peak", "Lt_fm", "Lt_area"):
        rec[k] = np.asarray(rec[k])
    # admissibility-projection record (FH box-clip) + regularization eps
    rec["proj_dofs"] = int(st.proj_dofs)
    rec["proj_max"] = float(st.proj_max)
    rec["fh_reg_eps"] = 1e-4 if energy == "fh" else 0.0
    return rec
