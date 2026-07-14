"""OrgElMorph course - Physics P6: Allen-Cahn crystallization.

Importable core.  Phase separation (Cahn-Hilliard) is only half of
organic-film morphology; the other half is CRYSTALLIZATION -- a
component ordering from an amorphous melt into a crystal.  Crystallinity
is a NON-CONSERVED order parameter psi in [0,1] (crystal can be created,
unlike composition), so it obeys ALLEN-CAHN (non-conserved gradient
flow) rather than Cahn-Hilliard.  This concept isolates the
crystallization physics: a seeded crystal that GROWS below the melting
point and MELTS above it, the growth KINETICS (the Avrami / JMAK law),
the INTERFACE VELOCITY as a function of undercooling, the CRITICAL
RADIUS that separates growing from redissolving seeds, and ORIENTATION
markers that distinguish grains.

We use the (M, K)-generic brick with M=1 (one composition) and K=1 (one
crystallizable species): fields (phi, mu, psi, theta).  The
crystallization free energy (2310.11844 "r14" form, PCBM-class
energetics from materials.yaml) drives psi:

    dpsi/dt = -L_psi [ df/dpsi - eps2 lap psi ],
    f_cr = phi [ q(psi) dsig + p(psi) drive ],
    q(psi) = psi^2 (1-psi)^2  (double-well barrier),
    p(psi) = psi^2 (3-2psi)   (interpolation),
    drive  = dh (T/Tm - 1)    (Turnbull undercooling; NEGATIVE for
                               T < Tm, so psi -> 1 lowers f -> growth).

  * SIGN: T < Tm gives drive < 0 -> the crystal GROWS; T > Tm gives
    drive > 0 -> the crystal MELTS.  The melting point Tm is where the
    driving force changes sign.
  * KINETICS (Avrami/JMAK): with athermal nuclei placed at t=0 growing
    at a roughly constant interface velocity v, the crystalline area of
    each grain grows like (v t)^2 in 2-D before grains impinge, so the
    crystalline FRACTION follows X(t) = 1 - exp[-(k t)^n] with Avrami
    exponent n ~ 2 (2-D, pre-existing nuclei, interface-limited).  We
    fit n from the measured X(t) after removing the pre-placed seed
    baseline X0 (the generalized-JMAK conversion X* = (X-X0)/(1-X0)),
    and we report the fit window, R^2, and a 95% CI on n.
  * INTERFACE VELOCITY: below Tm a planar/curved crystal front advances
    at a roughly constant velocity v(dT).  run_interface_velocity
    measures v = d r_eff/dt from the crystal area and sweeps the
    undercooling dT = Tm - T.  With the CONSTANT mobility used here
    (no thermally-activated hop) v rises monotonically with dT -- the
    real thermal-transport maximum is a modelling extension (P6 exercise).
  * CRITICAL RADIUS: a curved crystal pays interface energy ~ 2 pi r
    sigma while gaining bulk energy ~ pi r^2 |drive|.  dF/dr = 0 at a
    critical radius r* ~ sigma / |drive|: seeds larger than r* grow,
    smaller ones redissolve (the Gibbs-Thomson floor).  Deeper
    undercooling raises |drive| and so LOWERS r*.  run_critical_radius
    brackets r* by seeding a range of radii and classifying grow/shrink.
  * ORIENTATION: each seed carries an orientation marker theta.  theta
    here is a FIXED grain LABEL, not an evolved field: the stepper runs
    in theta_mode="frozen" (alpha_th = beta_th = 0), so there is NO
    grain-boundary energy and NO theta dynamics.  grain_labels reads the
    frozen theta plateaus to separate crystals for counting.  The
    Kobayashi-Warren-Carter grain-boundary physics (a theta that
    evolves and resists impinging crystals) is the coupled concept P7.

ACCELERATED PEDAGOGICAL PARAMETERS.  The Avrami demo runs at T = 250 K
and a kinetic prefactor L_psi = 11 that are NOT the physical PCBM values;
they are accelerated so the crystalline fraction sweeps the full (0, 1)
range inside a short tutorial run (a physical-rate run pins X near 0 and
the double-log fit is meaningless).  The grow/melt and interface-velocity
demos likewise pick temperatures for a clear, fast signal.  What is
physical and hardware-independent is the SIGN of the driving force at Tm,
the SHAPE of the JMAK sigmoid, the MONOTONE v(dT) trend, and the
EXISTENCE of a critical radius that shrinks with undercooling.
"""
import os
import sys

import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.multiphase import MultiPhaseStepper, grain_labels
from diffsim.diagnostics import conservation as _cons

# PCBM-class crystallization energetics are READ from the materials database
# (materials.yaml crystallization.PCBM_class, the extended-Flory-Huggins "r14"
# form of Siber-Ronsin-Harting), never hand-copied.  Call
# ``CRYST.record()`` / ``save_resolved_materials(CRYST, ...)`` to archive the
# resolved values + provenance next to a run's results.
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "materials")))
from loader import load_crystallization                      # noqa: E402

CRYST = load_crystallization("PCBM_class")
_chi_aa, _chi_ca = CRYST.value("chi_aa"), CRYST.value("chi_ca")
CHI_AA = np.array([[0.0, _chi_aa], [_chi_aa, 0.0]])
CHI_CA = np.array([[0.0, _chi_ca], [0.0, 0.0]])
DSIG, DH, TM, EPS2 = (CRYST.value("dsig"), CRYST.value("dh"),
                      CRYST.value("Tm"), CRYST.value("eps2"))
CRYST_N = list(CRYST.value("N"))


def build_mesh_dm(level=5, p=1, device="cuda:0"):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2), device)
    return dm, mesh, cons


def _stepper(dm, T, eps2=EPS2, L_psi=5.0, dt=2e-3):
    return MultiPhaseStepper(
        dm, M=1, K=1, chi_aa=CHI_AA, chi_ac=CHI_CA.T.copy(),
        chi_ca=CHI_CA, N=CRYST_N, onsager=[[0.1]], kappa=[2e-4],
        dsig=[DSIG], dh=[DH], Tm=[TM], eps2=[eps2], L_psi=[L_psi],
        alpha_th=[0.0], beta_th=[0.0], L_th=[5.0], T=T, dt=dt,
        bulk="r14", kg_delta=1e-2, p_floor=1e-6,
        newton_tol=1e-7, newton_max=60, linsolver="cudss")


def drive_of(T):
    """Turnbull driving force drive = dh (T/Tm - 1) for this material.
    Negative below Tm (growth), positive above (melting)."""
    return DH * (T / TM - 1.0)


def _disc(c, r0, w=0.02, amp=0.95):
    return lambda x: amp * 0.5 * (1.0 - np.tanh(
        (np.sqrt((x[:, 0] - c[0]) ** 2
                 + (x[:, 1] - c[1]) ** 2) - r0) / w))


def _grid_map(mesh):
    fc = mesh.node_coords
    xs = np.unique(np.round(fc[:, 0], 12))
    ys = np.unique(np.round(fc[:, 1], 12))
    ix = np.searchsorted(xs, np.round(fc[:, 0], 12))
    iy = np.searchsorted(ys, np.round(fc[:, 1], 12))
    return len(xs), len(ys), ix, iy


# ---------------------------------------------------------------------
# quadrature crystallinity (PRIMARY measure; thresholded area SECONDARY)
# ---------------------------------------------------------------------
def _quad_weights(st):
    """Gauss-point quadrature weights (w_ref * detJ) in the same (e, q)
    flat order the stepper's _gp evaluator uses, per degree bin."""
    w = {}
    for pv in sorted(st.dm.bins):
        tb = st.dm.tables_by_p[pv]
        eids = st.mesh.bins[pv]
        h = st.mesh.tree.h()[eids]
        jac = (0.5 * h) ** st.dm.dim
        w[pv] = np.outer(jac, tb.w).ravel()
    return w


def quad_mean_psi(st, k=0):
    """Domain-averaged crystallinity <psi> = INT psi dV / INT dV by
    Gauss quadrature (diagnostics.conservation).  This is the PRIMARY,
    threshold-free crystallinity measure; it counts partial (diffuse-
    interface) crystallinity honestly instead of a hard psi>0.5 cut."""
    vpsi, _ = st._gp(st.psi(k))
    wts = _quad_weights(st)
    keys = sorted(vpsi)
    field = np.concatenate([vpsi[pv] for pv in keys])
    weight = np.concatenate([wts[pv] for pv in keys])
    return _cons.mean_field(field, weight)


def thresholded_area(st, cons, thr=0.5, k=0):
    """SECONDARY measure: nodal area fraction with psi > thr.  Reported
    alongside the quadrature mean and swept over thr for sensitivity."""
    return float(np.mean(np.asarray(cons.T @ st.psi(k)) > thr))


# ---------------------------------------------------------------------
# grow / melt
# ---------------------------------------------------------------------
def run_grow_melt(dm, mesh, cons, T, phi0=0.6, r0=0.15, t_end=0.5,
                  dt_max=0.02, thresholds=(0.4, 0.5, 0.6), device="cuda:0"):
    """Single seeded disc in a uniform undercooled (or superheated)
    blend.  Returns the quadrature crystallinity history <psi>(t) (the
    primary measure) plus the thresholded area history and a threshold
    sensitivity table for the end state."""
    st = _stepper(dm, T)
    st.set_initial([lambda x: np.full(len(x), phi0)],
                   [_disc((0.5, 0.5), r0)],
                   [lambda x: np.zeros(len(x))])
    nx, ny, ix, iy = _grid_map(mesh)

    ts = [0.0]
    qpsi = [quad_mean_psi(st)]
    area = [thresholded_area(st, cons)]
    checks = np.linspace(t_end / 30, t_end, 30)
    for tc in checks:
        st.march(t_end=tc, dt_max=dt_max, max_steps=1500, dt_min=1e-7)
        ts.append(st.t)
        qpsi.append(quad_mean_psi(st))
        area.append(thresholded_area(st, cons))
    thr_end = {f"{thr:g}": thresholded_area(st, cons, thr=thr)
               for thr in thresholds}
    psf = np.zeros((nx, ny)); psf[ix, iy] = np.asarray(cons.T @ st.psi(0))
    return dict(t=np.array(ts), qpsi=np.array(qpsi), area=np.array(area),
                psi=psf, T=T, drive=drive_of(T),
                qpsi0=qpsi[0], qpsi1=qpsi[-1],
                a0=area[0], a1=area[-1], thr_end=thr_end,
                nx=nx, ny=ny)


# ---------------------------------------------------------------------
# interface velocity vs undercooling
# ---------------------------------------------------------------------
def _eff_radius(area_frac):
    """Effective disc radius from a crystalline area fraction on the unit
    square: A = pi r^2 -> r = sqrt(A/pi).  Valid while the crystal is a
    growing disc that has not yet touched a boundary."""
    return np.sqrt(np.maximum(area_frac, 0.0) / np.pi)


def run_interface_velocity(dm, mesh, cons, T, phi0=0.6, r0=0.15,
                           t_end=0.3, dt_max=0.02, n_checks=18,
                           device="cuda:0"):
    """Seed one disc below Tm and track its effective radius r_eff(t) =
    sqrt(A/pi) from the QUADRATURE crystallinity.  In the linear-growth
    regime dr/dt is the constant interface velocity v; we fit v by least
    squares over the growth window (before the crystal fills the box).
    Returns v, the fit R^2, and the trajectory."""
    st = _stepper(dm, T)
    st.set_initial([lambda x: np.full(len(x), phi0)],
                   [_disc((0.5, 0.5), r0)],
                   [lambda x: np.zeros(len(x))])
    ts = [0.0]
    qpsi = [quad_mean_psi(st)]
    checks = np.linspace(t_end / n_checks, t_end, n_checks)
    for tc in checks:
        st.march(t_end=tc, dt_max=dt_max, max_steps=1500, dt_min=1e-7)
        ts.append(st.t)
        qpsi.append(quad_mean_psi(st))
    ts = np.array(ts)
    reff = _eff_radius(np.array(qpsi))
    # fit v over the interior growth window (skip the first settling
    # point and stop before the crystal fills > 60% of the box)
    grow = (np.arange(reff.size) >= 1) & (np.array(qpsi) < 0.6)
    if grow.sum() >= 2:
        A = np.vstack([ts[grow], np.ones(grow.sum())]).T
        (v, b), res, *_ = np.linalg.lstsq(A, reff[grow], rcond=None)
        yhat = A @ np.array([v, b])
        ss_res = float(np.sum((reff[grow] - yhat) ** 2))
        ss_tot = float(np.sum((reff[grow] - reff[grow].mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        v, r2 = float("nan"), float("nan")
    return dict(T=T, dT=TM - T, drive=drive_of(T), v=float(v), r2=float(r2),
                t=ts, reff=reff, qpsi=np.array(qpsi))


def sweep_interface_velocity(dm, mesh, cons, temps, device="cuda:0"):
    """Interface velocity v(dT) across a set of undercoolings (all
    T < Tm).  Returns per-T dicts and the aligned (dT, v) arrays."""
    runs = [run_interface_velocity(dm, mesh, cons, T, device=device)
            for T in temps]
    dT = np.array([r["dT"] for r in runs])
    v = np.array([r["v"] for r in runs])
    drive = np.array([r["drive"] for r in runs])
    return dict(runs=runs, dT=dT, v=v, drive=drive, temps=np.array(temps))


# ---------------------------------------------------------------------
# critical radius (sub- vs super-critical seed)
# ---------------------------------------------------------------------
def _seed_fate(dm, mesh, cons, T, r0, phi0=0.6, t_end=0.25, dt_max=0.02,
               n_checks=8):
    """March a single seed of radius r0 at temperature T and report
    whether its quadrature crystallinity grows or decays.  A grown seed
    is supercritical, a decayed one subcritical."""
    st = _stepper(dm, T)
    st.set_initial([lambda x: np.full(len(x), phi0)],
                   [_disc((0.5, 0.5), r0)],
                   [lambda x: np.zeros(len(x))])
    q0 = quad_mean_psi(st)
    checks = np.linspace(t_end / n_checks, t_end, n_checks)
    for tc in checks:
        st.march(t_end=tc, dt_max=dt_max, max_steps=1500, dt_min=1e-7)
    q1 = quad_mean_psi(st)
    return dict(r0=r0, T=T, q0=float(q0), q1=float(q1),
                grew=bool(q1 > q0 * 1.02))


def run_critical_radius(dm, mesh, cons, T, radii, device="cuda:0",
                        **kw):
    """Seed a range of radii at a single undercooling and classify each
    as grow/shrink.  The transition radius r* is bracketed between the
    largest shrinking and the smallest growing seed.  Returns the per-
    radius fates and the bracket."""
    fates = [_seed_fate(dm, mesh, cons, T, r0, **kw) for r0 in radii]
    radii = np.array([f["r0"] for f in fates])
    grew = np.array([f["grew"] for f in fates])
    shrank = radii[~grew]
    grown = radii[grew]
    r_lo = float(shrank.max()) if shrank.size else float("nan")
    r_hi = float(grown.min()) if grown.size else float("nan")
    r_star = 0.5 * (r_lo + r_hi) if np.isfinite(r_lo) and np.isfinite(r_hi) \
        else (r_hi if np.isfinite(r_hi) else r_lo)
    return dict(T=T, drive=drive_of(T), radii=radii, grew=grew,
                r_lo=r_lo, r_hi=r_hi, r_star=float(r_star), fates=fates)


# ---------------------------------------------------------------------
# Avrami / JMAK kinetics
# ---------------------------------------------------------------------
def run_avrami(dm, mesh, cons, T=250.0, L_psi=11.0, n_seeds=4, r0=0.045,
               t_end=1.8, dt_max=0.02, phi0=0.62, seed=3,
               thresholds=(0.4, 0.5, 0.6), device="cuda:0"):
    """Multiple pre-placed nuclei with distinct orientation markers.
    Track the crystalline fraction and fit the Avrami exponent.

    ACCELERATED (pedagogical) run: the deep undercooling T = 250 K and
    the large kinetic prefactor L_psi = 11 are chosen so X sweeps the
    full (0, 1) range in a short run; they are not the physical PCBM
    rate (see module docstring).

    The primary fraction is the QUADRATURE crystallinity <psi>(t); the
    thresholded area at several psi cuts is reported for sensitivity.
    The Avrami fit removes the pre-placed-seed baseline X0 via the
    generalized-JMAK conversion X* = (X - X0)/(1 - X0) and reports the
    fit window, R^2, and a 95% CI on n."""
    rng = np.random.default_rng(seed)
    # well-separated random seed centers
    ctrs, tries = [], 0
    while len(ctrs) < n_seeds and tries < 8000:
        c = rng.uniform(0.1, 0.9, size=2)
        if all(np.hypot(c[0] - q[0], c[1] - q[1]) > 0.2 for q in ctrs):
            ctrs.append(c)
        tries += 1
    st = _stepper(dm, T, L_psi=L_psi)
    coords = st.free_coords
    psi0 = np.zeros(st.nfree)
    th0 = np.zeros(st.nfree)
    for k, c in enumerate(ctrs):
        d = _disc(c, r0)(coords)
        m = d > psi0
        psi0[m] = d[m]
        r = np.hypot(coords[:, 0] - c[0], coords[:, 1] - c[1])
        th0[r < r0 + 0.06] = 0.15 + 0.7 * (k / max(n_seeds - 1, 1))
    st.set_initial([lambda x: np.full(len(x), phi0)],
                   [lambda x: psi0], [lambda x: th0])
    nx, ny, ix, iy = _grid_map(mesh)

    ts = [0.0]
    Xq = [quad_mean_psi(st)]                     # primary (quadrature)
    Xa = {f"{thr:g}": [thresholded_area(st, cons, thr=thr)]
          for thr in thresholds}
    checks = np.linspace(t_end / 40, t_end, 40)
    for tc in checks:
        st.march(t_end=tc, dt_max=dt_max, max_steps=2000, dt_min=1e-7)
        ts.append(st.t)
        Xq.append(quad_mean_psi(st))
        for thr in thresholds:
            Xa[f"{thr:g}"].append(thresholded_area(st, cons, thr=thr))
    ts = np.array(ts)
    Xq = np.array(Xq)
    Xa = {kk: np.array(vv) for kk, vv in Xa.items()}
    # fit on the psi>0.5 area fraction (the classical JMAK observable),
    # baseline-corrected for the pre-placed seeds
    X = Xa["0.5"]
    fit = _fit_avrami(ts, X)
    psf = np.zeros((nx, ny)); psf[ix, iy] = np.asarray(cons.T @ st.psi(0))
    thf = np.zeros((nx, ny)); thf[ix, iy] = np.asarray(cons.T @ st.theta(0))
    labels, sizes = grain_labels(mesh.node_coords,
                                 np.asarray(cons.T @ st.psi(0)),
                                 np.asarray(cons.T @ st.theta(0)))
    return dict(t=ts, X=X, Xq=Xq, Xa=Xa, fit=fit,
                n_avrami=fit["n"], k_avrami=fit["k"], r2=fit["r2"],
                ci95=fit["ci95"], X0=fit["X0"], fit_lo=fit["lo"],
                fit_hi=fit["hi"], fit_npts=fit["npts"],
                psi=psf, theta=thf, nx=nx, ny=ny,
                n_grains=int((sizes > 0).sum()), n_seeds=len(ctrs),
                X_end=float(X[-1]), Xq_end=float(Xq[-1]))


def _fit_avrami(t, X, X0=None, lo=0.05, hi=0.9):
    """Fit X* = 1 - exp[-(k t)^n] via the double-log linearization
    ln(-ln(1-X*)) = n ln t + n ln k, where X* = (X - X0)/(1 - X0) is the
    generalized-JMAK fraction that removes the pre-placed-seed baseline
    X0 (default X0 = X[0]).  The fit runs over the pre-impingement window
    lo < X* < hi (the late stage X* -> 1 is impingement-dominated and
    rolls the apparent exponent down).  Returns the exponent n, the
    prefactor k, the coefficient of determination R^2, a 95% CI on n, the
    number of fit points, the window, and X0."""
    t = np.asarray(t, float)
    X = np.asarray(X, float)
    if X0 is None:
        X0 = float(X[0])
    denom = max(1.0 - X0, 1e-9)
    Xs = (X - X0) / denom
    m = (Xs > lo) & (Xs < hi) & (t > 1e-6) & np.isfinite(Xs)
    nan = float("nan")
    if m.sum() < 3:
        return dict(n=nan, k=nan, r2=nan, ci95=nan, npts=int(m.sum()),
                    lo=lo, hi=hi, X0=float(X0))
    yy = np.log(-np.log(1.0 - Xs[m]))
    xx = np.log(t[m])
    A = np.vstack([xx, np.ones_like(xx)]).T
    coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
    n, b = float(coef[0]), float(coef[1])
    yhat = A @ coef
    ss_res = float(np.sum((yy - yhat) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else nan
    npts = int(m.sum())
    dof = npts - 2
    if dof > 0:
        sigma2 = ss_res / dof
        cov = sigma2 * np.linalg.inv(A.T @ A)
        ci95 = float(1.96 * np.sqrt(cov[0, 0]))
    else:
        ci95 = nan
    k = float(np.exp(b / n)) if n != 0 else nan
    return dict(n=n, k=k, r2=r2, ci95=ci95, npts=npts, lo=lo, hi=hi,
                X0=float(X0))
