"""Thermodynamic + morphology analysis helpers for the film front end.

The interface-width constructions are PROMOTED from the validated
campaign code (benchmarks/phase-field/wodo_fig67.py — the fig67 gate
lineage); the FH curvature matches the wodo_film kernel EXACTLY
(including the C1-regularized inverse floor 1e-4 and the b-regularizer
floor 1e-3), so preflight verdicts describe the discrete model the
solver actually integrates, not an idealized FH.

CHARACTERISTIC LENGTH (documented choice): first-moment structure-
factor length on the DETRENDED field,
    L_c = sum(E) / sum(|k| E) = <|k|>^-1,   E = |FFT2(q)|^2,
with k in CYCLES per unit length, i.e. L_c is the energy-weighted mean
WAVELENGTH of the pattern. q = phi - lateral mean per row (removing the
vertical drying/enrichment trend so L_c measures the lateral domain
structure, not the film-scale stratification). Physical wavenumbers:
lateral spacing Lx/(nx-1), vertical h/(ny-1). This is the standard
coarsening-literature <k>^-1 measure; larger L_c = coarser domains.

ONSET convention: a row (fixed theta) is 'separated' when the LATERAL
standard deviation of phi_f exceeds AMP_ON = 0.1 — the wodo_fig3 house
amplitude for order-one composition variation (AMP_ON there bounds
|phi_p - phi_f| row means; here lateral std, since lateral structure is
what 2-D onset means).
"""
import numpy as np

AMP_ON = 0.1          # wodo_fig3 house convention: order-one separation


# ---------------------------------------------------------------------
# FH curvature (kernel-exact: _rinv floor 1e-4, _binv3 floor 1e-3)
# ---------------------------------------------------------------------
def _rinv(x):
    return 1.0 / np.maximum(x, 1e-4)


def _binv3(x):
    return 1.0 / np.maximum(x, 1e-3) ** 3


def fh_curvature(pp, pf, chi, N, b_reg=0.0):
    """(d11, d12, d22) of the regularized FH exchange free energy at
    (phi_p, phi_f) — the wodo_film kernel's Jacobian entries."""
    c12, c1s, c2s = chi
    Np, Nf, Ns = N
    ps = 1.0 - pp - pf
    d11 = _rinv(pp) / Np + _rinv(ps) / Ns - 2.0 * c1s \
        + 2.0 * b_reg * (_binv3(pp) + _binv3(ps))
    d22 = _rinv(pf) / Nf + _rinv(ps) / Ns - 2.0 * c2s \
        + 2.0 * b_reg * (_binv3(pf) + _binv3(ps))
    d12 = _rinv(ps) / Ns + c12 - c1s - c2s + 2.0 * b_reg * _binv3(ps)
    return d11, d12, d22


def curvature_eigs(pp, pf, chi, N, b_reg=0.0):
    """Eigenvalues (ascending) of [[d11, d12], [d12, d22]]."""
    d11, d12, d22 = fh_curvature(pp, pf, chi, N, b_reg)
    m = 0.5 * (d11 + d22)
    r = np.sqrt(0.25 * (d11 - d22) ** 2 + d12 ** 2)
    return m - r, m + r


def drying_line_crossing(pp0, pf0, chi, N, b_reg=0.0, nscan=2000):
    """Scan the drying line (fixed p:f ratio, solvent evaporating):
    total solute c = (pp0+pf0)/h. Returns (h_star, phis_star, c_star)
    at the first spinodal entry (min curvature eig < 0), or None if the
    line never destabilizes before c = 0.98."""
    c0 = pp0 + pf0
    fp = pp0 / c0
    for c in np.linspace(c0, 0.98, nscan):
        lo, _ = curvature_eigs(c * fp, c * (1.0 - fp), chi, N, b_reg)
        if lo < 0.0:
            return c0 / c, 1.0 - c, c
    return None


# ---------------------------------------------------------------------
# interface widths (common-tangent; promoted from wodo_fig67.py)
# ---------------------------------------------------------------------
def _clip01(u):
    return np.clip(u, 1e-12, 1.0 - 1e-12)


def _binary_f(u, Np, Nf, chi):
    u = _clip01(u)
    return (u / Np * np.log(u) + (1.0 - u) / Nf * np.log(1.0 - u)
            + chi * u * (1.0 - u))


def _lower_hull(uu, ff):
    hull = []
    for i in range(len(uu)):
        while len(hull) > 1:
            i0, i1 = hull[-2], hull[-1]
            if ((ff[i1] - ff[i0]) * (uu[i] - uu[i1])
                    >= (ff[i] - ff[i1]) * (uu[i1] - uu[i0])):
                hull.pop()
            else:
                break
        hull.append(i)
    return hull


_UGRID = np.unique(np.concatenate([
    np.linspace(1e-9, 1 - 1e-9, 4001),
    np.geomspace(1e-9, 0.5, 2001),
    1.0 - np.geomspace(1e-9, 0.5, 2001)]))


def interface_width(Np, Nf, chi_pf, kap):
    """delta = dphi_e sqrt(kap / barrier) on the fully-dried p/f binary
    (the SMALLEST late-stage interface). Binodal via the lower convex
    hull of f (robust for deep quenches)."""
    uu = _UGRID
    ff = _binary_f(uu, Np, Nf, chi_pf)
    hull = _lower_hull(uu, ff)
    gaps = np.diff(uu[hull])
    j = int(np.argmax(gaps))
    a, b = uu[hull[j]], uu[hull[j + 1]]
    if b - a < 1e-3:
        return np.nan                       # single phase
    fa = _binary_f(a, Np, Nf, chi_pf)
    s = (_binary_f(b, Np, Nf, chi_pf) - fa) / (b - a)
    um = np.linspace(a, b, 2001)[1:-1]
    barrier = np.max(_binary_f(um, Np, Nf, chi_pf) - (fa + s * (um - a)))
    if barrier <= 0:
        return np.nan
    return (b - a) * np.sqrt(kap / barrier)


def interface_width_ternary(Np, Nf, chi, kap, phis, Ns=1.0):
    """Same construction on the pseudo-binary p-f exchange at fixed
    phi_s (the LARGER early/mid-drying interface). nan = single phase."""
    c12, c1s, c2s = chi
    c = 1.0 - phis

    def g(u):
        p1, p2 = _clip01(c * u), _clip01(c * (1.0 - u))
        return (p1 / Np * np.log(p1) + p2 / Nf * np.log(p2)
                + phis / Ns * np.log(phis) + c12 * p1 * p2
                + c1s * p1 * phis + c2s * p2 * phis)

    uu = _UGRID
    gg = g(uu)
    hull = _lower_hull(uu, gg)
    gaps = np.diff(uu[hull])
    j = int(np.argmax(gaps))
    a, b = uu[hull[j]], uu[hull[j + 1]]
    if b - a < 1e-3:
        return np.nan
    ga = g(a)
    s = (g(b) - ga) / (b - a)
    um = np.linspace(a, b, 2001)[1:-1]
    barrier = np.max(g(um) - (ga + s * (um - a)))
    if barrier <= 0:
        return np.nan
    return c * (b - a) * np.sqrt(kap / barrier)


def earliest_interface_width(pp0, pf0, chi, N, b_reg, kap):
    """Width of the FIRST interfaces the march will form: scan phi_s
    down from the drying-line spinodal entry until the pseudo-binary
    common tangent opens; -> (width, phis_at_formation) or (nan, nan).

    Note the b-regularizer shifts the spinodal (drying_line_crossing
    carries it) but not the tangent construction (its energy is tiny
    away from the simplex edges); widths quoted from pure FH."""
    Np, Nf, Ns = N
    cross = drying_line_crossing(pp0, pf0, chi, N, b_reg)
    if cross is None:
        return np.nan, np.nan
    _, phis_star, _ = cross
    for phis in np.arange(phis_star, 0.005, -0.02):
        w = interface_width_ternary(Np, Nf, chi, kap, max(phis, 0.01),
                                    Ns=Ns)
        if np.isfinite(w):
            return w, phis
    return interface_width(Np, Nf, chi[0], kap), 0.0


# ---------------------------------------------------------------------
# morphology metrics
# ---------------------------------------------------------------------
def characteristic_length(grid, Lx, h):
    """First-moment structure-factor length L_c = <|k|>^-1 (k in cycles
    per unit length: an energy-weighted mean wavelength) of the
    row-detrended field (module docstring). grid[iy, ix] on the
    computational strip; physical spacing (Lx/(nx-1), h/(ny-1))."""
    q = grid - grid.mean(axis=1, keepdims=True)   # remove vertical trend
    ny, nx = q.shape
    E = np.abs(np.fft.fft2(q)) ** 2
    kx = np.fft.fftfreq(nx, d=Lx / max(nx - 1, 1))
    ky = np.fft.fftfreq(ny, d=h / max(ny - 1, 1))
    KX, KY = np.meshgrid(kx, ky)
    kmag = np.hypot(KX, KY)
    E[0, 0] = 0.0
    denom = float((kmag * E).sum())
    if denom <= 0:
        return np.nan
    return float(E.sum()) / denom                 # 2pi/<k>, k in cycles


def lateral_std_profile(grid):
    """Per-row lateral standard deviation (onset detector input)."""
    return grid.std(axis=1)
