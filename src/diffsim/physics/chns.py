"""CHNS physics helpers — Task 1 (SP-0): phase-mixture closures, capillary
force density, and VMS stabilization parameter for the coupled
Cahn-Hilliard–Navier-Stokes brick.

All functions are pure NumPy; Warp kernels (Tasks 3–7) call these host forms
for CPU reference paths and gate their correctness against these results.

Spec reference
--------------
  Spec §1 (task-1-brief.md), functions mix_props / capillary_gp / tau_m_gp.
  CHNSIntegrandsGenForm.hpp:125-217 pullback pattern is mirrored here for the
  phase-mixture averaging (linear-in-phi interpolation).

tau_m_gp legacy
---------------
  Generalizes the constant-viscosity h-based form in
  ``src/diffsim/physics/vms.py:tau_hbased_host`` (lines 48-56) to
  per-GP local rho, eta:
    - vms.py uses `nu = eta / (rho * Re)` as a single scalar;
      here rho and eta vary per GP so we write the formula in primitive vars.
    - Ci0 maps to `c1`  (advective coefficient, 4.0).
    - The transient term uses (Ci0/dt^2) which equals (2/dt)^2 when Ci0=4,
      matching vms.py's (2*b0/dt)^2 with b0=1.
    - Ci1 maps to the viscous coefficient (36.0), matching CI_F in vms.py.
    - The outer /rho factor converts the pressure-stabilized tauM (units of
      time/density) into the momentum form required by the CHNS variational
      residual (see spec §1).
"""

import numpy as np

__all__ = ["mix_props", "capillary_gp", "tau_m_gp"]

# ---------------------------------------------------------------------------
# Phase-mixture linear interpolation
# ---------------------------------------------------------------------------


def mix_props(
    phi: np.ndarray,
    rho_h: float,
    rho_l: float,
    eta_h: float,
    eta_l: float,
    floor_frac: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Linear phase-mixture density and viscosity at arbitrary phi values.

    Interpolates linearly in phi: for phi in [-1, 1]
        rho = a_rho * phi + b_rho,   a = (hi-lo)/2, b = (hi+lo)/2
        eta = a_eta * phi + b_eta
    Clamps results from below at ``floor_frac * lo`` to prevent zero or
    negative material properties when phi overshoots the [-1, 1] interval.

    Parameters
    ----------
    phi : ndarray, any shape
        Phase-field values.
    rho_h, rho_l : float
        Density of the heavy (phi=+1) and light (phi=-1) phase.
    eta_h, eta_l : float
        Dynamic viscosity of the heavy (phi=+1) and light (phi=-1) phase.
    floor_frac : float, optional
        Clamping floor as a fraction of the low-phase value.  Default 1e-3.

    Returns
    -------
    rho : ndarray, same shape as phi
        Mixture density, clamped from below.
    eta : ndarray, same shape as phi
        Mixture viscosity, clamped from below.
    n_clamped : int
        Number of entries where at least one property was clamped.
    """
    phi = np.asarray(phi, dtype=float)

    a_rho = 0.5 * (rho_h - rho_l)
    b_rho = 0.5 * (rho_h + rho_l)
    a_eta = 0.5 * (eta_h - eta_l)
    b_eta = 0.5 * (eta_h + eta_l)

    rho_raw = a_rho * phi + b_rho
    eta_raw = a_eta * phi + b_eta

    rho_floor = floor_frac * rho_l
    eta_floor = floor_frac * eta_l

    rho = np.maximum(rho_raw, rho_floor)
    eta = np.maximum(eta_raw, eta_floor)

    # Count GPs where any property was clamped
    clamped_mask = (rho_raw < rho_floor) | (eta_raw < eta_floor)
    n_clamped = int(np.sum(clamped_mask))

    return rho, eta, n_clamped


# ---------------------------------------------------------------------------
# Capillary body-force density
# ---------------------------------------------------------------------------


def capillary_gp(
    mu_gp: np.ndarray,
    grad_phi_gp: np.ndarray,
    Cn: float,
    We: float,
) -> np.ndarray:
    """Capillary force density at Gauss points: f_cap = (Cn*We)^{-1} mu grad(phi).

    Parameters
    ----------
    mu_gp : ndarray, shape [ngp]
        Chemical potential at each Gauss point.
    grad_phi_gp : ndarray, shape [ngp, dim]
        Gradient of the phase field at each Gauss point.
    Cn : float
        Cahn number (interface thickness / length scale).
    We : float
        Weber number.

    Returns
    -------
    f_gp : ndarray, shape [ngp, dim]
        Capillary force density at each Gauss point.
    """
    mu_gp = np.asarray(mu_gp, dtype=float)
    grad_phi_gp = np.asarray(grad_phi_gp, dtype=float)

    prefactor = 1.0 / (Cn * We)
    # mu_gp[:, None] broadcasts over the dim axis
    return prefactor * mu_gp[:, None] * grad_phi_gp


# ---------------------------------------------------------------------------
# Per-GP VMS stabilization parameter
# ---------------------------------------------------------------------------


def tau_m_gp(
    u_gp: np.ndarray,
    rho_gp: np.ndarray,
    eta_gp: np.ndarray,
    h: float,
    dt: float,
    Re: float,
    Ci: tuple[float, float] = (4.0, 36.0),
) -> np.ndarray:
    """Per-Gauss-point VMS momentum stabilization parameter tau_m.

    Implements the h-based form generalized to local (rho, eta):

        tau_m = [ (Ci0/dt^2) + (Ci0*|u|/h)^2
                              + (Ci1*eta/(rho*Re*h^2))^2 ]^{-1/2} / rho

    This is the variable-density extension of ``physics.vms.tau_hbased_host``
    (vms.py:48-56).  Constants Ci0=4, Ci1=36 match CI_F=36 and c1=4 from
    vms.py; with constant nu=eta/(rho*Re) the formulas agree up to the outer
    /rho factor and the squaring of the advective term (h-form vs metric-form
    difference, recorded in vms.py:13-18).

    Parameters
    ----------
    u_gp : ndarray, shape [ngp, dim]
        Velocity at each Gauss point.
    rho_gp : ndarray, shape [ngp]
        Local mixture density at each Gauss point.
    eta_gp : ndarray, shape [ngp]
        Local mixture viscosity at each Gauss point.
    h : float
        Element size (mesh spacing).
    dt : float
        Time step size.
    Re : float
        Reynolds number.
    Ci : tuple (Ci0, Ci1), optional
        Stabilization constants.  Default (4.0, 36.0).

    Returns
    -------
    tau : ndarray, shape [ngp]
        Stabilization parameter at each Gauss point.
    """
    u_gp = np.asarray(u_gp, dtype=float)
    rho_gp = np.asarray(rho_gp, dtype=float)
    eta_gp = np.asarray(eta_gp, dtype=float)

    Ci0, Ci1 = float(Ci[0]), float(Ci[1])

    u_mag = np.sqrt(np.sum(u_gp ** 2, axis=-1))  # [ngp]

    term_trans = Ci0 / dt ** 2
    term_adv = (Ci0 * u_mag / h) ** 2
    term_visc = (Ci1 * eta_gp / (rho_gp * Re * h ** 2)) ** 2

    denom = np.sqrt(term_trans + term_adv + term_visc)
    return 1.0 / (denom * rho_gp)
