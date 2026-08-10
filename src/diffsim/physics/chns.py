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

  Element-matrix / RHS constant-viscosity kernels being generalized:
  ``src/diffsim/api/ns_bricks.py:341-514`` (make_linear_ns_Ae / make_linear_ns_be).

tau_m_gp legacy
---------------
  Generalizes the constant-viscosity h-based tau formula in
  ``src/diffsim/physics/vms.py:48-56`` (``tau_hbased_host``) to per-GP local
  rho, eta.  Constant mapping (tau_hbased_host symbol -> chns.py symbol):

    tau_hbased_host            chns.py                    Value              Role
    ------------------         ---------------            -----              -------------------------
    (2*b0/dt)^2  (b0=1)        4.0 / dt^2                 4/dt^2             transient term
    c1=4.0                     ci0=4.0                    4.0                advective coefficient
    c2CI=CI_F*16*dim (dim-dep) ci_f*16*dim * nu^2/h^4     1152 @ dim=2       viscous coefficient
                                                           1728 @ dim=3
    nu = eta/(rho*Re)          eta_gp/(rho_gp*Re)         --                 kinematic viscosity, local

  The outer ``/ rho_gp`` converts the pressure-stabilized tauM (units of
  time/density) into the momentum stabilization form required by the CHNS
  variational residual (see spec §1).
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
    assert mu_gp.ndim == 1, f"mu_gp must be 1-D, got shape {mu_gp.shape}"
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

    Implements the h-based form from ``src/diffsim/physics/vms.py:48-56``
    (``tau_hbased_host``) generalized to per-GP local (rho, eta):

        tau_m = [ (2*b0/dt)^2 + c1*|u|^2/h^2
                              + c2CI*nu_local^2/h^4 ]^{-1/2} / rho_gp

    where b0=1, c1=ci0=4, nu_local = eta_gp/(rho_gp*Re), dim inferred from
    u_gp.shape[-1], and c2CI = CI_F * 16 * dim = 36 * 16 * dim
    (1152 for dim=2, 1728 for dim=3).

    Constant mapping to ``tau_hbased_host`` (vms.py:48-56):
      - transient: (2*b0/dt)^2 with b0=1  <=>  4/dt^2  (Ci[0]/dt^2)
      - advective: c1*|u|^2/h^2  (c1=4)   <=>  Ci[0]*u_mag**2/h**2
      - viscous:   c2CI*nu^2/h^4           <=>  Ci[1]*16*dim*nu_local**2/h**4

    See also: ``src/diffsim/api/ns_bricks.py:341-514`` for the constant-nu
    element kernels (make_linear_ns_Ae / make_linear_ns_be) being generalized.

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
    Ci : tuple (ci0, ci_f), optional
        Stabilization constants: ci0=advective coefficient (default 4.0),
        ci_f=CI_F in the house constant c2CI = ci_f * 16 * dim (default 36.0).
        Default (4.0, 36.0).

    Returns
    -------
    tau : ndarray, shape [ngp]
        Stabilization parameter at each Gauss point.
    """
    u_gp = np.asarray(u_gp, dtype=float)
    rho_gp = np.asarray(rho_gp, dtype=float)
    eta_gp = np.asarray(eta_gp, dtype=float)

    ci0, ci_f = float(Ci[0]), float(Ci[1])
    dim = u_gp.shape[-1]

    u_mag = np.sqrt(np.sum(u_gp ** 2, axis=-1))  # [ngp]

    nu_loc = eta_gp / (rho_gp * Re)               # local kinematic viscosity

    c2CI = ci_f * 16.0 * dim                      # house constant: 36*16*dim
    term_trans = ci0 / dt ** 2                    # (2*b0/dt)^2 with b0=1, ci0=4
    term_adv = ci0 * u_mag ** 2 / h ** 2          # c1*|u|^2/h^2, c1=ci0=4
    term_visc = c2CI * nu_loc ** 2 / h ** 4       # c2CI*nu^2/h^4 (dim-aware)

    denom = np.sqrt(term_trans + term_adv + term_visc)
    return 1.0 / (denom * rho_gp)
