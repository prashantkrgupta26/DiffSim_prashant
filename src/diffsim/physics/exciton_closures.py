"""Exciton closure physics module (SP-1 Task A3).

Onsager-Braun dissociation, Langevin recombination, generation + waveform engine,
region mobility. Pure numpy reference — no warp/torch.

Block B GPU kernels verified against this; R3 learned heads replace these closures.

Sign convention for dist (from morphology module):
  dist > 0 : acceptor side
  dist < 0 : donor side
  dist = 0 : at the interface
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from diffsim.xdd.params import XDDParams, _KB, _Q, _EPS0
from diffsim.xdd.morphology import tanh_mask, interface_mask


# ──────────────────────────────────────────────────────────────────────────────
# OnsagerBraunDissociation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OnsagerBraunDissociation:
    """Onsager-Braun field-enhanced exciton dissociation rate.

    Physics:
      eps_r(dist) = eps_D + (eps_A - eps_D) * tanh_mask(dist, width)
      eps = EPS0 * eps_r
      E_B = Q^2 / (4*pi*eps*a)                     [J]
      E   = grad_phi_hat_mag * phi0 / x0             [V/m]
      b   = Q^3 * E / (8*pi*eps*(kT)^2)
      Phi(b) = 1 + b + b^2/3 + b^3/18 + b^4/180 + b^5/2700
      k_dim = (3*gamma0/(4*pi*a^3)) * exp(-E_B/(kT)) * Phi(b)
      k_hat = k_dim * t0
      mask  = interface_mask(dist, interface_thk/2)
      k_hat_d = k_hat * mask * ex_diss_d_scaling
      k_hat_a = k_hat * mask * ex_diss_a_scaling

    Analytic Jacobian w.r.t. grad_phi_hat_mag is also returned.
    """

    params: XDDParams
    width: float  # relaxed-mask width [m] for interface localization

    def __call__(
        self,
        grad_phi_hat_mag: float | np.ndarray,
        dist: np.ndarray,
    ) -> tuple:
        """Compute Onsager-Braun dissociation rates and derivative.

        Parameters
        ----------
        grad_phi_hat_mag : float or array
            Nondimensional electric field magnitude |∇φ̂|.
        dist : np.ndarray
            Signed distance field [m].

        Returns
        -------
        (k_hat_d, k_hat_a, (dk_dgrad_d, dk_dgrad_a))
            All arrays with same shape as dist.
            k_hat_d, k_hat_a : nondim dissociation rates for donor/acceptor excitons.
            dk_dgrad_d, dk_dgrad_a : analytic d(k_hat)/d(grad_phi_hat_mag).
        """
        p = self.params
        s = p.scales()

        phi0 = s.phi0
        x0   = s.x0
        t0   = s.t0
        kT   = _KB * p.T

        # Spatially varying permittivity
        eps_r = p.eps_D + (p.eps_A - p.eps_D) * tanh_mask(dist, self.width)
        eps   = _EPS0 * eps_r                          # [F/m]

        # Coulomb binding energy
        E_B = _Q**2 / (4.0 * math.pi * eps * p.a)    # [J]

        # Electric field (re-dimensioned)
        E = grad_phi_hat_mag * phi0 / x0              # [V/m]

        # Poole-Frenkel parameter
        b = _Q**3 * E / (8.0 * math.pi * eps * kT**2)

        # Braun–Onsager enhancement polynomial Phi(b)
        Phi = 1.0 + b + b**2 / 3.0 + b**3 / 18.0 + b**4 / 180.0 + b**5 / 2700.0

        # Dissociation prefactor
        prefactor = (3.0 * s.gamma0 / (4.0 * math.pi * p.a**3))
        exp_EB = np.exp(-E_B / kT)
        k_dim = prefactor * exp_EB * Phi             # [1/s]

        # Interface mask: peaks at dist=0, half-width = interface_thk/2
        mask = interface_mask(dist, p.interface_thk / 2.0)

        # Nondimensional rates
        k_hat_base = k_dim * t0
        k_hat_d = k_hat_base * mask * p.ex_diss_d_scaling
        k_hat_a = k_hat_base * mask * p.ex_diss_a_scaling

        # Analytic derivative w.r.t. grad_phi_hat_mag
        # dPhi/db = 1 + 2b/3 + b^2/6 + b^3/45 + b^4/540
        dPhi_db = 1.0 + 2.0 * b / 3.0 + b**2 / 6.0 + b**3 / 45.0 + b**4 / 540.0
        # db/dE = Q^3/(8*pi*eps*(kT)^2)
        db_dE = _Q**3 / (8.0 * math.pi * eps * kT**2)
        # dE/d(grad_phi_hat) = phi0/x0
        dE_dgrad = phi0 / x0
        # dk_dim/dE = prefactor * exp(-E_B/kT) * dPhi_db * db_dE
        dk_dim_dE = prefactor * exp_EB * dPhi_db * db_dE
        dk_dgrad_base = dk_dim_dE * dE_dgrad * t0 * mask
        dk_dgrad_d = dk_dgrad_base * p.ex_diss_d_scaling
        dk_dgrad_a = dk_dgrad_base * p.ex_diss_a_scaling

        return k_hat_d, k_hat_a, (dk_dgrad_d, dk_dgrad_a)


# ──────────────────────────────────────────────────────────────────────────────
# LangevinRecombination
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LangevinRecombination:
    """Langevin bimolecular recombination rate.

    Physics:
      eps_bar = EPS0*(eps_A+eps_D)/2
      gamma (by strategy):
        "sum":         gamma0 = 2*(mu_n+mu_p)*Q/(EPS0*(eps_A+eps_D))
        "min":         Q*min(mu_n,mu_p)/eps_bar
        "image_force": |((eps_D-eps_A)/(eps_D+eps_A))| * Q*mu_p/(EPS0*eps_D)
      gamma_hat = zeta * gamma * C0^2 / U0   [nondim]
      Spatial modulation:
        "uniform":   as-is
        "interface": gamma_hat * interface_mask(dist, width)
        "disabled":  0
      R_hat    = gamma_hat * n_hat * p_hat
      dR/dn    = gamma_hat * p_hat
      dR/dp    = gamma_hat * n_hat
    """

    params: XDDParams
    strategy: str = "sum"       # "sum" | "min" | "image_force"
    zeta: float = 1.0           # reduction factor in (0, 1]
    spatial: str = "uniform"    # "disabled" | "uniform" | "interface"
    width: float = 1e-9         # interface width [m] for "interface" spatial

    def __call__(
        self,
        n_hat: np.ndarray,
        p_hat: np.ndarray,
        dist: np.ndarray,
    ) -> tuple:
        """Compute Langevin recombination rate and derivatives.

        Parameters
        ----------
        n_hat : np.ndarray
            Nondimensional electron density.
        p_hat : np.ndarray
            Nondimensional hole density.
        dist : np.ndarray
            Signed distance field [m].

        Returns
        -------
        (R_hat, dR_dn, dR_dp)
            All arrays with same shape as inputs.
        """
        p = self.params
        s = p.scales()

        eps_bar = _EPS0 * (p.eps_A + p.eps_D) / 2.0

        if self.strategy == "sum":
            gamma = s.gamma0
        elif self.strategy == "min":
            gamma = _Q * min(p.mu_n, p.mu_p) / eps_bar
        elif self.strategy == "image_force":
            gamma = abs((p.eps_D - p.eps_A) / (p.eps_D + p.eps_A)) * _Q * p.mu_p / (_EPS0 * p.eps_D)
        else:
            raise ValueError(f"Unknown Langevin strategy: {self.strategy!r}")

        gamma_hat = self.zeta * gamma * s.C0**2 / s.U0

        if self.spatial == "uniform":
            gh = gamma_hat
        elif self.spatial == "interface":
            gh = gamma_hat * interface_mask(dist, self.width)
        elif self.spatial == "disabled":
            gh = 0.0
        else:
            raise ValueError(f"Unknown Langevin spatial: {self.spatial!r}")

        R_hat = gh * n_hat * p_hat
        dR_dn = gh * p_hat
        dR_dp = gh * n_hat

        return R_hat, dR_dn, dR_dp


# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Generation:
    """Exciton generation — spatial profile + waveform engine.

    Profiles:
      "constant"     — uniform volumetric rate using Gx_donor / Gx_acceptor params.
      "beer_lambert" — exponential attenuation: alpha0*Gamma0*exp(-alpha0*x0*(1-h_hat)).

    Waveforms:
      "cw"       — constant 1.0
      "step"     — Heaviside at t_on [s]
      "pulse"    — 1 while t < pulse_duration [s], else 0
      "rect_sin" — |sin(2*pi*freq*t)| where t = t_hat * t0
      "user"     — user_waveform(t_hat) callable

    RET (Resonance Energy Transfer):
      Additional acceptor-side generation from Förster transfer of donor excitons.
      Added to G_a_dim only, using ret_factor and interface_mask.
    """

    params: XDDParams
    profile: str = "constant"           # "constant" | "beer_lambert"
    waveform: str = "cw"                # "cw" | "step" | "pulse" | "rect_sin" | "user"
    alpha0: float = 1e7                 # absorption coefficient [1/m]
    Gamma0: float = 4.5e21             # photon flux [photons/m^2/s]
    t_on: float = 0.0                   # step onset time [s]
    pulse_duration: float = 0.0         # pulse duration [s]
    freq: float = 0.0                   # rect_sin frequency [Hz]
    user_waveform: object = field(default=None, compare=False, hash=False)  # callable t_hat -> float
    width: float = 1e-9                 # region-mask width [m]

    def spatial(
        self,
        dist: np.ndarray,
        h_hat: np.ndarray,
    ) -> tuple:
        """Static spatial generation profile (nondim).

        Parameters
        ----------
        dist : np.ndarray   signed distance [m]
        h_hat : np.ndarray  nondim position along device height (0..1)

        Returns
        -------
        (G_hat_d, G_hat_a)  nondimensional generation rates
        """
        p = self.params
        s = p.scales()
        U0 = s.U0
        x0 = s.x0

        w_acceptor = tanh_mask(dist, self.width)
        w_donor    = 1.0 - w_acceptor

        if self.profile == "constant":
            G_d_dim = p.Gx_donor    * w_donor
            G_a_dim = p.Gx_acceptor * w_acceptor
            # RET: Förster transfer from donor to acceptor at the interface
            G_a_dim = G_a_dim + (
                p.ret_factor * p.Gx_donor
                * interface_mask(dist, p.interface_thk / 2.0)
                * w_acceptor
            )

        elif self.profile == "beer_lambert":
            # G_dim(x) = alpha0 * Gamma0 * exp(-alpha0 * x0 * (1 - h_hat))
            # h_hat=1 → front surface (entrance), h_hat=0 → back
            G_dim = self.alpha0 * self.Gamma0 * np.exp(-self.alpha0 * x0 * (1.0 - h_hat))
            G_d_dim = 0.5 * G_dim * w_donor
            G_a_dim = 0.5 * G_dim * w_acceptor
            # RET (using donor contribution)
            G_a_dim = G_a_dim + (
                p.ret_factor * G_d_dim
                * interface_mask(dist, p.interface_thk / 2.0)
                * w_acceptor
            )

        else:
            raise ValueError(f"Unknown generation profile: {self.profile!r}")

        # Transport layer zeroing: zero G where h_hat is in the transport layer
        if p.transport_layer > 0.0:
            tl = p.transport_layer
            in_tl = (h_hat < tl) | (h_hat > 1.0 - tl)
            G_d_dim = np.where(in_tl, 0.0, G_d_dim)
            G_a_dim = np.where(in_tl, 0.0, G_a_dim)

        G_hat_d = G_d_dim / U0
        G_hat_a = G_a_dim / U0

        return G_hat_d, G_hat_a

    def amplitude(self, t_hat: float) -> float:
        """Waveform amplitude w(t) at nondimensional time t_hat.

        Parameters
        ----------
        t_hat : float
            Nondimensional time.

        Returns
        -------
        float
            Amplitude factor (>= 0).
        """
        s = self.params.scales()
        t0 = s.t0
        t  = t_hat * t0   # dimensional time [s]

        if self.waveform == "cw":
            return 1.0
        elif self.waveform == "step":
            return 1.0 if t >= self.t_on else 0.0
        elif self.waveform == "pulse":
            return 1.0 if t < self.pulse_duration else 0.0
        elif self.waveform == "rect_sin":
            return abs(math.sin(2.0 * math.pi * self.freq * t))
        elif self.waveform == "user":
            return float(self.user_waveform(t_hat))
        else:
            raise ValueError(f"Unknown waveform: {self.waveform!r}")

    def __call__(
        self,
        dist: np.ndarray,
        h_hat: np.ndarray,
        t_hat: float,
    ) -> tuple:
        """Full generation: spatial profile * waveform amplitude.

        Parameters
        ----------
        dist : np.ndarray   signed distance [m]
        h_hat : np.ndarray  nondim position along device height
        t_hat : float       nondim time

        Returns
        -------
        (G_hat_d, G_hat_a)
        """
        G_hat_d, G_hat_a = self.spatial(dist, h_hat)
        w = self.amplitude(t_hat)
        return G_hat_d * w, G_hat_a * w


# ──────────────────────────────────────────────────────────────────────────────
# RegionMobility
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RegionMobility:
    """Region-dependent carrier and exciton mobilities.

    Physics (nondim, using mu0 = max(mu_n, mu_p)):
      w_a = tanh_mask(dist, width)          (acceptor weight → 1 deep in acceptor)
      w_d = 1 - w_a                          (donor weight)

      mu_n_hat  = (mu_n/mu0)  * (w_a + mu_ratio * w_d)   majority in acceptor
      mu_p_hat  = (mu_p/mu0)  * (w_d + mu_ratio * w_a)   majority in donor
      mu_xd_hat = (mu_x_donor/mu0)    * (w_d + 1e-6 * w_a)
      mu_xa_hat = (mu_x_acceptor/mu0) * (w_a + 1e-6 * w_d)
      eps_r     = eps_D + (eps_A - eps_D) * w_a
    """

    params: XDDParams
    width: float = 1e-9   # tanh mask width [m]

    def __call__(self, dist: np.ndarray) -> dict:
        """Compute nondimensional mobilities and dielectric constant.

        Parameters
        ----------
        dist : np.ndarray
            Signed distance field [m].

        Returns
        -------
        dict with keys:
            mu_n_hat  : nondim electron mobility
            mu_p_hat  : nondim hole mobility
            mu_xd_hat : nondim donor exciton mobility
            mu_xa_hat : nondim acceptor exciton mobility
            eps_r     : local relative permittivity
        """
        p  = self.params
        s  = p.scales()
        mu0 = s.mu0

        w_a = tanh_mask(dist, self.width)
        w_d = 1.0 - w_a

        mu_n_hat  = (p.mu_n          / mu0) * (w_a + p.mu_ratio * w_d)
        mu_p_hat  = (p.mu_p          / mu0) * (w_d + p.mu_ratio * w_a)
        mu_xd_hat = (p.mu_x_donor    / mu0) * (w_d + 1e-6 * w_a)
        mu_xa_hat = (p.mu_x_acceptor / mu0) * (w_a + 1e-6 * w_d)
        eps_r     = p.eps_D + (p.eps_A - p.eps_D) * w_a

        return {
            "mu_n_hat":  mu_n_hat,
            "mu_p_hat":  mu_p_hat,
            "mu_xd_hat": mu_xd_hat,
            "mu_xa_hat": mu_xa_hat,
            "eps_r":     eps_r,
        }
