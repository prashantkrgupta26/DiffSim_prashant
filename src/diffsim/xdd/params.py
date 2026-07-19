"""XDDParams — parameter surface for the excitonic drift-diffusion solver (SP-1).

Follows the FilmParams idiom (src/diffsim/film/params.py):
  - Frozen dataclass with SI defaults for PM6:Y6
  - replace(**kw) returns new instance
  - scales() returns XDDScales (nondimensionalisation following DDFields.h)
  - from_cpu_config(path) parses the legacy config.txt
  - read_spec(path) reads .spec wavelength/value files
  - ret_factor_from_spectra(...) computes RET overlap integral
  - generation_from_spectra(...) computes photon-flux-weighted G rates

Block A: PURE numpy/scipy/stdlib — do NOT import warp or torch.
"""
from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Physical constants (CODATA 2018 exact values)
# ──────────────────────────────────────────────────────────────────────────────
_KB   = 1.380649e-23        # J/K  — Boltzmann
_Q    = 1.602176634e-19     # C    — elementary charge
_EPS0 = 8.8541878128e-12   # F/m  — vacuum permittivity
_H    = 6.62607015e-34      # J·s  — Planck
_C    = 2.99792458e8        # m/s  — speed of light


# ──────────────────────────────────────────────────────────────────────────────
# XDDScales — nondimensionalisation following DDFields.h
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class XDDScales:
    """Derived nondimensionalisation scales from XDDParams.

    All values in SI units.  Each field carries a one-line unit comment.
    Ground truth: DDFields.h from the CPU excitonic drift-diffusion code.
    """
    x0:      float   # [m]        characteristic length = height
    phi0:    float   # [V]        thermal voltage k_B T / q
    C0:      float   # [m^-3]     reference carrier density = N_C
    mu0:     float   # [m^2/Vs]   max(mu_n, mu_p)
    t0:      float   # [s]        x0^2 / (mu0 * phi0)  — drift time
    eps_m:   float   # [F/m]      EPS0 * max(eps_A, eps_D)
    U0:      float   # [m^-3 s^-1] mu0 * phi0 * C0 / x0^2
    J0:      float   # [A/m^2]    U0 * x0 * q
    lambda2: float   # [-]        phi0 * eps_m / (x0^2 * C0 * q)  (Debye^2)
    gamma0:  float   # [m^3/s]    2*(mu_n+mu_p)*q / (EPS0*(eps_A+eps_D)) Langevin


# ──────────────────────────────────────────────────────────────────────────────
# XDDParams — frozen dataclass with PM6:Y6 defaults
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class XDDParams:
    """Parameter surface for the excitonic drift-diffusion solver.

    SI inputs throughout.  Defaults are the PM6:Y6 canonical values from the
    CPU config.txt.  Use replace(**kw) to create variants.

    Attributes
    ----------
    unrecognized : dict
        Keys from from_cpu_config() that are not mapped to dataclass fields
        (e.g. PETSc solver options).  Always empty when constructed directly.
    """

    # -- Materials -----------------------------------------------------------
    mu_n:            float = 2e-7        # [m^2/Vs] electron mobility
    mu_p:            float = 1.5e-7      # [m^2/Vs] hole mobility
    mu_ratio:        float = 1e-6        # [-]       minority carrier mobility ratio
    mu_x_donor:      float = 3.9e-9     # [m^2/Vs] exciton mobility in donor
    mu_x_acceptor:   float = 3.9e-9     # [m^2/Vs] exciton mobility in acceptor
    eps_A:           float = 3.9         # [-]       relative permittivity of acceptor
    eps_D:           float = 3.0         # [-]       relative permittivity of donor
    E_g:             float = 1.1         # [eV]      LUMO_A – HOMO_D gap
    T:               float = 300.0       # [K]       temperature
    a:               float = 1.8e-9     # [m]       e-h separation distance
    interface_thk:   float = 2e-9       # [m]       interfacial thickness
    height:          float = 100e-9     # [m]       domain height
    N_C:             float = 2.5e25     # [m^-3]    effective DOS conduction band
    N_V:             float = 2.5e25     # [m^-3]    effective DOS valence band

    # -- Excitons ------------------------------------------------------------
    tau_x_donor:     float = 1e-9       # [s]       exciton lifetime donor
    tau_x_acceptor:  float = 1e-9       # [s]       exciton lifetime acceptor
    q_r_donor:       float = 1.0        # [-]       radiative fraction donor (tau_r/tau_x)
    q_r_acceptor:    float = 1.0        # [-]       radiative fraction acceptor

    # -- Generation ----------------------------------------------------------
    Gx_donor:          float = 1e28    # [m^-3 s^-1] exciton generation donor
    Gx_acceptor:       float = 1e28    # [m^-3 s^-1] exciton generation acceptor
    ret_factor:        float = 0.2     # [-]          Förster energy transfer fraction
    ex_diss_d_scaling: float = 1.0    # [-]          exciton dissociation scaling donor
    ex_diss_a_scaling: float = 1.0    # [-]          exciton dissociation scaling acceptor

    # -- Protocol ------------------------------------------------------------
    Voc:                       float = 0.0     # [V]  sweep end voltage
    V_app_start:               float = 0.0     # [V]  sweep start voltage
    dt:                        float = 1e-11   # [s]  initial time step
    dt_max:                    float = 5e-6    # [s]  max time step
    time_limit:                float = 1e-6    # [s]  simulation time limit
    solver_strategy:           int   = 3       # 0=PULSE_GEN,1=SS_JV,2=TR_JV,3=STEADY_PULSE
    pulse_duration:            float = 0.0     # [s]  pulse duration
    time_adaptivity_strategy:  int   = 4       # 4=LOGSPACE
    recombination_strategy:    int   = 1       # 0=DISABLED,1=UNIFORM,2=INTERFACE
    block_iteration_tol:       float = 1e-3    # [-]  block iteration tolerance
    time_stepping_tol:         float = 1e-3    # [-]  time stepping tolerance
    if_acceptor_excitons:      bool  = True    # whether acceptor excitons are active
    transport_layer:           float = 0.0     # [-]  transport layer fraction

    # -- Numerics ------------------------------------------------------------
    refine_lvl:          int   = 8     # mesh refinement level
    boundary_refine_lvl: int   = 8     # boundary mesh refinement level
    boundary_depth:      float = 0.0   # boundary depth
    morphology_file:     str   = ""    # path to morphology file

    # -- Parser residue (never set by direct construction) -------------------
    unrecognized: Dict = dataclasses.field(
        default_factory=dict, compare=False, hash=False)

    # ── Properties: tau_r / tau_nr splits ───────────────────────────────────

    @property
    def tau_r_donor(self) -> float:
        """[s] Radiative lifetime donor: tau_x_donor / q_r_donor."""
        return self.tau_x_donor / self.q_r_donor

    @property
    def tau_nr_donor(self) -> float:
        """[s] Non-radiative lifetime donor: tau_x_donor / (1 - q_r_donor).
        Returns math.inf when q_r_donor == 1 (purely radiative).
        """
        if self.q_r_donor >= 1.0:
            return math.inf
        return self.tau_x_donor / (1.0 - self.q_r_donor)

    @property
    def tau_r_acceptor(self) -> float:
        """[s] Radiative lifetime acceptor: tau_x_acceptor / q_r_acceptor."""
        return self.tau_x_acceptor / self.q_r_acceptor

    @property
    def tau_nr_acceptor(self) -> float:
        """[s] Non-radiative lifetime acceptor: tau_x_acceptor / (1 - q_r_acceptor).
        Returns math.inf when q_r_acceptor == 1 (purely radiative).
        """
        if self.q_r_acceptor >= 1.0:
            return math.inf
        return self.tau_x_acceptor / (1.0 - self.q_r_acceptor)

    # ── scales() ────────────────────────────────────────────────────────────

    def scales(self) -> XDDScales:
        """Return the XDDScales nondimensionalisation from this parameter set.

        Follows DDFields.h (CPU ground truth).  All quantities in SI.
        """
        x0   = self.height
        phi0 = _KB * self.T / _Q                         # thermal voltage [V]
        C0   = self.N_C
        mu0  = max(self.mu_n, self.mu_p)
        t0   = x0 * x0 / (mu0 * phi0)                    # drift time [s]
        eps_m = _EPS0 * max(self.eps_A, self.eps_D)       # [F/m]
        U0   = mu0 * phi0 * C0 / (x0 * x0)               # [m^-3 s^-1]
        J0   = U0 * x0 * _Q                               # [A/m^2]
        lam2 = phi0 * eps_m / (x0 * x0 * C0 * _Q)        # nondim Debye^2
        gam0 = (2.0 * (self.mu_n + self.mu_p) * _Q
                / (_EPS0 * (self.eps_A + self.eps_D)))    # [m^3/s] Langevin
        return XDDScales(
            x0=x0, phi0=phi0, C0=C0, mu0=mu0, t0=t0,
            eps_m=eps_m, U0=U0, J0=J0, lambda2=lam2, gamma0=gam0,
        )

    # ── replace() ───────────────────────────────────────────────────────────

    def replace(self, **kw) -> "XDDParams":
        """Return a new XDDParams with fields overridden by kw."""
        return dataclasses.replace(self, **kw)

    # ── from_cpu_config() ───────────────────────────────────────────────────

    @classmethod
    def from_cpu_config(cls, path: str | Path) -> "XDDParams":
        """Parse the CPU legacy config.txt and return an XDDParams instance.

        Format: ``key = value`` lines (or ``key=value``), ``#`` comments,
        quoted strings for filenames, booleans ``true/false``.
        Unknown keys (e.g. PETSc solver options) accumulate in ``.unrecognized``
        and never cause a crash.

        CPU name → dataclass field mapping:
          muRatio → mu_ratio
          interfaceThk → interface_thk
          Height → height
          transportLayer → transport_layer
          (all others are identity-mapped)
        """
        raw = _parse_cpu_config(path)

        # CPU name → (field_name, coerce_fn)
        _MAP: dict[str, tuple[str, type]] = {
            "mu_n":                     ("mu_n",                    float),
            "mu_p":                     ("mu_p",                    float),
            "muRatio":                  ("mu_ratio",                float),
            "mu_ratio":                 ("mu_ratio",                float),
            "mu_x_donor":               ("mu_x_donor",              float),
            "mu_x_acceptor":            ("mu_x_acceptor",           float),
            "eps_A":                    ("eps_A",                   float),
            "eps_D":                    ("eps_D",                   float),
            "E_g":                      ("E_g",                     float),
            "T":                        ("T",                       float),
            "a":                        ("a",                       float),
            "interfaceThk":             ("interface_thk",           float),
            "interface_thk":            ("interface_thk",           float),
            "Height":                   ("height",                  float),
            "height":                   ("height",                  float),
            "N_C":                      ("N_C",                     float),
            "N_V":                      ("N_V",                     float),
            "tau_x_donor":              ("tau_x_donor",             float),
            "tau_x_acceptor":           ("tau_x_acceptor",          float),
            "q_r_donor":                ("q_r_donor",               float),
            "q_r_acceptor":             ("q_r_acceptor",            float),
            "Gx_donor":                 ("Gx_donor",                float),
            "Gx_acceptor":              ("Gx_acceptor",             float),
            "ret_factor":               ("ret_factor",              float),
            "ex_diss_d_scaling":        ("ex_diss_d_scaling",       float),
            "ex_diss_a_scaling":        ("ex_diss_a_scaling",       float),
            "Voc":                      ("Voc",                     float),
            "V_app_start":              ("V_app_start",             float),
            "dt":                       ("dt",                      float),
            "dt_max":                   ("dt_max",                  float),
            "time_limit":               ("time_limit",              float),
            "solver_strategy":          ("solver_strategy",         int),
            "pulse_duration":           ("pulse_duration",          float),
            "time_adaptivity_strategy": ("time_adaptivity_strategy", int),
            "recombination_strategy":   ("recombination_strategy",  int),
            "block_iteration_tol":      ("block_iteration_tol",     float),
            "time_stepping_tol":        ("time_stepping_tol",       float),
            "if_acceptor_excitons":     ("if_acceptor_excitons",    _parse_bool),
            "transportLayer":           ("transport_layer",         float),
            "transport_layer":          ("transport_layer",         float),
            "refine_lvl":               ("refine_lvl",              int),
            "boundary_refine_lvl":      ("boundary_refine_lvl",     int),
            "boundary_depth":           ("boundary_depth",          float),
            "morphology_file":          ("morphology_file",         str),
        }

        kw: dict = {}
        unrecognized: dict = {}

        for cpu_key, val_str in raw.items():
            if cpu_key in _MAP:
                field_name, coerce = _MAP[cpu_key]
                kw[field_name] = coerce(val_str)
            else:
                unrecognized[cpu_key] = val_str

        obj = cls(**kw)
        # Bypass frozen to attach unrecognized (use object.__setattr__)
        object.__setattr__(obj, "unrecognized", unrecognized)
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_bool(s: str) -> bool:
    """Parse 'true'/'false' (case-insensitive) to bool."""
    sl = s.strip().lower()
    if sl in ("true", "1", "yes"):
        return True
    if sl in ("false", "0", "no"):
        return False
    raise ValueError(f"Cannot parse boolean: {s!r}")


def _parse_cpu_config(path: str | Path) -> dict[str, str]:
    """Parse legacy config.txt into a flat {key: value_str} dict.

    Rules:
    - Lines starting with # (after stripping) are comments
    - solver_options_dd/ex {{ ... }} blocks are parsed but their keys are
      kept under their raw names (they will land in unrecognized)
    - Quoted string values have quotes stripped
    - Inline # comments are stripped
    - Multi-line blocks like ``solver_options_dd = { ... }`` are collapsed
      and their interior keys extracted individually
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    # Remove block comments that span multiple lines (solver_options = { ... })
    # We parse them by tracking brace depth
    result: dict[str, str] = {}

    # Preprocess: collapse block-value assignments into single lines we skip,
    # but extract their interior key=value pairs.
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip pure comment lines
        if stripped.startswith("#") or stripped == "":
            i += 1
            continue

        # Detect block opening: key = { (possibly with more text)
        # e.g. "solver_options_dd = {"
        block_match = re.match(r"^(\w+)\s*=\s*\{(.*)$", stripped)
        if block_match:
            block_name = block_match.group(1)
            rest = block_match.group(2)
            # Collect until matching closing brace
            depth = rest.count("{") + 1 - rest.count("}")
            block_lines = [rest] if rest.strip() else []
            i += 1
            while i < len(lines) and depth > 0:
                bl = lines[i]
                depth += bl.count("{") - bl.count("}")
                if bl.strip() and not bl.strip().startswith("#"):
                    block_lines.append(bl.strip())
                i += 1
            # Parse interior lines of the block (ignore the block key itself)
            for bl in block_lines:
                bl = bl.rstrip("};").strip()
                if not bl or bl.startswith("#"):
                    continue
                # Strip inline comments
                bl = re.sub(r"\s*#.*$", "", bl).strip()
                if "=" in bl:
                    k, _, v = bl.partition("=")
                    k = k.strip()
                    v = v.strip().strip('"').strip("'").rstrip(";").strip()
                    if k:
                        result[k] = v
            continue

        # Normal key = value line
        # Strip inline comments
        stripped_no_comment = re.sub(r"\s*#.*$", "", stripped).strip()
        if "=" in stripped_no_comment:
            k, _, v = stripped_no_comment.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'").strip()
            if k:
                result[k] = v

        i += 1

    return result


# ──────────────────────────────────────────────────────────────────────────────
# read_spec
# ──────────────────────────────────────────────────────────────────────────────

def read_spec(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a .spec file: first line = integer count N, then N lines of 'wl val'.

    Returns
    -------
    wavelengths : np.ndarray  [nm]
    values      : np.ndarray  (arbitrary units; depends on file type)
    """
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    # First non-empty line is the count
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    n = int(lines[idx].strip())
    idx += 1

    wl = np.empty(n, dtype=np.float64)
    val = np.empty(n, dtype=np.float64)

    row = 0
    while idx < len(lines) and row < n:
        ln = lines[idx].strip()
        idx += 1
        if not ln:
            continue
        parts = ln.split()
        wl[row] = float(parts[0])
        val[row] = float(parts[1])
        row += 1

    if row != n:
        raise ValueError(
            f"read_spec({path}): expected {n} rows, got {row}")

    return wl, val


# ──────────────────────────────────────────────────────────────────────────────
# ret_factor_from_spectra
# ──────────────────────────────────────────────────────────────────────────────

def ret_factor_from_spectra(
    donor_pl: Tuple[np.ndarray, np.ndarray],
    acceptor_abs: Tuple[np.ndarray, np.ndarray],
) -> float:
    """Compute Förster energy transfer (RET) overlap integral.

    Parameters
    ----------
    donor_pl     : (wavelengths_nm, pl_values)   — donor photoluminescence
    acceptor_abs : (wavelengths_nm, abs_values)  — acceptor absorption

    Returns
    -------
    ret : float in [0, 1]
        trapz(min(acc_norm, pl_norm), λ) / trapz(pl_norm, λ)
        over the PL grid (acceptor interpolated onto PL grid, clamped to 0
        outside its range).
    """
    wl_pl, pl = donor_pl
    wl_acc, acc = acceptor_abs

    # Normalise each spectrum by its max
    pl_max = np.max(pl)
    acc_max = np.max(acc)
    if pl_max == 0 or acc_max == 0:
        return 0.0

    pl_norm  = pl  / pl_max
    acc_norm_full = acc / acc_max

    # Interpolate acceptor onto the PL wavelength grid; clamp outside to 0
    acc_on_pl = np.interp(wl_pl, wl_acc, acc_norm_full,
                          left=0.0, right=0.0)

    overlap = np.trapezoid(np.minimum(acc_on_pl, pl_norm), wl_pl)
    denom   = np.trapezoid(pl_norm, wl_pl)

    if denom == 0:
        return 0.0
    return float(overlap / denom)


# ──────────────────────────────────────────────────────────────────────────────
# generation_from_spectra
# ──────────────────────────────────────────────────────────────────────────────

def generation_from_spectra(
    solar:       Tuple[np.ndarray, np.ndarray],
    blend_abs:   Tuple[np.ndarray, np.ndarray],
    donor_abs:   Tuple[np.ndarray, np.ndarray],
    acceptor_abs: Tuple[np.ndarray, np.ndarray],
    height:      float,
) -> Tuple[float, float, float]:
    """Compute photon-flux-weighted exciton generation rates.

    Follows DDFields.h lines 111-190 (CPU ground truth).

    Parameters
    ----------
    solar       : (wl_nm, solar_irradiance)  solar spectrum
    blend_abs   : (wl_nm, blend_absorption)  blend absorption
    donor_abs   : (wl_nm, donor_absorption)
    acceptor_abs: (wl_nm, acceptor_absorption)
    height      : [m]  active layer thickness

    Returns
    -------
    (G_total, G_donor, G_acceptor) : [m^-3 s^-1]
        G_total = trapz(blend_abs * solar/100 / (h*c/λ_m), λ_nm) / height
        split:  G_acceptor/G_total = acc_abs_int / (acc_abs_int + donor_abs_int)
    """
    wl_sol, sol = solar
    wl_bl, bl   = blend_abs
    wl_d, d_abs = donor_abs
    wl_a, a_abs = acceptor_abs

    # Use blend wavelength grid as common grid (interpolate others onto it)
    wl = wl_bl

    sol_on_bl = np.interp(wl, wl_sol, sol,  left=0.0, right=0.0)
    d_on_bl   = np.interp(wl, wl_d,   d_abs, left=0.0, right=0.0)
    a_on_bl   = np.interp(wl, wl_a,   a_abs, left=0.0, right=0.0)

    # Wavelength in meters for photon energy
    wl_m = wl * 1e-9  # nm → m

    # Photon flux factor: solar/100 / (h*c/λ)  [photons m^-2 s^-1 nm^-1]
    # (solar units in the CPU code are W m^-2 nm^-1 / 10 for mW cm^-2 nm^-1)
    hc_over_wl = (_H * _C) / wl_m   # [J per photon]

    photon_flux_factor = sol_on_bl / 100.0 / hc_over_wl  # [photons m^-2 s^-1 nm^-1 per unit abs]

    # Total generation
    G_total = float(np.trapezoid(bl * photon_flux_factor, wl)) / height

    # Split: use donor/acceptor individual absorption integrals
    donor_abs_int   = float(np.trapezoid(d_on_bl * photon_flux_factor, wl))
    acceptor_abs_int = float(np.trapezoid(a_on_bl * photon_flux_factor, wl))

    denom_split = donor_abs_int + acceptor_abs_int
    if denom_split == 0:
        G_donor    = G_total / 2.0
        G_acceptor = G_total / 2.0
    else:
        frac_acc    = acceptor_abs_int / denom_split
        frac_donor  = donor_abs_int   / denom_split
        G_donor     = G_total * frac_donor
        G_acceptor  = G_total * frac_acc

    return G_total, G_donor, G_acceptor
