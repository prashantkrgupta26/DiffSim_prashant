"""OrgElMorph course - Physics P0: nondimensionalization + resolution calculator.

The model-hierarchy chapter is mostly written, but the nondimensional groups it
quotes are COMPUTED here (through the harness, with provenance) rather than
hand-copied.  Everything is deterministic arithmetic on the dimensional inputs,
so it is a fast, exact, tolerance-checkable record of the one worked
nondimensionalization the chapter walks through.

The binary Cahn-Hilliard nondimensionalization (the worked example)
-------------------------------------------------------------------
Dimensional:  dphi/dt = div(M grad mu),  mu = f'(phi) - kappa lap phi,
  f  ~ e0 = kT/v0        [energy / volume]   (bulk free-energy scale)
  kappa = e0 * lam**2    [energy / length]   (lam = a microscopic length)
  M                       [length**2 volume / (energy time)]  (mobility)

Scale by a length L (the domain), energy e0, and the diffusive time
  tau = L**2 / (M e0).
Then, with x~=x/L, t~=t/tau, mu~=mu/e0, f~=f/e0,

  dphi/dt~ = div~ grad~ mu~,   mu~ = f~'(phi) - kappa~ lap~ phi,
  kappa~ = kappa / (e0 L**2) = (lam / L)**2 .

So the ONLY nondimensional group of binary CH is the **Cahn number**
kappa~ = (lam/L)**2 = (interface width / domain size)**2.  The equilibrium
interface width in nondimensional units is:
  poly double well:   ell~ = sqrt(2 kappa~)        (exact tanh profile, W=1)
  FH spinodal length: ell~ = sqrt(kappa~ / |f''|)  (linear-instability length)
and the number of mesh cells it spans at level L (2**level cells/side) is
ell~ * 2**level -- the resolution constraint.
"""
from __future__ import annotations

import math


def cahn_number(interface_length, domain_size):
    """kappa~ = (lam / L)**2 -- the sole nondimensional group of binary CH."""
    return (interface_length / domain_size) ** 2


def poly_interface_cells(kappa_tilde, level, well=1.0):
    """Cells spanned by the polynomial double-well interface ell~=sqrt(2 k~/W)
    on a 2**level-per-side unit box."""
    return math.sqrt(2.0 * kappa_tilde / well) * (2 ** level)


def fh_spinodal_length(kappa_tilde, fpp):
    """FH spinodal length ell~ = sqrt(kappa~ / |f''|) (interface set by the
    balance of the gradient penalty against the bulk curvature)."""
    return math.sqrt(kappa_tilde / abs(fpp))


def fh_curvature(c, A, B):
    """f''(c) = A(1/c + 1/(1-c)) - 2B for Flory-Huggins; <0 => spinodal."""
    return A * (1.0 / c + 1.0 / (1.0 - c)) - 2.0 * B


def min_level_for_interface(interface_cells_target, ell_tilde):
    """Smallest mesh level that resolves the interface with >= target cells:
    ell~ * 2**level >= target  =>  level >= log2(target / ell~)."""
    return int(math.ceil(math.log2(interface_cells_target / ell_tilde)))


def diffusive_time(L, M, e0):
    """tau = L**2 / (M e0), the natural CH time scale (consistent units)."""
    return L ** 2 / (M * e0)


def turnbull_drive(dh, T, Tm):
    """Crystallization driving force drive = dh (T/Tm - 1): <0 below Tm (grow),
    >0 above (melt).  The sign the melting point sets (P6)."""
    return dh * (T / Tm - 1.0)


def compute(cfg):
    """Return the dimensional/nondimensional/numerical group dictionary the
    chapter quotes, from the config's dimensional inputs."""
    lam = cfg["interface_length_nm"]         # microscopic length lam (nm)
    L = cfg["domain_size_nm"]                # domain size L (nm)
    level = cfg["level"]
    target_cells = cfg["target_interface_cells"]
    A, B = cfg["fh_A"], cfg["fh_B"]          # FH entropy/enthalpy (nondim)
    c_bar = cfg["fh_c_bar"]

    k_tilde = cahn_number(lam, L)
    poly_ell = math.sqrt(2.0 * k_tilde)
    poly_cells = poly_interface_cells(k_tilde, level)
    fpp = fh_curvature(c_bar, A, B)
    fh_ell = fh_spinodal_length(k_tilde, fpp) if fpp < 0 else float("nan")
    fh_cells = fh_ell * (2 ** level) if fpp < 0 else float("nan")
    min_level = min_level_for_interface(target_cells, poly_ell)

    # crystallization accelerated-vs-physical driving force (P6 PCBM class)
    Tm = cfg["cryst_Tm_K"]; dh = cfg["cryst_dh"]
    drive_accel = turnbull_drive(dh, cfg["cryst_T_accel_K"], Tm)
    drive_phys = turnbull_drive(dh, cfg["cryst_T_phys_K"], Tm)

    return {
        # --- nondimensional groups ---
        "cahn_number": k_tilde,                       # kappa~ = (lam/L)^2
        "lam_over_L": lam / L,
        "poly_interface_ell_tilde": poly_ell,
        "poly_interface_cells": poly_cells,
        "fh_curvature_fpp": fpp,
        "fh_spinodal_ell_tilde": fh_ell,
        "fh_interface_cells": fh_cells,
        "spinodal_unstable": bool(fpp < 0),
        # --- numerical resolution constraint ---
        "level": level,
        "cells_per_side": 2 ** level,
        "min_level_for_target": min_level,
        "target_interface_cells": target_cells,
        # --- crystallization drive (accelerated vs physical, P6) ---
        "cryst_Tm_K": Tm,
        "cryst_drive_accel": drive_accel,
        "cryst_drive_phys": drive_phys,
        "cryst_drive_ratio": (drive_accel / drive_phys
                              if drive_phys != 0 else float("nan")),
        # echo the dimensional inputs so results.json is self-contained
        "interface_length_nm": lam, "domain_size_nm": L,
        "fh_A": A, "fh_B": B, "fh_c_bar": c_bar,
    }
