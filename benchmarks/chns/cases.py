"""benchmarks/chns/cases.py — CHNSCase dataclass and instances from legacy configs.

Legacy source: baskargroup-chns_nonnewtonian-b4cc4ff1aaf0.zip
  (at /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim/local_code_old/AM/)
Configs extracted with:
  unzip -p $Z.zip "$Z/config/bubble_rise_2d/Re35We10/config.txt"
  unzip -p $Z.zip "$Z/config/bubble_rise_2d/Re35We125/config.txt"
  unzip -p $Z.zip "$Z/config/Dam_break_2d/config.txt"
  unzip -p $Z.zip "$Z/config/RT_instability/2D/config.txt"

Global Constraints:
  - rho_ratio and eta_ratio are always rhoH/rhoL and etaH/etaL respectively,
    where rhoH (or etaH) is the heavier/more-viscous phase (phi=+1 side).
  - If a legacy ratio exceeds 1e3, it is capped at 1e3 (recorded per case below).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Tuple


@dataclass
class CHNSCase:
    """Benchmark case descriptor for Cahn-Hilliard/Navier-Stokes simulations.

    Fields consumed by later tasks (DO NOT rename without updating consumers):
        name         : human-readable identifier
        dim          : spatial dimension (2 or 3)
        level        : octree refinement level for the canonical benchmark grid
        Re           : Reynolds number
        We           : Weber number
        Cn           : Cahn number (interface thickness parameter)
        Pe           : Peclet number (Pe = 1/(Cn * Mobility))
        Fr           : Froude number
        rho_ratio    : density ratio rhoH / rhoL  (>= 1 by convention)
        eta_ratio    : viscosity ratio etaH / etaL (>= 1 by convention)
        t_end        : end time (non-dimensional)
        dt0          : initial time step
        ic_fn        : callable x: [n, dim] -> phi0: [n]  (initial phase field)
        domain_aspect: (Lx, Ly) aspect ratio tuple, default (1, 1)
    """

    name: str
    dim: int
    level: int
    Re: float
    We: float
    Cn: float
    Pe: float
    Fr: float
    rho_ratio: float
    eta_ratio: float
    t_end: float
    dt0: float
    ic_fn: Callable[[np.ndarray], np.ndarray]
    domain_aspect: Tuple[float, float] = (1, 1)


# ---------------------------------------------------------------------------
# Utility: tanh drop IC
# ---------------------------------------------------------------------------

def _tanh_drop_ic(
    xc: float,
    yc: float,
    radius: float,
    Cn: float,
    domain_Lx: float = 1.0,
    domain_Ly: float = 1.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return an ic_fn for a circular drop centred at (xc, yc) with given radius.

    The tanh profile spans [-1, 1] with interface half-width ~ Cn * referenceLengthScale.
    phi = +1 inside bubble (rhoL phase), phi = -1 outside (rhoH phase).
    Coordinates x[:,0], x[:,1] are assumed to be in [0, domain_Lx] x [0, domain_Ly].
    """
    _eps = Cn  # interface half-width parameter

    def ic_fn(x: np.ndarray) -> np.ndarray:
        # x: [n, 2] in physical coords
        r = np.sqrt((x[:, 0] - xc) ** 2 + (x[:, 1] - yc) ** 2)
        # tanh profile: +1 inside bubble, -1 outside
        return -np.tanh((r - radius) / (_eps * np.sqrt(2)))

    return ic_fn


def _tanh_dambreak_ic(
    vert_interface_x: float,
    Cn: float,
    domain_Lx: float = 1.0,
    domain_Ly: float = 1.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return an ic_fn for a dam-break: fluid column left of vert_interface_x.

    The legacy config uses caseTypeCHinit = "damBreak", vertInterfaceLoc = 2.0
    on a domain max=[4.0, 3.0]. We normalise to [0,1]^2 by dividing by domain size.
    phi = +1 for the heavy fluid (left of dam), phi = -1 for light fluid (right).
    """
    _eps = Cn

    def ic_fn(x: np.ndarray) -> np.ndarray:
        # x: [n, 2] assumed normalised to unit domain [0,1]^2
        # Normalise the interface location from legacy physical coords
        x_iface = vert_interface_x / domain_Lx
        return -np.tanh((x[:, 0] - x_iface) / (_eps * np.sqrt(2)))

    return ic_fn


def _tanh_rt_ic(
    vert_interface_y: float,
    amplitude: float,
    Cn: float,
    domain_Lx: float = 1.0,
    domain_Ly: float = 1.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return an ic_fn for Rayleigh-Taylor instability.

    Legacy config: caseTypeCHinit = "rayleighTaylor", amplitudeInitial = 0.1,
    vertInterfaceLoc = 6.0 on domain max=[1.0, 8.0].
    Heavy fluid (rhoH) on top (phi = +1), light fluid (rhoL) below (phi = -1).
    Perturbed interface y = y_iface + A * cos(2*pi*x / Lx_norm).
    """
    _eps = Cn

    def ic_fn(x: np.ndarray) -> np.ndarray:
        # x: [n, 2] assumed normalised to unit domain [0,1]^2
        # Normalise interface location and amplitude from legacy physical coords
        y_iface = vert_interface_y / domain_Ly
        A_norm = amplitude / domain_Ly
        # Cosine perturbation in x (normalised)
        y_pert = y_iface + A_norm * np.cos(2.0 * np.pi * x[:, 0])
        return -np.tanh((x[:, 1] - y_pert) / (_eps * np.sqrt(2)))

    return ic_fn


# ---------------------------------------------------------------------------
# Case 1: Bubble Rise Re=35, We=10  (Hysing et al. 2009 Test Case 1)
# ---------------------------------------------------------------------------
# Source: config/bubble_rise_2d/Re35We10/config.txt
# Key legacy parameters:
#   Re = 35.0, We = 10.0, Cn = 0.01, Pe = 3333.33, Fr = 1.0
#   rhoH = 1.0, rhoL = 0.1  -> rho_ratio = rhoH/rhoL = 10.0
#   etaH = 1.0, etaL = 0.1  -> eta_ratio = etaH/etaL = 10.0
#   domain max = [2.0, 4.0] (Lx=2, Ly=4); dropxc=1.0, dropyc=1.0, radiusds=0.5
#   dt = 2.5e-3, totalTime = 6
# Normalised drop centre: (xc, yc) = (1.0/2.0, 1.0/4.0) = (0.5, 0.25)
# Normalised radius: r = 0.5/2.0 = 0.25 (in x-direction units)
# Note: legacy domain is [0,2]x[0,4]; we keep domain_aspect=(1,2) to preserve shape.
BUBBLE_RISE_RE35_WE10 = CHNSCase(
    name="bubble_rise_Re35_We10",
    dim=2,
    level=6,   # legacy refine_lvl_base = 6 (interface at 11)
    # Source: Re = 35.0, We = 10.0
    Re=35.0,
    We=10.0,
    # Source: Cn = 0.01, Pe = 3333.33
    Cn=0.01,
    Pe=3333.33,
    # Source: Fr = 1.0
    Fr=1.0,
    # Source: rhoH=1.0, rhoL=0.1 -> ratio=10.0 (below 1e3 cap)
    rho_ratio=10.0,
    # Source: etaH=1.0, etaL=0.1 -> ratio=10.0 (below 1e3 cap)
    eta_ratio=10.0,
    # Source: totalTime = 6, dt = 2.5e-3
    t_end=6.0,
    dt0=2.5e-3,
    # IC: drop at (1.0, 1.0) radius 0.5 on [0,2]x[0,4]
    # Normalised to [0,1]x[0,2]: centre (0.5, 0.25), radius 0.25 in unit domain
    ic_fn=_tanh_drop_ic(
        xc=0.5, yc=0.25, radius=0.25, Cn=0.01,
        domain_Lx=2.0, domain_Ly=4.0,
    ),
    domain_aspect=(1, 2),  # legacy mesh max = [2.0, 4.0]
)


# ---------------------------------------------------------------------------
# Case 2: Bubble Rise Re=35, We=125  (Hysing et al. 2009 Test Case 2)
# ---------------------------------------------------------------------------
# Source: config/bubble_rise_2d/Re35We125/config.txt
# Key legacy parameters:
#   Re = 35.0, We = 125.0, Cn = 0.005, Pe = 13333.33, Fr = 1.0
#   rhoH = 1.0, rhoL = 0.001  -> rho_ratio = 1000.0 (exactly at cap)
#   etaH = 1.0, etaL = 0.01   -> eta_ratio = 100.0
#   domain max = [2.0, 4.0]; dropxc=1.0, dropyc=1.0, radiusds=0.5
#   dt = 2.5e-3, totalTime = 6
# NOTE: rhoH/rhoL = 1.0/0.001 = 1000 exactly — at the cap limit, no rounding needed.
BUBBLE_RISE_RE35_WE125 = CHNSCase(
    name="bubble_rise_Re35_We125",
    dim=2,
    level=6,   # legacy refine_lvl_base = 6 (interface at 12)
    # Source: Re = 35.0, We = 125.0
    Re=35.0,
    We=125.0,
    # Source: Cn = 0.005, Pe = 13333.33
    Cn=0.005,
    Pe=13333.33,
    # Source: Fr = 1.0
    Fr=1.0,
    # Source: rhoH=1.0, rhoL=0.001 -> ratio=1000.0 (at 1e3 cap, no truncation needed)
    rho_ratio=1000.0,
    # Source: etaH=1.0, etaL=0.01 -> ratio=100.0 (below 1e3 cap)
    eta_ratio=100.0,
    # Source: totalTime = 6, dt = 2.5e-3
    t_end=6.0,
    dt0=2.5e-3,
    # IC: same geometry as Re35We10 — drop at (1.0, 1.0) radius 0.5 on [0,2]x[0,4]
    ic_fn=_tanh_drop_ic(
        xc=0.5, yc=0.25, radius=0.25, Cn=0.005,
        domain_Lx=2.0, domain_Ly=4.0,
    ),
    domain_aspect=(1, 2),  # legacy mesh max = [2.0, 4.0]
)


# ---------------------------------------------------------------------------
# Case 3: Dam Break 2D
# ---------------------------------------------------------------------------
# Source: config/Dam_break_2d/config.txt
# Key legacy parameters:
#   Re = 280000, We = 5444.44, Cn = 0.005, Pe = 13333.33, Fr = 1.0
#   rhoH = 1.0, rhoL = 0.001  -> rho_ratio = 1000.0 (at 1e3 cap)
#   etaH = 1.0, etaL = 0.01   -> eta_ratio = 100.0
#   domain max = [4.0, 3.0]; caseTypeCHinit = "damBreak", vertInterfaceLoc = 2.0
#   dt = 2.5e-4, totalTime = 6
# NOTE: Re = 280000 is transcribed faithfully (high Re turbulent dam break).
#       rhoH/rhoL = 1000 exactly — at cap limit, no rounding needed.
DAM_BREAK_2D = CHNSCase(
    name="dam_break_2d",
    dim=2,
    level=6,   # legacy refine_lvl_base = 6 (interface at 12)
    # Source: Re = 280000, We = 5444.44
    Re=280000.0,
    We=5444.44,
    # Source: Cn = 0.005, Pe = 13333.33
    Cn=0.005,
    Pe=13333.33,
    # Source: Fr = 1.0
    Fr=1.0,
    # Source: rhoH=1.0, rhoL=0.001 -> ratio=1000.0 (at 1e3 cap)
    rho_ratio=1000.0,
    # Source: etaH=1.0, etaL=0.01 -> ratio=100.0 (below 1e3 cap)
    eta_ratio=100.0,
    # Source: totalTime = 6, dt = 2.5e-4
    t_end=6.0,
    dt0=2.5e-4,
    # IC: dam-break; vertInterfaceLoc=2.0 on domain [0,4]x[0,3]
    # Normalised interface at x=2.0/4.0=0.5 in unit domain
    ic_fn=_tanh_dambreak_ic(
        vert_interface_x=2.0, Cn=0.005,
        domain_Lx=4.0, domain_Ly=3.0,
    ),
    domain_aspect=(4, 3),  # legacy mesh max = [4.0, 3.0]
)


# ---------------------------------------------------------------------------
# Case 4: Rayleigh-Taylor Instability 2D
# ---------------------------------------------------------------------------
# Source: config/RT_instability/2D/config.txt
# Key legacy parameters:
#   Re = 3000, We = 100, Cn = 0.00125, Pe = 213333.33, Fr = 1.0
#   rhoH = 1.0, rhoL = 10.0  -> rho_ratio = rhoL/rhoH = 10.0
#     NOTE: In RT the legacy config has rhoL > rhoH (heavy fluid is "L" side,
#     which is on top). We store rho_ratio = max(rhoH,rhoL)/min(rhoH,rhoL) = 10.0
#     (rhoL=10.0, rhoH=1.0 in legacy naming; heavy fluid is actually rhoL=10).
#   etaH = 1.0, etaL = 1.0   -> eta_ratio = 1.0
#   domain max = [1.0, 8.0]; amplitudeInitial=0.1, vertInterfaceLoc=6.0
#   dt = 1e-4, totalTime = 6
RT_2D = CHNSCase(
    name="rt_instability_2d",
    dim=2,
    level=8,   # legacy refine_lvl_base = 8 (interface at 14)
    # Source: Re = 3000, We = 100
    Re=3000.0,
    We=100.0,
    # Source: Cn = 0.00125, Pe = 213333.33
    Cn=0.00125,
    Pe=213333.33,
    # Source: Fr = 1.0
    Fr=1.0,
    # Source: rhoH=1.0, rhoL=10.0 -> rho_ratio = max/min = 10.0 (below 1e3 cap)
    # (Legacy naming: heavy fluid labelled rhoL=10; see docstring note.)
    rho_ratio=10.0,
    # Source: etaH=1.0, etaL=1.0 -> ratio=1.0
    eta_ratio=1.0,
    # Source: totalTime = 6, dt = 1e-4
    t_end=6.0,
    dt0=1e-4,
    # IC: RT with interface at y=6.0 (Ly=8.0), amplitude=0.1, domain [0,1]x[0,8]
    # Normalised: y_iface=6/8=0.75, A_norm=0.1/8=0.0125
    ic_fn=_tanh_rt_ic(
        vert_interface_y=6.0, amplitude=0.1, Cn=0.00125,
        domain_Lx=1.0, domain_Ly=8.0,
    ),
    domain_aspect=(1, 8),  # legacy mesh max = [1.0, 8.0]
)
