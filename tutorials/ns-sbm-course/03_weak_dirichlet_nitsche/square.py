"""NS-SBM course — flow past a square core (Chapter 03, weak Nitsche, d=0).

A square obstacle carved from a unit-box octree channel, sized so its faces are
CELL-ALIGNED (`half = k / 2^level`). Classification `lam=0.0` keeps only
fully-interior cells, so the surrogate faces coincide EXACTLY with the true box
boundary: `d = 0`, `corr = 1`. At `d = 0` the SBM shifted-Nitsche form
`S N_a = N_a + (grad N_a).d` DEGENERATES to standard Nitsche — so this module
isolates the two ways of imposing no-slip on an immersed body:

  MODE "weak"    — no-slip imposed WEAKLY by the Nitsche block
                   (sbm_vector_dirichlet): consistency + adjoint-consistency +
                   penalty alpha*nu/h; obstacle nodes stay FREE.
  MODE "strong"  — obstacle nodes join the strong Dirichlet set (row
                   replacement).

Both modes march the projection engine (consistent_projection) AND the
same-mesh MONOLITHIC saddle oracle with identical BC treatment. Drag
Cd = F_x / (0.5 U^2 D) is read by integrating the traction on the obstacle
faces (surrogate_traction); at d=0 that is the exact body-fitted traction.

This CURATES the validated ladder fixtures/marchers (tests/ladder_*), so the
numbers are exactly the verified ladder numbers — no toy re-implementation.
"""
from __future__ import annotations

import os
import sys

# the validated ladder fixtures + marchers live in tests/ (rung A = strong,
# rung B = weak Nitsche). Ensure the repo tests/ dir is importable.
_TESTS = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                     "..", "..", "..", "tests"))
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

from ladder_fixtures import build_square_channel_2d          # noqa: E402
from ladder_rungA_square_strong import (                     # noqa: E402
    march_projection as march_proj_strong,
    march_monolithic as march_mono_strong)
from ladder_rungB_square_nitsche import (                    # noqa: E402
    march_projection as march_proj_weak,
    march_monolithic as march_mono_weak, ALPHA)

HALF = 0.125     # cell-aligned half-width (k/2^level) -> exact body-fitted carve


def run_square(mode="weak", level=4, Re=40, alpha=ALPHA, device="cpu",
               dt=0.02, nsteps=600):
    """Build the aligned square-in-channel fixture (d=0) and march projection +
    monolithic with the chosen no-slip mode. Returns a results dict."""
    fx = build_square_channel_2d(level, Re, half=HALF, offset=0, device=device)
    D = 2.0 * HALF
    assert fx["dmax"] == 0.0, (
        f"expected an exact body-fitted carve (dmax==0), got {fx['dmax']}")

    if mode == "weak":
        pr = march_proj_weak(fx, dt=dt, nsteps=nsteps, rate_tol=5e-4,
                             alpha=alpha, rot_pin_wall=True)
        mo = march_mono_weak(fx, dt=dt, nsteps=nsteps, rate_tol=2e-4,
                             backflow_beta=0.5, boundary_vorticity=True,
                             alpha=alpha)
    elif mode == "strong":
        pr = march_proj_strong(fx, dt=dt, nsteps=nsteps, rate_tol=5e-4,
                               consistent_projection=True)
        mo = march_mono_strong(fx, dt=dt, nsteps=nsteps, rate_tol=2e-4,
                               backflow_beta=0.5, boundary_vorticity=True)
    else:
        raise ValueError(f"mode must be 'weak' or 'strong', got {mode!r}")

    cd_rel = abs(pr["cd"] - mo["cd"]) / abs(mo["cd"])
    mu_rel = abs(pr["mean_u"] - mo["mean_u"]) / abs(mo["mean_u"])
    return dict(
        mode=mode, level=level, Re=Re, D=D, dmax=fx["dmax"],
        n_fluid_cells=int(fx["n_fluid_cells"]),
        obstacle_nodes=int(fx["obstacle_node_mask"].sum()),
        cd_proj=pr["cd"], cd_mono=mo["cd"], cd_rel=cd_rel,
        mean_u_proj=pr["mean_u"], mean_u_mono=mo["mean_u"], mu_rel=mu_rel,
        blew_up=bool(pr.get("blew_up", False)),
        div_proj=pr["div"], div_mono=mo["div"],
        steps_proj=int(pr["steps"]), steps_mono=int(mo["steps"]))
