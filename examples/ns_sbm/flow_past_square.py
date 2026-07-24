#!/usr/bin/env python
r"""Worked example 7.2 — Flow past a square in a channel (SBM d=0).

A square obstacle carved from a unit-box octree channel, sized so its faces are
CELL-ALIGNED (`half = k / 2^level`). The classification `lam=0.0` keeps only
fully-interior cells, so the surrogate faces coincide EXACTLY with the true box
boundary: `d = 0`, `corr = 1`. At `d = 0` the SBM shifted-Nitsche form
`S N_a = N_a + (grad N_a).d` DEGENERATES to standard Nitsche — so this example
isolates the two ways of imposing no-slip on the immersed body:

  MODE "strong"  — the obstacle nodes join the strong Dirichlet set (u = 0
                   imposed by row replacement). Classic strong BC.
  MODE "weak"    — no-slip imposed WEAKLY by the Nitsche block
                   (sbm_vector_dirichlet): consistency + adjoint-consistency +
                   penalty alpha*nu/h, added to the momentum predictor; the
                   obstacle nodes are LEFT FREE (they skip the strong overwrite).

Both modes run the projection engine (consistent-projection mode) AND compare
against the same-mesh MONOLITHIC saddle oracle with the identical BC treatment.
Drag Cd = F_x / (0.5 U^2 D) is read by integrating the traction on the obstacle
faces (surrogate_traction). Because d=0 this is the exact body-fitted traction.

WHAT TO EXPECT (verified on gpubox, level 4, Re=40, dt=0.02, steady):

  weak Nitsche:  projection Cd = +1.3529    monolithic Cd = +1.3941  (rel 3.0%)
                 projection mean|u| = 0.9057  monolithic 0.8550      (rel 5.9%)

  (The "strong" mode's base single-pass split is pressure-unstable on the open
   outflow and can blow up — that is a KNOWN rung-A finding; the weak-Nitsche
   consistent-projection path is the robust, validated one. See the tutorial
   Section 7.2 for the full story. This example runs the WEAK path by default.)

The bar is faithfulness to the SAME-MESH monolithic (not the literature square
Cd, which needs a far larger domain to remove confinement/blockage).

KNOBS:
  * MODE    — "weak" (default) or "strong".
  * LEVEL   — mesh refinement.
  * RE      — Reynolds number (40 steady; 100 sheds).
  * ALPHA   — Nitsche penalty scale (weak mode; 0 removes the penalty -> the
              obstacle stops being felt, an anti-vacuity check).

Run:
    PYTHONPATH=src:tests:examples python examples/ns_sbm/flow_past_square.py
"""
import os
import sys

import numpy as np

# The validated ladder fixtures + marchers live in tests/ (rung A = strong,
# rung B = weak Nitsche). This example CURATES them into one driver so the
# numbers are exactly the verified ladder numbers. Ensure tests/ is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))

from ladder_fixtures import build_square_channel_2d          # noqa: E402
from ladder_rungA_square_strong import (                     # noqa: E402
    march_projection as march_proj_strong,
    march_monolithic as march_mono_strong)
from ladder_rungB_square_nitsche import (                    # noqa: E402
    march_projection as march_proj_weak,
    march_monolithic as march_mono_weak, ALPHA)

HALF = 0.125     # cell-aligned half-width (k/2^level) -> exact body-fitted carve


def run(mode="weak", level=4, Re=40, alpha=ALPHA, device="cpu",
        dt=0.02, nsteps=600):
    """Build the aligned square-in-channel fixture (d=0) and march projection +
    monolithic with the chosen no-slip mode. Returns a results dict."""
    fx = build_square_channel_2d(level, Re, half=HALF, offset=0, device=device)
    D = 2.0 * HALF
    assert fx["dmax"] == 0.0, (
        f"expected an exact body-fitted carve (dmax==0), got {fx['dmax']}")
    print(f"\n=== Flow past a square  mode={mode}  Re={Re}  level={level} "
          f"(D={D}, dmax={fx['dmax']} => standard Nitsche)  dt={dt} ===")
    print(f" fluid cells={fx['n_fluid_cells']}  "
          f"obstacle nodes={int(fx['obstacle_node_mask'].sum())}")

    if mode == "weak":
        # consistent-projection + weak Nitsche + rotational wall pin (the
        # validated rung-B path — THE drag fix is rot_pin_wall=True).
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
    print(f" projection : Cd={pr['cd']:+.4f}  mean|u|={pr['mean_u']:.4f}  "
          f"||div||={pr['div']:.3e}  steps={pr['steps']}"
          + ("  BLEW UP" if pr.get("blew_up") else ""))
    print(f" monolithic : Cd={mo['cd']:+.4f}  mean|u|={mo['mean_u']:.4f}  "
          f"||div||={mo['div']:.3e}  steps={mo['steps']}")
    print(f" Cd rel-diff     = {cd_rel:.3%}   (bar: match same-mesh monolithic)")
    print(f" mean|u| rel-diff = {mu_rel:.3%}")
    return dict(mode=mode, cd_proj=pr["cd"], cd_mono=mo["cd"], cd_rel=cd_rel,
                mean_u_proj=pr["mean_u"], mean_u_mono=mo["mean_u"],
                mu_rel=mu_rel, blew_up=pr.get("blew_up", False),
                div_proj=pr["div"], div_mono=mo["div"])


if __name__ == "__main__":
    # KNOBS: python flow_past_square.py [MODE] [LEVEL] [RE]
    mode = sys.argv[1] if len(sys.argv) > 1 else "weak"
    level = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    Re = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    run(mode=mode, level=level, Re=Re)
