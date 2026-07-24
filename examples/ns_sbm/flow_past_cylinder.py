#!/usr/bin/env python
r"""Worked example 7.3 — Flow past an immersed body with a GENUINE SBM shift.

The Shifted Boundary Method (SBM) lets the immersed body sit OFF the mesh grid:
the true boundary Gamma is approximated by a grid-aligned SURROGATE boundary
Gamma~, and a Taylor shift `S N_a = N_a + (grad N_a).d` transfers the boundary
condition from the surrogate to the true wall (d = surrogate-GP -> closest point
on Gamma). This example turns the shift ON.

We carve a square whose center is offset by a SUB-CELL fraction (offset=0.05),
so the grid-aligned surrogate no longer coincides with the true `Box` face:
`0 < dmax < h`. Now the shift terms do REAL work:
  * the Taylor `(grad N_a).d` term in the Nitsche block, and
  * the area-correction `geo.corr = n_tilde . n` in the surrogate traction
(unlike examples 7.1/7.2 where d=0 and corr=1, the shift is inert).

This is the ladder's "rung C". We march the consistent-projection engine with
weak Nitsche + the SBM shift, and compare Cd against the same-mesh MONOLITHIC
saddle oracle carrying the IDENTICAL shifted geometry (same d, same corr) — an
apples-to-apples same-mesh-with-shift comparison.

WHAT TO EXPECT (verified on gpubox, level 4, Re=40, offset=0.05, dt=0.02):

    dmax = 0.0500   (0 < dmax < h=0.0625;  dmax/h = 0.80 -> a real sub-cell shift)
    projection Cd = +1.5407    monolithic Cd = +1.5457    rel-diff = 0.32%
    projection mean|u| = 0.9090  monolithic 0.8527         rel-diff = 6.6%

ANTI-VACUITY (the shift is LOAD-BEARING): ZEROING the shift (`geo.d=0`,
`geo.corr=1`) throws the projection well off the TRUE shifted oracle (~tens of
percent at this offset). Pass `--zero-shift` to see the match break — proof the
Taylor `(grad N).d` term and the area correction are actually doing the work.

KNOBS:
  * LEVEL   — mesh refinement.
  * RE      — Reynolds number (40 steady).
  * OFFSET  — sub-cell center shift (0 => back to body-fitted; must stay < h).

Run:
    PYTHONPATH=src:tests:examples python examples/ns_sbm/flow_past_cylinder.py
"""
import dataclasses
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))

from ladder_fixtures import build_square_channel_2d          # noqa: E402
from ladder_rungB_square_nitsche import (                    # noqa: E402
    march_projection, march_monolithic, ALPHA)

HALF = 0.125
DEFAULT_OFFSET = 0.05


def zero_shift(fx):
    """Return a fixture COPY with the SBM shift zeroed (geo.d=0, geo.corr=1) —
    everything else identical. Running with this and comparing against the TRUE
    shifted oracle must FAIL: the shift is doing real work at d!=0."""
    geo0 = dataclasses.replace(fx["geo"], d=np.zeros_like(fx["geo"].d),
                               corr=np.ones_like(fx["geo"].corr))
    fx0 = dict(fx)
    fx0["geo"] = geo0
    fx0["dmax"] = 0.0
    return fx0


def run(level=4, Re=40, offset=DEFAULT_OFFSET, do_zero_shift=False,
        device="cpu", dt=0.02, nsteps=600, alpha=ALPHA):
    fx = build_square_channel_2d(level, Re, half=HALF, offset=offset,
                                 device=device)
    h = 1.0 / 2 ** level
    D = 2.0 * HALF
    assert fx["dmax"] > 0, (f"expected a genuine SBM shift (dmax>0); got "
                            f"dmax={fx['dmax']} (offset={offset} too small)")
    assert fx["dmax"] < h, f"want a sub-cell shift (dmax<h={h}); got {fx['dmax']}"
    corr = np.asarray(fx["geo"].corr)
    n_corr = int((np.abs(corr - 1.0) > 1e-9).sum())
    print(f"\n=== Flow past body — SBM SHIFT  Re={Re}  level={level}  "
          f"offset={offset} ===")
    print(f" dmax={fx['dmax']:.5f}  (0<dmax<h={h:.5f}, dmax/h={fx['dmax']/h:.3f})"
          f"  area-corrected GPs={n_corr}")

    # The monolithic oracle ALWAYS uses the true shifted geometry (fx).
    mo = march_monolithic(fx, dt=dt, nsteps=nsteps, rate_tol=2e-4,
                          backflow_beta=0.5, boundary_vorticity=True,
                          alpha=alpha)
    # The projection may optionally ZERO the shift (anti-vacuity break).
    fx_proj = zero_shift(fx) if do_zero_shift else fx
    pr = march_projection(fx_proj, dt=dt, nsteps=nsteps, rate_tol=5e-4,
                          alpha=alpha, rot_pin_wall=True)

    cd_rel = abs(pr["cd"] - mo["cd"]) / abs(mo["cd"])
    mu_rel = abs(pr["mean_u"] - mo["mean_u"]) / abs(mo["mean_u"])
    tag = " (SHIFT ZEROED — expect a broken match)" if do_zero_shift else ""
    print(f" projection{tag}: Cd={pr['cd']:+.4f}  mean|u|={pr['mean_u']:.4f}  "
          f"steps={pr['steps']}")
    print(f" monolithic (true shift): Cd={mo['cd']:+.4f}  "
          f"mean|u|={mo['mean_u']:.4f}  steps={mo['steps']}")
    print(f" Cd rel-diff vs TRUE shifted oracle = {cd_rel:.3%}")
    print(f" mean|u| rel-diff                    = {mu_rel:.3%}")
    return dict(dmax=fx["dmax"], cd_proj=pr["cd"], cd_mono=mo["cd"],
                cd_rel=cd_rel, mean_u_proj=pr["mean_u"],
                mean_u_mono=mo["mean_u"], mu_rel=mu_rel,
                zero_shift=do_zero_shift)


if __name__ == "__main__":
    # KNOBS: python flow_past_cylinder.py [LEVEL] [RE] [--zero-shift]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_zero = "--zero-shift" in sys.argv
    level = int(args[0]) if len(args) > 0 else 4
    Re = int(args[1]) if len(args) > 1 else 40
    run(level=level, Re=Re, do_zero_shift=do_zero)
