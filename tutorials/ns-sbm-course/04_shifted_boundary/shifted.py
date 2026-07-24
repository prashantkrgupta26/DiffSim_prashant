"""NS-SBM course — flow past a shifted body core (Chapter 04, genuine SBM d!=0).

The Shifted Boundary Method lets the immersed body sit OFF the mesh grid: the
true boundary Gamma is approximated by a grid-aligned SURROGATE boundary, and a
Taylor shift `S N_a = N_a + (grad N_a).d` transfers the boundary condition from
the surrogate to the true wall (d = surrogate-GP -> closest point on Gamma).

We carve a square whose center is offset by a SUB-CELL fraction (offset=0.05),
so the grid-aligned surrogate no longer coincides with the true `Box` face:
`0 < dmax < h`. Now the shift terms do REAL work — the Taylor `(grad N).d` term
in the Nitsche block AND the area correction `geo.corr = n_tilde . n` in the
surrogate traction (unlike Chapter 03 where d=0 and corr=1, the shift is inert).

We march the consistent-projection engine (weak Nitsche + SBM shift) and compare
Cd against the same-mesh MONOLITHIC oracle carrying the IDENTICAL shifted
geometry — an apples-to-apples same-mesh-with-shift comparison.

Anti-vacuity: zeroing the shift (geo.d=0, geo.corr=1) throws the projection off
the TRUE shifted oracle — proof the Taylor term and area correction are real.

This CURATES the validated ladder fixtures/marchers (tests/ladder_*).
"""
from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np

_TESTS = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                     "..", "..", "..", "tests"))
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

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


def run_shifted(level=4, Re=40, offset=DEFAULT_OFFSET, do_zero_shift=False,
                device="cpu", dt=0.02, nsteps=600, alpha=ALPHA):
    """Build the SHIFTED square-in-channel fixture (0<dmax<h) and compare the
    consistent-projection split against the same-mesh-with-shift monolithic."""
    fx = build_square_channel_2d(level, Re, half=HALF, offset=offset,
                                 device=device)
    h = 1.0 / 2 ** level
    D = 2.0 * HALF
    assert fx["dmax"] > 0, (f"expected a genuine SBM shift (dmax>0); got "
                            f"dmax={fx['dmax']} (offset={offset} too small)")
    assert fx["dmax"] < h, f"want a sub-cell shift (dmax<h={h}); got {fx['dmax']}"
    corr = np.asarray(fx["geo"].corr)
    n_corr = int((np.abs(corr - 1.0) > 1e-9).sum())

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
    return dict(
        level=level, Re=Re, offset=offset, D=D, h=h,
        dmax=fx["dmax"], dmax_over_h=fx["dmax"] / h,
        area_corrected_gps=n_corr,
        cd_proj=pr["cd"], cd_mono=mo["cd"], cd_rel=cd_rel,
        mean_u_proj=pr["mean_u"], mean_u_mono=mo["mean_u"], mu_rel=mu_rel,
        zero_shift=bool(do_zero_shift),
        steps_proj=int(pr["steps"]), steps_mono=int(mo["steps"]))
