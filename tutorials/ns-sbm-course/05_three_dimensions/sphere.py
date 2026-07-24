"""NS-SBM course — 3-D immersed sphere core (Chapter 05, the hero regime).

The end-to-end SBM-NS engine in 3-D: an immersed `Sphere` carved from a unit-box
octree channel, weak no-slip on the sphere, strong inflow/walls, free outflow.
The sphere is NOT grid-aligned, so the SBM shift d != 0 is genuinely active in
3-D.

TWO ENGINES, an HONEST 3-D verdict:
  * MONOLITHIC 3-D SBM-NS  — the WORKING drag path. Stable and physical:
    recovers from the startup transient to a positive steady Cd (= +0.381 at
    Re=100, alpha=10, level 4, reproducing the M1b lock). It scales via GPU
    direct (solver="cudss") / block-preconditioned FGMRES past the host-splu
    wall (~level 5 / 143k DOF).
  * PROJECTION + volumetric SBM (LeraySBMStepper) — the composition is
    DE-RISKED (finite, axisymmetric |C_lat| << |Cd|, BDF2 engages, PPE-space
    weak divergence machine-zero), BUT its long-time drag transient is UNSTABLE
    at feasible 3-D mesh. So in 3-D we validate the pipeline invariants and take
    DRAG from the monolithic.

This CURATES the validated sphere de-risk fixture (tests/p2r0_task10_*).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_TESTS = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                     "..", "..", "..", "tests"))
if _TESTS not in sys.path:
    sys.path.insert(0, _TESTS)

from p2r0_task10_sphere_derisk import (                       # noqa: E402
    build_sphere_3d, march_projection, monolithic_cd, qref)


def check_pipeline_invariants(fx, alpha=10.0, dt=0.05, nsteps=4):
    """Exercise the projection+SBM COMPOSITION over a short window and report
    the de-risk invariants: finite (u, p), axisymmetric traction, BDF2 engaged.
    (Does NOT march to the unstable long-time transient.)"""
    pr = march_projection(fx, alpha, dt, nsteps, rate_tol=-1.0,
                          order=2, beta_backflow=1.0, picard_iters=2)
    q = qref()
    st = pr["st"]
    F = st.surrogate_traction()
    cd = F[0] / q
    clat = float(np.hypot(F[1], F[2]) / q)
    axisym = clat < 0.1 * max(abs(cd), 1e-6)
    return dict(finite=bool(pr["finite"]), bdf2=bool(pr["bdf2_engaged"]),
                axisym=bool(axisym), cd_window=float(cd), clat=clat,
                clat_ratio=float(clat / max(abs(cd), 1e-9)))


def run_sphere(level=4, Re=100, alpha=10.0, device="cpu", dt=0.05,
               max_steps=60, rate_tol=5e-3, pipeline=False):
    """Build the 3-D sphere fixture and march the MONOLITHIC drag path (the hero
    number). Optionally check the projection composition invariants."""
    fx = build_sphere_3d(device, level, Re)
    n_free = len(fx["coords"])
    n_faces = int(fx["sf"].elem.size)
    dmax = float(np.abs(fx["geo"].d).max())
    D_over_h = 2 * 0.12 / (1.0 / 2 ** level)

    mo = monolithic_cd(fx, alpha, dt, max_steps, rate_tol)
    out = dict(level=level, Re=Re, alpha=alpha,
               cd_mono=float(mo["cd"]), steps=int(mo["steps"]),
               n_free=n_free, n_faces=n_faces, dmax=dmax,
               D_over_h=float(D_over_h))
    if pipeline:
        out["pipeline"] = check_pipeline_invariants(fx, alpha=alpha, dt=dt)
    return out
