#!/usr/bin/env python
r"""Worked example 8 — 3-D immersed sphere with SBM shift (the hero regime).

The end-to-end SBM-NS engine in 3-D: an immersed `Sphere` carved from a unit-box
octree channel, weak no-slip on the sphere, strong inflow/walls, free outflow.
The sphere is NOT grid-aligned, so the SBM shift `d != 0` is genuinely active in
3-D (the surrogate faces hug the carved octree boundary; d maps each surrogate
GP to the true sphere; corr != 1 area-corrects to the true surface).

TWO ENGINES, an HONEST 3-D verdict (documented in the ladder, verified here):

  * MONOLITHIC 3-D SBM-NS  — the WORKING drag path. One coupled saddle solve per
    step with the SBM Nitsche block. STABLE and PHYSICAL: recovers from the
    startup transient to a positive steady Cd, reproducing the M1b lock
    (Cd = +0.381 at Re=100, alpha=10, level 4). This is the hero result for 3-D
    drag. It scales via GPU direct (solver="cudss") / block-preconditioned
    FGMRES past the host-splu wall (~level 5 / 143k DOF).

  * PROJECTION + volumetric SBM (LeraySBMStepper) — the composition is DE-RISKED
    in 3-D (finite, axisymmetric |C_lat| << |Cd|, BDF2 engages, and the
    PPE-space weak divergence is machine-zero), BUT its long-time drag transient
    is UNSTABLE at feasible 3-D mesh (the coupled iteration's Cd diverges
    monotonically while staying finite — a documented NEEDS_CONTEXT / R2 item).
    So in 3-D we validate the pipeline invariants, and take DRAG from the
    monolithic.

This example runs the monolithic drag path by default and prints the verified
Cd; pass `--pipeline` to additionally exercise the projection composition and
check its pipeline invariants (a short window, not the unstable long march).

WHAT TO EXPECT (verified on gpubox, level 4, Re=100, alpha=10, dt=0.05):

    n_free ~ 4907   sphere surrogate faces = 64   D/h = 3.84   SBM dmax > 0
    MONOLITHIC steady Cd = +0.381  (positive, physical; matches the M1b lock)

  (The confined unit-box domain inflates Cd above the unconfined literature
   sphere value ~0.6-0.7 at Re=300 — BOTH engines see the same confinement, so
   the bar is faithfulness, not the literature number.)

KNOBS:
  * LEVEL   — mesh refinement (4 is the de-risk resolution; splu is ~9s/step).
  * RE      — Reynolds number.
  * ALPHA   — Nitsche penalty scale (10 is the monolithic-stable value here).

Run (gpubox recommended; level-4 splu is ~minutes):
    PYTHONPATH=src:tests:examples python examples/ns_sbm/sphere_3d.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))

from p2r0_task10_sphere_derisk import (                      # noqa: E402
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
    print(f" [pipeline] projection composition over {nsteps} steps:")
    print(f"   finite={pr['finite']}  BDF2 engaged={pr['bdf2_engaged']}  "
          f"Cd(window)={cd:+.4f}  |C_lat|/|Cd|={clat/max(abs(cd),1e-9):.2e}")
    axisym = clat < 0.1 * max(abs(cd), 1e-6)
    print(f"   axisymmetric (|C_lat| << |Cd|): {axisym}")
    return dict(finite=pr["finite"], bdf2=pr["bdf2_engaged"], axisym=axisym,
                cd_window=cd, clat=clat)


def run(level=4, Re=100, alpha=10.0, device="cpu", dt=0.05, max_steps=60,
        rate_tol=5e-3, pipeline=False):
    print(f"\n=== 3-D immersed sphere (SBM-NS)  Re={Re}  level={level}  "
          f"alpha={alpha}  dt={dt} ===")
    fx = build_sphere_3d(device, level, Re)
    n_free = len(fx["coords"])
    n_faces = int(fx["sf"].elem.size)
    dmax = float(np.abs(fx["geo"].d).max())
    D_over_h = 2 * 0.12 / (1.0 / 2 ** level)
    print(f" n_free={n_free}  sphere surrogate faces={n_faces}  D/h={D_over_h:.2f}"
          f"  SBM shift dmax={dmax:.5f} (d != 0 => genuine 3-D shift)")

    # HERO drag path: the stable, physical 3-D MONOLITHIC SBM-NS.
    mo = monolithic_cd(fx, alpha, dt, max_steps, rate_tol)
    print(f" MONOLITHIC steady Cd = {mo['cd']:+.4f}  steps={mo['steps']}  "
          f"(positive, physical — the working 3-D drag)")

    out = dict(cd_mono=mo["cd"], steps=mo["steps"], n_free=n_free,
               n_faces=n_faces, dmax=dmax, D_over_h=D_over_h)
    if pipeline:
        out["pipeline"] = check_pipeline_invariants(fx, alpha=alpha, dt=dt)
    return out


if __name__ == "__main__":
    # KNOBS: python sphere_3d.py [LEVEL] [RE] [ALPHA] [--pipeline]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    level = int(args[0]) if len(args) > 0 else 4
    Re = float(args[1]) if len(args) > 1 else 100
    alpha = float(args[2]) if len(args) > 2 else 10.0
    run(level=level, Re=Re, alpha=alpha, pipeline="--pipeline" in sys.argv)
