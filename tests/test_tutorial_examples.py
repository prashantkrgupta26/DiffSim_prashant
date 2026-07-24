r"""Tests for the NS-SBM tutorial examples (docs/tutorials/ns-sbm-tutorial.md,
examples/ns_sbm/*.py).

Each test runs the corresponding worked example on a small/fast fixture and
asserts the key result against its VERIFIED reference (numbers measured on
gpubox; see docs/tutorials/ns-sbm-tutorial.md Sections 7-8). The fast legs run
in CI (CPU/splu); the level-4 3-D monolithic sphere drag is a slow opt-in
(DIFFSIM_SLOW=1) because splu is ~seconds/step there.

Reference numbers (gpubox, CPU/splu, verified 2026-07-24):

  lid-driven cavity  L4 Re100 200 steps:
      max|proj - mono| = 0.0486   max|proj - Ghia| = 0.0164
  flow past square   L4 Re40 (weak Nitsche, d=0):
      proj Cd = +1.3529   mono Cd = +1.3941   rel = 2.96%
  flow past cylinder L4 Re40 (SBM shift, offset=0.05, dmax=0.05):
      proj Cd = +1.5407   mono Cd = +1.5457   rel = 0.32%
  sphere 3-D         L4 Re100 alpha=10 (monolithic):  Cd = +0.381
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples",
                                "ns_sbm"))


# --------------------------------------------------------------------------
# 7.1 Lid-driven cavity (Ghia)
# --------------------------------------------------------------------------
def test_cavity_ghia(device):
    """Both engines track the Ghia centerline and each other on the same mesh."""
    import lid_driven_cavity as ex
    r = ex.run(level=4, Re=100, dt=0.05, nsteps=200, device=device)
    # projection matches the same-mesh monolithic (coarse-mesh band)
    assert r["d_proj_mono"] < 0.08, r
    # both track the Ghia 129^2 table within coarse-mesh tolerance
    assert r["d_proj_ghia"] < 0.05, r
    assert r["d_mono_ghia"] < 0.08, r
    # divergence finite and bounded (weak-divergence VMS scheme, not a blow-up)
    assert np.isfinite(r["div_proj"]) and r["div_proj"] < 5.0, r
    # spot-check the lid-adjacent station tracks Ghia's +0.841
    assert abs(r["u_proj"][1] - 0.841) < 0.05, r


# --------------------------------------------------------------------------
# 7.2 Flow past a square — weak Nitsche, d = 0
# --------------------------------------------------------------------------
def test_square_weak_nitsche(device):
    """Weak-Nitsche projection matches the same-mesh weak-Nitsche monolithic
    (Cd and mean|u|), the validated rung-B result."""
    import flow_past_square as ex
    r = ex.run(mode="weak", level=4, Re=40, device=device)
    assert not r["blew_up"], r
    # verified: proj Cd +1.3529 vs mono +1.3941 (rel 2.96%)
    assert abs(r["cd_proj"] - 1.3529) < 0.05, r
    assert r["cd_rel"] < 0.15, r
    assert r["mu_rel"] < 0.20, r


# --------------------------------------------------------------------------
# 7.3 Flow past body — genuine SBM shift, d != 0
# --------------------------------------------------------------------------
def test_cylinder_sbm_shift(device):
    """Consistent-projection + genuine SBM shift matches the same-mesh-with-shift
    monolithic (rung C). The shift is load-bearing (see the zero-shift break)."""
    import flow_past_cylinder as ex
    r = ex.run(level=4, Re=40, offset=0.05, device=device)
    # the shift is genuinely active
    assert r["dmax"] > 0.0 and r["dmax"] < 0.0625, r
    # verified: proj Cd +1.5407 vs mono +1.5457 (rel 0.32%)
    assert abs(r["cd_proj"] - 1.5407) < 0.05, r
    assert r["cd_rel"] < 0.15, r
    assert r["mu_rel"] < 0.20, r


@pytest.mark.skipif(not os.environ.get("DIFFSIM_SLOW"),
                    reason="anti-vacuity zero-shift break (extra full march); "
                           "set DIFFSIM_SLOW=1")
def test_cylinder_shift_is_load_bearing(device):
    """ANTI-VACUITY: zeroing the SBM shift (geo.d=0, geo.corr=1) breaks the match
    against the TRUE shifted oracle — the Taylor (grad N).d + area correction are
    doing real work."""
    import flow_past_cylinder as ex
    good = ex.run(level=4, Re=40, offset=0.05, do_zero_shift=False, device=device)
    broke = ex.run(level=4, Re=40, offset=0.05, do_zero_shift=True, device=device)
    assert good["cd_rel"] < 0.15, good
    # zeroing the shift throws the projection materially off the true oracle
    assert broke["cd_rel"] > good["cd_rel"], (good, broke)


# --------------------------------------------------------------------------
# 8 Sphere 3-D — pipeline invariants (fast) + monolithic drag (slow)
# --------------------------------------------------------------------------
def test_sphere_pipeline_invariants(device):
    """The 3-D projection+SBM COMPOSITION is de-risked: finite, axisymmetric,
    BDF2 engages (a short window — NOT the unstable long-time transient). Level 3
    is fast/small."""
    import sphere_3d as ex
    from p2r0_task10_sphere_derisk import build_sphere_3d
    fx = build_sphere_3d(device, level=3, Re=100)
    inv = ex.check_pipeline_invariants(fx, alpha=10.0, dt=0.05, nsteps=4)
    assert inv["finite"], inv
    assert inv["bdf2"], inv
    assert inv["axisym"], inv


@pytest.mark.skipif(not os.environ.get("DIFFSIM_SLOW"),
                    reason="level-4 3-D monolithic splu march (~seconds/step); "
                           "set DIFFSIM_SLOW=1")
def test_sphere_monolithic_drag(device):
    """The 3-D MONOLITHIC SBM-NS drag is stable and physical: Cd = +0.381 at
    Re=100, alpha=10, level 4 (reproducing the M1b lock)."""
    import sphere_3d as ex
    r = ex.run(level=4, Re=100, alpha=10.0, max_steps=60, device=device)
    assert r["cd_mono"] > 0.0, r                    # positive (physical)
    assert abs(r["cd_mono"] - 0.381) < 0.05, r      # matches the M1b lock


if __name__ == "__main__":
    # allow `python tests/test_tutorial_examples.py` as a quick smoke.
    class _D:
        pass
    dev = "cpu"
    test_cavity_ghia(dev)
    test_square_weak_nitsche(dev)
    test_cylinder_sbm_shift(dev)
    test_sphere_pipeline_invariants(dev)
    print("tutorial example smoke: OK")
