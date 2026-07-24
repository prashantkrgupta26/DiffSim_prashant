"""Task 7 fast asserted case — the 3-D CUBE confirmation (rung C').

The 2-D ladder is COMPLETE: the consistent-projection + SBM scheme is faithful to
the same-mesh monolithic across projection (rung A), weak Nitsche (rung B), and the
genuine SBM shift (rung C). Task 7 confirms the SAME working scheme lifts to 3-D —
the original sphere regime — with the 2-D config UNCHANGED, using cuDSS on the GPU.

This test locks in the DECISIVE 3-D rung: C' — the OFFSET cube with a GENUINE SBM
SHIFT (dmax>0). It is the 3-D analogue of the 2-D rung C, and it is the concept
that matters for the sphere regime (weak Nitsche + Taylor shift + area correction).

MEASURED VERDICT (level 4, Re=40, cuDSS/GPU — this test locks it as a regression):
  * RUNG C' PROJECTION+SHIFT vs MONOLITHIC+SHIFT (same offset mesh, same shifted
    geo): the consistent-projection split with the shift ACTIVE matches the
    same-mesh-with-shift monolithic Cd to ~9.6% (< the 15% R0 tol) and tracks
    mean|u| to ~4% — adding the Taylor (grad N).d shift + the area correction
    PRESERVES the faithfulness the 2-D ladder established, IN 3-D.
  * SHIFT-ACTIVE guard (anti-vacuity, OPPOSITE of A''s dmax==0): 0 < dmax < h.
  * SHIFT LOAD-BEARING (anti-vacuity): ZEROING the shift (geo.d=0, geo.corr=1)
    throws the projection ~90% off the TRUE shifted monolithic oracle (>> 15%) —
    the shift terms are genuinely doing work at d!=0 in 3-D.

RUNG A' (body-fitted cube, STRONG no-slip) is a documented FINDING, not asserted
as PASS: it reproduces the 2-D rung-A wall-pressure drag consistency gap — the
strong-Dirichlet-wall projection Cd sits ~18% off the same-mesh monolithic at L4
(2-D rung A was ~11%), while mean|u| matches to ~1.6%. The FN4 rotational wall
pin that closes this gap is a WEAK-wall fix (rungs B/C/C'), so it does not apply
to the strong-wall A'. The velocity field is faithful; the strong-wall drag
carries the same intrinsic split-vs-saddle wall-pressure gap seen in 2-D. See
tests/ladder_rung3d_cube.py::run_rungA3d and .superpowers/sdd/task-7-report.md.

GPU-only: cuDSS direct solve on cuda:0 (3-D at L4 ~19.5k saddle DOF is where the
direct-GPU solve pays off). Skips when Warp/CUDA is unavailable.

Run (gpubox GPU0):
    CUDA_VISIBLE_DEVICES=0 \
    LD_LIBRARY_PATH=/usr/lib/wsl/lib:$PWD/.venv/lib/python3.12/site-packages/nvidia/cu12/lib \
    PYTHONPATH=src:tests .venv/bin/pytest -q tests/test_ladder_rung3d.py
"""
import numpy as np
import pytest

try:
    import warp as wp
    wp.init()
    _HAS_CUDA = wp.get_cuda_device_count() > 0
except Exception:
    wp = None
    _HAS_CUDA = False

pytestmark = pytest.mark.skipif(
    not _HAS_CUDA, reason="Task-7 3-D rung needs cuDSS on a CUDA device")

from ladder_fixtures import build_cube_channel_3d
from ladder_rungB_square_nitsche import march_projection, TOL_CD_REL, ALPHA
from ladder_rung3d_cube import (march_monolithic_3d, zero_shift, _recd,
                                qref3d)

LEVEL = 4
HALF = 0.125          # k/2^level aligned at level>=3; offset breaks the alignment
OFFSET = 0.05         # dmax = 0.05, dmax/h = 0.80 (genuine sub-cell shift)
RE = 40               # steady, below shedding onset
DT = 0.02
NSTEPS = 300          # both marches converge (rate_tol) well inside this
DEVICE = "cuda:0"
SOLVER = "cudss"


@pytest.fixture(scope="module")
def rungC3d():
    fx = build_cube_channel_3d(LEVEL, RE, half=HALF, offset=OFFSET,
                               device=DEVICE)
    # MONOLITHIC weak-Nitsche + SHIFT same-mesh oracle (3-D, cuDSS). qref3d.
    mo = march_monolithic_3d(fx, dt=DT, nsteps=NSTEPS, rate_tol=2e-4,
                             log_every=0, solver=SOLVER, device=DEVICE,
                             strong_obstacle=False, alpha=ALPHA)
    # PROJECTION (consistent scheme) weak-Nitsche + SHIFT + FN4 wall pin — rung
    # B's marcher VERBATIM. Cd comes back on the 2-D length qref => rescale to
    # the 3-D area qref (divide by D) for the apples-to-apples comparison.
    pr = march_projection(fx, dt=DT, nsteps=NSTEPS, rate_tol=5e-4, alpha=ALPHA,
                          solver=SOLVER, rot_pin_wall=True)
    pr["cd"] = _recd(pr["cd"], fx, HALF)
    # SHIFT-ZEROED projection (drops Taylor (grad N).d AND area corr) — must NOT
    # match the TRUE shifted oracle.
    fx0 = zero_shift(fx)
    pr0 = march_projection(fx0, dt=DT, nsteps=NSTEPS, rate_tol=5e-4,
                           alpha=ALPHA, solver=SOLVER, rot_pin_wall=True)
    pr0["cd"] = _recd(pr0["cd"], fx, HALF)
    return dict(fx=fx, mo=mo, pr=pr, pr_zero=pr0)


def test_shift_active_guard(rungC3d):
    """ANTI-VACUITY (OPPOSITE of A''s dmax==0): the offset cube carve gives a
    GENUINE sub-cell SBM shift — 0 < dmax < h — so the Taylor (grad N).d term
    and the area correction geo.corr are load-bearing, AND the obstacle
    surrogate set is non-empty (the weak no-slip is actually imposed)."""
    fx = rungC3d["fx"]
    h = 1.0 / 2 ** LEVEL
    assert fx["dim"] == 3
    assert fx["dmax"] > 0, "offset did not break the body-fitted alignment"
    assert fx["dmax"] < h, "shift is not sub-cell (dmax >= h)"
    assert int(fx["obstacle_node_mask"].sum()) > 0
    corr = np.asarray(fx["geo"].corr)
    assert int((np.abs(corr - 1.0) > 1e-9).sum()) > 0


def test_projection_with_shift_matches_monolithic_3d(rungC3d):
    """THE RUNG-C' VERDICT (3-D): the consistent-projection split WITH the
    genuine SBM shift active (dmax>0) matches the same-mesh-with-shift
    weak-Nitsche monolithic Cd (within the R0 15% tol) and tracks mean|u| — the
    2-D projection+SBM conclusion LIFTS to 3-D (the sphere regime)."""
    pr, mo = rungC3d["pr"], rungC3d["mo"]
    assert not pr.get("blew_up", False), "projection-with-shift blew up (3-D)"
    assert np.isfinite(pr["cd"]) and np.isfinite(pr["mean_u"])
    cd_rel = abs(pr["cd"] - mo["cd"]) / abs(mo["cd"])
    assert cd_rel < TOL_CD_REL, (
        f"3-D projection-with-shift Cd={pr['cd']:+.4f} does NOT match the "
        f"same-shifted-mesh monolithic Cd={mo['cd']:+.4f} (rel {cd_rel:.3%}, "
        f"tol {TOL_CD_REL:.0%}) — the SBM shift broke faithfulness in 3-D.")
    mu_frac = pr["mean_u"] / mo["mean_u"]
    assert 0.7 < mu_frac < 1.3, (
        f"3-D projection-with-shift mean|u|={pr['mean_u']:.4f} does not track "
        f"monolithic {mo['mean_u']:.4f} (frac {mu_frac:.3f}).")


def test_shift_is_load_bearing_3d(rungC3d):
    """ANTI-VACUITY: the shift terms must be load-bearing at d!=0 in 3-D too.
    ZEROING the shift (geo.d=0 AND geo.corr=1) throws the projection OFF the TRUE
    shifted monolithic oracle by MORE than the tol — the shift is doing real
    work (not a body-fitted degenerate case in disguise)."""
    pr_zero, mo = rungC3d["pr_zero"], rungC3d["mo"]
    cd_rel = abs(pr_zero["cd"] - mo["cd"]) / abs(mo["cd"])
    assert cd_rel > TOL_CD_REL, (
        f"3-D shift-zeroed projection Cd={pr_zero['cd']:+.4f} unexpectedly "
        f"matches the TRUE shifted monolithic Cd={mo['cd']:+.4f} (rel "
        f"{cd_rel:.3%}) — the SBM shift terms are NOT load-bearing at d!=0.")
