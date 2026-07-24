"""Rung C fast asserted case — OFFSET square, GENUINE SBM SHIFT.

Rung B (weak Nitsche on the EXACT body-fitted square, dmax==0, the shift inert)
PASSES: the consistent-projection split + FN4 rotational wall pin is faithful to
the same-mesh weak-Nitsche monolithic in velocity AND drag. Rung C OFFSETS the
square center by a SUB-CELL fraction (offset=0.05) so the grid-aligned surrogate
boundary no longer coincides with the true Box face: 0 < dmax < h. This ACTIVATES
the genuine SBM Taylor shift — the (grad N_a).d term in the Nitsche block AND the
area-correction geo.corr in the surrogate traction now do real work.

MEASURED RUNG-C VERDICT (this test locks it in as a regression):
  * PROJECTION+SHIFT vs MONOLITHIC+SHIFT (same offset mesh, same shifted geo) —
    the consistent-projection split with the shift ACTIVE matches the same-mesh-
    with-shift monolithic Cd to ~0.3% and tracks mean|u| — adding the Taylor
    shift + area correction PRESERVES the faithfulness rung B established. The
    drivers/steppers are UNCHANGED from rung B (the shift is fixture DATA in
    geo.d/geo.corr, not solver code) — rung C reuses march_projection /
    march_monolithic verbatim.
  * SHIFT-ACTIVE guard (anti-vacuity, the OPPOSITE of rung A/B's dmax==0):
    dmax>0 and 0<dmax<h — the shift is genuinely on.
  * SHIFT LOAD-BEARING (anti-vacuity): ZEROING the shift (geo.d=0 AND geo.corr=1,
    dropping the Taylor (grad N_a).d term AND the area correction) throws the
    projection ~36% off the TRUE shifted monolithic oracle (>> the 15% tol) —
    the shift terms are actually doing work at d!=0.
  * RUNG C ~= RUNG B (sub-cell shift ~= same body): the offset Cd sits within the
    shift's consistency error of the body-fitted (offset=0) Cd — a small
    correction, NOT a regime change.

DECISIVE READ: projection+Nitsche works (rung B); ADDING the genuine SBM shift
(rung C) preserves faithfulness => projection+SBM works end-to-end in 2-D.

Kept CHEAP (level 4, short march). Full Re-40/100 numbers live in the driver
(tests/ladder_rungC_square_shift.py).
"""
import numpy as np
import pytest

from ladder_fixtures import build_square_channel_2d
from ladder_rungB_square_nitsche import (march_projection, march_monolithic,
                                         TOL_CD_REL, ALPHA)
from ladder_rungC_square_shift import zero_shift, DEFAULT_OFFSET

LEVEL = 4
HALF = 0.125          # k/2^level aligned at level>=3; offset breaks the alignment
OFFSET = DEFAULT_OFFSET   # 0.05 => dmax = 0.05, dmax/h = 0.80 (genuine sub-cell shift)
RE = 40               # steady, below square-cylinder shedding onset
DT = 0.02
NSTEPS = 400


@pytest.fixture(scope="module")
def rungC_results():
    fx = build_square_channel_2d(LEVEL, RE, half=HALF, offset=OFFSET,
                                 device="cpu")
    # MONOLITHIC weak-Nitsche + SHIFT same-mesh oracle (the shifted bar).
    mo = march_monolithic(fx, dt=DT, nsteps=NSTEPS, rate_tol=2e-4,
                          backflow_beta=0.5, boundary_vorticity=True,
                          alpha=ALPHA)
    # PROJECTION (consistent scheme) weak-Nitsche + SHIFT + FN4 wall pin — the
    # decisive test (same config as rung B's verdict; shift enters via geo).
    pr = march_projection(fx, dt=DT, nsteps=NSTEPS, rate_tol=5e-4, alpha=ALPHA,
                          rot_pin_wall=True)
    # SHIFT-ZEROED projection (drops the Taylor (grad N).d AND the area corr) —
    # anti-vacuity: this must NOT match the TRUE shifted oracle `mo`.
    fx0 = zero_shift(fx)
    pr_zero = march_projection(fx0, dt=DT, nsteps=NSTEPS, rate_tol=5e-4,
                               alpha=ALPHA, rot_pin_wall=True)
    # RUNG B reference (offset=0, shift inert) — for the rung-C ~= rung-B check.
    fxB = build_square_channel_2d(LEVEL, RE, half=HALF, offset=0, device="cpu")
    moB = march_monolithic(fxB, dt=DT, nsteps=NSTEPS, rate_tol=2e-4,
                           backflow_beta=0.5, boundary_vorticity=True,
                           alpha=ALPHA)
    return dict(fx=fx, mo=mo, pr=pr, pr_zero=pr_zero, moB=moB)


def test_shift_active_guard(rungC_results):
    """ANTI-VACUITY (the OPPOSITE of rung A/B's dmax==0): the offset carve gives
    a GENUINE sub-cell SBM shift — 0 < dmax < h — so the Taylor (grad N).d term
    and the area correction geo.corr are actually load-bearing, AND the obstacle
    surrogate set is non-empty (the weak no-slip is actually imposed)."""
    fx = rungC_results["fx"]
    h = 1.0 / 2 ** LEVEL
    assert fx["dmax"] > 0, "offset did not break the body-fitted alignment"
    assert fx["dmax"] < h, "shift is not sub-cell (dmax >= h)"
    assert int(fx["obstacle_node_mask"].sum()) > 0
    # the area correction is genuinely non-trivial (drops falsely-intersected
    # surrogate faces) — corr != 1 somewhere.
    corr = np.asarray(fx["geo"].corr)
    assert int((np.abs(corr - 1.0) > 1e-9).sum()) > 0


def test_projection_with_shift_matches_monolithic(rungC_results):
    """THE RUNG-C VERDICT: the consistent-projection split WITH the genuine SBM
    shift active (dmax>0) matches the same-mesh-with-shift weak-Nitsche
    monolithic Cd (within the R0 tol) and tracks mean|u| — adding the Taylor
    shift + area correction PRESERVES the faithfulness rung B established.
    projection+SBM works end-to-end in 2-D."""
    pr, mo = rungC_results["pr"], rungC_results["mo"]
    assert not pr.get("blew_up", False), "projection-with-shift blew up"
    assert np.isfinite(pr["cd"]) and np.isfinite(pr["mean_u"])
    cd_rel = abs(pr["cd"] - mo["cd"]) / abs(mo["cd"])
    assert cd_rel < TOL_CD_REL, (
        f"projection-with-shift Cd={pr['cd']:+.4f} does NOT match same-shifted-"
        f"mesh monolithic Cd={mo['cd']:+.4f} (rel {cd_rel:.3%}, tol "
        f"{TOL_CD_REL:.0%}) — the SBM shift broke faithfulness.")
    mu_frac = pr["mean_u"] / mo["mean_u"]
    assert 0.7 < mu_frac < 1.3, (
        f"projection-with-shift mean|u|={pr['mean_u']:.4f} does not track "
        f"monolithic {mo['mean_u']:.4f} (frac {mu_frac:.3f}).")


def test_shift_is_load_bearing(rungC_results):
    """ANTI-VACUITY: the shift terms must be load-bearing at d!=0. ZEROING the
    shift (geo.d=0 AND geo.corr=1 — dropping the Taylor (grad N).d term AND the
    area correction) throws the projection OFF the TRUE shifted monolithic oracle
    by MORE than the tol — proving the shift is actually doing work (not a
    body-fitted degenerate case in disguise)."""
    pr_zero, mo = rungC_results["pr_zero"], rungC_results["mo"]
    cd_rel = abs(pr_zero["cd"] - mo["cd"]) / abs(mo["cd"])
    assert cd_rel > TOL_CD_REL, (
        f"shift-zeroed projection Cd={pr_zero['cd']:+.4f} unexpectedly matches "
        f"the TRUE shifted monolithic Cd={mo['cd']:+.4f} (rel {cd_rel:.3%}) — "
        f"the SBM shift terms are NOT load-bearing at d!=0 (dropping the Taylor "
        f"(grad N).d + area correction should change the answer).")


def test_rungC_approximates_rungB(rungC_results):
    """The sub-cell offset is nearly the same body — rung C (offset=0.05, shift
    active) should reproduce rung B (offset=0, shift inert) within the shift's
    consistency error: a SMALL correction, NOT a regime change. Compares the
    shifted monolithic Cd to the body-fitted monolithic Cd (same solver, same
    Re/level — the only difference is the sub-cell offset)."""
    mo, moB = rungC_results["mo"], rungC_results["moB"]
    cd_rel = abs(mo["cd"] - moB["cd"]) / abs(moB["cd"])
    assert cd_rel < 0.25, (
        f"rung-C shifted Cd={mo['cd']:+.4f} departs rung-B body-fitted Cd="
        f"{moB['cd']:+.4f} by {cd_rel:.3%} (> 25%) — the sub-cell shift is a "
        f"regime change, not a small consistency correction.")
