"""Rung B fast asserted case — body-fitted square, WEAK NITSCHE no-slip.

Rung A (strong Dirichlet on the exact body-fitted square) PASSED: the
consistent-projection split is FAITHFUL to the same-mesh monolithic. Rung B keeps
the SAME body-fitted mesh (aligned Box, offset=0 => dmax==0, so the SBM
shifted-Nitsche form S N_a = N_a DEGENERATES TO STANDARD NITSCHE exactly) and
swaps the obstacle BC treatment strong Dirichlet -> WEAK NITSCHE. This isolates
the Nitsche coupling on the already-repaired projection.

MEASURED RUNG-B VERDICT (post-FN1 fix, this test locks it in as a regression):
  * MONOLITHIC (same mesh, weak Nitsche) — CLEAN: converges to a bounded ‖div u‖,
    a physical steady Cd (~+1.4 at level 4, softer than the strong-Dirichlet
    ~+4.2 by the Nitsche consistency error), mean|u| ~ 0.86. The Nitsche block
    is correct.
  * The Nitsche penalty is LOAD-BEARING (anti-vacuity): zeroing it (alpha=0) in
    the CLEAN monolithic BREAKS the no-slip — Cd runs to ~+71, ‖div‖ ~ 21, and
    it never converges (the obstacle stops being felt).
  * PROJECTION (consistent scheme, weak Nitsche) — the FN1 fix makes it STABLE and
    VELOCITY-FAITHFUL: the bare split DIVERGED (|p*| past 1e6, mean|u| ~13x mono,
    seeded monolithic NOT a fixed point). The fix — grad-div (LSIC) stabilization
    (graddiv_gamma) + a correction-step Nitsche re-pin of the weak wall trace —
    arrests the growing interior-divergence mode: NO blow-up, ‖div‖ bounded to the
    monolithic's level, mean|u| tracks the monolithic to ~5%, |p*| bounded, and the
    SEEDED monolithic is a BOUNDED fixed point (ladder_rungB_seed_probe.py).
  * RESIDUAL (honest, NOT yet fixed): the drag Cd (dominated by the WALL PRESSURE-
    TRACTION) is NOT recovered — proj Cd ~ +0.36 vs mono ~+1.40 at level 4. The
    split's steady wall pressure differs from the monolithic saddle; the lagged
    wall pressure-traction term helps Cd when SEEDED but destabilizes from rest,
    and more grad-div drives Cd more negative. So weak-Nitsche projection is
    velocity-faithful but drag-unfaithful.

DECISIVE READ: strong Dirichlet on this mesh (rung A) is a stable fixed point of
the consistent projection; swapping strong -> WEAK NITSCHE BROKE it (divergence).
The FN1 grad-div + re-pin fix RESTORES stability and velocity-faithfulness (the
seeded monolithic is now a bounded fixed point), leaving the wall pressure-traction
(drag) as the open residual. The defect isolated to the Nitsche coupling (base
projection cleared by rung A; shift absent, dmax==0).

Kept CHEAP (level 4, short march). Full Re-40/100 numbers live in the driver
(tests/ladder_rungB_square_nitsche.py); the seed fixed-point gate lives in
tests/ladder_rungB_seed_probe.py.
"""
import numpy as np
import pytest

from ladder_fixtures import build_square_channel_2d
from ladder_rungB_square_nitsche import (march_projection, march_monolithic,
                                         TOL_CD_REL, ALPHA)

LEVEL = 4
HALF = 0.125          # k/2^level aligned at level>=3 -> dmax==0 (body-fitted)
RE = 40               # steady, below square-cylinder shedding onset
DT = 0.02
NSTEPS = 400


@pytest.fixture(scope="module")
def rungB_results():
    fx = build_square_channel_2d(LEVEL, RE, half=HALF, offset=0, device="cpu")
    # MONOLITHIC weak-Nitsche same-mesh oracle (the clean bar).
    mo = march_monolithic(fx, dt=DT, nsteps=NSTEPS, rate_tol=2e-4,
                          backflow_beta=0.5, boundary_vorticity=True,
                          alpha=ALPHA)
    # MONOLITHIC with the penalty REMOVED (anti-vacuity: no-slip must break).
    # graddiv_gamma=0 so the anti-vacuity signal isolates the NITSCHE PENALTY —
    # the FN1 grad-div would otherwise bound ‖div‖ even with no wall penalty,
    # masking the div balloon (the Cd departure alone stays diagnostic, but this
    # keeps the ‖div‖ check meaningful too).
    mo_nopen = march_monolithic(fx, dt=DT, nsteps=NSTEPS, rate_tol=2e-4,
                                backflow_beta=0.5, boundary_vorticity=True,
                                alpha=0.0, graddiv_gamma=0.0)
    # Penalized reference at graddiv_gamma=0 (matched to mo_nopen), so the
    # load-bearing test compares penalty-vs-no-penalty at the SAME grad-div.
    mo_g0 = march_monolithic(fx, dt=DT, nsteps=NSTEPS, rate_tol=2e-4,
                             backflow_beta=0.5, boundary_vorticity=True,
                             alpha=ALPHA, graddiv_gamma=0.0)
    # PROJECTION (consistent scheme) weak-Nitsche — the decisive test.
    pr = march_projection(fx, dt=DT, nsteps=NSTEPS, rate_tol=5e-4, alpha=ALPHA)
    return dict(fx=fx, mo=mo, mo_nopen=mo_nopen, mo_g0=mo_g0, pr=pr)


def test_body_fitted_guard(rungB_results):
    """Anti-vacuity: the carve is EXACT body-fitted (dmax==0) so this rung tests
    STANDARD NITSCHE (the shifted form degenerates, d=0) — NOT an SBM shift — and
    the obstacle surrogate set is non-empty (the weak no-slip is actually
    imposed)."""
    fx = rungB_results["fx"]
    assert fx["dmax"] == 0
    assert int(fx["obstacle_node_mask"].sum()) > 0


def test_monolithic_weak_nitsche_is_clean_oracle(rungB_results):
    """The same-mesh weak-Nitsche monolithic oracle develops the flow and is
    well-behaved: bounded ‖div u‖, physical positive steady Cd, mean|u| ~ U_IN.
    This confirms the SBM Nitsche block is CORRECT; it is the bar the projection
    must meet."""
    mo = rungB_results["mo"]
    assert np.isfinite(mo["cd"]) and np.isfinite(mo["mean_u"])
    assert mo["div"] < 5.0
    assert mo["cd"] > 0.0
    assert 0.6 < mo["mean_u"] < 1.3


def test_nitsche_penalty_is_load_bearing(rungB_results):
    """ANTI-VACUITY: the Nitsche penalty must be load-bearing. Removing it
    (alpha=0) in the CLEAN monolithic BREAKS the no-slip — Cd departs the
    penalized value by far more than the R0 tol and ‖div‖ balloons — proving the
    penalty (not the mesh alignment) is what imposes the wall."""
    # compare penalty-vs-no-penalty at the SAME grad-div (gamma=0) so the signal
    # isolates the Nitsche penalty, not the FN1 grad-div stabilization.
    mo_g0, mo_nopen = rungB_results["mo_g0"], rungB_results["mo_nopen"]
    cd_rel = abs(mo_nopen["cd"] - mo_g0["cd"]) / abs(mo_g0["cd"])
    assert cd_rel > TOL_CD_REL, (
        f"alpha=0 Cd={mo_nopen['cd']:+.4f} unexpectedly matches penalized "
        f"Cd={mo_g0['cd']:+.4f} (rel {cd_rel:.3%}) — the Nitsche penalty is not "
        f"load-bearing (mesh alignment alone should NOT impose no-slip).")
    assert mo_nopen["div"] > 2.0 * mo_g0["div"], (
        f"alpha=0 ‖div‖={mo_nopen['div']:.3f} not elevated vs penalized "
        f"‖div‖={mo_g0['div']:.3f} — penalty not load-bearing.")


def test_projection_weak_nitsche_stable_and_velocity_faithful(rungB_results):
    """THE RUNG-B VERDICT (post-FN1 fix, locked as a regression): the
    consistent-projection split with WEAK NITSCHE, with the FN1 fix (grad-div
    stabilization + correction-step Nitsche re-pin), is STABLE and
    VELOCITY-FAITHFUL to the same-mesh weak-Nitsche monolithic — it does NOT
    blow up (the bare split diverged, |p*| past 1e6), ‖div‖ stays at the
    monolithic's bounded level, and mean|u| tracks the monolithic.

    This is the flip from the pre-fix FAIL assertion: the growing
    interior-divergence mode that the weak wall excited is arrested. See
    ladder_rungB_seed_probe.py for the decisive seed-monolithic-one-step gate
    (the seeded monolithic becomes a BOUNDED fixed point of the split)."""
    pr, mo = rungB_results["pr"], rungB_results["mo"]
    assert not pr.get("blew_up", False), "projection blew up — FN1 fix regressed"
    assert np.isfinite(pr["cd"]) and np.isfinite(pr["mean_u"])
    # ‖div‖ bounded to the monolithic's level (was a runaway before the fix).
    assert pr["div"] < 2.0 * mo["div"], (
        f"projection ‖div‖={pr['div']:.3f} not bounded to the monolithic "
        f"level ‖div‖={mo['div']:.3f} — FN1 stabilization regressed")
    # mean|u| tracks the monolithic (NO 13x runaway, NO weak plateau).
    mu_frac = pr["mean_u"] / mo["mean_u"]
    assert 0.7 < mu_frac < 1.3, (
        f"projection mean|u|={pr['mean_u']:.4f} does not track monolithic "
        f"{mo['mean_u']:.4f} (frac {mu_frac:.3f}) — velocity not faithful")


def test_projection_weak_nitsche_drag_residual(rungB_results):
    """HONEST RESIDUAL (locked): the FN1 fix restores stability + velocity
    faithfulness but does NOT recover the drag Cd (wall pressure-traction). The
    projection's steady wall pressure differs from the monolithic saddle, so
    Cd is decisively off (well outside the R0 15% oracle tol). If this ever
    starts matching, the wall pressure-traction has been fixed and the rung-B
    residual must be revisited."""
    pr, mo = rungB_results["pr"], rungB_results["mo"]
    cd_rel = abs(pr["cd"] - mo["cd"]) / abs(mo["cd"])
    assert cd_rel > TOL_CD_REL, (
        f"consistent-projection weak-Nitsche Cd={pr['cd']:+.4f} unexpectedly "
        f"matches monolithic Cd={mo['cd']:+.4f} (rel {cd_rel:.3%}) — the wall "
        f"pressure-traction may have been fixed; revisit the rung-B residual.")
