"""Rung B fast asserted case — body-fitted square, WEAK NITSCHE no-slip.

Rung A (strong Dirichlet on the exact body-fitted square) PASSED: the
consistent-projection split is FAITHFUL to the same-mesh monolithic. Rung B keeps
the SAME body-fitted mesh (aligned Box, offset=0 => dmax==0, so the SBM
shifted-Nitsche form S N_a = N_a DEGENERATES TO STANDARD NITSCHE exactly) and
swaps the obstacle BC treatment strong Dirichlet -> WEAK NITSCHE. This isolates
the Nitsche coupling on the already-repaired projection.

MEASURED RUNG-B VERDICT (this test locks it in as a regression):
  * MONOLITHIC (same mesh, weak Nitsche) — CLEAN: converges to a bounded ‖div u‖,
    a physical steady Cd (~+2.0 at level 4, softer than the strong-Dirichlet
    ~+4.2 by the Nitsche consistency error), mean|u| ~ 0.87. The Nitsche block
    is correct.
  * The Nitsche penalty is LOAD-BEARING (anti-vacuity): zeroing it (alpha=0) in
    the CLEAN monolithic BREAKS the no-slip — Cd runs to ~+71, ‖div‖ ~ 21, and
    it never converges (the obstacle stops being felt).
  * PROJECTION (consistent scheme, weak Nitsche) — FAILS: the projection split is
    UNFAITHFUL to the same-mesh weak-Nitsche monolithic. It does NOT track — mean|u|
    runs to ~13x the monolithic and ‖div‖ blows past the monolithic's bounded
    value. Seeding the EXACT monolithic (u,p) and iterating the projection
    diverges too (the monolithic is NOT a stable fixed point of the
    projection-Nitsche split), and the production LeraySBMStepper
    (consistent_projection=True, with/without sbm_pressure_coupling T3+T6) shows
    the SAME divergence — so this is a real coupling instability, not a driver bug.

DECISIVE READ: strong Dirichlet on this mesh (rung A) is a stable fixed point of
the consistent projection; swapping strong -> WEAK NITSCHE breaks it. The defect
isolates to the NITSCHE COUPLING (not the base projection — rung A cleared it —
and not the shift — dmax==0 here). Weak imposition does NOT preserve faithfulness
on the projection scheme.

Kept CHEAP (level 4, short march). Full Re-40/100 numbers live in the driver
(tests/ladder_rungB_square_nitsche.py).
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
    mo_nopen = march_monolithic(fx, dt=DT, nsteps=NSTEPS, rate_tol=2e-4,
                                backflow_beta=0.5, boundary_vorticity=True,
                                alpha=0.0)
    # PROJECTION (consistent scheme) weak-Nitsche — the decisive test.
    pr = march_projection(fx, dt=DT, nsteps=NSTEPS, rate_tol=5e-4, alpha=ALPHA)
    return dict(fx=fx, mo=mo, mo_nopen=mo_nopen, pr=pr)


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
    mo, mo_nopen = rungB_results["mo"], rungB_results["mo_nopen"]
    cd_rel = abs(mo_nopen["cd"] - mo["cd"]) / abs(mo["cd"])
    assert cd_rel > TOL_CD_REL, (
        f"alpha=0 Cd={mo_nopen['cd']:+.4f} unexpectedly matches penalized "
        f"Cd={mo['cd']:+.4f} (rel {cd_rel:.3%}) — the Nitsche penalty is not "
        f"load-bearing (mesh alignment alone should NOT impose no-slip).")
    assert mo_nopen["div"] > 2.0 * mo["div"], (
        f"alpha=0 ‖div‖={mo_nopen['div']:.3f} not elevated vs penalized "
        f"‖div‖={mo['div']:.3f} — penalty not load-bearing.")


def test_projection_weak_nitsche_is_unfaithful(rungB_results):
    """THE RUNG-B VERDICT (measured, locked as a regression): the
    consistent-projection split with WEAK NITSCHE is UNFAITHFUL to the same-mesh
    weak-Nitsche monolithic — it does NOT track. mean|u| runs far past the
    monolithic magnitude and ‖div‖ blows past the monolithic's bounded value.

    Strong Dirichlet on this mesh (rung A) IS a stable fixed point of the
    consistent projection; swapping strong -> weak Nitsche breaks it. The defect
    isolates to the NITSCHE COUPLING (base projection cleared by rung A; shift
    absent, dmax==0). If this ever starts PASSING (projection matches the
    weak-Nitsche monolithic), the Nitsche coupling has been fixed and rung B's
    verdict must be revisited."""
    pr, mo = rungB_results["pr"], rungB_results["mo"]
    # UNFAITHFUL: either it blew up, or ‖div‖/mean|u| ran far past the monolithic.
    div_runaway = pr["div"] > 5.0 * mo["div"]
    mu_runaway = pr["mean_u"] > 3.0 * mo["mean_u"]
    blew = pr.get("blew_up", False)
    assert blew or div_runaway or mu_runaway, (
        f"consistent-projection weak-Nitsche UNEXPECTEDLY tracks the monolithic: "
        f"proj Cd={pr['cd']:+.4f} meanu={pr['mean_u']:.4f} div={pr['div']:.3e} vs "
        f"mono Cd={mo['cd']:+.4f} meanu={mo['mean_u']:.4f} div={mo['div']:.3e}. "
        f"The Nitsche coupling may have been fixed — revisit the rung-B verdict.")
    # And the Cd is NOT within the same-mesh oracle tolerance (decisively off).
    cd_rel = abs(pr["cd"] - mo["cd"]) / abs(mo["cd"])
    assert cd_rel > TOL_CD_REL, (
        f"consistent-projection weak-Nitsche Cd={pr['cd']:+.4f} unexpectedly "
        f"matches monolithic Cd={mo['cd']:+.4f} (rel {cd_rel:.3%}).")
