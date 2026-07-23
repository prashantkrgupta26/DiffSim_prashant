"""Rung A fast asserted case — body-fitted square, STRONG Dirichlet.

The DECISIVE control at the steady Re (Re=40) on a small level (5): with STRONG
no-slip on an EXACT body-fitted square (dmax==0, so NO SBM shift and NO weak
Nitsche), does single-pass pressure-projection match the same-mesh monolithic Cd
AND develop mean|u| to the monolithic magnitude — or fail?

MEASURED RUNG-A VERDICT (this test locks it in as a regression):
  * MONOLITHIC (same mesh) — CLEAN: converges to a bounded ‖div u‖ ~ O(1), a
    physical steady Cd (~+4.2 confined), mean|u| ~ 1.04.
  * PROJECTION (base, single-pass) — FAILS: on the OPEN outflow the lagged-p*
    incremental split is PRESSURE-UNSTABLE — ‖p‖ and ‖div u‖ grow unbounded and
    Cd runs to large negative values; it does NOT reach the monolithic steady
    state. (The failure is NOT caused by the obstacle: the same base projection
    blows up on the obstacle-free open channel too.) With PSPG added
    (`ppe_fine_scale=True`, a Task-4-class stabilizer, NOT part of rung A) ‖p‖
    bounds but mean|u| then PINS at ~5% of monolithic — the lagged-p* from-rest
    weak fixed point (docs/dev/2026-07-23-projection-sbm-weak-fixed-point-
    verdict.md).

So single-pass base projection is UNFAITHFUL to the same-mesh monolithic on this
external flow. This is the EXPECTED-and-important rung-A result. The `dmax==0`
body-fitted guard is the anti-vacuity check — ALWAYS asserted.

TASK-4 UPDATE (2026-07-23) — the fix did NOT flip rung A; the split is
OPERATOR-INCONSISTENT (Lane 1c). Both candidate fixes were implemented and run
head-to-head against this same-mesh oracle:
  * Stabilized inner predictor<->PPE iteration (inner_iterate + inner_relax +
    inner_accel="anderson", divergence guard) — the intended vehicle. The
    within-step predictor<->PPE map is NON-CONTRACTIVE on this open-outflow
    external flow (inner residual never converges at any omega/Anderson); it
    develops mean|u| but Cd runs strongly NEGATIVE and ‖div‖ grows.
  * Consistent PPE operator L = G^T M^-1 G (consistent_ppe=True) in place of the
    FE Laplacian K_p. K_p differs from the true projection operator L by ~67%
    (measured on this mesh), so the classic PPE does NOT project the corrected
    field onto the discretely divergence-free space. The DECISIVE diagnostic:
    seeding the projection with the EXACT monolithic (u, p) — a would-be fixed
    point — and taking ONE step throws Cd +4.18 -> -55 and RAISES ‖div‖
    1.22 -> 6.0 (the correction corrupts a perfect field). With the consistent L
    a static seeded projection IS idempotent (B^T u_corr -> 1e-15), but LIVE it
    diverges FASTER (mean|u| overshoots, Cd -> -19..-63) — idempotency != coupled
    -loop stability, even paired with the damped/Anderson inner iteration.

VERDICT: the plain lagged-p* split fixed point is not the monolithic fixed
point on rung A; neither the stabilized inner iteration nor the consistent PPE
operator (alone or together) makes the projection faithful. Reportable per the
Task-4 brief (Lane 1c operator-inconsistency). The knobs ship default-OFF and
bit-for-bit; see docs/dev/2026-07-23-projection-sbm-weak-fixed-point-verdict.md
(§Task-4 addendum) and .superpowers/sdd/task-4-report.md.

Kept CHEAP (level 5, 250-step march). Full Re-40/100 numbers + Strouhal live in
the driver (tests/ladder_rungA_square_strong.py); the fix experiments live in
tests/rungA_inner_experiment.py and tests/rungA_outflow_diag.py.
"""
import numpy as np
import pytest

from ladder_fixtures import build_square_channel_2d
from ladder_rungA_square_strong import march_projection, march_monolithic

LEVEL = 5
HALF = 0.125          # k/2^level aligned at level>=3 -> dmax==0 (body-fitted)
RE = 40               # steady, below square-cylinder shedding onset
DT = 0.02
NSTEPS = 250          # short march (cheap): monolithic is steady, projection
                      # already visibly diverging by here.


@pytest.fixture(scope="module")
def rungA_results():
    fx = build_square_channel_2d(LEVEL, RE, half=HALF, offset=0, device="cpu")
    pr = march_projection(fx, dt=DT, nsteps=NSTEPS, rate_tol=5e-4)
    mo = march_monolithic(fx, dt=DT, nsteps=NSTEPS, rate_tol=2e-4)
    return dict(fx=fx, pr=pr, mo=mo)


def test_body_fitted_guard(rungA_results):
    """Anti-vacuity: the carve is EXACT body-fitted (dmax==0) so this rung tests
    strong Dirichlet on a clean mesh — NOT an SBM shift — and the obstacle set
    is non-empty (the strong no-slip is actually imposed)."""
    fx = rungA_results["fx"]
    assert fx["dmax"] == 0
    assert int(fx["obstacle_node_mask"].sum()) > 0


def test_monolithic_is_clean_same_mesh_oracle(rungA_results):
    """The same-mesh monolithic oracle develops the flow and is well-behaved:
    bounded ‖div u‖, physical positive steady Cd, mean|u| ~ U_IN. This is the
    bar the projection must (and, single-pass, does NOT) meet."""
    mo = rungA_results["mo"]
    assert np.isfinite(mo["cd"]) and np.isfinite(mo["mean_u"])
    assert mo["div"] < 5.0                      # bounded weak divergence
    assert mo["cd"] > 0.0                        # drag points downstream
    assert 0.8 < mo["mean_u"] < 1.3             # flow developed to ~U_IN


def test_projection_does_not_match_monolithic_operator_inconsistent(
        rungA_results):
    """THE decisive rung-A read, LOCKED as the Task-4 verdict: the projection
    is UNFAITHFUL to the same-mesh monolithic on this open-outflow external flow
    and NEITHER candidate fix flips it — the split is operator-inconsistent
    (Lane 1c; see the module docstring for the full Task-4 investigation).

    Single-pass base projection: ‖div u‖ grows far past the monolithic's
    bounded value (~30 vs ~1.2) and its Cd does not reach the monolithic steady
    Cd — the pressure-instability signature. This asserts the ESTABLISHED
    unfaithfulness; the fix knobs (inner_iterate, consistent_ppe) ship
    default-OFF so this base single-pass regression is unchanged."""
    pr, mo = rungA_results["pr"], rungA_results["mo"]
    # projection divergence is NOT controlled to the monolithic's bounded level
    # (measured ~30 vs ~1.2) — the pressure-instability signature.
    assert pr["div"] > 5.0 * mo["div"], (
        f"projection ‖div‖={pr['div']:.3f} unexpectedly close to monolithic "
        f"‖div‖={mo['div']:.3f} — single-pass may have started matching; "
        f"re-open the Task-4 operator-inconsistency verdict and update this "
        f"rung-A regression")
    # and the projection Cd does NOT match the monolithic same-mesh Cd.
    cd_rel = abs(pr["cd"] - mo["cd"]) / abs(mo["cd"])
    assert cd_rel > 0.15, (
        f"projection Cd={pr['cd']:+.4f} unexpectedly matches monolithic "
        f"Cd={mo['cd']:+.4f} (rel {cd_rel:.3%}) — re-open the Task-4 verdict "
        f"and update this rung-A regression")
