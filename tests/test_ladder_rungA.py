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

TASK-F1 UPDATE (2026-07-23) — THE FIX FLIPPED RUNG A. The consistent-projection
mode (`consistent_projection=True`) implements the group's VMS-stabilized
Helmholtz-Leray scheme (ns_projection_vms_paper Eq 44a-c / Algorithm 1) as the
coherent set of changes #1-#4:
  #1 PSPG-consistent PPE — the coarse divergence is assembled COLLOCATED with q
     (-sigma(div u_h, q)_h, NOT integrated by parts) so it matches the monolithic
     PSPG continuity row (q, div u) + tau_m(grad q, r_m) term-for-term; only the
     fine scale u' = -tau_m r_m is taken by parts. The prior ppe_fine_scale path
     put the WHOLE flux by parts, fabricating a boundary term sigma(u_h.n, q)_Gamma
     that is nonzero at an OPEN outflow -> the spurious phi that corrupted a
     seeded monolithic field. Fixing this makes the monolithic steady state a
     FIXED POINT.
  #2 fine-scale u' in BOTH PPE source and L2 correction (consistent mass).
  #3 disjoint outflow BCs (velocity natural traction-free in the predictor;
     Dirichlet p'=0 on the pressure CORRECTION at outflow via
     pressure_outflow_nodes; node-0 gauge removed for the open flow).
  #4 rotational-incremental pressure update.

MEASURED (level 5, Re=40, this test):
  * SEED-MONOLITHIC-ONE-STEP is now a FIXED POINT: seeding (u_mono, p_mono) and
    taking one consistent-projection step PRESERVES it (Cd +4.18 -> +4.12,
    ‖div‖ 1.22 -> 1.29, du_from_seed 6e-2) — vs the base split's Cd -> -55,
    ‖div‖ 1.22 -> 6.0. (Probe: tests/rungA_outflow_diag.py.)
  * FROM REST, Re=40 steady: projection Cd -> +3.73 (monolithic +4.18, ~11%),
    mean|u| -> 1.044 (monolithic 1.038, <1%) — matches the same-mesh oracle;
    no weak plateau, no blow-up.

The BASE single-pass split (consistent_projection=False) is UNCHANGED and still
fails (kept as the anti-vacuity contrast). The fix ships default-OFF, bit-for-bit
(the test_leray/test_p2r0_parity/test_ladder_rung0 regressions stay green).

Kept CHEAP (level 5, short march). Full Re-40/100 numbers + Strouhal live in the
driver (tests/ladder_rungA_square_strong.py, arg "consistent").
"""
import numpy as np
import pytest

from ladder_fixtures import build_square_channel_2d
from ladder_rungA_square_strong import march_projection, march_monolithic

LEVEL = 5
HALF = 0.125          # k/2^level aligned at level>=3 -> dmax==0 (body-fitted)
RE = 40               # steady, below square-cylinder shedding onset
DT = 0.02
NSTEPS_BASE = 250     # base split already visibly diverging by here (cheap).
NSTEPS_FIX = 400      # consistent projection converges to the monolithic state.


@pytest.fixture(scope="module")
def rungA_results():
    fx = build_square_channel_2d(LEVEL, RE, half=HALF, offset=0, device="cpu")
    # BASE single-pass split (the fix OFF) — the anti-vacuity contrast.
    pr = march_projection(fx, dt=DT, nsteps=NSTEPS_BASE, rate_tol=5e-4)
    # CONSISTENT-PROJECTION split (the fix ON) — must match the oracle.
    prc = march_projection(fx, dt=DT, nsteps=NSTEPS_FIX, rate_tol=5e-4,
                           consistent_projection=True)
    mo = march_monolithic(fx, dt=DT, nsteps=NSTEPS_FIX, rate_tol=2e-4)
    return dict(fx=fx, pr=pr, prc=prc, mo=mo)


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
    bar the projection must meet."""
    mo = rungA_results["mo"]
    assert np.isfinite(mo["cd"]) and np.isfinite(mo["mean_u"])
    assert mo["div"] < 5.0                      # bounded weak divergence
    assert mo["cd"] > 0.0                        # drag points downstream
    assert 0.8 < mo["mean_u"] < 1.3             # flow developed to ~U_IN


def test_base_split_still_fails_open_outflow(rungA_results):
    """ANTI-VACUITY contrast (default-OFF regression): the BASE single-pass split
    (consistent_projection=False) is still UNFAITHFUL on the open outflow —
    ‖div u‖ grows far past the monolithic's bounded value and its Cd does not
    reach the monolithic steady Cd. If this ever starts passing, the fix's
    load-bearing terms have leaked into the default path."""
    pr, mo = rungA_results["pr"], rungA_results["mo"]
    assert pr["div"] > 5.0 * mo["div"], (
        f"BASE projection ‖div‖={pr['div']:.3f} unexpectedly close to monolithic "
        f"‖div‖={mo['div']:.3f} — the fix may have leaked into the default path.")
    cd_rel = abs(pr["cd"] - mo["cd"]) / abs(mo["cd"])
    assert cd_rel > 0.15, (
        f"BASE projection Cd={pr['cd']:+.4f} unexpectedly matches monolithic "
        f"Cd={mo['cd']:+.4f} (rel {cd_rel:.3%}) — the fix may have leaked into "
        f"the default path.")


def test_consistent_projection_matches_monolithic(rungA_results):
    """THE FIX (Task F1): the consistent-projection split MATCHES the same-mesh
    monolithic on this open-outflow external flow — the decisive proof-of-fix.

    Projection Cd == monolithic Cd within R0 tol (15%) AND mean|u| ->
    monolithic magnitude (no weak plateau). Measured: Cd +3.73 vs +4.18 (~11%),
    mean|u| 1.044 vs 1.038 (<1%)."""
    prc, mo = rungA_results["prc"], rungA_results["mo"]
    assert not prc.get("blew_up", False), "consistent projection blew up"
    # ‖div‖ bounded to the monolithic's level (not the base split's runaway).
    assert prc["div"] < 5.0 * mo["div"], (
        f"consistent-projection ‖div‖={prc['div']:.3f} not controlled to the "
        f"monolithic level ‖div‖={mo['div']:.3f}")
    # Cd matches the same-mesh oracle (R0 tol 15%).
    cd_rel = abs(prc["cd"] - mo["cd"]) / abs(mo["cd"])
    assert cd_rel < 0.15, (
        f"consistent-projection Cd={prc['cd']:+.4f} does not match monolithic "
        f"Cd={mo['cd']:+.4f} (rel {cd_rel:.3%})")
    # mean|u| develops to the monolithic magnitude — NO weak plateau.
    mu_frac = prc["mean_u"] / mo["mean_u"]
    assert mu_frac > 0.8, (
        f"consistent-projection mean|u|={prc['mean_u']:.4f} pins weak vs "
        f"monolithic {mo['mean_u']:.4f} (frac {mu_frac:.3f})")
    assert abs(prc["mean_u"] - mo["mean_u"]) / mo["mean_u"] < 0.20
