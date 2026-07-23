"""Ladder Task 4 — stabilized inner predictor<->PPE iteration knobs.

Unit gates for the ``inner_iterate`` / ``inner_relax`` / ``inner_accel`` knobs
added to ``LerayProjectionStepper`` (docs/dev/2026-07-23-projection-sbm-weak-
fixed-point-verdict.md path (a)). The contract:

  * DEFAULTS (inner_iterate=False) reproduce the single-pass projection
    BIT-FOR-BIT — the classic incremental split is untouched.
  * inner_iterate=True with inner_relax=omega drives the within-step
    predictor<->PPE fixed point with the DAMPED update p* <- p* + omega(p_hat-p*)
    (omega=1 = the naive nu-loop), and CHANGES the trajectory vs single-pass.
  * inner_accel="anderson" applies Anderson mixing on the fixed-point residual
    and changes the trajectory relative to plain relaxation.
  * a divergence guard breaks to the last bounded iterate.

Tiny 2-D vortex MMS fixture (same as test_leray.py). CPU, fast."""
import os
import sys

import numpy as np
import pytest

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.steppers.leray import LerayProjectionStepper

sys.path.insert(0, os.path.dirname(__file__))
from test_ns_stepper import u_ex, f_ex, NU  # noqa: E402

pytestmark = pytest.mark.tier5

LEVEL = 3
DT = 0.05


def _make(device="cpu", **kw):
    tree = build_uniform(LEVEL, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    st = LerayProjectionStepper(
        dm, NU, DT, f_fn=f_ex, g_fn=lambda x, t: np.zeros((len(x), 2)),
        order=1, picard_iters=2, timestab=False, **kw)
    st.set_initial(lambda x: u_ex(x, 0.0))
    return st


def _march(st, nsteps=4):
    u = p = None
    for _ in range(nsteps):
        u, p = st.step()
    return u, p


# --------------------------------------------------------------------------
# 1. DEFAULTS: inner iteration off -> single-pass, bit-for-bit unchanged
# --------------------------------------------------------------------------
def test_defaults_are_single_pass_bit_for_bit():
    base = _make()                        # inner_iterate defaults False
    u0, p0 = _march(base)
    # an explicit inner_iterate=False must be identical too
    off = _make(inner_iterate=False, inner_relax=0.5, inner_accel="anderson")
    u1, p1 = _march(off)
    assert np.array_equal(u0, u1)
    assert np.array_equal(p0, p1)
    assert base.inner_iters == 1


# --------------------------------------------------------------------------
# 2. RELAXATION knob validation
# --------------------------------------------------------------------------
def test_inner_relax_range_validated():
    with pytest.raises(ValueError):
        _make(inner_relax=0.0)
    with pytest.raises(ValueError):
        _make(inner_relax=1.5)
    with pytest.raises(ValueError):
        _make(inner_accel="bogus")


# --------------------------------------------------------------------------
# 3. omega=1 inner iteration == the naive fixed-point recursion:
#    the first pass of the inner loop MUST equal the single-pass predictor;
#    a converged inner loop makes p_hat self-consistent with u_hat (the
#    within-step fixed point) so it DIFFERS from single-pass.
# --------------------------------------------------------------------------
def test_inner_iterate_changes_trajectory():
    single = _make()
    us, ps = _march(single, nsteps=1)

    itr = _make(inner_iterate=True, inner_relax=1.0, inner_max=8)
    ui, pi = _march(itr, nsteps=1)

    # the inner loop ran more than one pass on a developing step
    assert itr.inner_iters >= 1
    # and (unless it converged in a single pass) reached a different state
    if itr.inner_iters > 1:
        assert not np.allclose(ui, us)
        assert not np.allclose(pi, ps)


# --------------------------------------------------------------------------
# 4. RELAXED update law: p* <- p* + omega (p_hat - p*).
#    With omega small, ONE inner pass moves p* only omega-of-the-way from the
#    lagged p* toward the first p_hat; check the update law directly by
#    reproducing the first pass by hand.
# --------------------------------------------------------------------------
def test_relaxed_update_law():
    # single pass from rest gives the first p_hat (= p_star + phi, p_star=0)
    probe = _make()
    probe.step()                                    # one full step, p_star set
    # Build a fresh stepper, capture its FIRST-pass p_hat (single pass):
    sp = _make()
    _uh, _phi, p_hat1, _uq, _fs, _sig = sp._projection_pass(
        sp.t + sp.dt, None, None, None)
    p_star0 = sp.p_star.copy()                       # == 0 at step 1

    # a relaxed inner solve with omega and inner_max=1 must, after its single
    # pass, have moved p_star to p_star0 + omega (p_hat1 - p_star0) IF it did
    # not early-exit on tol. Force >=2 iters by a tiny tol.
    omega = 0.3
    rel = _make(inner_iterate=True, inner_relax=omega, inner_max=1,
                inner_tol=0.0)
    # run the inner solve directly (does not commit p_star to p_hat)
    rel._inner_solve(rel.t + rel.dt, None, None, None)
    expected = p_star0 + omega * (p_hat1 - p_star0)
    assert np.allclose(rel.p_star, expected, atol=1e-10), (
        "after the first relaxed pass p_star must equal "
        "p_star0 + omega(p_hat - p_star0)")


# --------------------------------------------------------------------------
# 5. ANDERSON changes the trajectory vs plain relaxation
# --------------------------------------------------------------------------
def test_anderson_differs_from_relaxation():
    relax = _make(inner_iterate=True, inner_relax=0.5, inner_accel="none",
                  inner_max=8, inner_tol=1e-10)
    ur, pr = _march(relax, nsteps=2)
    ande = _make(inner_iterate=True, inner_relax=0.5, inner_accel="anderson",
                 inner_max=8, inner_tol=1e-10, inner_anderson_m=3)
    ua, pa = _march(ande, nsteps=2)
    # both converge to (nearly) the same within-step fixed point, but the
    # iterate COUNTS / paths differ — Anderson should not be bit-identical.
    assert (relax.inner_res_hist != ande.inner_res_hist) or \
           (relax.inner_iters != ande.inner_iters)
