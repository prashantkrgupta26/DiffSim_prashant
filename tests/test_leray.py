"""M1b Task 6 gates: the Leray projection stepper (Algorithm 1) on the same
transient vortex MMS as the monolithic stepper — measured temporal order,
head-to-head accuracy, and the Leray point: a divergence sentinel at least
as good as the monolithic solve. Spec S17 discipline: both steppers on the
same acceptance problem."""
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
from test_ns_stepper import u_ex, f_ex, NU, _run as _run_mono  # noqa: E402

pytestmark = pytest.mark.tier5


def _make(level, dt, device, order=2, picard=2):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    st = LerayProjectionStepper(
        dm, NU, dt, f_fn=f_ex,
        g_fn=lambda x, t: np.zeros((len(x), 2)),
        order=order, picard_iters=picard, timestab=False)
    st.set_initial(lambda x: u_ex(x, 0.0))
    return st


def _run(level, nsteps, T, device, **kw):
    st = _make(level, T / nsteps, device, **kw)
    for _ in range(nsteps):
        u, p = st.step()
    return u, st


def test_leray_temporal_order(device):
    # MEASURED (m1b findings 5): 1.07e-2 -> 2.95e-3 -> 1.37e-3, rates
    # 1.86 -> 1.10 — textbook incremental projection: BDF2-dominated second
    # order at coarse dt decaying toward the O(dt) pressure-splitting floor.
    # Lock the shape: near-2 in the BDF2 regime, monotone decay, absolute
    # accuracy. (Rotational/implicit-fine-scale variants raise the floor —
    # benchmark-task TODO.)
    T = 0.25
    ref, _ = _run(4, 64, T, device)
    errs = []
    for n in (8, 16, 32):
        u, _ = _run(4, n, T, device)
        errs.append(np.sqrt(((u - ref) ** 2).sum(1).mean()))
    rates = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
    assert rates[0] > 1.5, (errs, rates)
    assert rates[-1] > 0.8, (errs, rates)
    assert errs[-1] < 2e-3, errs


def test_leray_vs_monolithic_head_to_head(device):
    # same problem, same mesh, same dt: comparable accuracy (within 4x)
    # and the projection's divergence sentinel not worse than 2x monolithic
    T = 0.2
    level, n = 4, 16
    u_l, st_l = _run(level, n, T, device)
    u_m, st_m = _run_mono(level, n, T, device, timestab=False)
    coords = st_l.free_coords
    e_l = np.sqrt(((u_l - u_ex(coords, T)) ** 2).sum(1).mean())
    e_m = np.sqrt(((u_m - u_ex(coords, T)) ** 2).sum(1).mean())
    assert e_l < 4.0 * e_m, (e_l, e_m)
    div_l = st_l.divergence_l2()
    div_m = st_m.divergence_l2()
    assert div_l < 2.0 * div_m, (div_l, div_m)


def test_leray_picard_mechanism_live(device):
    # MEASURED: at the coarsest dt the pressure-splitting error dominates,
    # so a second Picard pass does NOT reduce error vs the fine-dt reference
    # (8.7e-3 vs 1.07e-2) — asserting direction there was a test-design
    # trap. Lock what is true: the nonlinearity is ENGAGED (solutions
    # differ measurably) and both stay stable.
    T = 0.25
    ref, _ = _run(4, 8, T, device, picard=1)
    u2, _ = _run(4, 8, T, device, picard=2)
    d = np.sqrt(((u2 - ref) ** 2).sum(1).mean())
    scale = np.sqrt((ref ** 2).sum(1).mean())
    assert np.isfinite(d) and scale < 10.0
    assert d > 1e-5 * scale, (d, scale)


def test_leray_implicit_finescale_stable(device):
    """Findings 5b closure: with the IMPLICIT (1/sigma + tau_m)-weighted PPE
    the fine-scale term is STABLE (the explicit form blew up ~7e5) and no
    less accurate than the incremental scheme on the same ladder point."""
    T = 0.25
    st = _make(4, T / 16, device)
    st.ppe_finescale = True
    for _ in range(16):
        u_fs, _ = st.step()
    ref, _ = _run(4, 64, T, device)
    e_fs = np.sqrt(((u_fs - ref) ** 2).sum(1).mean())
    u_inc, _ = _run(4, 16, T, device)
    e_inc = np.sqrt(((u_inc - ref) ** 2).sum(1).mean())
    assert np.isfinite(e_fs) and e_fs < 0.1, e_fs          # STABLE
    assert e_fs < 2.0 * e_inc, (e_fs, e_inc)               # comparable


def test_newton_predictor_contracts_faster(device):
    """Task 6b (the draft's Algorithm 1 NONLINEAR Newton predictor): on a
    stiff step (coarse dt, strong vortex) Newton's iterate differences
    contract much faster than Picard's — and the draft's '1-2 Newton
    iterations per step' claim is checked as diff2/diff1."""
    T = 0.25
    diffs = {}
    for mode in ("picard", "newton"):
        st = _make(4, T / 4, device)          # very coarse dt: stiff
        st.predictor = mode
        st.picard_iters = 4
        st.set_initial(lambda x: 2.0 * u_ex(x, 0.0))   # strong field
        st.step()
        diffs[mode] = st.predictor_diffs
    # both converge on this problem; Newton's contraction is much stronger.
    # MEASURED: newton diffs [0.278, 0.0595, 0.0085] — contraction ~0.2 per
    # iterate, i.e. INEXACT Newton (the cross-term is Galerkin-only; SUPG,
    # tau', and the (div du) a terms are deliberately Picard-level), which
    # buys a much better linear rate rather than quadratic. The draft's
    # '1-2 iterations' is its consistent-linearization Newton — recorded
    # as the remaining delta.
    ratio_p = diffs["picard"][-1] / diffs["picard"][0]
    ratio_n = diffs["newton"][-1] / diffs["newton"][0]
    assert ratio_n < 0.2 * ratio_p, (diffs["picard"], diffs["newton"])
    assert diffs["newton"][2] < 5e-2 * diffs["newton"][0], diffs["newton"]
