"""P2-R0 velocity-update EXPERIMENT — can a different Step-3 velocity update (or
a grad-div predictor penalty) REDUCE the O(1) POINTWISE divergence of the
SBM-projection velocity WITHOUT hurting Cd / temporal order?

Motivation (see .superpowers/sdd/p2r0-divergence-diagnostic-report.md): the
composed projection is CORRECT in the metrics it controls — the weak divergence
``||B^T u||`` is ~130x below the pointwise ``div_l2``, the PPE space is weakly
solenoidal to machine zero, and Cd matches the SAME-mesh monolithic (no-split)
SBM-NS. The residual O(1) ``div_l2 ~ 12`` comes from the Step-3 CONSISTENT-MASS
L2 velocity update not preserving the discrete pointwise
``B^T u_hat - (1/sigma) K_p phi = 0`` identity. The NAMED lever for a smaller
pointwise divergence is therefore the VELOCITY UPDATE, exposed here as a knob
``velocity_update`` on ``LerayProjectionStepper`` / ``LeraySBMStepper``:

  A "consistent" (DEFAULT, unchanged) — consistent-mass L2 re-projection.
  B "lumped"                          — row-sum diagonal mass in Step 3 (nodal
                                        collocation of u = u_hat-(1/sigma)grad phi).
  C "graddiv"                         — consistent-mass update PLUS a
                                        graddiv_scale * tau_C (div w, div u)
                                        grad-div (LSIC) penalty in the predictor.

The deliverable is a MEASURED comparison, gate-hygiene-clean:

  * pointwise div_l2 and weak ||B^T u|| — the SAME independent grad(N).u loop the
    diagnostic uses (not a re-run of the PPE);
  * Cd vs the INDEPENDENT same-mesh monolithic (no-split) reference Cd (imported
    from the diagnostic's ``_monolithic_cd``);
  * temporal MMS order on the vortex problem (test_leray's own gate) per variant.

Success criterion (report verdict): a variant that reduces div_l2 substantially
while keeping weak-div small, Cd ~ monolithic (2.85), and MMS order within
+-0.10 of the "consistent" baseline. The tests below RECORD the numbers and
assert only the robust facts (the default is unchanged; each variant runs and
its div_l2 is finite; lumped DOES reduce div_l2; the div/order trade-off is
real). The verdict is in the report — no variant is adopted as a new default.

Host-side only (numpy / scipy splu). ``uv run pytest -q``. Never gpubox.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.geometry.csg import Sphere
from diffsim.sbm.surrogate import (classify_lambda, extract_surrogate,
                                    GeometryData)
from diffsim.steppers.leray import LerayProjectionStepper
from diffsim.steppers.leray_sbm import LeraySBMStepper

# Reuse the diagnostic's Re20 fixture + the INDEPENDENT monolithic-Cd reference.
from p2r0_divergence_diagnostic import (_build_re20, weak_divergence,
                                        _monolithic_cd, R, U_IN, NU)
from p2r0_harness import march_to_steady

pytestmark = pytest.mark.tier5

VARIANTS = [("consistent", 1.0), ("lumped", 1.0),
            ("graddiv", 1.0), ("graddiv", 5.0)]


# ---------------------------------------------------------------------------
# Re20 measurement: march to steady, report div_l2 / weak / Cd per variant.
# ---------------------------------------------------------------------------

def _march_re20(device, velocity_update, graddiv_scale):
    oracle, dm, sf, geo, strong_mask, u_inf, mesh, cons = _build_re20(device)

    def f_fn(x, t):
        return np.zeros((len(x), 2))

    st = LeraySBMStepper(oracle, dm, NU, 0.05, f_fn, u_inf=u_inf,
                         strong_mask=strong_mask, lam=0.5, domain="outside",
                         order=1, picard_iters=2, solver="splu",
                         ppe_finescale=False, alpha=100.0,
                         velocity_update=velocity_update,
                         graddiv_scale=graddiv_scale)
    st.set_initial(lambda c: np.zeros((len(c), 2)))
    cd, cl, nsteps = march_to_steady(st, dt=0.05, U_in=U_IN, D=2.0 * R,
                                     max_steps=200, rate_tol=5e-3)
    div = float(st.divergence_l2())
    w2, winf = weak_divergence(st)
    return dict(Cd=float(cd), Cl=float(cl), steps=int(nsteps),
                div_l2=div, weak_l2=w2)


# ---------------------------------------------------------------------------
# Temporal MMS order per variant (test_leray's own vortex gate).
# ---------------------------------------------------------------------------

def _mms_order(velocity_update, graddiv_scale, device):
    from test_ns_stepper import u_ex, f_ex, NU as NU_MMS

    def make(level, dt):
        tree = build_uniform(level, dim=2)
        mesh = build_mesh(tree, p=1)
        cons = build_constraints(mesh)
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
        st = LerayProjectionStepper(
            dm, NU_MMS, dt, f_fn=f_ex,
            g_fn=lambda x, t: np.zeros((len(x), 2)),
            order=2, picard_iters=2, timestab=False,
            velocity_update=velocity_update, graddiv_scale=graddiv_scale)
        st.set_initial(lambda x: u_ex(x, 0.0))
        return st

    def run(level, nsteps, T):
        st = make(level, T / nsteps)
        for _ in range(nsteps):
            u, p = st.step()
        return u

    T = 0.25
    ref = run(4, 64, T)
    errs = [np.sqrt(((run(4, n, T) - ref) ** 2).sum(1).mean())
            for n in (8, 16, 32)]
    rates = [float(np.log2(errs[i] / errs[i + 1])) for i in range(2)]
    return errs, rates


# ---------------------------------------------------------------------------
# THE deliverable: one table across all variants, printed for the report.
# ---------------------------------------------------------------------------

def test_velupdate_variants_table(device):
    """A/B/C measured comparison: for each velocity_update variant, report the
    Re20 steady {div_l2, weak ||B^T u||, Cd} and the vortex-MMS temporal order,
    against the INDEPENDENT monolithic (no-split) Cd reference. This is the
    experiment's headline deliverable; the verdict is in the report."""
    cd_mono = _monolithic_cd(device)
    rows = []
    for vu, scale in VARIANTS:
        re20 = _march_re20(device, vu, scale)
        errs, rates = _mms_order(vu, scale, device)
        rows.append(dict(vu=vu, scale=scale, **re20,
                         mms_err=errs[-1], mms_rates=rates))
    print("\n[P2-R0 velocity-update experiment — Re20 (alpha=100 stable) "
          f"+ vortex-MMS temporal order]  monolithic no-split Cd = {cd_mono:.4f}")
    hdr = (f"  {'variant':22s} {'div_l2':>8s} {'weak||Bu||':>11s} "
           f"{'Cd':>7s} {'dCd%mono':>9s} {'MMS_err':>9s} {'MMS_rates':>16s} "
           f"{'steps':>6s}")
    print(hdr)
    for r in rows:
        label = (f"{r['vu']}(x{r['scale']:g})" if r['vu'] == 'graddiv'
                 else r['vu'])
        dcd = 100.0 * (r['Cd'] - cd_mono) / cd_mono
        rates_s = "[" + ", ".join(f"{x:.2f}" for x in r['mms_rates']) + "]"
        print(f"  {label:22s} {r['div_l2']:8.3f} {r['weak_l2']:11.4e} "
              f"{r['Cd']:7.4f} {dcd:+8.1f}% {r['mms_err']:9.2e} "
              f"{rates_s:>16s} {r['steps']:6d}")

    # Robust assertions (facts, not the verdict):
    by = {(_r['vu'], _r['scale']): _r for _r in rows}
    base = by[("consistent", 1.0)]
    for r in rows:
        assert np.isfinite(r['div_l2']) and np.isfinite(r['Cd'])
    # (i) lumped DOES reduce the pointwise div_l2 (the named lever works)...
    assert by[("lumped", 1.0)]['div_l2'] < base['div_l2'], (
        "lumped should reduce pointwise div_l2 vs consistent")
    # (ii) ...but at a real MMS-order cost (lumped mass is only 1st order):
    assert by[("lumped", 1.0)]['mms_err'] > 5.0 * base['mms_err'], (
        "expected lumped to degrade the temporal-MMS accuracy (trade-off)")
    # (iii) the baseline reproduces the diagnostic's numbers (regression guard):
    assert abs(base['Cd'] - 2.68) < 0.2 and abs(base['div_l2'] - 12.2) < 1.0
    pytest.velupdate_table = rows


def test_default_velocity_update_is_consistent_and_unchanged(device):
    """DEFAULT-preservation guard: the knob defaults to "consistent" and the
    "consistent" path is BIT-IDENTICAL to omitting the knob entirely (no
    default was changed by this experiment). Checked on the vortex MMS."""
    from test_ns_stepper import u_ex, f_ex, NU as NU_MMS
    tree = build_uniform(4, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)

    def make(**kw):
        dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
        st = LerayProjectionStepper(
            dm, NU_MMS, 0.25 / 16, f_fn=f_ex,
            g_fn=lambda x, t: np.zeros((len(x), 2)),
            order=2, picard_iters=2, timestab=False, **kw)
        st.set_initial(lambda x: u_ex(x, 0.0))
        return st

    st_default = make()                                  # no knob -> default
    st_explicit = make(velocity_update="consistent")
    assert st_default.velocity_update == "consistent"
    u_d = u_e = None
    for _ in range(16):
        u_d, _ = st_default.step()
        u_e, _ = st_explicit.step()
    assert np.array_equal(u_d, u_e), (
        "explicit velocity_update='consistent' must be bit-identical to the "
        "default path — the experiment changed no default")


def test_lumped_and_graddiv_run_and_are_finite(device):
    """Smoke: each non-default variant constructs and runs a step from rest on
    the Re20 fixture with a finite div_l2 (the knob is wired end-to-end through
    LeraySBMStepper into the base predictor + Step-3 update)."""
    oracle, dm, sf, geo, strong_mask, u_inf, mesh, cons = _build_re20(device)

    def f_fn(x, t):
        return np.zeros((len(x), 2))

    for vu, scale in [("lumped", 1.0), ("graddiv", 1.0), ("graddiv", 5.0)]:
        oracle2, dm2, sf2, geo2, sm2, uinf2, mesh2, cons2 = _build_re20(device)
        st = LeraySBMStepper(oracle2, dm2, NU, 0.05, f_fn, u_inf=uinf2,
                             strong_mask=sm2, lam=0.5, domain="outside",
                             order=1, picard_iters=2, solver="splu",
                             alpha=100.0, velocity_update=vu,
                             graddiv_scale=scale)
        st.set_initial(lambda c: np.zeros((len(c), 2)))
        for _ in range(3):
            st.step()
        assert np.isfinite(st.divergence_l2()), (vu, scale)


def test_invalid_velocity_update_rejected():
    """The knob validates its argument (fail-fast on a typo)."""
    tree = build_uniform(3, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), "cpu")
    with pytest.raises(ValueError):
        LerayProjectionStepper(dm, 0.01, 0.1, f_fn=lambda x, t: np.zeros(
            (len(x), 2)), g_fn=lambda x, t: np.zeros((len(x), 2)),
            velocity_update="bogus")
