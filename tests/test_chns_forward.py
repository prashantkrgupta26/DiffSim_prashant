"""tests/test_chns_forward.py — Task 5 (SP-0): CHNSDiscrete numpy mirror.

TDD gates (task-5-brief.md):
  1. test_chns_discrete_jacobian_vs_fd — analytic J @ v vs central FD of the
     coupled residual, rel error < 1e-6 on a level-3 2-D mesh.
  2. test_chns_discrete_stationary_drop — tanh disk, u0=0, gravity OFF:
     after 5 steps max|u| < 1e-3 and mass |sum(phi)-sum(phi0)| flat.
  3. test_chns_discrete_bubble_rises — ratio 10, gravity ON: light bubble
     (phi=-1) centroid_y strictly increases over the last 5 steps.

benchmarks/ is not on sys.path in tests — mirror the bootstrap from
tests/test_chns_scaffold.py.
"""
import os as _os
import sys as _sys

import numpy as np
import pytest

_BENCH_DIR = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "benchmarks")
)
if _BENCH_DIR not in _sys.path:
    _sys.path.insert(0, _BENCH_DIR)
_REPO_DIR = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")
)
if _REPO_DIR not in _sys.path:
    _sys.path.insert(0, _REPO_DIR)

from chns.cases import BUBBLE_RISE_RE35_WE10  # noqa: E402
from chns import metrics  # noqa: E402


# ---------------------------------------------------------------------------
# Mesh helper (uniform p=1, constraints.T == identity)
# ---------------------------------------------------------------------------
def _make_dm(level, dim=2):
    import warp as wp
    wp.init()
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh
    from diffsim import default_device

    tree = build_uniform(level, dim=dim)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    device = default_device()
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), device)
    return dm, mesh, cons


# ---------------------------------------------------------------------------
# Step 1: analytic Jacobian vs central finite differences
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("interface", ["ch", "cac"])
def test_chns_discrete_jacobian_vs_fd(interface):
    """Analytic J @ v vs central-FD of the residual, rel error < 1e-6.

    CH: standard frozen-nothing (mu is a PDE unknown).
    CAC: _freeze_beta=True is set before the FD perturb so the numerically-
    differenced residual uses the same frozen beta as the analytic Jacobian.
    The reviewer measured ~3.6e-10 for CAC; tolerance 1e-6 is comfortable.
    """
    from diffsim.adjoint.chns import CHNSDiscrete

    dm, mesh, cons = _make_dm(level=3, dim=2)
    op = CHNSDiscrete(level=3, dim=2, case=BUBBLE_RISE_RE35_WE10, dt=1e-2,
                      dm=dm, interface=interface)

    rng = np.random.default_rng(0)
    coords = mesh.node_coords
    # smooth admissible phi in [-0.9, 0.9]
    phi = 0.9 * np.tanh(
        (np.sqrt((coords[:, 0] - 0.5) ** 2 + (coords[:, 1] - 0.5) ** 2) - 0.25)
        / (2.0 * op.h)
    )
    n = op.nn
    dim = op.dim
    x = np.zeros(op.ndof)
    # small velocities, admissible phi, zero p, mu from a small field
    for d in range(dim):
        x[d::op.blk] = 1e-3 * rng.standard_normal(n)
    x[dim::op.blk] = 0.0                       # p
    x[dim + 1::op.blk] = phi                   # phi
    x[dim + 2::op.blk] = 1e-2 * rng.standard_normal(n)  # mu

    # history: previous state = current phi, u=0 (only needed by residual)
    op.set_history(u_n=np.zeros((n, dim)), phi_n=phi.copy())

    # For CAC: freeze beta at the value computed from the base point x.
    # The analytic Jacobian treats beta as a constant (Picard-on-beta);
    # the FD residuals must use the same frozen beta so the linearisation
    # is consistent.  _freeze_beta=True prevents _assemble from recomputing
    # beta when we evaluate residual(x ± eps*v).
    if interface == "cac":
        # Compute residual once to trigger beta freeze at x.
        _ = op.residual(x)
        op._freeze_beta = True

    R0 = op.residual(x)
    J = op.jacobian(x)

    eps = 1e-6
    for _ in range(3):
        v = rng.standard_normal(op.ndof)
        v /= np.linalg.norm(v)
        Rp = op.residual(x + eps * v)
        Rm = op.residual(x - eps * v)
        fd = (Rp - Rm) / (2.0 * eps)
        Jv = J @ v
        rel = np.linalg.norm(Jv - fd) / max(np.linalg.norm(Jv), 1e-30)
        assert rel < 1e-6, (
            f"[{interface}] Jacobian vs FD rel error {rel:.3e} >= 1e-6")

    # Restore (no side-effect for CAC after this test)
    if interface == "cac":
        op._freeze_beta = False


# ---------------------------------------------------------------------------
# Step 2: stationary drop (gravity off) — parasitic bound + mass flat
# ---------------------------------------------------------------------------
def test_chns_discrete_stationary_drop():
    from diffsim.adjoint.chns import CHNSDiscrete

    level = 5
    dm, mesh, cons = _make_dm(level=level, dim=2)
    op = CHNSDiscrete(level=level, dim=2, case=BUBBLE_RISE_RE35_WE10,
                      dt=1e-3, dm=dm, gravity=False, Cn_override="2h")

    coords = mesh.node_coords
    r = np.sqrt((coords[:, 0] - 0.5) ** 2 + (coords[:, 1] - 0.5) ** 2)
    phi0 = -np.tanh((r - 0.25) / (op.Cn * np.sqrt(2.0)))
    op.set_initial(phi0)

    M = op.lumped_mass()
    mass0 = float(M @ phi0)

    for _ in range(5):
        op.step()

    umax = float(np.abs(op.u).max())
    mass1 = float(M @ op.phi)
    assert umax < 1e-3, f"parasitic current max|u|={umax:.3e} >= 1e-3"
    assert abs(mass1 - mass0) < 1e-12 * op.nn, (
        f"mass drift {abs(mass1 - mass0):.3e} >= 1e-12*n"
    )


# ---------------------------------------------------------------------------
# Step 3: bubble rise — light bubble centroid strictly increases
# ---------------------------------------------------------------------------
def test_chns_discrete_bubble_rises():
    from diffsim.adjoint.chns import CHNSDiscrete

    level = 5
    dm, mesh, cons = _make_dm(level=level, dim=2)
    op = CHNSDiscrete(level=level, dim=2, case=BUBBLE_RISE_RE35_WE10,
                      dt=BUBBLE_RISE_RE35_WE10.dt0, dm=dm, gravity=True,
                      Cn_override="2h")

    coords = mesh.node_coords
    r = np.sqrt((coords[:, 0] - 0.5) ** 2 + (coords[:, 1] - 0.35) ** 2)
    phi0 = -np.tanh((r - 0.2) / (op.Cn * np.sqrt(2.0)))  # phi=-1 light bubble
    op.set_initial(phi0)

    cy = []
    for _ in range(10):
        op.step()
        cy.append(metrics.centroid_y(-op.phi, coords))  # bubble = phi<0
    cy = np.asarray(cy)
    last = cy[-5:]
    assert np.all(np.diff(last) > 0), (
        f"bubble centroid_y not strictly increasing over last 5 steps: {last}"
    )


# ---------------------------------------------------------------------------
# Task 6: staggered CH -> NS projection prototype smoke
# ---------------------------------------------------------------------------
def test_staggered_bubble_smoke():
    """CHNSStaggeredStepper, BUBBLE_RISE_RE35_WE10 at level 5 (32x32),
    rho_ratio pinned to 10 for the smoke (the case value is already 10;
    pinned via dataclasses.replace so a future case edit cannot silently
    change this gate), 20 steps at case dt0:
      - no NaN in phi/u/p at any snapshot,
      - |mass drift| printed and < 1e-6 (staggered is NOT exactly
        conservative: convective-form CH advection + the split — the
        loose bound is the gate, the actual value is spike evidence),
      - light bubble (phi=-1) centroid strictly rising over the last 5
        steps.
    Cn_override="2h" mirrors the CHNSDiscrete twin tests above (case
    Cn=0.01 is below the level-5 h=1/32; the sibling monolithic gates
    use the same 2h resolvability convention)."""
    from dataclasses import replace
    from diffsim.steppers.chns import CHNSStaggeredStepper

    level = 5
    dm, mesh, cons = _make_dm(level=level, dim=2)
    case = replace(BUBBLE_RISE_RE35_WE10, rho_ratio=10.0)
    st = CHNSStaggeredStepper(dm, case, dt=case.dt0, Cn_override="2h")

    xc = mesh.node_coords[cons.free_nodes]
    r = np.sqrt((xc[:, 0] - 0.5) ** 2 + (xc[:, 1] - 0.35) ** 2)
    phi0 = np.tanh((r - 0.2) / (st.Cn * np.sqrt(2.0)))  # phi=-1 light bubble
    st.set_initial(phi0)

    mass0 = st.mass_phi()
    snaps = st.march(t_end=20 * case.dt0)
    assert len(snaps) == 20

    cy = []
    for s in snaps:
        assert np.isfinite(s["phi"]).all(), f"NaN phi at t={s['t']}"
        assert np.isfinite(s["u"]).all(), f"NaN u at t={s['t']}"
        assert np.isfinite(s["p"]).all(), f"NaN p at t={s['t']}"
        cy.append(metrics.centroid_y(-s["phi"], xc))  # bubble = phi<0

    drift = abs(st.mass_phi() - mass0)
    wall = float(np.mean([s["wall_per_step"] for s in snaps]))
    print(f"[staggered smoke] |mass drift| = {drift:.3e} (bound 1e-6), "
          f"wall/step = {wall:.3f} s, "
          f"newton_iters(last) = {st.last_newton_iters}, "
          f"clamped(last) = {st.last_clamped}")
    assert drift < 1e-6, f"|mass drift| {drift:.3e} >= 1e-6"

    last = np.asarray(cy[-5:])
    assert np.all(np.diff(last) > 0), (
        f"bubble centroid_y not strictly rising over last 5 steps: {last}"
    )


# ---------------------------------------------------------------------------
# Task 7: monolithic (u,p,phi,mu) Warp kernel + stepper
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tstep", ["bdf1", "bdf2"])
def test_monolithic_parity_vs_mirror(tstep):
    """CHNSMonolithicStepper must reproduce CHNSDiscrete (the numpy mirror,
    ground truth) to 1e-10 relative on (phi, u, p) after 3 steps.
    Level-4 (16x16) 2-D, BUBBLE_RISE_RE35_WE10 (rho_ratio already 10.0),
    same dt, same tstep, and Newton tolerance 1e-12.
    For bdf2: 3 steps engage the 2-step history past the BDF1 bootstrap
    (step 1 = BDF1, steps 2-3 = BDF2 with stored history)."""
    from diffsim.adjoint.chns import CHNSDiscrete
    from diffsim.steppers.chns import CHNSMonolithicStepper

    level = 4
    dm, mesh, cons = _make_dm(level=level, dim=2)
    case = BUBBLE_RISE_RE35_WE10
    dt = case.dt0

    coords = mesh.node_coords
    # light bubble phi=-1 inside; use Cn_override="2h" for resolvability
    # (matches the sibling gates); build phi0 with the mirror's Cn.
    ref = CHNSDiscrete(level=level, dim=2, case=case, dt=dt, dm=dm,
                       gravity=True, Cn_override="2h", newton_tol=1e-12,
                       tstep=tstep)
    r = np.sqrt((coords[:, 0] - 0.5) ** 2 + (coords[:, 1] - 0.35) ** 2)
    phi0 = -np.tanh((r - 0.2) / (ref.Cn * np.sqrt(2.0)))
    ref.set_initial(phi0)

    mono = CHNSMonolithicStepper(dm, case, dt=dt, linsolver="splu",
                                 Cn_override="2h", newton_tol=1e-12,
                                 tstep=tstep)
    mono.set_initial(phi0)

    for _ in range(3):
        ref.step()
        mono.step()

    def relmax(a, b):
        a = np.asarray(a).ravel()
        b = np.asarray(b).ravel()
        return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))

    rphi = relmax(mono.phi, ref.phi)
    ru = relmax(mono.u, ref.u)
    rp = relmax(mono.p, ref.p)
    print(f"[monolithic parity tstep={tstep}] rel(phi)={rphi:.3e} "
          f"rel(u)={ru:.3e} rel(p)={rp:.3e}")
    assert rphi <= 1e-10, f"phi parity {rphi:.3e} > 1e-10 (tstep={tstep})"
    assert ru <= 1e-10, f"u parity {ru:.3e} > 1e-10 (tstep={tstep})"
    assert rp <= 1e-10, f"p parity {rp:.3e} > 1e-10 (tstep={tstep})"


def test_monolithic_parity_body_fn():
    """body_fn contract: CHNSMonolithicStepper and CHNSDiscrete must both
    accept a nonzero smooth body force via body_fn and produce identical
    results to 1e-10 relative.  BDF1, 2 steps, level-4, same setup as
    test_monolithic_parity_vs_mirror[bdf1].

    Body force: f(x,t) = [0.1*sin(pi*x0)*cos(pi*x1), 0.05*cos(pi*x0)]
    (smooth, nonzero momentum source; matches the MMS body_fn signature
    fn(xq[ngp,dim], t) -> [ngp, dim]).
    This closes the two-sided contract over the fbody_gp hook."""
    from diffsim.adjoint.chns import CHNSDiscrete
    from diffsim.steppers.chns import CHNSMonolithicStepper

    level = 4
    dm, mesh, cons = _make_dm(level=level, dim=2)
    case = BUBBLE_RISE_RE35_WE10
    dt = case.dt0

    def body_fn(xq, t):
        x0, x1 = xq[:, 0], xq[:, 1]
        f0 = 0.1 * np.sin(np.pi * x0) * np.cos(np.pi * x1)
        f1 = 0.05 * np.cos(np.pi * x0)
        return np.stack([f0, f1], axis=1)

    coords = mesh.node_coords
    ref = CHNSDiscrete(level=level, dim=2, case=case, dt=dt, dm=dm,
                       gravity=True, Cn_override="2h", newton_tol=1e-12,
                       body_fn=body_fn)
    r = np.sqrt((coords[:, 0] - 0.5) ** 2 + (coords[:, 1] - 0.35) ** 2)
    phi0 = -np.tanh((r - 0.2) / (ref.Cn * np.sqrt(2.0)))
    ref.set_initial(phi0)

    mono = CHNSMonolithicStepper(dm, case, dt=dt, linsolver="splu",
                                 Cn_override="2h", newton_tol=1e-12,
                                 body_fn=body_fn)
    mono.set_initial(phi0)

    for _ in range(2):
        ref.step()
        mono.step()

    def relmax(a, b):
        a = np.asarray(a).ravel()
        b = np.asarray(b).ravel()
        return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))

    rphi = relmax(mono.phi, ref.phi)
    ru = relmax(mono.u, ref.u)
    rp = relmax(mono.p, ref.p)
    print(f"[body_fn parity] rel(phi)={rphi:.3e} rel(u)={ru:.3e} "
          f"rel(p)={rp:.3e}")
    assert rphi <= 1e-10, f"phi parity {rphi:.3e} > 1e-10 (body_fn)"
    assert ru <= 1e-10, f"u parity {ru:.3e} > 1e-10 (body_fn)"
    assert rp <= 1e-10, f"p parity {rp:.3e} > 1e-10 (body_fn)"


def test_monolithic_bubble_smoke():
    """CHNSMonolithicStepper smoke: 20 steps at level 5, no NaN, light-bubble
    centroid strictly rising over the last 5 steps, and (conservative CH
    advection) mass drift machine-exact — bound 1e-11 (the mirror achieves
    0.0; actual printed)."""
    from dataclasses import replace
    from diffsim.steppers.chns import CHNSMonolithicStepper

    level = 5
    dm, mesh, cons = _make_dm(level=level, dim=2)
    case = replace(BUBBLE_RISE_RE35_WE10, rho_ratio=10.0)
    st = CHNSMonolithicStepper(dm, case, dt=case.dt0, Cn_override="2h")

    xc = mesh.node_coords[cons.free_nodes]
    r = np.sqrt((xc[:, 0] - 0.5) ** 2 + (xc[:, 1] - 0.35) ** 2)
    phi0 = -np.tanh((r - 0.2) / (st.Cn * np.sqrt(2.0)))  # phi=-1 light bubble
    st.set_initial(phi0)

    mass0 = st.mass_phi()
    snaps = st.march(t_end=20 * case.dt0)
    assert len(snaps) == 20

    cy = []
    for s in snaps:
        assert np.isfinite(s["phi"]).all(), f"NaN phi at t={s['t']}"
        assert np.isfinite(s["u"]).all(), f"NaN u at t={s['t']}"
        assert np.isfinite(s["p"]).all(), f"NaN p at t={s['t']}"
        cy.append(metrics.centroid_y(-s["phi"], xc))  # bubble = phi<0

    drift = abs(st.mass_phi() - mass0)
    wall = float(np.mean([s["wall_per_step"] for s in snaps]))
    print(f"[monolithic smoke] |mass drift| = {drift:.3e} (bound 1e-11), "
          f"wall/step = {wall:.3f} s, "
          f"newton_iters(last) = {st.last_newton_iters}, "
          f"clamped(last) = {st.last_clamped}")
    assert drift < 1e-11, f"|mass drift| {drift:.3e} >= 1e-11"

    last = np.asarray(cy[-5:])
    assert np.all(np.diff(last) > 0), (
        f"bubble centroid_y not strictly rising over last 5 steps: {last}"
    )


# ---------------------------------------------------------------------------
# Task 10: Conservative Allen-Cahn (CAC) interface variant
# ---------------------------------------------------------------------------
_ATANH_08 = np.arctanh(0.8)      # tanh(z)=0.8 -> z=atanh(0.8)~1.0986


def _interface_eps_from_cut(phi, coords, y_cut, h):
    r"""Effective interface scale eps = sqrt(2)*Cn measured from the 10-90%
    tanh distance along the horizontal cut y ~ y_cut.

    For phi(x) = -tanh((x - x_i)/eps) the value crosses +-0.8 (the 10%-90% band
    of the [-1,1]-scaled tanh) at |x - x_i| = eps*atanh(0.8).  So the raw
    10-90% x-distance is  d = 2*eps*atanh(0.8), and the physical interface
    scale is  eps = d / (2*atanh(0.8)).  Returns eps (directly comparable to
    the analytic sqrt(2)*Cn), or nan if the cut has no clean crossing.
    """
    yv = coords[:, 1]
    y0 = yv[np.argmin(np.abs(yv - y_cut))]
    row = np.where(np.abs(yv - y0) < 0.25 * h)[0]
    if row.size < 4:
        return float("nan")
    xr = coords[row, 0]
    order = np.argsort(xr)
    xr = xr[order]
    pr = phi[row][order]

    def x_at(level):
        # first monotone crossing of phi == level (left interface of the drop)
        for i in range(len(pr) - 1):
            a, b = pr[i], pr[i + 1]
            if (a - level) * (b - level) <= 0 and a != b:
                t = (level - a) / (b - a)
                return xr[i] + t * (xr[i + 1] - xr[i])
        return None
    x_hi = x_at(0.8)
    x_lo = x_at(-0.8)
    if x_hi is None or x_lo is None:
        return float("nan")
    d = abs(x_lo - x_hi)                       # raw 10-90% distance
    return d / (2.0 * _ATANH_08)              # -> eps = sqrt(2)*Cn


@pytest.mark.parametrize("interface", ["ch", "cac"])
def test_cac_parity_kernel_vs_mirror(interface):
    """CHNSMonolithicStepper(interface=) must reproduce CHNSDiscrete(interface=)
    (the numpy mirror) to 1e-10 relative on (phi, u, p) after 3 steps.  Runs
    for BOTH interfaces (the "ch" leg re-covers the existing gate; the "cac"
    leg is the new Task-10 parity gate, incl. the frozen-beta multiplier)."""
    from diffsim.adjoint.chns import CHNSDiscrete
    from diffsim.steppers.chns import CHNSMonolithicStepper

    level = 4
    dm, mesh, cons = _make_dm(level=level, dim=2)
    case = BUBBLE_RISE_RE35_WE10
    dt = case.dt0

    coords = mesh.node_coords
    ref = CHNSDiscrete(level=level, dim=2, case=case, dt=dt, dm=dm,
                       gravity=True, Cn_override="2h", newton_tol=1e-12,
                       interface=interface)
    r = np.sqrt((coords[:, 0] - 0.5) ** 2 + (coords[:, 1] - 0.35) ** 2)
    phi0 = -np.tanh((r - 0.2) / (ref.Cn * np.sqrt(2.0)))
    ref.set_initial(phi0)

    mono = CHNSMonolithicStepper(dm, case, dt=dt, linsolver="splu",
                                 Cn_override="2h", newton_tol=1e-12,
                                 interface=interface)
    mono.set_initial(phi0)

    for _ in range(3):
        ref.step()
        mono.step()

    def relmax(a, b):
        a = np.asarray(a).ravel()
        b = np.asarray(b).ravel()
        return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))

    rphi = relmax(mono.phi, ref.phi)
    ru = relmax(mono.u, ref.u)
    rp = relmax(mono.p, ref.p)
    print(f"[cac parity interface={interface}] rel(phi)={rphi:.3e} "
          f"rel(u)={ru:.3e} rel(p)={rp:.3e} "
          f"newton(last)={mono.last_newton_iters}")
    assert rphi <= 1e-10, f"phi parity {rphi:.3e} > 1e-10 ({interface})"
    assert ru <= 1e-10, f"u parity {ru:.3e} > 1e-10 ({interface})"
    assert rp <= 1e-10, f"p parity {rp:.3e} > 1e-10 ({interface})"


@pytest.mark.parametrize("interface", ["ch", "cac"])
def test_interface_mass_source_conservation(interface):
    """Brief Step-1 gate: with a FIXED Gaussian src_fns blob on the phi-row,
    BOTH interfaces satisfy the source-respecting mass bookkeeping
        | Int phi(t_n) - Int phi(0) - n*dt*Int s | / |Int phi(0)|  <= 1e-6
    (Int via lumped mass, partition of unity), AND the interface width along a
    mid-domain horizontal cut stays within 20% (CAC) / 40% (CH) of the analytic
    sqrt(2)*Cn over the run.  Static source -> Int_0^t Int s dt = n*dt*Int s."""
    from diffsim.steppers.chns import CHNSStepper

    level = 5
    dm, mesh, cons = _make_dm(level=level, dim=2)
    case = BUBBLE_RISE_RE35_WE10
    dt = case.dt0
    coords = mesh.node_coords
    h = float(dm.mesh.tree.h().min())
    Cn = 2.0 * h                                  # Cn_override="2h"

    # Fixed Gaussian blob source on the phi-row (index dim+1 = 3 in 2-D).
    A = 5.0
    x0, y0, rad = 0.5, 0.5, 3.0 * h
    blk = 2 + 3

    def src_phi(xq, t):
        rr2 = (xq[:, 0] - x0) ** 2 + (xq[:, 1] - y0) ** 2
        return A * np.exp(-rr2 / (2.0 * rad ** 2))

    src_fns = [None] * blk
    src_fns[2 + 1] = src_phi

    st = CHNSStepper(dm, case, dt, mode="monolithic", Cn_override="2h",
                     gravity=False, interface=interface, src_fns=src_fns)
    # a resolved tanh drop so an interface exists for the width metric
    r = np.sqrt((coords[:, 0] - x0) ** 2 + (coords[:, 1] - y0) ** 2)
    phi0 = -np.tanh((r - 0.25) / (Cn * np.sqrt(2.0)))
    st.set_initial(phi0)

    M = st.lumped_mass()
    mass0 = float(M @ phi0)
    src_int = float(M @ src_phi(coords, 0.0))     # Int s dOmega (partition of 1)

    n_steps = 10
    widths = []
    for k in range(1, n_steps + 1):
        st.step()
        mass_k = float(M @ st.phi)
        expected = mass0 + k * dt * src_int
        rel = abs(mass_k - expected) / max(abs(mass0), 1e-30)
        assert rel <= 1e-6, (
            f"[{interface}] mass-source bookkeeping rel {rel:.3e} > 1e-6 "
            f"at step {k}")
        w = _interface_eps_from_cut(st.phi, coords, y_cut=0.5, h=h)
        if np.isfinite(w):
            widths.append(w)

    analytic = np.sqrt(2.0) * Cn
    tol = 0.20 if interface == "cac" else 0.40
    w_arr = np.asarray(widths)
    assert w_arr.size > 0, f"[{interface}] no finite interface width measured"
    dev = float(np.max(np.abs(w_arr - analytic) / analytic))
    print(f"[mass-src {interface}] final rel-mass ok; width mean="
          f"{w_arr.mean():.4e} analytic={analytic:.4e} maxdev={dev:.2%} "
          f"(tol {tol:.0%})")
    assert dev <= tol, (
        f"[{interface}] interface width dev {dev:.2%} > {tol:.0%} of "
        f"sqrt(2)Cn={analytic:.3e}")


def test_cac_stationary_drop():
    """CAC stationary drop (gravity off): after 5 steps the parasitic current
    stays BOUNDED (no spurious blow-up) and mass is machine-flat.  The
    parasitic level is REPORTED alongside the CH (potential-form) baseline in
    the SAME config — the Korteweg surface tension is intentionally less
    well-balanced than Jacqmin's potential form, so CAC's max|u| is O(1e-2) vs
    CH's O(1e-4); this ~30x gap is direct A/B evidence (see the decision memo
    docs/dev/2026-08-10-sp0-interface-decision.md), not a failure.  The strict
    parasitic gate lives on the CH mirror (test_chns_discrete_stationary_drop);
    here we bound CAC loosely and assert the well-balanced CH baseline."""
    from diffsim.steppers.chns import CHNSStepper

    level = 5
    dm, mesh, cons = _make_dm(level=level, dim=2)
    coords = mesh.node_coords
    r = np.sqrt((coords[:, 0] - 0.5) ** 2 + (coords[:, 1] - 0.5) ** 2)

    umax = {}
    drift = {}
    for interface in ("ch", "cac"):
        st = CHNSStepper(dm, BUBBLE_RISE_RE35_WE10, dt=1e-3, mode="monolithic",
                         gravity=False, Cn_override="2h", interface=interface)
        phi0 = -np.tanh((r - 0.25) / (st.Cn * np.sqrt(2.0)))
        st.set_initial(phi0)
        M = st.lumped_mass()
        mass0 = float(M @ phi0)
        for _ in range(5):
            st.step()
        umax[interface] = float(np.abs(st.u).max())
        drift[interface] = abs(float(M @ st.phi) - mass0)

    print(f"[stationary drop A/B] parasitic max|u|: CH={umax['ch']:.3e} "
          f"CAC={umax['cac']:.3e} (ratio {umax['cac']/umax['ch']:.1f}x); "
          f"mass drift CH={drift['ch']:.3e} CAC={drift['cac']:.3e}")
    # CH potential form: well-balanced, strict bound (same as the mirror gate).
    assert umax["ch"] < 1e-3, f"CH parasitic max|u|={umax['ch']:.3e} >= 1e-3"
    # CAC Korteweg: bounded (no blow-up) — loose bound documenting the level.
    assert umax["cac"] < 5e-2, f"CAC parasitic max|u|={umax['cac']:.3e} >= 5e-2"
    # both interfaces: mass machine-flat on the closed box.
    for interface in ("ch", "cac"):
        assert drift[interface] < 1e-11 * dm.n_nodes, (
            f"{interface} mass drift {drift[interface]:.3e} >= 1e-11*n")
