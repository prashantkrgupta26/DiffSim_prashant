import numpy as np
import pytest

pytestmark = pytest.mark.ad

from diffsim.xdd.adjoint import ClosureControl
from tests._xdd_adjoint_fixtures import build_small_lit_system  # Task 1 helper


# ── Task 4: Mode-A implicit steady adjoint + G2 dot-product ────────────────────

def test_g2_dot_product_transpose(device):
    """G2: ⟨A v, w⟩ = ⟨v, Aᵀ w⟩ to machine precision for the Mode-A operator."""
    sysm, state = build_small_lit_system(device)
    A, _ = sysm.assemble_newton_system(state)          # 5*n_free square CSR
    A = A.tocsr()
    rng = np.random.default_rng(11)
    v = rng.standard_normal(A.shape[0]); w = rng.standard_normal(A.shape[0])
    lhs = float(w @ (A @ v)); rhs = float(v @ (A.T @ w))
    res = abs(lhs - rhs) / abs(lhs + 1e-30)
    print(f"G2 dot-product transpose residual = {res:.2e}")
    assert res < 1e-12, (lhs, rhs)


def test_task4_steady_gradient_closure_vs_fd(device):
    from diffsim.xdd.adjoint import XDDSteadyAdjoint
    from diffsim.xdd.observables import SteadyCurrentQoI
    from tests._xdd_adjoint_fixtures import remarch_steady, steady_polish
    sysm, state = build_small_lit_system(device)
    # Polish onto the exact steady root so the adjoint's base state and the FD
    # leg converge the SAME R_steady(u;p)=0 map (the IFT the adjoint models).
    state = steady_polish(sysm, state)
    ctrl = ClosureControl(sysm, param="langevin_zeta")
    qoi = SteadyCurrentQoI(contact="anode")
    adj = XDDSteadyAdjoint(sysm, [ctrl]); adj.factorize(state)
    g = adj.gradient(state, qoi)["langevin_zeta"][0]
    # FD: re-solve the STEADY implicit problem at zeta ± eps (re-march), compare
    p0 = ctrl.get()[0]; eps = 1e-6 * max(1.0, abs(p0))

    def J_of(z):
        ctrl.set(np.array([z])); st = remarch_steady(sysm, state)
        val = qoi.value(sysm, st); ctrl.set(np.array([p0])); return val
    fd = (J_of(p0 + eps) - J_of(p0 - eps)) / (2 * eps)
    rel = abs(g - fd) / max(abs(fd), 1e-12)
    print(f"Mode-A steady dJ/dzeta adj={g:.6e} fd={fd:.6e} rel={rel:.2e}")
    assert rel < 1e-6, (g, fd)


def test_task4_steady_gradient_mutation_fails(device):
    """GATE HYGIENE: a planted (×1.5) error in the analytic steady gradient must
    make the FD gate FAIL — proves the check is non-vacuous."""
    from diffsim.xdd.adjoint import XDDSteadyAdjoint
    from diffsim.xdd.observables import SteadyCurrentQoI
    from tests._xdd_adjoint_fixtures import remarch_steady, steady_polish
    sysm, state = build_small_lit_system(device)
    state = steady_polish(sysm, state)
    ctrl = ClosureControl(sysm, param="langevin_zeta")
    qoi = SteadyCurrentQoI(contact="anode")
    adj = XDDSteadyAdjoint(sysm, [ctrl]); adj.factorize(state)
    g = adj.gradient(state, qoi)["langevin_zeta"][0]
    p0 = ctrl.get()[0]; eps = 1e-6 * max(1.0, abs(p0))

    def J_of(z):
        ctrl.set(np.array([z])); st = remarch_steady(sysm, state)
        val = qoi.value(sysm, st); ctrl.set(np.array([p0])); return val
    fd = (J_of(p0 + eps) - J_of(p0 - eps)) / (2 * eps)
    rel_good = abs(g - fd) / max(abs(fd), 1e-12)
    assert rel_good < 1e-6, rel_good
    bad = 1.5 * g
    rel_bad = abs(bad - fd) / max(abs(fd), 1e-12)
    print(f"Mode-A gate: good rel = {rel_good:.2e}, mutated(×1.5) rel = {rel_bad:.2e}")
    assert rel_bad > 1e-6, ("steady gate vacuous — mutation not rejected", rel_bad)


def test_task4_sensitivity_rows_reuse_factored_AT(device):
    """Requirement-2 map: sensitivity_rows reuses the SAME factored Aᵀ across
    observable seeds.  Seeded with a QoI's own ∂J/∂u it must reproduce that
    QoI's steady gradient (−λᵀ∂R/∂p, the explicit ∂J/∂p being 0 here), and a
    stacked multi-seed call must equal the per-seed rows."""
    from diffsim.xdd.adjoint import XDDSteadyAdjoint, MaterialControl
    from diffsim.xdd.observables import SteadyCurrentQoI
    from tests._xdd_adjoint_fixtures import steady_polish
    sysm, state = build_small_lit_system(device)
    state = steady_polish(sysm, state)
    c1 = ClosureControl(sysm, param="langevin_zeta")
    c2 = MaterialControl(sysm, param="tau_inv_d")
    qoi = SteadyCurrentQoI(contact="anode")
    adj = XDDSteadyAdjoint(sysm, [c1, c2]); adj.factorize(state)
    grad = adj.gradient(state, qoi)                       # dict per control

    seed = qoi.dJ_du(sysm, state)[None, :]                # (1, 5*n_free)
    rows = adj.sensitivity_rows(state, seed)              # (1, size1+size2)
    assert rows.shape == (1, c1.size + c2.size)
    # column layout ordered by controls: [c1 | c2]
    np.testing.assert_allclose(rows[0, 0], grad["langevin_zeta"][0], rtol=1e-10)
    np.testing.assert_allclose(rows[0, 1], grad["tau_inv_d"][0], rtol=1e-10)

    # stacking two seeds must equal two independent single-seed solves (Aᵀ reuse)
    seed2 = np.vstack([seed[0], -2.0 * seed[0]])
    rows2 = adj.sensitivity_rows(state, seed2)
    np.testing.assert_allclose(rows2[0], rows[0], rtol=1e-10)
    np.testing.assert_allclose(rows2[1], -2.0 * rows[0], rtol=1e-10)
    print(f"sensitivity_rows: row0={rows[0]}  (matches gradient dict)")


def _fd_residual_dp(sysm, state, control, eps_rel=1e-6):
    """Central-FD of the reduced residual w.r.t. the control's scalar param."""
    p0 = control.get().copy()
    eps = eps_rel * max(1.0, abs(float(p0[0])))
    control.set(p0 + eps)
    Rp = _reduced_residual(sysm, state)
    control.set(p0 - eps)
    Rm = _reduced_residual(sysm, state)
    control.set(p0)
    return (Rp - Rm) / (2 * eps)          # shape (5*n_free,)


def _reduced_residual(sysm, state):
    from diffsim.physics.exciton_system import NDOF
    R = sysm.residual_full(state)
    return np.concatenate([np.asarray(sysm.T.T @ R[f]) for f in range(NDOF)])


def test_task1_closure_control_residual_derivative_matches_fd(device):
    sysm, state = build_small_lit_system(device)
    ctrl = ClosureControl(sysm, param="langevin_zeta")
    rows = ctrl.sensitivity_rows(sysm, state)          # (1, 5*n_free)
    fd = _fd_residual_dp(sysm, state, ctrl)            # (5*n_free,)
    analytic = rows[0]
    scale = max(np.abs(fd).max(), 1e-12)
    rel = np.abs(analytic - fd).max() / scale
    print(f"ClosureControl(langevin_zeta) dR/dp adj/fd rel = {rel:.2e}")
    assert rel < 1e-6, (analytic, fd)


# -- Task-1 review fold-in: the Onsager exciton-sink closure params were left
#    untested by Task 1 (only Langevin ζ was exercised).  Gate them here. -----
@pytest.mark.parametrize("param", ("ex_diss_d_scaling", "ex_diss_a_scaling"))
def test_task1_onsager_closure_residual_derivative_matches_fd(device, param):
    sysm, state = build_small_lit_system(device)
    ctrl = ClosureControl(sysm, param=param)
    rows = ctrl.sensitivity_rows(sysm, state)          # (1, 5*n_free)
    fd = _fd_residual_dp(sysm, state, ctrl)            # (5*n_free,)
    analytic = rows[0]
    scale = max(np.abs(fd).max(), 1e-12)
    rel = np.abs(analytic - fd).max() / scale
    print(f"ClosureControl({param}) dR/dp adj/fd rel = {rel:.2e}")
    assert rel < 1e-6, (param, analytic, fd)


# MaterialControl param -> residual-FD gate step.  The carrier µ params use a
# DECORRELATED gate step (1e-4) — deliberately different from the analytic's
# internal directional-derivative step (1e-6) so the semi-analytic carrier-µ
# derivative and its gate are NOT bit-for-bit identical (the gate is a real,
# non-vacuous check).  Closed-form params (ε, exciton-µ, τ⁻¹) use the tight 1e-6.
_MATERIAL_GATE_EPS = {
    "mu_n": 1e-4, "mu_p": 1e-4,
    "mu_x_donor": 1e-6, "mu_x_acceptor": 1e-6,
    "tau_inv_d": 1e-6, "tau_inv_a": 1e-6,
    "eps_A": 1e-6, "eps_D": 1e-6,
}


@pytest.mark.parametrize("param", (
    "mu_n", "mu_p", "mu_x_donor", "mu_x_acceptor",
    "tau_inv_d", "tau_inv_a", "eps_A", "eps_D"))
def test_task2_material_control_residual_derivative_matches_fd(device, param):
    from diffsim.xdd.adjoint import MaterialControl
    sysm, state = build_small_lit_system(device)
    ctrl = MaterialControl(sysm, param=param)
    rows = ctrl.sensitivity_rows(sysm, state)
    fd = _fd_residual_dp(sysm, state, ctrl, eps_rel=_MATERIAL_GATE_EPS[param])
    scale = max(np.abs(fd).max(), 1e-12)
    rel = np.abs(rows[0] - fd).max() / scale
    print(f"MaterialControl({param}) dR/dp adj/fd rel = {rel:.2e}")
    assert rel < 1e-6, (param, rows[0], fd)


def test_task2_material_mu_gate_is_non_vacuous(device):
    """Prove the carrier-µ FD gate can FAIL: plant a wrong (×1.5) factor in the
    analytic mu_n derivative and confirm the gate rejects it.  The gate and the
    analytic use DECORRELATED steps (1e-4 vs 1e-6), so a genuine mismatch of this
    size is caught — the gate is not the tautology it was in the review verdict."""
    from diffsim.xdd.adjoint import MaterialControl
    sysm, state = build_small_lit_system(device)
    ctrl = MaterialControl(sysm, param="mu_n")
    good = ctrl.sensitivity_rows(sysm, state)[0]
    fd = _fd_residual_dp(sysm, state, ctrl, eps_rel=_MATERIAL_GATE_EPS["mu_n"])
    scale = max(np.abs(fd).max(), 1e-12)
    rel_good = np.abs(good - fd).max() / scale
    assert rel_good < 1e-6, rel_good           # correct analytic passes
    bad = 1.5 * good                            # planted wrong factor
    rel_bad = np.abs(bad - fd).max() / scale
    print(f"mu_n gate: good rel = {rel_good:.2e}, mutated(×1.5) rel = {rel_bad:.2e}")
    assert rel_bad > 1e-6, ("gate is vacuous — mutated derivative not rejected",
                            rel_bad)


def test_task2_vector_illumination_residual_derivative_matches_fd(device):
    from diffsim.xdd.adjoint import IlluminationControl
    sysm, state, dist_gp, gen = build_small_lit_system(device, want_gen=True)
    ctrl = IlluminationControl(sysm, dist_gp, gen, mode="vector", n_bands=4)
    rows = ctrl.sensitivity_rows(sysm, state)          # (4, 5*n_free)
    assert rows.shape[0] == 4
    p0 = ctrl.get().copy()
    for j in range(4):                                  # every component
        eps = 1e-6 * max(1.0, abs(float(p0[j])))
        pj = p0.copy(); pj[j] += eps; ctrl.set(pj)
        Rp = _reduced_residual(sysm, state)
        pj = p0.copy(); pj[j] -= eps; ctrl.set(pj)
        Rm = _reduced_residual(sysm, state)
        ctrl.set(p0)
        fd = (Rp - Rm) / (2 * eps)
        scale = max(np.abs(fd).max(), 1e-12)
        rel = np.abs(rows[j] - fd).max() / scale
        print(f"IlluminationControl[band {j}] dR/dp adj/fd rel = {rel:.2e}")
        assert rel < 1e-6, (j, rows[j], fd)


# ── Task 5: Mode-B taped frozen-dt transient adjoint ───────────────────────────

def test_g3_checkpoint_equivalence(device):
    """G3a: the checkpointing march reproduces the plain march trajectory
    (converged state per step is bit-identical) — the record adds no drift."""
    from diffsim.xdd.run import march_with_checkpoints, _march_to_steady
    sysm, state = build_small_lit_system(device)
    # Plain march: fixed step count (time_stepping_tol=0 disables early stop).
    fs_plain, info = _march_to_steady(sysm, state, dt0_hat=1e-4, dt_max_hat=1e-2,
                                      max_steps=12, time_stepping_tol=0.0)
    sysm2, state2 = build_small_lit_system(device)
    fs_ckpt, steps = march_with_checkpoints(sysm2, state2, dt0_hat=1e-4,
                                            dt_max_hat=1e-2, max_steps=12, order=1)
    from diffsim.physics.exciton_system import NDOF
    for f in range(NDOF):
        assert np.allclose(fs_plain[f], fs_ckpt[f], atol=1e-12, rtol=0), f
    # frozen-dt tape sanity: recorded dt's are the FORWARD-adapted values and the
    # sigma per step is exactly 1/dt_hat (BDF1) — the reverse sweep freezes these.
    for s in steps:
        assert abs(s["sigma"] - 1.0 / s["dt_hat"]) < 1e-14, s["dt_hat"]


def _traj_J(all_steps):
    from diffsim.physics.exciton_system import IXD
    return 0.5 * sum(float(s["state"][IXD] @ s["state"][IXD]) for s in all_steps)


def _traj_dJdx(sysm, steps):
    from diffsim.physics.exciton_system import IXD, NDOF
    dJdx = []
    for s in steps:
        seed = {f: np.zeros(sysm.dm.n_nodes) for f in range(NDOF)}
        seed[IXD] = s["state"][IXD]
        dJdx.append(np.concatenate([np.asarray(sysm.T.T @ seed[f])
                                    for f in range(NDOF)]))
    return dJdx


def test_task5_transient_gradient_lifetime_vs_fd(device):
    """G3b: frozen-dt reverse sweep reproduces the tape gradient (independent FD
    leg re-runs the FULL forward march per perturbation)."""
    from diffsim.xdd.run import march_with_checkpoints
    from diffsim.xdd.adjoint import XDDTransientAdjoint, MaterialControl
    sysm, state = build_small_lit_system(device)
    ctrl = MaterialControl(sysm, param="tau_inv_d")
    N = 6
    fs, steps = march_with_checkpoints(sysm, state, dt0_hat=1e-4,
                                       dt_max_hat=1e-2, max_steps=N, order=1)
    # frozen-dt: the reverse sweep uses the RECORDED per-step dt (σ=1/dt) and
    # never re-adapts.  (A longer march genuinely adapts dt off the log
    # schedule — see test_task5_frozen_dt_adaptation; here the tape's σ/dt tie
    # is the contract the reverse freezes.)
    for s in steps:
        assert abs(s["sigma"] - 1.0 / s["dt_hat"]) < 1e-14
    dJdx = _traj_dJdx(sysm, steps)
    adj = XDDTransientAdjoint(sysm, [ctrl], steps)
    g = adj.gradient(dJdx)["tau_inv_d"][0]
    p0 = ctrl.get()[0]; eps = 1e-6 * max(1.0, abs(p0))

    def J_of(v):
        ctrl.set(np.array([v]))
        _, st2 = march_with_checkpoints(sysm, state, dt0_hat=1e-4,
                                        dt_max_hat=1e-2, max_steps=N, order=1)
        val = _traj_J(st2); ctrl.set(np.array([p0])); return val
    fd = (J_of(p0 + eps) - J_of(p0 - eps)) / (2 * eps)
    rel = abs(g - fd) / max(abs(fd), 1e-12)
    print(f"Mode-B transient dJ/dtau adj={g:.6e} fd={fd:.6e} rel={rel:.2e}")
    assert rel < 5e-5, (g, fd)


def test_task5_transient_gradient_mutation_fails(device):
    """GATE HYGIENE: a planted (×1.5) error in the transient adjoint gradient must
    make the FD gate FAIL — proves the check is non-vacuous."""
    from diffsim.xdd.run import march_with_checkpoints
    from diffsim.xdd.adjoint import XDDTransientAdjoint, MaterialControl
    sysm, state = build_small_lit_system(device)
    ctrl = MaterialControl(sysm, param="tau_inv_d")
    N = 6
    fs, steps = march_with_checkpoints(sysm, state, dt0_hat=1e-4,
                                       dt_max_hat=1e-2, max_steps=N, order=1)
    dJdx = _traj_dJdx(sysm, steps)
    adj = XDDTransientAdjoint(sysm, [ctrl], steps)
    g = adj.gradient(dJdx)["tau_inv_d"][0]
    p0 = ctrl.get()[0]; eps = 1e-6 * max(1.0, abs(p0))

    def J_of(v):
        ctrl.set(np.array([v]))
        _, st2 = march_with_checkpoints(sysm, state, dt0_hat=1e-4,
                                        dt_max_hat=1e-2, max_steps=N, order=1)
        val = _traj_J(st2); ctrl.set(np.array([p0])); return val
    fd = (J_of(p0 + eps) - J_of(p0 - eps)) / (2 * eps)
    rel_good = abs(g - fd) / max(abs(fd), 1e-12)
    bad = 1.5 * g
    rel_bad = abs(bad - fd) / max(abs(fd), 1e-12)
    print(f"Mode-B mutation gate: good rel={rel_good:.2e} mutated(×1.5) rel={rel_bad:.2e}")
    assert rel_good < 5e-5, rel_good
    assert rel_bad > 5e-5, ("gate is vacuous — mutated gradient not rejected",
                            rel_bad)


def test_task5_frozen_dt_adaptation(device):
    """Frozen-dt contract: the FORWARD march adapts dt off the log schedule, and
    the recorded tape freezes those exact per-step dt's (σₙ = 1/dtₙ) — the
    reverse sweep rebuilds Aₙ at the recorded dt and NEVER calls _dt_schedule."""
    from diffsim.xdd.run import march_with_checkpoints, _dt_schedule
    sysm, state = build_small_lit_system(device)
    # dt0=1e-3, dt_max=1e-1: after t crosses 1e-2, 1e-1 the schedule grows dt.
    fs, steps = march_with_checkpoints(sysm, state, dt0_hat=1e-3,
                                       dt_max_hat=1e-1, max_steps=40, order=1)
    dts = [s["dt_hat"] for s in steps]
    assert max(dts) > min(dts) + 1e-30, ("forward march did not adapt dt", dts)
    # every recorded step ties σ=1/dt (frozen BDF1 coefficient)
    for s in steps:
        assert abs(s["sigma"] - 1.0 / s["dt_hat"]) < 1e-14
    # the recorded dt sequence equals the deterministic forward schedule replayed
    # (proves the tape is the genuine adaptive-forward dt, not a re-derived one).
    t = 0.0; dt = 1e-3
    for s in steps:
        assert abs(s["dt_hat"] - dt) < 1e-14, (s["dt_hat"], dt)
        t += dt
        dt = max(_dt_schedule(t, dt, 1e-3, 1e-1), dt)
    print(f"frozen-dt: {len(steps)} steps, dt {min(dts):.1e}→{max(dts):.1e} (adapted)")


# ══════════════════════════════════════════════════════════════════════════════
# Task 6 fold-in: carrier-param (µ_n) transient gate + BDF2 transient gate
# ══════════════════════════════════════════════════════════════════════════════

def test_task6_transient_gradient_mu_n_vs_fd(device):
    """Carrier-parameter transient gate: dJ/dµ_n through Mode B exercises the
    carrier SUPG history operators (Hload_n with the drift/τ_M-consistent
    kernel), which the exciton-sink tau_inv_d gate never touches.  Adjoint vs an
    INDEPENDENT full-re-march central FD ≤ 5e-5 (µ_n's analytic is the semi-
    analytic decorrelated derivative — DIFFERENT internal step than the FD)."""
    from diffsim.xdd.run import march_with_checkpoints
    from diffsim.xdd.adjoint import XDDTransientAdjoint, MaterialControl
    sysm, state = build_small_lit_system(device)
    ctrl = MaterialControl(sysm, param="mu_n")
    N = 6
    fs, steps = march_with_checkpoints(sysm, state, dt0_hat=1e-4,
                                       dt_max_hat=1e-2, max_steps=N, order=1)
    dJdx = _traj_dJdx(sysm, steps)
    adj = XDDTransientAdjoint(sysm, [ctrl], steps)
    g = adj.gradient(dJdx)["mu_n"][0]
    p0 = ctrl.get()[0]; eps = 1e-4 * max(1.0, abs(p0))

    def J_of(v):
        ctrl.set(np.array([v]))
        _, st2 = march_with_checkpoints(sysm, state, dt0_hat=1e-4,
                                        dt_max_hat=1e-2, max_steps=N, order=1)
        val = _traj_J(st2); ctrl.set(np.array([p0])); return val
    fd = (J_of(p0 + eps) - J_of(p0 - eps)) / (2 * eps)
    rel = abs(g - fd) / max(abs(fd), 1e-12)
    print(f"\nMode-B transient dJ/dµ_n adj={g:.6e} fd={fd:.6e} rel={rel:.2e}")
    assert rel < 5e-5, (g, fd)


def test_task6_transient_gradient_mu_n_mutation_fails(device):
    """GATE HYGIENE for the µ_n transient gate: a planted (×1.5) error must make
    the FD comparison FAIL — proves the carrier-history check is non-vacuous."""
    from diffsim.xdd.run import march_with_checkpoints
    from diffsim.xdd.adjoint import XDDTransientAdjoint, MaterialControl
    sysm, state = build_small_lit_system(device)
    ctrl = MaterialControl(sysm, param="mu_n")
    N = 6
    fs, steps = march_with_checkpoints(sysm, state, dt0_hat=1e-4,
                                       dt_max_hat=1e-2, max_steps=N, order=1)
    dJdx = _traj_dJdx(sysm, steps)
    adj = XDDTransientAdjoint(sysm, [ctrl], steps)
    g = adj.gradient(dJdx)["mu_n"][0]
    p0 = ctrl.get()[0]; eps = 1e-4 * max(1.0, abs(p0))

    def J_of(v):
        ctrl.set(np.array([v]))
        _, st2 = march_with_checkpoints(sysm, state, dt0_hat=1e-4,
                                        dt_max_hat=1e-2, max_steps=N, order=1)
        val = _traj_J(st2); ctrl.set(np.array([p0])); return val
    fd = (J_of(p0 + eps) - J_of(p0 - eps)) / (2 * eps)
    rel_good = abs(g - fd) / max(abs(fd), 1e-12)
    rel_bad = abs(1.5 * g - fd) / max(abs(fd), 1e-12)
    print(f"\nMode-B µ_n mutation gate: good rel={rel_good:.2e} "
          f"mutated(×1.5) rel={rel_bad:.2e}")
    assert rel_good < 5e-5, rel_good
    assert rel_bad > 5e-5, ("gate vacuous — mutated µ_n gradient not rejected",
                            rel_bad)


def test_task6_transient_gradient_bdf2_vs_fd(device):
    """BDF2 transient gate: drive order=2 so the tape carries genuine BDF2 steps
    (hist = (2uⁿ − 0.5uⁿ⁻¹)/dt, coeffs 2/dt on uⁿ and −0.5/dt on uⁿ⁻¹, the n−2
    prev2 cotangent).  Confirms the frozen-dt reverse handles a BDF2 step: adj
    vs INDEPENDENT full-re-march central FD ≤ 5e-5, mutation-verified."""
    from diffsim.xdd.run import march_with_checkpoints
    from diffsim.xdd.adjoint import XDDTransientAdjoint, MaterialControl
    sysm, state = build_small_lit_system(device)
    ctrl = MaterialControl(sysm, param="tau_inv_d")
    N = 6
    fs, steps = march_with_checkpoints(sysm, state, dt0_hat=1e-4,
                                       dt_max_hat=1e-2, max_steps=N, order=2)
    # the tape MUST contain at least one true BDF2 step (order==2 with a prev2),
    # else this gate would silently degrade to a BDF1 check.
    n_bdf2 = sum(1 for s in steps if s["order"] == 2 and s["prev2"] is not None)
    assert n_bdf2 >= 1, ("march did not produce a BDF2 step", [s["order"] for s in steps])
    dJdx = _traj_dJdx(sysm, steps)
    adj = XDDTransientAdjoint(sysm, [ctrl], steps)
    g = adj.gradient(dJdx)["tau_inv_d"][0]
    p0 = ctrl.get()[0]; eps = 1e-6 * max(1.0, abs(p0))

    def J_of(v):
        ctrl.set(np.array([v]))
        _, st2 = march_with_checkpoints(sysm, state, dt0_hat=1e-4,
                                        dt_max_hat=1e-2, max_steps=N, order=2)
        val = _traj_J(st2); ctrl.set(np.array([p0])); return val
    fd = (J_of(p0 + eps) - J_of(p0 - eps)) / (2 * eps)
    rel_good = abs(g - fd) / max(abs(fd), 1e-12)
    rel_bad = abs(1.5 * g - fd) / max(abs(fd), 1e-12)
    print(f"\nMode-B BDF2 transient ({n_bdf2} BDF2 steps) dJ/dtau adj={g:.6e} "
          f"fd={fd:.6e} rel={rel_good:.2e} mutated rel={rel_bad:.2e}")
    assert rel_good < 5e-5, (g, fd)
    assert rel_bad > 5e-5, ("BDF2 gate vacuous — mutated gradient not rejected",
                            rel_bad)
