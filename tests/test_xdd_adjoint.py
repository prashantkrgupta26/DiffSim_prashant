import numpy as np
import pytest

pytestmark = pytest.mark.ad

from diffsim.xdd.adjoint import ClosureControl
from tests._xdd_adjoint_fixtures import build_small_lit_system  # Task 1 helper


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
