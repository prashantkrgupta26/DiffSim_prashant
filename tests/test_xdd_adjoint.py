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


def test_task2_material_control_residual_derivative_matches_fd(device):
    from diffsim.xdd.adjoint import MaterialControl
    sysm, state = build_small_lit_system(device)
    for param in ("mu_n", "tau_inv_d"):
        ctrl = MaterialControl(sysm, param=param)
        rows = ctrl.sensitivity_rows(sysm, state)
        fd = _fd_residual_dp(sysm, state, ctrl)
        scale = max(np.abs(fd).max(), 1e-12)
        rel = np.abs(rows[0] - fd).max() / scale
        print(f"MaterialControl({param}) dR/dp adj/fd rel = {rel:.2e}")
        assert rel < 1e-6, (param, rows[0], fd)


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
