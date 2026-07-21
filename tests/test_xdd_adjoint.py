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
