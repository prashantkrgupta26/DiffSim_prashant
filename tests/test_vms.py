"""M1b Task 3 gates: tau references (hand-computed), device/host parity,
and THE adjoint identity of M_{a,s} — the s=1/2 energy-stability and M1c
adjoint mechanism, locked before any NS kernel exists."""
import numpy as np
import pytest
import warp as wp
from diffsim.physics.vms import (tau_metric_host, tau_hbased_host,
                                 tau_m_metric, tau_c_metric, tau_m_hbased,
                                 advection_matrix_dense, CI_F)

pytestmark = pytest.mark.tier3


def test_tau_metric_hand_reference():
    # h=0.25, |u|=2, nu=0.01, dt=0.1, b0=1.5 (BDF2), dim=2
    h, u, nu, dt, b0 = 0.25, 2.0, 0.01, 0.1, 1.5
    trans = (2 * b0 / dt) ** 2                     # 900
    uGu = 4 * u ** 2 / h ** 2                      # 256
    GG = 2 * (2 / h) ** 4                          # 8192
    tauM_ref = 1 / np.sqrt(trans + uGu + 36 * nu ** 2 * GG)
    tauC_ref = 1 / (tauM_ref * 2 * 4 / h ** 2)
    tauM, tauC = tau_metric_host(u, h, nu, dt=dt, b0=b0, dim=2)
    assert abs(tauM - tauM_ref) < 1e-15
    assert abs(tauC - tauC_ref) < 1e-15
    # timestab off drops the transient term
    tauM_s, _ = tau_metric_host(u, h, nu, dt=dt, b0=b0, dim=2, timestab=False)
    assert abs(tauM_s - 1 / np.sqrt(uGu + 36 * nu ** 2 * GG)) < 1e-15


def test_tau_hbased_matches_metric_on_cubes():
    # draft-faithful defaults coincide with the metric form on our cubes
    for dim in (2, 3):
        for (u, h, nu, dt, b0) in ((1.3, 0.125, 0.02, 0.05, 1.0),
                                   (0.0, 0.5, 1.0, None, 1.0)):
            tm, _ = tau_metric_host(u, h, nu, dt=dt, b0=b0, dim=dim)
            th = tau_hbased_host(u, h, nu, dt=dt, b0=b0, dim=dim)
            assert abs(tm - th) < 1e-15 * max(tm, th), (dim, u, tm, th)


def test_tau_device_matches_host(device):
    vals = np.array([[2.0, 0.25, 0.01, 900.0],
                     [0.7, 0.125, 0.05, 0.0],
                     [0.0, 0.5, 1.0, 16.0]])
    out = wp.zeros(6, dtype=wp.float64, device=device)
    vd = wp.array(vals, dtype=wp.float64, device=device)

    @wp.kernel(module="unique", enable_backward=False)
    def probe(v: wp.array2d(dtype=wp.float64),
              o: wp.array(dtype=wp.float64)):
        i = wp.tid()
        tm = tau_m_metric(v[i, 0], v[i, 1], v[i, 2], v[i, 3], wp.float64(2.0))
        o[2 * i] = tm
        o[2 * i + 1] = tau_c_metric(tm, v[i, 1], wp.float64(2.0))

    wp.launch(probe, dim=3, inputs=[vd, out], device=device)
    o = out.numpy()
    for i, (u, h, nu, sig2) in enumerate(vals):
        uGu = 4 * u ** 2 / h ** 2
        GG = 2 * (2 / h) ** 4
        tm_ref = 1 / np.sqrt(sig2 + uGu + CI_F * nu ** 2 * GG)
        assert abs(o[2 * i] - tm_ref) < 1e-15
        assert abs(o[2 * i + 1] - 1 / (tm_ref * 2 * 4 / h ** 2)) < 1e-15


A_FN = lambda x, y: (x * (1 - x) * (1 + 0.3 * y), y * (1 - y) * (1 - 0.2 * x))
DIV_A = lambda x, y: (1 - 2 * x) * (1 + 0.3 * y) + (1 - 2 * y) * (1 - 0.2 * x)


@pytest.mark.parametrize("s", [0.0, 0.5, 1.0])
def test_advection_adjoint_identity(s):
    """<M_{a,s}u, w> = <u, -M_{a,1-s}w> for a.n = 0 on the boundary:
    A_{a,s} = -A_{a,1-s}^T exactly (quadrature-exact polynomial a)."""
    A_s = advection_matrix_dense(9, 2, A_FN, DIV_A, s, nq=4)
    A_1ms = advection_matrix_dense(9, 2, A_FN, DIV_A, 1.0 - s, nq=4)
    scale = np.abs(A_s).max()
    assert np.abs(A_s + A_1ms.T).max() < 1e-14 * scale


def test_advection_s_half_skew_symmetric():
    # s = 1/2 is exactly skew-symmetric => <M u, u> = 0: THE energy-stability
    # mechanism (draft Prop. 2) and the self-adjointness M1c relies on
    A_h = advection_matrix_dense(9, 2, A_FN, DIV_A, 0.5, nq=4)
    scale = np.abs(A_h).max()
    assert np.abs(A_h + A_h.T).max() < 1e-14 * scale
    rng = np.random.default_rng(2)
    u = rng.standard_normal(A_h.shape[0])
    assert abs(u @ A_h @ u) < 1e-13 * scale * (u @ u)


def test_advection_s0_s1_energy_term_sign():
    # for s != 1/2 the symmetric part is +-(s - 1/2)(div a) mass-weighted;
    # with div a != 0 it is NONZERO — the degenerate-MMS rule: assert the
    # mechanism is live, not just the identity
    A_0 = advection_matrix_dense(9, 2, A_FN, DIV_A, 0.0, nq=4)
    sym = 0.5 * (A_0 + A_0.T)
    assert np.abs(sym).max() > 1e-3 * np.abs(A_0).max()
