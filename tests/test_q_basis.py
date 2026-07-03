import numpy as np
import pytest
from diffsim.mesh.basis import gauss_1d, basis_tables

pytestmark = pytest.mark.tier3

def test_Q1_polynomial_exactness_1d():
    for p in (1, 2):
        pts, wts = gauss_1d(p)
        for deg in range(2 * (p + 1)):        # 2n-1 exactness: degrees 0..2p+1
            quad = np.sum(wts * pts**deg)
            exact = (1.0 - (-1.0) ** (deg + 1)) / (deg + 1)
            assert abs(quad - exact) < 1e-14, (p, deg)

def test_partition_of_unity_and_derivative_sum():
    for p in (1, 2):
        tb = basis_tables(p)
        assert np.allclose(tb.N.sum(axis=1), 1.0, atol=1e-14)
        assert np.allclose(tb.dN.sum(axis=1), 0.0, atol=1e-13)

def test_M1_local_mass_matrix():
    h = 0.25
    for p in (1, 2):
        tb = basis_tables(p)
        detJxW = tb.w * (h / 2.0) ** 3
        Me = np.einsum("qa,qb,q->ab", tb.N, tb.N, detJxW)
        assert np.allclose(Me, Me.T, atol=1e-15)
        assert abs(Me.sum() - h**3) < 1e-14                       # partition of unity
        assert np.all(np.linalg.eigvalsh(Me) > 0)                 # SPD
        if p == 1:
            assert np.allclose(Me.sum(axis=1), h**3 / 8.0, atol=1e-15)  # row sums

def test_M2_local_stiffness_matrix():
    h = 0.25
    for p in (1, 2):
        tb = basis_tables(p)
        detJxW = tb.w * (h / 2.0) ** 3
        dN = tb.dN * (2.0 / h)
        Ke = np.einsum("qak,qbk,q->ab", dN, dN, detJxW)
        assert np.allclose(Ke, Ke.T, atol=1e-13)
        ev = np.linalg.eigvalsh(Ke)
        assert abs(ev[0]) < 1e-13 and ev[1] > 1e-8                # nullspace = constants only
        assert np.allclose(Ke @ np.ones(tb.nbf), 0.0, atol=1e-13)

def test_lagrange_1d_exact_values():
    from diffsim.mesh.basis import lagrange_1d
    xi = 0.6
    N1, dN1 = lagrange_1d(1, xi)
    assert np.allclose(N1, [0.2, 0.8], atol=1e-15)
    assert np.allclose(dN1, [-0.5, 0.5], atol=1e-15)
    N2, dN2 = lagrange_1d(2, xi)
    assert np.allclose(N2, [0.5 * 0.6 * (0.6 - 1.0), 1.0 - 0.36, 0.5 * 0.6 * 1.6], atol=1e-15)
    assert np.allclose(dN2, [0.6 - 0.5, -1.2, 0.6 + 0.5], atol=1e-15)

def test_dN_axis_identity():
    # dN[q,a,d] must be the derivative along axis d of the tensor basis:
    # verify against a manually assembled entry at one qp for p=2
    from diffsim.mesh.basis import basis_tables, gauss_1d, lagrange_1d
    tb = basis_tables(2)
    pts, _ = gauss_1d(2)
    qi, qj, qk = 2, 0, 1            # arbitrary distinct 1D qp indices
    q = qi + 3 * qj + 9 * qk
    i, j, k = 1, 2, 0               # arbitrary distinct 1D node indices
    a = i + 3 * j + 9 * k
    Nx, dNx = lagrange_1d(2, pts[qi]); Ny, dNy = lagrange_1d(2, pts[qj]); Nz, dNz = lagrange_1d(2, pts[qk])
    assert np.isclose(tb.dN[q, a, 0], dNx[i] * Ny[j] * Nz[k], atol=1e-15)
    assert np.isclose(tb.dN[q, a, 1], Nx[i] * dNy[j] * Nz[k], atol=1e-15)
    assert np.isclose(tb.dN[q, a, 2], Nx[i] * Ny[j] * dNz[k], atol=1e-15)
