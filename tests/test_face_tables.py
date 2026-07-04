"""Face-restricted basis tables (M1a Task 4): N/dN/d2N of the volume
tensor-product basis at face Gauss points. Face id f = 2*ax + side, side 0 =
minus face (n_tilde = -e_ax), matching face_offsets/BoundaryTypes.WALL order."""
import numpy as np
import pytest
from diffsim.mesh.faces import face_tables

pytestmark = pytest.mark.tier2


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("p", [1, 2])
def test_face_tables_pou_and_quadrature(dim, p):
    ft = face_tables(p, dim)
    nf = 2 * dim
    assert ft.N.shape == (nf, ft.nqf, (p + 1) ** dim)
    assert ft.dN.shape == (nf, ft.nqf, (p + 1) ** dim, dim)
    assert ft.d2N.shape == (nf, ft.nqf, (p + 1) ** dim, dim, dim)
    assert np.allclose(ft.N.sum(axis=2), 1.0, atol=1e-14)          # PoU on every face
    assert np.allclose(ft.dN.sum(axis=2), 0.0, atol=1e-13)
    assert abs(ft.w.sum() - 2.0 ** (dim - 1)) < 1e-13              # reference face measure


@pytest.mark.parametrize("dim", [2, 3])
def test_face_normal_axis_values(dim):
    # On face f the fixed axis sits at +-1: p1 basis values of nodes NOT on
    # that face must vanish (the trace property).
    from diffsim.mesh.nodes import _local_offsets
    ft = face_tables(1, dim)
    offs = _local_offsets(1, dim)
    for ax in range(dim):
        for side in (0, 1):
            f = 2 * ax + side
            off_face = offs[:, ax] != (0 if side == 0 else 1)
            assert np.abs(ft.N[f][:, off_face]).max() < 1e-14


def test_dN_exact_for_linear_on_faces():
    # interpolate u = 2 xi - 3 eta + 0.5 zeta: gradient exact at all face GPs
    from diffsim.mesh.nodes import _local_offsets
    ft = face_tables(1, 3)
    offs = _local_offsets(1, 3)
    xi_n = 2.0 * offs.astype(float) - 1.0
    coef = np.array([2.0, -3.0, 0.5])
    ue = xi_n @ coef
    for f in range(6):
        gN = np.einsum("qad,a->qd", ft.dN[f], ue)
        assert np.allclose(gN, coef, atol=1e-14)


def test_d2N_exact_for_quadratic():
    # p2, dim=2: interpolate u = xi^2 eta + 3 eta^2; Hessian exact at face GPs.
    from diffsim.mesh.nodes import _local_offsets
    ft = face_tables(2, 2)
    offs = _local_offsets(2, 2)
    xi_n = offs.astype(float) - 1.0                                 # nodes at -1, 0, 1
    ue = xi_n[:, 0] ** 2 * xi_n[:, 1] + 3.0 * xi_n[:, 1] ** 2
    for f in range(4):
        xg = np.einsum("qa,ad->qd", ft.N[f], xi_n)                  # face GP coords
        H_exact = np.empty((ft.nqf, 2, 2))
        H_exact[:, 0, 0] = 2 * xg[:, 1]
        H_exact[:, 0, 1] = 2 * xg[:, 0]
        H_exact[:, 1, 0] = 2 * xg[:, 0]
        H_exact[:, 1, 1] = 6.0
        H_num = np.einsum("qaij,a->qij", ft.d2N[f], ue)
        assert np.allclose(H_num, H_exact, atol=1e-12)


def test_p1_d2N_diagonal_zero_mixed_nonzero():
    # Tensor-product Q1: pure second derivatives vanish, but MIXED partials
    # do not (d^2/dxdy of N_x(x)N_y(y) = dN dN) — unlike simplicial P1. The
    # S13.1 representability failure is about the missing diagonal terms.
    d2 = face_tables(1, 3).d2N
    diag = np.stack([d2[..., i, i] for i in range(3)], axis=-1)
    assert np.abs(diag).max() == 0.0
    assert np.abs(d2).max() > 0.1


def test_lagrange_1d_d2():
    from diffsim.mesh.basis import lagrange_1d_d2
    assert np.array_equal(lagrange_1d_d2(1, 0.3), np.zeros(2))
    assert np.allclose(lagrange_1d_d2(2, -0.7), [1.0, -2.0, 1.0], atol=1e-15)


@pytest.mark.parametrize("nq1", [3, 4, 5])
def test_face_tables_quadrature_order_override(nq1):
    # Raised face quadrature (differentiable-safe knob): PoU and the exact
    # face measure hold at any order; point count follows nq1.
    ft = face_tables(1, 2, nq1=nq1)
    assert ft.nqf == nq1
    assert np.allclose(ft.N.sum(axis=2), 1.0, atol=1e-14)
    assert abs(ft.w.sum() - 2.0) < 1e-13
    # a non-polynomial face integrand: integrate exp(xi) over one face; the
    # error drops ~2.5 orders per added point (measured: 6.5e-5 / 3.0e-7 /
    # 8.2e-10 for nq1 = 3/4/5)
    val = (ft.w * np.exp(ft.xi[2, :, 0])).sum()      # face y=-1, integrand exp(x)
    exact = np.e - 1.0 / np.e
    assert abs(val - exact) < 10.0 ** (-2 * nq1 + 2.5)


@pytest.mark.parametrize("p", [1, 2])
def test_face_tables_4d(p):
    # k=4 structural invariants (dim-coverage policy): PoU, face measure,
    # and shape bookkeeping at the space-time dimension.
    ft = face_tables(p, 4)
    assert ft.N.shape == (8, (p + 1) ** 3, (p + 1) ** 4)
    assert np.allclose(ft.N.sum(axis=2), 1.0, atol=1e-14)
    assert np.allclose(ft.dN.sum(axis=2), 0.0, atol=1e-13)
    assert abs(ft.w.sum() - 8.0) < 1e-13                    # reference 3-face measure
