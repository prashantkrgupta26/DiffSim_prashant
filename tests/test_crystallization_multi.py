"""Tests for CrystalEnergy protocol + AdditiveCrystalEnergy (Task 1).

TDD RED→GREEN sequence:
  1. test_crystal_energy_dfdphi_matches_fh_plus_coupling
  2. test_crystal_energy_derivs_complex_step
  3. test_crystal_energy_cross_hessian_d2fdphidpsi  (extra: pins the cross coupling)
  4. test_crystal_energy_d2fdphidphi_equals_fh      (extra: chi is psi-independent in v1)
"""
import numpy as np
import pytest
from diffsim.adjoint.crystallization_multi import AdditiveCrystalEnergy
from diffsim.adjoint import FHMultiEnergy


def _chiN(M=2):
    chi = np.zeros((M + 1, M + 1))
    chi[0, 1] = chi[1, 0] = 2.5
    chi[0, 2] = chi[2, 0] = 1.0
    chi[1, 2] = chi[2, 1] = 0.8
    return chi, np.ones(M + 1)


def test_crystal_energy_dfdphi_matches_fh_plus_coupling():
    chi, N = _chiN()
    K = (0,)
    en = AdditiveCrystalEnergy(chi, N, crystallizable=K,
                               dsig={0: 1.2}, dh={0: -1.0}, Tm={0: 1.0}, T=0.5)
    fh = FHMultiEnergy(chi, N)
    phis = [np.full(4, 0.30), np.full(4, 0.32)]
    psis = [np.full(4, 0.4)]
    mu = en.dfdphi(phis, psis)
    # species 0 crystallizes: dfdphi_0 = mu_FH_0 + q(psi)*dsig + p(psi)*drive
    q = 0.4 ** 2 * (1 - 0.4) ** 2
    p = 0.4 ** 2 * (3 - 2 * 0.4)
    drive = -1.0 * (0.5 / 1.0 - 1.0)
    assert np.allclose(mu[0], fh.mu(phis)[0] + q * 1.2 + p * drive)
    assert np.allclose(mu[1], fh.mu(phis)[1])   # species 1 no crystal term


def test_crystal_energy_derivs_complex_step():
    chi, N = _chiN()
    K = (0, 1)
    en = AdditiveCrystalEnergy(chi, N, crystallizable=K,
                               dsig={0: 1.2, 1: 0.8},
                               dh={0: -1.0, 1: -1.3},
                               Tm={0: 1.0, 1: 1.1}, T=0.5)
    phis = [np.array([0.30, 0.28]), np.array([0.32, 0.34])]
    psis = [np.array([0.4, 0.5]), np.array([0.3, 0.45])]
    h = 1e-30
    # d(dfdpsi_0)/dpsi_0 vs d2fdpsidpsi[0][0]
    pc = [p.astype(complex) for p in psis]
    pc[0] = pc[0] + 1j * h
    cs = (en.dfdpsi(phis, pc)[0].imag) / h
    an = en.d2fdpsidpsi(phis, psis)[0][0]
    assert np.allclose(cs, an, atol=1e-9)
    # d(dfdphi_0)/d(dsig_0) via complex-step on the param vs dfdphi_dparam
    en.dsig[0] = en.dsig[0] + 1j * h
    cs2 = (en.dfdphi(phis, psis)[0].imag) / h
    en.dsig[0] = en.dsig[0].real
    assert np.allclose(cs2, en.dfdphi_dparam(phis, psis, "dsig_0")[0], atol=1e-9)

    # d(dfdphi)/d(dh_k) complex-step tests
    for k in K:
        en.dh[k] = complex(en.dh[k]) + 1j * h
        cs_dh = (en.dfdphi(phis, psis)[k].imag) / h
        en.dh[k] = en.dh[k].real
        an_dh = en.dfdphi_dparam(phis, psis, f"dh_{k}")[k]
        assert np.allclose(cs_dh, an_dh, atol=1e-9), \
            f"dfdphi_dparam(dh_{k}) mismatch: cs={cs_dh}, an={an_dh}"

    # d(dfdpsi)/d(dh_k) complex-step tests
    for j, k in enumerate(K):
        en.dh[k] = complex(en.dh[k]) + 1j * h
        cs_dh_psi = (en.dfdpsi(phis, psis)[j].imag) / h
        en.dh[k] = en.dh[k].real
        an_dh_psi = en.dfdpsi_dparam(phis, psis, f"dh_{k}")[j]
        assert np.allclose(cs_dh_psi, an_dh_psi, atol=1e-9), \
            f"dfdpsi_dparam(dh_{k}) mismatch: cs={cs_dh_psi}, an={an_dh_psi}"

    # d(dfdphi)/d(Tm_k) complex-step tests
    for k in K:
        en.Tm[k] = complex(en.Tm[k]) + 1j * h
        cs_Tm = (en.dfdphi(phis, psis)[k].imag) / h
        en.Tm[k] = en.Tm[k].real
        an_Tm = en.dfdphi_dparam(phis, psis, f"Tm_{k}")[k]
        assert np.allclose(cs_Tm, an_Tm, atol=1e-9), \
            f"dfdphi_dparam(Tm_{k}) mismatch: cs={cs_Tm}, an={an_Tm}"

    # d(dfdpsi)/d(Tm_k) complex-step tests
    for j, k in enumerate(K):
        en.Tm[k] = complex(en.Tm[k]) + 1j * h
        cs_Tm_psi = (en.dfdpsi(phis, psis)[j].imag) / h
        en.Tm[k] = en.Tm[k].real
        an_Tm_psi = en.dfdpsi_dparam(phis, psis, f"Tm_{k}")[j]
        assert np.allclose(cs_Tm_psi, an_Tm_psi, atol=1e-9), \
            f"dfdpsi_dparam(Tm_{k}) mismatch: cs={cs_Tm_psi}, an={an_Tm_psi}"


def test_crystal_energy_cross_hessian_d2fdphidpsi():
    """Perturb psi_k imaginary, check d(dfdphi_k)/dpsi_k == d2fdphidpsi[k][j].

    This pins the cross-coupling the engine's Jacobian will rely on.
    """
    chi, N = _chiN()
    K = (0, 1)
    en = AdditiveCrystalEnergy(chi, N, crystallizable=K,
                               dsig={0: 1.2, 1: 0.8},
                               dh={0: -1.0, 1: -1.3},
                               Tm={0: 1.0, 1: 1.1}, T=0.5)
    phis = [np.array([0.30, 0.28]), np.array([0.32, 0.34])]
    psis = [np.array([0.4, 0.5]), np.array([0.3, 0.45])]
    h = 1e-30
    H = en.d2fdphidpsi(phis, psis)

    # for each crystallizable species, perturb its psi and check d(dfdphi)/dpsi
    for j, k in enumerate(K):
        pc = [p.astype(complex) for p in psis]
        pc[j] = pc[j] + 1j * h
        dfdphi_pert = en.dfdphi(phis, pc)
        cs = dfdphi_pert[k].imag / h
        # H[k][j] is d(dfdphi_k)/d(psi_j)
        assert np.allclose(cs, H[k][j], atol=1e-9), \
            f"Cross Hessian mismatch for species k={k}, j={j}: cs={cs}, an={H[k][j]}"


def test_crystal_energy_d2fdphidphi_equals_fh():
    """d2fdphidphi reduces to FHMultiEnergy.dmu_dphi (chi is psi-independent in v1)."""
    chi, N = _chiN()
    K = (0, 1)
    en = AdditiveCrystalEnergy(chi, N, crystallizable=K,
                               dsig={0: 1.2, 1: 0.8},
                               dh={0: -1.0, 1: -1.3},
                               Tm={0: 1.0, 1: 1.1}, T=0.5)
    fh = FHMultiEnergy(chi, N)
    phis = [np.array([0.30, 0.28]), np.array([0.32, 0.34])]
    psis = [np.array([0.4, 0.5]), np.array([0.3, 0.45])]

    H_crystal = en.d2fdphidphi(phis, psis)
    H_fh = fh.dmu_dphi(phis)

    M = 2
    for i in range(M):
        for j in range(M):
            assert np.allclose(H_crystal[i][j], H_fh[i][j], atol=1e-12), \
                f"d2fdphidphi[{i}][{j}] != FH dmu_dphi[{i}][{j}]"
