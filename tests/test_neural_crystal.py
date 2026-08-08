"""Sub-project 2: NeuralCrystalEnergy — non-parametric coupled free energy."""
import numpy as np
import pytest
from diffsim.adjoint.neural_crystal import NeuralCrystalEnergy, _legendre2_np
from diffsim.adjoint.crystallization_multi import AdditiveCrystalEnergy

pytestmark = pytest.mark.ad


def _chiN(M=2):
    chi = np.zeros((M + 1, M + 1))
    chi[0, 1] = chi[1, 0] = 2.5
    for a in range(M + 1):
        for b in range(a + 1, M + 1):
            if (a, b) != (0, 1):
                chi[a, b] = chi[b, a] = 1.0
    N = 1.0 + 0.3 * np.arange(M + 1)
    return chi, N


def _rand_state(M, K, nn=7, seed=0):
    r = np.random.default_rng(seed)
    phis = [0.15 + 0.1 * r.random(nn) for _ in range(M)]
    psis = [0.2 + 0.3 * r.random(nn) for _ in range(K)]
    return phis, psis


def _mk(M=2, crystallizable=(0,), deg_psi=(1, 2, 3), seed=1):
    chi, N = _chiN(M)
    r = np.random.default_rng(seed)
    coeffs = {f"cpl_{k}_{b}": 0.3 * (r.random() - 0.5)
              for k in crystallizable for b in deg_psi}
    return NeuralCrystalEnergy(chi, N, crystallizable, deg_psi=deg_psi,
                               coeffs=coeffs)


def test_legendre2_matches_finite_diff():
    u = np.linspace(-0.9, 0.9, 11)
    from diffsim.adjoint.neural_multiphase import _legendre_np
    h = 1e-6
    for k in range(0, 6):
        _, dp_plus = _legendre_np(u + h, k)
        _, dp_minus = _legendre_np(u - h, k)
        d2_fd = (dp_plus - dp_minus) / (2 * h)
        d2 = _legendre2_np(u, k)
        assert np.allclose(d2, d2_fd, atol=1e-6), f"L''_{k} mismatch"


def test_coupling_excludes_constant_gauge_mode():
    e = _mk(deg_psi=(1, 2, 3))
    # h_k spanned by L_{b>=1} -> zero projection on the constant mode L_0.
    for k in e.crystallizable:
        assert abs(e.coupling_gauge_residual(k)) < 1e-12
    # and deg_psi=0 is rejected
    chi, N = _chiN()
    with pytest.raises(AssertionError):
        NeuralCrystalEnergy(chi, N, (0,), deg_psi=(0, 1))


def _cs(fn, arr_lists, li, node, h=1e-30):
    """Complex-step one entry of arr_lists[li][node] through fn -> imag/h."""
    pert = [[a.astype(complex) for a in group] for group in arr_lists]
    pert[li[0]][li[1]][node] += 1j * h
    return fn(*[g for g in pert])


def test_protocol_derivs_complex_step():
    e = _mk(M=2, crystallizable=(0,), deg_psi=(1, 2, 3))
    phis, psis = _rand_state(2, 1, seed=3)
    node = 2
    # dfdphi_i wrt phi_j == d2fdphidphi[i][j]; wrt psi == d2fdphidpsi
    H_pp = e.d2fdphidphi(phis, psis)
    H_pc = e.d2fdphidpsi(phis, psis)
    for j in range(2):
        pert = [p.astype(complex) for p in phis]
        pert[j][node] += 1j * 1e-30
        got = [(x.imag / 1e-30) for x in e.dfdphi(pert, psis)]
        for i in range(2):
            assert np.isclose(got[i][node], H_pp[i][j][node], atol=1e-9)
    pert = [p.astype(complex) for p in psis]
    pert[0][node] += 1j * 1e-30
    got = [(x.imag / 1e-30) for x in e.dfdphi(phis, pert)]
    for i in range(2):
        assert np.isclose(got[i][node], H_pc[i][0][node], atol=1e-9)
    # dfdpsi_k wrt psi == d2fdpsidpsi[j][j]
    H_cc = e.d2fdpsidpsi(phis, psis)
    got = [(x.imag / 1e-30) for x in e.dfdpsi(phis, pert)]
    assert np.isclose(got[0][node], H_cc[0][0][node], atol=1e-9)


def test_cpl_param_derivs_complex_step():
    e = _mk(M=2, crystallizable=(0,), deg_psi=(1, 2, 3))
    phis, psis = _rand_state(2, 1, seed=5)
    node = 1
    for nm in ["cpl_0_1", "cpl_0_2", "cpl_0_3"]:
        _, sk, sb = nm.split("_")
        k, b = int(sk), int(sb)
        an_phi = e.dfdphi_dparam(phis, psis, nm)
        an_psi = e.dfdpsi_dparam(phis, psis, nm)
        # complex-step: bump c[(k,b)]
        ec = _mk(M=2, crystallizable=(0,), deg_psi=(1, 2, 3))
        ec.c[(k, b)] = complex(ec.c[(k, b)]) + 1j * 1e-30
        cs_phi = [(x.imag / 1e-30) for x in ec.dfdphi(phis, psis)]
        cs_psi = [(x.imag / 1e-30) for x in ec.dfdpsi(phis, psis)]
        for i in range(2):
            assert np.allclose(an_phi[i], cs_phi[i], atol=1e-9), (nm, "phi", i)
        assert np.allclose(an_psi[0], cs_psi[0], atol=1e-9), (nm, "psi")
