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


def test_reduction_matches_additive_up_to_gauge():
    from diffsim.adjoint.crystallization_multi import AdditiveCrystalEnergy
    M, cryst = 2, (0,)
    chi, N = _chiN(M)
    dsig, dh, Tm, T = {0: 0.7}, {0: -0.9}, {0: 1.1}, 0.5
    add = AdditiveCrystalEnergy(chi, N, cryst, dsig, dh, Tm, T=T)

    # project h_add onto shifted-Legendre L_b(u=2psi-1) for b in 1..4
    from diffsim.adjoint.neural_multiphase import _legendre_np
    x, w = np.polynomial.legendre.leggauss(64)
    psi_q = 0.5 * (x + 1.0)
    drive = dh[0] * (T / Tm[0] - 1.0)
    h_add = (psi_q ** 2 * (1 - psi_q) ** 2) * dsig[0] + \
            (3 * psi_q ** 2 - 2 * psi_q ** 3) * drive
    deg = (1, 2, 3, 4)
    coeffs = {}
    for b in deg:
        Lb, _ = _legendre_np(x, b)
        coeffs[f"cpl_0_{b}"] = float((w * h_add * Lb).sum()
                                     / (w * Lb * Lb).sum())
    neu = NeuralCrystalEnergy(chi, N, cryst, deg_psi=deg, coeffs=coeffs,
                              basis_degrees=(2, 3))  # basis coeffs 0 -> f_base == FH

    phis, psis = _rand_state(M, 1, seed=7)
    # dfdpsi exact (constant-independent)
    assert np.allclose(neu.dfdpsi(phis, psis)[0],
                       add.dfdpsi(phis, psis)[0], atol=1e-11)
    # Hessian blocks exact
    assert np.allclose(neu.d2fdphidpsi(phis, psis)[0][0],
                       add.d2fdphidpsi(phis, psis)[0][0], atol=1e-11)
    assert np.allclose(neu.d2fdpsidpsi(phis, psis)[0][0],
                       add.d2fdpsidpsi(phis, psis)[0][0], atol=1e-11)
    # dfdphi matches up to a per-species constant (the b=0 gauge)
    dneu = neu.dfdphi(phis, psis)[0]
    dadd = add.dfdphi(phis, psis)[0]
    diff = dneu - dadd
    assert np.allclose(diff - diff.mean(), 0.0, atol=1e-9)  # only a constant differs


def test_twin_cpl_grads_vs_fd():
    import torch
    from diffsim.adjoint.torch_twin import CrystalCHTwin
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh
    tree = build_uniform(2, dim=2)
    mesh = build_mesh(tree, p=1)
    dm = DeviceMesh.from_mesh(mesh, build_constraints(mesh),
                              basis_tables(1, dim=2), "cpu")
    M, cryst, deg = 2, (0,), (1, 2)
    chi, N = _chiN(M)
    coeffs = {"cpl_0_1": 0.12, "cpl_0_2": -0.08}
    neu = NeuralCrystalEnergy(chi, N, cryst, deg_psi=deg, coeffs=coeffs)
    nn = dm.n_nodes
    cc = np.cos(np.pi * mesh.node_coords[:, 0])
    phi0 = [0.28 + 0.03 * cc, 0.30 + 0.03 * cc]
    psi0 = [0.25 + 0.02 * cc]
    eng = dict(chi=chi, N=N, dsig={0: 0.0}, dh={0: 0.0}, Tm={0: 1.0}, T=0.5)
    enp = dict(onsager=np.eye(M), kappa=[0.01, 0.02], eps2={0: 0.015}, L={0: 1.2})
    names = ["cpl_0_1", "cpl_0_2"]
    tw = CrystalCHTwin(dm, M, cryst, dt=0.01, order=1, device="cpu")
    g = tw.grads(phi0, psi0, eng, enp, 3, names, 0.28, neural_energy=neu)
    # central FD via the twin's own detached forward at perturbed coeffs
    def loss_at(cmod):
        e2 = NeuralCrystalEnergy(chi, N, cryst, deg_psi=deg, coeffs=cmod)
        return tw.loss_only(phi0, psi0, eng, enp, 3, 0.28, neural_energy=e2)
    for nm in names:
        base = dict(coeffs)
        base[nm] += 1e-6; lp = loss_at(base)
        base[nm] -= 2e-6; lm = loss_at(base)
        fd = (lp - lm) / (2e-6)
        assert abs(g[nm] - fd) / max(abs(fd), 1e-12) < 1e-6, (nm, g[nm], fd)
