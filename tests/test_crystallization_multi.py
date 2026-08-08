"""Tests for CrystalEnergy protocol + AdditiveCrystalEnergy (Task 1) and
CrystalCHDiscrete 2M+K block residual/Jacobian (Task 2).

TDD RED→GREEN sequence:
  1. test_crystal_energy_dfdphi_matches_fh_plus_coupling
  2. test_crystal_energy_derivs_complex_step
  3. test_crystal_energy_cross_hessian_d2fdphidpsi  (extra: pins the cross coupling)
  4. test_crystal_energy_d2fdphidphi_equals_fh      (extra: chi is psi-independent in v1)
  5. test_crystal_discrete_block_and_jacobian_complex_step  (Task 2: full Jacobian)
  6. test_crystal_discrete_dR_dparam_complex_step   (Task 2: dR/dparam)
"""
import numpy as np
import pytest
from diffsim.adjoint.crystallization_multi import AdditiveCrystalEnergy
from diffsim.adjoint import FHMultiEnergy

import pytest
pytestmark = pytest.mark.ad


# ---- shared mesh helper (mirrors test_multiphase_adjoint._dm) ---------------
def _dm(level, dim=2):
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh
    tree = build_uniform(level, dim=dim)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), "cpu")
    return dm, mesh


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


# ==========================================================================
# Task 2: CrystalCHDiscrete — 2M+K block residual/Jacobian
# ==========================================================================
def _rand_crystal_energy(M=2, crystallizable=(0,), seed=42):
    """Build a random AdditiveCrystalEnergy for tests."""
    r = np.random.default_rng(seed)
    chi = 0.3 + 0.4 * r.random((M + 1, M + 1))
    chi = 0.5 * (chi + chi.T)
    np.fill_diagonal(chi, 0.0)
    N = 1.0 + 0.5 * r.random(M + 1)
    dsig = {k: 0.5 + r.random() for k in crystallizable}
    dh = {k: -(0.5 + r.random()) for k in crystallizable}
    Tm = {k: 1.0 + 0.2 * r.random() for k in crystallizable}
    return AdditiveCrystalEnergy(chi, N, crystallizable=crystallizable,
                                 dsig=dsig, dh=dh, Tm=Tm, T=0.5)


def _rand_crystal_state(op, seed):
    """Random (phis, mus, psis) for CrystalCHDiscrete with M,K from op."""
    r = np.random.default_rng(seed)
    M, K = op.M, op.crystallizable
    nn = op.nn
    phis = [0.10 + 0.08 * r.random(nn) for _ in range(M)]
    mus = [0.05 * r.random(nn) for _ in range(M)]
    psis = [0.10 + 0.08 * r.random(nn) for _ in range(len(K))]
    return phis, mus, psis


def test_crystal_discrete_block_and_jacobian_complex_step():
    """CrystalCHDiscrete M=2 ternary with K=(0,):
    1. Block sizing: ndof == (2*M + len(K)) * n_nodes.
    2. Full Jacobian dR/dx vs column-by-column complex-step (catches all
       cross-blocks: Rmu/dpsi, Rpsi/dphi, Rpsi/dpsi).  Agreement ~1e-8.
    """
    from diffsim.adjoint.crystallization_multi import CrystalCHDiscrete
    from diffsim.adjoint.neural_multiphase import MobilityClosure

    M = 2
    crystallizable = (0,)
    dm, _mesh = _dm(2)
    op = CrystalCHDiscrete(dm, M, crystallizable)

    # 1. Block sizing check
    blk = 2 * M + len(crystallizable)
    assert op.blk == blk
    assert op.ndof == blk * op.nn

    # 2. Build energy and params
    energy = _rand_crystal_energy(M=M, crystallizable=crystallizable, seed=10)
    onsager = np.array([[1.0, 0.2], [0.2, 0.8]])
    mobility = MobilityClosure("const", M=M, onsager=onsager)
    kappa = [0.01, 0.02]
    eps2 = {0: 0.015}
    L = {0: 1.5}
    sigma = 13.7

    params = dict(mobility=mobility, kappa=kappa, eps2=eps2, L=L,
                  energy=energy, sigma=sigma)

    phis, mus, psis = _rand_crystal_state(op, seed=11)

    # BDF history load: zeros (testing the Newton operator, not BDF step)
    hist_phi_gp = [[np.zeros_like(B["dJxW"]) for B in op.bins]
                   for _ in range(M)]
    hist_psi_gp = [[np.zeros_like(B["dJxW"]) for B in op.bins]
                   for _ in range(len(crystallizable))]

    R, J = op.assemble(phis, mus, psis, hist_phi_gp, hist_psi_gp,
                       params, want_jac=True)

    assert R.shape == (op.ndof,)
    assert J.shape == (op.ndof, op.ndof)

    # 3. Full Jacobian complex-step column check
    def col_cs(k):
        h = 1e-30
        node, fld = divmod(k, blk)
        pf = [x.astype(complex) for x in phis]
        mf = [x.astype(complex) for x in mus]
        sf = [x.astype(complex) for x in psis]
        if fld < 2 * M:
            # phi or mu fields
            field_idx = fld // 2      # species index
            if fld % 2 == 0:          # phi field
                pf[field_idx][node] += 1j * h
            else:                     # mu field
                mf[field_idx][node] += 1j * h
        else:
            # psi field: fld - 2*M gives psi index
            psi_idx = fld - 2 * M
            sf[psi_idx][node] += 1j * h
        Rc, _ = op.assemble(pf, mf, sf, hist_phi_gp, hist_psi_gp,
                             params, want_jac=False)
        return Rc.imag / h

    rng = np.random.default_rng(99)
    sample = rng.integers(0, op.ndof, size=24)
    for k in sample:
        k = int(k)
        got = np.asarray(J[:, k].todense()).ravel()
        cs = col_cs(k)
        assert np.allclose(got, cs, atol=1e-8, rtol=1e-6), \
            f"Jacobian mismatch at dof {k} (node={k // blk}, fld={k % blk})"


def test_crystal_discrete_dR_dparam_complex_step():
    """dR_dparam for all parameter kinds (eps2, L, kappa, mob, bulk) vs
    complex-step of the residual.  Covers the new psi-row scatter paths."""
    from diffsim.adjoint.crystallization_multi import CrystalCHDiscrete
    from diffsim.adjoint.neural_multiphase import MobilityClosure

    M = 2
    crystallizable = (0,)
    dm, _mesh = _dm(2)
    op = CrystalCHDiscrete(dm, M, crystallizable)

    energy = _rand_crystal_energy(M=M, crystallizable=crystallizable, seed=20)
    onsager = np.array([[1.0, 0.15], [0.15, 0.9]])
    mobility = MobilityClosure("const", M=M, onsager=onsager)
    kappa = [0.01, 0.02]
    eps2 = {0: 0.015}
    L = {0: 1.5}
    sigma = 9.0

    params = dict(mobility=mobility, kappa=kappa, eps2=eps2, L=L,
                  energy=energy, sigma=sigma)
    phis, mus, psis = _rand_crystal_state(op, seed=21)

    hist_phi_gp = [[np.zeros_like(B["dJxW"]) for B in op.bins]
                   for _ in range(M)]
    hist_psi_gp = [[np.zeros_like(B["dJxW"]) for B in op.bins]
                   for _ in range(len(crystallizable))]

    def resid(pp, pphis=None, mmus=None, ppsis=None):
        ph = pphis if pphis is not None else phis
        mu = mmus if mmus is not None else mus
        ps = ppsis if ppsis is not None else psis
        R, _ = op.assemble(ph, mu, ps, hist_phi_gp, hist_psi_gp, pp,
                            want_jac=False)
        return R

    h = 1e-30

    # --- engine params: eps2_0, L_0, kappa_0, kappa_1 ---
    for name in ["eps2_0", "L_0", "kappa_0", "kappa_1"]:
        an = op.dR_dparam(phis, mus, psis, params, name)
        if name.startswith("eps2_"):
            k = int(name.split("_")[1])
            e2c = {kk: complex(v) for kk, v in eps2.items()}
            e2c[k] += 1j * h
            pp = dict(params, eps2=e2c)
        elif name.startswith("L_"):
            k = int(name.split("_")[1])
            Lc = {kk: complex(v) for kk, v in L.items()}
            Lc[k] += 1j * h
            pp = dict(params, L=Lc)
        elif name.startswith("kappa_"):
            i = int(name.split("_")[1])
            kc = [complex(v) for v in kappa]
            kc[i] += 1j * h
            pp = dict(params, kappa=kc)
        cs = resid(pp).imag / h
        assert np.allclose(an, cs, atol=1e-8, rtol=1e-6), \
            f"dR_dparam mismatch for {name!r}"

    # --- bulk params via energy: chi_0_1, dsig_0, dh_0, Tm_0 ---
    chi = energy.fh.chi
    N_arr = energy.fh.N
    for name in ["chi_0_1", "N_0", "dsig_0", "dh_0", "Tm_0"]:
        an = op.dR_dparam(phis, mus, psis, params, name)
        if name.startswith("chi_"):
            _, a, b = name.split("_"); a, b = int(a), int(b)
            cc = chi.astype(complex); cc[a, b] += 1j * h; cc[b, a] += 1j * h
            ec = AdditiveCrystalEnergy(cc, N_arr, crystallizable,
                                       energy.dsig, energy.dh, energy.Tm, energy.T)
            pp = dict(params, energy=ec)
        elif name.startswith("N_"):
            j = int(name.split("_")[1])
            Nc = N_arr.astype(complex); Nc[j] += 1j * h
            ec = AdditiveCrystalEnergy(chi, Nc, crystallizable,
                                       energy.dsig, energy.dh, energy.Tm, energy.T)
            pp = dict(params, energy=ec)
        elif name.startswith("dsig_"):
            k = int(name.split("_")[1])
            dsigc = {kk: complex(v) for kk, v in energy.dsig.items()}
            dsigc[k] += 1j * h
            ec = AdditiveCrystalEnergy(chi, N_arr, crystallizable,
                                       dsigc, energy.dh, energy.Tm, energy.T)
            pp = dict(params, energy=ec)
        elif name.startswith("dh_"):
            k = int(name.split("_")[1])
            dhc = {kk: complex(v) for kk, v in energy.dh.items()}
            dhc[k] += 1j * h
            ec = AdditiveCrystalEnergy(chi, N_arr, crystallizable,
                                       energy.dsig, dhc, energy.Tm, energy.T)
            pp = dict(params, energy=ec)
        elif name.startswith("Tm_"):
            k = int(name.split("_")[1])
            Tmc = {kk: complex(v) for kk, v in energy.Tm.items()}
            Tmc[k] += 1j * h
            ec = AdditiveCrystalEnergy(chi, N_arr, crystallizable,
                                       energy.dsig, energy.dh, Tmc, energy.T)
            pp = dict(params, energy=ec)
        cs = resid(pp).imag / h
        assert np.allclose(an, cs, atol=1e-8, rtol=1e-6), \
            f"dR_dparam mismatch for {name!r}"
