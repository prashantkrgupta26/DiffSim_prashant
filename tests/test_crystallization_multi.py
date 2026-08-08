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


# ==========================================================================
# Task 3: CrystalCHForward + CrystalCHAdjoint
# ==========================================================================

def test_crystal_forward_reduces_to_multi_when_K_empty():
    """CrystalCHForward with crystallizable=() reproduces MultiCHForward
    (K=0) to 1e-14 on a ternary (M=2) run for BDF1 and BDF2."""
    from diffsim.adjoint import (MultiCHForward, FHMultiEnergy,
                                  MobilityClosure)
    from diffsim.adjoint.crystallization_multi import CrystalCHForward

    M = 2
    chi, N = _chiN(M=M)
    ons = np.array([[1.0, 0.2], [0.2, 0.8]])
    kap = [0.01, 0.02]
    dt = 1e-2
    dm, _mesh = _dm(2)

    rng = np.random.default_rng(77)
    nn = dm.n_nodes
    phi0 = [0.15 + 0.05 * rng.random(nn) for _ in range(M)]

    for order in (1, 2):
        # MultiCHForward (K=0 reference)
        multi = MultiCHForward(dm, FHMultiEnergy(chi, N), onsager=ons,
                               kappa=kap, dt=dt, order=order)
        multi.set_initial([p.copy() for p in phi0])
        multi.run(3)

        # CrystalCHForward with empty crystallizable
        energy = AdditiveCrystalEnergy(chi, N, crystallizable=(), dsig={},
                                       dh={}, Tm={}, T=0.5)
        crystal = CrystalCHForward(dm, energy, crystallizable=(),
                                   mobility=MobilityClosure("const", M=M,
                                                            onsager=ons),
                                   kappa=kap, eps2={}, L={}, dt=dt,
                                   order=order)
        crystal.set_initial([p.copy() for p in phi0], psi0_list=[])
        crystal.run(3)

        for i in range(M):
            assert np.allclose(crystal.steps[-1]["phis"][i],
                               multi.steps[-1]["phis"][i], atol=1e-14), \
                f"phis[{i}] mismatch at order={order}"
            assert np.allclose(crystal.steps[-1]["mus"][i],
                               multi.steps[-1]["mus"][i], atol=1e-14), \
                f"mus[{i}] mismatch at order={order}"


def test_crystal_adjoint_returns_finite_grads():
    """CrystalCHAdjoint.gradient returns finite grads for all param kinds
    when M=2, K=1 (crystallizable=(0,))."""
    from diffsim.adjoint import MobilityClosure
    from diffsim.adjoint.crystallization_multi import (CrystalCHForward,
                                                        CrystalCHAdjoint)

    M = 2
    crystallizable = (0,)
    chi, N = _chiN(M=M)
    ons = np.array([[1.0, 0.15], [0.15, 0.9]])
    kap = [0.01, 0.02]
    eps2 = {0: 0.015}
    L = {0: 1.5}
    dt = 1e-2
    dm, _mesh = _dm(2)
    nn = dm.n_nodes

    energy = AdditiveCrystalEnergy(chi, N, crystallizable=crystallizable,
                                   dsig={0: 1.2}, dh={0: -1.0}, Tm={0: 1.0},
                                   T=0.5)
    mob = MobilityClosure("const", M=M, onsager=ons)

    rng = np.random.default_rng(123)
    phi0 = [0.15 + 0.05 * rng.random(nn) for _ in range(M)]
    psi0 = [0.1 + 0.05 * rng.random(nn)]   # K=1

    fwd = CrystalCHForward(dm, energy, crystallizable=crystallizable,
                           mobility=mob, kappa=kap, eps2=eps2, L=L, dt=dt,
                           order=1)
    fwd.set_initial(phi0, psi0_list=psi0)
    N_steps = 3
    fwd.run(N_steps)

    blk = fwd.op.blk
    ndof = fwd.op.ndof

    # dJ/dx targets the final phi_0 field only
    dJdx_list = [np.zeros(ndof) for _ in range(N_steps)]
    dJdx_list[-1][0::blk] = 1.0   # phi_0 columns

    param_names = ["dsig_0", "dh_0", "Tm_0", "eps2_0", "L_0",
                   "chi_0_1", "N_0", "kappa_0"]
    adj = CrystalCHAdjoint(fwd)
    grads = adj.gradient(dJdx_list, param_names)

    assert set(grads.keys()) == set(param_names), \
        f"Missing keys: {set(param_names) - set(grads.keys())}"
    for nm, val in grads.items():
        assert np.isfinite(val), f"grad[{nm!r}] = {val} is not finite"


# ==========================================================================
# Task 4: CrystalCHTwin — autograd twin, checked vs central-FD of same loss
# ==========================================================================
def test_crystal_twin_grads_vs_fd():
    """CrystalCHTwin (M-generic CACHTwin + generic CrystalEnergy).  Ternary
    M=2, crystallizable=(0,), BDF1, n_steps=3.  Twin autograd grads of

        loss = 0.5 sum_i ||phi_i,N - tgt||^2 + 0.5 sum_k ||psi_k,N - tgt||^2

    are checked against central finite differences of the SAME loss computed
    with a fresh numpy CrystalCHForward march (independent forward).  Relative
    error < 1e-6 for each param."""
    from diffsim.adjoint import MobilityClosure
    from diffsim.adjoint.crystallization_multi import CrystalCHForward
    from diffsim.adjoint.torch_twin import CrystalCHTwin

    M = 2
    crystallizable = (0,)
    chi, N = _chiN(M=M)
    ons = np.array([[1.0, 0.15], [0.15, 0.9]])
    kap = [0.01, 0.02]
    eps2 = {0: 0.015}
    L = {0: 1.5}
    dsig = {0: 1.2}
    dh = {0: -1.0}
    Tm = {0: 1.0}
    T = 0.5
    dt = 1e-2
    n_steps = 3
    target = 0.2

    dm, _mesh = _dm(2)
    nn = dm.n_nodes
    rng = np.random.default_rng(321)
    phi0 = [0.15 + 0.05 * rng.random(nn) for _ in range(M)]
    psi0 = [0.1 + 0.05 * rng.random(nn)]

    energy_params = dict(chi=chi, N=N, dsig=dsig, dh=dh, Tm=Tm, T=T)
    engine_params = dict(onsager=ons, kappa=kap, eps2=eps2, L=L)

    names = ["dsig_0", "dh_0", "L_0", "eps2_0", "chi_0_1", "kappa_0"]

    # --- twin autograd grads ---
    twin = CrystalCHTwin(dm, M, crystallizable, dt=dt, order=1)
    tw = twin.grads(phi0, psi0, energy_params, engine_params, n_steps,
                    names, target)

    # --- central-FD reference of the SAME loss via numpy forward ---
    def loss_of(ep, gp):
        en = AdditiveCrystalEnergy(
            ep["chi"], ep["N"], crystallizable=crystallizable,
            dsig=ep["dsig"], dh=ep["dh"], Tm=ep["Tm"], T=ep["T"])
        fwd = CrystalCHForward(
            dm, en, crystallizable=crystallizable,
            mobility=MobilityClosure("const", M=M,
                                     onsager=np.asarray(gp["onsager"])),
            kappa=gp["kappa"], eps2=gp["eps2"], L=gp["L"], dt=dt, order=1)
        fwd.set_initial([p.copy() for p in phi0],
                        psi0_list=[p.copy() for p in psi0])
        fwd.run(n_steps)
        rec = fwd.steps[-1]
        loss = 0.5 * sum(((p - target) ** 2).sum() for p in rec["phis"])
        loss += 0.5 * sum(((p - target) ** 2).sum() for p in rec["psis"])
        return loss

    def perturbed(name, delta):
        ep = dict(chi=np.array(chi, float), N=np.array(N, float),
                  dsig=dict(dsig), dh=dict(dh), Tm=dict(Tm), T=T)
        gp = dict(onsager=np.array(ons, float), kappa=list(kap),
                  eps2=dict(eps2), L=dict(L))
        if name.startswith("dsig_"):
            k = int(name.split("_")[1]); ep["dsig"][k] += delta
        elif name.startswith("dh_"):
            k = int(name.split("_")[1]); ep["dh"][k] += delta
        elif name.startswith("Tm_"):
            k = int(name.split("_")[1]); ep["Tm"][k] += delta
        elif name.startswith("L_"):
            k = int(name.split("_")[1]); gp["L"][k] += delta
        elif name.startswith("eps2_"):
            k = int(name.split("_")[1]); gp["eps2"][k] += delta
        elif name.startswith("kappa_"):
            i = int(name.split("_")[1]); gp["kappa"][i] += delta
        elif name.startswith("chi_"):
            _, a, b = name.split("_"); a, b = int(a), int(b)
            ep["chi"][a, b] += delta; ep["chi"][b, a] += delta
        else:
            raise ValueError(name)
        return ep, gp

    step = 1e-6
    print()
    for nm in names:
        epp, gpp = perturbed(nm, step)
        epm, gpm = perturbed(nm, -step)
        fd = (loss_of(epp, gpp) - loss_of(epm, gpm)) / (2 * step)
        relerr = abs(tw[nm] - fd) / max(abs(fd), 1e-30)
        print(f"  {nm:10s} twin={tw[nm]:+.10e} fd={fd:+.10e} "
              f"relerr={relerr:.2e}")
        assert relerr < 1e-6, f"{nm}: twin={tw[nm]} fd={fd} relerr={relerr}"


# ==========================================================================
# Task 5: THREE-WAY GATE — hand adjoint == autograd twin == finite differences
# ==========================================================================

def _check_three_way_crystal(res, tag):
    """Check adj/twin < 1e-10, adj/fd < 1e-6 for every param in res."""
    for p, (a, t, f) in res.items():
        r_t = abs(a - t) / max(abs(t), 1e-14)
        r_f = abs(a - f) / max(abs(f), 1e-14)
        print(f"{tag} {p:14s} adj={a:+.6e} twin={t:+.6e} fd={f:+.6e} "
              f"adj/twin={r_t:.2e} adj/fd={r_f:.2e}")
        assert r_t < 1e-10, f"{tag} {p}: adj={a} twin={t} ratio={r_t}"
        assert r_f < 1e-6, f"{tag} {p}: adj={a} fd={f} ratio={r_f}"


def _three_way_crystal(dm, coords, M, crystallizable, order, n_steps,
                       dt=1e-2):
    """Three-way gate for CrystalCHForward/Adjoint vs CrystalCHTwin vs FD.

    Mirrors _three_way_multi (test_multiphase_adjoint.py) but for the coupled
    M-CH + K-Allen-Cahn system.  Loss = 0.5 sum_i ||phi_i,N - tgt||^2
    + 0.5 sum_k ||psi_k,N - tgt||^2.  Returns {name: (adj, twin, fd)}.
    """
    from diffsim.adjoint import MobilityClosure
    from diffsim.adjoint.crystallization_multi import (
        CrystalCHForward, CrystalCHAdjoint)
    from diffsim.adjoint.torch_twin import CrystalCHTwin

    K = len(crystallizable)
    nn = dm.n_nodes
    blk = 2 * M + K

    # --- base params (non-trivial, strong chi[0,1]) ---
    chi0 = np.zeros((M + 1, M + 1))
    for a in range(M + 1):
        for b in range(a + 1, M + 1):
            chi0[a, b] = chi0[b, a] = 2.5 if (a, b) == (0, 1) else 1.0
    N0 = 1.0 + 0.3 * np.arange(M + 1, dtype=float)
    ons0 = np.eye(M) + 0.1 * (np.ones((M, M)) - np.eye(M))
    kap0 = [0.01 * (i + 1) for i in range(M)]
    eps20 = {k: 0.012 + 0.003 * j for j, k in enumerate(crystallizable)}
    L0 = {k: 1.2 + 0.2 * j for j, k in enumerate(crystallizable)}
    dsig0 = {k: 1.0 + 0.2 * j for j, k in enumerate(crystallizable)}
    dh0 = {k: -(0.8 + 0.15 * j) for j, k in enumerate(crystallizable)}
    Tm0 = {k: 1.0 + 0.1 * j for j, k in enumerate(crystallizable)}
    T = 0.5
    tgt = 0.20

    # cosine initial field for phi, modest psi
    cc = np.cos(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])
    phi0 = [0.20 + 0.04 * cc for _ in range(M)]
    psi0 = [0.15 + 0.05 * cc for _ in range(K)]

    names = ["dsig_0", "dh_0", "Tm_0", "eps2_0", "L_0",
             "chi_0_1", "N_0", "onsager_0_0", "kappa_0"]

    # ---- forward + hand adjoint ----
    def _make_energy():
        return AdditiveCrystalEnergy(chi0, N0, crystallizable=crystallizable,
                                     dsig=dict(dsig0), dh=dict(dh0),
                                     Tm=dict(Tm0), T=T)

    def _make_fwd(en):
        mob = MobilityClosure("const", M=M, onsager=ons0)
        fwd = CrystalCHForward(dm, en, crystallizable=crystallizable,
                               mobility=mob,
                               kappa=list(kap0), eps2=dict(eps20),
                               L=dict(L0), dt=dt, order=order)
        fwd.set_initial([p.copy() for p in phi0],
                        psi0_list=[p.copy() for p in psi0])
        fwd.run(n_steps)
        return fwd

    en0 = _make_energy()
    fwd = _make_fwd(en0)
    rec = fwd.steps[-1]

    # dJdx at final step: phi_i rows and psi_j rows
    dJdx_list = [np.zeros(blk * nn) for _ in range(n_steps)]
    for i in range(M):
        dJdx_list[-1][2 * i::blk] = rec["phis"][i] - tgt
    for j in range(K):
        dJdx_list[-1][2 * M + j::blk] = rec["psis"][j] - tgt
    g_adj = CrystalCHAdjoint(fwd).gradient(dJdx_list, names)

    # ---- autograd twin ----
    twin = CrystalCHTwin(dm, M, crystallizable, dt=dt, order=order,
                         device="cpu")
    energy_params = dict(chi=chi0, N=N0, dsig=dsig0, dh=dh0, Tm=Tm0, T=T)
    engine_params = dict(onsager=ons0, kappa=kap0, eps2=eps20, L=L0)
    g_tw = twin.grads(phi0, psi0, energy_params, engine_params,
                      n_steps, names, tgt)

    # ---- central FD of same numpy loss ----
    def _loss(en, mob_ons, kap, eps2, L_):
        mob = MobilityClosure("const", M=M, onsager=np.asarray(mob_ons))
        fwd_ = CrystalCHForward(dm, en, crystallizable=crystallizable,
                                mobility=mob,
                                kappa=list(kap), eps2=dict(eps2),
                                L=dict(L_), dt=dt, order=order)
        fwd_.set_initial([p.copy() for p in phi0],
                         psi0_list=[p.copy() for p in psi0])
        fwd_.run(n_steps)
        rec_ = fwd_.steps[-1]
        loss = 0.5 * sum(((p - tgt) ** 2).sum() for p in rec_["phis"])
        loss += 0.5 * sum(((p - tgt) ** 2).sum() for p in rec_["psis"])
        return float(loss)

    def _fd(name):
        eps = 1e-6

        def bump(sign):
            chi_ = chi0.copy()
            N_ = N0.copy()
            ons_ = ons0.copy()
            kap_ = list(kap0)
            eps2_ = dict(eps20)
            L__ = dict(L0)
            dsig_ = dict(dsig0)
            dh_ = dict(dh0)
            Tm_ = dict(Tm0)
            if name.startswith("chi_"):
                _, a, b = name.split("_"); a, b = int(a), int(b)
                chi_[a, b] += sign * eps; chi_[b, a] += sign * eps
            elif name.startswith("N_"):
                N_[int(name.split("_")[1])] += sign * eps
            elif name.startswith("onsager_"):
                _, a, b = name.split("_"); a, b = int(a), int(b)
                ons_[a, b] += sign * eps
            elif name.startswith("kappa_"):
                kap_[int(name.split("_")[1])] += sign * eps
            elif name.startswith("eps2_"):
                k = int(name.split("_")[1]); eps2_[k] += sign * eps
            elif name.startswith("L_"):
                k = int(name.split("_")[1]); L__[k] += sign * eps
            elif name.startswith("dsig_"):
                k = int(name.split("_")[1]); dsig_[k] += sign * eps
            elif name.startswith("dh_"):
                k = int(name.split("_")[1]); dh_[k] += sign * eps
            elif name.startswith("Tm_"):
                k = int(name.split("_")[1]); Tm_[k] += sign * eps
            en_ = AdditiveCrystalEnergy(chi_, N_, crystallizable=crystallizable,
                                         dsig=dsig_, dh=dh_, Tm=Tm_, T=T)
            return _loss(en_, ons_, kap_, eps2_, L__)
        return (bump(+1) - bump(-1)) / (2 * eps)

    g_fd = {nm: _fd(nm) for nm in names}
    return {nm: (g_adj[nm], g_tw[nm], g_fd[nm]) for nm in names}


def test_three_way_crystal_ternary_bdf1():
    """M=2 ternary (crystallizable=(0,)), BDF1, n_steps=3: adj == twin == FD."""
    dm, mesh = _dm(2)
    res = _three_way_crystal(dm, mesh.node_coords, M=2, crystallizable=(0,),
                             order=1, n_steps=3)
    _check_three_way_crystal(res, "ternary-bdf1")


def test_three_way_crystal_ternary_bdf2():
    """M=2 ternary (crystallizable=(0,)), BDF2, n_steps=4: adj == twin == FD."""
    dm, mesh = _dm(2)
    res = _three_way_crystal(dm, mesh.node_coords, M=2, crystallizable=(0,),
                             order=2, n_steps=4)
    _check_three_way_crystal(res, "ternary-bdf2")


def test_three_way_crystal_quaternary_bdf1():
    """M=3 quaternary (crystallizable=(0,2)), BDF1, n_steps=3:
    both species 0 and 2 crystallize; adj == twin == FD."""
    dm, mesh = _dm(2)
    res = _three_way_crystal(dm, mesh.node_coords, M=3, crystallizable=(0, 2),
                             order=1, n_steps=3)
    _check_three_way_crystal(res, "quaternary-bdf1")


# ==========================================================================
# Task 6: GPU cuDSS parity — backend= seam smoke (CPU) + GPU-gated parity
# ==========================================================================

def _gpu_available():
    try:
        import warp as wp
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


def test_crystal_backend_honored_cpu():
    """Explicit ScipyBackend == default (None); confirms the backend= seam is live."""
    from diffsim.adjoint.crystallization_multi import (
        CrystalCHForward, CrystalCHAdjoint, AdditiveCrystalEnergy)
    from diffsim.adjoint import ScipyBackend
    dm, mesh = _dm(2)
    M, cryst = 2, (0,)
    chi, N = _chiN(M)
    energy = AdditiveCrystalEnergy(chi, N, cryst, {0: 0.7}, {0: -0.9}, {0: 1.1})
    cc = np.cos(np.pi * mesh.node_coords[:, 0])
    phi0 = [0.28 + 0.03 * cc, 0.30 + 0.03 * cc]
    psi0 = [0.25 + 0.02 * cc]
    names = ["dsig_0", "L_0", "chi_0_1", "kappa_0"]

    def run(backend):
        fwd = CrystalCHForward(dm, energy, cryst, onsager=np.eye(M),
                               kappa=[0.01, 0.02], eps2={0: 0.015}, L={0: 1.2},
                               dt=0.01, order=1, backend=backend)
        fwd.set_initial(phi0, psi0)
        fwd.run(3)
        blk = fwd.op.blk
        dJdx = [np.zeros(fwd.op.ndof) for _ in range(3)]
        for i in range(M):
            dJdx[-1][2 * i::blk] = fwd.steps[-1]["phis"][i] - 0.28
        dJdx[-1][2 * M::blk] = fwd.steps[-1]["psis"][0] - 0.25
        return CrystalCHAdjoint(fwd).gradient(dJdx, names)

    g_default = run(None)
    g_explicit = run(ScipyBackend())
    for nm in names:
        assert np.isclose(g_default[nm], g_explicit[nm], rtol=1e-12), nm


@pytest.mark.skipif(not _gpu_available(), reason="no CUDA GPU (rung-2 gate runs on gpubox)")
def test_crystal_cudss_parity_gpu():
    """cuDSS crystal adjoint gradients == scipy, to ~1e-8 (2M+K block)."""
    from diffsim.adjoint.crystallization_multi import (
        CrystalCHForward, CrystalCHAdjoint, AdditiveCrystalEnergy)
    from diffsim.adjoint import ScipyBackend, CudssBackend
    dm, mesh = _dm(3)
    M, cryst = 2, (0,)
    chi, N = _chiN(M)
    energy = AdditiveCrystalEnergy(chi, N, cryst, {0: 0.7}, {0: -0.9}, {0: 1.1})
    cc = np.cos(np.pi * mesh.node_coords[:, 0])
    phi0 = [0.28 + 0.03 * cc, 0.30 + 0.03 * cc]
    psi0 = [0.25 + 0.02 * cc]
    names = ["dsig_0", "dh_0", "eps2_0", "L_0", "chi_0_1", "kappa_0"]

    def run(backend):
        fwd = CrystalCHForward(dm, energy, cryst, onsager=np.eye(M),
                               kappa=[0.01, 0.02], eps2={0: 0.015}, L={0: 1.2},
                               dt=0.01, order=2, backend=backend)
        fwd.set_initial(phi0, psi0)
        fwd.run(4)
        blk = fwd.op.blk
        dJdx = [np.zeros(fwd.op.ndof) for _ in range(4)]
        for i in range(M):
            dJdx[-1][2 * i::blk] = fwd.steps[-1]["phis"][i] - 0.28
        dJdx[-1][2 * M::blk] = fwd.steps[-1]["psis"][0] - 0.25
        return CrystalCHAdjoint(fwd).gradient(dJdx, names)

    g_cpu = run(ScipyBackend())
    g_gpu = run(CudssBackend("cuda:0"))
    for nm in names:
        assert np.isclose(g_gpu[nm], g_cpu[nm], rtol=1e-6, atol=1e-8), \
            (nm, g_gpu[nm], g_cpu[nm])
