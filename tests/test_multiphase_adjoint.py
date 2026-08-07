"""M-component phase-separation adjoint gates (K=0) — the M-generic sibling of
test_phasefield_adjoint.py.  Three-way verification (hand IFT adjoint == torch
autograd twin == central finite difference) is the milestone gate.

Rung 1 scope: FHMultiEnergy + MultiCHDiscrete/Forward/Adjoint + MultiCHTwin,
verified against physics/multiphase.MultiPhaseStepper (K=0, bulk='p1', const)."""
import numpy as np
import pytest

from diffsim.physics.multiphase import np_potentials

pytestmark = pytest.mark.ad


# ---- mesh helper ---------------------------------------------------------
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


def _rng_phis(M, n, seed):
    r = np.random.default_rng(seed)
    x = 0.1 + 0.15 * r.random((M, n))          # interior, sum < 0.9
    return [x[i] for i in range(M)]


# ==========================================================================
# Task 1: FHMultiEnergy
# ==========================================================================
@pytest.mark.parametrize("M", [2, 3])
def test_energy_mu_parity(M):
    """mu_i matches the production numpy reference np_potentials at K=0."""
    from diffsim.adjoint.multiphase import FHMultiEnergy
    r = np.random.default_rng(0)
    chi = 0.3 + 0.5 * r.random((M + 1, M + 1))
    chi = 0.5 * (chi + chi.T)
    np.fill_diagonal(chi, 0.0)
    N = np.ones(M + 1)
    phis = _rng_phis(M, 50, seed=1)
    en = FHMultiEnergy(chi, N)
    got = en.mu(phis)
    z = np.zeros_like(chi)
    pars = dict(chi_aa=chi, chi_ac=z, chi_ca=z, chi_cc=z, Ninv=1.0 / N,
                dsig=[], drive=[], breg=0.0, bulk="p1")
    ref, _ = np_potentials(phis, [], pars)
    for i in range(M):
        assert np.allclose(got[i], ref[i], atol=1e-12, rtol=0), i


def _cstep_dmu_dphi(en, phis, j, i):
    h = 1e-30
    p = [x.astype(complex) for x in phis]
    p[j] = p[j] + 1j * h
    return en.mu(p)[i].imag / h


@pytest.mark.parametrize("M", [2, 3])
def test_energy_derivs_complex_step(M):
    """dmu_dphi via complex step; dmu_dparam via central FD of mu."""
    from diffsim.adjoint.multiphase import FHMultiEnergy
    r = np.random.default_rng(3)
    chi = 0.3 + 0.5 * r.random((M + 1, M + 1))
    chi = 0.5 * (chi + chi.T)
    np.fill_diagonal(chi, 0.0)
    N = 1.0 + r.random(M + 1)
    phis = _rng_phis(M, 20, seed=4)
    en = FHMultiEnergy(chi, N)
    H = en.dmu_dphi(phis)
    for i in range(M):
        for j in range(M):
            cs = _cstep_dmu_dphi(en, phis, j, i)
            assert np.allclose(H[i][j], cs, atol=1e-9, rtol=1e-7), (i, j)
    for name in en.param_names:
        an = en.dmu_dparam(phis, name)
        eps = 1e-6
        for i in range(M):
            if name.startswith("chi_"):
                _, a, b = name.split("_")
                a, b = int(a), int(b)
                c1 = chi.copy(); c1[a, b] += eps; c1[b, a] += eps
                c2 = chi.copy(); c2[a, b] -= eps; c2[b, a] -= eps
                fd = (FHMultiEnergy(c1, N).mu(phis)[i]
                      - FHMultiEnergy(c2, N).mu(phis)[i]) / (2 * eps)
            else:
                jj = int(name.split("_")[1])
                n1 = N.copy(); n1[jj] += eps
                n2 = N.copy(); n2[jj] -= eps
                fd = (FHMultiEnergy(chi, n1).mu(phis)[i]
                      - FHMultiEnergy(chi, n2).mu(phis)[i]) / (2 * eps)
            assert np.allclose(an[i], fd, atol=1e-6, rtol=1e-5), (name, i)


# ==========================================================================
# Task 2: MultiCHDiscrete
# ==========================================================================
def _rand_state(op, seed):
    r = np.random.default_rng(seed)
    M, nn = op.M, op.nn
    phis = [0.20 + 0.06 * r.random(nn) for _ in range(M)]
    mus = [0.05 * r.random(nn) for _ in range(M)]
    return phis, mus


def _rand_energy(M, seed):
    from diffsim.adjoint.multiphase import FHMultiEnergy
    r = np.random.default_rng(seed)
    chi = 0.3 + 0.4 * r.random((M + 1, M + 1))
    chi = 0.5 * (chi + chi.T)
    np.fill_diagonal(chi, 0.0)
    return FHMultiEnergy(chi, 1.0 + 0.5 * r.random(M + 1)), chi


def test_discrete_jacobian_complex_step():
    """J = dR/dx column-by-column vs complex-step of the residual (M=2)."""
    from diffsim.adjoint.multiphase import MultiCHDiscrete
    M = 2
    dm, mesh = _dm(2)
    op = MultiCHDiscrete(dm, M)
    en, _ = _rand_energy(M, 7)
    phis, mus = _rand_state(op, 8)
    onsager = np.array([[1.0, 0.2], [0.2, 0.8]])
    params = dict(onsager=onsager, kappa=[0.01, 0.02], energy=en, sigma=13.7)
    hist = [[np.zeros_like(B["dJxW"]) for B in op.bins] for _ in range(M)]
    R, J = op.assemble(phis, mus, hist, params, want_jac=True)
    blk = 2 * M

    def col(k):
        h = 1e-30
        pf = [x.astype(complex) for x in phis]
        mf = [x.astype(complex) for x in mus]
        node, fld = divmod(k, blk)
        (pf if fld % 2 == 0 else mf)[fld // 2][node] += 1j * h
        Rc, _ = op.assemble(pf, mf, hist, params, want_jac=False)
        return Rc.imag / h

    r = np.random.default_rng(9)
    for k in r.integers(0, op.ndof, size=16):
        got = np.asarray(J[:, int(k)].todense()).ravel()
        assert np.allclose(got, col(int(k)), atol=1e-8, rtol=1e-6), int(k)


def test_dR_dparam_complex_step():
    """dR/dp vs complex-step of the residual, all rung-1 parameter kinds."""
    from diffsim.adjoint.multiphase import MultiCHDiscrete
    M = 2
    dm, mesh = _dm(2)
    op = MultiCHDiscrete(dm, M)
    en, chi = _rand_energy(M, 11)
    phis, mus = _rand_state(op, 12)
    onsager = np.array([[1.0, 0.2], [0.2, 0.8]])
    kappa = [0.01, 0.02]
    params = dict(onsager=onsager, kappa=kappa, energy=en, sigma=9.0)
    hist = [[np.zeros_like(B["dJxW"]) for B in op.bins] for _ in range(M)]

    def resid(pp):
        R, _ = op.assemble(phis, mus, hist, pp, want_jac=False)
        return R

    names = ["chi_0_1", "chi_0_2", "N_0", "N_2", "onsager_0_1", "kappa_0"]
    from diffsim.adjoint.multiphase import FHMultiEnergy
    for name in names:
        an = op.dR_dparam(phis, mus, params, name)
        h = 1e-30
        if name.startswith("chi_"):
            _, a, b = name.split("_"); a, b = int(a), int(b)
            cc = chi.astype(complex); cc[a, b] += 1j * h; cc[b, a] += 1j * h
            pp = dict(params, energy=FHMultiEnergy(cc, en.N))
        elif name.startswith("N_"):
            j = int(name.split("_")[1])
            NN = en.N.astype(complex); NN[j] += 1j * h
            pp = dict(params, energy=FHMultiEnergy(chi, NN))
        elif name.startswith("onsager_"):
            _, a, b = name.split("_"); a, b = int(a), int(b)
            oo = onsager.astype(complex); oo[a, b] += 1j * h
            pp = dict(params, onsager=oo)
        else:
            i = int(name.split("_")[1])
            kk = [complex(v) for v in kappa]; kk[i] += 1j * h
            pp = dict(params, kappa=kk)
        cs = resid(pp).imag / h
        assert np.allclose(an, cs, atol=1e-8, rtol=1e-6), name
