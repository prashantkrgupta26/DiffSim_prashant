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


# ==========================================================================
# Task 3: MultiCHForward — forward parity vs production MultiPhaseStepper (K=0)
# ==========================================================================
def test_forward_parity_ternary():
    from diffsim.adjoint.multiphase import MultiCHForward, FHMultiEnergy
    from diffsim.physics.multiphase import MultiPhaseStepper
    M = 2
    dm, mesh = _dm(3)
    coords = mesh.node_coords
    chi = np.zeros((M + 1, M + 1))
    chi[0, 1] = chi[1, 0] = 2.5
    chi[0, 2] = chi[2, 0] = 1.0
    chi[1, 2] = chi[2, 1] = 0.8
    N = np.ones(M + 1)
    onsager = np.eye(M)
    kappa = [1e-2, 1e-2]
    dt = 1e-2
    cc = np.cos(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])
    phi0 = [0.30 + 0.05 * cc, 0.30 + 0.05 * cc]
    st = MultiPhaseStepper(dm, M=M, K=0, chi_aa=chi, N=N.tolist(),
                           onsager=onsager.tolist(), kappa=kappa, dt=dt,
                           bulk="p1", newton_tol=1e-12, newton_max=40,
                           linsolver="splu", tstep="bdf1")
    st.set_initial([(lambda x, v=p: v) for p in phi0])
    for _ in range(3):
        st.step()
    ref = [np.asarray(st.phi(i)) for i in range(M)]
    fwd = MultiCHForward(dm, FHMultiEnergy(chi, N), onsager=onsager,
                         kappa=kappa, dt=dt, order=1)
    fwd.set_initial(phi0)
    fwd.run(3)
    for i in range(M):
        rel = (np.abs(fwd.steps[-1]["phis"][i] - ref[i]).max()
               / max(np.abs(ref[i]).max(), 1e-12))
        print(f"parity ternary phi{i}: max rel {rel:.2e}")
        assert np.allclose(fwd.steps[-1]["phis"][i], ref[i],
                           atol=1e-9, rtol=1e-7), i


def test_forward_parity_quaternary():
    from diffsim.adjoint.multiphase import MultiCHForward, FHMultiEnergy
    from diffsim.physics.multiphase import MultiPhaseStepper
    M = 3
    dm, mesh = _dm(2)
    coords = mesh.node_coords
    chi = np.zeros((M + 1, M + 1))
    for a in range(M + 1):
        for b in range(a + 1, M + 1):
            chi[a, b] = chi[b, a] = 1.5 if b == M else 2.0
    N = np.ones(M + 1)
    onsager = np.eye(M)
    kappa = [1e-2] * M
    dt = 1e-2
    cc = np.cos(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])
    phi0 = [0.22 + 0.04 * cc for _ in range(M)]
    st = MultiPhaseStepper(dm, M=M, K=0, chi_aa=chi, N=N.tolist(),
                           onsager=onsager.tolist(), kappa=kappa, dt=dt,
                           bulk="p1", newton_tol=1e-12, newton_max=40,
                           linsolver="splu", tstep="bdf1")
    st.set_initial([(lambda x, v=p: v) for p in phi0])
    for _ in range(2):
        st.step()
    ref = [np.asarray(st.phi(i)) for i in range(M)]
    fwd = MultiCHForward(dm, FHMultiEnergy(chi, N), onsager=onsager,
                         kappa=kappa, dt=dt, order=1)
    fwd.set_initial(phi0)
    fwd.run(2)
    for i in range(M):
        assert np.allclose(fwd.steps[-1]["phis"][i], ref[i],
                           atol=1e-9, rtol=1e-7), i


# ==========================================================================
# Task 4: MultiCHAdjoint — interface smoke (three-way gate is Task 5)
# ==========================================================================
def test_stiffness_matrix():
    """K symmetric; K @ const == 0 (gradient energy of a constant is zero);
    phi^T K phi > 0 for a non-constant field (used by the interfacial-energy
    objective in the daisy-morph gradients path)."""
    from diffsim.adjoint.multiphase import MultiCHDiscrete
    dm, mesh = _dm(3)
    op = MultiCHDiscrete(dm, 2)
    K = op.stiffness_matrix()
    assert (abs(K - K.T)).nnz == 0 or np.allclose((K - K.T).data, 0.0)
    const = np.ones(op.nn)
    assert np.allclose(K @ const, 0.0, atol=1e-10)
    cc = np.cos(np.pi * mesh.node_coords[:, 0])
    assert float(cc @ (K @ cc)) > 1e-6


def test_adjoint_interface():
    from diffsim.adjoint.multiphase import (MultiCHForward, MultiCHAdjoint,
                                            FHMultiEnergy)
    M = 2
    dm, mesh = _dm(3)
    coords = mesh.node_coords
    chi = np.zeros((M + 1, M + 1))
    chi[0, 1] = chi[1, 0] = 2.5
    chi[0, 2] = chi[2, 0] = 1.0
    chi[1, 2] = chi[2, 1] = 0.8
    N = np.ones(M + 1)
    fwd = MultiCHForward(dm, FHMultiEnergy(chi, N), onsager=np.eye(M),
                         kappa=[1e-2, 1e-2], dt=1e-2, order=1)
    cc = np.cos(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])
    fwd.set_initial([0.30 + 0.05 * cc, 0.30 + 0.05 * cc])
    fwd.run(3)
    blk = 2 * M
    nn = dm.n_nodes
    dJdx = [np.zeros(blk * nn) for _ in range(3)]
    for i in range(M):
        dJdx[-1][2 * i::blk] = fwd.steps[-1]["phis"][i] - 0.30
    names = ["chi_0_1", "N_0", "onsager_0_0", "kappa_0"]
    g = MultiCHAdjoint(fwd).gradient(dJdx, names)
    assert set(g) == set(names)
    for k, v in g.items():
        assert np.isfinite(v), k


# ==========================================================================
# Task 5: MultiCHTwin + the THREE-WAY gate (hand adjoint == twin == FD)
# ==========================================================================
def _three_way_multi(dm, coords, M, order, n_steps, dt=0.01):
    from diffsim.adjoint.multiphase import (MultiCHForward, MultiCHAdjoint,
                                            FHMultiEnergy)
    from diffsim.adjoint.torch_twin import MultiCHTwin
    nn = dm.n_nodes
    blk = 2 * M
    chi0 = np.zeros((M + 1, M + 1))
    for a in range(M + 1):
        for b in range(a + 1, M + 1):
            chi0[a, b] = chi0[b, a] = 2.5 if (a, b) == (0, 1) else 1.0
    N0 = 1.0 + 0.3 * np.arange(M + 1)          # non-trivial N to exercise N grads
    ons0 = np.eye(M) + 0.1 * (np.ones((M, M)) - np.eye(M))
    kap0 = [0.01 * (i + 1) for i in range(M)]
    cc = np.cos(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])
    phi0 = [0.28 + 0.05 * cc for _ in range(M)]
    tgt = 0.28
    names = (["chi_0_1", f"chi_0_{M}"] + [f"N_{i}" for i in range(M + 1)]
             + ["onsager_0_0", "onsager_0_1"] + ["kappa_0"])

    def run(chi, N, ons, kap, record=False):
        fwd = MultiCHForward(dm, FHMultiEnergy(chi, N), onsager=ons,
                             kappa=list(kap), dt=dt, order=order)
        fwd.set_initial(phi0)
        fwd.run(n_steps)
        xs = [fwd.steps[-1]["phis"][i] for i in range(M)]
        J = 0.5 * float(sum(((x - tgt) ** 2).sum() for x in xs))
        return (fwd, J) if record else J

    fwd, _ = run(chi0, N0, ons0, kap0, record=True)
    adj = MultiCHAdjoint(fwd)
    dJdx = [np.zeros(blk * nn) for _ in range(n_steps)]
    for i in range(M):
        dJdx[-1][2 * i::blk] = fwd.steps[-1]["phis"][i] - tgt
    g_adj = adj.gradient(dJdx, names)

    def fd(name):
        eps = 1e-6

        def bump(sign):
            c_, N_, o_, k_ = (chi0.copy(), N0.copy(), ons0.copy(),
                              list(kap0))
            if name.startswith("chi_"):
                _, a, b = name.split("_"); a, b = int(a), int(b)
                c_[a, b] += sign * eps; c_[b, a] += sign * eps
            elif name.startswith("N_"):
                N_[int(name.split("_")[1])] += sign * eps
            elif name.startswith("onsager_"):
                _, a, b = name.split("_"); a, b = int(a), int(b)
                o_[a, b] += sign * eps
            else:
                k_[int(name.split("_")[1])] += sign * eps
            return run(c_, N_, o_, k_)
        return (bump(+1) - bump(-1)) / (2 * eps)
    g_fd = {nm: fd(nm) for nm in names}

    twin = MultiCHTwin(dm, M, dt=dt, order=order, device="cpu")
    g_tw = twin.grads(phi0, chi0, N0, ons0, kap0, n_steps, names, tgt)
    return {nm: (g_adj[nm], g_tw[nm], g_fd[nm]) for nm in names}


def _check_three_way(res, tag):
    for p, (a, t, f) in res.items():
        r_t = abs(a - t) / max(abs(t), 1e-14)
        r_f = abs(a - f) / max(abs(f), 1e-14)
        print(f"{tag} {p:12s} adj={a:+.6e} twin={t:+.6e} fd={f:+.6e} "
              f"adj/twin={r_t:.2e} adj/fd={r_f:.2e}")
        assert r_t < 1e-10, (p, a, t)
        assert r_f < 1e-6, (p, a, f)


def _three_way_meanphi(dm, coords, M, order, n_steps, dt=0.01):
    from diffsim.adjoint.multiphase import (MultiCHForward, MultiCHAdjoint,
                                            FHMultiEnergy)
    from diffsim.adjoint.torch_twin import MultiCHTwin
    nn = dm.n_nodes
    blk = 2 * M
    chi0 = np.zeros((M + 1, M + 1))
    for a in range(M + 1):
        for b in range(a + 1, M + 1):
            chi0[a, b] = chi0[b, a] = 2.5 if (a, b) == (0, 1) else 1.0
    N0 = 1.0 + 0.3 * np.arange(M + 1)
    ons0 = np.eye(M) + 0.1 * (np.ones((M, M)) - np.eye(M))
    kap0 = [0.01 * (i + 1) for i in range(M)]
    cc = np.cos(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])
    base = [0.28 + 0.05 * cc for _ in range(M)]
    tgt = 0.28
    names = [f"phi0_{i}" for i in range(M)]

    def run(phi0, record=False):
        fwd = MultiCHForward(dm, FHMultiEnergy(chi0, N0), onsager=ons0,
                             kappa=list(kap0), dt=dt, order=order)
        fwd.set_initial(phi0)
        fwd.run(n_steps)
        xs = [fwd.steps[-1]["phis"][i] for i in range(M)]
        J = 0.5 * float(sum(((x - tgt) ** 2).sum() for x in xs))
        return (fwd, J) if record else J

    fwd, _ = run(base, record=True)
    dJdx = [np.zeros(blk * nn) for _ in range(n_steps)]
    for i in range(M):
        dJdx[-1][2 * i::blk] = fwd.steps[-1]["phis"][i] - tgt
    g_adj = MultiCHAdjoint(fwd).gradient(dJdx, names)

    eps = 1e-6
    g_fd = {}
    for i in range(M):
        hi = [base[j] + (eps if j == i else 0.0) for j in range(M)]
        lo = [base[j] - (eps if j == i else 0.0) for j in range(M)]
        g_fd[f"phi0_{i}"] = (run(hi) - run(lo)) / (2 * eps)

    twin = MultiCHTwin(dm, M, dt=dt, order=order, device="cpu")
    g_tw = twin.grads(base, chi0, N0, ons0, kap0, n_steps, names, tgt)
    return {nm: (g_adj[nm], g_tw[nm], g_fd[nm]) for nm in names}


def test_three_way_meanphi_ternary_bdf1(device):
    dm, mesh = _dm(3)
    res = _three_way_meanphi(dm, mesh.node_coords, M=2, order=1, n_steps=3)
    _check_three_way(res, "MP-T-bdf1")


def test_three_way_meanphi_ternary_bdf2(device):
    dm, mesh = _dm(3)
    res = _three_way_meanphi(dm, mesh.node_coords, M=2, order=2, n_steps=4)
    _check_three_way(res, "MP-T-bdf2")


def test_three_way_meanphi_quaternary_bdf1(device):
    dm, mesh = _dm(2)
    res = _three_way_meanphi(dm, mesh.node_coords, M=3, order=1, n_steps=2)
    _check_three_way(res, "MP-Q-bdf1")


def test_three_way_ternary_bdf1(device):
    dm, mesh = _dm(3)
    res = _three_way_multi(dm, mesh.node_coords, M=2, order=1, n_steps=3)
    _check_three_way(res, "T-bdf1")


def test_three_way_ternary_bdf2(device):
    dm, mesh = _dm(3)
    res = _three_way_multi(dm, mesh.node_coords, M=2, order=2, n_steps=4)
    _check_three_way(res, "T-bdf2")


def test_three_way_quaternary_bdf1(device):
    dm, mesh = _dm(2)
    res = _three_way_multi(dm, mesh.node_coords, M=3, order=1, n_steps=2)
    _check_three_way(res, "Q-bdf1")


# ==========================================================================
# Task 6: MultiCHTwin mean-phi offset leaf (twin == FD for phi0 group)
# ==========================================================================
def test_meanphi_twin_vs_fd_ternary_bdf1():
    from diffsim.adjoint.torch_twin import MultiCHTwin
    import numpy as np
    dm, mesh = _dm(3)
    coords = mesh.node_coords
    M, order, n_steps, dt = 2, 1, 3, 0.01
    chi0 = np.zeros((M + 1, M + 1))
    chi0[0, 1] = chi0[1, 0] = 2.5
    chi0[0, 2] = chi0[2, 0] = 1.0
    chi0[1, 2] = chi0[2, 1] = 1.0
    N0 = 1.0 + 0.3 * np.arange(M + 1)
    ons0 = np.eye(M)
    kap0 = [0.01, 0.02]
    cc = np.cos(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])
    base = [0.28 + 0.05 * cc for _ in range(M)]
    tgt = 0.28
    names = [f"phi0_{i}" for i in range(M)]
    twin = MultiCHTwin(dm, M, dt=dt, order=order, device="cpu")
    g_tw = twin.grads(base, chi0, N0, ons0, kap0, n_steps, names, tgt)

    import torch
    def loss_of(phi0):
        out = twin.march(
            [torch.tensor(np.asarray(p)) for p in phi0],
            torch.tensor(chi0), torch.tensor(N0), torch.tensor(ons0),
            [torch.tensor(k) for k in kap0], n_steps)
        xN = out[-1]
        return 0.5 * float(sum(((xN[2 * i::2 * M] - tgt) ** 2).sum()
                               for i in range(M)))
    eps = 1e-6
    for i in range(M):
        hi = [base[j] + (eps if j == i else 0.0) for j in range(M)]
        lo = [base[j] - (eps if j == i else 0.0) for j in range(M)]
        fd = (loss_of(hi) - loss_of(lo)) / (2 * eps)
        rel = abs(g_tw[f"phi0_{i}"] - fd) / max(abs(fd), 1e-14)
        print(f"phi0_{i}: twin={g_tw[f'phi0_{i}']:+.6e} fd={fd:+.6e} rel={rel:.2e}")
        assert rel < 1e-6, (i, g_tw[f"phi0_{i}"], fd, rel)


# ==========================================================================
# Task 1 (Rung 2): LinearBackend protocol + ScipyBackend
# ==========================================================================
def test_scipy_backend_solve_and_transpose():
    import numpy as np
    import scipy.sparse as sp
    from diffsim.adjoint import ScipyBackend
    A = sp.csr_matrix(np.array([[3.0, 1.0, 0.0],
                                [0.0, 2.0, 1.0],
                                [1.0, 0.0, 4.0]]))
    b = np.array([1.0, -2.0, 3.0])
    be = ScipyBackend()
    x = be.solve(A, b)
    assert np.allclose(A @ x, b, atol=1e-12)
    xt = be.solve_T(A, b)
    assert np.allclose(A.T @ xt, b, atol=1e-12)


# ==========================================================================
# Task 2 (Rung 2): Route engine solves through LinearBackend
# ==========================================================================
def test_forward_accepts_backend_and_defaults_scipy():
    from diffsim.adjoint import ScipyBackend
    from diffsim.adjoint.multiphase import MultiCHForward, FHMultiEnergy
    import numpy as np
    # Build a ternary dm inline (same as test_forward_parity_ternary)
    dm, mesh = _dm(3)
    M = 2
    chi = np.zeros((M + 1, M + 1))
    chi[0, 1] = chi[1, 0] = 2.5
    chi[0, 2] = chi[2, 0] = 1.0
    chi[1, 2] = chi[2, 1] = 0.8
    N = np.ones(M + 1)
    en = FHMultiEnergy(chi, N)
    fwd = MultiCHForward(dm, en, onsager=np.eye(2), kappa=[1e-3, 1e-3], dt=1e-3)
    assert isinstance(fwd.backend, ScipyBackend)
    be = ScipyBackend()
    fwd2 = MultiCHForward(dm, en, onsager=np.eye(2), kappa=[1e-3, 1e-3],
                          dt=1e-3, backend=be)
    assert fwd2.backend is be


def test_scipy_to_torch_csr_roundtrip_and_transpose():
    import scipy.sparse as sp
    torch = pytest.importorskip("torch")
    from diffsim.adjoint.linsolve_backend import scipy_to_torch_csr
    A = sp.csr_matrix(np.array([[3.0, 1.0, 0.0],
                                [0.0, 2.0, 1.0],
                                [1.0, 0.0, 4.0]]))
    At = scipy_to_torch_csr(A, torch.device("cpu"), torch)
    # dense round-trip matches scipy
    assert np.allclose(At.to_dense().cpu().numpy(), A.toarray())
    # transpose path (what solve_T feeds cuDSS) matches scipy A.T
    ATt = scipy_to_torch_csr(A.T.tocsr(), torch.device("cpu"), torch)
    assert np.allclose(ATt.to_dense().cpu().numpy(), A.toarray().T)
    assert At.dtype == torch.float64


# ==========================================================================
# Task 1 (mean-phi / x0 cotangent): adj vs FD for initial-condition gradient
# ==========================================================================
def _meanphi_adj_vs_fd(dm, coords, M, order, n_steps, dt=0.01):
    from diffsim.adjoint.multiphase import (MultiCHForward, MultiCHAdjoint,
                                            FHMultiEnergy)
    nn = dm.n_nodes
    blk = 2 * M
    chi0 = np.zeros((M + 1, M + 1))
    for a in range(M + 1):
        for b in range(a + 1, M + 1):
            chi0[a, b] = chi0[b, a] = 2.5 if (a, b) == (0, 1) else 1.0
    N0 = 1.0 + 0.3 * np.arange(M + 1)
    ons0 = np.eye(M) + 0.1 * (np.ones((M, M)) - np.eye(M))
    kap0 = [0.01 * (i + 1) for i in range(M)]
    cc = np.cos(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])
    base = [0.28 + 0.05 * cc for _ in range(M)]
    tgt = 0.28
    names = [f"phi0_{i}" for i in range(M)]

    def run(phi0, record=False):
        fwd = MultiCHForward(dm, FHMultiEnergy(chi0, N0), onsager=ons0,
                             kappa=list(kap0), dt=dt, order=order)
        fwd.set_initial(phi0)
        fwd.run(n_steps)
        xs = [fwd.steps[-1]["phis"][i] for i in range(M)]
        J = 0.5 * float(sum(((x - tgt) ** 2).sum() for x in xs))
        return (fwd, J) if record else J

    fwd, _ = run(base, record=True)
    dJdx = [np.zeros(blk * nn) for _ in range(n_steps)]
    for i in range(M):
        dJdx[-1][2 * i::blk] = fwd.steps[-1]["phis"][i] - tgt
    g_adj = MultiCHAdjoint(fwd).gradient(dJdx, names)

    eps = 1e-6
    g_fd = {}
    for i in range(M):
        hi = [base[j] + (eps if j == i else 0.0) for j in range(M)]
        lo = [base[j] - (eps if j == i else 0.0) for j in range(M)]
        g_fd[f"phi0_{i}"] = (run(hi) - run(lo)) / (2 * eps)
    return {nm: (g_adj[nm], g_fd[nm]) for nm in names}


def test_meanphi_adj_vs_fd_ternary_bdf1():
    dm, mesh = _dm(3)
    res = _meanphi_adj_vs_fd(dm, mesh.node_coords, M=2, order=1, n_steps=3)
    for p, (a, f) in res.items():
        rel = abs(a - f) / max(abs(f), 1e-14)
        print(f"meanphi-bdf1 {p} adj={a:+.6e} fd={f:+.6e} rel={rel:.2e}")
        assert rel < 1e-6, (p, a, f)


def test_meanphi_adj_vs_fd_ternary_bdf2():
    dm, mesh = _dm(3)
    res = _meanphi_adj_vs_fd(dm, mesh.node_coords, M=2, order=2, n_steps=4)
    for p, (a, f) in res.items():
        rel = abs(a - f) / max(abs(f), 1e-14)
        print(f"meanphi-bdf2 {p} adj={a:+.6e} fd={f:+.6e} rel={rel:.2e}")
        assert rel < 1e-6, (p, a, f)


def test_cudss_backend_importable_on_cpu():
    # construction must not require CUDA (torch imported lazily); this lets the
    # engine + daisy-morph wiring be unit-tested on the Mac.
    from diffsim.adjoint import CudssBackend
    be = CudssBackend(device="cuda:0")
    assert be.device_str == "cuda:0"


def _gpu_available():
    try:
        import warp as wp
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_available(), reason="no CUDA GPU (rung-2 gate runs on gpubox)")
def test_cudss_backend_matches_scipy_on_gpu():
    import scipy.sparse as sp
    from diffsim.adjoint import ScipyBackend, CudssBackend
    rng = np.random.default_rng(0)
    n = 200
    A = sp.random(n, n, density=0.02, format="csr", random_state=0)
    A = A + sp.eye(n) * 5.0            # well-conditioned, nonsymmetric
    b = rng.standard_normal(n)
    xs = ScipyBackend().solve(A, b)
    xg = CudssBackend("cuda:0").solve(A, b)
    assert np.allclose(xs, xg, rtol=1e-9, atol=1e-11)
    xts = ScipyBackend().solve_T(A, b)
    xtg = CudssBackend("cuda:0").solve_T(A, b)
    assert np.allclose(xts, xtg, rtol=1e-9, atol=1e-11)

    # nnz-flap: reuse ONE backend instance across two different-nnz matrices so
    # the plan-rebuild/.free() branch is exercised on hardware (production saw
    # the CSR pattern grow mid-march; that branch is otherwise untested).
    be2 = CudssBackend("cuda:0")
    A1 = sp.random(n, n, density=0.02, format="csr", random_state=1) + sp.eye(n) * 5.0
    x1 = be2.solve(A1, b)
    assert np.allclose(x1, ScipyBackend().solve(A1, b), rtol=1e-9, atol=1e-11)
    A2 = A1.tolil(); A2[0, n - 1] += 1.0; A2 = A2.tocsr()   # adds one entry -> different nnz
    assert A2.nnz != A1.nnz
    x2 = be2.solve(A2, b)
    assert np.allclose(x2, ScipyBackend().solve(A2, b), rtol=1e-9, atol=1e-11)
