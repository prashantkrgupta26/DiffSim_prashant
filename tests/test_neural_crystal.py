"""Sub-project 2: NeuralCrystalEnergy — non-parametric coupled free energy."""
import numpy as np
import pytest
from diffsim.adjoint.neural_crystal import NeuralCrystalEnergy, _legendre2_np

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


# ==========================================================================
# Task 4: THREE-WAY GATE — hand adjoint == autograd twin == finite differences
#          for NeuralCrystalEnergy (cpl_* + basis_* + chi/N + engine params)
# ==========================================================================

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


def _check_three_way(res, tag):
    """Check adj/twin < 1e-10, adj/fd < 1e-6 for every param in res."""
    for p, (a, t, f) in res.items():
        r_t = abs(a - t) / max(abs(t), 1e-14)
        r_f = abs(a - f) / max(abs(f), 1e-14)
        print(f"{tag} {p:14s} adj={a:+.6e} twin={t:+.6e} fd={f:+.6e} "
              f"adj/twin={r_t:.2e} adj/fd={r_f:.2e}")
        assert r_t < 1e-10, f"{tag} {p}: adj={a} twin={t} ratio={r_t}"
        assert r_f < 1e-6, f"{tag} {p}: adj={a} fd={f} ratio={r_f}"


def _three_way_neural_crystal(dm, coords, M, crystallizable, deg_psi,
                               order, n_steps, dt=1e-2):
    """Three-way gate for NeuralCrystalEnergy with CrystalCHForward/Adjoint
    vs CrystalCHTwin vs central FD.

    Loss = 0.5 sum_i ||phi_i,N - tgt||^2 + 0.5 sum_k ||psi_k,N - tgt||^2.
    Returns {name: (adj, twin, fd)}.
    """
    from diffsim.adjoint import MobilityClosure
    from diffsim.adjoint.crystallization_multi import (
        CrystalCHForward, CrystalCHAdjoint)
    from diffsim.adjoint.torch_twin import CrystalCHTwin

    K = len(crystallizable)
    nn = dm.n_nodes
    blk = 2 * M + K

    # --- base params: strong 0-1 chi, non-uniform N, non-trivial coupling ---
    chi0 = np.zeros((M + 1, M + 1))
    for a in range(M + 1):
        for b in range(a + 1, M + 1):
            chi0[a, b] = chi0[b, a] = 2.5 if (a, b) == (0, 1) else 1.0
    N0 = 1.0 + 0.3 * np.arange(M + 1, dtype=float)
    ons0 = np.eye(M) + 0.1 * (np.ones((M, M)) - np.eye(M))
    kap0 = [0.01 * (i + 1) for i in range(M)]
    eps20 = {k: 0.012 + 0.003 * j for j, k in enumerate(crystallizable)}
    L0 = {k: 1.2 + 0.2 * j for j, k in enumerate(crystallizable)}

    # nonzero coupling coefficients for each crystallizable species
    coeffs0 = {}
    for k in crystallizable:
        for b in deg_psi:
            # stagger by species and degree for variety
            coeffs0[f"cpl_{k}_{b}"] = 0.10 * (1 + k * 0.3) * (1 - 0.2 * b % 3)
    # basis_coeffs set to zero: CrystalCHTwin._mu_and_H does not implement the
    # BasisMultiEnergy polynomial correction so nonzero basis coefficients
    # would cause the twin forward to diverge from the numpy forward, breaking
    # the adj/twin leg.  basis_* gradient verification is deferred until the
    # twin gains the basis_leaves branch.
    basis_coeffs0 = {}

    tgt = 0.22

    # cosine initial field: per-species offset to make chi_0_1 gradient non-tiny
    cc = np.cos(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])
    phi0 = [0.28 + 0.05 * cc * (-1) ** i for i in range(M)]
    psi0 = [0.20 + 0.04 * cc for _ in range(K)]

    # param names to gate (species-0 params; valid for all configs)
    # NOTE: basis_0_2 is excluded from names because CrystalCHTwin does not yet
    # implement the BasisMultiEnergy polynomial correction in its torch forward
    # march (its _mu_and_H uses only FH). Including basis_0_2 here would break
    # the adj/twin leg since the twin forward diverges from the numpy forward
    # when basis_coeffs are nonzero. This is a Task-3 gap: CrystalCHTwin needs
    # the same basis_leaves branch that MultiCHTwin already has.
    # TODO: re-enable "basis_0_2" once CrystalCHTwin._mu_and_H supports it.
    names = ["cpl_0_1", "cpl_0_2",
             "chi_0_1", "N_0", "onsager_0_0", "kappa_0", "eps2_0", "L_0"]

    # ---- helper: build a NeuralCrystalEnergy with given coeffs/basis --------
    def _make_neural_energy(coeffs=None, basis_coeffs=None):
        return NeuralCrystalEnergy(
            chi0, N0, crystallizable, deg_psi=deg_psi,
            coeffs=coeffs if coeffs is not None else dict(coeffs0),
            basis_coeffs=basis_coeffs if basis_coeffs is not None
            else dict(basis_coeffs0))

    # ---- helper: build and run CrystalCHForward ----------------------------
    def _make_fwd(energy, ons, kap, eps2, L_):
        mob = MobilityClosure("const", M=M, onsager=np.asarray(ons))
        fwd = CrystalCHForward(dm, energy, crystallizable=crystallizable,
                               mobility=mob,
                               kappa=list(kap), eps2=dict(eps2),
                               L=dict(L_), dt=dt, order=order)
        fwd.set_initial([p.copy() for p in phi0],
                        psi0_list=[p.copy() for p in psi0])
        fwd.run(n_steps)
        return fwd

    # ---- hand adjoint -------------------------------------------------------
    en0 = _make_neural_energy()
    fwd = _make_fwd(en0, ons0, kap0, eps20, L0)
    rec = fwd.steps[-1]

    dJdx_list = [np.zeros(blk * nn) for _ in range(n_steps)]
    for i in range(M):
        dJdx_list[-1][2 * i::blk] = rec["phis"][i] - tgt
    for j in range(K):
        dJdx_list[-1][2 * M + j::blk] = rec["psis"][j] - tgt
    g_adj = CrystalCHAdjoint(fwd).gradient(dJdx_list, names)

    # ---- autograd twin ------------------------------------------------------
    twin = CrystalCHTwin(dm, M, crystallizable, dt=dt, order=order,
                         device="cpu")
    # energy_params: zero additive coupling (coupling is purely neural)
    energy_params = dict(chi=chi0, N=N0,
                         dsig={k: 0.0 for k in crystallizable},
                         dh={k: 0.0 for k in crystallizable},
                         Tm={k: 1.0 for k in crystallizable},
                         T=0.5)
    engine_params = dict(onsager=ons0, kappa=kap0, eps2=eps20, L=L0)
    g_tw = twin.grads(phi0, psi0, energy_params, engine_params,
                      n_steps, names, tgt, neural_energy=en0)

    # ---- central FD of same numpy loss --------------------------------------
    def _loss(energy, ons, kap, eps2, L_):
        fwd_ = _make_fwd(energy, ons, kap, eps2, L_)
        rec_ = fwd_.steps[-1]
        loss = 0.5 * sum(((p - tgt) ** 2).sum() for p in rec_["phis"])
        loss += 0.5 * sum(((p - tgt) ** 2).sum() for p in rec_["psis"])
        return float(loss)

    def _fd(name):
        eps = 1e-6

        def bump(sign):
            ons_ = ons0.copy()
            kap_ = list(kap0)
            eps2_ = dict(eps20)
            L__ = dict(L0)
            # default: energy rebuilt from base coeffs
            cpl_ = dict(coeffs0)
            basis_ = dict(basis_coeffs0)
            chi_ = chi0.copy()
            N_ = N0.copy()
            if name.startswith("cpl_"):
                cpl_ = dict(coeffs0)
                cpl_[name] = cpl_.get(name, 0.0) + sign * eps
                en_ = _make_neural_energy(coeffs=cpl_, basis_coeffs=basis_)
            elif name.startswith("basis_"):
                basis_ = dict(basis_coeffs0)
                basis_[name] = basis_.get(name, 0.0) + sign * eps
                en_ = _make_neural_energy(coeffs=cpl_, basis_coeffs=basis_)
            elif name.startswith("chi_"):
                _, a, b = name.split("_"); a, b = int(a), int(b)
                chi_[a, b] += sign * eps; chi_[b, a] += sign * eps
                en_ = NeuralCrystalEnergy(chi_, N0, crystallizable,
                                           deg_psi=deg_psi, coeffs=cpl_,
                                           basis_coeffs=basis_)
            elif name.startswith("N_"):
                N_[int(name.split("_")[1])] += sign * eps
                en_ = NeuralCrystalEnergy(chi0, N_, crystallizable,
                                           deg_psi=deg_psi, coeffs=cpl_,
                                           basis_coeffs=basis_)
            elif name.startswith("onsager_"):
                _, a, b = name.split("_"); a, b = int(a), int(b)
                ons_[a, b] += sign * eps
                en_ = _make_neural_energy(coeffs=cpl_, basis_coeffs=basis_)
            elif name.startswith("kappa_"):
                kap_[int(name.split("_")[1])] += sign * eps
                en_ = _make_neural_energy(coeffs=cpl_, basis_coeffs=basis_)
            elif name.startswith("eps2_"):
                k = int(name.split("_")[1]); eps2_[k] += sign * eps
                en_ = _make_neural_energy(coeffs=cpl_, basis_coeffs=basis_)
            elif name.startswith("L_"):
                k = int(name.split("_")[1]); L__[k] += sign * eps
                en_ = _make_neural_energy(coeffs=cpl_, basis_coeffs=basis_)
            else:
                raise ValueError(name)
            return _loss(en_, ons_, kap_, eps2_, L__)

        return (bump(+1) - bump(-1)) / (2 * eps)

    g_fd = {nm: _fd(nm) for nm in names}
    return {nm: (g_adj[nm], g_tw[nm], g_fd[nm]) for nm in names}


def test_three_way_neural_ternary_bdf1():
    """M=2 ternary (crystallizable=(0,)), BDF1, n_steps=3: adj == twin == FD."""
    dm, mesh = _dm(2)
    res = _three_way_neural_crystal(
        dm, mesh.node_coords, M=2, crystallizable=(0,),
        deg_psi=(1, 2), order=1, n_steps=3)
    _check_three_way(res, "neural-ternary-bdf1")


def test_three_way_neural_ternary_bdf2():
    """M=2 ternary (crystallizable=(0,)), BDF2, n_steps=4: adj == twin == FD."""
    dm, mesh = _dm(2)
    res = _three_way_neural_crystal(
        dm, mesh.node_coords, M=2, crystallizable=(0,),
        deg_psi=(1, 2), order=2, n_steps=4)
    _check_three_way(res, "neural-ternary-bdf2")


def test_three_way_neural_quaternary_bdf1():
    """M=3 quaternary (crystallizable=(0,2)), BDF1, n_steps=3:
    both species 0 and 2 crystallize with neural coupling;
    adj == twin == FD for species-0 params."""
    dm, mesh = _dm(2)
    res = _three_way_neural_crystal(
        dm, mesh.node_coords, M=3, crystallizable=(0, 2),
        deg_psi=(1, 2), order=1, n_steps=3)
    _check_three_way(res, "neural-quaternary-bdf1")
