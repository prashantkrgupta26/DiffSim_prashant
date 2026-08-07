"""Tests for BasisMultiEnergy (Task 1, M6: gauge-anchored learnable bulk energy)
and MobilityClosure (Task 2, M6: named learnable mobility).

TDD structure:
  - test_basis_reduces_to_fh_at_zero_coeffs  (brief Step 1 / parity)
  - test_basis_correction_is_gauge_orthogonal (brief Step 1 / gauge)
  - test_basis_derivs_complex_step           (added per task description)
  - test_const_closure_matches_current_engine (Task 2 / const back-compat)
  - test_phi_diag_closure_complex_step        (Task 2 / mob param dR/dparam)
  - test_phi_diag_jacobian_complex_step       (Task 2 / phi-dep mobility Jacobian)
"""
import numpy as np
import pytest

from diffsim.adjoint import BasisMultiEnergy, FHMultiEnergy


def _chiN(M=2):
    chi = np.zeros((M + 1, M + 1))
    chi[0, 1] = chi[1, 0] = 2.5
    chi[0, 2] = chi[2, 0] = 1.0
    chi[1, 2] = chi[2, 1] = 0.8
    return chi, np.ones(M + 1)


# --------------------------------------------------------------------------
# Test 1: FH parity when all basis coefficients are zero
# --------------------------------------------------------------------------
def test_basis_reduces_to_fh_at_zero_coeffs():
    chi, N = _chiN()
    be = BasisMultiEnergy(chi, N, degrees=(2, 3))          # coeffs default 0
    fh = FHMultiEnergy(chi, N)
    phis = [np.full(5, 0.30), np.full(5, 0.32)]
    for a, b in zip(be.mu(phis), fh.mu(phis)):
        assert np.allclose(a, b)
    assert any(nm.startswith("basis_") for nm in be.param_names)


# --------------------------------------------------------------------------
# Test 2: gauge orthogonality — correction is L2-orthogonal to {1, phi_i}
# --------------------------------------------------------------------------
def test_basis_correction_is_gauge_orthogonal():
    chi, N = _chiN()
    be = BasisMultiEnergy(chi, N, degrees=(2, 3),
                          coeffs={"basis_0_2": 0.7, "basis_0_3": -0.4})
    # <corr_mu_0, 1> and <corr_mu_0, phi_0> over the domain must be ~0
    r0, r1 = be.gauge_residual(species=0)
    assert abs(r0) < 1e-10 and abs(r1) < 1e-10


# --------------------------------------------------------------------------
# Test 3: complex-step verification of dmu_dphi and dmu_dparam
# --------------------------------------------------------------------------
def _cstep_mu(be, phis, j):
    """Complex-step dmu/d(phis[j]) — returns list of M arrays."""
    h = 1e-30
    p = [x.astype(complex) for x in phis]
    p[j] = p[j] + 1j * h
    return [mu_i.imag / h for mu_i in be.mu(p)]


def test_basis_derivs_complex_step():
    """dmu_dphi diagonal correction and dmu_dparam('basis_i_k') vs complex step.

    Non-trivial coefficients so the correction path is exercised.
    Uses 1e-30 imaginary perturbation (same idiom as test_energy_derivs_complex_step).
    """
    chi, N = _chiN(M=2)
    coeffs = {"basis_0_2": 0.5, "basis_0_3": -0.3,
              "basis_1_2": 0.2, "basis_1_3": 0.1}
    be = BasisMultiEnergy(chi, N, degrees=(2, 3), coeffs=coeffs)
    rng = np.random.default_rng(42)
    # compositions well inside dom=(0.05, 0.95)
    phis = [0.20 + 0.15 * rng.random(8), 0.25 + 0.15 * rng.random(8)]

    M = 2
    H = be.dmu_dphi(phis)

    # --- dmu_dphi: diagonal ---
    for j in range(M):
        cs_col = _cstep_mu(be, phis, j)
        for i in range(M):
            assert np.allclose(H[i][j], cs_col[i], atol=1e-9, rtol=1e-7), \
                f"dmu_dphi[{i}][{j}] mismatch"

    # --- dmu_dparam for basis names ---
    h = 1e-30
    for name in [n for n in be.param_names if n.startswith("basis_")]:
        _, si, sk = name.split("_")
        i_sp, k_deg = int(si), int(sk)
        # complex-step w.r.t. this gamma: perturb gamma[(i_sp, k_deg)] by 1j*h
        old = be.gamma[(i_sp, k_deg)]
        be.gamma[(i_sp, k_deg)] = old + 1j * h
        p_c = [x.astype(complex) for x in phis]
        mu_c = be.mu(p_c)
        cs_dparam = [mu_c[i].imag / h for i in range(M)]
        be.gamma[(i_sp, k_deg)] = old

        an_dparam = be.dmu_dparam(phis, name)
        for i in range(M):
            assert np.allclose(an_dparam[i], cs_dparam[i], atol=1e-9, rtol=1e-7), \
                f"dmu_dparam({name})[{i}] mismatch"


# ==========================================================================
# Task 2 (M6): MobilityClosure + engine routing
# ==========================================================================

def _dm(level, dim=2):
    """Reuse mesh helper from test_multiphase_adjoint."""
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


def test_const_closure_matches_current_engine():
    """A const closure must reproduce the existing constant-Onsager forward
    bit-for-bit: plain matrix and explicit MobilityClosure('const') give identical
    trajectories."""
    from diffsim.adjoint import MobilityClosure
    from diffsim.adjoint.multiphase import MultiCHForward, FHMultiEnergy
    M = 2
    dm, mesh = _dm(3)
    chi = np.zeros((M + 1, M + 1))
    chi[0, 1] = chi[1, 0] = 2.5
    chi[0, 2] = chi[2, 0] = 1.0
    chi[1, 2] = chi[2, 1] = 0.8
    en = FHMultiEnergy(chi, np.ones(M + 1))
    ons = np.eye(M)
    cc = np.cos(np.pi * mesh.node_coords[:, 0]) * np.cos(np.pi * mesh.node_coords[:, 1])
    ic = [0.30 + 0.05 * cc, 0.30 + 0.05 * cc]
    # plain matrix path (legacy API)
    f_mat = MultiCHForward(dm, en, mobility=ons, kappa=[1e-2, 1e-2], dt=1e-3)
    f_mat.set_initial(ic)
    f_mat.run(3)
    # explicit closure path
    f_cls = MultiCHForward(dm, en,
                           mobility=MobilityClosure("const", M=M, onsager=ons),
                           kappa=[1e-2, 1e-2], dt=1e-3)
    f_cls.set_initial(ic)
    f_cls.run(3)
    for i in range(M):
        assert np.allclose(f_mat.steps[-1]["phis"][i],
                           f_cls.steps[-1]["phis"][i]), i


def test_phi_diag_closure_complex_step():
    """dR/d(mob_param) via complex step matches dR_dparam for phi_diag closure."""
    from diffsim.adjoint import MobilityClosure
    from diffsim.adjoint.multiphase import MultiCHDiscrete, FHMultiEnergy
    M = 2
    dm, mesh = _dm(2)
    op = MultiCHDiscrete(dm, M)
    rng = np.random.default_rng(20)
    chi = 0.3 + 0.4 * rng.random((M + 1, M + 1))
    chi = 0.5 * (chi + chi.T); np.fill_diagonal(chi, 0.0)
    en = FHMultiEnergy(chi, 1.0 + 0.5 * rng.random(M + 1))
    phis = [0.20 + 0.06 * rng.random(op.nn) for _ in range(M)]
    mus  = [0.05 * rng.random(op.nn) for _ in range(M)]
    mob_coeffs = {"mob_m0": 1.2, "mob_c": 0.5}
    mob = MobilityClosure("phi_diag", M=M, coeffs=mob_coeffs)
    params = dict(mobility=mob, kappa=[0.01, 0.02], energy=en, sigma=9.0)
    hist = [[np.zeros_like(B["dJxW"]) for B in op.bins] for _ in range(M)]

    h = 1e-30
    for pname in ("mob_m0", "mob_c"):
        an = op.dR_dparam(phis, mus, params, pname)
        # complex-step: perturb the coefficient and re-evaluate residual
        old_val = mob.coeffs[pname]
        mob.coeffs[pname] = old_val + 1j * h
        mob_c = MobilityClosure("phi_diag", M=M,
                                coeffs={k: mob.coeffs[k] for k in mob.coeffs})
        pp = dict(params, mobility=mob_c)
        pf = [x.astype(complex) for x in phis]
        mf = [x.astype(complex) for x in mus]
        Rc, _ = op.assemble(pf, mf, hist, pp, want_jac=False)
        cs = Rc.imag / h
        mob.coeffs[pname] = old_val
        assert np.allclose(an, cs, atol=1e-8, rtol=1e-6), pname


def test_phi_diag_jacobian_complex_step():
    """Full assemble Jacobian for phi_diag closure vs complex-step: verifies the
    extra phi-dependent-mobility dM/dphi term in Ae[2i::blk, 2i::blk]."""
    from diffsim.adjoint import MobilityClosure
    from diffsim.adjoint.multiphase import MultiCHDiscrete, FHMultiEnergy
    M = 2
    dm, mesh = _dm(2)
    op = MultiCHDiscrete(dm, M)
    rng = np.random.default_rng(30)
    chi = 0.3 + 0.4 * rng.random((M + 1, M + 1))
    chi = 0.5 * (chi + chi.T); np.fill_diagonal(chi, 0.0)
    en = FHMultiEnergy(chi, 1.0 + 0.5 * rng.random(M + 1))
    phis = [0.20 + 0.06 * rng.random(op.nn) for _ in range(M)]
    mus  = [0.05 * rng.random(op.nn) for _ in range(M)]
    mob = MobilityClosure("phi_diag", M=M, coeffs={"mob_m0": 1.3, "mob_c": 0.4})
    params = dict(mobility=mob, kappa=[0.01, 0.02], energy=en, sigma=11.5)
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

    rng2 = np.random.default_rng(31)
    for k in rng2.integers(0, op.ndof, size=20):
        got = np.asarray(J[:, int(k)].todense()).ravel()
        assert np.allclose(got, col(int(k)), atol=1e-8, rtol=1e-6), int(k)


# ==========================================================================
# Task 3 (M6): MultiCHTwin — basis/mob/tau grads vs central FD
# ==========================================================================
def test_twin_basis_mob_tau_vs_fd():
    """MultiCHTwin.grads for basis_0_2 / mob_m0 / tau vs central FD.

    Ternary (M=2), BDF1, 3 steps.  BasisMultiEnergy(degrees=(2,3)) + phi_diag
    MobilityClosure.  Compare each twin grad against central FD of the same
    loss = 0.5 * sum_i || phi_i,N - tgt ||^2.  Rel error < 1e-6.
    """
    import torch
    from diffsim.adjoint import BasisMultiEnergy, MobilityClosure
    from diffsim.adjoint.torch_twin import MultiCHTwin

    M = 2
    dm, mesh = _dm(3)
    coords = mesh.node_coords
    cc = np.cos(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])

    # ---- base parameters --------------------------------------------------
    chi0 = np.zeros((M + 1, M + 1))
    chi0[0, 1] = chi0[1, 0] = 2.5
    chi0[0, 2] = chi0[2, 0] = 1.0
    chi0[1, 2] = chi0[2, 1] = 0.8
    N0 = np.ones(M + 1)
    ons0 = np.eye(M)   # used only as placeholder (overridden by mob_closure)
    kap0 = [0.01, 0.02]
    dt = 0.01
    n_steps = 3
    tgt = 0.28
    phi0 = [0.28 + 0.05 * cc, 0.28 + 0.05 * cc]

    # non-zero basis coefficients so the gradient is non-trivial
    basis_coeffs = {"basis_0_2": 0.15, "basis_0_3": -0.10,
                    "basis_1_2": 0.08, "basis_1_3": 0.05}
    basis_en = BasisMultiEnergy(chi0, N0, degrees=(2, 3), coeffs=basis_coeffs)

    mob_coeffs = {"mob_m0": 1.2, "mob_c": 0.4}
    mob_cl = MobilityClosure("phi_diag", M=M, coeffs=mob_coeffs)

    twin = MultiCHTwin(dm, M, dt=dt, order=1, device="cpu")

    names = ["basis_0_2", "mob_m0", "tau"]
    g_tw = twin.grads(phi0, chi0, N0, ons0, kap0, n_steps, names, tgt,
                      basis_energy=basis_en, mob_closure=mob_cl)

    # ---- central FD reference using the SAME loss via twin.march ----------
    def loss_of(basis_coeffs_d, mob_coeffs_d, tau_val):
        """Re-run twin march with perturbed params; return scalar loss."""
        mid, half = 0.5, 0.45
        degrees = (2, 3)
        # build basis_leaves (constant tensors, no grad)
        bl = {}
        for i in range(M):
            for k in degrees:
                nm = f"basis_{i}_{k}"
                bl[(i, k)] = torch.tensor(
                    float(basis_coeffs_d.get(nm, 0.0)), dtype=torch.float64)
        bmeta = dict(mid=mid, half=half, degrees=degrees)
        mob_lvs = {pn: torch.tensor(float(mob_coeffs_d[pn]), dtype=torch.float64)
                   for pn in ("mob_m0", "mob_c")}
        tau_t = torch.tensor(float(tau_val), dtype=torch.float64)
        chi_t = torch.tensor(chi0, dtype=torch.float64)
        N_t = torch.tensor(N0, dtype=torch.float64)
        phis_t = [torch.tensor(np.asarray(phi0[i]), dtype=torch.float64)
                  for i in range(M)]
        kap_t = [torch.tensor(k, dtype=torch.float64) for k in kap0]
        out = twin.march(phis_t, chi_t, N_t, None, kap_t, n_steps,
                         tau=tau_t, basis_leaves=bl, basis_meta=bmeta,
                         mob_leaves=mob_lvs)
        xN = out[-1]
        blk = 2 * M
        tgt_t = torch.tensor(float(tgt), dtype=torch.float64)
        return float(0.5 * sum(((xN[2 * i::blk] - tgt_t) ** 2).sum()
                               for i in range(M)))

    eps = 1e-5  # central FD step (tau/mob are order-1 params; 1e-5 is safe)

    # FD for basis_0_2
    bc_hi = dict(basis_coeffs); bc_hi["basis_0_2"] += eps
    bc_lo = dict(basis_coeffs); bc_lo["basis_0_2"] -= eps
    fd_basis02 = (loss_of(bc_hi, mob_coeffs, 1.0)
                  - loss_of(bc_lo, mob_coeffs, 1.0)) / (2 * eps)

    # FD for mob_m0
    mc_hi = dict(mob_coeffs); mc_hi["mob_m0"] += eps
    mc_lo = dict(mob_coeffs); mc_lo["mob_m0"] -= eps
    fd_mob_m0 = (loss_of(basis_coeffs, mc_hi, 1.0)
                 - loss_of(basis_coeffs, mc_lo, 1.0)) / (2 * eps)

    # FD for tau
    fd_tau = (loss_of(basis_coeffs, mob_coeffs, 1.0 + eps)
              - loss_of(basis_coeffs, mob_coeffs, 1.0 - eps)) / (2 * eps)

    fd = {"basis_0_2": fd_basis02, "mob_m0": fd_mob_m0, "tau": fd_tau}

    for nm in names:
        tw_val = g_tw[nm]
        fd_val = fd[nm]
        rel = abs(tw_val - fd_val) / max(abs(fd_val), 1e-14)
        print(f"{nm}: twin={tw_val:+.6e}  fd={fd_val:+.6e}  rel={rel:.2e}")
        assert rel < 1e-6, (nm, tw_val, fd_val, rel)
