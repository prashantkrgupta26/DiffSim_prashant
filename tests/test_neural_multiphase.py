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
