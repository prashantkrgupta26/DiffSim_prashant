# NeuralCrystalEnergy (sub-project ②) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A learnable, non-parametric coupled crystallization free energy `f(φ,ψ) = BasisMultiEnergy(φ) + Σ_k φ_k·h_k(ψ_k)` that drops into the `CrystalEnergy` protocol with zero engine change, three-way verified and recovered by trajectory-matching.

**Architecture:** Compose the M6 `BasisMultiEnergy` (FH + gauge-anchored φ-correction) as the pure-φ base and add a linear-in-φ, Legendre-in-ψ coupling `h_k(ψ)=Σ_b c_{k,b}L_b(2ψ−1)` with the constant mode (b=0) excluded as the conserved-φ_k gauge. The new energy object satisfies the existing `CrystalEnergy` protocol so `CrystalCHDiscrete/Forward/Adjoint` are untouched; `CrystalCHTwin` gets an additive coupling extension for the `cpl_*` leaves; the three-way gate and a synthetic-recovery driver close the loop.

**Tech Stack:** numpy (energy + hand adjoint), torch (autograd twin), scipy (sparse solve), pytest. CPU-only, `.venv/bin/pytest`.

**Spec:** `docs/superpowers/specs/2026-08-08-neural-crystal-energy-design.md`.

## Global Constraints

- **Coupling form:** linear in φ_k; `h_k(ψ)=Σ_{b∈deg_ψ} c_{k,b}·L_b(û_k)`, `û_k=2ψ_k−1` (ψ domain [0,1] → Legendre domain [−1,1]). `deg_ψ ⊆ {1,2,3,…}`; **b=0 excluded** (conserved-φ_k gauge — provably unidentifiable).
- **Base:** the pure-φ energy is `BasisMultiEnergy` (FH + Legendre φ-correction). The full `f(φ,ψ)` is learnable.
- **Parameter split:** bulk-energy params (`chi_a_b`, `N_i`, `basis_i_k`, `cpl_k_b`) route through the energy object; engine params (mobility/onsager, kappa, eps2, L) stay engine-level. No engine (`CrystalCHDiscrete/Forward/Adjoint`) edits.
- **Verification:** three-way rule — hand adjoint (`CrystalCHAdjoint`) == autograd twin (`CrystalCHTwin`) == central FD; adj/twin < 1e-10, adj/fd < 1e-6. Complex-step (1e-30) unit gate on analytic derivatives. Reduction gate against `AdditiveCrystalEnergy` on identifiable quantities.
- **Complex-dtype preservation:** no float64 casts on the φ/ψ/param path (the `_legendre*` helpers are pure polynomials; coeffs multiply arrays). Mirrors `BasisMultiEnergy`.
- **θ (orientation) absent.** Full 2-D `g_k(φ,ψ)`, χ(ψ), MD ingest all out of scope.
- **Env:** `.venv/bin/pytest` only (no bare `python`, no `timeout`). Mac CPU-only.
- **Commits** end with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

## File Structure

- `src/diffsim/adjoint/neural_crystal.py` — **new.** `_legendre2_np` + `NeuralCrystalEnergy` (implements `CrystalEnergy`).
- `src/diffsim/adjoint/__init__.py` — **modify.** Export `NeuralCrystalEnergy`.
- `src/diffsim/adjoint/torch_twin.py` — **modify.** Extend `CrystalCHTwin` with the torch Legendre-in-ψ coupling correction (additive; `_q_t/_p_t`-style helpers already there).
- `tests/test_neural_crystal.py` — **new.** All ② tests (unit complex-step, reduction gate, twin-vs-FD, three-way, recovery).
- `examples/crystal_learn_from_synthetic.py` — **new.** Recovery demo (Task 5).

Opportunistic ①-minor sweep (from the ① whole-branch review, both cosmetic): in `src/diffsim/adjoint/crystallization_multi.py` remove the dead `_rowslices()` method and simplify `self.K = len(tuple(crystallizable))` → `len(crystallizable)`. Fold into Task 1's commit.

---

### Task 1: `NeuralCrystalEnergy` energy object + complex-step unit gate

**Files:**
- Create: `src/diffsim/adjoint/neural_crystal.py`
- Modify: `src/diffsim/adjoint/__init__.py`, `src/diffsim/adjoint/crystallization_multi.py` (minor sweep)
- Test: `tests/test_neural_crystal.py`

**Interfaces:**
- Consumes: `BasisMultiEnergy`, `_legendre_np` (from `neural_multiphase.py`); `CrystalEnergy` (from `crystallization_multi.py`).
- Produces: `NeuralCrystalEnergy(chi, N, crystallizable, deg_psi=(1,2), coeffs=None, dom_phi=(0.05,0.95), breg=0.0, basis_degrees=(2,3), basis_coeffs=None)` implementing the full `CrystalEnergy` protocol (`dfdphi`, `dfdpsi`, `d2fdphidphi`, `d2fdphidpsi`, `d2fdpsidpsi`, `dfdphi_dparam`, `dfdpsi_dparam`; attrs `M`, `crystallizable`, `param_names`) plus `coupling_gauge_residual(k, n_quad=64)`. `coeffs` keyed `"cpl_{k}_{b}"`. Module-level `_legendre2_np(u, k)`.

- [ ] **Step 1: Write the failing tests** — `tests/test_neural_crystal.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest tests/test_neural_crystal.py -x -q`
Expected: FAIL — `ModuleNotFoundError: diffsim.adjoint.neural_crystal`.

- [ ] **Step 3: Implement** `src/diffsim/adjoint/neural_crystal.py`:

```python
r"""Sub-project 2: learnable coupled crystallization free energy.

f(phi, psi) = f_base(phi)  +  sum_{k in crystallizable} phi_k * h_k(psi_k)
  f_base  = BasisMultiEnergy (FH + gauge-anchored Legendre phi-correction)
  h_k(psi) = sum_{b in deg_psi} c_{k,b} L_b(u),  u = 2*psi - 1   (b >= 1)

The b=0 constant mode is EXCLUDED: h_k -> h_k + const shifts mu_k by a constant,
a pure {phi_k} conservation gauge on the conserved phi_k -> unidentifiable.

Implements the CrystalEnergy protocol so CrystalCHDiscrete/Forward/Adjoint are a
drop-in (zero engine change).  Complex-dtype preserving (pure-polynomial basis,
coeffs multiply arrays) for complex-step verification.
"""
import numpy as np
from .neural_multiphase import BasisMultiEnergy, _legendre_np
from .crystallization_multi import CrystalEnergy


def _legendre2_np(u, k):
    """Second derivative d^2 P_k / du^2 on [-1, 1], complex-safe, k in 0..5."""
    if k == 0 or k == 1:
        return np.zeros_like(u)
    if k == 2:
        return 3.0 * np.ones_like(u)
    if k == 3:
        return 15.0 * u
    if k == 4:
        return (105.0 * u * u - 15.0) / 2.0
    if k == 5:
        return (315.0 * u ** 3 - 105.0 * u) / 2.0
    raise ValueError(f"basis degree {k} not in 0..5")


class NeuralCrystalEnergy(CrystalEnergy):
    def __init__(self, chi, N, crystallizable, deg_psi=(1, 2), coeffs=None,
                 dom_phi=(0.05, 0.95), breg=0.0, basis_degrees=(2, 3),
                 basis_coeffs=None):
        self.base = BasisMultiEnergy(chi, N, degrees=basis_degrees,
                                     coeffs=basis_coeffs, dom=dom_phi, breg=breg)
        self.M = self.base.M
        self.crystallizable = tuple(int(k) for k in crystallizable)
        self.deg_psi = tuple(int(b) for b in deg_psi)
        assert all(b >= 1 for b in self.deg_psi), \
            "deg_psi excludes b=0 (conserved-phi_k gauge, unidentifiable)"
        self.c = {(k, b): 0.0
                  for k in self.crystallizable for b in self.deg_psi}
        if coeffs:
            for nm, v in coeffs.items():
                _, sk, sb = nm.split("_")
                self.c[(int(sk), int(sb))] = v
        self.param_names = self.base.param_names + tuple(
            f"cpl_{k}_{b}" for k in self.crystallizable for b in self.deg_psi)

    # -- coupling helpers (h_k and derivatives in psi) --------------------
    def _kpsi(self, psis, k):
        return psis[self.crystallizable.index(k)]

    def _h(self, psi, k):
        u = 2.0 * psi - 1.0
        out = np.zeros_like(psi)
        for b in self.deg_psi:
            pk, _ = _legendre_np(u, b)
            out = out + self.c[(k, b)] * pk
        return out

    def _hp(self, psi, k):                       # dh/dpsi = 2 * sum c L_b'(u)
        u = 2.0 * psi - 1.0
        out = np.zeros_like(psi)
        for b in self.deg_psi:
            _, dpk = _legendre_np(u, b)
            out = out + self.c[(k, b)] * dpk * 2.0
        return out

    def _hpp(self, psi, k):                       # d2h/dpsi2 = 4 * sum c L_b''(u)
        u = 2.0 * psi - 1.0
        out = np.zeros_like(psi)
        for b in self.deg_psi:
            out = out + self.c[(k, b)] * _legendre2_np(u, b) * 4.0
        return out

    # -- CrystalEnergy protocol ------------------------------------------
    def dfdphi(self, phis, psis):
        mu = list(self.base.mu(phis))
        for k in self.crystallizable:
            mu[k] = mu[k] + self._h(self._kpsi(psis, k), k)
        return mu

    def dfdpsi(self, phis, psis):
        return [phis[k] * self._hp(self._kpsi(psis, k), k)
                for k in self.crystallizable]

    def d2fdphidphi(self, phis, psis):
        return self.base.dmu_dphi(phis)          # coupling is linear in phi

    def d2fdphidpsi(self, phis, psis):
        K = len(self.crystallizable)
        H = [[np.zeros_like(phis[0]) for _ in range(K)] for _ in range(self.M)]
        for j, k in enumerate(self.crystallizable):
            H[k][j] = self._hp(self._kpsi(psis, k), k)
        return H

    def d2fdpsidpsi(self, phis, psis):
        K = len(self.crystallizable)
        H = [[np.zeros_like(phis[0]) for _ in range(K)] for _ in range(K)]
        for j, k in enumerate(self.crystallizable):
            H[j][j] = phis[k] * self._hpp(self._kpsi(psis, k), k)
        return H

    def dfdphi_dparam(self, phis, psis, name):
        if name.startswith("cpl_"):
            _, sk, sb = name.split("_")
            k, b = int(sk), int(sb)
            z = [np.zeros_like(phis[0]) for _ in range(self.M)]
            pk, _ = _legendre_np(2.0 * self._kpsi(psis, k) - 1.0, b)
            z[k] = pk
            return z
        return self.base.dmu_dparam(phis, name)   # chi/N/basis_i_k

    def dfdpsi_dparam(self, phis, psis, name):
        K = len(self.crystallizable)
        z = [np.zeros_like(phis[0]) for _ in range(K)]
        if name.startswith("cpl_"):
            _, sk, sb = name.split("_")
            k, b = int(sk), int(sb)
            j = self.crystallizable.index(k)
            _, dpk = _legendre_np(2.0 * self._kpsi(psis, k) - 1.0, b)
            z[j] = phis[k] * dpk * 2.0
        return z

    # -- diagnostic -------------------------------------------------------
    def coupling_gauge_residual(self, k, n_quad=64):
        """<h_k, 1> over u in [-1, 1].  ~0 by construction: h_k is spanned by
        L_{b>=1}, all L2-orthogonal to the constant mode L_0."""
        x, w = np.polynomial.legendre.leggauss(n_quad)
        psi = 0.5 * (x + 1.0)          # u = 2psi-1 = x
        return float((w * self._h(psi, k)).sum())
```

Then export in `src/diffsim/adjoint/__init__.py` (follow the existing pattern — add `NeuralCrystalEnergy` to the import from `.neural_crystal` and to `__all__`). And apply the ①-minor sweep in `crystallization_multi.py`: delete the `_rowslices` method; change `self.K = len(tuple(crystallizable))` to `self.K = len(crystallizable)` in `CrystalCHForward.__init__`.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_neural_crystal.py -q`
Expected: PASS (5 tests). Also `.venv/bin/pytest tests/test_crystallization_multi.py -q` (the minor sweep must not regress — 12 pass).

- [ ] **Step 5: Commit**

```bash
git add src/diffsim/adjoint/neural_crystal.py src/diffsim/adjoint/__init__.py \
        src/diffsim/adjoint/crystallization_multi.py tests/test_neural_crystal.py
git commit -m "feat(neural-crystal): NeuralCrystalEnergy coupled energy + complex-step gate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Exact-match reduction gate vs `AdditiveCrystalEnergy`

**Files:** Test only — `tests/test_neural_crystal.py`.

**Interfaces:**
- Consumes: `NeuralCrystalEnergy` (Task 1), `AdditiveCrystalEnergy` (from `crystallization_multi.py`), the `_q/_p` polynomials (`from diffsim.adjoint.crystallization import _q, _p`).
- Produces: nothing (pure gate).

The additive coupling is `h_k^add(ψ) = q(ψ)dσ_k + p(ψ)drive_k`, `q=ψ²(1−ψ)²` (deg 4), `p=3ψ²−2ψ³` (deg 3), `drive_k = dh_k(T/Tm_k−1)`. Choose `deg_psi=(1,2,3,4)` so the ψ-VARYING part is representable. The constant (b=0) projection of `h_k^add` is the gauge direction the neural form drops, so match `dfdpsi`/Hessians exactly and `dfdphi` up to a per-species constant.

- [ ] **Step 1: Write the gate**

```python
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
```

- [ ] **Step 2: Run to verify** — this should PASS immediately if Task 1 is correct (no new implementation). Run `.venv/bin/pytest tests/test_neural_crystal.py::test_reduction_matches_additive_up_to_gauge -q`. If `dfdpsi` mismatches beyond 1e-11, the bug is in Task 1's `_hp`/`_legendre2` — fix there.

- [ ] **Step 3: Commit**

```bash
git add tests/test_neural_crystal.py
git commit -m "test(neural-crystal): reduction gate == AdditiveCrystalEnergy up to phi gauge

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `CrystalCHTwin` coupling extension (cpl_* leaves)

**Files:**
- Modify: `src/diffsim/adjoint/torch_twin.py` (`CrystalCHTwin` only — additive)
- Test: `tests/test_neural_crystal.py`

**Interfaces:**
- Consumes: `NeuralCrystalEnergy` (Task 1); the existing `CrystalCHTwin(dm, M, crystallizable, dt, order, device).grads(phi0_list, psi0_list, energy_params, engine_params, n_steps, names, target)`.
- Produces: `CrystalCHTwin.grads` recognises `cpl_{k}_{b}` names — building a leaf per coupling coeff, applying the torch Legendre-in-ψ correction `Δdfdφ_k += Σ_b c_{k,b} L_b(2ψ_k−1)` and `Δdfdψ_k += φ_k·2Σ_b c_{k,b} L_b'(2ψ_k−1)` inside the coupled `_assemble`, and returning `{name: leaf.grad}`. The additive coupling `q/p·dσ/drive` path must remain intact (both coexist: the twin evaluates BOTH the additive coupling from `energy_params` AND, when `cpl_*` leaves/an energy with `.c` is supplied, the neural coupling).

**Implementation direction** (research task — derive against the existing code, do not transcribe blindly): `CrystalCHTwin` currently hard-codes the additive coupling via `_q_t/_p_t`. The neural coupling is structurally identical — replace/augment the scalar `coup = qp*dsig + pp*drive` with the general `h_k'`, and the potential term `q*dsig+p*drive` with `h_k`. Mirror how `MultiCHTwin.grads` handles `basis_{i}_{k}` leaves (torch `_legendre` correction to μ, always applied when a `basis_energy` is provided, only requested names get grads) — the `cpl_*` path is the exact analogue for the coupling. Add torch `_legendre_t(u,b)` and `_legendre_t_d(u,b)` helpers (value + first derivative) mirroring `neural_multiphase._legendre_np`, complex-not-needed (torch autograd). Pass the neural energy (or its `deg_psi`/`c`) into `grads`/`march` via a new optional arg (e.g. `neural_energy=None`), following the `basis_energy`/`mob_closure` precedent in `MultiCHTwin.grads`.

- [ ] **Step 1: Write the failing test**

```python
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
```

(The test needs a `CrystalCHTwin.loss_only(...)` detached-forward helper returning the scalar φ+ψ loss without building autograd leaves — add it alongside `grads` if not present; it is a thin wrapper over `march` + the loss formula with `torch.no_grad()`.)

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/pytest tests/test_neural_crystal.py::test_twin_cpl_grads_vs_fd -q`
Expected: FAIL — `grads` does not yet recognise `cpl_*` / `neural_energy` (KeyError or unknown-name), or `loss_only` missing.

- [ ] **Step 3: Implement** the `cpl_*` leaf handling + neural coupling in `CrystalCHTwin.grads`/`march`/`_assemble` and the `loss_only` helper, per the Implementation direction above. Keep `CACHTwin`/`MultiCHTwin` and the additive `CrystalCHTwin` path unchanged (additive-only).

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_neural_crystal.py::test_twin_cpl_grads_vs_fd -q` → PASS.
Then `.venv/bin/pytest tests/test_crystallization_multi.py tests/test_multiphase_adjoint.py -q` → no regression (12 + 23/1s).

- [ ] **Step 5: Commit**

```bash
git add src/diffsim/adjoint/torch_twin.py tests/test_neural_crystal.py
git commit -m "feat(neural-crystal): CrystalCHTwin cpl_* coupling grads (torch Legendre-in-psi)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Three-way gate (cpl_* + basis_* + chi/N + engine)

**Files:** Test only — `tests/test_neural_crystal.py`.

**Interfaces:**
- Consumes: `NeuralCrystalEnergy` (Task 1), `CrystalCHForward`/`CrystalCHAdjoint` (from `crystallization_multi.py`), `CrystalCHTwin` w/ neural coupling (Task 3). Mirror `_three_way_crystal`/`_check_three_way` from `tests/test_crystallization_multi.py`.
- Produces: nothing (pure gate).

- [ ] **Step 1: Write the gate** — `_three_way_neural_crystal(dm, coords, M, crystallizable, deg_psi, order, n_steps)` mirroring `tests/test_crystallization_multi.py::_three_way_crystal`, but the forward energy is a `NeuralCrystalEnergy` (nonzero `cpl_*` and a nonzero `basis_0_2`), and:
  - `names = ["cpl_0_1","cpl_0_2","basis_0_2","chi_0_1","N_0","onsager_0_0","kappa_0","eps2_0","L_0"]`
  - hand adjoint: `CrystalCHAdjoint(fwd).gradient(dJdx, names)` with dJdx from `loss = 0.5Σ‖φ_i,N−tgt‖² + 0.5Σ‖ψ_k,N−tgt‖²` (dJdx[-1][2i::blk]=φ_i−tgt, [2M+j::blk]=ψ_j−tgt)
  - twin: `CrystalCHTwin(...).grads(..., names, tgt, neural_energy=fwd_energy)`
  - FD central on each name (perturb the coupling coeff via `energy.c`, the basis coeff via `energy.base.gamma`, chi/N via the energy, engine params directly)
  - `_check_three_way`: adj/twin < 1e-10, adj/fd < 1e-6.

  Three test fns: `test_three_way_neural_ternary_bdf1` (M=2,K=(0,),deg=(1,2)), `test_three_way_neural_ternary_bdf2` (M=2,K=(0,),deg=(1,2),order=2), `test_three_way_neural_quaternary_bdf1` (M=3,K=(0,2),deg=(1,2)). Reuse the module `_dm` helper (add one if `tests/test_neural_crystal.py` lacks it — copy from `test_crystallization_multi.py`).

- [ ] **Step 2-4: Run each.** `.venv/bin/pytest tests/test_neural_crystal.py -q`. Expected: all pass at adj/twin < 1e-10, adj/fd < 1e-6. BDF2 and quaternary are the stress cases — if a leg misses tolerance there, investigate a real bug (don't loosen). Confirm no regression across `tests/test_crystallization_multi.py tests/test_multiphase_adjoint.py`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_neural_crystal.py
git commit -m "test(neural-crystal): three-way gate (cpl/basis/chi/N/engine, ternary+quaternary, bdf1+2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Synthetic recovery driver + example

**Files:**
- Create: `examples/crystal_learn_from_synthetic.py`
- Test: `tests/test_neural_crystal.py`

**Interfaces:**
- Consumes: `NeuralCrystalEnergy`, `CrystalCHForward`, `CrystalCHAdjoint`.
- Produces: `recover_coupling(dm, coords, planted_coeffs, names, n_steps, order, n_iter, lr)` → `(loss_hist, theta_hat, theta_true)`; assert loss-drop and identifiable-direction recovery.

- [ ] **Step 1: Write the failing test**

```python
def test_synthetic_recovery_drops_loss():
    # 1. plant a NeuralCrystalEnergy with known cpl_* + basis, generate phi+psi traj
    # 2. from a zeroed guess, gradient-descend cpl_* via the hand adjoint to match
    #    the planted final phi+psi fields (loss = 0.5||phi_N-phi*||^2+0.5||psi_N-psi*||^2)
    # 3. assert loss drops >= 20x and recovered cpl_0_1 within 25% of planted
    dm, coords = _dm_neural(2)
    res = recover_coupling(dm, coords, planted={"cpl_0_1": 0.15, "cpl_0_2": -0.1},
                           names=["cpl_0_1", "cpl_0_2"], n_steps=4, order=1,
                           n_iter=40, lr=0.5)
    loss_hist, th_hat, th_true = res
    assert loss_hist[-1] <= loss_hist[0] / 20.0
    assert abs(th_hat["cpl_0_1"] - th_true["cpl_0_1"]) \
        <= 0.25 * abs(th_true["cpl_0_1"])
```

(`_dm_neural` = the module's `_dm`. `recover_coupling` lives in `examples/crystal_learn_from_synthetic.py` and is imported by the test; keep it importable — guard any CLI under `if __name__ == "__main__":`.)

- [ ] **Step 2: Run to verify fail** — `ImportError`/`NameError` for `recover_coupling`.

- [ ] **Step 3: Implement** `examples/crystal_learn_from_synthetic.py`. `recover_coupling`:
  1. Build a planted `NeuralCrystalEnergy` (planted coeffs + a small basis correction), run `CrystalCHForward` `n_steps`, capture the final `phis`/`psis` as targets `phi*`, `psi*`.
  2. Initialise a guess energy with the coupling coeffs zeroed. Loop `n_iter`: run forward at current coeffs, form `dJdx` from the field mismatch at the final step (φ-rows and ψ-rows), get `grads = CrystalCHAdjoint(fwd).gradient(dJdx, names)`, gradient-descend `energy.c[(k,b)] -= lr * grads[name]`, record loss.
  3. Re-evaluate loss at the final coeffs (avoid the M6 off-by-one: the recorded loss_hist and returned params must correspond). Return `(loss_hist, {name: recovered}, {name: planted})`.
  Note weak-identifiability of higher ψ-degrees from a single final snapshot in a docstring (motivates ③ multi-snapshot), mirroring M6 Plan A Task 5.

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest tests/test_neural_crystal.py::test_synthetic_recovery_drops_loss -q` → PASS. Full-file + regression check.

- [ ] **Step 5: Commit**

```bash
git add examples/crystal_learn_from_synthetic.py tests/test_neural_crystal.py
git commit -m "feat(neural-crystal): synthetic recovery of coupled energy via hand adjoint

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** `NeuralCrystalEnergy` object + complex-step gate → Task 1. Exact-match reduction gate → Task 2. Twin coupling extension → Task 3. Three-way gate (cpl/basis/chi/N/engine, ternary+quaternary, BDF1+2) → Task 4. Synthetic recovery + example → Task 5. Functional form (linear-in-φ, Legendre-in-ψ, b=0 dropped), gauge analysis (`coupling_gauge_residual`), the `L_b''` helper (`_legendre2_np`), and the "match up to gauge constant" reduction are all covered. Deferrals (full 2-D g_k, χ(ψ), MD, θ, GPU) correctly absent.

**Placeholder scan:** Task 1 carries full energy-object code + tests. Tasks 2, 4 are pure test gates with full test code. Tasks 3, 5 are research tasks (twin torch coupling; recovery loop) with full test code + concrete implementation direction anchored to named precedents (`MultiCHTwin.grads` basis_* handling; `_three_way_crystal`; M6 Plan A Task 5 recovery) — the numeric derivation is best done against those files, consistent with how the ① plan handled the twin/gate.

**Type consistency:** coupling coeff key `c[(k,b)]` and name `cpl_{k}_{b}` consistent across energy/twin/gate/recovery. `deg_psi` (b≥1) consistent. `_hp = 2·Σc·L'`, `_hpp = 4·Σc·L''` chain factors consistent (û=2ψ−1). Protocol method names match the `CrystalEnergy` contract used by `CrystalCHDiscrete`. Engine untouched — same `CrystalCHForward`/`Adjoint`/`Twin` signatures as ①. `names` list identical between Task 4 gate and the recovery driver's supported set.
