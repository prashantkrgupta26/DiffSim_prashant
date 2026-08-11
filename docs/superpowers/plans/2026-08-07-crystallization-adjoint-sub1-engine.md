# ψ-Crystallization Adjoint Engine (sub-project ①) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the M-generic coupled multi-CH × multi-Allen-Cahn (φ+ψ, no θ) IFT adjoint — three-way verified gradients w.r.t. the crystallization params + χ/N/mobility/κ — with the bulk energy behind a `CrystalEnergy` protocol so a non-parametric learnable energy drops in later.

**Architecture:** Generalize the binary `adjoint/crystallization.py` (node block φ,μ,ψ) to the M-generic `2M+K` block, and the K=0 `adjoint/multiphase.py` engine to add the ψ Allen-Cahn block. **All** crystallization physics routes through a `CrystalEnergy` object (unlike the binary, which hard-codes q/p/dσ/drive in `assemble`) — so the additive parametric form is one implementation and the non-parametric `NeuralCrystalEnergy` (sub-project ②) is a drop-in. Reuses `FHMultiEnergy` (φ part) and `MobilityClosure` (mobility).

**Tech Stack:** numpy/scipy (hand engine), torch (autograd twin), pytest. Pure CPU. Reuses `src/diffsim/adjoint/{multiphase.py, crystallization.py, torch_twin.py, neural_multiphase.py}`.

## Global Constraints

- **Block layout:** node-major, per node `[(φ_0,μ_0)…(φ_{M-1},μ_{M-1}), ψ_{k}…]` for k in the crystallizable subset K ⊆ {0..M-1}. `ndof_per_node = 2M + |K|`. φ_i/μ_i at `2i`/`2i+1`; ψ for the j-th crystallizable species at `2M+j`.
- **Dynamics (spec §Architecture):** `R_φi = ∫N(σφ_i−hist) + Σ_j M_ij∫∇N·∇μ_j`; `R_μi = ∫N(μ_i−∂f/∂φ_i) − κ_i∫∇N·∇φ_i`; `R_ψk = ∫N(σψ_k−hist_ψk) + L_k[∫N ∂f/∂ψ_k + ε_k²∫∇N·∇ψ_k]`. ψ is a single dof (2nd-order AC, no μ-splitting).
- **Energy behind the `CrystalEnergy` protocol:** `dfdphi(phis,psis)` (M list, → μ), `dfdpsi(phis,psis)` (K list, → AC driving force), `d2fdphidphi`/`d2fdphidpsi`/`d2fdpsidpsi` (Hessian blocks), `dfdphi_dparam`/`dfdpsi_dparam(name)`, `param_names`, `crystallizable` (the K tuple). Methods PRESERVE INPUT DTYPE (complex-step).
- **Parameter split:** bulk-energy params (χ, N, dσ_k, dh_k, Tm_k) route THROUGH the energy object; engine-level params (mobility via `MobilityClosure`, κ_i, ε²_k, L_k) are engine params. `T` (temperature) is external/fixed in ①.
- **Additive form (first `CrystalEnergy` impl):** `f = f_FH(φ;χ) + Σ_k φ_k[q(ψ_k)dσ_k + p(ψ_k)drive_k]`, `q(ψ)=ψ²(1−ψ)²`, `p(ψ)=ψ²(3−2ψ)`, `drive_k = dh_k(T/Tm_k − 1)`. Use the `_q/_qp/_qpp/_p/_pp/_ppp` helpers already in `crystallization.py` (import them).
- **Verification (house rule):** three-way gate hand adjoint == torch twin == FD, per parameter, ternary + one quaternary with a crystallizable subset, BDF1+BDF2. Tolerances adj/twin < 1e-10, adj/fd < 1e-6. Analytic energy/engine derivatives additionally complex-step gated (1e-30 imag). Plus a **K=∅ reduction gate**: with no crystallizing species, `CrystalCHForward` reproduces `MultiCHForward` (K=0) to machine precision.
- **Run tests with** `.venv/bin/pytest` (no bare `python`; no `timeout` on macOS).
- **Branch:** `feat/crystallization-adjoint` (spec committed there). Commit-per-green.

---

## File Structure

- Create `src/diffsim/adjoint/crystallization_multi.py` — `CrystalEnergy` protocol, `AdditiveCrystalEnergy`, `CrystalCHDiscrete`, `CrystalCHForward`, `CrystalCHAdjoint`.
- Modify `src/diffsim/adjoint/torch_twin.py` — add `CrystalCHTwin` (M-generalize `CACHTwin`; consume a `CrystalEnergy` + `MobilityClosure`).
- Modify `src/diffsim/adjoint/__init__.py` — export `CrystalEnergy`, `AdditiveCrystalEnergy`, `CrystalCHForward`, `CrystalCHAdjoint`.
- Create `tests/test_crystallization_multi.py` — energy complex-step, three-way gate, K=∅ reduction.

---

## Task 1: `CrystalEnergy` protocol + `AdditiveCrystalEnergy`

**Files:**
- Create: `src/diffsim/adjoint/crystallization_multi.py`
- Modify: `src/diffsim/adjoint/__init__.py`
- Test: `tests/test_crystallization_multi.py`

**Interfaces:**
- Consumes: `FHMultiEnergy` (from `multiphase.py`) for the φ part; `_q,_qp,_qpp,_p,_pp,_ppp` (from `crystallization.py`).
- Produces: `AdditiveCrystalEnergy(chi, N, crystallizable, dsig, dh, Tm, T=0.5, breg=0.0)` where `crystallizable` is a tuple of species indices, and `dsig`/`dh`/`Tm` are dicts or arrays keyed by crystallizable index. Implements the `CrystalEnergy` contract (Global Constraints). `param_names` = FH `chi_*`/`N_*` + `dsig_k`/`dh_k`/`Tm_k` for k in crystallizable.

- [ ] **Step 1: Write the failing test** (reduces to FH when no crystal coupling; ∂f/∂ψ correct; complex-step)

```python
import numpy as np, pytest
from diffsim.adjoint.crystallization_multi import AdditiveCrystalEnergy
from diffsim.adjoint import FHMultiEnergy

def _chiN(M=2):
    chi = np.zeros((M+1,M+1)); chi[0,1]=chi[1,0]=2.5; chi[0,2]=chi[2,0]=1.0; chi[1,2]=chi[2,1]=0.8
    return chi, np.ones(M+1)

def test_crystal_energy_dfdphi_matches_fh_plus_coupling():
    chi,N=_chiN(); K=(0,)
    en = AdditiveCrystalEnergy(chi,N,crystallizable=K,dsig={0:1.2},dh={0:-1.0},Tm={0:1.0},T=0.5)
    fh = FHMultiEnergy(chi,N)
    phis=[np.full(4,0.30),np.full(4,0.32)]; psis=[np.full(4,0.4)]
    mu = en.dfdphi(phis,psis)
    # species 0 crystallizes: dfdphi_0 = mu_FH_0 + q(psi)dsig + p(psi)drive
    q=0.4**2*(1-0.4)**2; p=0.4**2*(3-2*0.4); drive=-1.0*(0.5/1.0-1.0)
    assert np.allclose(mu[0], fh.mu(phis)[0] + q*1.2 + p*drive)
    assert np.allclose(mu[1], fh.mu(phis)[1])            # species 1 no crystal term

def test_crystal_energy_derivs_complex_step():
    chi,N=_chiN(); K=(0,1)
    en = AdditiveCrystalEnergy(chi,N,crystallizable=K,dsig={0:1.2,1:0.8},dh={0:-1.0,1:-1.3},Tm={0:1.0,1:1.1},T=0.5)
    phis=[np.array([0.30,0.28]),np.array([0.32,0.34])]; psis=[np.array([0.4,0.5]),np.array([0.3,0.45])]
    h=1e-30
    # d(dfdpsi_0)/dpsi_0 vs d2fdpsidpsi[0][0]
    pc=[p.astype(complex) for p in psis]; pc[0]=pc[0]+1j*h
    cs=(en.dfdpsi(phis,pc)[0].imag)/h
    an=en.d2fdpsidpsi(phis,psis)[0][0]
    assert np.allclose(cs, an, atol=1e-9)
    # d(dfdphi_0)/d(dsig_0) via complex-step on the param vs dfdphi_dparam
    en.dsig[0]=en.dsig[0]+1j*h
    cs2=(en.dfdphi(phis,psis)[0].imag)/h
    en.dsig[0]=en.dsig[0].real
    assert np.allclose(cs2, en.dfdphi_dparam(phis,psis,"dsig_0")[0], atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_crystallization_multi.py -k crystal_energy -v`
Expected: FAIL — `ModuleNotFoundError`/`AttributeError`.

- [ ] **Step 3: Write minimal implementation**

Create `src/diffsim/adjoint/crystallization_multi.py`:

```python
import numpy as np
from .multiphase import FHMultiEnergy
from .crystallization import _q, _qp, _qpp, _p, _pp, _ppp


class CrystalEnergy:
    """Protocol: coupled bulk free energy f(phi, psi).  Exposes the M exchange
    potentials df/dphi_i, the K Allen-Cahn driving forces df/dpsi_k, the Hessian
    blocks, and per-parameter cotangents.  Implementations: AdditiveCrystalEnergy
    (parametric); NeuralCrystalEnergy (sub-project 2, drop-in)."""
    M = 0; crystallizable = (); param_names = ()
    def dfdphi(self, phis, psis): raise NotImplementedError
    def dfdpsi(self, phis, psis): raise NotImplementedError
    def d2fdphidphi(self, phis, psis): raise NotImplementedError
    def d2fdphidpsi(self, phis, psis): raise NotImplementedError
    def d2fdpsidpsi(self, phis, psis): raise NotImplementedError
    def dfdphi_dparam(self, phis, psis, name): raise NotImplementedError
    def dfdpsi_dparam(self, phis, psis, name): raise NotImplementedError


class AdditiveCrystalEnergy(CrystalEnergy):
    r"""f = f_FH(phi;chi) + sum_k phi_k[q(psi_k)dsig_k + p(psi_k)drive_k],
    drive_k = dh_k(T/Tm_k - 1).  chi/N via FHMultiEnergy; dsig/dh/Tm are the
    learnable crystal-bulk params.  K = crystallizable species subset."""

    def __init__(self, chi, N, crystallizable, dsig, dh, Tm, T=0.5, breg=0.0):
        self.fh = FHMultiEnergy(chi, N, breg=breg)
        self.M = self.fh.M
        self.crystallizable = tuple(int(k) for k in crystallizable)
        self.dsig = {int(k): dsig[k] for k in self.crystallizable}
        self.dh = {int(k): dh[k] for k in self.crystallizable}
        self.Tm = {int(k): Tm[k] for k in self.crystallizable}
        self.T = float(T)
        cp = tuple(f"{b}_{k}" for k in self.crystallizable
                   for b in ("dsig", "dh", "Tm"))
        self.param_names = self.fh.param_names + cp

    def _drive(self, k):
        return self.dh[k] * (self.T / self.Tm[k] - 1.0)

    def _kpsi(self, psis, k):          # psi array for crystallizable species k
        return psis[self.crystallizable.index(k)]

    def dfdphi(self, phis, psis):
        mu = list(self.fh.mu(phis))
        for k in self.crystallizable:
            ps = self._kpsi(psis, k)
            mu[k] = mu[k] + _q(ps) * self.dsig[k] + _p(ps) * self._drive(k)
        return mu

    def dfdpsi(self, phis, psis):
        out = []
        for k in self.crystallizable:
            ps = self._kpsi(psis, k)
            out.append(phis[k] * (_qp(ps) * self.dsig[k] + _pp(ps) * self._drive(k)))
        return out

    def d2fdphidphi(self, phis, psis):
        return self.fh.dmu_dphi(phis)            # chi psi-independent in v1

    def d2fdphidpsi(self, phis, psis):
        # [M][K]; only the diagonal crystallizable coupling is nonzero:
        # d(dfdphi_k)/dpsi_k = q'(psi_k)dsig_k + p'(psi_k)drive_k
        K = len(self.crystallizable)
        H = [[np.zeros_like(phis[0]) for _ in range(K)] for _ in range(self.M)]
        for j, k in enumerate(self.crystallizable):
            ps = self._kpsi(psis, k)
            H[k][j] = _qp(ps) * self.dsig[k] + _pp(ps) * self._drive(k)
        return H

    def d2fdpsidpsi(self, phis, psis):
        K = len(self.crystallizable)
        H = [[np.zeros_like(phis[0]) for _ in range(K)] for _ in range(K)]
        for j, k in enumerate(self.crystallizable):
            ps = self._kpsi(psis, k)
            H[j][j] = phis[k] * (_qpp(ps) * self.dsig[k] + _ppp(ps) * self._drive(k))
        return H

    def dfdphi_dparam(self, phis, psis, name):
        z = [np.zeros_like(phis[0]) for _ in range(self.M)]
        if name.startswith(("dsig_", "dh_", "Tm_")):
            b, k = name.rsplit("_", 1); k = int(k); ps = self._kpsi(psis, k)
            if b == "dsig": z[k] = _q(ps)
            elif b == "dh": z[k] = _p(ps) * (self.T / self.Tm[k] - 1.0)
            else:  # Tm
                z[k] = _p(ps) * (-self.dh[k] * self.T / self.Tm[k] ** 2)
            return z
        return self.fh.dmu_dparam(phis, name)     # chi/N

    def dfdpsi_dparam(self, phis, psis, name):
        K = len(self.crystallizable)
        z = [np.zeros_like(phis[0]) for _ in range(K)]
        if name.startswith(("dsig_", "dh_", "Tm_")):
            b, k = name.rsplit("_", 1); k = int(k); j = self.crystallizable.index(k)
            ps = self._kpsi(psis, k)
            if b == "dsig": z[j] = phis[k] * _qp(ps)
            elif b == "dh": z[j] = phis[k] * _pp(ps) * (self.T / self.Tm[k] - 1.0)
            else: z[j] = phis[k] * _pp(ps) * (-self.dh[k] * self.T / self.Tm[k] ** 2)
        return z
```

Export `CrystalEnergy`, `AdditiveCrystalEnergy` in `__init__.py`.

> **Complex-dtype note:** `_q/_p` etc. are pure polynomials in ψ (dtype-preserving); `FHMultiEnergy` already preserves dtype; the `dsig/dh/Tm` dict values are plain scalars multiplied against arrays, so complex-step on a param flows through. No casts.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_crystallization_multi.py -k crystal_energy -v`
Expected: PASS (FH+coupling parity; complex-step agreement ~1e-15).

- [ ] **Step 5: Commit**

```bash
git add src/diffsim/adjoint/crystallization_multi.py src/diffsim/adjoint/__init__.py tests/test_crystallization_multi.py
git commit -m "feat(crystallization): CrystalEnergy protocol + AdditiveCrystalEnergy (coupled phi-psi bulk)"
```

---

## Task 2: `CrystalCHDiscrete` — the 2M+K block residual/Jacobian

**Files:**
- Modify: `src/diffsim/adjoint/crystallization_multi.py`
- Test: `tests/test_crystallization_multi.py`

**Interfaces:**
- Consumes: `CrystalEnergy` (Task 1), `MobilityClosure` (from `neural_multiphase.py`), the mesh-bin machinery pattern from `MultiCHDiscrete` (`adjoint/multiphase.py`) and `CACHDiscrete` (`adjoint/crystallization.py`).
- Produces: `CrystalCHDiscrete(dm, M, crystallizable)` with `blk = 2*M + len(crystallizable)`, `assemble(phis, mus, psis, hist_phi_gp, hist_psi_gp, params, want_jac=True) -> (R, J)`, `dR_dparam(phis, mus, psis, params, name)`, `mass_matrix()`, `stiffness_matrix()`. `params` carries `mobility` (closure), `kappa` (list), `eps2` (dict by k), `L` (dict by k), `energy` (CrystalEnergy), `sigma`.

- [ ] **Step 1: Write the failing test** (block sizing + complex-step Jacobian, incl. the φ–ψ cross-blocks)

```python
def test_crystal_discrete_block_and_jacobian_complex_step():
    # M=2 ternary, K=(0,); build dm via the shared helper; assemble R,J at a
    # random state; verify the full Jacobian vs complex-step (this catches the
    # Rmu/dpsi, Rpsi/dphi, Rpsi/dpsi cross-blocks). Mirror
    # test_discrete_jacobian_complex_step in tests/test_multiphase_adjoint.py,
    # extended with psi rows. ndof == (2*M+len(K)) * n_nodes.
    ...
```

- [ ] **Step 2: Run to verify it fails** — `CrystalCHDiscrete` undefined.

- [ ] **Step 3: Implement.** `CrystalCHDiscrete.assemble` generalizes `CACHDiscrete.assemble` (crystallization.py:133-201) to the M-generic block, but routes ALL bulk terms through the energy object:
  - φ_i rows (`2i`): `R_φi = ∫N(σφ_i − hist) + Σ_j Lam_ij ∫∇N·∇μ_j`; Jacobian `Ae[2i,2i]+=σNN`, `Ae[2i,2j+1]+=Lam_ij·LL` (+ the φ-dependent-mobility term if the closure is non-const, per `MobilityClosure`).
  - μ_i rows (`2i+1`): `R_μi = ∫N(μ_i − dfdphi_i) − κ_i∫∇N·∇φ_i`; Jacobian `Ae[2i+1,2j]+=−WM(d2fdphidphi[i][j])`, `Ae[2i+1,2i]−=κ_i·LL`, `Ae[2i+1,2i+1]+=NN`, and the **cross** `Ae[2i+1, 2M+jψ]+=−WM(d2fdphidpsi[i][jψ])` for each crystallizable.
  - ψ_k rows (`2M+jψ`): `R_ψk = ∫N(σψ_k − hist_ψk) + L_k[∫N·dfdpsi_k + ε²_k∫∇N·∇ψ_k]`; Jacobian `Ae[2M+jψ, 2k]+=L_k·WM(d2fdphidpsi[k][jψ])` (Rpsi/dphi), `Ae[2M+jψ, 2M+jψ]+=σNN + L_k(WM(d2fdpsidpsi[jψ][jψ]) + ε²_k·LL)`, and cross `Ae[2M+jψ, 2M+lψ]+=L_k·WM(d2fdpsidpsi[jψ][lψ])`.
  - `dR_dparam`: `eps2_k`/`L_k` engine params (mirror `CACHDiscrete.dR_dparam` `eps2`/`L`), `mob_*`/`kappa_i` (mirror K=0), bulk (chi/N/dsig/dh/Tm) via `energy.dfdphi_dparam`/`dfdpsi_dparam` scattered onto the μ-rows and ψ-rows respectively. Promote `R` dtype for complex-step (as `MultiCHDiscrete.assemble` does).

- [ ] **Step 4: Run to verify passes** — block sizing + full-Jacobian complex-step agreement ~1e-9.

- [ ] **Step 5: Commit** `feat(crystallization): CrystalCHDiscrete 2M+K block (phi-psi coupled Jacobian), complex-step gated`

---

## Task 3: `CrystalCHForward` + `CrystalCHAdjoint`

**Files:** Modify `src/diffsim/adjoint/crystallization_multi.py`, `src/diffsim/adjoint/__init__.py`; test `tests/test_crystallization_multi.py`.

**Interfaces:**
- Produces: `CrystalCHForward(dm, energy, crystallizable, mobility, kappa, eps2, L, dt, order, ...)` with `set_initial(phi0_list, psi0_list)`, `run(n_steps)`, recording per-step `phis`/`mus`/`psis`/`x`/`sigma`/`ch`/`dt`/`params`. `CrystalCHAdjoint(fwd).gradient(dJdx_list, param_names)`.

- [ ] **Step 1: Write the failing tests** (a) **K=∅ reduction**: `CrystalCHForward` with `crystallizable=()` reproduces `MultiCHForward` (K=0) to 1e-14 on a ternary run; (b) adjoint interface returns finite grads for the crystallization names.

- [ ] **Step 2: Run to verify fail.**

- [ ] **Step 3: Implement.** `CrystalCHForward` mirrors `CACHForward` (crystallization.py:260+) generalized to lists of φ/ψ + the closure; the BDF `_bdf`, `_hist_gp` (separately for φ and ψ history), and the Newton loop follow `MultiCHForward`/`CACHForward`. `CrystalCHAdjoint.gradient` mirrors `MultiCHAdjoint.gradient` (reverse sweep, `solve_T` via the backend) with the history cotangent coupling BOTH the conserved φ-rows (mass term, as K=0) AND the non-conserved ψ-rows (mass term on `2M+jψ`). Reuse the `LinearBackend` default (ScipyBackend).

- [ ] **Step 4: Run to verify pass** — K=∅ reduction ~1e-14; adjoint returns finite grads.

- [ ] **Step 5: Commit** `feat(crystallization): CrystalCHForward + CrystalCHAdjoint (psi Allen-Cahn history coupling)`

---

## Task 4: `CrystalCHTwin`

**Files:** Modify `src/diffsim/adjoint/torch_twin.py`; test `tests/test_crystallization_multi.py`.

**Interfaces:**
- Produces: `CrystalCHTwin(dm, M, crystallizable, dt, order, device).grads(phi0_list, psi0_list, energy_params, engine_params, n_steps, names, target)` returning `{name: leaf.grad}` for the crystallization params + χ/N/mobility/κ.

- [ ] **Step 1: Write the failing test** — twin grads for `["dsig_0","dh_0","L_0","eps2_0","chi_0_1","kappa_0"]` vs central FD (ternary, K=(0,), bdf1), rel < 1e-6.

- [ ] **Step 2: Run to verify fail.**

- [ ] **Step 3: Implement.** M-generalize `CACHTwin` (`torch_twin.py`): node block `2M+K`; the differentiable energy correction consumes a torch mirror of `AdditiveCrystalEnergy` (dfdphi/dfdpsi via the torch `_q_t/_p_t` helpers already in `CACHTwin`), the mobility via the closure, and the ψ Allen-Cahn residual/Jacobian per the spec. `grads()` builds leaves for the requested names and backprops `loss = 0.5 Σ_i‖φ_i,N−tgt‖² + 0.5 Σ_k‖ψ_k,N−tgt‖²` (or the φ-only objective — match the hand-adjoint dJdx).

- [ ] **Step 4: Run to verify pass** — twin==FD < 1e-6.

- [ ] **Step 5: Commit** `feat(crystallization): CrystalCHTwin (M-generic CACHTwin, generic CrystalEnergy + closure)`

---

## Task 5: three-way gate + reduction gate

**Files:** Modify `tests/test_crystallization_multi.py`.

- [ ] **Step 1: Write the gate.** `_three_way_crystal(dm, coords, M, crystallizable, order, n_steps)` mirroring `_three_way_multi` (tests/test_multiphase_adjoint.py) but with a `CrystalCHForward`/`AdditiveCrystalEnergy` forward and `names = ["dsig_0","dh_0","Tm_0","eps2_0","L_0","chi_0_1","N_0","mob_m0","kappa_0"]`. Hand adjoint via `CrystalCHAdjoint.gradient`, twin via `CrystalCHTwin.grads`, FD central. Three test fns: ternary K=(0,) bdf1, ternary K=(0,) bdf2, quaternary K=(0,2) bdf1 — each `_check_three_way`. Plus assert the K=∅ reduction gate from Task 3 (promote it to a hard test if it was a smoke test).

- [ ] **Step 2-4:** Run each; adj/twin < 1e-10, adj/fd < 1e-6. Run the full file + `tests/test_multiphase_adjoint.py` (no regression).

- [ ] **Step 5: Commit** `test(crystallization): three-way gate (crystal params + chi/N/mob/kappa) + K=empty reduction`

---

## Self-Review

**Spec coverage:** `CrystalEnergy` protocol + `AdditiveCrystalEnergy` → Task 1. `CrystalCHDiscrete` (2M+K block) → Task 2. `CrystalCHForward`/`CrystalCHAdjoint` (ψ history coupling) → Task 3. `CrystalCHTwin` → Task 4. Three-way gate + K=∅ reduction → Task 5. Bulk/engine param split honored (dσ/dh/Tm via energy; ε²/L/κ/mobility engine-level). θ deferred (no θ dof anywhere). ② `NeuralCrystalEnergy` / ①b χ(ψ) / ③ MD are out of scope (spec §Designed-for), correctly absent.

**Placeholder scan:** Tasks 2–5 give interfaces, the exact block-index map, per-block Jacobian formulas, and test intent but abbreviate some step code (marked `...` / prose), anchored to named precedents (`crystallization.py` `CACHDiscrete/Forward/Adjoint`, `CACHTwin`, K=0 `MultiCHDiscrete/Forward/Adjoint`, `_three_way_multi`). Task 1 (the crux — the coupled energy object) carries full code. The research math (2M+K assembly, twin) is best derived against those files.

**Type consistency:** block map `φ_i@2i, μ_i@2i+1, ψ_k@2M+j` consistent across assemble/adjoint/twin/gate. `CrystalEnergy` methods (`dfdphi`/`dfdpsi`/`d2f*`/`*_dparam`) match between the protocol, `AdditiveCrystalEnergy`, and `CrystalCHDiscrete`'s consumption. Param names (`dsig_k`/`dh_k`/`Tm_k` bulk; `eps2_k`/`L_k`/`mob_*`/`kappa_i` engine) consistent across energy/engine/twin/gate. `crystallizable` is a tuple of species indices throughout.
```
