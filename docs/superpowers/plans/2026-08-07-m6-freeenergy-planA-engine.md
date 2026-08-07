# M6 Free-Energy Learning — Plan A (engine capability) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the verified K=0 multi-CH adjoint a learnable bulk free energy and a named learnable mobility closure, three-way verified, and prove end-to-end learning via a synthetic-recovery gate — all inside DiffSim, no MD data.

**Architecture:** A new `MultiEnergy` implementation with a **gauge-anchored polynomial correction** to Flory-Huggins (the M-component analog of `neural_energy.BasisCorrEnergy` — exactly orthogonal to the `{1, φᵢ}` gauge modes, analytic `mu`/Hessian). A **named mobility closure** `M(φ;a)` threaded as an engine-level parameter with analytic `∂R/∂a`. `MultiCHTwin` extended to consume a generic energy + closure + an optional learnable time-scale τ. The existing `MultiCHForward`/`MultiCHAdjoint` are unchanged — gradients w.r.t. {θ, a, τ} flow through them via the `MultiEnergy`/closure `dR_dparam` path. Verified by the three-way gate and a synthetic-recovery test (recover a known energy+closure from a self-generated trajectory).

**Tech Stack:** numpy/scipy (hand engine), torch (autograd twin), pytest. Pure CPU. Reuses `src/diffsim/adjoint/{multiphase.py, torch_twin.py, neural_energy.py}`.

## Plan-level decisions (deviating from spec §Components — flag at handoff)

1. **v1 correction = gauge-anchored POLYNOMIAL BASIS** (`BasisMultiEnergy`), not the MLP head. Rationale: exact simplex gauge (no differentiable Gram-projection), analytic `mu`/`dmu_dphi` (no autograd-through-Hessian inside Newton), and its coefficient Gramian is the identifiability number. The MLP head (`NeuralMultiEnergy`) is a follow-on with the identical `MultiEnergy` contract (spec §Designed-for). Both were named in the spec; this picks the basis first.
2. **Scope split:** this Plan A is the engine capability + synthetic recovery (field-L2 loss, no MD). The MD ingest + hybrid CG/descriptor loss + Gramian pre-check + real-MD demo are **Plan B**.

## Global Constraints

- **Reuse the verified engine unchanged:** `MultiCHForward`/`MultiCHAdjoint` (K=0, node-major 2M block) are NOT modified except to route the mobility through a closure object (a strict generalization that reduces to the current constant-Onsager behavior). The existing three-way gate (`tests/test_multiphase_adjoint.py`) must stay green.
- **`MultiEnergy` protocol (verbatim contract):** an energy exposes `mu(phis)` (length-M list), `dmu_dphi(phis)` (M×M list-of-lists), `dmu_dparam(phis, name)` (length-M list), and `param_names` (tuple). `phis` is a length-M list of arrays; methods broadcast elementwise and **preserve input dtype** (complex-step verification passes complex φ/params — do not cast to float64).
- **Gauge (house rule):** the beyond-FH correction lives INSIDE the energy object and is orthogonal to `{1, φ₀,…,φ_{M-1}}` over the composition domain (the M-generalization of the binary `{1,c}` T0/T1 gauge). For the polynomial basis this is exact-by-construction (basis degrees ≥2 in the gauge-orthogonal complement).
- **Mobility closures stay NAMED:** the closure is registered by name (`const` = current behavior; the v1 learnable closure name is defined in Task 2). No anonymous mobility matrices.
- **Three-way gate (house rule):** hand IFT adjoint == torch twin == finite differences, per parameter, ternary + one quaternary, BDF1+BDF2. Tolerances: adj/twin < 1e-10, adj/fd < 1e-6. Analytic energy/closure derivatives additionally complex-step gated (1e-30 imag perturbation).
- **Run tests with** `.venv/bin/pytest` (no bare `python`; no `timeout` on macOS).
- **Branch:** `feat/m6-freeenergy` (spec already committed there). Commit-per-green.

---

## File Structure

- Create `src/diffsim/adjoint/neural_multiphase.py` — `BasisMultiEnergy(MultiEnergy)` (FH base + gauge-anchored polynomial correction) and the named mobility closure `MobilityClosure` (`const` + the v1 learnable closure). One responsibility: learnable bulk energy + learnable transport, behind the existing protocol boundaries.
- Modify `src/diffsim/adjoint/multiphase.py` — `MultiCHDiscrete.assemble`/`dR_dparam` route the Onsager mobility through a closure object (default `const` = today's behavior); `MultiCHForward.__init__` accepts a `mobility=` closure. Strict generalization.
- Modify `src/diffsim/adjoint/torch_twin.py` — `MultiCHTwin` consumes a generic `MultiEnergy` (mirror `CHTwin`'s `obj_energy` branch) + a mobility closure + an optional τ; `grads()` handles `basis_k`, `mob_j`, `tau` names.
- Modify `src/diffsim/adjoint/__init__.py` — export `BasisMultiEnergy`, `MobilityClosure`.
- Create `tests/test_neural_multiphase.py` — energy-object + closure complex-step gates; three-way gate for `dJ/dθ`, `dJ/da`, `dJ/dτ`.
- Create `tests/test_m6_recovery.py` — synthetic-recovery pipeline gate.

---

## Task 1: `BasisMultiEnergy` — gauge-anchored learnable bulk energy

**Files:**
- Create: `src/diffsim/adjoint/neural_multiphase.py`
- Modify: `src/diffsim/adjoint/__init__.py`
- Test: `tests/test_neural_multiphase.py`

**Interfaces:**
- Consumes: `MultiEnergy`, `FHMultiEnergy` (from `multiphase.py`) — reuse the FH `mu`/`dmu_dphi`/`dmu_dparam` for the base.
- Produces: `BasisMultiEnergy(chi, N, degrees=(2,3), coeffs=None, dom=(0.05,0.95), breg=0.0)`. `param_names` = FH `chi_*`/`N_*` **plus** `basis_{i}_{k}` (species i in 0..M-1, degree k in `degrees`). Implements `mu`, `dmu_dphi`, `dmu_dparam`. The correction to μ_i is `Σ_k γ_{i,k} P_k(û_i)` with `û_i = (φ_i − mid)/half`, `P_k` shifted-Legendre degree k≥2 — orthogonal to `{1, φ_i}` on the domain by construction (per-species, separable v1; cross-species coupling deferred to the MLP head). `dmu_dphi` correction is diagonal: `∂(corr μ_i)/∂φ_i = Σ_k γ_{i,k} P'_k(û_i)/half`, off-diagonal 0.

- [ ] **Step 1: Write the failing test** (mu parity to FH at γ=0, then non-trivial correction + gauge orthogonality)

```python
import numpy as np, pytest
from diffsim.adjoint import BasisMultiEnergy, FHMultiEnergy

def _chiN(M=2):
    chi = np.zeros((M+1, M+1))
    chi[0,1]=chi[1,0]=2.5; chi[0,2]=chi[2,0]=1.0; chi[1,2]=chi[2,1]=0.8
    return chi, np.ones(M+1)

def test_basis_reduces_to_fh_at_zero_coeffs():
    chi, N = _chiN()
    be = BasisMultiEnergy(chi, N, degrees=(2,3))          # coeffs default 0
    fh = FHMultiEnergy(chi, N)
    phis = [np.full(5, 0.30), np.full(5, 0.32)]
    for a, b in zip(be.mu(phis), fh.mu(phis)):
        assert np.allclose(a, b)
    assert any(nm.startswith("basis_") for nm in be.param_names)

def test_basis_correction_is_gauge_orthogonal():
    chi, N = _chiN()
    be = BasisMultiEnergy(chi, N, degrees=(2,3),
                          coeffs={"basis_0_2": 0.7, "basis_0_3": -0.4})
    # <corr_mu_0, 1> and <corr_mu_0, phi_0> over the domain must be ~0
    r0, r1 = be.gauge_residual(species=0)
    assert abs(r0) < 1e-10 and abs(r1) < 1e-10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_neural_multiphase.py -k basis -v`
Expected: FAIL — `ImportError: cannot import name 'BasisMultiEnergy'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/diffsim/adjoint/neural_multiphase.py`. Reuse the shifted-Legendre `_legendre(u,k)` from `neural_energy.py` (import it). Structure (grounded in `neural_energy.BasisCorrEnergy`, generalized per-species):

```python
import numpy as np
from .multiphase import MultiEnergy, FHMultiEnergy
from .neural_energy import _legendre   # shifted-Legendre P_k, P'_k (k>=2 gauge-anchored)


class BasisMultiEnergy(MultiEnergy):
    """FH base + a per-species gauge-anchored polynomial correction to each
    exchange potential mu_i.  Correction basis are shifted-Legendre degrees k>=2
    on the composition domain [lo,hi], L2-orthogonal to {1, phi_i} by
    construction (the M-component T0/T1 gauge; see neural_energy)."""

    def __init__(self, chi, N, degrees=(2, 3), coeffs=None, dom=(0.05, 0.95),
                 breg=0.0):
        self.fh = FHMultiEnergy(chi, N, breg=breg)
        self.M = self.fh.M
        self.degrees = tuple(int(k) for k in degrees)
        self.lo, self.hi = float(dom[0]), float(dom[1])
        self.mid = 0.5 * (self.hi + self.lo)
        self.half = 0.5 * (self.hi - self.lo)
        # gamma[i][k] learnable; complex-safe dtype preserved on use
        self.gamma = {(i, k): 0.0 for i in range(self.M) for k in self.degrees}
        if coeffs:
            for nm, v in coeffs.items():
                _, i, k = nm.split("_"); self.gamma[(int(i), int(k))] = v
        self.param_names = self.fh.param_names + tuple(
            f"basis_{i}_{k}" for i in range(self.M) for k in self.degrees)

    def _corr_mu(self, phis, i):
        u = (phis[i] - self.mid) / self.half
        out = np.zeros_like(phis[i])
        for k in self.degrees:
            pk, _ = _legendre_np(u, k)
            out = out + self.gamma[(i, k)] * pk
        return out

    def _corr_dmu(self, phis, i):        # d(corr mu_i)/d phi_i  (diagonal)
        u = (phis[i] - self.mid) / self.half
        out = np.zeros_like(phis[i])
        for k in self.degrees:
            _, dpk = _legendre_np(u, k)
            out = out + self.gamma[(i, k)] * dpk / self.half
        return out

    def mu(self, phis):
        base = self.fh.mu(phis)
        return [base[i] + self._corr_mu(phis, i) for i in range(self.M)]

    def dmu_dphi(self, phis):
        H = self.fh.dmu_dphi(phis)
        for i in range(self.M):
            H[i][i] = H[i][i] + self._corr_dmu(phis, i)
        return H

    def dmu_dparam(self, phis, name):
        if name.startswith("basis_"):
            _, i, k = name.split("_"); i, k = int(i), int(k)
            u = (phis[i] - self.mid) / self.half
            pk, _ = _legendre_np(u, k)
            z = [np.zeros_like(phis[0]) for _ in range(self.M)]
            z[i] = pk
            return z
        return self.fh.dmu_dparam(phis, name)

    def gauge_residual(self, species=0, n_quad=64):
        x, w = np.polynomial.legendre.leggauss(n_quad)
        cq = self.mid + self.half * x; wq = self.half * w
        phis = [np.full_like(cq, self.mid) for _ in range(self.M)]
        phis[species] = cq
        r = self._corr_mu(phis, species)
        return float((wq * r).sum()), float((wq * cq * r).sum())
```

Add a complex-safe `_legendre_np(u, k)` helper in this module (numpy port of `neural_energy._legendre`, preserving dtype) — the torch `_legendre` cannot take numpy complex arrays. Copy the closed forms for k in 2..5 verbatim from `neural_energy._legendre`, using `np` ops.

Export in `__init__.py`: add `BasisMultiEnergy` (and `MobilityClosure`, Task 2) to the import and `__all__`.

> **Gauge note for the implementer:** shifted-Legendre P_k (k≥2) are L2-orthogonal to {1, u} on [-1,1], hence to {1, φ_i} on [lo,hi] — so `gauge_residual` is ~0 by construction, no projection needed (unlike the MLP head). The test pins this.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_neural_multiphase.py -k basis -v`
Expected: PASS (FH parity at γ=0; gauge residual ~1e-16).

- [ ] **Step 5: Commit**

```bash
git add src/diffsim/adjoint/neural_multiphase.py src/diffsim/adjoint/__init__.py tests/test_neural_multiphase.py
git commit -m "feat(m6): BasisMultiEnergy — gauge-anchored learnable bulk energy (MultiEnergy)"
```

---

## Task 2: named mobility closure + engine routing

**Files:**
- Modify: `src/diffsim/adjoint/neural_multiphase.py` (add `MobilityClosure`)
- Modify: `src/diffsim/adjoint/multiphase.py` (`MultiCHDiscrete.assemble` transport term + `dR_dparam` `mob_*`; `MultiCHForward.__init__` `mobility=`)
- Test: `tests/test_neural_multiphase.py`

**Interfaces:**
- Produces: `MobilityClosure(name="const", M=2, onsager=None, coeffs=None)`. `.matrix(phi_gp)` returns the M×M mobility at a Gauss-point array (constant → the given Onsager matrix broadcast; the v1 learnable closure `"phi_diag"` → `M_ii(φ) = m0_i·(1 + c_i·φ_i)`, off-diagonal 0). `.param_names` = `mob_{j}` coefficients. `.dmatrix_dparam(phi_gp, name)` returns the M×M derivative. The engine's transport flux uses `Lam = closure.matrix(mu_gp?)` — **NOTE:** mobility multiplies `∇μ`; the closure depends on φ at the Gauss point (evaluate on `phi_gp`). `MultiCHForward(dm, energy, mobility=<closure or matrix>, kappa, dt, order, backend=None)` — a plain matrix/array is wrapped as `MobilityClosure("const", onsager=…)` (back-compat).

- [ ] **Step 1: Write the failing test**

```python
def test_const_closure_matches_current_engine():
    # a const closure must reproduce the existing constant-Onsager forward
    import numpy as np
    from diffsim.adjoint import BasisMultiEnergy, MobilityClosure
    from diffsim.adjoint.multiphase import MultiCHForward
    from tests.test_multiphase_adjoint import _dm     # reuse mesh helper
    dm, mesh = _dm(3)
    chi = np.zeros((3,3)); chi[0,1]=chi[1,0]=2.5; chi[0,2]=chi[2,0]=1.0; chi[1,2]=chi[2,1]=0.8
    en = BasisMultiEnergy(chi, np.ones(3))
    ons = np.eye(2)
    cc = np.cos(np.pi*mesh.node_coords[:,0])*np.cos(np.pi*mesh.node_coords[:,1])
    ic = [0.30+0.05*cc, 0.30+0.05*cc]
    f_mat = MultiCHForward(dm, en, mobility=ons, kappa=[1e-2,1e-2], dt=1e-3); f_mat.set_initial(ic); f_mat.run(3)
    f_cls = MultiCHForward(dm, en, mobility=MobilityClosure("const", M=2, onsager=ons), kappa=[1e-2,1e-2], dt=1e-3); f_cls.set_initial(ic); f_cls.run(3)
    for i in range(2):
        assert np.allclose(f_mat.steps[-1]["phis"][i], f_cls.steps[-1]["phis"][i])

def test_phi_diag_closure_complex_step():
    # dR/d(mob_j) via complex-step matches dR_dparam
    ...  # mirror the existing test_dR_dparam_complex_step pattern for a mob_j name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_neural_multiphase.py -k "closure" -v`
Expected: FAIL — `MobilityClosure` undefined / `MultiCHForward` rejects a closure.

- [ ] **Step 3: Write minimal implementation**

Add `MobilityClosure` to `neural_multiphase.py`:

```python
class MobilityClosure:
    """NAMED mobility M(phi;a).  'const' = the given Onsager matrix (today's
    behavior); 'phi_diag' = M_ii = m0_i*(1 + c_i*phi_i), off-diag 0 (learnable
    m0_i, c_i)."""
    def __init__(self, name="const", M=2, onsager=None, coeffs=None):
        self.name = name; self.M = int(M)
        self.onsager = None if onsager is None else np.asarray(onsager)
        self.coeffs = dict(coeffs or {})
        if name == "const":
            self.param_names = ()
        elif name == "phi_diag":
            self.coeffs.setdefault("mob_m0", 1.0); self.coeffs.setdefault("mob_c", 0.0)
            self.param_names = ("mob_m0", "mob_c")
        else:
            raise ValueError(f"unknown mobility closure {name!r}")

    def matrix(self, phi_gp):        # returns list-of-lists [M][M] of [ne,nqp] arrays
        M = self.M
        if self.name == "const":
            return [[self.onsager[i, j] * np.ones_like(phi_gp[0]) for j in range(M)] for i in range(M)]
        m0, c = self.coeffs["mob_m0"], self.coeffs["mob_c"]
        out = [[np.zeros_like(phi_gp[0]) for _ in range(M)] for _ in range(M)]
        for i in range(M):
            out[i][i] = m0 * (1.0 + c * phi_gp[i])
        return out

    def dmatrix_dparam(self, phi_gp, name):
        M = self.M
        out = [[np.zeros_like(phi_gp[0]) for _ in range(M)] for _ in range(M)]
        if self.name == "phi_diag" and name == "mob_m0":
            c = self.coeffs["mob_c"]
            for i in range(M): out[i][i] = (1.0 + c * phi_gp[i])
        elif self.name == "phi_diag" and name == "mob_c":
            m0 = self.coeffs["mob_m0"]
            for i in range(M): out[i][i] = m0 * phi_gp[i]
        return out
```

In `multiphase.py`: (a) `MultiCHForward.__init__` — wrap a non-closure `onsager` as `MobilityClosure("const", M, onsager=…)`; store `self.mobility`. (b) `MultiCHDiscrete.assemble` — replace the constant `Lam[i,j]` reads in the transport flux/Jacobian with `Lam = params["mobility"].matrix(phi_gp)` (now φ-dependent per Gauss point); the flux `Σ_j Lam[i][j]*gmu[j]` and the Jacobian block `Ae[2i, 2j+1] += Lam[i][j]*LL` use the arrays. For `phi_diag`, add the extra Jacobian term `∂flux_i/∂φ_i` (mobility depends on φ) — `∂(M_ii ∇μ_i)/∂φ_i` contributes to the `Ae[2i::blk, 2i::blk]` block: `+ (∂M_ii/∂φ_i) ∇μ_i · ∇N` term. (c) `dR_dparam` — add a `mob_*` branch delegating to `closure.dmatrix_dparam`, scattering `∫ ∇N·(dM·∇μ)` onto the φ-rows (generalizing the existing `onsager_*` branch). (d) thread `mobility` into `params` dict in `_params`.

> **Consistency with existing tests:** the `const` closure must keep the existing `onsager_*` `dR_dparam` and the three-way gate byte-behaviorally identical. Run `tests/test_multiphase_adjoint.py` after — must stay green.

- [ ] **Step 4: Run test to verify it passes + no regression**

Run: `.venv/bin/pytest tests/test_neural_multiphase.py -k closure -v && .venv/bin/pytest tests/test_multiphase_adjoint.py -q`
Expected: closure tests PASS; existing adjoint gate unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/diffsim/adjoint/neural_multiphase.py src/diffsim/adjoint/multiphase.py tests/test_neural_multiphase.py
git commit -m "feat(m6): named mobility closure (phi_diag) + engine routing (const back-compat)"
```

---

## Task 3: `MultiCHTwin` extension — generic energy + closure + optional τ

**Files:**
- Modify: `src/diffsim/adjoint/torch_twin.py` (`MultiCHTwin`)
- Test: `tests/test_neural_multiphase.py`

**Interfaces:**
- Produces: `MultiCHTwin.grads(...)` accepts names `basis_{i}_{k}`, `mob_{j}`, and `tau`. It builds torch leaves for each, marches with (i) a differentiable energy correction mirroring `BasisMultiEnergy` (add `Σ_k γ_{i,k} P_k(û_i)` to `_mu_and_H`'s `mus[i]` and the diagonal Hessian), (ii) the φ-dependent mobility from the closure, and (iii) a global time-scale τ multiplying `dt` (τ default 1.0; when `tau` requested, it is a leaf and `dt_eff = τ·dt`). Returns `{name: leaf.grad}`.

- [ ] **Step 1: Write the failing test** (twin basis/mob/τ grads vs central FD, ternary bdf1)

```python
def test_twin_basis_mob_tau_vs_fd():
    # build a BasisMultiEnergy + phi_diag closure; compare MultiCHTwin.grads
    # for names ["basis_0_2","mob_m0","tau"] against central FD of the same
    # loss 0.5*sum||phi_i,N - tgt||^2.  rel < 1e-6.
    ...
```

- [ ] **Step 2: Run to verify it fails** — `KeyError`/unsupported name in `grads`.

- [ ] **Step 3: Implement.** In `MultiCHTwin`:
  - Add an optional `energy_corr` path in `_mu_and_H`: if basis leaves are present, add `Σ_k γ_{i,k}·P_k((φ_i−mid)/half)` to `mus[i]` and `Σ_k γ_{i,k}·P'_k/half` to `H[i][i]` (use the torch `_legendre` from `neural_energy`).
  - Replace the constant `Lam[i,j]` in `_assemble` with the closure matrix on `phi_gp` (torch), plus the `phi_diag` extra Jacobian term (autograd handles it since the twin differentiates through convergence — but `_assemble` builds J explicitly, so add the analytic `∂M_ii/∂φ_i` block to keep Newton exact; autograd flows through the leaves regardless).
  - Thread `tau`: `dt_eff = tau * self.dt` in `march`; `sigma = c0_/dt_eff`, history `(cc/dt_eff)`.
  - In `grads`, build leaves for `basis_*`/`mob_*`/`tau` (init from provided base values), assemble the differentiable energy+closure, march, `loss.backward()`.

- [ ] **Step 4: Run to verify it passes** — twin grads == FD < 1e-6.

- [ ] **Step 5: Commit**
```bash
git commit -am "feat(m6): MultiCHTwin consumes generic energy + mobility closure + tau (twin==FD)"
```

---

## Task 4: three-way gate — `dJ/dθ`, `dJ/da`, `dJ/dτ`

**Files:** Modify `tests/test_neural_multiphase.py`.

**Interfaces:** Consumes Tasks 1–3. Reuses `_check_three_way` and the `_three_way_*` pattern from `tests/test_multiphase_adjoint.py`.

- [ ] **Step 1: Write the gate.** A `_three_way_m6(dm, coords, M, order, n_steps)` mirroring `_three_way_multi`, with `names = ["basis_0_2","basis_0_3","mob_m0","mob_c","tau"]`, a `BasisMultiEnergy` + `phi_diag` closure forward, hand adjoint via `MultiCHAdjoint.gradient` (the engine's `dR_dparam` now covers `basis_*`, `mob_*`; `tau` via the dt-scaling cotangent — see note), twin via `MultiCHTwin.grads`, and central FD. Three test functions: ternary bdf1, ternary bdf2, quaternary bdf1, each `_check_three_way(res, tag)`.

> **τ in the hand adjoint:** a global `dt_eff = τ·dt` scales `sigma=c0/dt_eff` and every history coefficient `ch_k/dt_eff`; `dJ/dτ = Σ_n λ_n·(∂R_n/∂τ)` where `∂R_n/∂τ` is the analytic derivative of the time term w.r.t. τ (`∂sigma/∂τ = −c0/(τ²dt)`, `∂(ch_k/dt_eff)/∂τ = −ch_k/(τ²dt)`). Implement `∂R/∂τ` in `MultiCHForward`/adjoint as a recorded per-step term (small addition, analogous to the mean-φ cotangent). If τ-in-hand-adjoint proves involved, gate `tau` twin-vs-FD only in Task 3 and defer the hand-adjoint `dJ/dτ` to a follow-up — but `basis_*`/`mob_*` MUST pass the full three-way here.

- [ ] **Step 2–4:** Run each; expect adj/twin < 1e-10, adj/fd < 1e-6.
- [ ] **Step 5: Commit** `test(m6): three-way gate for dJ/dtheta, dJ/da, dJ/dtau`

---

## Task 5: synthetic-recovery gate

**Files:** Create `tests/test_m6_recovery.py`, `examples/m6_learn_from_md.py` (synthetic path).

**Interfaces:** Consumes Tasks 1–4 + the rung-2 optimizer-agnostic gradient. No MD data.

- [ ] **Step 1: Write the recovery test.** Generate a "truth" trajectory from a known `BasisMultiEnergy(coeffs=γ*)` + `phi_diag(coeffs=a*)`; record the final φ-field as the target. Starting from γ=0, a=nominal, run gradient descent on {basis_*, mob_*} using the interfacial/field-L2 loss and `MultiCHAdjoint.gradient`; assert the loss decreases ≥10× and the recovered γ,a approach γ*,a* (gauge-invariant compare for γ). Small resolution (level 4–5), few steps — keep it a fast deterministic gate.

```python
def test_synthetic_recovery_of_energy_and_mobility():
    # truth -> target field; descend {basis_*, mob_*}; assert loss drops >=10x
    # and ||gamma - gamma*|| shrinks.  Fixed seed; level 4; ~30 opt steps.
    ...
```

- [ ] **Step 2–4:** Run; the descent must reduce the field-L2 loss ≥10× and move the coefficients toward truth (identifiability permitting — if the Gramian is rank-deficient in a direction, assert only the identifiable components; note this in the test).
- [ ] **Step 5: Commit** `test(m6): synthetic-recovery gate (recover known energy+mobility from a trajectory)`

---

## Self-Review

**Spec coverage:** `NeuralMultiEnergy` → Task 1 (as `BasisMultiEnergy`; MLP head deferred, flagged). Named mobility closure → Task 2. Twin extension → Task 3. Three-way `dJ/dθ,da,dτ` → Task 4. Synthetic recovery → Task 5. **MD ingest, hybrid CG+descriptor loss, Gramian pre-check → Plan B (out of scope here, stated).** κ fixed → honored (not learned). Gauge inside the energy object → Task 1 (exact-by-construction). Mobility NAMED → Task 2.

**Placeholder scan:** Tasks 3–5 give interfaces, key formulas, and test intent but abbreviate some step code (marked `...`) rather than full literal code — a deliberate concession for the novel research math (twin energy-correction, τ cotangent, recovery optimizer), each anchored to a named precedent (`neural_energy`, the existing `_three_way_multi`, the mean-φ cotangent). The implementer derives against those. Tasks 1–2 (the crux) carry full code.

**Type consistency:** `basis_{i}_{k}` / `mob_{j}` / `tau` names are identical across `BasisMultiEnergy.param_names`, `MobilityClosure.param_names`, `MultiCHTwin.grads`, `dR_dparam`, and the gates. `MobilityClosure.matrix` returns `[M][M]` list-of-arrays consumed identically in `assemble` and the twin. `MultiEnergy` contract (`mu`/`dmu_dphi`/`dmu_dparam`) matches `FHMultiEnergy`.
