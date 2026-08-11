# While-waiting crystallization extensions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** GPU-validate the crystallization adjoint (2M+K) through cuDSS, and build a reusable multi-snapshot recovery harness (φ+ψ field-L2 + structure-factor descriptor hook) that lifts ②'s single-snapshot identifiability limit.

**Architecture:** Both drop onto existing seams with **zero engine change** — `CrystalCHForward(backend=…)` already routes solves through a pluggable `LinearBackend`, and `CrystalCHAdjoint.gradient` already accepts a per-step `dJdx` list. #1 is a validation test + gpubox run. #2 is a new `crystal_recovery.py` module (descriptor + multi-snapshot loop) reusing the ② recovery helpers.

**Tech Stack:** numpy, scipy (sparse solve), NVIDIA cuDSS via nvmath (GPU, gpubox), pytest. CPU-testable on Mac; GPU parts skip-gated.

**Spec:** `docs/superpowers/specs/2026-08-08-crystal-while-waiting-design.md`.

## Global Constraints

- **No engine change.** Do not modify `CrystalCHDiscrete`/`CrystalCHForward`/`CrystalCHAdjoint`/`CrystalCHTwin`/`NeuralCrystalEnergy` logic. Both features consume existing interfaces (`backend=`, per-step `dJdx`, `fwd.steps[s]["phis"/"psis"]`).
- **Recovery uses the HAND adjoint** (`CrystalCHAdjoint`), the production gradient path — not the twin.
- **No off-by-one:** the returned recovered params and `loss_hist[-1]` must correspond to the same state (re-evaluate loss at final params after the loop) — the ② Task 5 lesson.
- **GPU tests skip-gated:** guard with the same GPU availability check the K=0 rung-2 tests use (find it in `tests/` — a `pytest.mark.skipif` on cuDSS/CUDA availability). Mac is CPU-only.
- **Env:** `.venv/bin/pytest` only (no bare `python`, no `timeout`). Commits end with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- θ absent; full 2-D coupling and χ(ψ) out of scope.

## File Structure

- `tests/test_crystallization_multi.py` — **modify** (Task 1): cuDSS parity + backend-honored tests.
- `src/diffsim/adjoint/crystal_recovery.py` — **new** (Tasks 2–3): `structure_factor`, `nodal_to_grid`, `recover_multisnapshot`.
- `tests/test_crystal_recovery.py` — **new** (Tasks 2–3).
- `examples/crystal_learn_multisnapshot.py` — **new** (Task 3).

---

### Task 1: GPU cuDSS parity for the 2M+K crystallization adjoint

**Files:** Modify `tests/test_crystallization_multi.py`.

**Interfaces:**
- Consumes: `CrystalCHForward(dm, energy, crystallizable, onsager=, kappa=, eps2=, L=, dt=, order=, backend=)`, `CrystalCHAdjoint(fwd).gradient(dJdx, names)`, `AdditiveCrystalEnergy`, `ScipyBackend`/`CudssBackend` (from `diffsim.adjoint.linsolve_backend` or `diffsim.adjoint`), the module's `_dm` helper.
- Produces: two tests (one CPU-always, one GPU-skip-gated).

- [ ] **Step 1: Find the GPU skip-guard** used by the existing rung-2 K=0 cuDSS test. Search: `grep -rn "cudss\|CudssBackend\|skipif" tests/test_multiphase_adjoint.py tests/` — reuse that exact `skipif` condition (e.g. a helper that probes CUDA/nvmath availability). Do NOT invent a new gating mechanism.

- [ ] **Step 2: Write the CPU backend-honored test**

```python
def test_crystal_backend_honored_cpu():
    """Explicit ScipyBackend == default; confirms the backend= seam is live."""
    from diffsim.adjoint.crystallization_multi import (
        CrystalCHForward, CrystalCHAdjoint, AdditiveCrystalEnergy)
    from diffsim.adjoint import ScipyBackend
    dm, mesh = _dm(2)
    M, cryst = 2, (0,)
    chi, N = _chiN(M)  # reuse the module helper
    energy = AdditiveCrystalEnergy(chi, N, cryst, {0: 0.7}, {0: -0.9}, {0: 1.1})
    cc = np.cos(np.pi * mesh.node_coords[:, 0])
    phi0 = [0.28 + 0.03 * cc, 0.30 + 0.03 * cc]; psi0 = [0.25 + 0.02 * cc]
    names = ["dsig_0", "L_0", "chi_0_1", "kappa_0"]
    def run(backend):
        fwd = CrystalCHForward(dm, energy, cryst, onsager=np.eye(M),
                               kappa=[0.01, 0.02], eps2={0: 0.015}, L={0: 1.2},
                               dt=0.01, order=1, backend=backend)
        fwd.set_initial(phi0, psi0); fwd.run(3)
        blk = fwd.op.blk
        dJdx = [np.zeros(fwd.op.ndof) for _ in range(3)]
        for i in range(M): dJdx[-1][2 * i::blk] = fwd.steps[-1]["phis"][i] - 0.28
        dJdx[-1][2 * M::blk] = fwd.steps[-1]["psis"][0] - 0.25
        return CrystalCHAdjoint(fwd).gradient(dJdx, names)
    g_default = run(None)
    g_explicit = run(ScipyBackend())
    for nm in names:
        assert np.isclose(g_default[nm], g_explicit[nm], rtol=1e-12), nm
```

- [ ] **Step 3: Write the GPU cuDSS parity test (skip-gated)**

```python
@<the same skipif guard found in Step 1>
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
    phi0 = [0.28 + 0.03 * cc, 0.30 + 0.03 * cc]; psi0 = [0.25 + 0.02 * cc]
    names = ["dsig_0", "dh_0", "eps2_0", "L_0", "chi_0_1", "kappa_0"]
    def run(backend):
        fwd = CrystalCHForward(dm, energy, cryst, onsager=np.eye(M),
                               kappa=[0.01, 0.02], eps2={0: 0.015}, L={0: 1.2},
                               dt=0.01, order=2, backend=backend)
        fwd.set_initial(phi0, psi0); fwd.run(4)
        blk = fwd.op.blk
        dJdx = [np.zeros(fwd.op.ndof) for _ in range(4)]
        for i in range(M): dJdx[-1][2 * i::blk] = fwd.steps[-1]["phis"][i] - 0.28
        dJdx[-1][2 * M::blk] = fwd.steps[-1]["psis"][0] - 0.25
        return CrystalCHAdjoint(fwd).gradient(dJdx, names)
    g_cpu = run(ScipyBackend())
    g_gpu = run(CudssBackend("cuda:0"))
    for nm in names:
        assert np.isclose(g_gpu[nm], g_cpu[nm], rtol=1e-6, atol=1e-8), \
            (nm, g_gpu[nm], g_cpu[nm])
```

- [ ] **Step 4: Run** — `.venv/bin/pytest tests/test_crystallization_multi.py -q`. The CPU test passes; the GPU test SKIPS on Mac. Confirm no regression (all prior crystallization tests pass).

- [ ] **Step 5: Commit** `test(crystallization): cuDSS parity for the 2M+K adjoint (GPU-gated) + backend-honored smoke`

(The GPU leg is executed on gpubox by the controller after merge; this task only lands the gated test.)

---

### Task 2: Structure-factor descriptor + nodal→grid helper

**Files:** Create `src/diffsim/adjoint/crystal_recovery.py`; test `tests/test_crystal_recovery.py`.

**Interfaces:**
- Produces: `nodal_to_grid(field_1d, mesh) -> (side_y, side_x) array` (lexsort reshape, mirrors `daisy_morph.gradients.grid_fields`); `structure_factor(field_2d, nbins=None) -> S (1d radially-averaged |FFT|^2)`; `structure_factor_grad(field_2d, target_S, nbins=None) -> (2d array)` = `∂(0.5‖S(field)−target_S‖²)/∂field` (the field-space cotangent of the descriptor mismatch).

- [ ] **Step 1: Write the failing test** (descriptor gradient vs central FD):

```python
import numpy as np
import pytest
from diffsim.adjoint.crystal_recovery import (
    structure_factor, structure_factor_grad, nodal_to_grid)

def test_structure_factor_grad_vs_fd():
    rng = np.random.default_rng(0)
    f = 0.3 + 0.1 * rng.standard_normal((8, 8))
    tgt = structure_factor(0.3 + 0.1 * rng.standard_normal((8, 8)))
    g = structure_factor_grad(f, tgt)
    # central FD on a few random pixels
    def loss(x):
        return 0.5 * float(((structure_factor(x) - tgt) ** 2).sum())
    h = 1e-6
    for (a, b) in [(0, 0), (3, 5), (7, 2), (4, 4)]:
        fp = f.copy(); fp[a, b] += h
        fm = f.copy(); fm[a, b] -= h
        fd = (loss(fp) - loss(fm)) / (2 * h)
        assert np.isclose(g[a, b], fd, rtol=1e-4, atol=1e-6), ((a, b), g[a, b], fd)
```

- [ ] **Step 2: Run to verify fail** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `crystal_recovery.py` descriptor block. `structure_factor`: `F = np.fft.fft2(field - field.mean()); P = np.abs(F)**2`; radially bin `P` by integer wavenumber magnitude into `nbins` (default `min(shape)//2`), return the per-bin mean → `S`. `structure_factor_grad`: analytic chain rule of `0.5‖S−target‖²` through the radial-average (linear map `A`: `P→S`) and `P=|F|²` back to `field` — `dL/dP = A^T((S−target)/counts)` broadcast to pixels in each bin; `dL/dfield = ifft2-based adjoint of |fft2|²` = `2·real(ifft2(dL/dP ⊙ F))` (with the mean-subtraction adjoint: subtract the spatial mean of the result). Keep it dtype-clean (real fields). `nodal_to_grid`: unique-x/unique-y lexsort reshape (copy the pattern from `src`-side `daisy_morph.gradients.grid_fields` — coords = `mesh.node_coords[:, :2]`, `order = np.lexsort((round(x), round(y)))`, reshape `(side_y, side_x)`).

- [ ] **Step 4: Run to verify pass** — descriptor grad matches FD < 1e-4.

- [ ] **Step 5: Commit** `feat(crystal-recovery): structure-factor descriptor (S(k)) + field-space gradient + nodal_to_grid`

---

### Task 3: Multi-snapshot recovery harness + identifiability gate

**Files:** Modify `src/diffsim/adjoint/crystal_recovery.py`; test `tests/test_crystal_recovery.py`; create `examples/crystal_learn_multisnapshot.py`.

**Interfaces:**
- Consumes: `CrystalCHForward`/`CrystalCHAdjoint`, `NeuralCrystalEnergy`, Task-2 `structure_factor`/`structure_factor_grad`/`nodal_to_grid`; the ② helpers in `examples/crystal_learn_from_synthetic.py` (`_make_dm`, `_make_energy`, `_make_fwd`) — import or mirror.
- Produces: `recover_multisnapshot(dm, mesh, planted, names, n_steps, snapshots, order, n_iter, lr, descriptor=False, lam_desc=0.0) -> (loss_hist, theta_hat, theta_true)`.

- [ ] **Step 1: Write the failing test** (multi-snapshot lifts identifiability vs single-snapshot):

```python
def test_multisnapshot_recovers_higher_mode():
    from diffsim.adjoint.crystal_recovery import recover_multisnapshot
    dm, mesh = _dm(2)  # add the _dm helper (copy from test_crystallization_multi)
    planted = {"cpl_0_1": 0.15, "cpl_0_2": -0.12}
    names = ["cpl_0_1", "cpl_0_2"]
    # multi-snapshot: observe several times -> higher mode cpl_0_2 identifiable
    lh, th, tt = recover_multisnapshot(
        dm, mesh, planted, names, n_steps=6, snapshots=(2, 4, 6),
        order=1, n_iter=60, lr=0.5)
    assert lh[-1] <= lh[0] / 20.0
    # the higher mode cpl_0_2 recovers where single-snapshot (② Task 5) fails
    assert abs(th["cpl_0_2"] - tt["cpl_0_2"]) <= 0.30 * abs(tt["cpl_0_2"])
```

- [ ] **Step 2: Run to verify fail.**

- [ ] **Step 3: Implement** `recover_multisnapshot`:
  1. Build a planted `NeuralCrystalEnergy` (planted `cpl_*`), run `CrystalCHForward` `n_steps`, capture targets at EACH observed step: `phi_star[s], psi_star[s] = fwd.steps[s-1]["phis"/"psis"]` for `s in snapshots` (1-indexed step → `steps[s-1]`).
  2. Guess energy with `cpl_*` zeroed. Loop `n_iter`: run forward; build a per-step `dJdx` list of length `n_steps`, zero except at observed steps where `dJdx[s-1][2i::blk] = w_s(φ_i(t_s)−φ*_i)`, `[2M+j::blk] = w_s(ψ_j−ψ*_j)` (uniform `w_s = 1/len(snapshots)`); if `descriptor`, add `lam_desc · structure_factor_grad(nodal_to_grid(field,mesh), target_S)` scattered back to the φ-rows at each observed step (compute `target_S` from the planted fields once). `grads = CrystalCHAdjoint(fwd).gradient(dJdx, names)`; descend `guess.c[(k,b)] -= lr·grads[name]`; record the multi-snapshot loss (field-L2 summed over snapshots, + descriptor term if on).
  3. Re-evaluate loss at final params; return `(loss_hist, {name: recovered}, {name: planted})`.
  Docstring: note this is the format-agnostic ③/M6-Plan-B core; the descriptor path is where real-MD S(k) targets plug in.

- [ ] **Step 4: Run to verify pass** — loss drop ≥20×, `cpl_0_2` within 30%. Tune `snapshots`/`n_iter`/`lr` if needed (do NOT loosen the assert beyond 30%; if the higher mode genuinely won't identify from these snapshots, add a 4th snapshot or a second initial condition — that's the physics of the fix, document it).

- [ ] **Step 5: Add a descriptor-on smoke** (`descriptor=True, lam_desc=1e-3` runs and still drops loss) and the example driver `examples/crystal_learn_multisnapshot.py` (guard CLI under `if __name__ == "__main__":`). Run the full file + `tests/test_crystallization_multi.py tests/test_neural_crystal.py` (no regression).

- [ ] **Step 6: Commit** `feat(crystal-recovery): multi-snapshot recovery harness (field-L2 + descriptor hook) — identifiability lift`

---

## Self-Review

**Spec coverage:** #1 GPU cuDSS parity + backend-honored → Task 1. #2 descriptor (`S(k)` + grad) → Task 2; multi-snapshot harness + hook + identifiability gate + example → Task 3. Zero-engine-change honored (only tests + a new module + an example). Deferrals (2-D coupling, χ(ψ), real MD, θ) correctly absent.

**Placeholder scan:** Task 1 carries full test code (the GPU skipif is deliberately delegated to Step 1's search — the existing guard must be reused, not invented). Tasks 2–3 carry full test code + concrete implementation direction anchored to named precedents (`daisy_morph.gradients.grid_fields`, ② `crystal_learn_from_synthetic` helpers, `CrystalCHAdjoint` per-step `dJdx`).

**Type consistency:** `dJdx` is a length-`n_steps` list of `ndof` vectors, populated at `steps[s-1]` for 1-indexed `snapshots` — consistent across Task 3. `structure_factor`→`S` (1d), `structure_factor_grad`→2d field cotangent, both keyed off the same `nbins` radial binning. `recover_multisnapshot` returns `(loss_hist, theta_hat, theta_true)` matching the ② `recover_coupling` shape.
