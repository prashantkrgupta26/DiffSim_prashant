# Differentiable OrgElMorph — Rung 2 (GPU cuDSS backend) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the three-way-verified rung-1 M-component adjoint a pluggable GPU/cuDSS linear-solver backend so its fixed-step forward+adjoint scales to 256×128 (~131k DOF), without touching the verified assembly or adjoint math.

**Architecture:** The rung-1 engine (`adjoint/multiphase.py`) solves exactly two square sparse systems per committed step — the forward Newton update `J·dx = −R` (line 402) and the adjoint `Jᵀλ = rhs` (line 458), both via scipy `splu`. Rung 2 introduces a `LinearBackend` abstraction with two implementations: `ScipyBackend` (host `splu`, the bit-identical rung-1 default) and `CudssBackend` (factor+solve on a CUDA device through nvmath's cuDSS `DirectSolver`, mirroring the production `MultiPhaseStepper._solve`). Assembly stays host-numpy (cheap at 131k DOF); only the expensive factorization+solve moves to device. This is the "scale the verified engine" path chosen 2026-08-07 over differentiating the production adaptive trajectory (that is deferred to rung 3, see the roadmap section).

**Tech Stack:** numpy 2.5.1, scipy 1.18.0 (host assembly + `splu`), torch 2.13.0 (sparse CSR device tensors), nvmath cuDSS `DirectSolver` (GPU direct solve), Warp (device probe only). Verification GPU hardware: **gpubox** (2× RTX 6000 Ada, 48 GB, WSL2). Development/CPU-testing host: office Mac (CPU-only Warp — cannot run cuDSS).

## Global Constraints

- **The office Mac cannot run cuDSS or any GPU solve** (CPU-only Warp). Every `CudssBackend`-executing step (Tasks 6–8) runs on **gpubox**, code shipped Mac→gpubox via `scripts/remote/gpubox-*.sh`; results reconciled back to master (see `docs/dev/remote-workflow.md`). The Mac is the git master and sole GitHub gatekeeper — never treat gpubox's tree as authority.
- **gpubox GPU access requires** the WSL lib-path prefix on every GPU command: `LD_LIBRARY_PATH=/usr/lib/wsl/lib` (else `Warp CUDA error 100`). gpubox repo: `/home/bglab/Baskar/DiffSim`, venv `.venv`.
- **256×128 ternary is ~131k DOF — comfortably under the ~812k cuDSS memory wall** on a 48 GB card. cuDSS is the correct solver at this scale; do not reach for iterative solvers here.
- **Non-regression is non-negotiable:** default backend stays `ScipyBackend`; the full existing three-way gate (`tests/test_multiphase_adjoint.py`, 21 tests) must pass **bit-identically** after the backend refactor. `ScipyBackend` reproduces the exact pre-backend `splu` calls.
- **Run tests with** `.venv/bin/pytest` on Mac (no bare `python`; no `timeout` command on macOS). On gpubox, `.venv/bin/pytest` prefixed with the WSL lib-path.
- **cuDSS call sequence is authoritative in production:** `CudssBackend` must mirror `src/diffsim/physics/multiphase.py:1934-1984` (`_solve`, `linsolver="cudss"` branch) — plan-once-per-pattern, `nnz`-change guard, `reset_operands`/`factorize`/`solve`. Since this cannot be exercised on the Mac, copy that proven sequence rather than improvising.
- **Branch:** `diff-orgelmorph` (continue the rung-1 branch). Commit-per-green.

---

## File Structure

**DiffSim (branch `diff-orgelmorph`):**
- Create `src/diffsim/adjoint/linsolve_backend.py` — `LinearBackend` protocol, `ScipyBackend`, `CudssBackend`, `scipy_to_torch_csr` helper. One responsibility: turn "solve this scipy CSR system (or its transpose)" into a host or device solve.
- Modify `src/diffsim/adjoint/multiphase.py` — `MultiCHForward.__init__` gains `backend=None`; forward Newton (line 402) and adjoint solve (line 458) route through the backend. No change to assembly or adjoint math.
- Modify `src/diffsim/adjoint/__init__.py` — export `LinearBackend`, `ScipyBackend`, `CudssBackend`.
- Modify `tests/test_multiphase_adjoint.py` — add backend-abstraction unit tests + a GPU-gated cuDSS-parity test (skipped on Mac).

**daisy-morph (branch `feat/gradients`):**
- Modify `src/daisy_morph/gradients.py` — `_forward`/`compute`/`objective_value` accept `backend=None`, thread it into `MultiCHForward`.
- Modify `src/daisy_morph/core.py` — `_run_gradients` accepts `device`, builds the backend by device (GPU → `CudssBackend`, else `ScipyBackend`/None), calls `ensure_gpu()` on CUDA, stamps GPU provenance.
- Modify `src/daisy_morph/tests/test_gradients.py` — CPU test that device routing preserves the default path; GPU-gated test that `device="cuda:0"` selects `CudssBackend`.
- Modify `pyproject.toml:13` — bump the DiffSim git-SHA pin (Task 8, on gpubox after parity passes).

---

## Task 1: LinearBackend protocol + ScipyBackend

**Files:**
- Create: `src/diffsim/adjoint/linsolve_backend.py`
- Modify: `src/diffsim/adjoint/__init__.py`
- Test: `tests/test_multiphase_adjoint.py`

**Interfaces:**
- Produces: `class LinearBackend` with `solve(A, b) -> ndarray` (solves `A x = b`) and `solve_T(A, b) -> ndarray` (solves `Aᵀ x = b`), `A` a `scipy.sparse` matrix, `b`/return length-`ndof` `ndarray`. `class ScipyBackend(LinearBackend)` — host `splu`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_multiphase_adjoint.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_multiphase_adjoint.py::test_scipy_backend_solve_and_transpose -v`
Expected: FAIL with `ImportError: cannot import name 'ScipyBackend'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/diffsim/adjoint/linsolve_backend.py`:

```python
"""Pluggable linear-solver backends for the M-component adjoint engine.

The rung-1 engine (adjoint/multiphase.py) solves two square sparse systems per
committed step: the forward Newton update J dx = -R and the adjoint J^T lam =
rhs.  Rung 2 keeps the verified numpy assembly and swaps ONLY these solves onto
a CUDA device via cuDSS, so 256x128 (~131k DOF, past scipy splu's practical CPU
ceiling) becomes reachable while the three-way-verified adjoint math is
untouched.

ScipyBackend is the CPU default (bit-identical to the pre-backend splu path);
CudssBackend runs factor+solve on a CUDA device through nvmath's cuDSS
DirectSolver, mirroring physics/multiphase.MultiPhaseStepper._solve.
"""
from scipy.sparse.linalg import splu


class LinearBackend:
    """Solve A x = b and A^T x = b for a square scipy-sparse A."""

    def solve(self, A, b):
        raise NotImplementedError

    def solve_T(self, A, b):
        raise NotImplementedError


class ScipyBackend(LinearBackend):
    """Host splu — the rung-1 default; preserves exact pre-backend behavior."""

    def solve(self, A, b):
        return splu(A.tocsc()).solve(b)

    def solve_T(self, A, b):
        return splu(A.T.tocsc()).solve(b)
```

Add to `src/diffsim/adjoint/__init__.py` (extend the existing import/`__all__` block):

```python
from .linsolve_backend import LinearBackend, ScipyBackend, CudssBackend
```

and add `"LinearBackend"`, `"ScipyBackend"`, `"CudssBackend"` to `__all__`.

> Note: `CudssBackend` is created in Task 4. To keep `__init__.py` importable now, either add the export in Task 1 and create a stub `class CudssBackend: ...` placeholder in `linsolve_backend.py` that raises on construction, replaced in Task 4 — OR add only `LinearBackend, ScipyBackend` here and add `CudssBackend` to the import/`__all__` in Task 4. **Choose the latter** (no throwaway stub): in this step import only `LinearBackend, ScipyBackend`.

So in this step the `__init__.py` line is:

```python
from .linsolve_backend import LinearBackend, ScipyBackend
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_multiphase_adjoint.py::test_scipy_backend_solve_and_transpose -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/diffsim/adjoint/linsolve_backend.py src/diffsim/adjoint/__init__.py tests/test_multiphase_adjoint.py
git commit -m "feat(orgelmorph-adj): LinearBackend protocol + ScipyBackend (rung-2 solve abstraction)"
```

---

## Task 2: Route the engine solves through the backend (non-regression)

**Files:**
- Modify: `src/diffsim/adjoint/multiphase.py:343-352` (`MultiCHForward.__init__`), `:402` (forward Newton solve), `:458` (adjoint solve)
- Test: `tests/test_multiphase_adjoint.py` (existing 21-test gate is the guard)

**Interfaces:**
- Consumes: `ScipyBackend` from Task 1.
- Produces: `MultiCHForward(dm, energy, onsager, kappa, dt=1e-2, order=1, newton_tol=1e-12, newton_max=30, backend=None)` — stores `self.backend = backend or ScipyBackend()`. `MultiCHAdjoint` reads `self.fwd.backend`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_multiphase_adjoint.py`:

```python
def test_forward_accepts_backend_and_defaults_scipy():
    from diffsim.adjoint import ScipyBackend
    from diffsim.adjoint.multiphase import MultiCHForward, FHMultiEnergy
    import numpy as np
    # reuse whatever dm/energy the existing forward-parity test builds; the
    # assertion is purely about the backend attribute + default type.
    en = FHMultiEnergy(np.array([[0.0, 3.0, 1.0],
                                 [3.0, 0.0, 0.8],
                                 [1.0, 0.8, 0.0]]), np.array([1.0, 1.0, 1.0]))
    dm = _ternary_dm(level=3)          # existing helper in this test module
    fwd = MultiCHForward(dm, en, onsager=np.eye(2), kappa=[1e-3, 1e-3], dt=1e-3)
    assert isinstance(fwd.backend, ScipyBackend)
    be = ScipyBackend()
    fwd2 = MultiCHForward(dm, en, onsager=np.eye(2), kappa=[1e-3, 1e-3],
                          dt=1e-3, backend=be)
    assert fwd2.backend is be
```

> If the test module has no `_ternary_dm(level=...)` helper, use the exact dm-construction lines the existing `test_forward_parity_ternary` uses (copy them inline). Do not invent a mesh API.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_multiphase_adjoint.py::test_forward_accepts_backend_and_defaults_scipy -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'backend'`.

- [ ] **Step 3: Write minimal implementation**

In `src/diffsim/adjoint/multiphase.py`, edit `MultiCHForward.__init__` signature (line 343) and body:

```python
    def __init__(self, dm, energy, onsager, kappa, dt=1e-2, order=1,
                 newton_tol=1e-12, newton_max=30, backend=None):
        from .linsolve_backend import ScipyBackend
        self.op = MultiCHDiscrete(dm, energy.M)
        self.M = energy.M
        self.energy = energy
        self.onsager = np.asarray(onsager, np.float64)
        self.kappa = [float(k) for k in kappa]
        self.dt = float(dt)
        self.order = order
        self.newton_tol, self.newton_max = newton_tol, newton_max
        self.backend = backend or ScipyBackend()
        self.t = 0.0
        self.dt_prev = None
        self.steps = []
```

Replace the forward Newton solve (line 402):

```python
            R, J = self.op.assemble(phis, mus, hist_gp, params, want_jac=True)
            dx = self.backend.solve(J, -R)
```

In `MultiCHAdjoint.gradient`, replace the adjoint solve (line 458):

```python
            rhs = np.asarray(dJdx_list[n], np.float64) + pending[n]
            lam = self.fwd.backend.solve_T(J, rhs)
```

(The `from scipy.sparse.linalg import splu` import at line 29 may now be unused in `multiphase.py`; leave it only if still referenced, otherwise remove it.)

- [ ] **Step 4: Run the new test AND the full existing gate**

Run: `.venv/bin/pytest tests/test_multiphase_adjoint.py -v`
Expected: PASS — all 21 original tests **plus** the two new backend tests (23 total). The three-way gate values must be unchanged (ScipyBackend reproduces the exact `splu` path).

- [ ] **Step 5: Commit**

```bash
git add src/diffsim/adjoint/multiphase.py tests/test_multiphase_adjoint.py
git commit -m "feat(orgelmorph-adj): route forward/adjoint solves through LinearBackend (ScipyBackend default, bit-identical gate)"
```

---

## Task 3: scipy↔torch CSR conversion + transpose helper (CPU-testable)

**Files:**
- Modify: `src/diffsim/adjoint/linsolve_backend.py`
- Test: `tests/test_multiphase_adjoint.py`

**Interfaces:**
- Produces: `scipy_to_torch_csr(A, device, torch) -> torch.Tensor` (a `torch.sparse_csr_tensor`, int64 indices, float64 values, on `device`). Pure array plumbing — the only cuDSS-independent piece of `CudssBackend`, so it is unit-tested on CPU torch (available on the Mac).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_multiphase_adjoint.py`:

```python
def test_scipy_to_torch_csr_roundtrip_and_transpose():
    import numpy as np
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_multiphase_adjoint.py::test_scipy_to_torch_csr_roundtrip_and_transpose -v`
Expected: FAIL with `ImportError: cannot import name 'scipy_to_torch_csr'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/diffsim/adjoint/linsolve_backend.py`:

```python
def scipy_to_torch_csr(A, device, torch):
    """scipy sparse -> torch.sparse_csr_tensor (int64 indices, float64 vals) on
    ``device``.  Mirrors the CSR handoff production uses for cuDSS zero-copy
    (physics/multiphase.MultiPhaseStepper.device_csr)."""
    A = A.tocsr()
    crow = torch.as_tensor(A.indptr.astype("int64"), device=device)
    col = torch.as_tensor(A.indices.astype("int64"), device=device)
    val = torch.as_tensor(A.data.astype("float64"), device=device)
    return torch.sparse_csr_tensor(crow, col, val, size=A.shape, device=device)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_multiphase_adjoint.py::test_scipy_to_torch_csr_roundtrip_and_transpose -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/diffsim/adjoint/linsolve_backend.py tests/test_multiphase_adjoint.py
git commit -m "feat(orgelmorph-adj): scipy->torch CSR helper for the cuDSS backend (CPU-tested transpose)"
```

---

## Task 4: CudssBackend (GPU factor+solve; CPU-importable, GPU-gated solve)

**Files:**
- Modify: `src/diffsim/adjoint/linsolve_backend.py`, `src/diffsim/adjoint/__init__.py`
- Test: `tests/test_multiphase_adjoint.py`

**Interfaces:**
- Consumes: `scipy_to_torch_csr` (Task 3), the production cuDSS sequence at `physics/multiphase.py:1934-1984`.
- Produces: `CudssBackend(device="cuda:0")` implementing `solve(A, b)` and `solve_T(A, b)` via cuDSS. Constructible on any host (imports torch lazily); the actual `DirectSolver` factor/solve only runs when a CUDA device is present.

- [ ] **Step 1: Write the failing test** (construction is CPU-testable; the solve is GPU-gated)

Add to `tests/test_multiphase_adjoint.py`:

```python
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
    import numpy as np
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
```

- [ ] **Step 2: Run tests to verify state (on Mac)**

Run: `.venv/bin/pytest tests/test_multiphase_adjoint.py::test_cudss_backend_importable_on_cpu -v`
Expected: FAIL with `ImportError: cannot import name 'CudssBackend'`.
Run: `.venv/bin/pytest tests/test_multiphase_adjoint.py::test_cudss_backend_matches_scipy_on_gpu -v`
Expected: SKIPPED (no CUDA GPU on Mac).

- [ ] **Step 3: Write minimal implementation**

Append to `src/diffsim/adjoint/linsolve_backend.py`:

```python
class CudssBackend(LinearBackend):
    """Factor+solve on a CUDA device via nvmath cuDSS DirectSolver.

    Mirrors physics/multiphase.MultiPhaseStepper._solve (linsolver='cudss'):
    plan once per sparsity pattern, refactorize per call, rebuild the plan if
    nnz flaps.  Assembly stays host-numpy; only the solve is on device.
    Construction is host-safe (torch imported lazily) so the engine + daisy-morph
    wiring unit-test on a CPU box; the DirectSolver path needs a real GPU.
    """

    def __init__(self, device="cuda:0"):
        self.device_str = str(device)
        self._torch = None
        self._device = None
        self._solver = None      # cuDSS DirectSolver, planned once per pattern
        self._nnz = None

    def _lazy(self):
        if self._torch is None:
            import torch
            self._torch = torch
            self._device = torch.device(self.device_str)
        return self._torch

    def _solve_csr(self, A, b):
        torch = self._lazy()
        from nvmath.sparse.advanced import DirectSolver
        At = scipy_to_torch_csr(A, self._device, torch)
        bt = torch.as_tensor(b.astype("float64"), device=self._device)
        # plan once per pattern; rebuild if nnz changes (production nnz guard).
        if self._solver is None or A.nnz != self._nnz:
            self._solver = DirectSolver(At, bt)
            self._solver.plan()
            self._nnz = A.nnz
        else:
            self._solver.reset_operands(a=At, b=bt)
        self._solver.factorize()
        x = self._solver.solve()
        return x.cpu().numpy()

    def solve(self, A, b):
        return self._solve_csr(A.tocsr(), b)

    def solve_T(self, A, b):
        return self._solve_csr(A.T.tocsr(), b)
```

> **Authority note for the implementer:** the exact `DirectSolver`/`plan`/`reset_operands`/`factorize`/`solve` API and the nnz-guard live in `src/diffsim/physics/multiphase.py:1934-1984`. If the nvmath surface differs from the sketch above (e.g. option objects, keyword names), copy production verbatim — it is the proven sequence. This cannot be validated on the Mac; Task 6 validates it on gpubox.

Add `CudssBackend` to the `linsolve_backend` import and `__all__` in `src/diffsim/adjoint/__init__.py`:

```python
from .linsolve_backend import (LinearBackend, ScipyBackend, CudssBackend,
                               scipy_to_torch_csr)
```

- [ ] **Step 4: Run tests (on Mac)**

Run: `.venv/bin/pytest tests/test_multiphase_adjoint.py -k "cudss or scipy_backend or scipy_to_torch" -v`
Expected: `test_cudss_backend_importable_on_cpu` PASS; `test_cudss_backend_matches_scipy_on_gpu` SKIPPED; Task 1/3 tests still PASS.

- [ ] **Step 5: Commit**

```bash
git add src/diffsim/adjoint/linsolve_backend.py src/diffsim/adjoint/__init__.py tests/test_multiphase_adjoint.py
git commit -m "feat(orgelmorph-adj): CudssBackend (device cuDSS solve, host-safe ctor, GPU-gated parity test)"
```

---

## Task 5: daisy-morph device routing (backend by device)

**Files:**
- Modify: `src/daisy_morph/gradients.py:46-65` (`_forward`), `:83-88` (`objective_value`), `:105-119` (`compute`)
- Modify: `src/daisy_morph/core.py:130-139` (call site), `:197-244` (`_run_gradients`)
- Test: `src/daisy_morph/tests/test_gradients.py`

**Interfaces:**
- Consumes: `diffsim.adjoint.CudssBackend`, `ScipyBackend`.
- Produces: `_run_gradients(..., device)` selects the backend; `_grad.compute(..., backend=<obj|None>)` threads it into `MultiCHForward`. On a CUDA `device`, `_run_gradients` calls `doctor.ensure_gpu()` first and stamps `grad_mode="gpu_cudss_fixed_step"`, `gradient_rung="rung2_gpu_cudss"`.

- [ ] **Step 1: Write the failing tests** (CPU routing + GPU-gated selection)

Add to `src/daisy_morph/tests/test_gradients.py`:

```python
def test_gradients_cpu_routing_unchanged():
    # device="cpu" must keep the rung-1 path: no backend object (ScipyBackend
    # default), grad_mode stamp stays the CPU reference string.
    from daisy_morph.core import simulate
    import daisy_morph.doctor as doc
    r = simulate(phi=(0.35, 0.35), chi=(3.5, 1.0, 0.6), device="cpu",
                 level=4, periodic=False, t_end=0.02, dt=5e-4, tstep="bdf1",
                 gradients=["chi"])
    assert r.gradients is not None and "chi_0_1" in r.gradients["chi"]
    assert r.health["grad_mode"] == "cpu_reference_fixed_step"


def _gpu_available():
    try:
        import warp as wp
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_available(), reason="no CUDA GPU (rung-2 gate runs on gpubox)")
def test_gradients_gpu_uses_cudss_and_matches_cpu():
    from daisy_morph.core import simulate
    import numpy as np
    kw = dict(phi=(0.35, 0.35), chi=(3.5, 1.0, 0.6), level=6, periodic=False,
              t_end=0.01, dt=5e-4, tstep="bdf1", gradients=["chi"])
    rc = simulate(device="cpu", **kw)
    rg = simulate(device="cuda:0", **kw)
    assert rg.health["grad_mode"] == "gpu_cudss_fixed_step"
    for k in rc.gradients["chi"]:
        assert np.isclose(rc.gradients["chi"][k], rg.gradients["chi"][k],
                          rtol=1e-8, atol=1e-10)
```

- [ ] **Step 2: Run tests (on Mac)**

Run: `.venv/bin/pytest tests/test_gradients.py::test_gradients_cpu_routing_unchanged -v` (from the daisy-morph dir, its `.venv`)
Expected: FAIL — currently `_run_gradients` sets `grad_mode="cpu_reference_fixed_step"` already, so the assertion on the stamp passes, but the routing signature change hasn't landed; if this test passes as-is it confirms the CPU stamp is preserved. If it errors on an unexpected kwarg, that surfaces a wiring gap. (The GPU test SKIPS on Mac.)

> If `test_gradients_cpu_routing_unchanged` passes immediately (the stamp is already correct), that is fine — it becomes the regression guard proving the device-routing edit did not disturb the CPU path. Proceed to Step 3 to add the GPU branch.

- [ ] **Step 3: Write minimal implementation**

In `src/daisy_morph/gradients.py`, thread `backend` through:

```python
def _forward(M, phi0, chi_m, N_arr, *, level, aspect, periodic, mobility,
             kappa, dt, tstep, t_end, seed, amp, backend=None):
    ...
    fwd = MultiCHForward(
        dm, FHMultiEnergy(np.asarray(chi_m, float), np.asarray(N_arr, float)),
        onsager=onsager, kappa=kappa_list, dt=dt, order=order, backend=backend)
    ...
```

```python
def objective_value(M, phi0, chi_m, N_arr, **kw):
    fwd, kap, _ = _forward(M, phi0, chi_m, N_arr, **kw)   # kw may carry backend
    ...
```

```python
def compute(M, phi0, chi_m, N_arr, groups, *, backend=None, **kw):
    from diffsim.adjoint.multiphase import MultiCHAdjoint
    fwd, kappa_list, mesh = _forward(M, phi0, chi_m, N_arr, backend=backend, **kw)
    ...
```

In `src/daisy_morph/core.py`, pass `device` into the gradients call (edit lines 136-139):

```python
        return _run_gradients(
            list(gradients), M, phi0, chi_m, N_arr, tp=tp, per=per,
            level=level, aspect=aspect, mobility=mobility, kappa=kappa,
            dt=dt, tstep=tstep, t_end=t_end, seed=seed, amp=amp, device=device)
```

Edit `_run_gradients` (line 197) signature + body to select the backend:

```python
def _run_gradients(groups, M, phi0, chi_m, N_arr, *, tp, per, level, aspect,
                   mobility, kappa, dt, tstep, t_end, seed, amp, device="cpu"):
    """gradients= path.  device='cpu' -> rung-1 CPU reference (ScipyBackend);
    a CUDA device -> rung-2 GPU cuDSS backend (fixed-step forward, cuDSS
    factor+solve).  Objective = interfacial (gradient) energy."""
    import platform
    from . import gradients as _grad

    is_gpu = not str(device).startswith("cpu")
    if is_gpu:
        from . import doctor
        doctor.ensure_gpu()
        from diffsim.adjoint import CudssBackend
        backend = CudssBackend(str(device))
        grad_mode, rung = "gpu_cudss_fixed_step", "rung2_gpu_cudss"
    else:
        backend = None
        grad_mode, rung = "cpu_reference_fixed_step", "rung1_cpu_reference"

    t0 = time.perf_counter()
    grads, J, fwd, mesh = _grad.compute(
        M, phi0, chi_m, N_arr, groups, backend=backend, level=level,
        aspect=aspect, periodic=per, mobility=mobility, kappa=kappa, dt=dt,
        tstep=tstep, t_end=t_end, seed=seed, amp=amp)
    fields = _grad.grid_fields(mesh, M, fwd)
    wall = round(time.perf_counter() - t0, 3)
    health = {
        "objective": "interface_energy",
        "objective_value": J,
        "n_steps": len(fwd.steps),
        "n_dofs": int(fwd.op.ndof),
        "grad_mode": grad_mode,
        "wall_time_s": wall,
    }
    record = {
        # ... unchanged provenance keys ...
        "gradient_groups": list(groups),
        "gradient_rung": rung,
        "device": device,
    }
    return SimResult(phi=fields, thermo=tp, health=health, record=record,
                     gradients=grads)
```

(Keep the existing `record` provenance keys; only `gradient_rung` becomes dynamic and `device` is added.)

- [ ] **Step 4: Run tests (on Mac)**

Run (daisy-morph dir): `.venv/bin/pytest tests/test_gradients.py -v`
Expected: `test_gradients_cpu_routing_unchanged` PASS; `test_gradients_gpu_uses_cudss_and_matches_cpu` SKIPPED; existing gradient tests still PASS.

- [ ] **Step 5: Commit** (daisy-morph, branch `feat/gradients`)

```bash
git add src/daisy_morph/gradients.py src/daisy_morph/core.py tests/test_gradients.py
git commit -m "feat: route gradients= to GPU cuDSS backend when device is CUDA (rung-2 wiring, CPU path unchanged)"
```

---

## Task 6: cuDSS parity gate — **runs on gpubox**

**Files:** none new — executes the GPU-gated tests from Tasks 4 and 5 on real hardware.

**Interfaces:**
- Consumes: everything above, shipped to gpubox at the current `diff-orgelmorph` HEAD (DiffSim) and `feat/gradients` HEAD (daisy-morph, editable-installed against the shipped DiffSim).

- [ ] **Step 1: Ship code Mac→gpubox**

Use the remote toolkit (see `scripts/remote/README.md` / `docs/dev/remote-workflow.md`). Verify no uncommitted clobber first (`git status` on gpubox). Sync the `diff-orgelmorph` DiffSim tree and the `feat/gradients` daisy-morph tree.

- [ ] **Step 2: Confirm the GPUs are visible**

On gpubox: `LD_LIBRARY_PATH=/usr/lib/wsl/lib .venv/bin/python -c "import warp as wp; print(wp.get_cuda_device_count())"`
Expected: `2`.

- [ ] **Step 3: Run the DiffSim cuDSS parity test**

On gpubox, DiffSim dir:
`LD_LIBRARY_PATH=/usr/lib/wsl/lib .venv/bin/pytest tests/test_multiphase_adjoint.py::test_cudss_backend_matches_scipy_on_gpu -v`
Expected: PASS (cuDSS solve == scipy `splu` to rtol 1e-9, both `solve` and `solve_T`).

- [ ] **Step 4: Run the daisy-morph GPU gradient parity test**

On gpubox, daisy-morph dir:
`LD_LIBRARY_PATH=/usr/lib/wsl/lib .venv/bin/pytest tests/test_gradients.py::test_gradients_gpu_uses_cudss_and_matches_cpu -v`
Expected: PASS (GPU χ-gradients == CPU rung-1 χ-gradients at level 6, rtol 1e-8). **This is the rung-2 acceptance gate: GPU gradients reproduce the three-way-verified CPU gradients at overlapping resolution.**

- [ ] **Step 5: Record the result**

Append the pass (with the gpubox commit sha it ran at) to a findings note `docs/dev/2026-08-07-rung2-gpu-parity.md`; reconcile any gpubox commits back to master. If parity FAILS, stop — do not proceed to Task 7/8; the cuDSS sequence or transpose handling is wrong and must be fixed against production `_solve` first.

---

## Task 7: 256×128 net-new GPU gradient run — **runs on gpubox**

**Files:** Create `docs/dev/2026-08-07-rung2-256x128-run.md` (the deliverable record).

**Interfaces:**
- Consumes: Task 6 green.

- [ ] **Step 1: Run 256×128 ternary gradients on gpubox**

The committed deliverable resolution. On gpubox, daisy-morph dir, a driver script:

```python
# scratch/rung2_256x128.py  (level=8, aspect=2 -> 256x128 grid; M=2 ternary)
import numpy as np, time
from daisy_morph.core import simulate
t0 = time.perf_counter()
r = simulate(phi=(0.35, 0.35), chi=(3.5, 1.0, 0.6), device="cuda:0",
             level=8, aspect=2, periodic=False, t_end=0.01, dt=5e-4,
             tstep="bdf1", gradients=["chi", "mobility", "kappa"])
print("ndof", r.health["n_dofs"], "grad_mode", r.health["grad_mode"])
print("wall_s", round(time.perf_counter() - t0, 1))
print("chi grads", r.gradients["chi"])
```

Run: `LD_LIBRARY_PATH=/usr/lib/wsl/lib .venv/bin/python scratch/rung2_256x128.py`
Expected: `n_dofs == 131072` (2M×256×128 = 4×32768), `grad_mode == gpu_cudss_fixed_step`, finite gradients for all three groups, wall-time recorded. (This resolution is impractical on CPU `splu` — that gap is the point of rung 2.)

- [ ] **Step 2: Sanity-check the gradients**

χ-gradients should be finite, sign-consistent with the exponential stiffness noted in rung 1 (larger χ → larger interfacial energy → `dJ/dχ_0_1 > 0` for the demixing pair). Record values + wall-time in the run doc.

- [ ] **Step 3: Commit the run record** (reconcile to master)

```bash
git add docs/dev/2026-08-07-rung2-256x128-run.md
git commit -m "docs(orgelmorph-adj): rung-2 256x128 GPU gradient run record (131k DOF, cuDSS)"
```

---

## Task 8: daisy-morph 6 GPU forward tests + DiffSim pin bump — **runs on gpubox**

**Files:** Modify `daisy-morph/pyproject.toml:13` (the DiffSim git-SHA pin).

**Interfaces:**
- Consumes: Task 6 green (DiffSim `diff-orgelmorph` HEAD sha).

- [ ] **Step 1: Run the 6 GPU forward integration tests**

These exercise the **production** `MultiPhaseStepper` forward (they check `r.phi`, not gradients) — the guard that the rung-2 changes did not regress production. On gpubox, daisy-morph dir:
`LD_LIBRARY_PATH=/usr/lib/wsl/lib .venv/bin/pytest tests/test_gpu_integration.py -v`
Expected: 6 passed (`test_ternary_reference_demixes`, `test_ternary_seed_reproducible`, `test_quaternary_smoke`, `test_image_conventions_on_real_run`, `test_noflux_boundary_conditions`, `test_aspect_rectangle`).

- [ ] **Step 2: Bump the DiffSim pin**

Edit `pyproject.toml:13` — replace the 40-char SHA in
`diffsim @ git+ssh://git@github.com/BaskarGS/DiffSim.git@<sha>`
with the `diff-orgelmorph` HEAD sha that passed Task 6 (once that sha is pushed to the BaskarGS/DiffSim remote — the Mac is the GitHub gatekeeper, so push from the Mac after reconciliation).

- [ ] **Step 3: Verify the pinned install resolves and the suite is green**

Reinstall against the bumped pin and run the daisy-morph suite (GPU-marked tests included on gpubox):
`LD_LIBRARY_PATH=/usr/lib/wsl/lib .venv/bin/pytest -v`
Expected: full suite green (the 6 GPU tests now run; CPU tests unchanged). Confirm `r.record["diffsim_commit"]` resolves to the new 40-char sha.

- [ ] **Step 4: Commit the pin bump** (daisy-morph)

```bash
git add pyproject.toml
git commit -m "chore: bump DiffSim pin to rung-2 GPU cuDSS adjoint HEAD"
```

---

## Roadmap: Rung 3 (deferred) — differentiate the production adaptive trajectory

Requested by Baskar 2026-08-07 ("once the low-risk path is done, could we revisit the adaptive time stepping approach?"). **Not in this plan.** After rung 2 lands, the follow-on is the higher-fidelity adjoint that differentiates the *actual* production `MultiPhaseStepper` adaptive-dt trajectory (the originally-specced Approach A): retain+transpose the production converged Jacobian on device, checkpoint the adaptive committed-state trajectory `{xₙ}`, and handle the variable-coefficient BDF2 / step-control history.

Feasibility note for the rung-3 spec: unlike NS (where variable-dt is explicitly *not* wired — see `diffsim-device-compute-orientation`), the **phase-field/multiphase LTE step controller `physics/lte.py` is proven** (`march(adapt="lte")`, PI(D) filter, noise→BDF1 fallback). So the adaptive-trajectory adjoint for multi-CH is materially more tractable than the general NS case. Rung 3 gets its own spec + plan via brainstorming → writing-plans when we return to it.

---

## Self-Review

**Spec coverage (spec §Rung plan rung 2 + §7):**
- "On-device Jᵀ/cuDSS adjoint" → Tasks 1–4 (`CudssBackend.solve_T`). ✓
- "reusing the production converged Jacobian" → **reinterpreted** per the 2026-08-07 decision: reuse the machine-precision-parity rung-1 Jacobian (self-contained numpy assembly), solve on device. Documented in Architecture + Roadmap. The literal production-Jacobian reuse moves to rung 3. ✓ (intentional, user-ratified deviation)
- "committed-state trajectory checkpointing" → already present (`MultiCHForward.steps`); no new work needed at fixed-step (memory point is modest at 131k DOF). Noted. ✓
- "parameter derivatives as device arrays" → **not required** on this path: `dR_dparam` stays host-numpy (cheap); only the linear solve is on device. This is consistent with "swap linear backend only." ✓
- "same gradients= signature" → Task 5 preserves the signature; only internal routing added. ✓
- "matches rung-1 CPU gradients at overlapping resolution" → Task 6 Step 4 (the acceptance gate). ✓
- "the 6 daisy-morph GPU integration tests → bump the DiffSim pin" → Task 8. ✓

**Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N" — each step carries real code or a real command. The one deliberate indirection (nvmath API details) points to an exact authoritative source (`physics/multiphase.py:1934-1984`) because it is not Mac-verifiable; that is a documented constraint, not a placeholder.

**Type consistency:** `LinearBackend.solve/solve_T(A, b)` signatures are identical across `ScipyBackend`/`CudssBackend` and both call sites (`multiphase.py:402`, `:458`). `MultiCHForward(..., backend=None)` matches the `_forward(..., backend=None)` and `compute(..., backend=None)` threading. `grad_mode` strings (`cpu_reference_fixed_step` / `gpu_cudss_fixed_step`) match between `_run_gradients` and the test assertions. `scipy_to_torch_csr(A, device, torch)` argument order matches its call in `CudssBackend._solve_csr`.
