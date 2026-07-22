# P2-R2b.1 — Host-Orchestrated Block-Preconditioned Monolithic 3-D Solve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `splu` direct solve in the monolithic 3-D SBM-NS Picard march with a **block-preconditioned FGMRES** built on the EXISTING `BlockAMGPreconditioner` (Cahouet–Chabard Schur + AMG-on-F, AMGX inner solves), host-orchestrated (outer Krylov on host, AMGX inner on GPU), to march past the splu scaling wall (splu STALLED at level-5 / 143k DOF overnight).

**Architecture:** Wire a new `solver="blockamgx"` branch into `solvers/linsolve.py::solve_linear` that constructs `BlockAMGPreconditioner` from the assembled monolithic CSR `A` plus geometry-static `Kp`/`Mp_diag`/`sigma`/`nu`/`dir_rows` passed through a `cache`-carried `meta` dict (mirroring the existing `blockch` `meta` pattern), then runs `solve_block_preconditioned` and returns the host solution. A `solver=` knob threads this into the monolithic march (`tests/p2r0_task10_sphere_derisk.py::monolithic_cd` and `steppers/linearized.py::LinearizedMonolithicStepper`), assembling `Kp`/`Mp_diag` ONCE (geometry-static) and passing them via `meta`. A gpubox validation script marches the sphere at Re=100 at levels {4,5,6} and reports Cd, wall-time/step, and FGMRES + AMGX inner iteration counts to show the outer count is ~mesh-independent. Full device residency (device outer FGMRES) is R2b.2, out of scope.

**Tech Stack:** Python / NumPy / SciPy (host outer Krylov via `scipy.sparse.linalg.gmres`); pyamgx / AMGX (GPU inner V-cycles, gpubox only); pytest; JSON baselines. Default `solver` path (`splu`) unchanged and bit-for-bit.

## Global Constraints

- Branch: **`p2-r2a`** (this work lands on `p2-r2a`, continuing the R2 series).
- REPO-IDENTITY GUARD: before EVERY commit, verify `git rev-parse --abbrev-ref HEAD` == `p2-r2a`; abort the commit if not.
- Agents NEVER push; the supervisor pushes.
- Every commit message ends with the trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- All AMGX / device runs (numerics gate, validation gate) run on **gpubox**. pyamgx is NOT importable on the Mac dev box (confirmed: `ModuleNotFoundError: No module named 'pyamgx'`), so any test that constructs `BlockAMGPreconditioner` is gpubox-gated behind an import skip; a host-only structural fallback (Task 2) is the off-box correctness check.
- The DEFAULT `solver` path stays `splu` everywhere (`monolithic_cd` default, `LinearizedMonolithicStepper(solver="splu")`); `solve_linear`'s existing branches are untouched. Only a NEW `solver == "blockamgx"` branch is added.
- Absolute paths only. Working directory: `/Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim`.
- DOF layout contract (the preconditioner ASSUMES it): node-major interleaved `[u_0..u_{dim-1}, p]` per free node, `ndof = dim+1`, over `n_nodes` FREE nodes. `Kp` and `Mp_diag` are the scalar pressure operators on the SAME free-node ordering.
- The SBM block `Af_c` is added into `A` BEFORE the solve (both in `monolithic_cd` and via the stepper's assembly) so the preconditioner sees the FULL saddle including the SBM contribution.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/diffsim/solvers/linsolve.py` | Add `solver == "blockamgx"` branch to `solve_linear`: read `("blockamgx_meta", cache_key)` from cache, construct `BlockAMGPreconditioner`, run `solve_block_preconditioned`, record `("blockamgx_iters", cache_key)`, return host x (Task 1) |
| Create | `tests/test_blockamgx_linsolve.py` | Unit tests: (a) gpubox numerics gate — `solver="blockamgx"` matches `solver="splu"` on the SAME small saddle to Krylov tol; (b) host-only structural fallback (no AMGX) validating the Schur/velocity block split + Cahouet–Chabard action via loose direct inner solves (Task 1 + Task 2) |
| Modify | `src/diffsim/solvers/block_precond.py` | Add an optional `inner="amgx"|"direct"` selector to `_AMGXCycle`/`BlockAMGPreconditioner` so the off-box structural test can run WITHOUT AMGX (a CPU direct/loose inner); store `dir_rows`; AMGX default path unchanged (Task 2) |
| Modify | `tests/p2r0_task10_sphere_derisk.py` | Add a `solver` kwarg to `monolithic_cd` (default `"splu"`); when `"blockamgx"`, assemble `Kp`/`Mp_diag` ONCE and pass `Kp`/`Mp_diag`/`sigma`/`nu`/`dir_rows` via `cache[("blockamgx_meta", key)]` each step; call `solve_linear` instead of `splu` (Task 3) |
| Modify | `src/diffsim/steppers/linearized.py` | Route `LinearizedMonolithicStepper` through `solve_linear(solver="blockamgx")` when selected: assemble `Kp`/`Mp_diag` once in `__init__`, publish `meta` into `self._solver_cache` each step (Task 4) |
| Create | `tests/p2r2b1_blockamgx_convergence.py` | gpubox validation gate: march the monolithic sphere Re=100 with `solver="blockamgx"` at levels {4,5,6}; assert L4 Cd reproduces splu (0.964); REACH L5/L6 (splu stalled); report wall/step + FGMRES outer + AMGX inner counts; assert outer count ~mesh-independent (Task 5) |
| Create | `tests/baselines/p2r2b1_blockamgx_convergence.json` | Per-level Cd / wall / iteration-count record written by the validation script (Task 5) |

---

## Verified Code Facts (read from source — do not re-derive)

- `BlockAMGPreconditioner.__init__(self, A, n_nodes, ndof, Kp, Mp_diag, sigma, nu, dir_rows=None)` (`src/diffsim/solvers/block_precond.py:38-59`). It builds `u_ids`/`p_ids` from an interleaved node-major layout, extracts `F = A[u_ids][:,u_ids]`, `G = A[u_ids][:,p_ids]`, stores `Kp`, `Mp_diag`, `sigma`, `nu`, and creates `self._amg_F = _AMGXCycle(F, sym=False, cycles=1)` and `self._amg_Kp = _AMGXCycle(Kp, sym=True, cycles=3)`.
- `.apply(r)` (`:61-71`): `z_p = sigma * amg_Kp.solve(r_p) + nu * (r_p / Mp_diag)`; `z_u = amg_F.solve(r_u - G @ z_p)`; scatter back.
- `.as_linear_operator()` (`:73-75`): `LinearOperator((n*ndof, n*ndof), matvec=self.apply)`.
- `solve_block_preconditioned(A, b, pre, tol=1e-9, maxiter=200)` (`:123-137`): right-preconditioned `scipy.sparse.linalg.gmres(A.tocsr(), b, M=pre.as_linear_operator(), rtol=tol, atol=1e-13, maxiter=maxiter, restart=50, callback=..., callback_type="pr_norm")`. Returns `(x, iters)`; raises `RuntimeError` on `info != 0`.
- **`dir_rows` is accepted by the constructor but NEVER stored or used** (`grep` shows references only in the signature and docstring). Confirm before relying on it (Task 2 addresses this).
- `_AMGXCycle.__init__` (`:86-112`) creates its OWN `pyamgx.Resources().create_simple(...)` per instance. `BlockAMGPreconditioner` therefore holds TWO independent Resources (F + Kp). `amgx.py:82` warns "Multiple live Resources sets in one process ... segfault inside AMGX." This is the top risk (see Global Gotchas).
- `solve_linear(A, b, solver="splu", sym=False, tol=1e-10, maxiter=40000, device="cuda:0", cache=None, cache_key=None, return_result=False)` (`src/diffsim/solvers/linsolve.py:1034`). The `blockch` branch reads `meta = (cache or {}).get(("blockch_meta", cache_key))` and records `cache[("blockch_iters", cache_key)] = (outer, inner)` — the pattern to mirror.
- `assemble_csr(dm)` (`src/diffsim/assembly/operators.py:459-465`) returns the constrained scalar stiffness `T^T K T` on FREE nodes — this is `Kp` (the same operator `LerayProjectionStepper.K_p` uses at `steppers/leray.py:137`).
- The scalar consistent mass `M_p = T^T M T` is built by `LerayProjectionStepper._mass_matrix` (`steppers/leray.py:159-178`); `Mp_diag = np.asarray(Mp.diagonal())`. This method is a standalone GP loop; the monolithic march (which has NO Leray stepper) must replicate it or import a shared helper.
- `monolithic_cd(fx, alpha, dt, max_steps, rate_tol)` (`tests/p2r0_task10_sphere_derisk.py:138-201`) assembles `A = (assemble_linear_ns(...) + Af_c)`, overwrites strong Dirichlet rows to identity, pins pressure at the far-corner node `pin = argmax(coords.sum(1))*ndof + dim`, then `x = splu(A.tocsr().tocsc()).solve(b)`. `nfree = T.shape[1]`, `sigma = 1.0/dt`. `strong` = `np.where(fx["strong_mask"])[0]`; the monolithic velocity Dirichlet rows are `strong[k]*ndof + c` for `c in range(dim)`.
- `LinearizedMonolithicStepper` (`src/diffsim/steppers/linearized.py:32`) already carries `self.solver`, `self._solver_cache = {}`, `self.dir_rows`, `self.pin_row = 0*ndof + dim`, and calls `solve_linear(A, b, solver=self.solver, ..., cache=self._solver_cache)` at `:238-240`.
- pyamgx is unavailable on the Mac (`ModuleNotFoundError`); the AMGX-touching tests are gpubox-only.

---

## Task 1: Wire `solver="blockamgx"` into `solve_linear`

**Files:**
- Modify: `src/diffsim/solvers/linsolve.py` (add a branch just before the final `ConfigError`, after the `cudss` branch at `:1366`)
- Test: `tests/test_blockamgx_linsolve.py`

**Interfaces:**
- Consumes: `BlockAMGPreconditioner`, `solve_block_preconditioned` from `diffsim.solvers.block_precond`.
- Produces: the contract `solve_linear(A, b, solver="blockamgx", tol=..., cache=CACHE, cache_key=KEY)` where the caller MUST have set `CACHE[("blockamgx_meta", KEY)] = {"n_nodes": int, "ndof": int, "Kp": csr, "Mp_diag": ndarray, "sigma": float, "nu": float, "dir_rows": ndarray|None}` before the call. On return, `CACHE[("blockamgx_iters", KEY)] = (outer_iters,)`.

- [ ] **Step 1: Write the failing gpubox numerics-gate test**

Add to `tests/test_blockamgx_linsolve.py`:

```python
import numpy as np
import scipy.sparse as sp
import pytest

pyamgx = pytest.importorskip("pyamgx")  # gpubox-only; skips on the Mac


def _small_saddle(n_nodes=40, dim=2, seed=0):
    """A small node-major interleaved (u,p) saddle system with a mass-
    dominated velocity block (sigma large) and a Cahouet-Chabard-friendly
    pressure block, so the block preconditioner is a good match."""
    rng = np.random.default_rng(seed)
    ndof = dim + 1
    N = n_nodes * ndof
    sigma, nu = 50.0, 0.02
    # scalar pressure stiffness Kp (SPD, pinned last row) and mass diag
    Kp = sp.diags([2.1] * n_nodes) - sp.eye(n_nodes, k=1) * 0.5 \
        - sp.eye(n_nodes, k=-1) * 0.5
    Kp = sp.csr_matrix(Kp)
    Mp_diag = np.full(n_nodes, 0.7)
    # assemble a monolithic A on interleaved layout
    idx = np.arange(N).reshape(n_nodes, ndof)
    u_ids = idx[:, :dim].ravel()
    p_ids = idx[:, dim].ravel()
    A = sp.lil_matrix((N, N))
    F = sigma * sp.eye(len(u_ids)) + 0.05 * sp.random(
        len(u_ids), len(u_ids), density=0.02, random_state=rng)
    G = 0.3 * sp.random(len(u_ids), n_nodes, density=0.05, random_state=rng)
    A[np.ix_(u_ids, u_ids)] = F.toarray()
    A[np.ix_(u_ids, p_ids)] = G.toarray()
    A[np.ix_(p_ids, u_ids)] = -G.T.toarray()
    A[np.ix_(p_ids, p_ids)] = (nu * sp.diags(Mp_diag)).toarray()
    A = A.tocsr()
    b = rng.standard_normal(N)
    return dict(A=A, b=b, n_nodes=n_nodes, ndof=ndof, Kp=Kp,
                Mp_diag=Mp_diag, sigma=sigma, nu=nu)


def test_blockamgx_matches_splu_on_small_saddle():
    from diffsim.solvers.linsolve import solve_linear
    s = _small_saddle()
    x_direct = solve_linear(s["A"], s["b"], solver="splu")
    cache, key = {}, "t1"
    cache[("blockamgx_meta", key)] = dict(
        n_nodes=s["n_nodes"], ndof=s["ndof"], Kp=s["Kp"],
        Mp_diag=s["Mp_diag"], sigma=s["sigma"], nu=s["nu"], dir_rows=None)
    x_block = solve_linear(s["A"], s["b"], solver="blockamgx", tol=1e-8,
                           cache=cache, cache_key=key)
    rel = np.linalg.norm(x_block - x_direct) / np.linalg.norm(x_direct)
    assert rel < 1e-6, f"blockamgx vs splu rel err {rel:.2e}"
    outer, = cache[("blockamgx_iters", key)]
    assert 0 < outer < 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_blockamgx_linsolve.py::test_blockamgx_matches_splu_on_small_saddle -v`
Expected (on the Mac): SKIPPED (`pyamgx` not importable). On gpubox before the branch exists: FAIL with `ConfigError: unknown solver 'blockamgx'`.

- [ ] **Step 3: Add the `blockamgx` branch to `solve_linear`**

In `src/diffsim/solvers/linsolve.py`, immediately AFTER the `cudss` branch (after line 1366) and BEFORE `from ..errors import ConfigError`:

```python
    if solver == "blockamgx":
        # P2-R2b.1: host-orchestrated block-preconditioned FGMRES for the
        # monolithic SBM-NS saddle. The Cahouet-Chabard Schur + AMG-on-F
        # preconditioner (block_precond.BlockAMGPreconditioner) runs AMGX
        # inner V-cycles on the GPU; the outer flexible GMRES is on the
        # host (few 10s of iterations, so the per-iter sync amortizes).
        # meta via cache: ("blockamgx_meta", cache_key) =
        #   {"n_nodes","ndof","Kp","Mp_diag","sigma","nu","dir_rows"}.
        from .block_precond import (BlockAMGPreconditioner,
                                    solve_block_preconditioned)
        meta = (cache or {}).get(("blockamgx_meta", cache_key))
        if meta is None:
            raise ValueError("blockamgx requires ('blockamgx_meta', "
                             "cache_key) = {'n_nodes','ndof','Kp',"
                             "'Mp_diag','sigma','nu','dir_rows'} in cache")
        pre = BlockAMGPreconditioner(
            A, meta["n_nodes"], meta["ndof"], meta["Kp"], meta["Mp_diag"],
            meta["sigma"], meta["nu"], dir_rows=meta.get("dir_rows"))
        x, iters = solve_block_preconditioned(A, b, pre, tol=tol,
                                              maxiter=200)
        if cache is not None and cache_key is not None:
            cache[("blockamgx_iters", cache_key)] = (iters,)
        return x
```

Note: `solve_linear` already skips the cache staleness fingerprint only for `blockch`; `blockamgx` uploads a NEW `A` (values change each Picard step) but shares one `cache_key`, so add `"blockamgx"` to the fingerprint-exempt tuple at `linsolve.py:1062`:

```python
    if cache is not None and cache_key is not None \
            and solver not in ("blockch", "blockamgx"):
```

- [ ] **Step 4: Run test to verify it passes (gpubox)**

Run (on gpubox): `.venv/bin/python -m pytest tests/test_blockamgx_linsolve.py::test_blockamgx_matches_splu_on_small_saddle -v`
Expected: PASS (`rel err < 1e-6`, outer iters in range).

- [ ] **Step 5: Commit**

```bash
test "$(git rev-parse --abbrev-ref HEAD)" = "p2-r2a" || { echo "WRONG BRANCH"; exit 1; }
git add src/diffsim/solvers/linsolve.py tests/test_blockamgx_linsolve.py
git commit -m "$(cat <<'MSG'
feat(p2-r2b1): wire solver=blockamgx into solve_linear

Host-orchestrated block-preconditioned FGMRES on the monolithic SBM-NS
saddle via BlockAMGPreconditioner + solve_block_preconditioned; meta
(Kp/Mp_diag/sigma/nu/dir_rows) passed through cache like blockch.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
MSG
)"
```

---

## Task 2: Off-box structural fallback + honor `dir_rows`

**Files:**
- Modify: `src/diffsim/solvers/block_precond.py`
- Test: `tests/test_blockamgx_linsolve.py`

**Interfaces:**
- Consumes: `BlockAMGPreconditioner` from Task 1's contract.
- Produces: `BlockAMGPreconditioner(..., inner="amgx"|"direct")` (default `"amgx"` — unchanged behavior) and `_AMGXCycle` replaced by a `_CycleInner` selector so the Schur/velocity block STRUCTURE is testable on the Mac with a CPU loose-direct inner. `dir_rows`, when given, is stored and the identity Dirichlet rows are left untouched by the preconditioner (`z[dir_rows] = r[dir_rows]` passthrough).

- [ ] **Step 1: Write the failing host-only structural test**

Add to `tests/test_blockamgx_linsolve.py` (this one does NOT importorskip pyamgx):

```python
def test_blockamgx_structure_host_direct_inner():
    """Off-box (no AMGX) check that the Cahouet-Chabard Schur + velocity-
    block split reproduces splu when the inner cycles are exact CPU direct
    solves. Validates the block extraction + apply arithmetic, not AMGX."""
    from diffsim.solvers.block_precond import (BlockAMGPreconditioner,
                                               solve_block_preconditioned)
    import scipy.sparse as sp
    import numpy as np
    from scipy.sparse.linalg import splu
    s = _small_saddle(seed=3)
    pre = BlockAMGPreconditioner(
        s["A"], s["n_nodes"], s["ndof"], s["Kp"], s["Mp_diag"],
        s["sigma"], s["nu"], dir_rows=None, inner="direct")
    x, iters = solve_block_preconditioned(s["A"], s["b"], pre, tol=1e-8,
                                          maxiter=200)
    x_direct = splu(s["A"].tocsc()).solve(s["b"])
    rel = np.linalg.norm(x - x_direct) / np.linalg.norm(x_direct)
    assert rel < 1e-6, f"structural block precond rel err {rel:.2e}"
    assert 0 < iters < 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_blockamgx_linsolve.py::test_blockamgx_structure_host_direct_inner -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'inner'`.

- [ ] **Step 3: Add the `inner=` selector and store `dir_rows`**

In `src/diffsim/solvers/block_precond.py`, change the constructor signature and inner-cycle creation:

```python
    def __init__(self, A, n_nodes, ndof, Kp, Mp_diag, sigma, nu,
                 dir_rows=None, inner="amgx"):
        self.n, self.ndof = n_nodes, ndof
        dim = ndof - 1
        idx = np.arange(n_nodes * ndof).reshape(n_nodes, ndof)
        self.u_ids = idx[:, :dim].ravel()
        self.p_ids = idx[:, dim].ravel()
        self.dir_rows = (None if dir_rows is None
                         else np.asarray(dir_rows, np.int64))
        Ac = A.tocsr()
        self.F = Ac[self.u_ids][:, self.u_ids].tocsr()
        self.G = Ac[self.u_ids][:, self.p_ids].tocsr()
        self.sigma, self.nu = sigma, nu
        self.Kp = Kp.tocsr()
        self.Mp_diag = np.asarray(Mp_diag)
        Cycle = _AMGXCycle if inner == "amgx" else _DirectCycle
        self._amg_F = Cycle(self.F, sym=False, cycles=1)
        self._amg_Kp = Cycle(self.Kp, sym=True, cycles=3)
```

Update `apply` to passthrough the Dirichlet identity rows (they are exact-solved by construction, so the preconditioner should be identity on them):

```python
    def apply(self, r):
        r = np.asarray(r)
        r_u, r_p = r[self.u_ids], r[self.p_ids]
        z_p = (self.sigma * self._amg_Kp.solve(r_p, tol=1e-3, iters=8)
               + self.nu * (r_p / self.Mp_diag))
        z_u = self._amg_F.solve(r_u - self.G @ z_p, tol=1e-2, iters=2)
        z = np.empty_like(r)
        z[self.u_ids], z[self.p_ids] = z_u, z_p
        if self.dir_rows is not None:
            z[self.dir_rows] = r[self.dir_rows]
        return z
```

Add the CPU direct inner class (alongside `_AMGXCycle`):

```python
class _DirectCycle:
    """OFF-BOX structural stand-in for _AMGXCycle: a loose CPU solve of
    the same block used as a preconditioner apply. NOT a GPU path — it
    exists so the Schur/velocity split (block extraction + apply
    arithmetic) is testable where AMGX is unavailable (the Mac dev box).
    Same .solve(b, **_ignored) contract as _AMGXCycle."""

    def __init__(self, A, sym, cycles=1):
        from scipy.sparse.linalg import splu
        self._lu = splu(A.tocsc())

    def solve(self, b, **_ignored):
        return self._lu.solve(np.ascontiguousarray(b, np.float64))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_blockamgx_linsolve.py::test_blockamgx_structure_host_direct_inner -v`
Expected: PASS (`rel err < 1e-6`). This runs on the Mac (no AMGX).

- [ ] **Step 5: Commit**

```bash
test "$(git rev-parse --abbrev-ref HEAD)" = "p2-r2a" || { echo "WRONG BRANCH"; exit 1; }
git add src/diffsim/solvers/block_precond.py tests/test_blockamgx_linsolve.py
git commit -m "$(cat <<'MSG'
feat(p2-r2b1): block precond inner selector + dir_rows passthrough

inner="direct" gives an off-box CPU-inner structural check of the
Cahouet-Chabard Schur / velocity-block split (AMGX default unchanged);
dir_rows is now stored and identity-passthrough in apply().

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
MSG
)"
```

---

## Task 3: Thread `solver=` into `monolithic_cd`

**Files:**
- Modify: `tests/p2r0_task10_sphere_derisk.py:138-201` (`monolithic_cd`)
- Test: `tests/test_blockamgx_linsolve.py`

**Interfaces:**
- Consumes: `solve_linear(solver="blockamgx", cache=..., cache_key=...)` from Task 1; `assemble_csr` from `diffsim.assembly.operators`.
- Produces: `monolithic_cd(fx, alpha, dt, max_steps, rate_tol, solver="splu")`. When `solver != "splu"` it assembles `Kp = assemble_csr(dm)` and `Mp_diag` ONCE, pins the SAME pressure row in `Kp` (identity row) as the monolithic pin, and passes `meta` via a per-call cache.

- [ ] **Step 1: Write the failing parity test (host-direct inner)**

Add to `tests/test_blockamgx_linsolve.py` — this exercises the real fixture on a tiny mesh with the CPU-direct inner (no AMGX), so it runs on the Mac. It monkeypatches the inner selector so `monolithic_cd`'s `blockamgx` path uses `_DirectCycle`:

```python
def test_monolithic_cd_blockamgx_matches_splu_level2(monkeypatch):
    import diffsim.solvers.block_precond as bp
    # force the off-box CPU inner so this runs without AMGX
    orig = bp.BlockAMGPreconditioner.__init__

    def patched(self, A, n_nodes, ndof, Kp, Mp_diag, sigma, nu,
                dir_rows=None, inner="amgx"):
        orig(self, A, n_nodes, ndof, Kp, Mp_diag, sigma, nu,
             dir_rows=dir_rows, inner="direct")
    monkeypatch.setattr(bp.BlockAMGPreconditioner, "__init__", patched)

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from p2r0_task10_sphere_derisk import build_sphere_3d, monolithic_cd
    fx = build_sphere_3d("cpu", level=2, Re=100.0)
    r_splu = monolithic_cd(fx, alpha=100.0, dt=0.05, max_steps=3,
                           rate_tol=1e-9, solver="splu")
    r_blk = monolithic_cd(fx, alpha=100.0, dt=0.05, max_steps=3,
                          rate_tol=1e-9, solver="blockamgx")
    assert abs(r_splu["cd"] - r_blk["cd"]) < 1e-4, \
        f"Cd splu {r_splu['cd']:.5f} vs blockamgx {r_blk['cd']:.5f}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_blockamgx_linsolve.py::test_monolithic_cd_blockamgx_matches_splu_level2 -v`
Expected: FAIL with `TypeError: monolithic_cd() got an unexpected keyword argument 'solver'`.

- [ ] **Step 3: Add the `solver` kwarg + `Kp`/`Mp_diag` assembly**

In `tests/p2r0_task10_sphere_derisk.py`, change the signature and the solve. Add near the top of the file (module imports):

```python
from diffsim.assembly.operators import assemble_csr
from diffsim.solvers.linsolve import solve_linear
```

Change the signature (line 138):

```python
def monolithic_cd(fx, alpha, dt, max_steps, rate_tol, solver="splu"):
```

After `Af_c`/`bf_c` are built and BEFORE the step loop (after line 154), assemble the geometry-static pressure operators once when needed:

```python
    Kp = Mp_diag = None
    solver_cache = {}
    if solver != "splu":
        Kp = assemble_csr(dm).tolil()          # scalar T^T K T on free nodes
        # scalar consistent mass diagonal (same GP loop as leray._mass_matrix)
        rows_m, cols_m, vals_m = [], [], []
        for pv, bb in dm.bins.items():
            tb = dm.tables_by_p[pv]
            h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
            jac = (h / 2.0) ** dim
            Me = np.einsum("qa,qb,q->ab", tb.N, tb.N, tb.w)
            conn = dm.mesh.conn_of[pv].astype(np.int64)
            ne, nbf = conn.shape
            Mee = Me[None, :, :] * jac[:, None, None]
            rows_m.append(np.repeat(conn, nbf, axis=1).ravel())
            cols_m.append(np.tile(conn, (1, nbf)).ravel())
            vals_m.append(Mee.ravel())
        Mp_full = sp.coo_matrix(
            (np.concatenate(vals_m),
             (np.concatenate(rows_m), np.concatenate(cols_m))),
            shape=(dm.n_nodes, dm.n_nodes)).tocsr()
        Mp = (T @ Mp_full @ T.T).tocsr()        # NOTE T is free->full here
```

IMPORTANT — resolve the `T` orientation before implementing: in `monolithic_cd`, `T = cons.T.tocsr()` and `nfree = T.shape[1]`, so `T` is `(n_full, n_free)` and the CONSTRAINED (free-node) operator is `T.T @ M_full @ T` (see `assemble_csr` / `_mass_matrix`, both use `T.T @ K @ T`). Use the SAME `T.T @ ... @ T` form as `assemble_csr` for both `Kp` and `Mp`. `assemble_csr(dm)` already returns the constrained `Kp` directly, so DO NOT re-apply `T`. Correct the mass block to:

```python
        Mp = (T.T @ Mp_full @ T).tocsr()
        Kp = assemble_csr(dm).tolil()
        # pin the SAME pressure node as the monolithic pin (row + col),
        # so Kp is nonsingular; pin node id in free-node space:
        pin_node = int(np.argmax(coords.sum(1)))
        Kp.rows[pin_node] = [pin_node]
        Kp.data[pin_node] = [1.0]
        Kp = Kp.tocsr()
        Mp_diag = np.asarray(Mp.diagonal())
        Mp_diag[Mp_diag == 0] = 1.0
        dir_rows = np.concatenate(
            [strong * ndof + c for c in range(dim)]).astype(np.int64)
```

Replace the solve (line 190) with a dispatch:

```python
        if solver == "splu":
            x = splu(A.tocsr().tocsc()).solve(b)
        else:
            cache = solver_cache
            cache[("blockamgx_meta", "mono")] = dict(
                n_nodes=nfree, ndof=ndof, Kp=Kp, Mp_diag=Mp_diag,
                sigma=sigma, nu=nu, dir_rows=dir_rows)
            x = solve_linear(A.tocsr(), b, solver=solver, tol=1e-8,
                             cache=cache, cache_key="mono")
```

(`A` at this point is the LIL after strong-row + pin overwrite; call `.tocsr()` for both paths. `nu = fx["nu"]` is already in scope via the unpack at line 143.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_blockamgx_linsolve.py::test_monolithic_cd_blockamgx_matches_splu_level2 -v`
Expected: PASS (`|Cd_splu - Cd_blockamgx| < 1e-4`).

- [ ] **Step 5: Commit**

```bash
test "$(git rev-parse --abbrev-ref HEAD)" = "p2-r2a" || { echo "WRONG BRANCH"; exit 1; }
git add tests/p2r0_task10_sphere_derisk.py tests/test_blockamgx_linsolve.py
git commit -m "$(cat <<'MSG'
feat(p2-r2b1): solver= knob on monolithic_cd (blockamgx path)

monolithic_cd assembles Kp (assemble_csr) + Mp_diag once, pins the same
pressure node in Kp, and routes each Picard solve through
solve_linear(solver="blockamgx") via cache meta; default stays splu.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
MSG
)"
```

---

## Task 4: Thread the block solve into `LinearizedMonolithicStepper`

**Files:**
- Modify: `src/diffsim/steppers/linearized.py`
- Test: `tests/test_blockamgx_linsolve.py`

**Interfaces:**
- Consumes: Task 1's `solve_linear(solver="blockamgx")` contract; `assemble_csr`.
- Produces: `LinearizedMonolithicStepper(..., solver="blockamgx")` publishes `self._solver_cache[("blockamgx_meta", <key>)]` before each `solve_linear` call. `Kp`/`Mp_diag` are assembled ONCE in `__init__` (geometry-static). `sigma` is refreshed per step (BDF/dt dependent).

- [ ] **Step 1: Write the failing test (host-direct inner, tiny 2-D dm)**

Add to `tests/test_blockamgx_linsolve.py`. Use the smallest `dm` fixture the repo's stepper tests use (mirror `tests/test_linearized.py`'s fixture; if it builds a 2-D box `dm`, reuse it). The test marches one step with `solver="blockamgx"` (monkeypatched to the direct inner) and compares to `solver="splu"`:

```python
def test_linearized_stepper_blockamgx_one_step(monkeypatch):
    import diffsim.solvers.block_precond as bp
    orig = bp.BlockAMGPreconditioner.__init__

    def patched(self, A, n_nodes, ndof, Kp, Mp_diag, sigma, nu,
                dir_rows=None, inner="amgx"):
        orig(self, A, n_nodes, ndof, Kp, Mp_diag, sigma, nu,
             dir_rows=dir_rows, inner="direct")
    monkeypatch.setattr(bp.BlockAMGPreconditioner, "__init__", patched)
    # build the SAME small dm the existing linearized-stepper test uses;
    # copy that fixture helper here (see tests/test_linearized.py).
    from diffsim.steppers.linearized import LinearizedMonolithicStepper
    # ... construct dm, nu, dt, f_fn, g_fn exactly as test_linearized.py ...
    def run(solver):
        st = LinearizedMonolithicStepper(dm, nu, dt, f_fn, g_fn,
                                         solver=solver)
        st.set_initial(lambda c: np.zeros((len(c), dm.dim)))
        return st.step()
    x_splu = run("splu")
    x_blk = run("blockamgx")
    rel = np.linalg.norm(x_blk - x_splu) / max(np.linalg.norm(x_splu), 1e-30)
    assert rel < 1e-5, f"stepper blockamgx vs splu rel {rel:.2e}"
```

(Before writing this, READ `tests/test_linearized.py` to copy its exact `dm`/`nu`/`dt`/`f_fn`/`g_fn` construction — do not invent a fixture.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_blockamgx_linsolve.py::test_linearized_stepper_blockamgx_one_step -v`
Expected: FAIL — the `blockamgx` branch of `solve_linear` raises `ValueError("blockamgx requires ('blockamgx_meta', cache_key) ...")` because the stepper does not publish `meta` yet.

- [ ] **Step 3: Assemble `Kp`/`Mp_diag` in `__init__` and publish `meta` per step**

In `src/diffsim/steppers/linearized.py::__init__`, after `self.xq = gauss_points(...)` (line 93), add (guarded so the default splu path pays nothing):

```python
        self._blockamgx = (self.solver == "blockamgx")
        if self._blockamgx:
            from ..assembly.operators import assemble_csr
            import scipy.sparse as _sp
            Kp = assemble_csr(dm).tolil()
            # pin the SAME pressure node as the monolithic pin (node 0)
            pin_node = 0
            Kp.rows[pin_node] = [pin_node]
            Kp.data[pin_node] = [1.0]
            self._Kp = Kp.tocsr()
            T = dm.constraints.T.tocsr()
            rows_m, cols_m, vals_m = [], [], []
            for pv, b in dm.bins.items():
                tb = dm.tables_by_p[pv]
                h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
                jac = (h / 2.0) ** dm.dim
                Me = np.einsum("qa,qb,q->ab", tb.N, tb.N, tb.w)
                conn = dm.mesh.conn_of[pv].astype(np.int64)
                ne, nbf = conn.shape
                Mee = Me[None, :, :] * jac[:, None, None]
                rows_m.append(np.repeat(conn, nbf, axis=1).ravel())
                cols_m.append(np.tile(conn, (1, nbf)).ravel())
                vals_m.append(Mee.ravel())
            Mp_full = _sp.coo_matrix(
                (np.concatenate(vals_m),
                 (np.concatenate(rows_m), np.concatenate(cols_m))),
                shape=(dm.n_nodes, dm.n_nodes)).tocsr()
            Mp = (T.T @ Mp_full @ T).tocsr()
            md = np.asarray(Mp.diagonal())
            md[md == 0] = 1.0
            self._Mp_diag = md
            # velocity Dirichlet rows + pin row are the identity rows
            self._blockamgx_dir = np.concatenate(
                [self.dir_rows, [self.pin_row]]).astype(np.int64)
```

Then, in the HOST branch of `step()`, immediately before the `solve_linear` call (line 237), publish `meta` when the block solver is active:

```python
            from ..solvers.linsolve import solve_linear
            if self._blockamgx:
                self._solver_cache[("blockamgx_meta", "lin")] = dict(
                    n_nodes=self.n_free, ndof=self.ndof, Kp=self._Kp,
                    Mp_diag=self._Mp_diag, sigma=sigma, nu=self.nu,
                    dir_rows=self._blockamgx_dir)
                x = solve_linear(A, b, solver=self.solver, sym=False,
                                 device=self.dm.device,
                                 cache=self._solver_cache,
                                 cache_key="lin", tol=1e-8)
            else:
                x = solve_linear(A, b, solver=self.solver, sym=False,
                                 device=self.dm.device,
                                 cache=self._solver_cache)
```

(Do the SAME publish in `_step_device`'s host-CSR fallback at line 369-371 if `blockamgx` must work under `use_device_assembly=True`; for R2b.1 the host path is sufficient — note this in the plan and leave the device-assembly + blockamgx combo for R2b.2.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_blockamgx_linsolve.py::test_linearized_stepper_blockamgx_one_step -v`
Expected: PASS (`rel < 1e-5`).

- [ ] **Step 5: Commit**

```bash
test "$(git rev-parse --abbrev-ref HEAD)" = "p2-r2a" || { echo "WRONG BRANCH"; exit 1; }
git add src/diffsim/steppers/linearized.py tests/test_blockamgx_linsolve.py
git commit -m "$(cat <<'MSG'
feat(p2-r2b1): LinearizedMonolithicStepper blockamgx path

Assemble Kp (pinned) + Mp_diag once in __init__; publish blockamgx meta
into the solver cache each step; default splu path unchanged.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
MSG
)"
```

---

## Task 5: R2b.1 validation gate (gpubox)

**Files:**
- Create: `tests/p2r2b1_blockamgx_convergence.py`
- Create: `tests/baselines/p2r2b1_blockamgx_convergence.json` (written by the script)

**Interfaces:**
- Consumes: `build_sphere_3d`, `monolithic_cd(solver="blockamgx")`, `R`, `U_IN` from `tests/p2r0_task10_sphere_derisk.py`; the `("blockamgx_iters", "mono")` cache record.
- Produces: a runnable gpubox script mirroring `tests/p2r2c_monolithic_sphere_convergence.py`. Env: `LEVELS` (default `"4,5,6"`), `RE` (100), `STEPS` (80), `ALPHA` (100).

- [ ] **Step 1: Extend `monolithic_cd` to return iteration counts**

`monolithic_cd` currently returns `dict(cd=..., steps=...)`. Add the FGMRES outer + AMGX inner telemetry so the validation script can report the scalability signature. In `tests/p2r0_task10_sphere_derisk.py`, inside the step loop AFTER the `solve_linear` call in the `blockamgx` branch, capture:

```python
            if solver != "splu":
                rec = cache.get(("blockamgx_iters", "mono"))
                if rec is not None:
                    last_outer = rec[0]
```

and return `dict(cd=cd, steps=steps, outer_iters=last_outer)` (initialize `last_outer = None` before the loop; splu path returns `outer_iters=None`).

- [ ] **Step 2: Write the validation script (no separate failing test — it IS the gate)**

Create `tests/p2r2b1_blockamgx_convergence.py`:

```python
"""P2-R2b.1 — host-orchestrated blockamgx monolithic sphere convergence.

Marches the monolithic SBM-NS sphere at Re=100 with solver="blockamgx"
(Cahouet-Chabard Schur + AMG-on-F, AMGX inner V-cycles, host FGMRES outer)
across mesh levels. GATE:
  (a) L4 Cd reproduces the splu reference (0.964, tests/baselines/
      p2r2c_monolithic_sphere_convergence.json);
  (b) REACHES L5 and L6 where splu STALLED (records wall/step + FGMRES
      outer + AMGX inner iteration counts);
  (c) the FGMRES outer count is ~mesh-independent (the scalability
      signature of a good block preconditioner).

Runs on gpubox (AMGX required). Env: LEVELS (default "4,5,6"), RE (100),
STEPS (80), ALPHA (100).

    LEVELS=4,5,6 .venv/bin/python tests/p2r2b1_blockamgx_convergence.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from p2r0_task10_sphere_derisk import build_sphere_3d, monolithic_cd, R

LEVELS = [int(x) for x in os.environ.get("LEVELS", "4,5,6").split(",")]
RE = float(os.environ.get("RE", "100"))
STEPS = int(os.environ.get("STEPS", "80"))
ALPHA = float(os.environ.get("ALPHA", "100"))
DT = 0.05
RATE_TOL = 1e-4
L4_SPLU_CD = 0.964


def main():
    print(f"[r2b1] blockamgx monolithic sphere  Re={RE} dt={DT} "
          f"alpha={ALPHA} steps<= {STEPS} levels={LEVELS}", flush=True)
    rows = []
    for level in LEVELS:
        t0 = time.time()
        fx = build_sphere_3d("cuda:0" if os.environ.get("DIFFSIM_CUDA")
                             else "cpu", level=level, Re=RE)
        n_free = int(len(fx["coords"]))
        n_dof = n_free * fx["ndof"]
        print(f"\n[r2b1] level {level}: n_free={n_free} n_dof={n_dof}",
              flush=True)
        res = monolithic_cd(fx, ALPHA, DT, STEPS, RATE_TOL,
                            solver="blockamgx")
        wall = time.time() - t0
        row = dict(level=level, n_free=n_free, n_dof=n_dof,
                   cd=float(res["cd"]), steps=int(res["steps"]),
                   outer_iters=res.get("outer_iters"),
                   wall_s=round(wall, 1),
                   wall_per_step=round(wall / max(res["steps"], 1), 2))
        rows.append(row)
        print(f"[r2b1] level {level}: Cd={row['cd']:+.4f} "
              f"outer_iters={row['outer_iters']} steps={row['steps']} "
              f"wall/step={row['wall_per_step']}s", flush=True)

    ok = rows
    print("\n[r2b1] ===== VALIDATION SUMMARY =====", flush=True)
    l4 = next((r for r in ok if r["level"] == 4), None)
    if l4 is not None:
        drift = abs(l4["cd"] - L4_SPLU_CD)
        print(f"[r2b1] (a) L4 Cd={l4['cd']:+.4f} vs splu {L4_SPLU_CD} "
              f"(|drift|={drift:.4f}) -> "
              f"{'PASS' if drift < 0.02 else 'FAIL'}", flush=True)
    reached = [r["level"] for r in ok if r["level"] in (5, 6)]
    print(f"[r2b1] (b) reached levels {reached} (splu stalled at L5) -> "
          f"{'PASS' if set(reached) >= {5, 6} else 'PARTIAL'}", flush=True)
    outer = [r["outer_iters"] for r in ok if r["outer_iters"] is not None]
    if len(outer) >= 2:
        spread = max(outer) / max(min(outer), 1)
        print(f"[r2b1] (c) outer iters {outer} (max/min={spread:.2f}) -> "
              f"{'mesh-independent' if spread < 2.0 else 'GROWING'}",
              flush=True)

    out = os.path.join(os.path.dirname(__file__), "baselines",
                       "p2r2b1_blockamgx_convergence.json")
    with open(out, "w") as fh:
        json.dump(dict(config=dict(Re=RE, dt=DT, alpha=ALPHA, steps=STEPS,
                                   levels=LEVELS, l4_splu_cd=L4_SPLU_CD),
                       rows=rows), fh, indent=2)
    print(f"[r2b1] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Dry-run the script structure on the Mac (import + arg parse only)**

Run: `.venv/bin/python -c "import ast; ast.parse(open('tests/p2r2b1_blockamgx_convergence.py').read()); print('parse OK')"`
Expected: `parse OK`. (The full march requires AMGX; it runs on gpubox.)

- [ ] **Step 4: Run the gate on gpubox**

Run (gpubox): `LEVELS=4,5,6 .venv/bin/python tests/p2r2b1_blockamgx_convergence.py`
Expected: (a) L4 Cd within 0.02 of 0.964; (b) L5 and L6 both complete (no stall / no OOM); (c) outer FGMRES iteration count roughly flat across levels (max/min < 2). Baseline JSON written.

- [ ] **Step 5: Commit**

```bash
test "$(git rev-parse --abbrev-ref HEAD)" = "p2-r2a" || { echo "WRONG BRANCH"; exit 1; }
git add tests/p2r2b1_blockamgx_convergence.py tests/p2r0_task10_sphere_derisk.py tests/baselines/p2r2b1_blockamgx_convergence.json
git commit -m "$(cat <<'MSG'
feat(p2-r2b1): blockamgx monolithic sphere convergence gate (gpubox)

Marches Re=100 sphere at L{4,5,6} with solver=blockamgx: reproduces L4
splu Cd, reaches L5/L6 (splu stalled), reports FGMRES outer + AMGX inner
counts to show mesh-independent outer iterations.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
MSG
)"
```

---

## Global Gotchas (read before implementing ANY task)

1. **AMGX multiple-Resources segfault (TOP RISK).** `_AMGXCycle.__init__` (`block_precond.py:103`) creates its OWN `pyamgx.Resources().create_simple(...)` per instance, and `BlockAMGPreconditioner` builds TWO (F + Kp). `amgx.py:82` explicitly warns: "Multiple live Resources sets in one process ... segfault inside AMGX." The block preconditioner has NOT been run end-to-end on gpubox (its module docstring says "v1 measured NOT yet effective — kept as the harness"). **The implementer MUST verify on gpubox that two coexisting `_AMGXCycle` Resources (F nonsym + Kp SPD) do not segfault BEFORE building anything on top.** If they do, the fix is to route both blocks through ONE shared Resources (refactor `_AMGXCycle` to accept a shared `rsc`, created once in `BlockAMGPreconditioner.__init__`), mirroring `amgx_solve`'s singleton discipline. Also note `BlockAMGPreconditioner`'s F/Kp Resources and the singleton in `amgx_solve` are DIFFERENT Resources — do not mix a `solver="amgx"` sub-solve into the same process as a live block preconditioner without checking.

2. **`solve_block_preconditioned` uses `scipy gmres`, not a flexible GMRES.** The preconditioner apply is NONLINEAR (AMGX inner cycles are not a fixed linear operator across iterations — and even with fixed cycles, the residual-based restart interacts). Right-preconditioned standard GMRES assumes a FIXED M. This is why the module docstring calls it "flexible GMRES/BiCGStab" but the code calls `gmres`. For R2b.1 the fixed-cycle inner (`cycles=1`/`3`, `tolerance=0.0`, `max_iters` capped) makes M a FIXED linear operator, so standard GMRES is valid. **Do NOT loosen the inner to a convergence-checked solve** (that makes M vary per apply and breaks standard GMRES — use FGMRES/`lgmres` if you ever do). The numerics gate (Task 1) is the guard.

3. **DOF-layout assumption.** `BlockAMGPreconditioner` hard-codes interleaved node-major `[u_0..u_{dim-1}, p]` via `idx = arange(n*ndof).reshape(n, ndof)`. The monolithic `A` from `assemble_linear_ns` + `Af_c` IS in this layout (`T_vec = kron(T, I_ndof)`), and `Kp`/`Mp_diag` are SCALAR operators indexed by free node — pass them on the free-node ordering, NOT the interleaved one. Getting `n_nodes` wrong (passing `n*ndof` instead of `n`) silently corrupts the block extraction.

4. **SBM block must be in `A` before the solve.** In `monolithic_cd`, `A = (assemble_linear_ns(...) + Af_c)` (line 178) — the SBM Nitsche block `Af_c` is folded in BEFORE the solve, so the preconditioner's `F = A[u_ids][:,u_ids]` already includes the SBM velocity coupling. This is correct and required: the preconditioner must see the FULL saddle. Do NOT construct the preconditioner from a pre-SBM `A`.

5. **Pressure-pin interaction.** The monolithic pin overwrites row `pin = argmax(coords.sum(1))*ndof + dim` to identity (line 186-189). `Kp` (the Schur operator) is a STANDALONE scalar Laplacian and is SINGULAR (pure-Neumann) unless the SAME pressure node is pinned in `Kp` too — Task 3/Task 4 pin `pin_node` in `Kp`. If `Kp` is left singular, AMGX PCG on it will not converge / gives a garbage Schur action. The pinned monolithic pressure row is one of the identity `dir_rows`; with the Task 2 `dir_rows` passthrough, the preconditioner is identity on it (consistent with the exact identity row in `A`). Verify the pin node id is the SAME in both the monolithic `A` and `Kp` (free-node index `argmax(coords.sum(1))` in `monolithic_cd`; node `0` in the stepper — confirm the stepper's `pin_row = 0*ndof + dim` corresponds to `Kp` node `0`).

6. **`nu` and `sigma` sourcing.** In `monolithic_cd`, `sigma = 1.0/dt` (line 169) and `nu = fx["nu"]`. In the stepper, `sigma = b0/dt` (BDF-dependent, refreshed per step) and `nu = self.nu`. The Cahouet–Chabard Schur `S~^{-1} = sigma*Kp^{-1} + nu*Mp^{-1}` needs BOTH; `sigma` MUST be refreshed in `meta` every step (BDF order changes it during bootstrap), `Kp`/`Mp_diag` are geometry-static (assemble once).

7. **pyamgx unavailable on Mac.** Every test that actually constructs a real AMGX `_AMGXCycle` uses `pytest.importorskip("pyamgx")` and runs only on gpubox. The off-box coverage is the `inner="direct"` structural path (Tasks 2-4) which validates the block arithmetic without AMGX. Do not let the gpubox-only test block local CI.

---

## Self-Review

**1. Spec coverage.**
- "wire `solver="blockamgx"` into `solve_linear`" → Task 1. ✓
- "CPU-runnable unit test asserting blockamgx matches splu ... gpubox-gated ... host-only fallback preconditioner check" → Task 1 (gpubox numerics gate, `importorskip`) + Task 2 (`inner="direct"` off-box structural check). ✓
- "thread the block solve into the monolithic march — a `solver` knob on `monolithic_cd` (and/or `LinearizedMonolithicStepper`) ... assembles Kp/Mp_diag once ... default splu" → Task 3 (`monolithic_cd`) + Task 4 (stepper). ✓
- "R2b.1 validation gate (gpubox) `tests/p2r2b1_blockamgx_convergence.py` ... levels {4,5,6} ... reproduces L4 Cd 0.964 ... reaches L5/L6 ... outer count mesh-independent ... mirror p2r2c" → Task 5. ✓
- "Global constraints / gotchas: AMGX singleton Resources caveat; DOF-layout; SBM block in A; pin-row interaction" → Global Gotchas 1,3,4,5. ✓
- "Header: branch p2-r2a, REPO-IDENTITY GUARD, agents never push, Co-Authored-By trailer, gpubox, default solver unchanged" → Global Constraints + every commit step. ✓

**2. Placeholder scan.** No "TBD/TODO/handle edge cases". Two deliberate READ-THE-SOURCE notes: Task 4 Step 1 ("copy `tests/test_linearized.py`'s exact dm fixture") and Task 3 Step 3 (the `T` orientation resolution). These are not placeholders — they point at exact source to read and give the corrected code (`T.T @ ... @ T`), because inventing a `dm` fixture or guessing `T`'s orientation would be wrong. Flagged, not vague.

**3. Type/name consistency.** `meta` key is `("blockamgx_meta", cache_key)` and iters record `("blockamgx_iters", cache_key)` consistently across Tasks 1,3,4,5. `meta` dict keys `{n_nodes, ndof, Kp, Mp_diag, sigma, nu, dir_rows}` match the `BlockAMGPreconditioner` constructor arg names exactly. `inner="amgx"|"direct"` consistent across Tasks 2,3,4. `solve_block_preconditioned` returns `(x, iters)` — used as `x, iters = ...` and recorded as `(iters,)` (1-tuple) consistently; Task 5 reads `rec[0]`. `monolithic_cd` return dict gains `outer_iters` (Task 5 Step 1) — consistent with the script's `res.get("outer_iters")`.
