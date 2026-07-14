# Critical-evaluation hardening (release readiness)

**Date:** 2026-07-14
**Source of truth:** `CriticalEvaluations/DiffSim_Current_Codebase_Critical_Evaluation.md`
(gitignored, main checkout). This note records the contained fixes made against
that review, and the two items that need Baskar (CI GPU runner activation; two
deferred lint sweeps).

The evaluation was written against commit `8029917`; this work landed on top of
`b857641` (master). Findings were re-verified against the current tree before
each change.

---

## Track B — code robustness

### B1. Runtime asserts → typed exceptions (eval P0.2)

New module **`src/diffsim/errors.py`** defines a small hierarchy:

| Exception | Base(s) | Use |
|---|---|---|
| `DiffSimError` | `Exception` | package base |
| `ConfigError` | `DiffSimError, ValueError` | invalid user config / option / mesh-order |
| `BackendError` | `DiffSimError, RuntimeError` | slot-map / 32-bit-overflow / unsupported backend path (silent-corruption guards) |
| `SolverError` | `DiffSimError, RuntimeError` | runtime solve failure |
| `ConvergenceError` | `SolverError` | did-not-converge |

**Back-compat is deliberate:** `ConfigError` is a `ValueError`; `SolverError`
(and `ConvergenceError`) are `RuntimeError`, so the film/multiphase
adaptive-timestep reject ladders that catch `RuntimeError` to turn an *expected*
non-convergence into a NaN divergence signal keep working unchanged.

Load-bearing asserts converted (the ones the eval flags as correctness-critical
— they can scatter wrong sparse entries or select wrong physics under
`python -O`, which strips `assert`):

- `assembly/device_assembly.py` — node-graph pattern mode restriction, blockmask
  requirement, `nnz < 2^31` int32-slot overflow, node-pair slot-map validation,
  constraint-aware scatter restriction, CSR slot lookup → `BackendError` /
  `ConfigError`.
- `solvers/linsolve.py` — blockch pair/AC symbolic-pattern validation and the
  blockch-device `nnz < 2^31` guard → `BackendError`; all 19 solver
  non-convergence `RuntimeError`s → `ConvergenceError`; unknown-solver and
  cache-staleness `ValueError`s → `ConfigError`.
- `physics/wodo_film.py` — `tstep` / `noise` / `mob_model` / `D_ratio`
  validation → `ConfigError`; top-surface-nodes mesh check → `ConfigError`;
  device identity-constraint restriction → `BackendError`.
- `physics/multiphase.py` — kernel-factory `bulk` / `mob` / `theta` / aniso
  selectors and the stepper `T_mode` / `tstep` / device-linsolver selectors →
  `ConfigError`.
- `sbm/reference.py`, `sbm/vector.py`, `sbm/poisson.py` — uniform-p assumptions
  in the SBM helper/twin paths → `ConfigError`.

Cheap *impossible-developer* invariants and pure math-invariant checks (e.g.
`indptr[-1] == nnz`, `chi` symmetry) were **left as `assert`** per the eval's
guidance — do not convert wholesale.

### B2. Broad `except` → surface defects (eval P0.3)

The cuDSS / block solve paths in `physics/wodo_film.py` (host + device) and
`physics/multiphase.py` (host + device) caught bare `Exception` and returned an
all-NaN vector so the dt-reject heuristic could react. That also swallowed
programming/environment defects (API rename → `AttributeError`, bad call →
`TypeError`, missing dep → `ImportError`, OOM → `MemoryError`) as if they were a
"physically hard timestep".

`errors.reraise_if_bug(exc)` is now called at the top of each of those four
handlers: it re-raises the programming/environment error set with a real
traceback, and lets genuine numerical failures (singular factor, non-convergence)
fall through to the NaN divergence signal. Happy path and legitimate
divergence-signalling are unchanged.

The best-effort `try: self._cudss.free() except Exception: pass` *cleanup*
blocks were left as-is (cleanup, not the load-bearing solve).

### B3. Structured solver results (eval P2.1)

New **`src/diffsim/solvers/result.py`**: `LinearSolveResult`
(`x/converged/iterations/residual_norm/backend/reason/info`) and
`NonlinearSolveResult`. Introduced **non-invasively**:

- `solvers/newton.py` `NonlinearSolver.solve()` now returns a
  `NonlinearSolveResult`. It defines `__iter__` and `__getitem__` so the historic
  `u, info = solver.solve(u0)` unpack and `result[0]/[1]` indexing keep working,
  and `info["iters"|"fnorm_history"|"converged"]` are preserved.
- `solvers/linsolve.py` `solve_linear(..., return_result=False)`: opt-in only.
  Default still returns the bare host array; `return_result=True` wraps it in a
  `LinearSolveResult`. A returned result is always `converged=True` because the
  backends raise `ConvergenceError` on failure rather than returning an
  unconverged vector.

---

## Track A — release polish

### A1. README claims reconciled (eval P0.4, P1.4)

- The absolute "the full assemble→solve→adjoint loop runs without leaving the
  GPU / host↔device traffic is reserved for output" claim was replaced with a
  path-distinguished, **measured** statement: device CSR assembly 7–11× over the
  host path across 2-D L6–L8 (→ solver-bound), block-CH *device* solve 8.4× over
  host cuDSS on the S2 system; and it names what is *not* yet device-resident
  (generic SciPy brick assembly, some adjoint reductions, verification twins),
  pointing to the device-assembly / block-CH dev notes for the per-path detail.
- The stale "~350-test suite" was corrected to the actual collected count
  (**467 tests / 61 modules**, Jul 2026) with the CPU-vs-GPU split noted.

### A2. Packaging metadata (eval P1.2)

`pyproject.toml` gained full PEP 621 metadata: `description`, `readme`, PEP 639
`license = "Apache-2.0"` + `license-files`, authors/maintainers, keywords,
classifiers, `[project.urls]`. `build-system` bumped to `setuptools>=77` for the
SPDX license expression. Verified: wheel builds, ships the AMGX JSON and film
YAML package data, and METADATA carries the new fields. Version left at `0.0.1`
(a bump is Baskar's call — see recommendations).

### A3. Lint config + a real bug it caught (eval P1.6)

Added `[tool.ruff]` — a deliberately **high-signal, currently-green** rule set
(`E9, F63, F7, F82`; `F821` ignored for Warp codegen) so CI lint is a true
signal, not style noise. This immediately caught a genuine portability bug:
`steppers/leray.py:140` used a star-expression in a subscript (PEP 646, 3.11+)
while `requires-python = ">=3.10"` — a `SyntaxError` on the declared floor.
Fixed by building the index tuple explicitly. `python -m compileall src/` on 3.10
in CI now guards this class of regression.

### A4. CI workflow (eval P0.1) — see handoff below.

---

## CI activation handoff — WHAT BASKAR MUST DO

`.github/workflows/ci.yml` has two jobs.

**`cpu` — fully authored, runs on GitHub-hosted runners, no action needed.**
Matrix Python 3.10 + 3.12: ruff lint, `compileall` (requires-python floor),
public-API/errors import smoke, the Warp-free `tests/test_errors.py`, wheel+sdist
build, AMGX-JSON-in-wheel check, and install-from-wheel smoke. All steps were
run locally and pass.

**`gpu-smoke` — present but GATED OFF.** It targets `runs-on: [self-hosted, gpu]`
and is guarded by `if: ${{ vars.ENABLE_GPU_CI == 'true' }}`, so until enabled it
is *skipped, not failed*. To turn it on:

1. **Register a self-hosted runner** on the GPU box (the one with the A100/RTX):
   GitHub → repo **Settings → Actions → Runners → New self-hosted runner** →
   run the given `./config.sh --url … --token …` and `./run.sh` (install it as a
   service for persistence). This uses a **runner registration token** that
   GitHub generates in that page — it is *not* a PAT and I cannot create it.
2. **Label the runner** `gpu` (add it during `config.sh`, or in the runner
   settings) so `runs-on: [self-hosted, gpu]` matches.
3. **Enable the job:** repo → **Settings → Secrets and variables → Actions →
   Variables → New repository variable** `ENABLE_GPU_CI = true`.
4. Ensure the runner's environment has CUDA 12+, an FP64 GPU, and can
   `pip install -e ".[dev]"` (Warp/Torch/cuDSS). The job runs tiers 1–3 plus one
   adjoint gate (`test_neumann_shape_gradient.py`); expand markers once stable.
5. **Branch protection (recommended):** repo → Settings → Branches → protect
   `master` → require the `cpu` checks (and `gpu-smoke` once the runner is
   reliable) before merge.

No token is stored in the workflow; the runner registration token is entered once
in the GitHub UI at step 1.

---

## Recommendations recorded (not done — need Baskar / broad refactor)

- **Version bump + tag automation (P1.2):** `0.0.1` understates the feature set.
  Consider `0.1.0` and a tag-driven version (`setuptools-scm`). API-affecting, so
  left for Baskar.
- **Deferred lint sweep (P1.6):** ruff `F401` (~18 unused imports in `src/`) and
  `F841` (~17 unused locals) are real polish but need per-item review because
  Warp `@wp.func` codegen injects free symbols — auto-`--fix` is unsafe here.
- **Backend capability matrix (P0.4 / P2.2):** the README now points at the dev
  notes; a single machine-readable capability descriptor per
  physics×solver×adjoint combination (validated before a run) is the fuller fix.
- **Compatibility/lock matrix (P1.3), kernel-cache service (P2.4),
  property-based mesh/constraint tests (P2.5), full-workflow telemetry (P3):**
  larger efforts, unchanged from the eval.

---

## Suite status at hand-off

- `tests/test_errors.py` — 21 passed (Warp-free CPU gate).
- `tests/test_newton_bratu.py` — 3 passed (new `NonlinearSolveResult` return).
- device_assembly / sbm_poisson / sbm_vector / stepper_solvers / s_solvers /
  krylov_dev — 72 passed, 1 skipped.
- multiphase / wodo_film / sbm_neumann / sbm_gap_fills / ternary_ch — (see the
  commit's verification log).
- `ruff check src/ tests/` — clean. `compileall src/` — clean. Wheel builds with
  AMGX/film package data and full metadata.

## Merge checklist

- [ ] `ruff check src/ tests/` green
- [ ] `pytest tests/test_errors.py` green (Warp-free)
- [ ] affected GPU suites green on the runner (device_assembly, multiphase*,
      wodo_film, sbm_*, stepper/s_solvers, newton_bratu)
- [ ] wheel builds; AMGX JSON present; `import diffsim` from wheel OK
- [ ] enable `gpu-smoke` per the handoff, then add both CI checks to branch
      protection on `master`
