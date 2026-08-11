# While-waiting crystallization extensions — GPU cuDSS (2M+K) + multi-snapshot recovery

**Date:** 2026-08-08
**Branch:** `feat/crystal-while-waiting` (off master `dc19496`, after ①+② merged)
**Context:** MD φ+ψ data is pending from a collaborator (blocks ③ ingest). Full 2-D coupling (②b) and χ(ψ) (①b) are deferred until the data decides which is needed. These two pieces are **data-independent** and valuable regardless of which extension wins.

## Goal

1. **GPU-validate** the crystallization adjoint (2M+K coupled block) through the existing cuDSS backend — the rung-2 move already done for K=0, re-applied to the bigger block.
2. **Lift the single-snapshot identifiability limit** ② found: a reusable **multi-snapshot** recovery harness (φ+ψ field-L2 across observed times) with a **structure-factor descriptor hook** — the format-agnostic core of ③ / M6 Plan B.

## Designed-for / out of scope

- **In:** GPU cuDSS parity for the coupled engine; a multi-snapshot recovery harness fitting φ+ψ at several time-snapshots, with an optional descriptor term; a structure-factor `S(k)` reference descriptor.
- **Out:** full 2-D coupling `g_k(φ,ψ)` (②b — pending MD), χ(ψ) (①b), real MD ingest/format (③, blocked on data), θ orientation. No changes to the verified ①/② engine.

## #1 — GPU cuDSS for the 2M+K block (validation)

Zero engine change: `CrystalCHForward(dm, energy, crystallizable, …, backend=None)` already resolves `backend or ScipyBackend()` and routes the forward Newton solve through `backend.solve(J, -R)` and the adjoint reverse sweep through `backend.solve_T(J, rhs)`. `CudssBackend` (from `linsolve_backend.py`) already carries the rung-2 plan-fresh-per-solve fix and handles `solve_T`. cuDSS factors/solves the `2M+K` CSR exactly like the `2M` one.

- **Gate (CPU):** an explicit `backend=ScipyBackend()` produces identical results to the default (plumbing honored).
- **Gate (GPU, skip-gated on Mac):** crystal forward+adjoint gradients for a set of params via `CudssBackend` match `ScipyBackend` to ~1e-8 — mirrors the K=0 rung-2 parity test.
- **Validation:** run the GPU test on gpubox at a coupled 256×128-class system; record DOF + wall-time.

## #2 — Multi-snapshot recovery harness

`CrystalCHAdjoint.gradient(dJdx_list, names)` already accepts a **per-step** cotangent list; single-snapshot recovery (② Task 5) is the special case that only populates `dJdx[-1]`. Multi-snapshot is the general case.

- **Observable:** φ,ψ at K selected snapshot steps `S = {s_1,…,s_K}` from `fwd.steps`.
- **Loss:** `J = Σ_{s∈S} w_s · 0.5‖(φ,ψ)(t_s) − target(t_s)‖²` (+ optional descriptor term). Cotangent: `dJdx[s][2i::blk] = w_s·(φ_i(t_s) − φ*_i(t_s))`, `dJdx[s][2M+j::blk] = w_s·(ψ_j(t_s) − ψ*_j(t_s))`; weights `w_s` default uniform.
- **Descriptor hook:** an optional `descriptor` term `D(fields)` added in field space — `dJdx[s] += ∂(λ_D·‖D(fields(t_s)) − D*‖²)/∂fields`. Ship `structure_factor(field_2d)` (radially-averaged `|FFT|²`, `S(k)`) + its field-space gradient as the reference descriptor; default `descriptor=None`.
- **Optimize:** gradient-descent `cpl_*` (optionally `basis_*`) via the hand adjoint (production path), re-evaluating loss at final params (no off-by-one).
- **Gate:** on a synthetic trajectory, multi-snapshot recovers a higher-ψ mode (e.g. `cpl_0_2`) to a tolerance that single-snapshot (② Task 5) demonstrably misses — the identifiability lift. Plus a descriptor unit test (`S(k)` gradient vs finite differences).

## File structure

- `src/diffsim/adjoint/crystal_recovery.py` — **new.** `structure_factor` (+ grad), `nodal_to_grid`, `recover_multisnapshot`.
- `tests/test_crystal_recovery.py` — **new.** descriptor grad-vs-FD; multi-vs-single identifiability gate.
- `tests/test_crystallization_multi.py` — **modify.** GPU cuDSS parity test (skip-gated) + CPU backend-honored smoke.
- `examples/crystal_learn_multisnapshot.py` — **new.** thin driver.

## Verification

- #1: CPU smoke (backend honored) always runs; GPU parity runs on gpubox (~1e-8 cuDSS vs scipy).
- #2: descriptor gradient three-way-style vs central FD (<1e-6); multi-snapshot identifiability gate (recovers `cpl_0_2` within tol where single-snapshot fails); reuse the ② `crystal_learn_from_synthetic` helpers where possible.
