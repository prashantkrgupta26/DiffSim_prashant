# Track A: R2b Saddle-Preconditioner Campaign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether a scalable iterative solve exists for the VMS-stabilized (u,p) saddle with the two-sided Nitsche shell block — iteration ladder 143k→~10M DOF — against the binding kill-gate (iteration growth ≤ ~2× across the 1M→10M decade, or Track A reports negative-with-measurements).

**Architecture:** Per the approved spec (`docs/superpowers/specs/2026-07-27-100m-two-track-design.md`, Track A). New module `src/diffsim/solvers/saddle_precond.py` provides preconditioner-apply closures for `fgmres_dev` (which already accepts `apply_dev`); `solve_linear` gains routed backends `"fgmres_bdiag"` (baseline) and `"fgmres_pcd"` (PCD Schur); the drivers' existing `MONO_SOLVER` knob reaches them with zero driver changes. An iteration-ladder harness runs the corrected-physics configs at four sizes and records iterations/step.

**Tech Stack:** `fgmres_dev` (src/diffsim/solvers/fgmres_dev.py — device FGMRES, `apply_dev` preconditioner hook), `cg_dev`/Jacobi inners, existing mass/poisson assembly for the pressure-space operators, the monolithic drivers + `MONO_SOLVER` routing, AMGX wrapper (velocity-block inner only — NEVER the outer on the saddle).

## Global Constraints

- Branch `track-a-r2b` off current master; repo cd prefix on every bash command; `.venv/bin/python`; box runs via the gpubox toolkit; nova for big-memory legs if needed.
- Default paths untouched: new solvers are OPT-IN backends; every existing gate stays green (tests/test_p2r1a_thin_plate_flow.py, test_p2r1c_thin_plate_flow_3d.py, test_p2r0_projection_sbm.py, test_device_assembly.py).
- AMGX only ever as a velocity-block INNER (it diverges on the raw saddle — documented). GPU parity/iteration claims from measured runs, ledgered; negative results first-class.
- Iteration-count telemetry mandatory on every new backend (`return_result=True` plumbing or the backend's info dict — surfaced to the driver's verbose print and the ladder table).
- KILL-GATE (spec, binding): iteration growth ≤ ~2× across 1M→10M or the campaign concludes negative; the plan budget is THIS plan — no extension without Baskar.
- Commits: explicit paths; trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task A1: `saddle_precond.py` — block-diagonal baseline + `fgmres_bdiag` backend

**Files:**
- Read FIRST: `src/diffsim/solvers/fgmres_dev.py` (:298 signature — `fgmres_dev(A_matvec, b_dev, apply_dev, N, device, ...)`; how `tests/test_fgmres_dev.py` builds `A_matvec` from a CSR and a Jacobi `apply_dev` — copy that wiring), `src/diffsim/solvers/linsolve.py` (the `"fused"` backend at :1100 shows CSROperator construction + backend dispatch pattern; add the new backends alongside)
- Create: `src/diffsim/solvers/saddle_precond.py`
- Modify: `src/diffsim/solvers/linsolve.py` (two new `solver ==` branches)
- Test: `tests/test_saddle_precond.py`

**Interfaces:**
- Produces: `make_bdiag_apply(A, ndof, device) -> apply_dev` — block-diagonal preconditioner closure: velocity dofs preconditioned by the velocity-block diagonal (Jacobi), pressure dofs by the pressure-block diagonal with a safe floor (`max(|d_p|, eps_p)` — the PSPG-stabilized p-p diagonal is nonzero but may be small; floor at 1e-12·max|d|). `solve_linear(..., solver="fgmres_bdiag", sym=False)` routes: CSR → device matvec (CSROperator, reuse the fused backend's construction) + `make_bdiag_apply` → `fgmres_dev`; raises ConvergenceError with the iteration count in the message on failure; on success returns host x (and iteration count via `return_result=True` path — read how LinearSolveResult carries backend telemetry and populate `iterations`).

- [ ] **Step 1: failing test** — `tests/test_saddle_precond.py`: build a small REAL saddle system (assemble via the 2-D driver's machinery at level=4: reuse `_build_shell` + `assemble_linear_ns` + `Af_c` + the LIL surgery exactly as `run_flow_past` does for ONE step — factor this as a test helper `_one_step_system()` returning (Acsr, b, x_splu)); assert `solve_linear(Acsr, b, solver="fgmres_bdiag", sym=False, device="cpu", tol=1e-10)` matches `x_splu` to rtol 1e-8, and that the result object reports `iterations > 0`. RED: unknown solver.
- [ ] **Step 2: implement** per Produces. VERIFY-FIRST: fgmres_dev's convergence criterion and restart length (choose restart=60 default, expose via kwarg); the CPU Warp device works for the test (all our device machinery runs on "cpu").
- [ ] **Step 3: gates** — new test + `tests/test_p2r1a_thin_plate_flow.py -q` green (no default-path drift).
- [ ] **Step 4: commit** — `feat(solvers): block-diagonal saddle FGMRES baseline (fgmres_bdiag)` + trailer.

---

### Task A2: iteration-ladder harness

**Files:**
- Create: `tests/gpu_saddle_ladder.py` (not pytest-collected)
- Test: `tests/test_saddle_ladder_cpu.py` (tiny wiring gate)

**Interfaces:**
- Produces: `run_ladder_point(tag, solver, dim, nsteps=5, **cfg) -> dict` with `{tag, solver, dofs, iters_per_step (list), iters_mean, s_per_step, converged}` — marches nsteps of the MONOLITHIC driver (2-D `run_flow_past` or 3-D `run_flow_past_3d` per `dim`) with `mono_solver=solver`, capturing per-solve iteration counts. Iteration capture: VERIFY-FIRST how to surface counts from the routed solve inside the march — if the driver's routed call can't return telemetry without modification, add an optional `solver_stats=None` list kwarg to both drivers (append per-step iteration count when the backend provides it; default None = untouched behavior; parity-test it like every prior knob).
- Ladder points (the spec's sizes): 2-D corrected-units r9 (~0.3M dof) and r11 (~1M+); 3-D L6 uniform (~1.1M) and base-L7/band-r9 (9.24M — the WP0-verified mesh). `__main__`: for SOLVERS (env, default "fgmres_bdiag") × points, run and print the table with flush; `SADDLE-LADDER-OK` sentinel; npz per point.
- CPU gate: one tiny point (level=4, 2 steps, fgmres_bdiag, cpu) — finite, iterations recorded, parity of the `solver_stats=None` default.

Steps: TDD; gates incl. both drivers' existing tests; commit `feat(solvers): saddle iteration-ladder harness` + trailer.

---

### Task A3: PCD Schur preconditioner (`fgmres_pcd`)

**Files:**
- Read FIRST: how the 2-D driver assembles operators (`assemble_linear_ns` internals for the blocks), `src/diffsim/physics/poisson.py` + mass assembly (the machinery for pressure-space Ap/Mp), the blockch two-factor structure (`src/diffsim/solvers/linsolve.py` `_blockch_pairs` docstring :312+) as the shape reference
- Modify: `src/diffsim/solvers/saddle_precond.py`, `src/diffsim/solvers/linsolve.py`
- Test: `tests/test_saddle_precond.py` (extend)

**Interfaces:**
- Produces: `make_pcd_apply(A, meta, device) -> apply_dev` implementing PCD: `P⁻¹ = [F⁻¹, 0; 0, S⁻¹]` with `S⁻¹ ≈ Mp⁻¹ Fp Ap⁻¹` (pressure convection-diffusion). `meta = {"ndof", "dim", "Mp", "Ap", "Fp"}` — the pressure-space operators assembled ONCE per mesh from dm (scalar mass, scalar stiffness scaled by ν... VERIFY the PCD scalings against the σ-dominated transient regime: Fp = σ·Mp + ν·Ap + N_p(a) — at production dt the σ·Mp term dominates, making S⁻¹ ≈ σ·Ap⁻¹-like; implement Fp WITHOUT the convection term first (σMp+νAp — the "Cahouet–Chabard-like" transient form, matching the docs' physical-time finding that block preconditioning works in physical time), note the omission honestly, add N_p only if the ladder demands it). Inners: `cg_dev` Jacobi-CG on F (velocity block, tol loose 1e-2, maxiter cap), on Ap and Mp (SPD). `solve_linear(..., solver="fgmres_pcd", cache=, cache_key=)` — meta built lazily per cache_key from dm... the backend can't see dm; ROUTE: the driver builds meta once (a helper `build_pcd_meta(dm, nu, sigma)` in saddle_precond) and passes via `cache[("pcd_meta", key)]` — VERIFY the least-invasive plumbing and document it.
- Gates: same one-step-system test vs splu (rtol 1e-8) + an ITERATION-COUNT assertion: pcd iterations < bdiag iterations on the same system (the point of PCD) — if NOT, that is a finding to report, not to hide (assert with a clear message, xfail-style recording allowed only with the number printed).

Steps: TDD; regression gates; commit `feat(solvers): PCD Schur saddle preconditioner (fgmres_pcd)` + trailer.

---

### Task A4: field-split with AMGX velocity inner (conditional)

Run ONLY if A3's ladder (A5 interim) shows PCD iteration growth > the gate at ≤1M: replace the F-inner Jacobi-CG with the AMGX wrapper on the VELOCITY BLOCK (extract the velocity sub-CSR — ndof-strided rows/cols; AMGX is valid there, it is not a saddle). Same test pattern; commit `feat(solvers): AMGX velocity-inner field-split option` + trailer. If A3 meets the gate at 1M, SKIP (YAGNI) and note in the ledger.

---

### Task A5: the ladder campaign (GPU) + kill-gate verdict

- Run the ladder (A2 harness) for bdiag + pcd (+ A4 if built) at all four points: 2-D on gpubox; 3-D L6 on gpubox; the 9.24M point on gpubox if memory allows (matrix host CSR ~16GB + solver working set — VERIFY; else nova Grace/GH200 via sbatch mirroring the WP0 pattern).
- Record per point: iterations/step (mean + range over 5 steps), s/step, memory, convergence failures (honest rows).
- **Apply the kill-gate verbatim:** growth ≤ ~2× across 1M→10M for ANY candidate ⇒ Track A viable (record which candidate + the projection to 100M with stated assumptions). Otherwise ⇒ NEGATIVE verdict with the full table.
- Recording: dated section in `docs/dev/thinshell-gpu-runbook.md` + a new `docs/dev/2026-07-27-r2b-saddle-campaign.md` (the campaign report). Commit docs.

---

### Task A6: ledger + track verdict handoff

Update the two-track ledger section with the Track A verdict; if negative, the engine defaults to Track B per the spec (no further Track A work without Baskar). Full CPU regression sweep before closing the branch; final whole-branch review; merge decision to Baskar.

## Self-Review

Spec coverage: candidates 1-3 → A3/A4/A1; ladder+sizes → A2/A5; kill-gate verbatim → A5/A6; telemetry mandate → A1/A2; AMGX-inner-only honored → A4. Placeholders: verify-first notes name their oracles (fgmres_dev test wiring, blockch docstring, PCD plumbing decision documented). Types: `run_ladder_point`/backend names consistent across tasks.
