# HANDOFF — make the monolithic block-preconditioner SCALE (L3→L5→L6+)

**For:** a fresh agent (Fable5) picking this up cold, working overnight.
**Date:** 2026-07-22. **Branch:** `r2b1` (off `master` @ `3afa487`). **Do NOT push.**
**Context doc:** `docs/dev/2026-07-22-p2-r2a-projection-3d-findings.md` (why we're on the
monolithic engine at all) and the ledger `.superpowers/sdd/progress.md` (full blow-by-blow).

---

## 0. TL;DR — the one problem to solve

The AMGX block-preconditioner (`BlockAMGPreconditioner`) makes FGMRES **converge on a tiny
L3 saddle (rel 1.9e-8) but CLIFF-FAILS at L5 (143k DOF)** — 2000+ FGMRES iterations, no
convergence — and this does NOT respond to strengthening the F solve, swapping the Schur, or
raising the GMRES restart. **A cliff (not gradual iteration-count growth) means the failure is
STRUCTURAL / scale-specific, not a weak-preconditioner tuning problem.** Your job: find and fix
the cliff so the block-preconditioned monolithic march converges at L5 and L6 (1.09M DOF), where
direct solvers (splu on CPU, cuDSS on GPU) run out of memory.

**FIRST THING TO DO: instrument the outer FGMRES to log the true residual per iteration.**
Everything so far was diagnosed blind (inferring from AMGX timing spam). You cannot debug a
convergence cliff without seeing the convergence curve. Add a `scipy.sparse.linalg.gmres`
`callback=` (with `callback_type="pr_norm"`) that logs the preconditioned residual each
iteration; print it in `solve_block_preconditioned`. This turns every run into a diagnosis.

---

## 1. Where R2b stands (what is DONE and BANKED)

- **R2a resolved + merged to master** (`3afa487`, not pushed): the 3-D projection split is
  unfaithful (weak fixed point); the **monolithic SBM-NS saddle solve is the R2 3-D forward
  engine**. See the findings doc. The projection is a separate research track.
- **cuDSS unblock DELIVERED (this is the banked R2b.1 win):** the monolithic sphere Cd
  mesh-convergence at Re=100, GPU-direct:
  - L4 (20k dof): **Cd=0.9641** (50s; == the CPU-splu value, 4.7× faster)
  - L5 (143k dof): **Cd=1.4476** (388s) — **the mesh CPU-splu STALLS on**
  - L6 (1.09M dof): cuDSS **`ALLOC_FAILED`** (direct-factorization GPU-memory wall)
  - Trend 0.96→1.45, refining toward the confined-box value (> unbounded Schiller-Naumann 1.092).
  - Baseline: `tests/baselines/p2r2c_monolithic_sphere_convergence.json`.
- **AMGX rebuilt from source on gpubox** (it was missing). Recipe: the box's `/usr/bin/nvcc` is
  CUDA 11.5 (too old for sm_89) — you MUST use CUDA 12.4:
  ```
  export CUDA_HOME=/usr/local/cuda-12.4; export PATH="$CUDA_HOME/bin:$PATH"
  cd $HOME/AMGX/build && cmake .. -DCMAKE_NO_MPI=1 -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc && make -j6 amgxsh
  ```
  `libamgxsh.so` (141 MB) is at `$HOME/AMGX/build/libamgxsh.so`. Run anything AMGX with
  `LD_LIBRARY_PATH=/usr/lib/wsl/lib:$HOME/AMGX/build`. `pyamgx.initialize()` works. (This also
  fixes the 2 `test_stepper_solvers::*_solver_parity[amgx]` failures in the suite.)

## 2. The system you're preconditioning

- **Monolithic SBM-NS saddle**, PSPG-stabilized **equal-order P1-P1** (this matters — see §4).
- **DOF layout:** node-major interleaved `[u_1, u_2, u_3, p]` per node, `ndof=4`, over `nfree`
  free nodes. Global dof id = `node*ndof + comp` (comp 0..2 velocity, 3 pressure).
- Assembled each Picard step in `tests/p2r0_task10_sphere_derisk.py::monolithic_cd`:
  `A = assemble_linear_ns(...) + Af_c(SBM block)`, then strong-Dirichlet velocity rows set to
  identity, ONE pressure DOF pinned to identity (`pin = argmax(coords.sum(1))*ndof + dim`).
  `sigma = 1/dt = 20` (dt=0.05), `nu = 2*U*R/Re = 0.0024` at Re=100.
- Block form: `[[F, G],[D, C]]` — F = velocity conv-diff-reaction (+ SBM Nitsche penalty),
  G = gradient/SUPG, D = divergence/PSPG, **C = PSPG pressure-pressure block ≈ τ ∇q·∇p (NONZERO —
  equal-order)**. True Schur `S = C - D F^{-1} G`.
- Mesh sizes: L4 `n_dof=19628`, L5 `143272`, L6 `1092924`.

## 3. The preconditioner + all the knobs (already built)

`src/diffsim/solvers/block_precond.py` — `BlockAMGPreconditioner`:
- Upper-triangular block Schur: `apply(r)`: `z_p = S~^{-1} r_p`; `z_u = F~^{-1}(r_u - G z_p)`.
- `F~^{-1}` = `_amg_F` AMGX cycle on F. `S~^{-1}` depends on `schur_mode`:
  - `"cahouet_chabard"` (default): `sigma * Kp^{-1} + nu * Mp^{-1}` (Kp = pressure stiffness via
    AMGX PCG, Mp_diag = pressure mass diagonal). **This assumes an inf-sup-STABLE saddle (C=0).**
  - `"pspg_c"` (added `2112774`): `C^{-1}` via AMGX, where `C = A[p_ids][:,p_ids]` (the actual
    PSPG pressure block). **Motivated because our system is PSPG equal-order, C≠0.**
- `solve_block_preconditioned(A, b, pre, tol=1e-9, maxiter=200)` → scipy `gmres(..., restart=50,
  maxiter=maxiter)`.
- **Tunable knobs** (all default to original behavior; committed `d996ad3` + `2112774`):
  `f_iters=2, f_tol=1e-2, kp_iters=8, kp_tol=1e-3, f_cycles=1, kp_cycles=3`, GMRES `restart=50`,
  `maxiter=200`, `schur_mode="cahouet_chabard"`.
- Plumbed via the `blockamgx_meta` cache dict in `linsolve.py` (`solver=="blockamgx"` branch,
  ~L1368) AND via ENV VARS in `monolithic_cd`: `F_ITERS F_TOL KP_ITERS KP_TOL F_CYCLES KP_CYCLES
  GMRES_RESTART GMRES_MAXITER SCHUR_MODE` (set only when present → default holds otherwise).

## 4. Experiments already run — ALL stagnate at L5 (do NOT repeat)

| # | config | result |
|---|---|---|
| de-risk | L3 saddle (~2916 dof), default | **CONVERGES** rel 1.9e-8 (`p2r2b1_blockamgx_derisk.py`) |
| 1 | L5, `F_ITERS=10 F_TOL=1e-3 KP_ITERS=15 GMRES_RESTART=100` | stagnate (~1500 iters, no conv) |
| 2 | L5, **near-exact F** `F_ITERS=50 F_TOL=1e-6 F_CYCLES=3 KP_ITERS=20 KP_TOL=1e-6` | **FAIL info=20 @ 2000 iters** → NOT the F solve |
| 3 | L5, `SCHUR_MODE=pspg_c KP_ITERS=20 KP_TOL=1e-4 GMRES_RESTART=100` | stagnate (~2750 inner, no conv) → NOT just the Schur formula |

**Interpretation:** the block-solve strengths (F, Schur) are NOT the lever. Converges at L3,
cliff-fails at L5. So look STRUCTURAL / scale-specific.

## 5. Hypotheses to test — IN PRIORITY ORDER

**H0 (do first, always): INSTRUMENT.** Add per-iteration residual logging to the outer FGMRES
(callback). Also log: the preconditioned residual reduction of a SINGLE `apply()` on a random
vector (does the preconditioner reduce anything at L5?), and AMGX's reported grid/operator
complexity + per-solve convergence for the F and Schur inner solves. Without this you are blind.

**H1 — preconditioner is silently ineffective at scale (block extraction/apply bug).**
Verify `u_ids`/`p_ids`, and the `G`/`D` sub-block extraction, are correct at L5 (they're derived
from `A` + `ndof` — check they match the node-major layout). Test: compare FGMRES with the
preconditioner vs WITHOUT (identity precond) at L5 — if the iteration curves are nearly identical,
the preconditioner is doing ~nothing (a bug), not merely weak. This is the highest-probability
structural cause given the cliff.

**H2 — restarted-GMRES(m) stagnation.** Restarted GMRES plateaus on hard nonsymmetric systems.
Test: L5 with `GMRES_RESTART` == `GMRES_MAXITER*something` (effectively NO restart / full GMRES) —
or switch the outer Krylov to a non-restarted method (BiCGStab, or FGMRES with a big restart). If
full-GMRES converges where GMRES(100) stalled, it's restart stagnation (cheap partial win).

**H3 — AMG coarsening fails on the SBM-structured F/C.** The F and C matrices carry SBM Nitsche
penalty entries (large near-boundary diagonals) + the pinned identity row. Classical AMG may build
a bad hierarchy at scale. Test: try different AMGX configs (smoothed-aggregation vs classical;
more/fewer levels; a stronger smoother) for `_amg_F`/`_amg_C`. The AMGX config files are in
`amgx_configs/` (see `src/diffsim/solvers/amgx.py::_config`). Log AMGX grid complexity.

**H4 — pinned-row / scaling pathology.** The single pinned pressure identity row (and the SBM
penalty `alpha=20` in the fixture) may ill-condition the block at scale. Test: row/col scaling
(diagonal equilibration) of A before preconditioning; or handle the pin via a null-space
projection instead of an identity row.

**H5 — the right Schur really is PCD, not C alone.** If H1-H4 are clean, the pressure Schur for
convection-dominated (Re=100) equal-order NS may need the **Pressure-Convection-Diffusion (PCD)**
approximation `S^{-1} ≈ Mp^{-1} Fp Kp^{-1}` (Fp = pressure convection-diffusion operator). This is
real implementation. Only after ruling out the structural causes above.

## 6. Reproduction — exact commands (gpubox)

Repo→box workflow (from repo root, `scripts/remote/`): `FORCE=1 bash scripts/remote/gpubox-sync.sh`
then `bash scripts/remote/gpubox-run.sh "<cmd>" <tag>` (detached tmux + run-lock; prints the log
path), poll `bash scripts/remote/gpubox-poll.sh <log>`. Box: `gpubox` host, repo at
`/home/bglab/Baskar/DiffSim`, 2× RTX 6000 Ada (48 GB each). **One run at a time (run-lock).**
AMGX runs need `LD_LIBRARY_PATH=/usr/lib/wsl/lib:$HOME/AMGX/build`.

- **Reproduce the cliff (L5, default precond):**
  `LD_LIBRARY_PATH=/usr/lib/wsl/lib:$HOME/AMGX/build SOLVER=blockamgx DEVICE=cuda:0 LEVELS=5 RE=100
  STEPS=3 GMRES_MAXITER=40 .venv/bin/python tests/p2r2c_monolithic_sphere_convergence.py`
- **Fast local (no AMGX) unit tests:** `.venv/bin/python -m pytest tests/test_blockamgx_linsolve.py`
  (18 pass, 1 skip). Use `_ExactCycleStub` for off-box structural tests.
- **The de-risk (L3, converges):** `.venv/bin/python tests/p2r2b1_blockamgx_derisk.py` (on box).
- **cuDSS convergence (works, for reference):** `SOLVER=cudss DEVICE=cuda:0 LEVELS=4,5 RE=100 ...`

## 7. Files

- `src/diffsim/solvers/block_precond.py` — the preconditioner (F/Kp/C, apply, schur modes, knobs,
  `solve_block_preconditioned`). **← most of your work is here.**
- `src/diffsim/solvers/linsolve.py` — `solver=="blockamgx"` branch (~L1368), reads `blockamgx_meta`.
- `src/diffsim/solvers/amgx.py` — AMGX singleton + configs (`amgx_configs/`); the Resources
  lifetime rule (2 max; the preconditioner respects it — F + one Schur op).
- `tests/p2r0_task10_sphere_derisk.py::monolithic_cd` — assembles A + Kp/Mp/C meta + env→meta.
- `tests/p2r2c_monolithic_sphere_convergence.py` — the convergence driver (SOLVER/LEVELS/RE/STEPS).
- `tests/p2r2b1_blockamgx_derisk.py` — STEP-A instantiate + STEP-B converge-vs-splu (small).
- `tests/test_blockamgx_linsolve.py` — local (off-box) tests + `_ExactCycleStub`.
- Reference: the survey findings in the ledger; the `blocktri` solver (`linsolve.py` ~L764) does
  **exact-F (cuDSS) + diagonal Schur** and converges in 2-3 iters at sigma=0 — a proven-in-a-regime
  alternative worth trying (exact-F would also localize H1 definitively).

## 8. Definition of done

The block-preconditioned monolithic march converges (rel < 1e-8, mesh-INDEPENDENT-ish outer
iteration count) at L5, then reaches L6 (1.09M) — producing the L6 sphere Cd for the R2c
convergence curve, at a memory footprint well below the direct cuDSS wall. Commit on `r2b1`; report
the L4/L5/L6 Cd + outer-iteration counts. That closes R2b.1's scalability goal on the iterative path.

## 9. Working rules

Branch `r2b1`, never push. `Co-Authored-By` trailer on commits. Subagent-driven if you fan out,
but this is tight iterative R&D — mostly one thread + gpubox runs. Keep the ledger
`.superpowers/sdd/progress.md` updated. Default solver paths (`splu`/`cudss`) MUST stay unchanged.
