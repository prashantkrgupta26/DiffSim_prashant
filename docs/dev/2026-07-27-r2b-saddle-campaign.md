# R2b Saddle-Preconditioner GPU Campaign Report

**Date:** 2026-07-27 / 2026-07-28  
**Branch:** `track-a-r2b`  
**Hardware:** gpubox — 2× NVIDIA RTX 6000 Ada Generation (48 GiB VRAM, sm_89, WSL2)  
**Harness:** `tests/gpu_saddle_ladder.py`  
**Commit at campaign:** 5b24496 (fix(solvers): wire pcd_meta into 2D/3D drivers; add SADDLE_POINTS env knob)

---

## 1. Harness Fixes (pre-campaign; committed before runs)

Before the GPU campaign could run, three bugs were found and fixed:

1. **pcd_meta wiring gap** — `fgmres_pcd` requires `("pcd_meta", cache_key)` in the
   solver cache dict (built once per mesh from `build_pcd_meta(dm, nu, sigma, p_pin)`),
   but neither `run_flow_past` (2D) nor `run_flow_past_3d` (3D) built or passed this.
   **Fix:** both drivers now rebuild pcd_meta once per BDF order (2 builds per run:
   BDF1 at step 0, BDF2 at step 1) and pass `cache/_pcd_cache, cache_key` to
   `solve_linear`. The 3D driver also wires `("blocktri_meta", "ns3d") = {"ndof": 4}`
   so `fgmres_bdiag` gets the correct 3D `ndof=4` (not the default 3).

2. **Fingerprint guard too strict** — the `solve_linear` staleness guard checked matrix
   shape/nnz per cache_key for all solvers; but for `fgmres_pcd/bdiag` the cache carries
   only mesh meta (constant per BDF order), not matrix factorizations, and the matrix A
   legitimately changes every step (Picard convection). **Fix:** exempted `fgmres_bdiag`
   and `fgmres_pcd` from the fingerprint guard (same exemption pattern as `blockch`).

3. **`nsteps` double-kwarg collision** — the ladder point dict includes `nsteps=5` which
   conflicted with the explicit `nsteps=NSTEPS` passed to `run_ladder_point`. **Fix:**
   excluded `nsteps` from the point-dict spread.

4. **`SADDLE_POINTS` env knob** — added to run each ladder point in a separate process,
   preventing the cross-leg memory accumulation / OOM documented in the brief.

CPU smoke test after fixes: bdiag=3867 iters, pcd=45 iters on 2d-r9 (both converged).

---

## 2. Campaign Matrix

Solvers: `fgmres_bdiag` (block-diagonal Jacobi), `fgmres_pcd` (Cahouet-Chabard PCD)  
Steps: nsteps=5, device=cuda:0  
Points run in SEPARATE processes via `SADDLE_POINTS=<tag>`.

---

## 3. Measured Results Table

| tag | solver | DOFs | iters/step | iters_mean | s/step | converged | log |
|-----|--------|------|-----------|-----------|--------|-----------|-----|
| 2d-r9 | fgmres_bdiag | 90,744 | [4274,3414,3344,2906,2989] | 3385.4 | 11.313 | YES | saddle-2dr9-bdiag-20260727-233643-42933.log |
| 2d-r9 | fgmres_pcd | 90,744 | [39,50,48,47,47] | **46.2** | 2.956 | YES | saddle-2dr9-pcd-20260727-233800-43285.log |
| 2d-r11 | fgmres_bdiag | 93,792 | [5853,6623,3542,3913,4277] | 4841.6 | 15.617 | YES | saddle-2dr11-bdiag-20260727-233832-43490.log |
| 2d-r11 | fgmres_pcd | 93,792¹ | — | N/A | N/A | **NO** (DIVERGED) | saddle-2dr11-pcd-20260727-234000-43808.log |
| 3d-L6 | fgmres_bdiag | 1,097,344 | [551,805,583,565,511] | **603.0** | 35.558 | YES | saddle-3dL6-bdiag-20260727-235440-46681.log |
| 3d-L6 | fgmres_pcd | 1,097,344 | [35,37,36,35,34] | **35.4** | 38.798 | YES | saddle-3dL6-pcd-20260727-235759-47448.log |
| 3d-L7r9 | fgmres_bdiag | ~9,240,000 | — | N/A | N/A | **OOM-KILLED** | saddle-3dL7r9-bdiag-20260728-000149-47900.log |
| 3d-L7r9 | fgmres_pcd | ~9,240,000 | — | N/A | N/A | NOT RUN | — |

All logs on gpubox: `/home/bglab/Baskar/DiffSim/logs/`  
All npz artifacts on gpubox: `/home/bglab/Baskar/DiffSim/results/`

¹ The 2d-r11/fgmres_pcd run DIVERGED (no npz written; summary recorded -1 DOFs).
The DOF count 93,792 is taken from the bdiag run on the identical r11 mesh.

---

## 4. Anomalies and Findings

### 4.1 3D-L7r9 OOM on gpubox (expected)

The 3D-L7r9 adaptive mesh (base-L7 + band-r9, ~9.24M DOF) was killed by the
Linux OOM-killer on gpubox (62 GB RAM). The log shows `Killed` immediately after
Warp init and before any solver output — the OOM occurred inside the first
`run_flow_past_3d` call, before any solver work.

**The mesh build is NOT the culprit here.** WP0 (nova GH200 job 11771926)
measured the r7b9 mesh build itself at **3.71 GB peak / 94 s** — the mesh build
is exonerated. The 209.7 GB MaxRSS figure recorded earlier was cross-leg
accumulation in the single-process ladder run (all ladder legs sharing one
process), not the mesh-build peak in isolation; see
`docs/dev/2026-07-27-100m-gh200-readiness.md` §2.1 "WP0 RESOLUTION".

**Leading hypothesis for the gpubox OOM** (no per-stage measurement on gpubox;
labeled as hypothesis): the host CSR-assembly and constraint-projection transients
at 9.24M-DOF ndof=4 scale exhaust the 62 GB RAM before any solve begins. The
plausible chain includes the `kron(T, I4)` constraint operator, the two-sided SBM
face system, the `T_vec.T @ Af_raw @ T_vec` sparse triple product, and the
step-0 `assemble_linear_ns` + `.tolil()` saddle surgery; the stored saddle CSR
alone is ~12–16 GB at this scale and the assembly transients multiply it.

**Gpubox has only 62 GB RAM — insufficient for 3D-L7r9 at this DOF scale.**  
The 9.24M point requires nova GH200 (480 GB Grace socket). A ready-to-submit
nova sbatch config can be derived from `cluster/slurm/thinshell_gh200_ladder.sbatch`
with the following env additions:
```
SADDLE_SOLVERS=fgmres_bdiag,fgmres_pcd
SADDLE_POINTS=3d-L7r9
SADDLE_NSTEPS=5
SADDLE_DEVICE=cuda:0
```
Submit with `--mem=400G` or higher. The GH200 verdict leg will capture the true
peak via RSS/SMI sampling and is the authoritative measurement for this scale.
The nova A100 (39 GiB VRAM) cannot run the 9.24M DOF solver; the GH200 (95 GiB
HBM3, 480 GB Grace) is the target.

### 4.2 2D-r11 fgmres_pcd DIVERGED

The PCD preconditioner that converged at r9 (90K DOF, 46 iters) failed to
converge at r11 (93K DOF, same DOF count but higher refinement level = finer
near-plate mesh). Relres stuck at 1.059e-03 after the maximum 12000 inner
iterations (200 restarts × 60-restart FGMRES).

Root cause hypothesis: the pressure Laplacian `Ap` assembled on the scalar Q1
space from `dm` has a condition number that grows with mesh refinement; the
Jacobi-CG inner solver for `Ap^{-1} r_p` may require more iterations at r11,
making the preconditioner quality degrade. At r9 the inner solves converge
within the `_INNER_MAX=500` cap; at r11 they may hit the cap without converging.

This is a genuine finding: **PCD as implemented (Cahouet-Chabard + Jacobi-CG
inner solves) is NOT robust across all mesh refinement levels.** The r9→r11
failure is at SIMILAR DOF COUNTS (90K vs 93K) — it is a REFINEMENT-QUALITY
problem, not a DOF-count scaling problem.

The 3D-L6 PCD result (35 iters, converged) is on a UNIFORM mesh (no adaptive
refinement), which may have better conditioned `Ap`. This deserves investigation
before interpreting the 3D results as "PCD scales."

### 4.3 2D vs 3D bdiag iteration counts

Bdiag shows a DECREASE in iterations from 2D (3385 at 90K DOF) to 3D (603 at
1.1M DOF). This is NOT a favorable scaling finding — it reflects the DIFFERENT
SADDLE SYSTEM being solved: 2D ndof=3 (u_x, u_y, p) vs 3D ndof=4 (u_x, u_y,
u_z, p). The 3D system has different conditioning, different block structure, and
a larger velocity block relative to the pressure block. These are not directly
comparable rungs on the same ladder.

The kill-gate is best evaluated WITHIN each dimension separately.

---

## 5. Kill-Gate Assessment

The kill-gate specification: "iteration growth ≤ ~2× across the 1M→10M decade
for ANY candidate ⇒ Track A viable."

### 5.1 Available evidence (0.3M→1.1M range only)

| solver | 90K DOF (2d-r9) | 1.1M DOF (3d-L6) | growth | note |
|--------|----------------|-----------------|--------|------|
| fgmres_bdiag | 3385 iters | 603 iters | 0.18× | cross-dim: 2D→3D, not apples-to-apples |
| fgmres_pcd | 46 iters | 35 iters | 0.76× | cross-dim: 2D→3D, not apples-to-apples |

Within 2D only (0.3M → 0.3M — both r9/r11 are ~90K DOF, same refinement regime):
- bdiag: 3385 → 4842 (growth 1.43×) — 2d-r9 to 2d-r11
- pcd: 46 → DIVERGED — PCD not robust at r11

Within 3D only: only one point measured (L6); no second point for growth assessment.

### 5.2 Gate verdict

**UNDECIDABLE** — the 9.24M 3D-L7r9 point required for the 1M→10M growth
measurement could not run on gpubox (OOM in the first `run_flow_past_3d` call on
the host-assembly path — see §4.1; the mesh build itself is WP0-exonerated) and was not submitted
to nova (out of scope for this task — controller handles cluster submissions).

**Interim evidence from 0.3M→1.1M:**
- fgmres_bdiag converges at both 2D (90K, 3385 iters) and 3D-L6 (1.1M, 603 iters).
  The cross-dimensional comparison is not a valid scaling measurement, but BOTH
  points converge with finite iteration counts, and the 3D count at 1.1M is lower
  than the 2D count at 90K.
- fgmres_pcd converges at 2D-r9 (46 iters) and 3D-L6 (35 iters), with PCD
  DRAMATICALLY outperforming bdiag at both measured points. PCD gives 73× fewer
  iterations at 2D-r9 and 17× fewer at 3D-L6 compared to bdiag.
- PCD FAILED at 2D-r11 (same DOF count as r9 but finer mesh). This is a
  robustness concern for the PCD implementation; the root cause is likely the
  inner Jacobi-CG quality on the finer pressure Laplacian, not the outer solver.

**Nova GH200 leg required before verdict:** The kill-gate cannot be decided from
the available 2D-only data (PCD failed at one 2D point) and 3D-L6 (one data
point). The 3D-L7r9 measurement at 9.24M DOF is needed for the 1M→10M decade.

**Projection to 100M (speculative, with stated assumptions):**
If PCD maintains ~35 iters at 1.1M and the growth is sub-linear (PCD theory
suggests O(1) iteration count with the exact Cahouet-Chabard Schur), projecting
to 100M DOF with 0× growth (optimistic, theoretical) would suggest ~35 iters
at 100M. With 2× growth (kill-gate bound), ~70 iters. This projection has
LARGE uncertainty — the 2D-r11 PCD failure and the absence of the 9.24M data
point make any extrapolation unreliable.

---

## 6. Config Provenance

- Harness: `tests/gpu_saddle_ladder.py` (commit 5b24496)
- Ladder point configs: LADDER_2D_R9, LADDER_2D_R11, LADDER_3D_L6, LADDER_3D_L7R9
  (verbatim from harness; see ladder docstring for geometry/physics params)
- GPU env: `LD_LIBRARY_PATH=/usr/lib/wsl/lib` (WSL2 CUDA passthrough, config.sh)
- Each point ran in a SEPARATE PROCESS (SADDLE_POINTS env knob) to prevent
  cross-leg memory accumulation
- NSTEPS=5 for all points
- Inner CG tolerance for PCD: `_INNER_TOL=1e-4`, `_INNER_MAX=500` (saddle_precond.py)

---

## 7. Follow-up Actions Required

1. **Nova GH200 3D-L7r9 run** (mandatory for kill-gate): Submit via sbatch from
   `cluster/slurm/` with SADDLE_SOLVERS=fgmres_bdiag,fgmres_pcd, SADDLE_POINTS=3d-L7r9,
   --mem=400G or higher. This is the controller's responsibility (per brief).

2. **PCD inner-solver robustness at fine mesh** (2d-r11 failure): The inner Jacobi-CG
   for `Ap^{-1}` hits `_INNER_MAX=500` at fine refinement. Options:
   - Increase `_INNER_MAX` (currently 500) to allow more inner iterations
   - Switch inner solver from Jacobi-CG to AMGX V-cycle for the Ap block
   - Use AMG-based pressure predictor (the R2b intended path)

3. **3D uniform ladder within-dimension scaling**: Run 3D-L7 uniform (if it fits
   on gpubox host RAM) to get a within-3D growth measurement. L7 uniform has
   ~8.4M DOF in 3D — also likely OOM on gpubox but worth checking.
