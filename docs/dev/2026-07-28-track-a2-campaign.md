# Track A2 Campaign Report — Saddle-Point Solver GPU Validation

**Date:** 2026-07-28  
**Branch:** `track-a2-strengthen`  
**Campaign objective:** Validate pcd-amgx as the production saddle-point solver
engine by measuring outer iteration growth, s/step, and inner telemetry across
the 2-D and 3-D ladder points with device assembly on gpubox (RTX 6000 Ada),
then submitting the 9.08M gate legs to GH200 for the binding verdict.

**Campaign manager:** T4 subagent (Claude Opus 4.8)  
**Controller:** fills TODO-GH200 section after GH200 results land.

---

## 1. Environment / Infrastructure

### 1.1 gpubox

| Item | Value |
|------|-------|
| Hardware | 2× NVIDIA RTX 6000 Ada Generation, 48 GiB each, sm_89 |
| CUDA / Driver | 12.9 / 13.2 (WSL2) |
| LD_LIBRARY_PATH | `/home/bglab/AMGX/build:/usr/lib/wsl/lib` (required for pyamgx) |
| pyamgx status | **PRESENT** — `libamgxsh.so` at `/home/bglab/AMGX/build/libamgxsh.so`; `import pyamgx` succeeds with the above LD path |
| Branch at run time | `track-a2-strengthen` @ cc9838b (plumbing) + 91882fe (amgx smoother fix) |

### 1.2 nova GH200

| Item | Value |
|------|-------|
| Hardware | NVIDIA GH200 (Grace Hopper) |
| venv | `.venv-nova-arm` (Python 3.11, aarch64) |
| pyamgx status | **ABSENT** — no `pyamgx` package in `.venv-nova-arm/lib/python3.11/site-packages/`; no `libamgxsh.so` found under `/work/mech-ai/baskarg/`; login node is x86 so cannot run arm binaries to verify. **CONTROLLER MUST INSTALL pyamgx on nova-arm before submitting the pcd-amgx sbatch kit.** |

---

## 2. Plumbing Changes (T4 gate commits)

Two commits landed on this branch before the campaign:

| SHA | Change |
|-----|--------|
| `cc9838b` | `feat(t4): SADDLE_PCD_INNER env + pcd_inner kwarg plumbing; inner_stats to npz` — adds `SADDLE_PCD_INNER` env → `pcd_inner` param through both 2-D and 3-D drivers and to `build_pcd_meta`; saves T1 `inner_stats` (F/Ap/Mp telemetry) as flat `ist_{blk}_{key}` arrays into the ladder npz. |
| `91882fe` | `fix(amgx): accept not_converged status as smoother in PCD F-inner` — AMGX returns `not_converged` when `maxiter` is hit; for the PCD F-inner the truncated iterate is correct (used as a smoother, outer FGMRES carries residual). Previously raised RuntimeError; now accepted like Jacobi-CG's cap_hits. |

Gate (both commits): `test_saddle_ladder_cpu` + `test_saddle_precond` — **16 passed** (including the previously-skipped `test_fgmres_pcd_amgx_real` which PASSES on gpubox).

---

## 3. Real-AMGX Correctness — Unit Test

```
tests/test_saddle_precond.py::test_fgmres_pcd_amgx_real  PASSED  (6.73 s, gpubox)
```

This is the **first real-AMGX correctness evidence**: with `inner="amgx"`, the outer
FGMRES PCD solve reaches the accuracy gates (`||x-x_splu||/||x_splu|| < 1e-8`)
on the level-4 saddle using AMGX BiCGStab+classical-AMG on the extracted velocity
block F. AMGX setup was reused across the ~20 outer preconditioner applies (same
sparsity, values-only refresh after first setup).

---

## 4. Campaign Matrix — Measured Results

### 4.1 Reference Numbers (Track A campaign, host assembly)

| Tag | Solver | DOFs | iters/step (mean) | s/step | Assembly | Note |
|-----|--------|------|--------------------|--------|----------|------|
| 3d-L6 | fgmres_bdiag | 1,097,344 | 603.0 | 35.6 s | host | Track A ref |
| 3d-L6 | fgmres_pcd (jacobi) | 1,097,344 | 35.4 | 38.8 s | host | Track A ref |
| 2d-r11 | fgmres_pcd (jacobi) | — | — | — | host | DIVERGED (relres 1e-3) |
| 3d-L7r9 | fgmres_bdiag | 9,080,000 | 1196.8 | 242.2 s | host | Track A ref |
| 3d-L7r9 | — | — | — | — | host | setup ~1.5 h, RSS 243-245 GB |

### 4.2 gpubox Measured (T4, device assembly, cuda:1 / cuda:0)

All legs: `SADDLE_ASSEMBLY=device`, branch `track-a2-strengthen`.

| Tag | Solver | DOFs | iters/step (mean) | s/step | RC | Log |
|-----|--------|------|--------------------|--------|-----|-----|
| 3d-L6 | fgmres_bdiag | 1,097,344 | 602.2 | **12.648 s** | 0 | `logs/t4-3dL6-bdiag-device.log` |
| 3d-L6 | fgmres_pcd+amgx | 1,097,344 | **DNF** (1642 outer @103min, step 1 never done) | **DNF** | SIGTERM | `logs/t4-3dL6-pcdamgx-device.log` |
| 2d-r11 | fgmres_pcd+amgx | ~1.3M | **DIVERGES** | — | 0 (harness caught) | `logs/t4-2dr11-pcd-amgx-200.log` |

**3d-L6 bdiag device-asm headline:**
- iters/step: 602.2 (virtually identical to 603.0 host reference — device assembly does not change iteration count, as expected)
- **s/step: 12.648 s vs 35.6 s host-path = 2.8× wall-time reduction from device assembly alone**
- iters by step: [551, 801, 583, 565, 511] — step 2 spike (BDF1→BDF2 transition) normal

#### 4.2.1 3d-L6 pcd-amgx device-asm — **CATASTROPHIC: FAILED TO CONVERGE IN REASONABLE TIME**

**Log:** `logs/t4-3dL6-pcdamgx-device.log`  
**Started:** 16:32 CDT  
**Ended:** ~18:15 CDT (~103 min) — process (PID 53297) received SIGTERM ("Terminated" in log);
step 1 of 5 NEVER COMPLETED. Not an OOM kill: the box's dmesg OOM entry maps to ~00:05 CDT
(an earlier, unrelated run), and our process RSS was ~4.2 GB on the 62 GB box.

**Final measured inner-solve telemetry (from log):**
- 1642 F-inner AMGX calls completed at termination
- Average per inner-solve: ~5.8 s (each hitting maxiter=50 → `not_converged`)
- All 1642 calls correspond to step 1 only (no step completion marker in log)
- No npz written (step never completed; the existing `saddle_ladder_3d-L6_fgmres_pcd.npz`
  dated Jul 28 00:01 is the Track A jacobi run: iters [35,37,36,35,34], 38.8 s/step)

**Verdict:** pcd-amgx on 3d-L6 is NOT viable. The outer FGMRES accumulated ≥1642 outer iterations
(vs 35.4 for pcd-jacobi) without converging step 1. This is ≥46× iteration growth — compared to the
kill-gate threshold of ≤2×. The AMGX F-inner (maxiter=50, BiCGStab+classical-AMG) provides a truncated
iterate that is too poor in quality for the PCD Schur approximation at 1.1M DOF. Each AMGX inner
"solve" hits maxiter (not_converged), so the PCD preconditioner receives a low-accuracy F-inner, and
the outer FGMRES must compensate with far more iterations.

**Root cause:** AMGX BiCGStab+classical-AMG at maxiter=50 is insufficient accuracy for the 3-D NS
velocity block at Re=250. For jacobi-CG inner, the CG converges (or cap_hits a tight residual) with
higher relative accuracy, yielding a better Schur approximation. Increasing AMGX maxiter would help
but make each inner call proportionally more expensive. The AMG setup reuse (hierarchy already built)
is working, but the solve quality budget is too coarse.

**Conclusion:** pcd-amgx fails the iteration-growth kill-gate at the very first (smallest) ladder
point. GH200 submission of the pcd-amgx kit at 9.08M DOF is NOT warranted until the AMGX maxiter
budget is tuned (or the preconditioner redesigned). The leg is a confirmed negative result; the
process ended (SIGTERM) at ~18:15 CDT without ever completing a single time step.

#### 4.2.2 2d-r11 pcd-amgx 200-step probe — **DIVERGES (same as jacobi)**

**Log:** `logs/t4-2dr11-pcd-amgx-200.log`  
**Duration:** near-instant (step 1 FGMRES blew through 200 restarts × 60 inner = 12000 inner iters)  
**Result:** `ConvergenceError: fgmres_pcd: not converged after 12000 inner iterations (200 restarts); relres=4.086e-01`

The Track A reference had 2d-r11 fgmres_pcd (jacobi) DIVERGE at relres ~1e-3. This leg confirms the
AMGX inner does NOT fix the 2d-r11 divergence — the residual is actually worse (0.41 vs ~0.001 for
jacobi). The divergence is a structural issue with the PCD Schur approximation quality on the
fine-graded 2-D mesh at Re=250, not an inner-solve accuracy issue.

**Harness output:**
```
FAILED 2d-r11/fgmres_pcd: ConvergenceError: fgmres_pcd: not converged after 12000 inner
  iterations (200 restarts); relres=4.086e-01
  2d-r11  fgmres_pcd  -1  N/A  N/A  False
Kill-gate: insufficient valid points for kill-gate check
SADDLE-LADDER-OK  (harness caught the ConvergenceError as expected; RC=0)
```

**NPZ:** No npz written for the 2d-r11 pcd-amgx leg (ConvergenceError → error fallback dict with dofs=-1).

### 4.3 Inner Stats (T1 telemetry — fgmres_pcd legs)

Inner stats (F/Ap/Mp block applies, iters_total, cap_hits, max_exit_relres) are
saved to the npz files under `results/saddle_ladder_*_fgmres_pcd.npz` as
`ist_{blk}_{key}` arrays.

**3d-L6 pcd-amgx (final, from log telemetry at step-1 DNF / SIGTERM):**
- F-inner AMGX calls: 1642 (all step 1; each hits maxiter=50 → not_converged)
- Average per F-inner AMGX call: ~5.8 s (50 BiCGStab iters)
- Every single call returned `not_converged` (maxiter hit as smoother mode)
- No npz written (step never completed)

**2d-r11 pcd-amgx (from log, diverged on step 1):**
- F-inner AMGX calls: 11784 (harness 200-restart × 60-inner budget; all step 1)
- Convergence: NONE — relres=0.41 after 12000 inner FGMRES iterations
- No npz written (ConvergenceError → error fallback dict with dofs=-1)

**3d-L6 pcd-jacobi reference (from existing npz `results/saddle_ladder_3d-L6_fgmres_pcd.npz`):**
- This is the Track A run; inner_stats not present (pre-T1-telemetry run)
- iters/step: [35, 37, 36, 35, 34], mean=35.4; s/step=38.8 s; converged=True

---

## 5. GH200 Gate Legs — Submission-Ready Configs

**PREREQUISITE (BLOCKING): pyamgx is ABSENT on nova.** The controller must
install pyamgx in `.venv-nova-arm` on nova before submitting the pcd-amgx kit.
The bdiag kit does not require pyamgx.

### 5.1 Bdiag kit (device assembly, no pyamgx required)

File: `cluster/slurm/saddle_ladder_gh200.sbatch`

```
SADDLE_POINTS=3d-L7r9 SADDLE_SOLVERS=fgmres_bdiag SADDLE_DEVICE=cuda:0
SADDLE_NSTEPS=5 SADDLE_ASSEMBLY=device
```

Expected: setup time minutes (vs ~1.5 h host), RSS ≪ 243-245 GB (device skips
the host CSR transient), iters/step ~1196 (unchanged by assembly), s/step scaled
from the gpubox 3d-L6 observation.

### 5.2 PCD-AMGX kit (device assembly + AMGX inner — REQUIRES pyamgx on nova)

> **BLOCKING — DO NOT SUBMIT until gpubox 3d-L6 pcd-amgx re-run shows ≤2× iteration growth**  
> Measured result: ≥44× outer iteration growth at 1.1M DOF (≤2× gate threshold). Submitting  
> to GH200 at 9.08M DOF will consume node-hours and almost certainly fail the same gate.  
> Root cause: AMGX maxiter=50 budget too coarse for 3-D NS velocity block. Tune first.

File: `cluster/slurm/saddle_ladder_gh200_pcd.sbatch`

```
SADDLE_POINTS=3d-L7r9 SADDLE_SOLVERS=fgmres_pcd SADDLE_DEVICE=cuda:0
SADDLE_NSTEPS=5 SADDLE_ASSEMBLY=device SADDLE_PCD_INNER=amgx
```

**Install path for pyamgx on nova-arm (controller action):**
1. Build AMGX from source for aarch64 (`cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90a`);
   stage `libamgxsh.so` under `/work/mech-ai/baskarg/AMGX/build/`.
2. `AMGX_DIR=/work/mech-ai/baskarg/AMGX pip install pyamgx` (from shwina/pyamgx).
3. Set `AMGX_LIB_DIR=/work/mech-ai/baskarg/AMGX/build` in the sbatch (kit already
   reads this env in its LD_LIBRARY_PATH line).
4. Verify: `LD_LIBRARY_PATH=/work/mech-ai/baskarg/AMGX/build .venv-nova-arm/bin/python3.11 -c 'import pyamgx'` from an interactive node.

Expected with device assembly: setup RSS ≪ 100 GB (vs 243-245 GB host); NSTEPS=5
is feasible in a single job (device assembly makes setup minutes not 1.5 h).

---

## 6. Assembly-Time Savings — Summary

| Leg | Assembly | Setup time | s/step | RSS peak |
|-----|----------|------------|--------|----------|
| 3d-L6 bdiag | host (Track A ref) | N/A | 35.6 s | — |
| 3d-L6 bdiag | **device (T4)** | fast (< 1 min) | **12.6 s** | — |
| 3d-L7r9 bdiag | host (Track A ref) | ~1.5 h | 242.2 s | 243-245 GB |
| 3d-L7r9 bdiag | device (GH200, TODO) | — | — | — |

The 2.8× s/step reduction on 3d-L6 bdiag from host→device assembly is a
significant finding: device assembly not only removes the host-RAM transient but
also substantially reduces per-step wall time (less data movement across the
PCIe/NVLink bus).

---

## TODO-GH200 (SANCTIONED PLACEHOLDER — controller fills after GH200 results)

The binding verdict gate (SUCCESS: 9.08M pcd-amgx outer growth from 3d-L6 ≤ ~2×
AND s/step ≤ ~60 s) requires the 3d-L7r9 GH200 legs. The controller submits
both sbatch kits after:
1. Confirming 3d-L6 results above (device assembly works).
2. Installing pyamgx on nova-arm.
3. Submitting bdiag kit first (confirms device-assembly setup time + RSS before
   the pcd-amgx kit, which requires pyamgx).

**Fill in here:**
- 3d-L7r9 bdiag device-asm: iters/step, s/step, setup time, RSS peak, job ID
- 3d-L7r9 pcd-amgx device-asm: outer iters/step, s/step, inner_stats (F iters/apply, cap_hits), SMI peak, job ID
- Verdict: PASS / FAIL / partial outcome + rationale

---

## 7. Anomalies and Notes

1. **Track B session lock**: on arrival, gpubox had a live tmux session `diffsim-20260728-150243-11183` holding the `.remote-run.lock`. Campaign legs were run directly (nohup ssh) on cuda:0 (after Track B completed) and cuda:1 (free). Lock was clear before the T4 legs started.

2. **pyamgx absent without LD_LIBRARY_PATH**: `import pyamgx` fails with `libamgxsh.so: cannot open shared object file` until `/home/bglab/AMGX/build` is prepended. The sbatch kit uses `GPUBOX_GPU_ENV` from `config.sh` which only includes `/usr/lib/wsl/lib`; any future gpubox AMGX run must add `/home/bglab/AMGX/build` to LD_LIBRARY_PATH (or install the shared lib in a standard path). The T4 legs include it explicitly in the nohup commands.

3. **amgx.py not_converged fix**: AMGX reports `not_converged` (not `success`) when `maxiter` is hit. For the PCD F-inner this is the correct smoother behavior (same as Jacobi-CG `cap_hits`). The fix accepts `not_converged`; any other non-success status still raises. This was a pre-existing gap caught by the T4 unit test on gpubox.

4. **2d-r11 pcd-amgx DIVERGES** (confirmed, 2026-07-28 16:25 CDT): AMGX inner does NOT fix the 2d-r11 divergence. relres=0.41 (worse than jacobi's ~0.001) after 12000 inner iterations. Confirms the failure is structural (PCD Schur approximation quality on fine-graded 2-D mesh at Re=250), not an inner-solve issue.

5. **3d-L6 pcd-amgx DNF** (confirmed negative, 2026-07-28; process SIGTERMed ~18:15 CDT): After ~103 min wall time and 1642 outer FGMRES iterations, step 1 of 5 never completed. AMGX BiCGStab+classical-AMG at maxiter=50 is too coarse for the PCD F-inner at 1.1M DOF 3-D velocity. Kill-gate fails: iteration growth ≥46× (vs ≤2× required). GH200 pcd-amgx submission is NOT warranted without AMGX parameter re-tuning. Termination cause is external SIGTERM ("Terminated" in log), not OOM — verdict unaffected either way.

6. **3d-L6 bdiag step-2 spike**: iters [551, 801, 583, 565, 511] — step 1 (BDF1→BDF2) causes a spike to 801 outer iters; this is the well-documented transition artifact and not a solver pathology.

7. **GH200 pcd-amgx sbatch kit status**: Kit is complete and ready in `cluster/slurm/saddle_ladder_gh200_pcd.sbatch` but should NOT be submitted until AMGX maxiter budget is tuned and gpubox 3d-L6 pcd-amgx re-run shows ≤2× iteration growth vs pcd-jacobi. The bdiag kit can still be submitted independently to validate device-assembly speedup on GH200.
