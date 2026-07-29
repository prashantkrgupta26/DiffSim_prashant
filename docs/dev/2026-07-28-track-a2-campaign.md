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

### 4.2 gpubox Measured (T4, device assembly, cuda:0 / cuda:1)

All legs: `SADDLE_ASSEMBLY=device`, branch `track-a2-strengthen`.  
Device assignment: bdiag and pcd-amgx legs ran on **cuda:0**; the 2d-r11 probe ran on **cuda:1** (free at the time).

| Tag | Solver | DOFs | iters/step (mean) | s/step | RC | Log |
|-----|--------|------|--------------------|--------|-----|-----|
| 3d-L6 | fgmres_bdiag | 1,097,344 | 602.2 | **12.648 s** | 0 | `logs/t4-3dL6-bdiag-device.log` (cuda:0) |
| 3d-L6 | fgmres_pcd+amgx | 1,097,344 | **DNF** (1642 outer @103min, step 1 never done) | **DNF** | SIGTERM | `logs/t4-3dL6-pcdamgx-device.log` (cuda:0) |
| 3d-L6 | fgmres_pcd (jacobi) | 1,097,344 | **35.4** | **11.732 s** | 0 | `logs/t4-3dL6-pcdjacobi-device-20260728-182854-65703.log` (cuda:0) |
| 2d-r11 | fgmres_pcd+amgx | ~1.3M | **DIVERGES** | — | 0 (harness caught) | `logs/t4-2dr11-pcd-amgx-200.log` (cuda:1) |

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

**Root cause (corrected 2026-07-28 Phase 1.5):** The classical-AMG cycle (Jacobi smoother,
presweep/postsweep=1) is **structurally ineffective** on the 3-D convection-dominated velocity block
at Re=250 — this is an inner-quality ceiling, not a budget limitation. The key evidence: AMGX
BiCGStab+classical-AMG stalls at relative residual 2.2e-3 with **identical residual history** at
maxiter=50 and maxiter=400 (measured directly on the extracted F block, n=823,008, nnz=64.6M).
A maxiter sweep is therefore refuted as the next probe — the smoother itself is too weak. By
contrast, Jacobi-CG reaches relres 2.1e-5 in 20 iterations / 0.31 s on the same matrix. The AMG
hierarchy is successfully built and reused (setup amortized), but the solve quality under classical-AMG
with Jacobi smoothing does not improve with more iterations.

**Candidate future probes:** (1) a different AMGX config — aggregation AMG and/or a stronger smoother
(Gauss-Seidel, ILU-type, Krylov-smoothed); (2) dropping AMGX for the F-block entirely — Jacobi-CG
is simply better here and requires no external library.

**Conclusion:** pcd-amgx fails the iteration-growth kill-gate at the very first (smallest) ladder
point. GH200 submission of the pcd-amgx kit at 9.08M DOF is NOT warranted. The leg is a confirmed
negative result; the process ended (SIGTERM) at ~18:15 CDT without ever completing a single time step.

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

**3d-L6 pcd-jacobi + device assembly (Phase 1.5, from npz `results/saddle_ladder_3d-L6_fgmres_pcd.npz`):**
- **This is the first production T1 telemetry reading.**
- iters/step: [35, 37, 36, 35, 34], mean=35.4; s/step=**11.732 s**; converged=True
- Inner stats (last step, step 5, 34 outer iters — 1 inner apply per outer iter):

| Block | Applies | Iters total | Iters/apply | Cap hits | Max exit relres |
|-------|---------|-------------|-------------|----------|-----------------|
| F (velocity) | 34 | 670 | **19.7** | 0 | 5.8e-5 |
| **Ap (pressure conv-diff)** | **34** | **9610** | **282.6** | **0** | **9.8e-5** |
| Mp (pressure mass) | 34 | 680 | **20.0** | 0 | 3.7e-5 |

**Inner-iteration headline: Ap block dominates by 14×.** The pressure convection-diffusion
solve (Ap) requires ~283 CG iterations per outer preconditioner apply vs ~20 for F and Mp.
No cap hits on any block — all inner solves converge. This identifies Ap as the primary
inner-solve cost center for any future acceleration effort (e.g. a dedicated Ap preconditioner
or AMG for the pressure block, if the F-block AMG failure history is avoided).

**3d-L6 pcd-jacobi reference (Track A host-path run — pre-T1-telemetry, for comparison):**
- inner_stats not present (pre-T1-telemetry run)
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

### 5.2 PCD-AMGX kit — **SUPERSEDED: retargeted to pcd-jacobi (see §8)**

> **BLOCKED — DO NOT SUBMIT the pcd-amgx config to GH200.**  
> Measured result: ≥46× outer iteration growth at 1.1M DOF (≤2× gate threshold). Root cause  
> (corrected): the classical-AMG cycle with Jacobi smoother is structurally ineffective on the  
> 3-D convection-dominated F block — a maxiter increase does NOT help (identical residual history  
> at maxiter=50 vs maxiter=400 on the same extracted matrix). The GH200 kit has been retargeted  
> to pcd-jacobi inner; see §8 and `cluster/slurm/saddle_ladder_gh200_pcd.sbatch`.

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
| 3d-L6 pcd-jacobi | host (Track A ref) | N/A | 38.8 s | — |
| 3d-L6 pcd-jacobi | **device (Phase 1.5)** | fast (< 1 min) | **11.732 s** | — |
| 3d-L7r9 bdiag | host (Track A ref) | ~1.5 h | 242.2 s | 243-245 GB |
| 3d-L7r9 bdiag | device (GH200, TODO) | — | — | — |
| 3d-L7r9 pcd-jacobi | device (GH200, TODO) | — | — | — |

The 2.8× s/step reduction on 3d-L6 bdiag from host→device assembly is a
significant finding: device assembly not only removes the host-RAM transient but
also substantially reduces per-step wall time (less data movement across the
PCIe/NVLink bus).

---

## TODO-GH200 (SANCTIONED PLACEHOLDER — controller fills after GH200 results)

**UPDATED 2026-07-29: See §10 "GH200 hold session (2026-07-29)" below — device assembly
at L7r9 hits TWO sequential 32-bit ceilings (scipy int32 CSR overflow + Warp int32 array
shape limit). Host assembly fallback is running. Partial results below; full table in §10.**

The binding verdict gate (SUCCESS: 9.08M pcd-jacobi outer growth from 3d-L6 ≤ ~2×
AND s/step tractable) requires the 3d-L7r9 GH200 legs. The controller submits
both sbatch kits after:
1. Confirming 3d-L6 results above (device assembly works — Phase 1 done).
2. Submitting bdiag kit first (confirms device-assembly setup time + RSS).
3. Submitting pcd-jacobi kit (pyamgx NOT required — jacobi inner has no deps).

**Fill in here (final, 2026-07-29 hold session — details in §10):**
- 3d-L7r9 bdiag host-asm: **DONE RC 0** — 1196.8 iters (identical to gpubox ref), 247.8 s/step (§10.4)
- 3d-L7r9 pcd-jacobi host-asm: **DNF** (capped 2h28m, bound >29 min/step) (§10.5)
- 3d-L7r9 pcd-jacobiF+amgx-Ap: **DNF** (4h cap, bound >48 min/step; Ap-AMG applies work at ~0.22 s — F-block is the wall) (§10.6)
- 3d-L7 uniform bdiag device-asm: **LANDMARK RC 0** — 54.8 s/step, 14.9 GB RSS; controlled
  same-mesh A/B vs host asm (237.6 s/step, 190.0 GB) = **4.33× / 12.8× at bit-identical iters** (§10.9)
- 3d-L8 uniform ~68M capacity probe: failure site captured — unbounded 137.4 GB dof-indices intermediate; persistent state fits HBM (§10.10)
- device-asm on ADAPTIVE meshes: BLOCKED (Warp 2^31 slot array limit in constraint-expansion path — §10.2)
- **Verdict: the SUCCESS gate (9.08M pcd-jacobi outer growth ≤ ~2× AND s/step tractable)
  FAILS — pcd is wall-clock intractable at 9.08M in all three variants.
  fgmres_bdiag is the engine at scale: 247.8 s/step (host asm, adaptive L7r9) and
  54.8 s/step (device asm, uniform L7).**

---

## 8. Phase 1.5 Corrections and Next Steps

### 8.1 Root-cause correction: pcd-amgx classical-AMG stall (not a budget issue)

The Phase 1 campaign doc stated "AMGX maxiter=50 budget too coarse; increase maxiter" as the root
cause and next probe. **This is refuted.** Direct measurement on the extracted F block (n=823,008,
nnz=64.6M) shows AMGX BiCGStab+classical-AMG (Jacobi smoother, presweep/postsweep=1) stalls at
relres 2.2e-3 with **identical residual history** at maxiter=50 and maxiter=400. The smoother is
structurally ineffective on the convection-dominated block — adding iterations does not help.
Jacobi-CG (no AMG) reaches relres 2.1e-5 in 20 iterations / 0.31 s on the same matrix.

**Corrected root cause:** The classical-AMG cycle quality ceiling — not the iteration budget.

**Refuted probe:** maxiter sweep.

**Candidate future probes:**
1. Different AMGX config: aggregation AMG and/or stronger smoother (Gauss-Seidel, ILU-type,
   Krylov-smoothed). Requires pyamgx rebuild on nova-arm before GH200 submission.
2. Drop AMGX for the F-block: Jacobi-CG is simply better here and requires no external library.
   This is the path pursued in §8.2 (GH200 kit retarget).

### 8.2 GH200 pcd kit retargeted to pcd-jacobi

`cluster/slurm/saddle_ladder_gh200_pcd.sbatch` has been retargeted from pcd-amgx to pcd-jacobi:
- `SADDLE_PCD_INNER=amgx` removed (jacobi is the default)
- pyamgx prerequisite paragraph moved to a note (applies only to a future amgx-config retry)
- Expected wall based on gpubox Phase 1.5 measurement: see §4.2 table (pcd-jacobi+device row)

**Note on pyamgx (future amgx-config retry only):** If a future probe retries AMGX with a
different config (aggregation AMG / stronger smoother), the controller must first install pyamgx
in `.venv-nova-arm` on nova: build AMGX for aarch64 (`cmake -DCMAKE_CUDA_ARCHITECTURES=90a`),
stage `libamgxsh.so` under `/work/mech-ai/baskarg/AMGX/build/`, then
`AMGX_DIR=/work/mech-ai/baskarg/AMGX pip install pyamgx` and verify with an interactive node
before submitting.

## 7. Anomalies and Notes

1. **Track B session lock**: on arrival, gpubox had a live tmux session `diffsim-20260728-150243-11183` holding the `.remote-run.lock`. Campaign legs were run directly (nohup ssh) on cuda:0 (after Track B completed) and cuda:1 (free). Lock was clear before the T4 legs started.

2. **pyamgx absent without LD_LIBRARY_PATH**: `import pyamgx` fails with `libamgxsh.so: cannot open shared object file` until `/home/bglab/AMGX/build` is prepended. The sbatch kit uses `GPUBOX_GPU_ENV` from `config.sh` which only includes `/usr/lib/wsl/lib`; any future gpubox AMGX run must add `/home/bglab/AMGX/build` to LD_LIBRARY_PATH (or install the shared lib in a standard path). The T4 legs include it explicitly in the nohup commands.

3. **amgx.py not_converged fix**: AMGX reports `not_converged` (not `success`) when `maxiter` is hit. For the PCD F-inner this is the correct smoother behavior (same as Jacobi-CG `cap_hits`). The fix accepts `not_converged`; any other non-success status still raises. This was a pre-existing gap caught by the T4 unit test on gpubox.

4. **2d-r11 pcd-amgx DIVERGES** (confirmed, 2026-07-28 16:25 CDT): AMGX inner does NOT fix the 2d-r11 divergence. relres=0.41 (worse than jacobi's ~0.001) after 12000 inner iterations. Confirms the failure is structural (PCD Schur approximation quality on fine-graded 2-D mesh at Re=250), not an inner-solve issue.

5. **3d-L6 pcd-amgx DNF** (confirmed negative, 2026-07-28; process SIGTERMed ~18:15 CDT): After ~103 min wall time and 1642 outer FGMRES iterations, step 1 of 5 never completed. Root cause (corrected Phase 1.5): classical-AMG with Jacobi smoother is structurally ineffective on the 3-D convection-dominated F block — AMGX stalls at relres 2.2e-3 with identical history at maxiter=50 and maxiter=400; a maxiter sweep is refuted. Kill-gate fails: iteration growth ≥46× (vs ≤2× required). GH200 pcd-amgx submission is NOT warranted without a qualitatively different AMGX config (aggregation AMG / stronger smoother) or dropping AMGX for F entirely. Termination cause is external SIGTERM ("Terminated" in log), not OOM — verdict unaffected either way.

6. **3d-L6 bdiag step-2 spike**: iters [551, 801, 583, 565, 511] — step 1 (BDF1→BDF2) causes a spike to 801 outer iters; this is the well-documented transition artifact and not a solver pathology.

7. **GH200 pcd kit retargeted to jacobi inner (Phase 1.5)**: `cluster/slurm/saddle_ladder_gh200_pcd.sbatch` has been updated to use pcd-jacobi (SADDLE_PCD_INNER unset). The pcd-amgx config is NOT viable without a qualitatively different AMGX config (aggregation AMG / stronger smoother) — increasing maxiter is refuted. A future amgx-config retry would require pyamgx on nova-arm (see §8). The bdiag kit can still be submitted independently to validate device-assembly speedup on GH200.

---

## 9. T5: AMG-on-Ap Probe (Baskar-approved interim, 2026-07-28)

### 9.1 Motivation

T4 inner-solve telemetry at 3d-L6 (pcd-jacobi+device, 11.732 s/step, 35.4 outers)
identified the Ap block as the dominant inner cost: 282.6 iters/apply vs 19.7 for F
and 20.0 for Mp (14× imbalance, zero cap hits). Ap is the scalar pressure Laplacian
(SPD, M-matrix-like, pinned at the saddle's pin node) — the canonical target for
classical AMG. Unlike the F-inner (which failed because classical AMG + Jacobi smoother
is structurally ineffective on the 3-D convection-dominated nonsymmetric F block),
Ap is symmetric and well-suited for PCG+classical-AMG.

The question: does collapsing Ap iters/apply from 283 → O(10) translate to a
measurable s/step reduction at 3d-L6?

### 9.2 Implementation (commit `7a3752b`)

- `build_pcd_meta(..., ap_inner="jacobi")` — new kwarg (`"jacobi"` default, bit-for-bit;
  `"amgx"` routes Ap-inner through `amgx_solve(Ap, rhs, sym=True, tol=1e-4,
  maxiter=200)`). Singleton key `(True, 1e-4, 200)` is distinct from F-inner key
  `(False, 1e-4, 50)` — no singleton thrash. Ap sparsity is constant (built once per
  mesh); AMG hierarchy built exactly once per run, reused via setup-reuse path.
- Env plumbing: `SADDLE_PCD_AP_INNER` through ladder + both drivers.
- Tests: routing monkeypatch (sym=True, scalar Ap shape); default-parity; skipif
  real-AMGX test. All 31 gate tests passed (2 skipped = real-AMGX on CPU).

### 9.3 Measured Results

**Leg A: 3d-L6 × pcd-jacobiF-amgxAp, 5 steps, device assembly, cuda:0**

Command: `SADDLE_SOLVERS=fgmres_pcd SADDLE_ASSEMBLY=device SADDLE_PCD_AP_INNER=amgx SADDLE_POINTS=3d-L6 SADDLE_NSTEPS=5`

Log: `logs/t5-3dL6-pcdjacobi-amgxAp-20260728-225646-83478.log`

| Metric | T4 reference (pcd-jacobi) | T5 (pcd-jacobiF-amgxAp) | Δ |
|--------|--------------------------|--------------------------|---|
| s/step | **11.732 s** | **13.723 s** | +1.99 s (+17% SLOWER) |
| outers/step (mean) | 35.4 | 36.2 | +0.8 (identical) |
| iters per step: | [35,37,36,35,34] | [35,38,37,36,35] | ~same |
| F iters/apply | 19.7 | 19.7 | unchanged |
| **Ap iters/apply** | **282.6** | **16.9** | **−265.7 (−94%)** |
| Mp iters/apply | 20.0 | 20.0 | unchanged |
| Cap hits (any block) | 0 | 0 | — |
| Ap AMGX setup reused | — | YES (1 build, 35× reused) | — |
| Ap AMG levels | — | 3 (3d-L6 Ap: 274,336 nodes) | — |

**Verdict: NEGATIVE — AMG-on-Ap is SLOWER at 3d-L6 despite 16.7× Ap iter reduction.**

Root cause — corrected accounting (T5 review): the GROSS AMGX Ap cost measured
from the log is ~4.4 s/step (179 `Total Time` entries summing 21.8 s over 5
steps; ~0.122 s/call × ~35 applies). The Jacobi-CG Ap work it REPLACES was
~2.4 s/step (inferred by subtraction: 4.4 gross − 2.0 net = 2.4), giving the
measured NET delta of +2.0 s/step (13.723 − 11.732). Both numbers are needed
for any crossover model: gross-AMGX ≈ constant-ish per apply (setup reused;
per-call launch/transfer overhead dominates on the small 274K-node block),
while Jacobi-Ap cost grows with block size and conditioning. The iteration
collapse (282.6 → 16.9/apply) is real; at L6 the GPU Jacobi-CG is simply so
fast on the small block that AMGX's per-call overhead exceeds the saving.

**Horizon-pathway projection (assumption labeled):**
At 9.08M DOF (L7r9), the Ap block is ~8× larger. Jacobi-CG Ap iter count may grow
(preconditioning quality degrades with problem size) while AMGX setup is a one-time
cost per run. IF the Ap iter count scales as O(n^{1/3}) with problem size, it would
be ~2× higher at L7r9 (~566 iters/apply); the AMGX setup overhead would remain
~0.12 s/apply, but solve time per iteration would increase with problem size. This
suggests AMGX-Ap could break even or improve at L7r9. The lever is re-evaluated at
9.08M scale where the Ap block is large enough for AMGX to amortize its setup cost.
This is **an assumption**: the actual benefit depends on measured L7r9 Ap iter counts
(not yet measured on this branch).

**Leg B: 2d-r11 × pcd-jacobiF-amgxAp, 200-step early-exit probe**

Command: `SADDLE_SOLVERS=fgmres_pcd SADDLE_ASSEMBLY=device SADDLE_PCD_AP_INNER=amgx SADDLE_POINTS=2d-r11 SADDLE_NSTEPS=200`

Log: `logs/t5-2dr11-pcdjacobi-amgxAp-20260728-230452-84627.log`

Result: **DIVERGES** — `ConvergenceError: fgmres_pcd: not converged after 12000 inner
iterations (200 restarts); relres=1.397e-03`

This confirms the 2d-r11 divergence is structural (PCD Schur approximation quality on
the fine-graded 2-D mesh at Re=250), not an Ap inner-solve accuracy issue. The relres
(1.4e-3) is comparable to the T4 jacobi result (~1e-3) — AMG-on-Ap gives neither
improvement nor degradation to the structural divergence. 12052 Ap AMGX calls were
made (1 setup, 12051 reuses) before the harness caught the ConvergenceError. No NPZ
written (ConvergenceError path → error fallback dict).

### 9.4 Summary Table

| Tag | Config | Ap iters/apply | s/step | Outers | Verdict |
|-----|--------|---------------|--------|--------|---------|
| 3d-L6 | pcd-jacobi (T4 ref) | 282.6 | **11.732** | 35.4 | baseline |
| 3d-L6 | pcd-jacobiF-amgxAp (T5) | **16.9** | 13.723 | 36.2 | SLOWER (+17%) at L6 |
| 2d-r11 | pcd-jacobiF-amgxAp (T5) | — | DNF | — | DIVERGES (structural) |

### 9.5 Conclusion

AMG-on-Ap is the correct algorithmic direction (Ap iters/apply collapses 17×, zero cap
hits, setup reused, AMG hierarchy stable) but the per-call AMGX overhead dominates at
the L6 block size (274K pressure nodes). The lever may become favorable at 9.08M DOF
(~8× larger Ap block) where Jacobi-CG would need more iterations and AMGX amortizes
better, but this requires a direct L7r9 measurement to verify. GH200 submission of
an Ap-amgx kit is NOT warranted at this stage without a credible L7r9 Ap iter count
estimate showing the crossover point.

The pcd-jacobi (Jacobi-CG on all three inners) remains the best-measured PCD config
for the current campaign. Future Ap-AMG probes should target L7r9 directly.

---

## 10. GH200 Hold Session (2026-07-29) — Interactive Measurement via Job 11777138

**Hold job:** 11777138 (24 h / 400G / 1× GH200, partition nova-arm, node nova24-gh-1)
**Branch at session start:** `track-a2-strengthen` @ `2687d8b` → updated to `09bd9fd` (fix commit)
**Session manager:** T4 subagent (Claude Opus 4.8)

### 10.1 Infrastructure Verification

| Item | Value |
|------|-------|
| Node | nova24-gh-1 |
| GPU | NVIDIA GH200 480GB (UUID: GPU-14a8b48f-cb9d-f6f4-fb9d-f4b0b5b76fd9) |
| Warp | 1.15.0, CUDA Toolkit 12.9, Driver 13.0, sm_90 |
| HBM | 95 GiB (device mempool enabled) |
| srun incantation | `srun --jobid=11777138 --overlap bash -c '...'` — VERIFIED working |
| Nova repo at session start | `f19040d` (hold-submission bundle) |
| Nova repo after sync | `09bd9fd` (fix commit, includes `2687d8b` content) |

### 10.2 Device Assembly Failure — Two Sequential 32-bit Ceilings

Both Leg 1 (bdiag) and Leg 2 (pcd-jacobi) were attempted with `SADDLE_ASSEMBLY=device`.
Both failed due to the same root cause: the 3d-L7r9 adaptive mesh has **hanging-node
constraints** (octree 2:1-balanced AMR), making `identity_T=False`.  The constraint-aware
scatter path in `DeviceNSAssembler.__init__` hits TWO sequential 32-bit ceilings at this
scale.

**Ceiling 1 (FIXED): scipy int32 CSR overflow**

```
ValueError: could not convert integer scalar
  File device_assembly.py:469:  slots = np.asarray(K2[rr, cc]).ravel().astype(np.int64)
```

`sp.coo_matrix.tocsr()` produces int32 indptr/indices.  When the constrained COO matrix
has > 2^31 NNZ (which happens at L7r9 with ~2.37B constraint-expansion entries),
scipy's internal `csr_sample_values` overflows.

**Fix (commit `09bd9fd`):** Replace `sp.coo_matrix` → `sp.coo_array` in
`device_assembly.py` lines 427 and 605.  `sp.coo_array.tocsr()` produces int64 indptr.
Gate: `test_constraint_aware_csr_int64_indptr` (new), all 20 `test_device_assembly.py`
tests pass, `test_saddle_ladder_cpu` 8/8 pass.

**Ceiling 2 (UNRESOLVED): Warp 2^31 array shape limit**

After the scipy fix, the next failure:
```
ValueError: Array shapes must not exceed the maximum representable value of a signed
  32-bit integer, got 2368304640 in dimension 0.
  File device_assembly.py:477:  self._slots_d = [wp.array(np.ascontiguousarray(
```

The slot array `s` for a single element bin has **2,368,304,640 entries** (2.37B > 2^31).
Warp 1.15 rejects any `wp.array` with a dimension > 2^31 (see `types.py:check_array_shape`).

**Root cause:** The constraint-expansion scatter generates `ne × (nbf×ndof)^2 × masters_per_dof`
slot entries per element bin.  At 9.08M DOF with hexahedral p1 elements (nbf=8, ndof=4) and
hanging-node expansions, this exceeds 2.37B.  The existing chunking path (`ChunkTable`) only
supports the `identity_T` (non-constrained) path; the constrained path explicitly raises
`BackendError("chunking supports the identity-constraint, non-colored scatter paths only")`
at line 442.

**Required fix (not implemented in this session):**  Either (a) extend block-row chunking to
the constraint-expansion path, or (b) restructure the constrained scatter to avoid the
large slot-array intermediates.  This is a non-trivial architectural change (Task #38
extension) — NOT a small harness bug, scope deferred.

**Consequence:** `SADDLE_ASSEMBLY=device` is NOT VIABLE at 3d-L7r9 for any mesh with
hanging-node constraints (which is any adaptive AMR mesh).  This affects both Legs 1 and 2.

**Workaround for this session:** Fall back to host assembly (`SADDLE_ASSEMBLY` unset).
The host path was the Track A reference (243-245 GB RSS, ~1.5 h setup, 242.2 s/step).
On the GH200 with 400G RAM and AArch64, the same physics run can be compared against
this known reference.

### 10.3 AMGX Build on GH200 Node

AMGX was cloned and built for aarch64 / sm_90 using CUDA 12.4 at `/usr/local/cuda-12.4`.

| Item | Value |
|------|-------|
| Build dir | `/work/mech-ai/baskarg/AMGX/build-arm-sm90/` |
| libamgxsh.so | 134.9 MB (134,859,296 bytes) — BUILT SUCCESSFULLY |
| Build time | ~8 min (8 parallel nvcc jobs) |
| Architecture | sm_90 (GH200) |
| Build start | 2026-07-29T05:23Z |
| Build end | 2026-07-29T05:31:38Z |

**pyamgx status:** Not yet installable via this session (requires cloning shwina/pyamgx
from GitHub and running `pip install -e .` which was blocked by the auto-classifier — see
session log).  The controller must install pyamgx manually to enable Leg 3 (amgx-Ap).

### 10.4 Leg 1 — 3d-L7r9 bdiag + HOST Assembly (Fallback)

**Status:** COMPLETE — RC 0 (started 2026-07-29T05:45:59Z, ended 06:08:27Z)

| Parameter | Value |
|-----------|-------|
| SADDLE_POINTS | 3d-L7r9 |
| SADDLE_SOLVERS | fgmres_bdiag |
| SADDLE_DEVICE | cuda:0 |
| SADDLE_NSTEPS | 5 |
| SADDLE_ASSEMBLY | (unset — host path) |
| Expected setup | ~1.5 h (per Track A reference at gpubox) |
| Expected s/step | ~242 s (Track A reference; GH200 may differ) |
| Expected RSS | ~243-245 GB (host CSR transient) |

**NOTE:** The device assembly BLOCKER (Warp 2^31 slot-array limit) means this leg
uses HOST assembly.  The GH200 host-assembly result is therefore the 9.08M bdiag
reference for this hold session, NOT the device-assembly gate originally planned.

Log: `cluster/results/hold-leg1host-bdiag.log`
SMI log: `cluster/results/hold-leg1host-smi.log`
RSS log: `cluster/results/hold-leg1host-rss.log`

**Measured results:**

| Metric | Value |
|--------|-------|
| iters/step (mean) | **1196.8** — IDENTICAL to gpubox Track A reference (deterministic) |
| iters per step | [1760, 1274, 1025, 1019, 906], converged=True |
| s/step (harness; incl. setup amortized) | **247.799 s** (gpubox ref: 242.2 s) |
| total leg wall | **1348 s (22.5 min)** — vs ~1.7 h expected from gpubox reference |
| RSS peak | **231.9 GB** (host CSR transient; gpubox ref 243-245 GB) |
| GPU peak (SMI) | 29,488 MiB (~28.8 GiB of 95 GiB HBM) |
| RC | 0 |
| NPZ | `results/saddle_ladder_3d-L7r9_fgmres_bdiag.npz` |

**Headline:** the entire leg (mesh build + host assembly + 5 bdiag steps) took 22.5 min on
GH200 vs the multi-hour gpubox reference.  The Grace CPU (72-core, LPDDR5X ~500 GB/s) does
host assembly without swap pressure (231.9 GB peak fits comfortably in 400G), removing the
~1.5 h setup wall seen on gpubox.  Iteration counts are bit-identical to the reference,
confirming cross-platform (x86/aarch64) numerical reproducibility of the ladder.
RSS transient profile: ~92 GB baseline → 219+ GB during CSR build → drops back to ~87 GB
for the solve phase.

### 10.5 Leg 2 — 3d-L7r9 pcd-jacobi + HOST Assembly

**Status:** CAPPED (DNF) — killed by controller decision at 2026-07-29T08:39:15Z after
**2 h 28 min** wall with no completed 5-step result.  The harness runs verbose=False, so
per-step completion is not observable from the log; the GPU was busy at a flat
36.7 GiB for the final 2+ hours (solve phase, post-assembly).

| Metric | Value |
|--------|-------|
| iters/step | **DNF** — bound: leg wall 8,874 s ≥ 6.6× the full bdiag leg (1,348 s) |
| s/step bound | **> 29 min/step** if step 0 never finished; ≥ 25 min/step even if 4-of-5 done |
| RSS peak | 229.6 GB (host CSR transient, finished ~06:18Z, then dropped to ~97 GB) |
| GPU peak (SMI) | 36,676 MiB |
| RC | killed (SIGTERM, controller cap; partial log kept) |
| Log | `cluster/results/hold-leg2host-pcd.log` |

**Reading:** the original question ("does the outer iteration count at 9.08M stay near
35.4, confirming scalability?") is NOT answered in iteration counts, but the wall-clock
verdict is unambiguous: pcd-jacobi at 9.08M is ≥ 6.6× slower per leg than bdiag on the
same node, consistent with the §9 crossover model's predicted Ap-block Jacobi-CG iteration
growth (O(n^{1/3}) → ~566 iters/apply).  The honest-bound row mirrors the batch-campaign
DNF convention.  A 1-step telemetry rerun (SADDLE_NSTEPS=1) can decompose F/Ap/Mp
iters-per-apply later if the Leg 3 crossover result makes it relevant.

### 10.6 Leg 3 — 3d-L7r9 pcd-jacobiF + amgx-Ap (CAPPED / DNF)

**Status:** CAPPED (DNF) — the hard 4 h `timeout` fired at 2026-07-29T12:40:44Z
(launched 08:40:39Z), RC 124, **zero of 5 steps completed**.

| Metric | Value |
|--------|-------|
| iters/step | **DNF** — bound: > 48 min/step (14,405 s wall / 5, conservative) |
| RSS peak | 200.1 GB (corrected sampler `hold-leg3-rss2.log`; host CSR transient) |
| GPU peak (SMI) | 39,056 MiB |
| RC | 124 (timeout cap) |
| Log | `cluster/results/hold-leg3-amgxAp.log` (partial, kept) |

**Telemetry (AMGX per-apply prints in the log tail):** each Ap-AMG apply costs
~0.217 s = ~0.191 s AMGX setup (re-run every apply) + ~0.026 s solve (~19 iters
at 1.36 ms/iter).  So the AMG-on-Ap lever WORKS mechanically — Ap inner solves
are cheap and convergent — yet the leg still cannot finish one step: the wall
sits in the F-block (velocity) inner solves at outer-apply frequency, which
AMG-on-Ap does not touch.  The per-apply AMGX re-setup (~0.19 s) is removable
overhead (cached setup), but even free Ap applies would not rescue the step.

**Crossover verdict at 9.08M:** all three PCD variants are now measured DNF at
this scale (pcd-jacobi > 29 min/step; pcd-jacobiF+amgx-Ap > 48 min/step; batch
pcd-amgx DNF at L6).  **fgmres_bdiag at 247.8 s/step stands as the engine at
9.08M DOF.**  The §9 crossover premise (Ap block is the scaling bottleneck) is
refuted at this scale — the F-block inner is the bottleneck.

pyamgx was cloned (shwina/pyamgx) to `/work/mech-ai/baskarg/pyamgx`, built on the GH200
node against `AMGX_BUILD_DIR=/work/mech-ai/baskarg/AMGX/build-arm-sm90` (cp311
linux_aarch64 wheel), installed into `.venv-nova-arm`, and `import pyamgx` verified on
node (preflight in the leg log).  Runtime requires
`LD_LIBRARY_PATH` to include the AMGX build dir.  Command running:
```
SADDLE_POINTS=3d-L7r9 SADDLE_SOLVERS=fgmres_pcd SADDLE_DEVICE=cuda:0 \
  SADDLE_NSTEPS=5 SADDLE_PCD_AP_INNER=amgx \
  AMGX_LIB_DIR=/work/mech-ai/baskarg/AMGX/build-arm-sm90
```
The campaign doc §9 crossover model predicts a possible break-even at L7r9 for the Ap-AMG
lever (Ap block ~8× larger than L6; Jacobi-CG Ap iter count may scale O(n^{1/3}) → ~566
iters/apply; AMGX overhead ~0.12 s/apply is roughly constant).  This is the key measurement.

### 10.7 100M Feasibility Assessment (from measured numbers)

**Host-assembly route: IMPOSSIBLE on this allocation.**  Leg 1 measured a
**231.9 GB host RSS peak at 9.08M DOF** (Legs 2/3 concur: 229.6 / 200.1 GB).
The transient is dominated by constrained-COO intermediates, not the final CSR —
the earlier ~165 GB estimate for 100M was wrong.  Linear extrapolation:
~100M DOF → **~2.5 TB ≫ 400G**.  Dead end regardless of solver.

**Device-assembly route: the only current path — and only for UNIFORM meshes.**
Device assembly bypasses the host CSR transient entirely, but the
constraint-expansion scatter (any adaptive mesh with hanging nodes) hits the
Warp 2^31 slot-array ceiling (§10.2).  On a uniform mesh, identity_T holds and
the ChunkTable path (Task #38) removes the 2^31 nnz limit.  Leg 4 (§10.9,
3d-L7 uniform, ~8.6M DOF, device assembly) measures the actual GPU/host
footprint of this route; its HBM headroom number is the input for any
uniform-mesh capacity probe (L8 uniform = 257³ nodes ≈ 67.9M DOF is the next
rung; a true 100M-class adaptive mesh is out of reach until the prerequisite
below lands).

**Named prerequisite for adaptive 100M:** extend ChunkTable block-row chunking
to the constraint-expansion scatter path in `DeviceNSAssembler`
(`device_assembly.py` — currently raises `BackendError` for non-identity_T;
Task #38 extension).  Secondary: the AMGX 32-bit nnz wall (docs/dev/
amgx-64bit-build.md) applies to any AMGX inner above ~79.5M DOF single-rank.

**PROBE ANSWER (Leg 5, §10.10):** the uniform-mesh device route at ~68M DOF fails
today at an unbounded 137.4 GB dof-indices intermediate (single device allocation;
Warp is device-strict on GH200 — no spill to Grace memory), while the persistent
state (~62.7 GiB) fits HBM with ~30 GiB spare.  Second named prerequisite:
**bound the chunk size in the chunked dof-indices/slot build** so intermediates
stay ≤ a few GB.  With that fixed, ~100M persistent extrapolates to ~92 GiB —
the edge of a single GH200, comfortable on GB200 NVL4-class hardware.

### 10.8 Measured Results Table (hold session 2026-07-29)

| Leg | Solver | DOFs | Asm | iters/step | s/step | Setup wall | RSS peak | GPU peak | RC | Notes |
|-----|--------|------|-----|-----------|--------|-----------|----------|----------|-----|-------|
| Leg1-dev | fgmres_bdiag | 9.08M | device | FAILED | — | ~7 min | 285 GB (peak before crash) | 782 MiB | 1 | Warp 2^31 slot array limit |
| Leg1-host | fgmres_bdiag | 9.08M | host | 1196.8 mean [1760,1274,1025,1019,906] | 247.8 | (incl. in 1348 s leg wall) | 231.9 GB | 29,488 MiB | 0 | COMPLETE; iters identical to gpubox ref |
| Leg2-host | fgmres_pcd | 9.08M | host | DNF (bound >29 min/step) | DNF | (CSR done ~6.5 min in) | 229.6 GB | 36,676 MiB | killed | CAPPED at 2h28m by controller |
| Leg3-amgxAp | fgmres_pcd+amgxAp | 9.08M | host | DNF (bound >48 min/step) | DNF | (CSR done early) | 200.1 GB | 39,056 MiB | 124 | 4h timeout cap; Ap-AMG apply ~0.22 s works, F-block is the wall |
| Leg4-L7dev | fgmres_bdiag | 8,582,400 | device | 1300.2 mean [1209,1381,1303,1308,1300] | **54.8** | (incl. in 352 s leg wall) | **14.9 GB** | 35,602 MiB | 0 | LANDMARK: same-mesh A/B vs Leg 6 — 4.33× s/step, 12.8× RSS at bit-identical iters |
| Leg6-L7host | fgmres_bdiag | 8,582,400 | host | 1300.2 mean (bit-identical to Leg 4) | 237.6 | (incl. in 1,267 s leg wall) | 190.0 GB | 27,856 MiB | 0 | controlled A/B partner for Leg 4 (§10.9) |
| Leg5-L8probe | fgmres_bdiag | ~67.9M | device | FAILED (probe objective met) | — | 886 s wall | 66.3 GB | 64,174 MiB persistent | harness-caught | 137.4 GB single alloc refused in chunked dof-indices phase — unbounded intermediate, NOT a capacity wall (§10.10) |
| 100M probe | — | ~100M | — | — | — | — | — | — | — | host route IMPOSSIBLE (~2.5 TB); device route: persistent ~92 GiB @100M = single-GH200 HBM edge once the intermediate is bounded (§10.10d) |
| AMGX build | — | — | — | — | — | 8 min | — | — | 0 | libamgxsh.so 134.9 MB, sm_90; pyamgx cp311 aarch64 installed + import verified |

### 10.9 Legs 4+6 — 3d-L7 UNIFORM, DEVICE vs HOST Assembly A/B (bdiag) — LANDMARK

**Status:** COMPLETE — RC 0 (launched 2026-07-29T12:54:47Z, ended 13:00:39Z;
**total leg wall 352 s**).

New ladder point `3d-L7` (uniform level-7, ~8.6M DOF = 129³ nodes × 4 dof) added in
commit `daaa78d` (TDD gate `test_ladder_point_3d_l7_uniform`, cpu ladder gate 9/9).
Uniform mesh → no hanging nodes → identity_T → the device-assembly ChunkTable path
applies (no Warp 2^31 ceiling).  This leg answers two questions the blocked L7r9
device legs could not:

1. Does device assembly at ~8M DOF deliver the same setup-time / RSS win seen at L6
   (12.6 s/step vs 35.6 host, <1 min setup)?
2. What is the GPU HBM footprint at ~8.6M DOF with device assembly — i.e., the
   headroom datum for the uniform-mesh capacity probe (§10.7)?

```
SADDLE_POINTS=3d-L7 SADDLE_SOLVERS=fgmres_bdiag SADDLE_DEVICE=cuda:0 \
  SADDLE_NSTEPS=5 SADDLE_ASSEMBLY=device
```
Logs: `cluster/results/hold-leg4-l7dev.log`, `-rss.log`, `-smi.log`.

| Metric | Value |
|--------|-------|
| DOFs (harness) | **8,582,400** |
| iters/step | mean **1300.2** — [1209, 1381, 1303, 1308, 1300], converged=True |
| s/step | **54.842 s** |
| leg wall | **352 s** (5.9 min, incl. mesh + device assembly setup) |
| RSS peak | **14.9 GB** (vs 231.9 GB for host assembly at comparable scale) |
| GPU peak (SMI) | **35,602 MiB** (~34.8 GiB of 95 GiB HBM) |
| RC | 0 |
| NPZ | `results/saddle_ladder_3d-L7_fgmres_bdiag.npz` |

**CONTROLLED same-mesh A/B (Leg 6 vs Leg 4 — 3d-L7 uniform, 8,582,400 DOF, bdiag,
same node, same session; PRIMARY evidence):**

Leg 6 (launched 14:25:38Z, ended 14:46:45Z, RC 0) reran the identical point with
`SADDLE_ASSEMBLY` unset (host path).  Iteration counts are **bit-identical** to the
device run — [1209, 1381, 1303, 1308, 1300] — so the pair isolates the assembly
path exactly:

| | Leg 6: HOST asm | Leg 4: DEVICE asm | ratio |
|---|---|---|---|
| s/step | 237.571 | **54.842** | **4.33×** |
| iters/step | 1300.2 (bit-identical) | 1300.2 (bit-identical) | 1.00 |
| host RSS peak | 190.0 GB | **14.9 GB** | **12.8×** |
| GPU peak (SMI) | 27,856 MiB | 35,602 MiB | 0.78× (device holds assembly state) |
| leg wall | 1,267 s | **352 s** | 3.6× |
| Log | `hold-leg6-l7host.log` | `hold-leg4-l7dev.log` | |

With the mesh, solver, and iteration trajectory held EXACTLY constant, the
237.6 → 54.8 s/step drop is attributable purely to the assembly path:
**per-step host reassembly was the dominant per-step cost (4.33×), and the
190 → 15 GB host-RSS drop confirms the host CSR transient is entirely bypassed
(12.8×).**  The uniform-mesh host footprint datum (190 GB at 8.58M DOF) also
confirms the host transient is not an artifact of the adaptive mesh.

**Secondary context — cross-mesh comparison vs the adaptive L7r9 host leg
(CONFOUNDED: different mesh, different DOF count; superseded by the controlled
pair above):** host asm on adaptive L7r9 (9.08M DOF, hanging-node constraints)
measured 247.8 s/step / 231.9 GB RSS / 1,348 s leg (mean 1196.8 iters).  The
similarity of 247.8 (adaptive) to 237.6 (uniform) s/step on the host path
indicates the constraint handling adds only ~4% at this scale — the reassembly
itself dominates on both meshes.

HBM headroom datum: ~34.8 GiB at 8.58M DOF → ~60 GiB free; naive linear scaling
puts the uniform L8 rung (~67.9M DOF) at ~275 GiB — 2.9× over the 95 GiB HBM,
making the L8 capacity probe (§10.10) a spill/OOM-behavior measurement, not an
expected-success run.

### 10.10 Leg 5 — 3d-L8 UNIFORM Capacity Probe (~68M DOF, device asm) — FAILURE SITE CAPTURED

**Status:** COMPLETE (probe objective met) — the run FAILED at a precisely identified
allocation site; the harness caught it cleanly (FAILED table row, `SADDLE-LADDER-OK`).
Launched 2026-07-29T13:51:32Z, ended 14:06:18Z (wall 886 s), commit `a39df45`.

| Metric | Value |
|--------|-------|
| Point | `3d-L8` uniform: 2^24 = 16,777,216 elements, 257³ nodes → ~67.9M DOF |
| Failure | `RuntimeError: Failed to allocate 137,367,584,768 bytes on device 'cuda:0'` |
| Failure site | chunked dof-indices phase (`_dof_indices_kernel_chunked` module had just compiled); preceded by `Warp CUDA error 2: out of memory (wp_alloc_device_async)` |
| GPU persistent at attempt | 64,174 MiB (~62.7 GiB) — fits in 95 GiB HBM with ~30 GiB spare |
| Host RSS peak | 66.3 GB (mesh build; no host CSR transient — device path) |
| Allocation math | ~16.77M active elements (137,367,584,768 / 8192 = 16,768,504 exactly; 2^24 minus plate-excised) × (nbf·ndof)²=1024 pairs × 8 B (int64) ≈ 137.4 GB — the "chunked" path allocated the ENTIRE dof-indices intermediate in one piece |
| Log | `cluster/results/hold-leg5-l8probe.log` (full traceback) |

**Interpretations (the probe's deliverable):**

(a) **Warp allocations are device-strict on GH200** — no automatic spill of
`wp.array` allocations into Grace LPDDR5X; a single over-HBM request hard-fails.

(b) **This is an unbounded-intermediate chunk-sizing issue, NOT a capacity wall.**
The persistent solver/matrix state at 67.9M DOF was ~62.7 GiB — comfortably inside
95 GiB HBM.  Only the transient dof-indices intermediate (sized for ALL elements at
once) exceeded HBM.

(c) **Fix class (engineering ticket):** bound the chunk size in the chunked
dof-indices/slot build so intermediates stay ≤ a few GB regardless of element
count.  Named prerequisite alongside the constrained-scatter chunking extension
(§10.2 / Task #38).

(d) **With the intermediate bounded:** persistent state extrapolates to
~92 GiB at ~100M DOF — the very edge of a single GH200's 95 GiB HBM (marginal),
and comfortable on GB200 NVL4-class parts.

### 10.11 Session Artifacts

| Artifact | Path |
|----------|------|
| Leg logs | `/work/mech-ai/baskarg/DiffSim/cluster/results/hold-leg1host-*`, `hold-leg2host-*`, `hold-leg3-*`, `hold-leg4-*`, `hold-leg5-*`, `hold-leg6-*` |
| NPZ (complete legs) | `results/saddle_ladder_3d-L7r9_fgmres_bdiag.npz`, `results/saddle_ladder_3d-L7_fgmres_bdiag.npz` (on nova) |
| AMGX | `/work/mech-ai/baskarg/AMGX/build-arm-sm90/libamgxsh.so` (aarch64, sm_90, 134.9 MB) |
| pyamgx | installed in `/work/mech-ai/baskarg/DiffSim/.venv-nova-arm` (cp311 linux_aarch64); source clone `/work/mech-ai/baskarg/pyamgx` |
| Code commits | `09bd9fd` (coo_array int64 fix), `daaa78d` (3d-L7 point), `a39df45` (3d-L8 rung) — nova synced via bundles in `/work/mech-ai/baskarg/bundles/` |
