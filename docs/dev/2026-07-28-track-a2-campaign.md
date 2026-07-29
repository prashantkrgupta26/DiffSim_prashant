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

The binding verdict gate (SUCCESS: 9.08M pcd-jacobi outer growth from 3d-L6 ≤ ~2×
AND s/step tractable) requires the 3d-L7r9 GH200 legs. The controller submits
both sbatch kits after:
1. Confirming 3d-L6 results above (device assembly works — Phase 1 done).
2. Submitting bdiag kit first (confirms device-assembly setup time + RSS).
3. Submitting pcd-jacobi kit (pyamgx NOT required — jacobi inner has no deps).

**Fill in here:**
- 3d-L7r9 bdiag device-asm: iters/step, s/step, setup time, RSS peak, job ID
- 3d-L7r9 pcd-jacobi device-asm: outer iters/step, s/step, inner_stats (F/Ap/Mp iters/apply, cap_hits), SMI peak, job ID
- Verdict: PASS / FAIL / partial outcome + rationale

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
