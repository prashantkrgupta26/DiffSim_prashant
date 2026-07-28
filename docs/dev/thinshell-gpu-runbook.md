# ThinShell GPU Runbook — Consolidated Operator's Guide

**Branch:** thinshell-gpu  
**Date:** 2026-07-26  
**CI state at consolidation:** 59 passed, 4 skipped (371 s, Mac warp-CPU, 2026-07-26)

Cross-links:
- 2-D thin-plate (Re=250, CPU + GPU campaign): [`docs/dev/p2r1a-thin-plate-runbook.md`](p2r1a-thin-plate-runbook.md)
- 3-D thin-plate + GH200 ladder: [`docs/dev/p2r1c-3d-plate-runbook.md`](p2r1c-3d-plate-runbook.md)

---

## 1. Targets and Environment Preparation

### 1.1 gpubox (WSL2, RTX 6000 Ada, 48 GiB)

**Hardware:** 2× NVIDIA RTX 6000 Ada Generation (48 GiB VRAM each, sm_89).  
**OS/CUDA:** WSL2, CUDA 12.9, Driver 13.2.

The critical WSL2 toolkit library path must be set so that Warp can locate the
CUDA runtime shared libraries.  Set it in your shell profile or prepend inline:

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH}
```

This is required for every GPU invocation on gpubox.  Without it, `warp.init()`
will fail with a missing-CUDA-runtime error.

**venv:** `.venv` (the standard project venv; CUDA extras installed).

**Smoke command** (2-D + 3-D, both solvers):

```bash
ssh gpubox "cd /home/bglab/Baskar/DiffSim && \
    LD_LIBRARY_PATH=/usr/lib/wsl/lib:\$LD_LIBRARY_PATH \
    DEVICE=cuda:0 MONO_SOLVER=cudss ASSEMBLY=device PPE_SOLVER=gpu_cg \
    .venv/bin/python tests/gpu_smoke_thinshell.py"
```

Expected output: `SMOKE-OK 2d-mono 2d-proj 3d-mono 3d-proj`

**NCCL workaround** (required if `gpu_cg` is in use with multi-GPU setup):

```bash
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_CUMEM_ENABLE=0
```

These env vars are harmless for single-GPU use and required if gpu_cg touches
multi-GPU. Prepend to any production command on gpubox as a precaution.

---

### 1.2 nova A100 (partition `nova`, x86_64)

**Hardware:** NVIDIA A100-PCIE-40GB (39 GiB, sm_80, x86_64).  
**CUDA/Driver:** 12.9 / 13.0.  
**venv:** `.venv-nova`

**Smoke sbatch:**

```bash
sbatch cluster/slurm/thinshell_a100_smoke.sbatch
```

Job 11764188 verified GREEN: `SMOKE-OK 2d-mono 2d-proj 3d-mono 3d-proj` (44 s,
including cold Warp kernel compilation). Exit 0.

---

### 1.3 nova GH200 (partition `nova-arm`, aarch64)

**Hardware:** NVIDIA GH200 480GB (95 GiB HBM3, sm_90, aarch64).  
**CUDA/Driver:** 12.9 / 13.0.  
**venv:** `.venv-nova-arm`  
**Warp kernel cache:** `.warp-cache-arm` (aarch64 kernels; separate from x86 cache).

GH200 requires the RPM library path active in the sbatch (already wired in the
provided sbatch script).

**Smoke sbatch:**

```bash
sbatch cluster/slurm/thinshell_gh200_smoke.sbatch
```

Job 11764189 verified GREEN: `SMOKE-OK 2d-mono 2d-proj 3d-mono 3d-proj` (44 s).
Exit 0.

**Note on GPU quota:** nova's mech-ai account cap is 19 GPUs (GrpTRES), frequently
saturated. `AssocGrpGRES` pending delays are expected during peak hours. The
`gh200-hold12` placeholder job (job 11763434, `cluster/slurm/gh200_hold12.sbatch`,
pure sleep for 12 h) was cancelled during Task 9 to free a slot. Re-submit if a
node reservation is needed: `sbatch cluster/slurm/gh200_hold12.sbatch`.

---

## 2. Knob Matrix

### 2.1 2-D Thin-Plate Driver (`tests/p2r1a_thin_plate_flow.py`)

| Env var | Meaning | Notes |
|---------|---------|-------|
| `MONO_SOLVER` | Monolithic saddle solver | `cudss` (GPU direct), `fused` (GPU BiCGStab), `splu` (CPU) |
| `PRED_SOLVER` | Projection predictor solver | `cudss` (production), `fused` (diverges at Re=250/L9), `splu` (CPU fallback) |
| `PPE_SOLVER` | Pressure-Poisson solver | `gpu_cg` (GPU Jacobi-CG), `splu` (CPU) |
| `ASSEMBLY` | Assembly device | `device` (GPU kernels), `host` (CPU CSR) |
| `DEVICE` | Warp device string | `cuda:0`, `cuda:1`, `cpu` |
| `BASE_LEVEL` | Octree base level | e.g. `7` (128² base) |
| `REFINE_LEVEL` | Near-plate refinement | e.g. `9` (512² near plate) |
| `WAKE_LEVEL` | Wake-region refinement | e.g. `9` |
| `L_INV` | Domain size (1/L, integer) | `16` = unit square (default), `32` = half-size (confinement probe) |
| `NSTEPS` | Number of BDF2 steps | |
| `DT` | Time step | |
| `NU` | Kinematic viscosity | **See §3: must be in octree units** |
| `U_INF` | Freestream velocity | |
| `PLATE_XC` | Plate center x (octree units) | |
| `PLATE_YC` | Plate center y (octree units) | |
| `PLATE_L` | Plate half-length (octree units) | |
| `PERT_EPS` | Transverse kick amplitude (fraction of U_inf) | Required to seed shedding |
| `PERT_T_END` | Duration of kick (time units) | |
| `T_START` | Post-transient averaging start | |
| `SOLVER` | `mono`, `proj`, or `both` | `both` runs compare_solvers |
| `COMPARE` | `1` to enable compare_solvers | Alias for `SOLVER=both` |
| `VERBOSE` | `1` for per-step output + timing | Silent by default |

### 2.2 3-D Thin-Plate Driver (`tests/p2r1c_thin_plate_flow_3d.py`)

| Env var | Meaning | Notes |
|---------|---------|-------|
| `MONO_SOLVER` | Monolithic saddle solver | Same options as 2-D |
| `ASSEMBLY` | Assembly device | `device` or `host` |
| `DEVICE` | Warp device string | |
| `LEVEL` | Octree level (uniform) | |
| `NSTEPS` | BDF2 steps | |
| `DT` | Time step | |
| `NU` | Kinematic viscosity | |
| `U_INF` | Freestream velocity | |
| `PLATE_XC/YC/ZC` | Plate center | |
| `PLATE_HALF_Y/Z` | Plate half-dimensions | |

### 2.3 3-D Projection Driver (`tests/p2r1c_thin_plate_flow_3d_projection.py`)

| Env var | Meaning | Notes |
|---------|---------|-------|
| `PPE_SOLVER` | PPE solver | `gpu_cg` (GPU CG), `splu` (CPU) |
| `PRED_SOLVER` | Predictor solver | `cudss` (production for GPU), `splu` (CPU, default); wired as of commit a40b329 |
| `DEVICE` | Warp device string | |
| `LEVEL` | Octree level | |
| `NSTEPS` | BDF2 steps | |
| (other geometry vars same as 3-D mono driver) | | |

### 2.4 GH200 Ladder Driver (`tests/gpu_gh200_ladder.py`)

Three-phase bluff-body + thin-plate capacity ladder. Launched via
`cluster/slurm/thinshell_gh200_ladder.sbatch`. Controlled by hard-coded rung
sequences inside the script; no env-var knobs.

---

## 3. THE UNIT-SYSTEM RULE

> **All quantities must be in OCTREE UNITS (the [0,1]^d native domain).  
> Paper-physical quantities do NOT transfer directly.**

The octree solver uses the unit square [0,1]² (2-D) or unit cube [0,1]³ (3-D)
as its native domain. Physical problem parameters — viscosity `nu`, plate
geometry, time — MUST be re-expressed in octree units before passing them to
the driver.

### The RE250_CONFIG Unit Bug

`RE250_CONFIG` in `tests/p2r1a_thin_plate_flow.py` (the configuration that
existed before the campaign) contained:

```
nu      = 0.004   # = 1/250  (paper-physical, assuming L_phys=1, U=1)
t_end   = 50.0    # paper-physical seconds (50+ shedding periods)
PLATE_L = 0.0625  # = 1/16 in the unit square — CORRECT octree value
```

**The bug:** `nu=0.004` is correct for Re=250 if the plate physical length is
L=1. But in the octree domain, the plate is L=1/16=0.0625 in octree units.
The effective plate Reynolds number is:

```
Re_eff = U * L_octree / nu = 1.0 * 0.0625 / 0.004 = 15.6
```

At Re≈16, the wake is steady (no shedding). This explains the O1 result: the
production run at `nu=0.004` with the corrected octree plate size produced
steady Cl→0 and Cd≈6.70 (consistent with Re~16 physics — correct physics,
wrong Re).

### Corrected Recipe

The consistent-octree configuration requires:

```
nu   = U * L_octree / Re = 1.0 * (1/16) / 250 = 2.5e-4
x_c  = 5/16 = 0.3125   (plate center in [0,1]² at 5 plate-lengths upstream)
t in L/U units, where L = L_octree = 1/16
St   = f * L_octree / U   (Strouhal number uses the octree plate length)
```

See `tests/gpu_re250_corrected.py` (commit e2c2675) for the corrected driver
invocation.

**Verification:** The corrected config probe immediately produced shedding
(Cl_std 2.7e-2, St=0.219, Cd_mean=5.33 over t=[1.25,3.5] — a short window).
The extended 16k-step run gave Cd_mean=5.47, St=0.203 (t≥3, ~17 periods).

---

## 4. Re=250 2-D Campaign Results

### Context

All runs below use the corrected-octree config (nu=2.5e-4, x_c=5/16) unless
marked MISCALIBRATED. Mesh: BASE_LEVEL=7, REFINE_LEVEL=9 (adaptive, ~30k nodes),
WAKE_LEVEL=9. Solver: monolithic cudss + device assembly, BDF2.

**Literature targets (ThinShell.pdf Table 1; Najjar & Balachandar 1995):**
Cd = 3.36 (lit mean); accepted band [3.29, 3.45]. St ≈ 0.15 (lit); accepted band [0.13, 0.17].

### Results Table

| Run | Config | Cd_mean | St | Steps / t_end | Shedding | Notes |
|-----|--------|---------|----|---------------|----------|-------|
| MISCALIBRATED (O1) | nu=0.004, BDF2 dt=5e-4, 100k steps | 6.70 | 2.0 (artifact) | 100k / t=50 | NO (Cl → 1e-9) | RE_eff≈15.6; steady-branch physics correct at that Re |
| r9 probe (corrected) | nu=2.5e-4, dt=5e-4, 16k steps | 5.47 | 0.203 | 16k / t≈8 (avg t≥3, ~17 periods) | YES (Cl_std 3e-2) | Window-insensitive; gap vs lit = resolution+confinement |
| r10 (corrected) | nu=2.5e-4, REFINE_LEVEL=10 | 5.72 | 0.203 | ~16k | YES | Cd +4.6% vs r9; St unchanged — roughly mesh-converged |
| L_INV=32 confinement probe | nu=2.5e-4, r10 cells/plate, blockage 3.1%, 5L upstream / 27L wake | 5.09 | 0.1875 | 16k | YES | Both move TOWARD lit; residual +51% Cd remains |

**Verdict:**
- Shedding is **robust** at Re=250 under the corrected config.
- Results are **resolution-converged** (r9→r10: Cd change +4.6%, St 0.0).
- Confinement is **measurable but not dominant** (L_INV=16→32: Cd 5.72→5.09, St 0.203→0.1875).
- **Residual gap** vs literature (Cd +51%, St +25% at L_INV=32): primary suspect is SBM-force systematic — the alpha=50 Nitsche penalty introduces a leak-drag / traction bias observable in `surrogate_traction`. 
- **Discriminating diagnostic (proposed):** compare momentum-deficit CV drag (integral of `u*(U_inf - u)` across a wake cross-section) vs surrogate-traction force on the same saved field (`results/re250_*_hist.npz` + VTU exist). If these two estimates agree with each other but both exceed the literature, the geometry is the driver (domain blockage, finite-L plate vs infinite-span). If they disagree, traction-bias dominates.

### dt-Ladder (BDF2 stability, t=0.1 horizon, mono cudss+dev-asm)

| dt | Multiplier vs dt=5e-5 | CFL | Steps/period (St≈0.15) | Cd(T) rel-err vs dt=5e-5 ref | Verdict |
|----|-----------------------|-----|------------------------|-------------------------------|---------|
| 5e-5 | 1× (reference) | ~0.026 | ~8333 | — | Reference |
| 1e-4 | 2× | ~0.051 | ~4167 | small | STABLE |
| 2.5e-4 | 5× | ~0.128 | ~1667 | moderate | STABLE |
| 5e-4 | 10× | 0.256 | ~833 | 5.9% | STABLE (production choice) |
| 1e-3 | 20× | 0.51 | ~417 | larger | STABLE (finite, positive) |

All rungs through dt=1e-3 (CFL 0.51) produced finite, stable results.

**Production setting:** BDF2 dt=5e-4 (10× speedup), NSTEPS=100000, t_end≈50 in physical/L/U
units. At this dt, Cd_mean is expected within ~6% of the fine-dt reference (transient-window
metric; the VMS tau transient term `(2b0/dt)^2` contributes to this sensitivity).

**Caveat:** If Cd_mean falls outside the [3.29,3.45] literature band after averaging, rerun
at dt=2e-4 to quantify the dt contribution before concluding physics residual.

---

## 5. Projection Status

### 2-D Projection — STRUCTURAL DIVERGENCE at Re=250/L9

The projection leg (LeraySBMShellStepper + gpu_cg PPE + cudss predictor) was run under
both the miscalibrated and the corrected Re=250 config. **Both diverge.**

- **With miscalibrated config (old nu=0.004):** Cd diverges to ~1e22 by step ~20 (Task 7 after-preflight).
- **With corrected config (nu=2.5e-4):** Cd diverges to ~1e25 by step 109 (Task O6, nova A100 job 11765210).
  The divergence is unambiguous and predates any knob tuning.

**Diagnosis:** structural to the current projection split at this Re/mesh (Re=250, L9 adaptive).
The fused predictor (BiCGStab) also diverges — relres 2.3e15 at step 1, breakdown at `rho`
(Task 7 preflight). The cudss predictor runs but the projection coupling itself diverges.

**Research-track suspects:**
1. p'-outflow scheme completeness in the 2-D projection path: the 3-D fix (p2r1c outflow-BC
   p'-scheme) may not be fully ported to the 2-D `LeraySBMShellStepper` path.
2. Lagged-p* split non-convergence at Re=250 (known: backflow beta=0.5 tuned at Re=100;
   "not universal" per ladder docs).
3. Inner iteration divergence under consistent_projection=True on the large adaptive mesh.

**Knob probes** (O6 task on nova A100): job 11765210 diverged on P0 by step 109; subsequent
knob probes (inner_relax, inner_max, dt, consistent_projection=False) were queued but their
completion within the 3-hr SLURM wall was uncertain. Recorded as pending — low marginal value
vs structural fix.

**6b device fallback validated:** The `cudss` predictor path silently fired the host-fallback
(`backflow slots absent; re-entering host predictor`) as designed. This confirms the 6b
forced-fallback mechanism works in production.

### 3-D Projection — L5 WORKS; L6 ALLOC_FAILED on predictor

After wiring PRED_SOLVER into the 3-D projection driver (commit a40b329):

- **L5 (LEVEL=5, ~131K DOF): FULL-GPU PROJECTION WORKS** — cudss predictor + gpu_cg PPE,
  artifacts written. This is the current practical ceiling for 3-D GPU projection.
- **L6 (LEVEL=6, ~1M DOF): predictor cudss ALLOC_FAILED** on gpubox 48 GiB. Consistent with
  the GH200 observation that the direct-factor transient peak at L6 (~51 GiB) barely fits on
  95 GiB but cannot fit on 48 GiB.

**Implication:** L6+ 3-D projection requires an iterative predictor (device FGMRES or AMGX).
This is the R2b block-preconditioner track.

---

## 6. Capacity Map

### 6.1 GPU Memory Envelope

| Target | VRAM | cuDSS 3-D saddle wall | Host mesh-build note |
|--------|------|----------------------|----------------------|
| gpubox RTX 6000 Ada | 48 GiB | ~1M DOF expected limit (L6 cudss not tested; L5 ~131K DOF confirmed OK; see L6 fused result below) | WSL2 host RAM not cgroup-limited in production use |
| nova A100-PCIE | 39 GiB | lower than gpubox — use only through L5 for 3-D direct | n/a (slurm job memory) |
| nova GH200 | 95 GiB HBM3 | **between ~1.1M and ~2.9M saddle DOF** (1.08M confirmed OK at ~51 GiB transient; 2.92M ALLOC_FAILED at 78.3 GiB peak; 8.46M ALLOC_FAILED early) | r7b9 adaptive OOM: MaxRSS 209.7 GB > 200G cgroup; GPU flat 13.5 GiB [209.7 GB was cross-leg accumulation — mesh build itself 3.71 GB; see §6.2 + WP0 §2.1] |

**cuDSS wall is fill/bandwidth-dependent, not a fixed DOF count.** The same 95-GiB GH200 fits
1.08M DOF (uniform L6) but fails at 626K DOF adaptive (a5r8, ~23 GiB transient) vs fitting it —
wait, a5r8 at 626K DOF succeeded (23 GiB transient, 9.8 GiB after). The discrepancy is in fill:
the adaptive mesh has a different sparsity pattern than the uniform mesh.

Revised numbers from Task 8b: fits at 1.08M (uniform) and 626K (adaptive a5r8); fails at
2.92M (adaptive a6r9) and 8.46M (uniform L7). The wall for this P1 saddle on GH200 is
**between ~1.1M and ~2.9M DOF** for the pathological fills tested.

### 6.2 Solver Decision Tree

```
Is the system 2-D thin-plate (Re=250)?
  ├─ DOF < ~800K (L7/L9 adaptive ≈ 30K nodes, 120K DOF): use MONO_SOLVER=cudss + ASSEMBLY=device
  └─ DOF > 800K: out of current scope (R2b)

Is the system 3-D thin-plate (uniform)?
  ├─ L4–L5 (~17K–131K DOF, < ~400K saddle DOF):
  │     Monolithic: MONO_SOLVER=cudss ASSEMBLY=device  →  OK (gpubox, A100, GH200)
  │     Projection: PPE_SOLVER=gpu_cg PRED_SOLVER=cudss  →  L5 confirmed OK (gpubox)
  ├─ L6 (~1M DOF, ~4M saddle):
  │     Monolithic cudss: ALLOC_FAILED expected on gpubox; uncertain on GH200
  │     Monolithic fused: MONO_SOLVER=fused ASSEMBLY=device  →  13.1 s/step, 3-step march (long-march stability unconfirmed)
  │     Projection cudss predictor: ALLOC_FAILED on gpubox 48 GiB; needs GH200
  └─ L7+ (~8M saddle): cuDSS ALLOC_FAILED on GH200; fused-BiCGSTAB path untested at this scale

Host mesh-build ceiling (GH200, nova):
  r7b9 (base-L7 + band-L9 adaptive): MaxRSS 209.7 GB > 200G cgroup  →  oom_kill
    [SUPERSEDED by WP0 (job 11771926): the 209.7 GB was cross-leg accumulation in the
     single-process ladder; the r7b9 mesh build itself measures 3.71 GB/94 s on Grace
     (mesh build exonerated) — see docs/dev/2026-07-27-100m-gh200-readiness.md §2.1.]
  Mitigation: raise --mem to approach the 480 GB Grace socket limit, or slim mesh-build intermediates
```

### 6.3 L6 Monolithic Fused — 3-Step Caveat

`MONO_SOLVER=fused ASSEMBLY=device` at L6 (~1M DOF) produced Cd=+22.79 in 3 steps (13.1 s/step)
on gpubox. This is a genuine result: the device BiCGStab converged on the nonsymmetric α=50
saddle. However, **only 3 steps were marched**; long-march stability (hundreds to thousands of
steps) is unconfirmed. This finding contradicts the initial expectation (convergence failure for
the nonsymmetric saddle) and is the iterative-mono-beyond-cudss-wall candidate path. Needs a
longer-march validation run before being declared production.

Contrast: fused predictor DIVERGED on the 2-D Re=250 projection predictor (relres 2.3e15, step 1
on the L9 Oseen system) — record both. The difference is likely the system conditioning
(3-D L6 saddle vs Re=250 fine-mesh nonsymmetric Oseen at L9).

---

## 7. Known Follow-Ups

### R2b Track (Iterative Solver Scale-Up)

1. **Iterative predictor for L6+ projection:** wire device FGMRES or AMGX as PRED_SOLVER=fgmres.
   Unblocks 3-D projection at L6+ (currently ALLOC_FAILED on cudss predictor at 48 GiB gpubox).
2. **Long-march validation of fused-BiCGStab at L6 3-D monolithic:** run 100+ steps to confirm
   convergence quality and step-to-step stability.

### Solver / Memory

3. **cudssDeviceMemHandler oversubscription:** wire cuDSS's own device-memory handler to a managed
   pool (solver-path change, not a Warp change). Required for true GPU-managed oversubscription
   beyond the cuDSS direct-factor wall. Distinct from `wp.set_device_allocator` (governs Warp
   arrays only; cuDSS allocates its factors internally via nvmath — the managed lever does NOT
   transfer to the torch/nvmath-cuDSS path).

### Host Mesh-Build Scaling

4. **Host mesh-build memory for base-L7 adaptive:** r7b9 exceeded 200G host cgroup during
   the HOST-side octree/mesh/constraints build (GPU idle at 13.5 GiB). Path: raise SLURM `--mem`
   toward 480 GB (Grace socket limit) and/or slim the host mesh-build intermediates. This is a
   binding constraint on the 100M-DOF path.

### Physics / Validation

5. **Re=250 SBM-force systematic:** discriminate via momentum-deficit CV drag vs
   `surrogate_traction` on the same saved field (`results/re250_*_hist.npz` + VTU exist).
   Alpha sweep (alpha<50) secondary. **RESOLVED — see verdict below.**

#### 2026-07-27 Leak-drag discriminator — VERDICT

**Instrument trail** (all GPU legs on gpubox RTX 6000 Ada, `tests/gpu_leakdrag_discriminator.py`):

(a) **3-leg α sweep** (8000 steps/leg, log `leakdrag-20260726-232649-53567.log`, ~45–53 min/leg):

| alpha | Cd_surr | CV(4L) | \|leak\| | St |
|-------|---------|--------|----------|-----|
| 20 | 4.228 | 2.619 | 7.7e-3 | 0.1875 |
| 50 | 5.437 | 3.090 | 4.7e-3 | 0.2188 |
| 100 | 5.982 | 3.565 | 5.0e-3 | 0.2188 |

Original {4,6,8}L box columns **withheld** — 6L/8L were off-domain (upstream caps at 5L for
x_c=5/16; a probe-design bug — the zero-meaned columns tripped the spread self-check exactly
as designed). Only the 4L column is valid.

(b) **Corrected-box leg** (α=50, margins {2,3,4}L): CV = 2.855 / 2.954 / 3.090 — spread 7.9%
(>5% gate), monotone in box size.

(c) **Double-window leg** (16k steps, ~2.9× window): CV = 2.853 / 2.980 / 3.209 — spread 11.8%,
**WORSE** ⇒ unsteady-residual hypothesis **REFUTED**; the line-quadrature CV method hit its
limit on the stabilized weak-divergence field (real CV integration systematic).

(d) **FINAL verdict leg** (LD-5 consistent-reaction arbiter, α=50, 8000 steps,
log `leakdrag-verdict-20260727-091605-26155.log`): **Cd_reaction = 2.3440** on both
plate-enclosing indicator sets (agreement 7.01e-15 ≤ 1e-6, **gate PASSED**),
**Cd_surr = 5.4369**, **St = 0.2188**. Printed verdict line:
`VERDICT: observable-overestimates+alpha-insensitive`, `LEAKDRAG-OK`.

**RESOLVED (final review 2026-07-27): the 2.5996-vs-2.3440 difference is an averaging-window asymmetry, NOT a code delta: the LD-5 verdict leg averaged the reaction over ALL 8000 steps (incl. the from-rest transient; its own comment says 'for gate') while its Cd_surr used the post-t_start window — the TD probe windows BOTH post-t_start. Ratio 2.5996/2.3440 = 1.109 matches the discrepancy exactly; git diff 4acaae1..0c8243f -- src/ is empty (no code delta). The developed-window value (2.5996) is the physically appropriate one. The earlier 'code-delta suspect' text above is superseded by this finding; the filed re-run diagnostic is unnecessary.**

**VERDICT: observable-overestimates** — the `surrogate_traction` observable reads **2.3×** the
variationally-consistent Nitsche reaction; the physics-carrying instruments (reaction 2.34,
CV 2.9–3.2) sit at/below literature 3.36. OPEN QUESTION (sharp form): the consistent reaction itself is ~30% BELOW literature — the consistent observable may UNDERestimate; the traction dissection must adjudicate both directions, not only the surrogate's excess. Two mandatory caveats: (i) the printed
"alpha-insensitive" tag is an artifact of the single-α verdict leg — the 3-leg sweep measured
**alpha-SENSITIVE** (+41% surr, +36% CV over α 20→100), which is the standing label;
(ii) the agreement gate is a mechanics self-check (zero by construction, per the documented
scoping in the probe), not set-independence.

**Follow-ups:** (a) traction-observable dissection — shifted-face σ·n integration vs the
consistent Nitsche functional; decompose the 2.3× gap (candidate terms: penalty virtual work,
adjoint-consistency, staircase-face area weighting) — **DONE, see the 2026-07-27 traction
dissection below**; (b) the
reaction-vs-CV ~25% gap (penalty virtual work vs divergence-error flux) as a secondary
reconciliation item; (c) α-scaling investigation (α~Pe·p² vs fixed 50) stands.

**Instrument-development cost (honest):** the probe shipped with a box-margin design bug
({4,6,8}L against a 5L upstream cap — caught by its own spread self-check, margins corrected
to {2,3,4}L, commit 969f391) and a printer KeyError (hardcoded tags) that crashed the final
table on two legs (results recovered from npz; printer fixed dynamically). Both fixed; neither
affected the physics numbers.

#### 2026-07-27 Traction dissection — term attribution + resolution/blockage verdict

**Probe:** `tests/gpu_traction_dissect.py` (commit 802590e), gpubox RTX 6000 Ada,
log `tractdissect-20260727-121213-49595.log`. All four legs GREEN, `TRACTDISSECT-OK`.
Per-leg npz on the box: `results/tractdissect_D{1..4}.npz`.

**Per-leg term table (verbatim from the probe):**

```
 Tag   Cd_rxn  consistency+adjoint              penalty             backflow  Cd_surr   bridge      St        dt       s
  D1   2.5996             0.070377             2.529259             0.000000   5.4369  77.2537  0.2188  5.00e-04  2042.4
  D2   2.8098             0.114229             2.695620             0.000000   5.8385  51.1125  0.2188  5.00e-04  2166.9
  D3   3.0483             0.129051             2.919280             0.000000   6.3351  49.0900  0.1875  5.00e-04  2028.9
  D4   2.4336             0.065880             2.367731             0.000000   5.0844  77.1765  0.1875  2.50e-04  2130.4
```

Legs: D1 = r9/α=50 baseline (L_inv=16, blockage 6.25%); D2 = r10; D3 = r11; D4 = L/32
plate at 32 cells/plate (r10-matched), blockage halved to 3.125%. **dt_used per leg:**
D1/D2/D3 = 5e-4 (D3 needed **NO dt fallback** at r11), D4 = 2.5e-4 (by design).
**Partition gates:** the in-march per-step gate `1e-12·max(1,|Cd_total|)` held on every
leg; post-hoc max abs residuals from the npz: D1 9.18e-13, D2 1.13e-12, D3 1.14e-12,
D4 1.12e-12 (values above 1e-12 absolute are within the enforced relative gate at
|Cd| ≈ 2.4–3.0).

**HEADLINE — the PENALTY term carries the force.** On every leg the α=50 penalty block
contributes ~96–97% of the consistent reaction (e.g. D1: 2.5293 of 2.5996); the
consistency+adjoint class is nearly inert (2.7–4.2%); backflow is identically zero.
Numerically, the "variationally-consistent reaction" at this configuration IS penalty
virtual work.

**Bridge ratio (bare σ·n vs consistency class):** 77.25 / 51.11 / 49.09 / 77.18 — FAR
from 1 on every leg. Bare σ·n does NOT approximate the consistency-class term, so the
2.3× surrogate excess cannot be read as "consistency-carried force plus penalty/adjoint
virtual work on top". Per the design's interpretation rule, the shifted-face σ·n
integration itself is implicated: at α=50 the surrogate and the reaction measure
essentially different functionals.

**Resolution trend (spec decision rule applied verbatim):** Cd_rxn(r9→r10→r11) =
2.5996 → 2.8098 → 3.0483; increments +0.2102, then +0.2385. The sequence is MONOTONE
toward 3.36 but the increments are NOT shrinking (they grew ~13%). The spec rule —
"monotone toward 3.36 with shrinking increments ⇒ deficit = resolution, extrapolated
value recorded; flat/oscillating ⇒ resolution exonerated ⇒ formulation under the
microscope" — has NEITHER branch obtain: the sequence is pre-asymptotic. No valid
extrapolation exists and mesh-convergence is NOT demonstrated; but the trend is
monotone toward literature, not flat/oscillating, so resolution is NOT exonerated
either.

**Blockage direction (D4 vs D2, against the CV trend):** halving blockage at matched
cells/plate moves Cd_rxn 2.8098 → 2.4336 (−13.4%) and Cd_surr 5.8385 → 5.0844 (−12.9%)
— the SAME direction and similar magnitude as the 2026-07-26 confinement probe
(Cd 5.72 → 5.09, −11%). Confinement inflates both observables by ~13%; removing it
DEEPENS the reaction's literature deficit (2.4336 at 3.1% blockage = 27.6% below 3.36).
St = 0.1875 at both D3 and D4 (literature ~0.15).

**Arbiter reproducibility — OPEN:** D1's Cd_rxn = 2.5996 differs 11% from the LD-5
verdict leg's 2.3440 at the nominally IDENTICAL config (r9, α=50, dt=5e-4, 8000 steps,
t_start=1.5), while Cd_surr matches to 4 decimals (5.4369 both) — trajectory-identical
yet reaction-differing: NOT chaotic divergence; UNEXPLAINED. Leading suspect: a code
delta in the reaction path — the LD leg ran at commit 4acaae1 (pre-TD-2), D1 runs
post-4472cf2 where TD-2 restructured the reaction computation; TD-2's parity test
pinned only `reaction_terms=False` against its own base, not `reaction_hist` against
the LD-era code. Diagnostic (filed as follow-up, not done): re-run the LD verdict
config on current code and/or diff the two reaction computations. The headline is
robust to it (5.44/2.60 = 2.1× vs 5.44/2.34 = 2.3×).

**OBSERVABLE DECISION — ESCALATED to Baskar (no adoption).** The spec adopts the
consistent reaction as the canonical force observable only "if the reaction
mesh-converges toward literature". Mesh-convergence is NOT demonstrated (monotone but
non-shrinking increments), so no adoption is recorded; the decision escalates with the
term table in hand. For Baskar: (i) the reaction is ~96% penalty virtual work at α=50 —
the "consistent reaction" is here an α-scaled penalty functional, which connects
directly to the standing α-sensitivity finding (+36–41% over α 20→100) and the
α-scaling follow-up; (ii) St ALSO moved toward literature at r11 (0.2188 → 0.1875) —
resolution is helping the physics broadly, so the sequence may still converge beyond
r11; (iii) the obvious next probe is a D3b leg at refine_to=12 (256 cells/plate; 2-D
fits the 48 GiB box; dt likely 2.5e-4 at that h; ~1–2 h wall) to test whether the
increments start shrinking — Baskar's call, not run in this campaign; (iv) if r12
still fails to shrink the increments, the two-sided-shell formulation (α-scaling,
penalty-dominated reaction) moves under the microscope as the systematic.

**Follow-ups filed:** (1) arbiter reproducibility diagnostic (LD-era vs TD-2-era
reaction path, above); (2) D3b r12 resolution probe (offered; Baskar's call);
(3) α-scaling investigation (α~Pe·p² vs fixed 50) — stands, now sharpened by
penalty-dominance; (4) canonical-observable adoption — BLOCKED on the escalation
decision, NOT implemented.

**D3b r12 UPDATE (2026-07-27, log `d3b-r12-retry-20260727-191316-76793.log`):**
refine_to=12 (256 cells/plate), dt=1.25e-4, 32k steps: **Cd_rxn = 3.1340**
(penalty 3.009, 96%), Cd_surr = 6.5299, bridge 52.3, St = 0.1562. Increment vs
r11 = **+0.0857** vs prior +0.2102/+0.2385 ⇒ **SHRINKING — convergent regime**;
geometric extrapolation Cd_∞ ≈ 3.18 at 6.25% blockage. Verdict: the literature
deficit is RESOLUTION, not formulation; α-scaling stays the open formulation
question. **ADOPTION (Baskar):** Cd_rxn (consistent Nitsche reaction) is the
CANONICAL force observable for thin-shell SBM campaigns — quoted values must
carry refine level + the still-rising (pre-asymptotic-in-absolute-terms) caveat
and the blockage note (6.25%-blockage limit ≈3.18 need not equal unconfined
3.36). Cd_surr is DEMOTED to diagnostic (approximates no partition term). Full
record: `docs/dev/p2r1a-thin-plate-runbook.md` D3b section.

6. **Projection research track:** fix the structural 2-D divergence at Re=250/L9. Candidate
   suspects listed in §5. Start with `consistent_projection=False` probe on a small mesh (L6,
   no adaptive) to isolate sub-step.

### Remaining ThinShell Cases (Phase 3)

7. **Re=126 both-solver bridge case:** nu=1/126 (octree-consistent), same corrected recipe.
   Projection leg EXPECTED to converge at this Re per ladder Re=40/100 heritage — if it does,
   both-solver deliverable rescued at a validated Re. Skipped in this campaign (time); launch
   after projection fix or as standalone validation.
8. **Two-plate case:** `tests/p2r1a_thin_plate_flow.py` two-plate geometry support — requires
   `Segment.distance_vector` implementation. Deferred to R3.
9. **Cylinder + attachments:** requires `Segment.distance_vector`. Deferred.

### CI / Infrastructure

10. **Regression tally:** the full CPU sweep (Mac, warp-CPU, `.venv/bin/python`) was run
    at consolidation: **59 passed, 4 skipped** in 371 s. All green.
    The 4 skips are GPU-only tests (warp-CUDA not available on Mac CPU). Run after every
    commit to `thinshell-gpu`. Gate command:

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && \
  .venv/bin/python -m pytest \
  tests/test_p2r1a_thin_plate_flow.py \
  tests/test_p2r1a_thin_plate_flow_projection.py \
  tests/test_p2r1c_thin_plate_flow_3d.py \
  tests/test_p2r1c_thin_plate_flow_3d_projection.py \
  tests/test_p2r0_projection_sbm.py \
  tests/test_device_assembly.py \
  tests/test_adaptive_cube_channel.py \
  -q 2>&1 | tail -5
```

---

## Appendix A: Timing Reference (measured, gpubox RTX 6000 Ada)

### 2-D Thin-Plate (L7 base, L9 adaptive, cudss + device assembly)

| Config | s/step | Notes |
|--------|--------|-------|
| Monolithic cudss + device assembly | 0.267 | Task 7 after-preflight, 100 steps, verbose ON (estimated from elapsed/steps) |
| Projection (all GPU) — cudss pred + gpu_cg PPE | 15.8 | Task 7 after-preflight, diverged (Cd -1e22), so step rate is from diverging path |

### 3-D Thin-Plate (gpubox, uniform levels)

| Level | Solver | s/step | Notes |
|-------|--------|--------|-------|
| L4 (~17K DOF) | Monolithic cudss + device | 0.38 | Task 8, 10 steps |
| L4 (~17K DOF) | Projection cpu-splu pred + gpu_cg PPE | 52.8 | Task 8, bottleneck = cpu predictor |
| L5 (~131K DOF) | Monolithic cudss + device | 2.22 | Task 8, 10 steps |
| L6 (~1M DOF) | Monolithic fused + device | 13.1 | Task 8, 3 steps only |

### 3-D Thin-Plate (GH200, Task 8b Phase 2, adaptive, device assembly)

| Rung | s/step | Cd[final] |
|------|--------|-----------|
| r5b8 (base-L5 / band-r8) | 3.36 | +7.268 |
| r6b9 (base-L6 / band-r9) | 22.21 | +6.721 |

### 3-D Bluff-Body (GH200, Task 8b Phase 1, uniform, host assembly + cuDSS)

| Level | s/step | saddle DOF |
|-------|--------|-----------|
| L4 | 0.54 | 19,508 |
| L5 | 2.89 | 142,180 |
| L6 | 26.12 | 1,084,100 |

---

---

## 2026-07-27 / 2026-07-28 — R2b Saddle-Preconditioner GPU Campaign (Task A5)

**Hardware:** gpubox RTX 6000 Ada (48 GiB, sm_89, WSL2). Branch: `track-a-r2b`.  
**Harness:** `tests/gpu_saddle_ladder.py` (SADDLE_POINTS per-process isolation).  
**Full report:** `docs/dev/2026-07-27-r2b-saddle-campaign.md`

### Campaign results table

| tag | solver | DOFs | iters_mean | s/step | converged |
|-----|--------|------|-----------|--------|-----------|
| 2d-r9 | fgmres_bdiag | 90,744 | 3385.4 | 11.3 | YES |
| 2d-r9 | fgmres_pcd | 90,744 | **46.2** | 3.0 | YES |
| 2d-r11 | fgmres_bdiag | 93,792 | 4841.6 | 15.6 | YES |
| 2d-r11 | fgmres_pcd | 93,792 | N/A | N/A | **DIVERGED** |
| 3d-L6 | fgmres_bdiag | 1,097,344 | 603.0 | 35.6 | YES |
| 3d-L6 | fgmres_pcd | 1,097,344 | **35.4** | 38.8 | YES |
| 3d-L7r9 | fgmres_bdiag | ~9,240,000 | N/A | N/A | OOM on gpubox (host-assembly path; see §4.1 note) |
| 3d-L7r9 | fgmres_bdiag | 9,080,064 | **1196.8** | 242.2 | YES (GH200, job 11772638) |
| 3d-L7r9 | fgmres_pcd | 9,080,064 | N/A | ≥5,160 | DNF (GH200 2×walltime, jobs 11772733/11772832) |

### Kill-gate verdict (FINAL, 2026-07-28)

**TRACK A VIABLE via fgmres_bdiag.** Within-3-D decade: 603.0 iters/step at 1.097M
→ 1196.8 at 9.08M = **1.985× growth ≤ ~2× — PASS** (late-step settled values put
it nearer 1.6×). PCD wins iterations wherever it converges (35 at 3-D L6, 17×
under bdiag) but is wall-time-infeasible at 9M with Jacobi-CG inners (≥86 min/step
vs bdiag 4.0 min/step, two GH200 timeouts) and non-robust on the 2-D fine-graded
mesh — A4 (AMGX velocity-inner) is the identified upgrade to keep PCD outer
counts at viable apply cost. **Memory (measured):** host-assembly transients
243–245 GB at 9.08M (mesh build 3.71 GB, WP0) ⇒ device assembly (`ASSEMBLY=device`)
MANDATORY at ≥10M; GPU peak 36.7 GiB/95. Full record + 100M projection:
`docs/dev/2026-07-27-r2b-saddle-campaign.md` §8.

---

## Appendix B: GPU Smoke Script Reference

`tests/gpu_smoke_thinshell.py` runs 4 legs (2d-mono, 2d-proj, 3d-mono, 3d-proj)
with minimal steps (2 steps each) to verify the full wiring end-to-end.

**Leg-by-leg env:**

| Leg | Key env vars |
|-----|-------------|
| 2d-mono | `MONO_SOLVER=cudss ASSEMBLY=device DEVICE=cuda:0` |
| 2d-proj | `PPE_SOLVER=gpu_cg PRED_SOLVER=cudss DEVICE=cuda:0` |
| 3d-mono | `MONO_SOLVER=cudss ASSEMBLY=device DEVICE=cuda:0` |
| 3d-proj | `PPE_SOLVER=gpu_cg PRED_SOLVER=cudss DEVICE=cuda:0` |

The script exits with a non-zero code and prints the failed leg if any leg fails.

---

## Appendix C: Track A2 Campaign — SADDLE_PCD_INNER=amgx Route (2026-07-28)

**Branch:** `track-a2-strengthen`  
**Campaign doc:** `docs/dev/2026-07-28-track-a2-campaign.md`

### C.1 Plumbing (T4 gate commits)

Two commits wired the AMGX inner route into the saddle-ladder harness and drivers:

1. **cc9838b** — `feat(t4): SADDLE_PCD_INNER env + pcd_inner kwarg plumbing`  
   `SADDLE_PCD_INNER=amgx` → `pcd_inner="amgx"` → `build_pcd_meta(..., inner="amgx")`.
   Inner stats (F/Ap/Mp) saved to npz as `ist_{blk}_{key}` arrays.

2. **91882fe** — `fix(amgx): accept not_converged as smoother`  
   AMGX `not_converged` (maxiter hit) is now accepted — the PCD F-inner uses AMGX
   as a smoother; the outer FGMRES carries the residual.

Gate (both commits): `test_saddle_ladder_cpu` + `test_saddle_precond` — 16 passed
on gpubox (including `test_fgmres_pcd_amgx_real` — first real-AMGX correctness
evidence).

### C.2 Invoking AMGX Inner on gpubox

```bash
# pyamgx requires the AMGX shared lib in LD_LIBRARY_PATH on gpubox:
export LD_LIBRARY_PATH=/home/bglab/AMGX/build:/usr/lib/wsl/lib

# Example: 3d-L6 pcd-amgx with device assembly (cuda:0)
cd /home/bglab/Baskar/DiffSim
LD_LIBRARY_PATH=/home/bglab/AMGX/build:/usr/lib/wsl/lib \
SADDLE_POINTS=3d-L6 SADDLE_SOLVERS=fgmres_pcd SADDLE_DEVICE=cuda:0 \
SADDLE_NSTEPS=5 SADDLE_ASSEMBLY=device SADDLE_PCD_INNER=amgx \
PYTHONPATH=src:tests .venv/bin/python tests/gpu_saddle_ladder.py
```

### C.3 pyamgx Status per Machine

| Machine | Status | Notes |
|---------|--------|-------|
| gpubox | **PRESENT** | `libamgxsh.so` at `/home/bglab/AMGX/build/`; needs LD path above |
| nova (GH200) | **ABSENT** | No pyamgx package, no libamgxsh.so; controller must build AMGX for aarch64 sm_90a before submitting pcd-amgx kit |

### C.4 GH200 sbatch kits (updated for T4)

| Kit | Key env change | Note |
|-----|---------------|-------|
| `saddle_ladder_gh200.sbatch` | Added `SADDLE_ASSEMBLY=device` | bdiag leg; no pyamgx needed |
| `saddle_ladder_gh200_pcd.sbatch` | Added `SADDLE_ASSEMBLY=device SADDLE_PCD_INNER=amgx SADDLE_NSTEPS=5` | pcd-amgx leg; REQUIRES pyamgx on nova-arm |
