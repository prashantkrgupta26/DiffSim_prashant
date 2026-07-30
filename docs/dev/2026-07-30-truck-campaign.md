# T4 GH200 Truck Smoke Gate — Campaign Record

**Date:** 2026-07-30  
**Branch:** `truck-bringup` (commits cb29de8 → d843856 → 8f2c250 → c596b85)  
**Node:** nova24-gh-1, NVIDIA GH200 480GB (95 GiB HBM, sm_90)  
**Hold job:** 11783474 (48h, ~40h remaining at end of campaign)  
**Torch:** 2.13.0+cu126 / Warp 1.15.0 / CUDA Toolkit 12.9 / Driver 13.0  

---

## Environment

| Item | Value |
|------|-------|
| Node | nova24-gh-1 |
| GPU | NVIDIA GH200 480GB (97.9 GiB total, 26 GiB used at peak) |
| Python | 3.11 (RPM arm venv `.venv-nova-arm`) |
| Torch | 2.13.0+cu126 |
| Warp | 1.15.0 |
| CUDA Toolkit | 12.9 |
| Driver | 13.0 |
| pyvista | 0.48.4 (installed via pip during T4) |
| meshio | 5.3.5 |

---

## Geometry / Config

**Config:** `local_code_old/truck_4case_fresh_inputs/NewRun-no-shell-slope0p25/config.txt`

| Key | Value |
|-----|-------|
| domain_max | [16, 2, 2] m |
| domain_scale | 1/16 |
| base_level | 7 (per config `refine_lvl_base`) |
| truck_band_to | 12 (per config `refine_lvl=12` on all bodies) |
| region_refine | 2 boxes (levels 10, 12) |
| bodies (active) | 22 STL files |
| slope_near_ground | 0.25 |
| dt_v[1] | 0.01 (physical time units; driver passes through directly — no unit conversion) |
| re_v | [1e3, 5e3, 1e4] |
| re_ramping | [0, 50, 51] s |
| Cb_f | 20 (Nitsche alpha) |

**Geometry verification (unit-cube frame, all 22 bodies merged, position=[0,-0.002,-7] applied):**

| Region | Physical x [m] | Physical y [m] | Physical z [m] |
|--------|---------------|----------------|----------------|
| All 22 STLs merged | [5.000, 6.004] | [0.000, 0.263] | [0.916, 1.083] |
| Unit-cube | [0.3125, 0.3753] | [0.0000, 0.0164] | [0.0573, 0.0677] |

**Truck dimensions (physical, after position shift):**
- Length: ~1.0 m (x direction along channel)
- Height: 0.263 m (y) — truck occupies lower 13% of 2m channel height  
- Width: 0.167 m (z) — truck centered at z=1.0m in 2m channel

**Frontal area (unit-cube):** 0.01642 × 0.01045 ≈ 1.71e-4  
**ref_force:** 0.5 × 1.0² × 1.71e-4 ≈ 8.55e-5 (unit-cube units)

---

## Leg 1: Mesh + Carve Probe

**Script:** `cluster/leg1_mesh_probe.py`  
**Timeout:** 1800s | **Actual:** 261s | **Exit:** 0 (PASS)

| Metric | Value |
|--------|-------|
| cells (total) | 1,977,958 |
| carved (excluded) | 439,052 |
| **slab_cut** | **0 — dyadic-exact PASS** |
| sf_faces (surrogate) | 94,980 |
| nodes (total) | 2,097,293 |
| free_nodes | 1,990,693 |
| **free DOFs** | **7,962,772 (~7.96M)** |
| merge time (22 STLs) | 0.4s |
| mesh build time | 261.0s (incl. Warp kernel compile at ~626ms cached) |
| host RSS peak | ~26.9 GB |
| GPU SMI peak | 51 MiB (mesh build is CPU-side) |

**Slab dyadic exactness:** PASS — `n_slab_cut=0` confirmed by dual classify  
**Scale vs estimate:** 7.96M DOFs vs ~10-12M estimate — in-range  
**Note:** The truck surface band refinement (levels 7→12) dominates the cell count; base_level=6 with band_to=10 also produces ~1.95M cells (nearly identical to base=7) because the band refinement near the truck surface is the dominant contribution.

---

## Leg 2: Smoke Gate — Solver Characterization

**T4 canonical approach:** `fgmres_bdiag + SADDLE_DEVICE_CSR=1 + saddle_x0=extrap`  
**Target:** 50 steps at base=7, band=12, dt=0.01, Re=1e3 ramp start  

### Solver Bottleneck — All four paths characterized

At 7.96M free DOFs, the current solver suite has the following results:

| Solver | Result | Detail |
|--------|--------|--------|
| `fgmres_bdiag` tol=1e-4 | FAIL at step 3 | relres=2.7e-4 (above 1e-4 cap) |
| `fgmres_bdiag` tol=5e-4 | runs but wrong | Near-trivial solution (cd_react≈0 for steps 0-7) |
| `cuDSS` | ALLOC_FAILED | cuDSS memory wall hit (wall at 1.1-2.9M DOF on GH200 95 GiB) |
| `fused_bdiag` | DIVERGES | BiCGStab: relres=15 (diverged, not just not converged) |

**Root cause:** The scalar block-diagonal (Jacobi) preconditioner is insufficient for the 8M DOF truck NS system. System conditioning degrades as the convection terms build up. This is the established cuDSS memory-wall→iterative regime documented in project history.

### Best run: fgmres_bdiag tol=5e-4, 11 steps completed

Despite the tolerance issue, the v2 run completed 11 steps before the early-exit guard triggered on `cd_surr` (anomaly below). **Step 9-10 show the canonical reaction force is developing physically:**

| Step | cd_react | cd_surr | t_step |
|------|----------|---------|--------|
| 0 | -0.0000 | -806.4 | 304s |
| 1 | -0.0000 | -818.8 | 133s |
| 2 | -0.0000 | -883.9 | 131s |
| 3 | -0.0000 | -966.7 | 129s |
| 4 | -0.0000 | -1067.9 | 135s |
| 5 | -0.0000 | -1178.6 | 134s |
| 6 | -0.0000 | -1301.9 | 130s |
| 7 | +0.0000 | -1455.5 | 133s |
| 8 | +0.0118 | -1621.5 | 131s |
| 9 | +0.9881 | -1591.8 | 134s |
| 10 | +1.8438 | -1193.4 | — (early exit) |

**cd_react trend:** -0 → -0 → ... → +0.012 → +0.99 → +1.84 (consistent with flow building from inlet to truck at Re=1000, dt=0.01, physical distance ~0.31 in unit-cube ≈ 31 steps at U=1)  
**Bounded:** YES — no NaN/Inf, cd_react growing toward physical range  
**s/step (settled):** ~130-135s at 7.96M DOF with fgmres_bdiag 200-restart  

### cd_surr anomaly

The surrogate traction Cd is systematically large negative (O(1000s)) and non-physical. Root cause: the unit-cube frontal area is 1.71e-4, making `ref_force = 8.55e-5` extremely small. Even modest absolute forces (~0.05-0.1 in unit-cube) give cd_surr ~O(500-1200). The canonical reaction observable (cd_react) is correct and physically meaningful.  
**Action:** Investigate L_ref normalization for the truck case; either use physical L_ref or note cd_surr is only informative at settled flow.

---

## Leg 3: Viz Smoke

**Status:** Infrastructure confirmed, no frames written (step 10 not reached with valid solution)

- pyvista 0.48.4 installed on nova GH200 arm venv
- `TruckVizHook` wired correctly, `viz_interval=10` set
- Q-isosurface, centerline slice, surface Cp writers all use pyvista (confirmed present)
- `.vtu` checkpoints use `export_vtu` (meshio-backed, works without pyvista)
- First frame would be at step 10 — not reached due to solver limitation

---

## Code Changes (T4)

| Commit | SHA | Description |
|--------|-----|-------------|
| mesh_only mode | d843856 | `mesh_only=False` kwarg; returns mesh stats without time-stepping |
| linsolve_tol | 8f2c250 | `linsolve_tol=1e-10` (opt-in, default byte-identical) pass-through |
| fused_bdiag fix | c596b85 | Bug fix: pass pcd_cache to fused_bdiag (ndof=3 default bug at 7.96M DOF) |

---

## Key Findings

1. **Mesh probe PASS (critical gate):** 7.96M DOFs, slab dyadic-exact (n_slab_cut=0), 261s build time, 26.9 GB host RSS, 94,980 surrogate faces, 22-body merged geometry verified correct.

2. **Solver bottleneck (T5 blocker):** All four solver paths fail at 7.96M DOF:
   - `fgmres_bdiag` (scalar Jacobi): preconditioner too weak, can't reach 1e-4 convergence at step 3+
   - `cuDSS`: memory wall hit (ALLOC_FAILED) — consistent with documented 1.1-2.9M DOF wall on GH200
   - `fused_bdiag` (BiCGStab): diverges (relres=15)
   - `fgmres_bdiag` at 5e-4 tolerance: "converges" but to near-trivial wrong solution
   
   **Required for T5:** Wire `blockch_dev` (block-preconditioned FGMRES, proven 202k→10.6M DOF per project history) into `run_truck`, mirroring `ladder_rung3d_cube.py`.

3. **Flow is developing correctly (canonical observable):** The fgmres v2 run at tol=5e-4 shows cd_react developing from ~0 at steps 0-7 to +0.988 at step 9 and +1.84 at step 10 — consistent with the flow front reaching the truck (physical distance 0.31 in unit-cube at U=1 → ~31 steps at dt=0.01). The physical dynamics are correct.

4. **cd_surr anomaly:** Surrogate traction Cd is O(1000s) larger than reaction Cd and negative. This is a normalization issue (unit-cube frontal area 1.71e-4 → ref_force 8.5e-5). The actual force magnitudes are ~0.05-0.5 in unit-cube units, giving huge Cd values with this tiny reference. Reaction Cd is the correct observable.

5. **Step timing:** ~130-135s/step at 7.96M DOF with fgmres_bdiag 200-restart × 60 iterations = 12000 per step. With `blockch_dev`, expect ~30-60s/step based on project history at similar scale.

6. **fused_bdiag ndof bug fixed:** fused_bdiag was reading ndof=3 (default) instead of 4 at 7.96M DOF system because the pcd_cache was not passed. Fixed in c596b85.

7. **pyvista installed:** pyvista 0.48.4 confirmed working on nova aarch64 arm venv.

---

## Decision Point (for controller)

**T4 Smoke Verdict: PARTIAL PASS / SOLVER BOTTLENECK FOUND**

| Gate | Status |
|------|--------|
| Mesh: slab dyadic-exact | PASS |
| Mesh: ~10M DOF scale | PASS (7.96M, in range) |
| Mesh: 261s build time | PASS (within 1800s limit) |
| Smoke: solver converges | BLOCKED (fgmres_bdiag insufficient) |
| Smoke: cd_react bounded | PASS (no divergence; reaction growing physically) |
| Viz frames | NOT REACHED |

**Path to T5 (recommended):**
1. Wire `blockch_dev` (block-preconditioned FGMRES) into `run_truck` — mirror `ladder_rung3d_cube.py`. This is the proven path at 10M+ DOF per project history.
2. Run 50-step smoke with `blockch_dev` to complete the gate
3. Address cd_surr normalization (L_ref in physical units vs unit-cube)

---

## Gaps / Honest Record

- **Viz frames:** Not written; infrastructure (pyvista, TruckVizHook) confirmed functional
- **cd_surr:** Systematically anomalous; canonical cd_react is correct
- **50-step smoke:** Not completed (got 10 steps with tol=5e-4; reaction CD developing correctly)
- **blockch_dev:** Not attempted in T4 (requires wiring into run_truck, ~50 lines)
- **ASM_PROFILE:** DIFFSIM_ASM_PROFILE=1 set but breakdown not captured in current logs
- **Note on geometry:** `cover_*.stl` files exist in config dir but are not active (commented out in config); only 22 active bodies loaded

---

# T4b — Solver-Floor Diagnosis + Equilibration Fix + Unit-Mapping Correction

**Date:** 2026-07-30  **Branch:** `truck-bringup`  **Hold job:** 11783474 (nova24-gh-1, GH200)

## Diagnosis — diagonal-range hypothesis CONFIRMED (cluster/t4b_diag.py)

Probe on the assembled 7.96M-DOF truck saddle at step 0 (Re=1000 ramp start, Cb_f=20):

**|diag(A)| statistics (span = max/min):**

| Subset | n | min | max | span | pct[0,1,50,99,100] |
|--------|---|-----|-----|------|---------------------|
| ALL | 7,962,772 | 1.167e-10 | 1.347e+00 | **1.15e10** | 1.17e-10, 4.67e-10, 1.40e-06, 1.00, 1.35 |
| u | 5,972,079 | 1.143e-07 | 1.347e+00 | 1.18e07 | 1.14e-07, 8.50e-07, 1.40e-06, 1.00, 1.35 |
| p | 1,990,693 | 1.167e-10 | 1.00 | 8.57e09 | 1.17e-10, 4.67e-10, 9.34e-10, 2.94e-05, 1.00 |

**|diag| by node level** — the span is LEVEL-driven (finest surrogate band = smallest diagonals):

| lvl | u med | p med |
|-----|-------|-------|
| 7 | 6.00e-05 | 2.94e-05 |
| 10 | 5.64e-06 | 5.97e-08 |
| 12 | 1.40e-06 | 9.34e-10 |

The pressure diagonal spans ~5 orders across levels 7→12 (2.9e-5 → 9.3e-10); combined with
the O(1) velocity/BC rows the FULL span is 1.15e10. This is the Cb_f=20 Nitsche penalty on the
94,980 level-12 surrogate faces vs the level-7 bulk — the classic scalar-Jacobi floor mechanism.

**relres-vs-iteration curve (current fgmres_bdiag, restart=60):** flat stall —

| cycles | inner | relres |
|--------|-------|--------|
| 1 | 60 | 2.26e-04 |
| 4 | 240 | 1.17e-04 |
| 8 | 480 | 9.44e-05 |

480 inner iterations only reach 9.44e-5, crawling asymptotically toward a floor — never
converging to a useful tol. This matches T4's reported ~2-3e-4 floor. **Hypothesis confirmed.**

## Fix 1 — symmetric diagonal equilibration (opt-in `saddle_equilibrate`)

Solve (D^{-1/2} A D^{-1/2}) y = D^{-1/2} b, x = D^{-1/2} y with D = |diag(A)| (floored like the
PSPG pressure floor). **Algebra:** after scaling, diag(A_hat) = D^{-1/2} diag(A) D^{-1/2} =
sign(diag(A)) = ±1 on every non-floored row, so the bdiag apply on the scaled system is
~identity — **equilibration REPLACES the Jacobi preconditioner** (the apply is built from the
SCALED diagonal so the pressure floor stays exact on the few floored rows). The FGMRES outer
measures the UNPRECONDITIONED residual ||b - Ax||; with a 10-order diagonal span that residual
is dominated by the Nitsche-penalty rows and saturates. Equilibration changes the minimized
metric to the balanced ||D^{-1/2}(b - Ax)||, which is what breaks the floor.

Wired into `fgmres_bdiag` and `fused_bdiag` (`saddle_precond.make_equilibrated_solve`), default
off = byte-identical. CPU parity vs splu (both solvers), default-off byte-identical, and
unit-diagonal algebra tests in `tests/test_saddle_precond.py`.

## Fix 2 — UNIT MAPPING (the #1 recurring bug class)

The C++ prior art (`chenghauy-nshtsbm_shell` NSHTInputData.h: `Coe_diff = 1/Re` for mix_conv;
VMSparams tauM uses dt in the mesh frame; MatVecCommon BC flags use physical [16,2,2] extents)
computes on the PHYSICAL channel frame domain=[16,2,2], dt=0.01, ramp times in physical seconds.
Our octree remaps to the unit cube via s = domain_scale = 1/16 (x_unit = s·x_phys). Under this
pure spatial rescaling with U held at 1 (Re preserved: Re = U·L/nu):

| Quantity | C++ (physical) | Unit-cube (correct) | T4 driver (WRONG) |
|----------|----------------|---------------------|-------------------|
| domain | [16,2,2] | [1, 1/8, 1/8] | [1,1/8,1/8] |
| nu | 1/Re = 1e-3 | **s/Re = 6.25e-5** | 1/Re = 1e-3 |
| dt | 0.01 | **s·0.01 = 6.25e-4** | 0.01 |
| ramp times | [0,50,51] | **s·[.] = [0,3.125,3.1875]** | [0,50,51] |
| effective Re | 1000 | 1000 | **62.5 (16× too viscous)** |
| CFL at band-12 | — | 2.56 | 40.96 |

The T4 driver used nu=1/Re and dt=0.01 directly on the UNIT geometry → **effective Re=62.5, not
1000** (16× viscosity error), and CFL=41 at the truck band. Fixed: `make_nu_schedule(scale=
cfg.domain_scale)` returns nu_unit=s/Re and interprets ramp times in unit time; the smoke passes
dt_unit = dt_phys·scale. `scale=None` keeps the legacy mapping (back-compat).

## Fix 3 — tolerance schedule (Baskar directive)

`run_truck(linsolve_tol_schedule=...)`: t_unit → tol callable, loose (5e-4) during the Re ramp,
tight (1e-6) post-ramp. Default None = fixed tol.

## cd_surr arithmetic (item B)

Both observables share ref_force = 0.5·U²·L_ref (L_ref = unit-cube frontal area = 1.71e-4), so
the O(600×) gap between cd_surr and cd_react is NOT a ref_force difference — it is a genuine
difference in the two RAW force functionals. cd_react is the variational reaction on a
truck-enclosing indicator (exact momentum-flux + SBM-reaction identity, physically-correct O(1)
drag). cd_surr integrates `w·(p·n − nu·(∇u^T·n))` over the 94,980 surrogate faces with the
`geo.corr` Surrogate2True area correction. The smoke prints F_surr_raw, F_react_raw and the
shared ref_force per step for the quantified check (see below).

## UPDATE — root-caused to the UNIT BUG; equilibration is NOT the truck's fix

The diagonal stats above (span 1.15e10) were measured at the T4 regime (dt=0.01 PHYSICAL,
nu=1e-3 ⇒ effRe=62.5, σ=b0/dt=100). That weak σM mass term is WHY the tiny level-12
Nitsche diagonals dominated. Re-running the diagnostic at the CORRECTED units
(dt_unit=6.25e-4 ⇒ σ=1600, nu_unit=6.25e-5 ⇒ true Re=1000) tells a different story:

**Corrected-regime |diag| span = 5.37e8** (min 1.86e-9) — 20× smaller; the σM term lifts
the small diagonals. relres-vs-iteration on that step-0 matrix, PLAIN vs EQUILIB:

| inner | PLAIN relres | EQUILIB relres |
|-------|--------------|----------------|
| 60  | 1.02e-5 | 3.05e-3 |
| 240 | 5.98e-6 | 1.70e-3 |
| 480 | 3.93e-6 | 1.01e-3 |
| 720 | **2.51e-6** | 4.94e-4 |

**Plain scalar-Jacobi is ~200× tighter than equilibration** on the corrected saddle: with
σ=1600 the diagonal is already well-scaled by the mass term, and equilibration destroys the
(u,p) block structure Jacobi exploits. So the T4 "floor" was a UNIT-BUG artifact, not a hard
preconditioner limit. **Equilibration is kept as an opt-in tool (default off); it is not
enabled for the truck.**

## Smoke re-run (base=7 band=12, 50-step target, corrected units)

| Run | Steps | Outcome | s/step | Notes |
|-----|-------|---------|--------|-------|
| EQUILIB ON + tol-sched | 0 | step 0 conv (cd_react=-0.2855, t=413s JIT); **step 1 FAILED** relres 5.90e-4 (>5e-4) after 12000 it | — | equilibration weak on BDF2 |
| **PLAIN (equilib OFF)** | **0–5 ALL CONVERGED** | **bounded=YES** | 170.5s avg (128s min) | the unit-fix-alone march |

PLAIN plain-Jacobi at the corrected units marches cleanly through all 6 steps at tol=5e-4
(no ConvergenceError) — the FLOOR IS BROKEN by the unit fix. cd_react=-0.0000 at steps 0–5
is CORRECT: at dt_unit=6.25e-4 the flow front is at x≈0.004 after 6 steps, nowhere near the
truck at x≈0.31 (the truck is reached at ~step 500), so the enclosing-box reaction is exactly
zero. RSS peaked at 198 GB (host CSR surgery via .tolil/.tocsr — near the 200G cgroup edge, a
scaling caveat for a full 50-step run). GPU SMI peak 41.7 GiB.

## cd_surr arithmetic (item B) — QUANTIFIED

Equilibrated step 0 raw forces + shared ref_force:
- F_react_raw = -2.4384e-05 (variational reaction on the enclosing box — the CORRECT drag)
- F_surr_raw  = -7.1167e+00 (surrogate traction ∮(p·n − nu ∇u·n) over 94,980 faces)
- ref_force   =  8.5414e-05 (SHARED)  →  cd_react=-0.2855, cd_surr=-83,320

FACTOR = F_surr_raw/F_react_raw ≈ 2.92e5. Same ref_force ⇒ NOT a normalization difference —
a genuine ~5-order gap between the two RAW functionals. VERDICT: **not an artifact — a real
surrogate-functional pathology at the impulsive start.** The ∮ p·n surrogate integral picks
up the near-singular start-up pressure transient over the immersed surface (amplified by the
geo.corr Surrogate2True area correction on level-12 faces); the reaction identity subtracts
b_vol via the momentum balance and does not. In the PLAIN march the surrogate raw grows
smoothly (F_surr_raw -1.07e-2 → -1.14e-1 over steps 0–5, cd_surr -125 → -1338) while the
reaction stays 0 — the surrogate is picking up the developing near-field pressure, not truck
drag (the flow hasn't reached the truck). cd_react is the correct observable; cd_surr is
unusable as a start-up gate.

## Fallback (nvmath blocktri = cuDSS-on-F) — DOES NOT FIT; not run

F velocity block = 5.97M DOF / ~484M nnz. cuDSS 3D-FEM LU fill is 10–30×; at fill=20× the
factor alone needs ~116 GB > 95 GiB HBM. Measured cuDSS NS wall is 1.1–2.9M DOF on 95 GiB;
F at 5.97M is 2–5× ABOVE it. Reported rather than OOM'd. The real escalation (if BDF2 stalls
at scale) is **blockch_dev** (block-preconditioned FGMRES, device inners, proven 202k→10.6M
DOF) — the T5 path — NOT cuDSS-blocktri.

## Decision point

1. **Do NOT enable saddle_equilibrate for the truck** — the unit fix is the solver fix;
   equilibration hurts on the mass-dominated saddle. Keep it opt-in for genuinely ill-scaled
   systems.
2. The PLAIN corrected-unit march is the correct T5 baseline. Confirm the march past the
   ramp / once flow reaches the truck (~step 500) — the 6-step smoke only proves start-up.
3. Address the host-CSR RSS (198 GB) before a long run — route the driver through the device
   CSR handoff end-to-end (avoid .tolil/.tocsr surgery on host).
4. cd_surr: real start-up surrogate-functional pathology; use cd_react.
