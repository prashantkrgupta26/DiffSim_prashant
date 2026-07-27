# P2-R1a Re=250 Thin Plate: gpubox Full-Resolution Runbook

## Target

Cd ≈ 3.29–3.45, St ≈ 0.15 (ThinShell.pdf Table 1; Najjar & Balachandar 1995: Cd=3.36, St=0.14).

## Symmetry-breaking perturbation (REQUIRED for Re=250 shedding)

A perfectly symmetric flow-past-a-symmetric-plate on a symmetric mesh NEVER
sheds spontaneously — it rides the unstable symmetric branch indefinitely.
The driver now supports a small transverse kick at the inflow nodes:

    u_y|_{inflow} = pert_eps * U_inf    for t < pert_t_end
    u_y|_{inflow} = 0.0                 for t >= pert_t_end

Default for Re=250 gpubox run: `pert_eps=0.03, pert_t_end=1.0` (hardcoded in
`RE250_CONFIG` in `tests/p2r1a_thin_plate_flow.py`).  The env-var knob is
`PERT_EPS` (pass as float string; empty = no kick).

**Validation (Mac CPU):** At Re=100, level=5, 200 steps (dt=0.02), with kick ON
Cl_std = 0.00206 vs OFF Cl_std = 0.000003 — ratio ~700x.  The kick clearly
seeds asymmetry; it does not force shedding at this coarse resolution (numerical
damping suppresses limit-cycle growth at level 5), but the asymmetry is present
for the wake instability to amplify at Re=250 on the fine gpubox mesh.

## Command (with perturbation, for Re=250 shedding run)

```bash
ssh gpubox "cd ~/DiffSim && \
    LEVEL=7 NSTEPS=1000000 DT=5e-5 NU=0.004 U_INF=1.0 \
    PLATE_XC=0.1389 PLATE_YC=0.5 PLATE_L=0.0625 \
    PERT_EPS=0.03 PERT_T_END=1.0 \
    .venv/bin/python tests/p2r1a_thin_plate_flow.py 2>&1 | tee logs/p2r1a_re250_$(date +%Y%m%d_%H%M%S).log"
```

Or using `scripts/remote/run.sh` (if wired for env-var passthrough):

```bash
LEVEL=7 NSTEPS=1000000 DT=5e-5 NU=0.004 U_INF=1.0 \
PLATE_XC=0.1389 PLATE_YC=0.5 PLATE_L=0.0625 \
bash scripts/remote/run.sh tests/p2r1a_thin_plate_flow.py
```

## Plate parameters (ThinShell.pdf §4.3)

| Parameter         | Physical value                       | Octree [0,1]² value |
|-------------------|--------------------------------------|---------------------|
| Plate length L    | 1.0 (= H/16 where H=16)             | 0.0625              |
| Plate center x    | 5.0 (in [0, 36] domain)              | 5/36 ≈ 0.1389       |
| Plate center y    | 8.0 (in [0, 16] domain)              | 8/16 = 0.5          |
| Re                | 250 (= U_inf * L / nu)               | —                   |
| nu (kinematic)    | 1.0 * 1.0 / 250 = 0.004             | —                   |
| U_inf             | 1.0                                  | —                   |
| dt                | 5e-5 (near-plate CFL)                | —                   |
| t_end             | 50.0 (50+ shedding periods at St≈0.15) | —                 |
| t_start (average) | 20.0 (post-transient)                | —                   |

## Python config dict (in tests/p2r1a_thin_plate_flow.py)

`RE250_CONFIG` at the bottom of the driver holds these exact parameters for
reference.  It is **not** run in CI.

## Computing Cd_mean and St from the log

After the run completes, the driver prints:

```
Cd_mean=<value>  St=<value>  freq=<value>
```

These use `diffsim.postproc.shedding.time_avg_cd` (t_start defaults to second
half, i.e. t ≥ 25 s) and `strouhal` (FFT of the detrended Cl tail).

For a manual post-processing run against a saved numpy log:

```python
import numpy as np
from diffsim.postproc.shedding import time_avg_cd, strouhal

data = np.load("p2r1a_re250_history.npz")
cd_mean = time_avg_cd(data["t"], data["cd"], t_start=20.0)
St, freq = strouhal(data["t"], data["cl"], U=1.0, L=1.0)
print(f"Cd_mean={cd_mean:.4f}  St={St:.4f}  freq={freq:.4f}")
```

Note: `L=1.0` is the PHYSICAL plate length (St = f·L/U uses physical units).
The octree-normalized value (0.0625 = 1/16) must NOT be used here — that would
yield St 16× too small.

## Output logs

Logs land at `~/DiffSim/logs/` on gpubox (create if absent):

```bash
ssh gpubox "mkdir -p ~/DiffSim/logs"
```

## Adaptive mesh (optional, for convergence)

Level 7 base + Level 9 wake + Level 9–11 plate.  Adaptive wiring via
`refine_elements()` + `balance2to1()` is deferred to Task 4 / R1b.  The
`RE250_CONFIG.level=7` entry covers base-only resolution.

## Expected wall time

~2–4 hours on a single A100 for 1 M steps at L7 (rough estimate based on
similar BDF2 runs at comparable problem size).  Adaptive L9–L11 can increase
this to 8–12 hours.

## Outlet BC: do-nothing + single pressure pin

The outflow boundary (x = x_max) uses a **do-nothing / natural outlet**: no
velocity Dirichlet is imposed there, and the IBP surface integral is dropped —
equivalent to a zero-traction condition `sigma.n = 0` (i.e. `p - nu(grad u).n = 0`).
A single pressure node is pinned (`p = 0` at the outflow–bottom corner) to
remove the pressure null-space.

**Assessment for Re=250 run:**

- In the RE250_CONFIG domain [0,36]×[0,16] the plate sits at x=5, giving
  **31 plate-lengths** of wake before the outlet — well beyond the ~20L
  typically recommended for minimal outlet influence.
- Do-nothing imposes zero pseudo-traction.  When a vortex convects through,
  it induces a transient `p - nu*(grad u).n ≈ 0` that back-drives a small
  spurious velocity.  At 31L downstream, vortices are substantially diffused.
- **Reflection risk to watch for:** spurious Cl oscillation phase-locked to
  the outlet convection time `T_conv = L_wake / U_inf ≈ 31 s`.  In the Cl(t)
  spectrum this appears as a peak at `f ~ 1/31 ≈ 0.032 Hz` — far from the
  expected shedding frequency `f_shed = St * U / L ≈ 0.15 Hz`.  If this peak
  is absent, outlet reflection is not significant.
- **Recommended fix if reflection is observed:** convective (advective) outlet
  `u_t + U_inf * u_x = 0` applied weakly, or Robin BC
  `p - nu(grad u).n = U_inf * u.n`.  This requires a stepper modification and
  is deferred; `assemble_backflow_block` in `ns_bricks.py` provides a partial
  scaffold (backflow stabilization) that can be extended.

## Pass/fail criteria

| Run type    | Gate                                      |
|-------------|-------------------------------------------|
| CI (Mac)    | Pipeline runs end-to-end; Cd_mean finite and O(1)-plausible; perturbation breaks symmetry (ON/OFF Cl_std ratio > 10) |
| gpubox full | Cd_mean ∈ [3.29, 3.45], St ∈ [0.12, 0.18] (ThinShell.pdf Table 1 band); no outlet-reflection peak at f~0.032 in Cl spectrum |

---

## BOTH-SOLVER vs LITERATURE (2026-07-25): monolithic + projection

The R1a gate is not "which solver matches the other" but **which solver matches
the LITERATURE** (Cd 3.29–3.45, St ~0.15; Najjar & Balachandar 3.36/0.14).
Per `ns_projection_vms_paper §4.3 + Fig.10` the monolithic VMS **OVERpredicts**
Cd (~25%, the pressure-fine-scale/grad-div term); the projection is the more
literature-faithful one — so monolithic is NOT ground truth.

The driver now has a projection path (`run_flow_past_projection`, marched via
`LeraySBMShellStepper` with the merged outflow-BC p'-scheme levers:
`consistent_projection=True, inner_iterate=True, inner_max=8, inner_relax=0.5,
rotational_pin_wall=True`, whole-outflow-line p'=0 Dirichlet) and a both-solver
comparison entry point (`compare_solvers`, or `SOLVER=both`/`COMPARE=1` in
`__main__`).  The PPE solver is `PPE_SOLVER` (splu on CPU, **gpu_cg on GPU**).

### Small-case verification (Mac CPU, done):

- L3, 6 steps, Re=10: projection Cd 123.0 → **+42.8** (positive, decaying);
  monolithic Cd 129.6 → +70.1.  Projection ~40% below monolithic — the SAME
  documented sign/shape-correct, magnitude-low behavior as the 3-D projection.
- Two-sided load-bearing confirmed (one-sided force differs by >700x).
- PPE ran (solve_linear `sym=True` once per step).

### Resolved Re=250 — MONOLITHIC (unchanged; splu saddle → gpubox):

```bash
ssh gpubox "cd ~/DiffSim && \
    LEVEL=7 NSTEPS=1000000 DT=5e-5 NU=0.004 U_INF=1.0 \
    PLATE_XC=0.1389 PLATE_YC=0.5 PLATE_L=0.0625 \
    PERT_EPS=0.03 PERT_T_END=1.0 \
    .venv/bin/python tests/p2r1a_thin_plate_flow.py 2>&1 | tee logs/p2r1a_re250_mono_$(date +%Y%m%d_%H%M%S).log"
```

### Resolved Re=250 — BOTH SOLVERS in one run (projection PPE on gpu_cg):

```bash
ssh gpubox "cd ~/DiffSim && \
    NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_CUMEM_ENABLE=0 \
    SOLVER=both PPE_SOLVER=gpu_cg \
    LEVEL=7 NSTEPS=1000000 DT=5e-5 NU=0.004 U_INF=1.0 \
    PLATE_XC=0.1389 PLATE_YC=0.5 PLATE_L=0.0625 \
    PERT_EPS=0.03 PERT_T_END=1.0 T_START=20.0 \
    .venv/bin/python tests/p2r1a_thin_plate_flow.py 2>&1 | tee logs/p2r1a_re250_both_$(date +%Y%m%d_%H%M%S).log"
```

This prints the Cd/St-vs-literature table for BOTH solvers (Cd time-averaged
from `T_START=20.0`, St from the FFT of Cl over the plate physical length
L=1.0).  `PPE_SOLVER=gpu_cg` routes the SPD pressure-Poisson through the
scalable Jacobi-CG device path; the Oseen predictor stays on splu (not SPD) at
L7 — for near-L9 the predictor is the bottleneck (FGMRES/AMGX follow-on).  The
NCCL env vars are the gpubox WSL2 workaround (harmless single-GPU; required if
gpu_cg touches multi-GPU).

### Expected mesh / wall:

- Mesh: base L7 (128² octree ≈ 16k cells uniform), wake refined L9, plate cells
  L9–11 via adaptive refinement (`--adaptive`, not yet wired into the driver;
  uniform L7 is the current path).
- Wall: ~2–4 h single-A100/GH200 for 1M steps at uniform L7 (monolithic splu);
  the projection leg's gpu_cg PPE is comparable-or-faster per step at L7 and is
  the ONLY path that scales to L9-near-plate (~186k nodes) where host-splu walls.
- Averaging window: `T_START=20.0`, march to t≈50 (≥ 20 shedding periods at
  St≈0.15, f_shed≈0.15).

### Expected literature outcome (the deliverable):

- **Monolithic** Cd expected ~25% HIGH of the 3.36 ref (VMS overprediction,
  paper Fig.10) → Cd ~4.0–4.2, likely ABOVE the 3.29–3.45 band.
- **Projection** Cd expected to land IN or NEAR the 3.29–3.45 band (the more
  literature-faithful solver).  If instead the projection converges ~40% LOW of
  monolithic (i.e. Cd ~2.5–3.0, BELOW the band) that is the immersed-shell
  deficiency (deeper p2-r2a-monolithic-pivot defect), NOT a feature — that is
  exactly the question this run settles.
- St ≈ 0.15 expected for both (shedding frequency is a wake property, less
  solver-sensitive than Cd magnitude).

> **⚠ SUPERSEDED by 2026-07-26 GPU campaign** — the commands and configurations
> in this section use `NU=0.004`, which is the miscalibrated (paper-physical) value.
> See the "2026-07-26 GPU Campaign" section below for the corrected recipe and results.
> The commands above are preserved for reference only.

---

## 2026-07-26 GPU Campaign: Unit Bug + Corrected Recipe + Results

**Branch:** thinshell-gpu. **Solvers validated on:** gpubox RTX 6000 Ada 48 GiB,
nova A100-PCIE 39 GiB, nova GH200 95 GiB (smoke OK all targets).

See [`docs/dev/thinshell-gpu-runbook.md`](thinshell-gpu-runbook.md) for the full
consolidated operator's guide. This section records the campaign-specific findings.

### The Unit Bug (RE250_CONFIG miscalibration)

The `RE250_CONFIG` block in `tests/p2r1a_thin_plate_flow.py` used `nu=0.004` with
`PLATE_L=0.0625` (octree units). The effective plate Re is:

```
Re_eff = U * L_octree / nu = 1.0 * 0.0625 / 0.004 = 15.6
```

At Re≈16, the wake is steady. The O1 production run (100k steps, dt=5e-4) correctly
produced no shedding: Cl decayed to 1e-9, Cd=6.70 (steady symmetric branch — CORRECT
physics at Re~16).

### Corrected Recipe

```python
nu   = U * L_octree / Re = 1.0 * (1/16) / 250 = 2.5e-4
x_c  = 5/16 = 0.3125          # plate center in [0,1]²
dt   = 5e-4                    # 10× speedup vs dt=5e-5
St   = f * L_octree / U        # Strouhal number in octree units
```

Runner: `tests/gpu_re250_corrected.py` (commit e2c2675).

```bash
ssh gpubox "cd /home/bglab/Baskar/DiffSim && \
    LD_LIBRARY_PATH=/usr/lib/wsl/lib:\$LD_LIBRARY_PATH \
    DEVICE=cuda:0 MONO_SOLVER=cudss ASSEMBLY=device \
    BASE_LEVEL=7 REFINE_LEVEL=9 WAKE_LEVEL=9 \
    NSTEPS=16000 DT=5e-4 NU=2.5e-4 U_INF=1.0 \
    PLATE_XC=0.3125 PLATE_L=0.0625 \
    PERT_EPS=0.03 PERT_T_END=0.5 T_START=3.0 \
    .venv/bin/python tests/gpu_re250_corrected.py 2>&1 | tee logs/re250-corrected-\$(date +%Y%m%d-%H%M%S).log"
```

### Re=250 Monolithic GPU Results (corrected config)

| Run | Cd_mean | St | t_avg window | Shedding | Notes |
|-----|---------|----|-----------|-----------| ------|
| MISCALIBRATED (O1, NU=0.004) | 6.70 | 2.0 (artifact) | t=[0,50] | NO | RE_eff=15.6; steady physics correct |
| r9 longstats (NU=2.5e-4, 16k steps) | 5.47 | 0.203 | t≥3, ~17 periods | YES (Cl_std 3e-2) | Window-insensitive |
| r10 (REFINE_LEVEL=10) | 5.72 | 0.203 | ~same | YES | +4.6% vs r9; mesh-converged |
| L_INV=32 confinement (blockage 3.1%) | 5.09 | 0.1875 | ~same | YES | Move toward lit; residual +51% |

**Literature:** Cd=3.36, St=0.14–0.15 (Najjar & Balachandar 1995; ThinShell.pdf Table 1 band [3.29,3.45]).

**Verdict:** Shedding robust. Resolution-converged (r9→r10). Confinement measurable but not dominant.
Residual gap primary suspect: SBM-force systematic (alpha=50 leak-drag / traction bias). Discriminating
diagnostic: momentum-deficit CV drag vs `surrogate_traction` on the saved `results/re250_*_hist.npz`.

### Projection Status (GPU)

The 2-D projection leg **diverges structurally** at Re=250/L9 under BOTH the miscalibrated and
the corrected configs. This is NOT a unit bug. Research-track suspects:
- p'-outflow scheme completeness in the 2-D LeraySBMShellStepper path
- Lagged-p* split non-convergence at Re=250 (backflow beta=0.5 tuned at Re=100)
- Inner iteration divergence under consistent_projection=True on the large adaptive mesh

The fused predictor (BiCGStab) also diverges at step 1 (relres 2.3e15, rho breakdown).
The cudss predictor runs but the projection coupling itself diverges. Both-solver@Re250
stays monolithic-only on GPU; projection = research track. See
[`docs/dev/thinshell-gpu-runbook.md §5`](thinshell-gpu-runbook.md) for the full diagnosis.

### 2026-07-27 Leak-drag discriminator — VERDICT

**Instrument trail** (GPU legs on gpubox RTX 6000 Ada, `tests/gpu_leakdrag_discriminator.py`):

(a) **3-leg α sweep** (8000 steps/leg, log `leakdrag-20260726-232649-53567.log`, ~45–53 min/leg):

| alpha | Cd_surr | CV(4L) | \|leak\| | St |
|-------|---------|--------|----------|-----|
| 20 | 4.228 | 2.619 | 7.7e-3 | 0.1875 |
| 50 | 5.437 | 3.090 | 4.7e-3 | 0.2188 |
| 100 | 5.982 | 3.565 | 5.0e-3 | 0.2188 |

Original {4,6,8}L box columns **withheld** — 6L/8L were off-domain (upstream caps at 5L for
x_c=5/16; probe-design bug — zero-meaned columns tripped the spread self-check exactly as
designed). Only the 4L column is valid.

(b) **Corrected-box leg** (α=50, margins {2,3,4}L): CV = 2.855 / 2.954 / 3.090 — spread 7.9%
(>5% gate), monotone in box size.

(c) **Double-window leg** (16k steps, ~2.9× window): CV = 2.853 / 2.980 / 3.209 — spread 11.8%,
**WORSE** ⇒ unsteady-residual hypothesis **REFUTED**; line-quadrature CV hit its method limit
on the stabilized weak-divergence field (real CV integration systematic).

(d) **FINAL verdict leg** (LD-5 consistent-reaction arbiter, α=50, 8000 steps,
log `leakdrag-verdict-20260727-091605-26155.log`): **Cd_reaction = 2.3440** on both
plate-enclosing indicator sets (agreement 7.01e-15 ≤ 1e-6, **gate PASSED**),
**Cd_surr = 5.4369**, **St = 0.2188**. Printed verdict line:
`VERDICT: observable-overestimates+alpha-insensitive`, `LEAKDRAG-OK`.

**VERDICT: observable-overestimates** — the `surrogate_traction` observable reads **2.3×** the
variationally-consistent Nitsche reaction; the physics-carrying instruments (reaction 2.34,
CV 2.9–3.2) sit at/below literature 3.36. OPEN QUESTION (sharp form): the consistent reaction itself is ~30% BELOW literature — the consistent observable may UNDERestimate; the traction dissection must adjudicate both directions, not only the surrogate's excess. Two mandatory caveats: (i) the printed
"alpha-insensitive" tag is an artifact of the single-α verdict leg — the 3-leg sweep measured
**alpha-SENSITIVE** (+41% surr, +36% CV over α 20→100), which is the standing label;
(ii) the agreement gate is a mechanics self-check (zero by construction, per the documented
scoping in the probe), not set-independence.

**Follow-ups:** (a) traction-observable dissection — shifted-face σ·n integration vs the
consistent Nitsche functional; decompose the 2.3× gap (candidate terms: penalty virtual work,
adjoint-consistency, staircase-face area weighting) — the immediate next work; (b) the
reaction-vs-CV ~25% gap (penalty virtual work vs divergence-error flux) as a secondary
reconciliation item; (c) α-scaling investigation (α~Pe·p² vs fixed 50) stands.

**Instrument-development cost (honest):** the probe shipped with a box-margin design bug
({4,6,8}L against a 5L upstream cap — caught by its own spread self-check, margins corrected
to {2,3,4}L, commit 969f391) and a printer KeyError (hardcoded tags) that crashed the final
table on two legs (results recovered from npz; printer fixed dynamically). Both fixed; neither
affected the physics numbers.

This resolves the "Residual gap primary suspect: SBM-force systematic" line above: the +51%
Cd excess vs literature is attributed to the traction observable, not to leak drag or real
flow physics.

### 2026-07-27 Traction dissection — term attribution + resolution/blockage verdict

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

Legs: D1 = r9/α=50 baseline (L_inv=16, blockage 6.25%); D2 = r10; D3 = r11;
D4 = L/32 plate at 32 cells/plate (r10-matched), blockage halved to 3.125%.
**dt_used per leg:** D1/D2/D3 = 5e-4 (D3 needed **NO dt fallback** at r11), D4 = 2.5e-4
(by design). **Partition gates:** the in-march per-step gate `1e-12·max(1,|Cd_total|)`
held on every leg; post-hoc max abs residuals from the npz: D1 9.18e-13, D2 1.13e-12,
D3 1.14e-12, D4 1.12e-12 (the values above 1e-12 absolute are within the enforced
relative gate at |Cd| ≈ 2.4–3.0).

**HEADLINE — the PENALTY term carries the force.** On every leg the α=50 penalty
block contributes ~96–97% of the consistent reaction (e.g. D1: 2.5293 of 2.5996);
the consistency+adjoint class is nearly inert (2.7–4.2%); backflow is identically
zero. Numerically, the "variationally-consistent reaction" at this configuration
IS penalty virtual work.

**Bridge ratio (bare σ·n vs consistency class):** 77.25 / 51.11 / 49.09 / 77.18 —
FAR from 1 on every leg. Bare σ·n does NOT approximate the consistency-class term,
so the 2.3× surrogate excess cannot be read as "consistency-carried force plus
penalty/adjoint virtual work on top". Per the design's interpretation rule, the
shifted-face σ·n integration itself is implicated: at α=50 the surrogate and the
reaction measure essentially different functionals.

**Resolution trend (spec decision rule applied verbatim):** Cd_rxn(r9→r10→r11) =
2.5996 → 2.8098 → 3.0483; increments +0.2102, then +0.2385. The sequence is
MONOTONE toward 3.36 but the increments are NOT shrinking (they grew ~13%). The
spec rule — "monotone toward 3.36 with shrinking increments ⇒ deficit = resolution,
extrapolated value recorded; flat/oscillating ⇒ resolution exonerated ⇒ formulation
under the microscope" — has NEITHER branch obtain: the sequence is pre-asymptotic.
No valid extrapolation exists and mesh-convergence is NOT demonstrated; but the
trend is monotone toward literature, not flat/oscillating, so resolution is NOT
exonerated either.

**Blockage direction (D4 vs D2, against the CV trend):** halving blockage at matched
cells/plate moves Cd_rxn 2.8098 → 2.4336 (−13.4%) and Cd_surr 5.8385 → 5.0844
(−12.9%) — the SAME direction and similar magnitude as the 2026-07-26 confinement
probe (Cd 5.72 → 5.09, −11%). Confinement inflates both observables by ~13%;
removing it DEEPENS the reaction's literature deficit (2.4336 at 3.1% blockage =
27.6% below 3.36). St = 0.1875 at both D3 and D4 (literature ~0.15).

**Arbiter reproducibility — OPEN:** D1's Cd_rxn = 2.5996 differs 11% from the LD-5
verdict leg's 2.3440 at the nominally IDENTICAL config (r9, α=50, dt=5e-4, 8000
steps, t_start=1.5), while Cd_surr matches to 4 decimals (5.4369 both) —
trajectory-identical yet reaction-differing: NOT chaotic divergence; UNEXPLAINED.
Leading suspect: a code delta in the reaction path — the LD leg ran at commit
4acaae1 (pre-TD-2), D1 runs post-4472cf2 where TD-2 restructured the reaction
computation; TD-2's parity test pinned only `reaction_terms=False` against its own
base, not `reaction_hist` against the LD-era code. Diagnostic (filed as follow-up,
not done): re-run the LD verdict config on current code and/or diff the two
reaction computations. The headline is robust to it (5.44/2.60 = 2.1× vs
5.44/2.34 = 2.3×).

**OBSERVABLE DECISION — ESCALATED to Baskar (no adoption).** The spec adopts the
consistent reaction as the canonical force observable only "if the reaction
mesh-converges toward literature". Mesh-convergence is NOT demonstrated (monotone
but non-shrinking increments), so no adoption is recorded; the decision escalates
with the term table in hand. For Baskar: (i) the reaction is ~96% penalty virtual
work at α=50 — the "consistent reaction" is here an α-scaled penalty functional,
which connects directly to the standing α-sensitivity finding (+36–41% over
α 20→100) and the α-scaling follow-up; (ii) St ALSO moved toward literature at r11
(0.2188 → 0.1875) — resolution is helping the physics broadly, so the sequence may
still converge beyond r11; (iii) the obvious next probe is a D3b leg at
refine_to=12 (256 cells/plate; 2-D fits the 48 GiB box; dt likely 2.5e-4 at that h;
~1–2 h wall) to test whether the increments start shrinking — Baskar's call, not
run in this campaign; (iv) if r12 still fails to shrink the increments, the
two-sided-shell formulation (α-scaling, penalty-dominated reaction) moves under the
microscope as the systematic.

**Follow-ups filed:** (1) arbiter reproducibility diagnostic (LD-era vs TD-2-era
reaction path, above); (2) D3b r12 resolution probe (offered; Baskar's call);
(3) α-scaling investigation (α~Pe·p² vs fixed 50) — stands, now sharpened by
penalty-dominance; (4) canonical-observable adoption — BLOCKED on the escalation
decision, NOT implemented.
