# P2-R1c 3-D Thin-Plate Flow: GH200 Runbook

## Target

3-D flow past a finite rectangular plate at moderate Re. No literature Cd target
yet — this is a first-principles 3-D case. Qualitative expectations: Cd > 0
(plate is a bluff body), |Cl_y|, |Cl_z| << Cd (transverse forces small by symmetry
at short time, may grow at higher Re/longer time).

## Geometry

Plate: `FiniteSheet(center=(0.375, 0.5, 0.5), half=(0.125, 0.125), normal=(1, 0, 0))`
- A 0.25 x 0.25 square patch in the y-z plane at x=0.375.
- Domain: unit cube [0,1]^3 (octree native).
- Plate area = 0.0625; reference force = 0.5 * U_inf^2 * 0.0625.
- VTP export: 4-vertex rectangle triangulated into 2 triangles (body.vtu via meshio).

## Mac CI Smoke (validated, warp-CPU)

```bash
cd ~/Dropbox/work/Projects/ClaudeCode/DiffSim
.venv/bin/python -m pytest tests/test_p2r1c_thin_plate_flow_3d.py -q
```

Expected: 2 passed, ~2 min on Mac warp-CPU.
Smoke result: Cd[-1]=+29.11 (level=4, Re=10), Cl_y=Cl_z~0 (machine zero).

## Hero Run (GH200, srun)

```bash
srun --gpus-per-node=1 --ntasks=1 \
  bash -c "cd ~/DiffSim && \
  LEVEL=6 NSTEPS=200 DT=0.005 NU=0.004 U_INF=1.0 \
  PLATE_XC=0.375 PLATE_YC=0.5 PLATE_ZC=0.5 \
  PLATE_HALF_Y=0.125 PLATE_HALF_Z=0.125 \
  .venv/bin/python tests/p2r1c_thin_plate_flow_3d.py 2>&1 | tee logs/p2r1c_3d_re250_$(date +%Y%m%d_%H%M%S).log"
```

Or using the remote deploy helper:

```bash
LEVEL=6 NSTEPS=200 DT=0.005 NU=0.004 \
bash scripts/remote/run.sh tests/p2r1c_thin_plate_flow_3d.py
```

## Environment variables

| Variable       | Default | Description                                                      |
|----------------|---------|------------------------------------------------------------------|
| LEVEL          | 4       | Octree level (L4=4096, L5=32768, L6=262144 cells)               |
| NSTEPS         | 5       | Number of BDF2 steps                                             |
| DT             | 0.01    | Time step                                                        |
| NU             | 0.1     | Kinematic viscosity (Re = U_inf / nu for sqrt(plate_area)=0.25) |
| U_INF          | 1.0     | Freestream velocity                                              |
| PLATE_XC       | 0.375   | Plate center x in octree [0,1]                                   |
| PLATE_YC       | 0.5     | Plate center y                                                   |
| PLATE_ZC       | 0.5     | Plate center z                                                   |
| PLATE_HALF_Y   | 0.125   | Plate y half-width (full plate width = 0.25)                     |
| PLATE_HALF_Z   | 0.125   | Plate z half-width (full plate height = 0.25)                    |

## Output artifacts

Results land in `results/` (Dropbox-synced):
- `results/figures/p2r1c_3d_re<N>_L<L>_forces.png` — Cd/Cl_y/Cl_z history plot
- `results/vtu/p2r1c_3d_re<N>_L<L>.vtu` — volumetric flow field (velocity_magnitude, pressure)
- `results/vtu/p2r1c_3d_re<N>_L<L>.pvsm` — ParaView state (color_by=velocity_magnitude)
- `results/vtu/p2r1c_3d_re<N>_L<L>_body.vtu` — rectangular plate surface (4 vertices, 2 triangles; meshio fallback when pyvista absent)

## Recommended GH200 levels

| Level | Cells (3-D)   | DOF (ndof=4)  | Estimated wall time (GH200) |
|-------|---------------|---------------|-----------------------------|
| 4     | 4,096         | ~17K          | < 2 min (validated on Mac)  |
| 5     | 32,768        | ~131K         | ~5-15 min                   |
| 6     | 262,144       | ~1M           | ~30-90 min                  |
| 7     | 2,097,152     | ~8M           | hours (use cuDSS or AMGX)   |

Note: 3-D at level L is 8x larger than 2-D at level L. Level 5 is the first
physically interesting scale for 3-D plate wakes.

## Physics notes

**Oracle strategy:** `FiniteSheet` is used for BOTH `classify_shell_intercepted`
(unsigned psi fine for classification) AND `extract_two_sided_surrogate`
(`distance_vector` returns the fixed patch normal as `n_grad` for all surrogate GPs,
giving correct Gamma~+/Gamma~- split). No need for a separate `Plane` oracle,
unlike the 2-D driver which used `Segment+Plane`.

**Force nondimensionalization:** F / (0.5 * U_inf^2 * plate_area) where
plate_area = 4 * half_y * half_z = 0.0625 for the default geometry.

**Symmetry:** At short time from rest, Cl_y and Cl_z are machine-zero (~1e-7).
They may develop at longer time / higher Re once the 3-D wake becomes asymmetric.

**Outlet BC:** Do-nothing (same as 2-D driver — see p2r1a-thin-plate-runbook.md
for the full discussion of reflection risk and convective outlet options).

## Pass/fail criteria

| Run type   | Gate                                                                              |
|------------|-----------------------------------------------------------------------------------|
| CI (Mac)   | 2 pytest tests green; Cd finite, nonzero, |Cd| < 1000; load-bearing check passes |
| GH200 hero | Cd > 0; |Cl_y|, |Cl_z| < |Cd| (symmetry); VTU+VTP written; no NaN in fields    |

---

## 2026-07-26: 3-D Both-Solver GPU Ladder on gpubox (L4–L6)

**Box:** gpubox — 2x NVIDIA RTX 6000 Ada Generation, 48 GiB VRAM each, sm_89 (WSL2/CUDA 12.9, Driver 13.2).
**Branch:** thinshell-gpu, HEAD e2c2675.
**Config:** DEVICE=cuda:0, DT=0.005, NU=0.004 (Re≈250), NSTEPS=10 (L4/L5) or 3 (L6 mono-attempt).

### Rung Table

| Rung | Solver config | Cells / est. DOF | Steps | Elapsed | s/step | Peak GPU MiB (est.) | Cd[-1] | Outcome |
|------|--------------|-----------------|-------|---------|--------|---------------------|--------|---------|
| L4 mono | MONO_SOLVER=cudss ASSEMBLY=device | 4096 / ~17K | 10 | 3.8s | 0.38 | <500 MiB | +12.49 | PASS — finite, positive, decaying |
| L4 proj | PPE_SOLVER=gpu_cg PRED_SOLVER=fused* | 4096 / ~17K | 10 | 527.5s | 52.8 | <500 MiB | +102.7 | PASS (runs end-to-end) — Cd positive but oscillates; cpu-splu predictor bottleneck dominates; 52.8 s/step vs 0.38 s/step for mono reveals predictor scale problem |
| L5 mono | MONO_SOLVER=cudss ASSEMBLY=device | 32768 / ~131K | 10 | 22.2s | 2.22 | ~1.5 GiB (est.) | +10.09 | PASS — under cuDSS wall, fits in 48 GiB |
| L5 proj | PPE_SOLVER=gpu_cg PRED_SOLVER=fused* | 32768 / ~131K | — | KILLED at 22 min | n/a | <500 MiB (CPU-bound) | — | KILLED (0 steps in budget) — assembly for step 0 still running at 22 min; cpu-splu predictor scales far worse than O(N) |
| L6 proj | PPE_SOLVER=gpu_cg PRED_SOLVER=fused* | 262144 / ~1M | — | KILLED at 2 min | n/a | <500 MiB (CPU-bound) | — | KILLED (0 steps in budget) — same cpu-splu predictor bottleneck, worse at 8x DOF |
| L6 mono-attempt | MONO_SOLVER=fused ASSEMBLY=device | 262144 / ~1M | 3 | 39.4s | 13.1 | ~3–5 GiB (est., device BiCGSTAB) | +22.79 | PASS — UNEXPECTED: fused iterative (BiCGSTAB, fused GPU kernels) converged for the nonsymmetric saddle at α=50; finite, positive Cd (3-step march only; long-march stability unconfirmed) |

*NOTE: PRED_SOLVER env var is not wired in p2r1c_thin_plate_flow_3d_projection.py — the driver always uses
predictor_solver="splu" (CPU scipy sparse LU) regardless of PRED_SOLVER setting. The PPE is gpu_cg (GPU CG,
working), but the predictor is CPU splu at all scales.

**Log files on gpubox:**
- L4 mono: `/home/bglab/Baskar/DiffSim/logs/l4-mono-20260726-013920-71335.log`
- L4 proj: `/home/bglab/Baskar/DiffSim/logs/l4-proj-20260726-013943-71787.log`
- L5 mono: `/home/bglab/Baskar/DiffSim/logs/l5-mono-20260726-014852-73045.log`
- L5 proj: `/home/bglab/Baskar/DiffSim/logs/l5-proj-20260726-014939-73510.log` (killed at 22 min, 11 lines)
- L6 proj: `/home/bglab/Baskar/DiffSim/logs/l6-proj-20260726-021337-76462.log` (killed at 2 min, 11 lines)
- L6 mono-attempt: `/home/bglab/Baskar/DiffSim/logs/l6-mono-attempt-20260726-021543-79843.log`

### Wall Analysis on 48 GiB gpubox

**Monolithic cuDSS wall (not directly tested at L6 — fused iterative tested instead):**
The brief anticipated an ALLOC_FAILED at L6 (~1M DOF) for monolithic cudss on gpubox (48 GiB), mirroring the
GH200 (95 GiB) where the wall was confirmed at ~812K DOF uniform L7. At L6 uniform (~1M DOF), the cuDSS
direct factorization of the 4x4M nonsymmetric saddle would require >10 GiB of fill — ALLOC_FAIL expected but
not confirmed in this run series (L6 rung used fused iterative instead).

**Monolithic fused-BiCGSTAB (L6, THIS RUN):**
MONO_SOLVER=fused with ASSEMBLY=device passed at L6 in 39.4s for 3 steps (13.1 s/step). The device-assembled
nonsymmetric saddle was solved iteratively by fused BiCGSTAB without ALLOC_FAIL. This contradicts the initial
brief expectation of convergence failure for the saddle system with α=50 — the result is HONEST: fused
BiCGSTAB converges here, making L6 monolithic-fused a viable (if slower) path beyond the cuDSS wall.

**Projection predictor wall (UNEXPECTED BOTTLENECK):**
The projection driver's scalable path at L6+ is blocked not by the PPE (gpu_cg works), but by the
predictor solver: the driver hardcodes predictor_solver="splu" (CPU scipy sparse LU), and PRED_SOLVER env var
is ignored. At L5 (~131K DOF), step-0 assembly still running at 22 min (killed). At L6 (~1M DOF), killed at
2 min during assembly (same pattern, 64x worse DOF). The "projection scalable path" requires wiring a
device-side predictor (device FGMRES or AMGX) — this is the R2b block-preconditioner track, NOT available
in the current driver.

### Summary of Boundaries

| Path | Practical ceiling on gpubox (48 GiB) | Next step |
|------|--------------------------------------|-----------|
| Monolithic cuDSS | L5 (~131K DOF) confirmed; L6 cudss NOT tested (would likely ALLOC_FAIL — see GH200 wall at 812K DOF for reference) | cuDSS wall probe deferred |
| Monolithic fused-BiCGSTAB | L6 (~1M DOF) PASSES in 13.1 s/step — viable iterative path beyond cuDSS (3-step march only; long-march stability unconfirmed) | Convergence quality / iteration count study needed |
| Projection gpu_cg PPE | PPE itself scales; L4 runs (52.8 s/step) — bottleneck is predictor, not PPE | Wire device predictor (R2b track) |
| Projection cpu-splu predictor | Dies at L5 (22 min for step 0, killed) — NOT a viable L5+ path | R2b block-preconditioner / FGMRES |

### Cross-reference

Task 8b (GH200 ladder, nova) maps the same envelope at 95 GiB with uniform L4→L7 volumetric-SBM cube-in-channel
(bluff-body first, most-trusted machinery). The gpubox runs here establish the 48 GiB reference points for
the monolithic cudss ceiling and the fused-BiCGSTAB alternative. The GH200's larger VRAM pushes the cudss
wall to ~812K DOF (confirmed); gpubox's cuDSS wall at L6 is anticipated but not confirmed (fused was tested
instead as the HONEST GATE rung).

## 2026-07-26: GH200 Large-DOF Ladder (Task 8b, nova job 11764437)

Three-phase capacity ladder on the nova GH200 (95 GiB HBM3, `--mem=200G`,
4 h): Phase 1 UNIFORM bluff-body cube-in-channel (host assembly + cuDSS),
Phase 1b ADAPTIVE band-refined cube (`tests/adaptive_cube_channel.py`,
CPU-gated), Phase 2 thin-plate adaptive (device assembly + cuDSS). Driver:
`tests/gpu_gh200_ladder.py` via `cluster/slurm/thinshell_gh200_ladder.sbatch`.
Full report: `.superpowers/sdd/task-8b-report.md`.

### Rung Table

Phase 1 — uniform cube (10 steps, dt=0.02, Re=40, offset=0.05):

| rung | n_nodes | saddle DOF | s/step | Cd[10] | smi | status |
|------|---------|------------|--------|--------|-----|--------|
| L4 | 4,877 | 19,508 | 0.54 | +2.076 | 748 MiB | OK |
| L5 | 35,545 | 142,180 | 2.89 | +2.710 | 1.3 GiB | OK |
| L6 | 271,025 | 1,084,100 | 26.12 | +3.379 | 3.2 GiB after (transient ~51 GiB during factor) | OK |
| L7 | 2,115,937 | 8,463,748 | — | — | ~35 GiB at failure | WALL cuDSS ALLOC_FAILED(2) |

Phase 1b — adaptive band-refined cube (same march):

| rung | h_fine | n_nodes (hanging) | saddle DOF | s/step | Cd[10] | smi | status |
|------|--------|-------------------|------------|--------|--------|-----|--------|
| a5r8 | 1/256 | 189,605 (32,976) | 626,516 | 17.71 | +4.366 | 9.8 GiB after (~23 GiB transient) | OK |
| a6r9 | 1/512 | 844,049 (113,616) | 2,921,732 | — | — | 78,338 MiB peak then fail | WALL cuDSS ALLOC_FAILED(2) |
| a6r10 | 1/1024 | — | — | — | — | — | skipped (wall) |

a5r8 reaches near-wall h=1/256 (uniform-L8 territory, ~67M DOF) at 626K DOF —
the adaptive lever buys ~2 orders of magnitude on near-wall resolution per DOF.

Phase 2 — thin-plate adaptive (10 steps, dt=0.005, nu=0.004, device assembly):

| rung | n_excluded | s/step | Cd | smi | status |
|------|------------|--------|----|-----|--------|
| r5b8 | 8,712 | 3.36 | +7.268 | 3.9 GiB | OK |
| r6b9 | 33,800 | 22.21 | +6.721 | 13.7 GiB | OK |
| r7b9 | — | — | — | GPU flat 13.5 GiB | HOST-OOM (200G cgroup) — job killed |

### Three-Wall Analysis (95 GiB GH200)

1. **GPU cuDSS factorization wall:** fits at 1.08M saddle DOF (uniform L6,
   ~51 GiB transient factor peak); fails at 2.92M (a6r9, after climbing to
   78.3 GiB resident — closest measured approach to the ceiling) and at 8.46M
   (L7, early fail at ~35 GiB resident). Measured wall: **between ~1.1M and
   ~2.9M DOF** for this 3-D P1 saddle. This REFINES the earlier "~812K DOF
   GH200 cuDSS wall" note in the cross-reference above: 1.08M DOF factorized
   cleanly — the wall is fill/bandwidth-dependent, not a fixed DOF count.
2. **Host mesh-build wall (new failure class):** r7b9 (base-L7 tree + band-L9
   refinement) exceeded the **200G host cgroup** during the HOST-side
   octree/mesh/constraints build — SLURM oom_kill (sacct MaxRSS 209.7 GB, exit
   137), GPU idle at 13.5 GiB. Mitigation: raise `--mem` toward Grace's 480 GB
   and/or slim the host mesh-build intermediates.
3. **Not run:** the projection secondary legs, final summary table, and the
   `GH200-LADDER-OK` sentinel never printed — the host OOM ended the job inside
   the r7b9 build. All rungs through r6b9 completed and are recorded above.

**WP0 RESOLUTION (2026-07-27, Grace job 11771926): the adaptive mesh build is EXONERATED.** The exact r7b9 case (base L7/band r9, 2.31M nodes / 9.24 M-DOF) builds in **3.71 GB peak, 94 s** on Grace (0.40 GB/M-DOF; L6/r9: 0.89 — sublinear per-DOF). The 209.7 GB OOM was CROSS-LEG ACCUMULATION in the single-process GH200 ladder (prior legs' assembled CSRs + the failed L7-uniform cuDSS factorization's host staging), mis-attributed to the build because the job died during that leg with the GPU idle. Mesh build extrapolates to ~40-90 GB at 100M DOF — inside Grace. The streaming build reclassifies to the medium-term remeshing/mixed-p architecture; the 100M critical path is the solver (R2b saddle preconditioner / matrix-free).

Oversubscription decision: the film's `--managed` lever
(`wp.set_device_allocator` + `CudaManagedAllocator`) governs Warp-owned arrays
only; cuDSS allocates its factors internally (nvmath), so the lever does NOT
transfer. Observed C2C behavior at the wall: **clean fast-fail ALLOC_FAILED, no
spill, no crawl** — GH200+cuDSS is binary fit/no-fit under default allocation.
A true managed probe needs cuDSS's `cudssDeviceMemHandler` wired to a managed
pool (solver-path change, out of 8b scope).

### Cross-reference

Complements the 2026-07-26 gpubox (48 GiB) ladder section above: gpubox
established cuDSS-at-L5 + fused-BiCGSTAB-at-L6 on the thin-plate driver; the
GH200 run bounds the cuDSS direct-factor envelope at 95 GiB on both uniform and
adaptive bluff-body meshes and adds the host-RAM mesh-build ceiling as the
binding constraint for base-L7 adaptive builds.
