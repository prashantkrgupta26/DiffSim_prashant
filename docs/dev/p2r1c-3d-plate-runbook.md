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
