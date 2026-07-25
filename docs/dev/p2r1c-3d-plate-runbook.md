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
