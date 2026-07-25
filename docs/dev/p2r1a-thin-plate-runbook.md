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
