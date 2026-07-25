# P2-R1a Re=250 Thin Plate: gpubox Full-Resolution Runbook

## Target

Cd ≈ 3.29–3.45, St ≈ 0.15 (ThinShell.pdf Table 1; Najjar & Balachandar 1995: Cd=3.36, St=0.14).

## Command

```bash
ssh gpubox "cd ~/DiffSim && \
    LEVEL=7 NSTEPS=1000000 DT=5e-5 NU=0.004 U_INF=1.0 \
    PLATE_XC=0.1389 PLATE_YC=0.5 PLATE_L=0.0625 \
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
St, freq = strouhal(data["t"], data["cl"], U=1.0, L=1.0/16.0)
print(f"Cd_mean={cd_mean:.4f}  St={St:.4f}  freq={freq:.4f}")
```

Note: `L=1.0/16.0` is the normalized plate length in the octree [0,1]² domain
(the reference length used in force nondimensionalization).

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

## Pass/fail criteria

| Run type    | Gate                                      |
|-------------|-------------------------------------------|
| CI (Mac)    | Pipeline runs end-to-end; Cd_mean finite and O(1)-plausible (no lit values asserted) |
| gpubox full | Cd_mean ∈ [3.29, 3.45], St ∈ [0.12, 0.18] (ThinShell.pdf Table 1 band) |
