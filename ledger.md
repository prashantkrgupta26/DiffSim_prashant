## Task A1 - MMS Convergence Test
**Date:** July 26, 2026  
**Script:** `tutorials/A_foundations/A1_mms_convergence.py`  
**Hardware:** NVIDIA RTX 2000 Ada Generation (16 GiB, CUDA Toolkit 12.9, Driver 12.8, Warp 1.15.0)  

### Observed Convergence Rates:
- **p=1 (Linear Elements):**
  - Errors: `[1.302e-03, 3.255e-04, 8.138e-05]`
  - Convergence Orders: **`2.00`**, **`2.00`** ($\mathcal{O}(h^2)$)
- **p=2 (Quadratic Elements - Patch Test for $u^* = x^2 + y^2$):**
  - Errors: `[1.985e-15, 5.943e-15, 2.943e-14]`
  - Convergence Orders: **Machine Precision Zero** ($\sim 10^{-15}$, exact discrete representation)

### Generated Figures:
- `tutorials/A_foundations/figures/convergence_p1.png`
- `tutorials/A_foundations/figures/convergence_p2.png`

### Notes / Findings:
- **Linear ($p=1$)**: Achieved exact theoretical $L_2$ convergence rate of $\mathcal{O}(h^2)$ (`2.00`, `2.00`).
- **Quadratic ($p=2$)**: Passed patch test for $u^* = x^2 + y^2$ with $L_2$ errors at machine precision zero ($\approx 10^{-15}$), since exact quadratic solution lies in the discrete $p=2$ space.
- CUDA kernel compilation and device table assembly executed cleanly with cached module load times ~1ms on NVIDIA RTX 2000 Ada Generation GPU.

### Explore (c) Performance Study
**Run context:** CPU backend (the current environment had no available CUDA driver). Timings are therefore CPU measurements, not RTX GPU measurements.

| Level | DOFs (`n`) | Mesh + constraints (s) | Assembly (s) | Solve (s) |
|---:|---:|---:|---:|---:|
| 4 | 289 | 0.0009 | 0.0010 | 0.0004 |
| 5 | 1089 | 0.0009 | 0.0022 | 0.0019 |
| 6 | 4225 | 0.0014 | 0.0082 | 0.0083 |
| 7 | 16641 | 0.0055 | 0.0244 | 0.0414 |
| 8 | 66049 | 0.0105 | 0.1451 | 0.2428 |

Measured exponents, computed as `log(time ratio) / log(DOF ratio)`:

- **Mesh:** `0.06, 0.33, 0.97, 0.47` — noisy at small sizes because fixed overhead dominates.
- **Assembly:** `0.56, 0.97, 0.80, 1.29` — approximately linear, consistent with `O(n)` overall behavior.
- **Solve:** `1.19, 1.08, 1.17, 1.28` — superlinear and trending toward the expected 2-D sparse-solve scaling near `O(n^1.5)`.

**Interpretation:** At level 8, the solve is the largest cost (`0.2428 s`) and is the main bottleneck. Assembly is next (`0.1451 s`) and also grows substantially. Mesh construction remains comparatively inexpensive. Small-level exponents fluctuate because the measured runtimes are very short; levels 6–8 provide the more meaningful scaling trend.

### Explore (c) GPU-Visible Rerun
The subsequent run detected and used `cuda:0`:

```text
NVIDIA RTX 2000 Ada Generation (16 GiB, sm_89)
CUDA Toolkit 12.9, Driver 12.8
```

| Level | DOFs (`n`) | Mesh + constraints (s) | Assembly (s) | Solve (s) |
|---:|---:|---:|---:|---:|
| 4 | 289 | 0.0007 | 0.0010 | 0.0004 |
| 5 | 1089 | 0.0008 | 0.0020 | 0.0019 |
| 6 | 4225 | 0.0013 | 0.0073 | 0.0084 |
| 7 | 16641 | 0.0054 | 0.0248 | 0.0415 |
| 8 | 66049 | 0.0105 | 0.1501 | 0.2433 |

Measured exponents:

- **Mesh:** `0.12, 0.40, 1.03, 0.49`
- **Assembly:** `0.52, 0.98, 0.89, 1.31`
- **Solve:** `1.22, 1.08, 1.16, 1.28`

The Warp element, load, and error kernels were loaded on `cuda:0`. The `splu` sparse solve remains CPU-based, so the solve timings are not GPU solve timings. At level 8, the solve remains the largest measured stage (`0.2433 s`), followed by assembly (`0.1501 s`).
