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

## Task A2 - Boundary Conditions: Explore (a)
**Date:** July 27, 2026  
**Experiment:** Changed the manufactured solution to $u^*(x,y)=\sin(\pi x)\sin(\pi y)$ while leaving the x-only case unchanged.

### Observed Results
- **Dirichlet on all faces:**
  - Errors: `[1.606e-03, 4.015e-04, 1.004e-04]`
  - Convergence orders: **`2.00`, `2.00`** ($\mathcal{O}(h^2)$)
- **Dirichlet on x-faces only:**
  - Errors: `[4.673e-01, 4.692e-01, 4.697e-01]`
  - Convergence orders: **`-0.01`, `-0.00`** (effectively zero order)

### Interpretation
The sine manufactured solution has nonzero normal flux on the y-faces:

$$
\frac{\partial u^*}{\partial y}=\pi\sin(\pi x)\cos(\pi y).
$$

Leaving the y-faces without an explicitly assembled boundary term silently imposes zero Neumann flux, which is inconsistent with this manufactured solution. Therefore the x-only error remains near `0.47` instead of decreasing under refinement: the boundary-condition error dominates and cannot be removed by refining the mesh. Applying Dirichlet conditions on all faces remains consistent and retains second-order convergence.

### Explore (b) Pinning Test — User-Provided Run
**Observed output:**

- **Dirichlet on all faces:** errors `[1.606e-03, 4.015e-04, 1.004e-04]`, orders `2.00`, `2.00`.
- **Dirichlet on x-faces only:** errors `[4.673e-01, 4.692e-01, 4.697e-01]`, orders `-0.01`, `-0.00`.
- **Pure Neumann with one pinned node:** errors `[2.192e+01, 2.544e+01, 2.896e+01]`, orders `-0.21`, `-0.19`.

**Context:** This run used the sine manufactured solution from Explore (a). The pinned pure-Neumann case is therefore not a valid pure-Neumann MMS test, because the sine solution has nonzero normal flux on the y-faces while the code imposes zero natural flux there. The large, increasing pin-case error is consequently expected. For a meaningful pinning test, restore the original cosine solution, whose normal flux is zero on all faces, and rerun the `all`, `x-only`, and `pin` cases.

### Explore (c) Dirichlet Row-Replacement Timing — User-Provided Run
**Observed output:**

| Level | DOFs (`n`) | Boundary nodes | Row-replacement time (s) |
|---:|---:|---:|---:|
| 5 | 1089 | 128 | 0.000223 |
| 6 | 4225 | 256 | 0.000458 |
| 7 | 16641 | 512 | 0.000911 |
| 8 | 66049 | 1024 | 0.001792 |

Measured exponents, computed as `log(time ratio) / log(DOF ratio)`:

```text
row replacement: 0.53  0.50  0.49
```

**Interpretation:** The row-replacement time approximately doubles whenever the boundary-node count doubles. Its scaling exponent is about `0.5` with respect to total DOFs, consistent with a 2-D boundary growing like $\sqrt{n}$. The loop remains inexpensive at these sizes, but its host-side Python cost grows and may matter for much larger meshes.

## Task A3 - Shifted Boundary MMS
**Date:** July 28, 2026  
**Run source:** User-provided terminal output.

### Observed Results

| Level | Mesh size `h` | L2 error |
|---:|---:|---:|
| 4 | 0.0625 | 3.210e-03 |
| 5 | 0.0312 | 6.615e-04 |
| 6 | 0.0156 | 1.508e-04 |

Observed convergence orders:

```text
level 4 -> 5: 2.28
level 5 -> 6: 2.13
```

**Interpretation:** Each refinement halves `h`, and the L2 error decreases by approximately a factor of four. The measured orders are slightly above 2 but move toward the expected second-order rate, which is normal in the pre-asymptotic regime. The A3 shifted-boundary implementation therefore shows the expected approximately `O(h^2)` convergence.

### A3 Explore (a) — Quadratic Basis (`p=2`)
**Observed output:**

| Level | Mesh size `h` | L2 error |
|---:|---:|---:|
| 4 | 0.0625 | 1.726e-04 |
| 5 | 0.0312 | 1.423e-05 |
| 6 | 0.0156 | 1.808e-06 |

Observed convergence orders:

```text
level 4 -> 5: 3.60
level 5 -> 6: 2.98
```

**Interpretation:** The order approaches `3.0`, as expected for quadratic elements in the L2 norm. The first interval is mildly pre-asymptotic (`3.60`), while the finer-mesh interval gives `2.98`, confirming the expected `O(h^3)` behavior. The terminal label should say the order should approach `3.0`, not `2.0`, for this `p=2` run.

### A3 Explore (b) — Retain Intercepted Elements (`lambda=1.0`)
**Observed output:**

| Level | Mesh size `h` | L2 error |
|---:|---:|---:|
| 4 | 0.0625 | 9.324e-03 |
| 5 | 0.0312 | 3.753e-03 |
| 6 | 0.0156 | 6.083e-04 |

Observed convergence orders:

```text
level 4 -> 5: 1.31
level 5 -> 6: 2.63
```

**Interpretation:** The coarser interval is pre-asymptotic, but the finer interval gives a super-second-order value of `2.63`; together these results are consistent with the expected convergence approaching `O(h^2)` as the mesh is refined. With `lambda=1.0`, intercepted elements are retained and the surrogate boundary lies on the outside side of the true circle. The shifted-boundary correction still recovers the expected asymptotic order.

### A3 Explore (c) — Sampled `GridSDF` Geometry
**Experiment:** Temporarily replaced the exact `Sphere` oracle with
`GridSDF.from_oracle(Sphere((0.5, 0.5), 0.3), n=128)`, while keeping `p=1` and `lambda=1.0`.

**Strict projection result:** The run stopped before solving because Newton closest-point projection failed at 8 of 80 surrogate quadrature points at level 4. The admissibility report gave `newton_ok_frac = 0.9`. This is evidence that the sampled, piecewise-linear GridSDF is less smooth than the analytical circle and does not satisfy the strict projection assumptions at every point.

**Fallback result:** With the optional `max_fail_frac=0.1` projection fallback enabled, the run completed:

| Level | Mesh size `h` | L2 error |
|---:|---:|---:|
| 4 | 0.0625 | 9.317e-03 |
| 5 | 0.0312 | 3.791e-03 |
| 6 | 0.0156 | 6.401e-04 |

Observed orders were `1.30` and `2.57`. These are very close to the earlier exact-circle `lambda=1.0` results (`1.31`, `2.63`) at these levels, but the sampled geometry introduces projection failures and a small error change. The temporary code changes were restored after testing; this is a diagnostic result, not a permanent change to the A3 tutorial.
