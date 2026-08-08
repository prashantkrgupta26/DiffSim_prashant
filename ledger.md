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

### Explore (b) — Break the MMS: keep `f`, set `g=0`

Ran two variants (no ledger entry existed previously for this task):

- **`u*=sin(pi x)sin(pi y)`** (f correct, g=0 everywhere): errors `[1.606e-03, 4.015e-04, 1.004e-04]`, orders `2.00, 2.00` — **identical** to the correct-g control, because this particular `u*` is already exactly zero on the whole boundary, so `g=0` happens to be exactly right. Nothing was actually broken.
- **`u*=x^2+y^2`** (the file's current default, nonzero on the boundary; f correct, g=0 everywhere): errors `[9.045e-01, 9.051e-01, 9.052e-01]`, orders `-0.001, -0.000` — error plateaus around `0.905` and does not shrink under refinement at all.

**Interpretation:** boundary-data errors don't average out under mesh refinement — they set a hard floor on accuracy, exactly like A2's Explore (a) finding for a Neumann-side flux error. The `sin*sin` case is a cautionary tale of its own: it shows a manufactured solution can accidentally be immune to a "broken" boundary condition if it happens to vanish there, which would make the experiment falsely look like a non-issue — the choice of `u*` matters for what a test can actually detect.

## Task A1 Addendum — Note on the file's own EXPECTED RESULTS vs. what's recorded

The docstring's EXPECTED RESULTS block (`p=1: ~1.61e-3/... order 2.00`; `p=2: ~2.05e-4/... order 2.99/3.00`) describes the original `u*=sin(pi x)sin(pi y)` MMS case. The file's module-level `u_star`/`f_star` currently implement `u*=x^2+y^2` instead (Explore (a)'s patch-test case) — so the "Observed Convergence Rates" recorded above are actually Explore (a)'s result, not a run of the docstring's own stated baseline. Both are legitimate MMS choices; flagging only so the numbers above are read in the right context.

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

### Performance Corner — `classify_lambda` timing vs. level (untitled task at the end of the file, no prior ledger entry)

The claim to test: cost should track the number of **intercepted** (narrow-band) elements, `O(2^level)`, not the total element count, `O(4^level)` — because of the two-pass design (a cheap center-distance check resolves most elements immediately; only the narrow band around the true boundary pays the expensive dense per-element quadrature).

Direct element-count check (recomputing the same narrow-band criterion `classify_lambda` uses internally: `|psi(center)| <= lipschitz_bound*(sqrt(dim)/2)*h`), levels 4-8:

| level | total elements | narrow-band elements | band growth |
|---:|---:|---:|---:|
| 4 | 256 | 88 | — |
| 5 | 1,024 | 176 | 2.00x |
| 6 | 4,096 | 336 | 1.91x |
| 7 | 16,384 | 688 | 2.05x |
| 8 | 65,536 | 1,368 | 1.99x |

Wall-clock timing (median/min of 11 reps, after warming all levels first) was too noisy to extract a reliable exponent at this scale — all runs are under 5ms, deep in Python-dispatch/launch-floor-dominated territory on this shared machine (e.g. level 7 median `5.07ms` vs. min `1.18ms`, a 4x spread from run to run).

**Interpretation:** the element-count check is a clean, direct, noise-free confirmation of the claim — the narrow band grows almost exactly `2.00x` per level (total elements always grow exactly `4.00x`), matching `O(2^level)` to within measurement precision. Wall-clock timing at these small absolute run times isn't trustworthy evidence either way; the algorithmic claim itself is nonetheless solidly confirmed by directly counting what the two-pass logic actually does, independent of timer noise. A real timing measurement of this claim would need levels large enough that the narrow-band work (tens of thousands of elements, not hundreds) dominates fixed per-call overhead.

## Task A4 - Mixed p1/p2 Elements and Minimum Rule
**Date:** August 1, 2026  
**Script:** `tutorials/A_foundations/A4_mixed_elements.py`

### Bug fix: p2 region was a thin column, not a half-domain

`solve()` normalized `xs` by `1 << level` instead of `tree.anchors().max()`, so p2 covered only the leftmost element column (`2^-level` of the mesh) instead of the claimed left half. Fixed to use `anchors[:, 0]` (already computed, previously unused); verified 50.00% p2 at levels 4/5/6. The old ledger entry above (`1.73`/`1.87` orders) was measured on the buggy mesh and is superseded below.

### Observed Results (corrected, true 50/50 split)

- **rich-L (p2 on left):** errors `[1.251e-03, 3.108e-04, 7.751e-05]`, orders `2.01`, `2.00` (finest mesh: `10401` DOFs).
- **rich-R (p2 on left):** errors `[6.426e-03, 1.614e-03, 4.039e-04]`, orders `1.99`, `2.00` (finest mesh: `10401` DOFs).

### Interpretation

Both cases still converge at order ~2 — the p1 half caps the global rate regardless of where p2 sits. Now rich-L is a stable ~5.2x more accurate than rich-R at every level (5.14x/5.19x/5.21x), unlike the buggy run's shrinking 1.5x margin (an artifact of the p2 fraction itself shrinking each level). With a real half-domain split, most of rich-L's rich zone near `x=0` sits clear of the p2/p1 interface and keeps full p2 accuracy; only a fixed-width band near `x=0.5` is taxed to p1.

### Explore (a) — Constraint counts at the p2/p1 interface

Level-4 mesh (128 p2, 128 p1 elements): `Nn=697`, free=`681`, hanging=`16`. All 16 hanging nodes sit on `x=0.5`, one per interface element (16 p2 elements border it). Sample interpolation row: node `(0.5, 0.03125) = 0.5*(0.5, 0.0) + 0.5*(0.5, 0.0625)`.

**Interpretation:** the minimum rule pins exactly one DOF per interface element — the quadratic mid-face mode — to the linear average of its two corners, i.e. a hat function riding on what would be a quadratic edge trace.

### Explore (b) — Strip `|x-0.5|<0.25` vs. half-domain, same p2 budget

| region | rich-L errors (lv 4/5/6) | rich-R errors (lv 4/5/6) | p2 frac |
|---|---|---|---|
| p1 | 6.537e-3 / 1.642e-3 / 4.110e-4 | 6.537e-3 / 1.642e-3 / 4.110e-4 | 0.0 |
| half | 1.251e-3 / 3.108e-4 / 7.751e-5 | 6.426e-3 / 1.614e-3 / 4.039e-4 | 0.5 |
| strip | 5.811e-3 / 1.463e-3 / 3.663e-4 | 5.811e-3 / 1.463e-3 / 3.663e-4 | 0.5 |
| p2 | 2.183e-4 / 2.733e-5 / 3.418e-6 | 2.183e-4 / 2.733e-5 / 3.418e-6 | 1.0 |

**Interpretation:** no — half the budget does not buy most of the benefit; placement dominates budget. The symmetric strip gives identical, modest gains (~11% over p1) for both L and R. `half` at the same budget swings from ~5.2x better (rich-L, well-placed) to ~1.7% better (rich-R, misplaced). `half`'s gap to the full-p2 ceiling also widens under refinement for rich-L (5.7x -> 22.7x, level 4 to 6), since full p2 holds order 3 while the interface-capped half is stuck at order 2.

### Explore (c) — Connection to m1a finding 4b (Neumann band)

`docs/dev/m1a-deferred-findings.md` 4b(c): band elements carrying the Hessian must be decoupled from the minimum-rule p1 trace constraints — thin rings (1-2 layers) cap order at ~1; >=3-4 layers restore order 2.

**Restated for A4:** the SBM Neumann p2-band's outer face is subject to the same minimum-rule constraint measured in Explore (a). A too-thin band lets the constrained trace strip overlap the elements meant to carry the Hessian, capping order at ~1. Explore (b) is the steady-state analogue: `half` keeps its rich zone several elements clear of the interface (thick enough), while `strip` puts the constrained band directly on the region that would benefit (too thin), and the payoff nearly vanishes.

### Explore (d) — Assembly time: p1 vs p2 vs mixed (level 6, 4096 elements each)

| config | DOFs | nnz | assemble median/min (ms) |
|---|---:|---:|---:|
| p1-uniform | 4225 | 37249 | 1.083 / 1.028 |
| p2-uniform | 16641 | 263169 | 6.694 / 5.766 |
| mixed 50/50 | 10401 | 149281 | 3.768 / 3.635 |

**Interpretation:** DOF ratio p2/p1 measured `3.94x`, not the docstring's predicted `(3/2)^2=2.25x` (same-level p2 node density approaches 4x p1 asymptotically). Assembly time ratio `6.18x` is close to the predicted per-element `(9/4)^2=5.06x` (matches the `nbf^2` scatter-size ratio 81/16). Mixed time (`3.768 ms`) matches `mean(p1,p2)=3.889 ms` within 3%: `volume_triplets` launches one Warp kernel per p-bin over its own disjoint elements, so a 50/50 split is just the mean of the two uniform costs, no cross term.

## Task A5 - Going 3-D
**Date:** August 2, 2026  
**Script:** `tutorials/A_foundations/A5_three_dimensions.py`

### Observed Results

- **box p1** (levels 3/4/5): errors `[4.548e-03, 1.136e-03, 2.840e-04]`, orders `2.00`, `2.00`.
- **box p2** (levels 2/3/4): errors `[1.386e-03, 1.772e-04, 2.227e-05]`, orders `2.97`, `2.99`.
- **sphere p1, immersed** (levels 3/4): errors `[3.014e-03, 1.190e-03]`, order `1.34` (preasymptotic, as the docstring predicts).

Matches the docstring's EXPECTED RESULTS exactly — no bug this time.

### Explore (a) — Staged cost table, levels 3-6

| level | DOFs | nnz | mesh+cons | assemble | solve (factorize+backsolve) | solve/assemble |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 729 | 15,625 | 0.001s | 0.002s | 0.003s | 1.3x |
| 4 | 4,913 | 117,649 | 0.002s | 0.008s | 0.105s | 13.6x |
| 5 | 35,937 | 912,673 | 0.019s | 0.061s | 13.40s | 220x |
| 6 | 274,625 | ~7.1M | — | — | **OOM-killed** | — |

Level 6 (p1-uniform box) was attempted directly and the process was killed by the OS OOM killer mid-factorization, `anon-rss` at `14.6 GB` on this machine's `15 GiB` RAM, before finishing.

**Interpretation:** solve time scaling exponent measured `1.83` (lv 3->4) then `2.44` (lv 4->5) — consistent with the docstring's ~O(n^2) claim (and trending above it, likely early swap pressure at level 5). Solve overtakes assembly by level 3 already (`n=729`) in 3-D; the 2-D crossover (from A1's performance-corner table) doesn't happen until between level 5 and 6 (`n≈1089-4225`). Extrapolating O(n^2) from level 5 predicts a ~13-minute level-6 solve, but that number is moot — the real bottleneck is **memory**, not time: splu fill-in in 3-D (~O(n^{4/3})) exhausted RAM before finishing. This is exactly why track-P's iterative solvers exist: a direct solver doesn't just get slow in 3-D, it becomes infeasible on commodity RAM well before it gets slow enough to notice.

### Explore (b) — k=4 estimate (no run)

| level | elements | p2 DOFs | nnz (~`5^4`/row) | assembled matrix (~16B/nnz) |
|---:|---:|---:|---:|---:|
| 2 | 256 | 6,561 | 4.1M | 0.07 GB |
| 3 | 4,096 | 83,521 | 52.2M | 0.84 GB |
| 4 | 65,536 | 1,185,921 | 741.2M | **11.9 GB** |

**Interpretation:** level-4 k=4 p2 needs `65,536` elements and `1,185,921` DOFs. Just the assembled matrix already approaches this machine's 16 GiB GPU budget, before counting solver fill-in or host-side triplet staging — level 3 fits comfortably, level 4 is right at the edge / likely over. Matches the m1a finding-6 reference point (~7,105 nodes was the practical scale for 4D p2 in the test suite) — that's between this table's level 2 and level 3.

### Explore (c) — Profile the sphere solve at level 5

| stage | time | share |
|---|---:|---:|
| solve | 0.102s | 42.9% |
| classify | 0.076s | 31.9% |
| geometry_data | 0.021s | 8.9% |
| extract_surrogate | 0.012s | 5.1% |
| build_constraints | 0.012s | 5.0% |
| build_mesh | 0.009s | 3.8% |
| build_uniform | 0.003s | 1.2% |
| sbm_setup | 0.002s | 0.7% |
| device_mesh | 0.001s | 0.5% |

(`n_elems=2968`, `n_dofs=3791`, total wall `0.27s`.)

**Interpretation:** this does NOT match what the explore prompt hints at ("the m0.5 findings say constraints") — here `build_constraints` is only 5% of the time; `solve` and `classify` dominate. Checked the referenced finding directly: m1a finding 6 measured `build_constraints` at `48s` for a 4D p2 mesh, but that was the OLD, unvectorized host-probe-loop implementation (one `LeafLookup.find` call per node per probe). The current `build_constraints` (vectorized campaign, 2026-07-05) batches all probes into one call and was already confirmed elsewhere (P1 tutorial) to drop from `222s` to `0.71s` at 66k 2-D DOFs. At this sphere mesh's tiny scale (3,791 DOFs), that fix means constraints is no longer the bottleneck — the old finding describes code that no longer exists in this form. Re-measuring beats trusting a stale reference.

## Task A6 - Complex Geometry (CSG Carves)
**Date:** August 4, 2026  
**Script:** `tutorials/A_foundations/A6_complex_geometry.py`  
**Run source:** User-provided terminal output.

### Observed Results

- **channel (inside a box):** errors `[2.522e-04, 1.373e-04]`, order `0.88`.
- **plate minus disk (outside):** errors `[1.024e-03, 2.493e-04]`, order `2.04`.
- **icosphere STL-path (outside, 3-D):** errors `[1.961e-02, 4.138e-03]`, order `2.25`.

Matches the docstring's EXPECTED RESULTS closely (`0.88` / `2.04` / `2.25`) — no bug.

### Interpretation

The two smooth-boundary carves (disk, icosphere) recover the expected order-2 SBM accuracy. The channel is the outlier at order `0.88`: a box has corners, where the SDF is not differentiable and the closest-point projection is ambiguous (multiple equally-close points), breaking the smoothness assumption the Taylor shift relies on. The corner elements locally degrade accuracy and drag down the global rate, even though most of the channel's boundary (the flat sides) is perfectly smooth.

### Explore (a) — Compose channel MINUS disk

Built `Intersection(Box(channel), Complement(Sphere(r=0.1)))`, domain=`"inside"` — flow-past-a-cylinder-in-a-pipe geometry. Control check: running plain `Box` alone through the same harness reproduced the base script's `0.88` exactly, confirming the implementation is correct.

**Result:** errors `[3.182e-04, 1.387e-04]` (levels 5/6), order `1.20`.

**Interpretation:** the prompt says "verify order 2" — measured order is `1.20`, not 2. The disk removal doesn't touch the box's four corners, so the same corner degradation from the base channel case persists (the disk only improves the constant: `1.20` vs the plain channel's `0.88`). The explore prompt's expectation doesn't hold here.

### Explore (b) — Corner-effect diagnosis (mask near-corner elements)

Re-solved the plain channel and measured L2 error excluding all quadrature points within `k*h` of any of the box's 4 corners, for `k = 1, 2, 3, 5, 10`.

| exclusion | order (levels 5/6) |
|---|---:|
| none (full domain) | 0.88 |
| 1h | 0.88 |
| 2h | 0.87 |
| 3h | 0.86 |
| 5h | 0.82 |
| 10h | 0.49 |
| 20h | degenerate — mask empties most of the small channel domain |

**Interpretation:** contrary to what the prompt expects ("smooth order should reappear"), masking away from the corners does NOT recover order 2 — order barely moves out to 5h, and gets *worse* at 10h. This means the corner's damage isn't just a local contribution to the error norm that can be filtered out after the fact: through the coupled linear system, a badly-conditioned corner region pollutes the discrete solution field more globally (elliptic PDEs don't guarantee numerical defects stay local). The corner is still the right root cause (the smooth-only cases converge cleanly), but the fix has to attack the corner itself, not the error measurement — e.g. round the corner (blend the Box SDF with a small-radius fillet via `Union`/smooth-min), or add local h-refinement concentrated at the corners to better resolve the ambiguous closest-point region there.

### Explore (c) — Stanford bunny (real STL-path test)

Downloaded the real `bun_zipper.ply` from the Stanford 3D Scanning Repository (`graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz`; 35,947 verts, 69,451 tris), scaled to fit `[0,1]^3` (max extent `0.6`, centered at `0.5`), wrapped in `TriMeshOracle`, and ran the same `domain="outside"` case-3 pipeline.

| level | DOFs | error | wall |
|---:|---:|---:|---:|
| 3 | 707 | 1.499e-02 | — |
| 4 | 4,744 | 3.053e-03 | — |
| 5 | 34,546 | 7.498e-04 | 9.1s |
| 6 | 263,518 | 1.781e-04 | 785s |

Orders: `2.30` (3→4), `2.03` (4→5), `2.07` (5→6) — no plateau yet.

Mean/max triangle edge length after scaling: `0.0057` / `0.0189`. Sagitta estimate `s²/(8R)`: `~4e-5` for smooth body regions (`R~0.1`) up to `~2.2e-3` for thin/curved features like the ears (`R~0.02`, max edge).

**Interpretation:** no error floor observed through level 6 — order stays clean at ~2.0-2.3 throughout, because octree `h` at level 6 (`0.0156`) is still larger than the mesh's own mean facet size (`0.0057`); the octree hasn't yet out-resolved the geometry. The level-6 error (`1.78e-4`) already falls inside the broad sagitta estimate range, hinting the ears/thin features may already be facet-limited even while the smooth body is still octree-limited — but confirming a true global floor needs level 7-8, which was not attempted: level 6 alone cost 785s, and the level 5→6 solve-time growth was already steep enough that level 7 risked the same OOM/impractical-runtime outcome as A5's level 6.

### Explore (d) — Performance corner: `GeometryData.evaluate` timing vs level

Isolated `GeometryData.evaluate` (icosphere case), timed with repeated calls (median/min of 7-9 reps) per level to reduce noise.

| level | n surrogate faces | median | min |
|---:|---:|---:|---:|
| 3 | 192 | 0.001s | 0.001s |
| 4 | 528 | 0.009s | 0.006s |
| 5 | 1,992 | 0.014s | 0.009s |
| 6 | 7,320 | 0.075s | 0.070s |
| 7 | 28,512 | 0.148s | 0.102s |
| 8 | 112,344 | 2.506s | 2.009s |

**Interpretation:** expected exponent is ~1 — each Gauss point is one independent BVH query against a fixed-size triangle mesh, so cost should scale linearly with the number of query points. Small-level measurements (levels 3-6) are launch-floor-dominated and noisy — repeated runs of the *same* level transition gave wildly different exponents (`0.18` to `2.3`) run-to-run on this shared machine, so they're not trustworthy. At the largest, most reliable scale (level 7→8, min times): faces grew `3.94x` but time grew `~19.7x` — exponent `~2.17`, clearly super-linear, not the ~1 the "independent per-point query" model predicts. Something in `GeometryData.evaluate` (likely the FP64 torch closest-point/region-masking machinery, or host-device sync overhead) stops scaling cleanly once the query count gets large — worth a closer look if this geometry path is ever pushed to production mesh sizes.

## Task B1 - Bratu Newton
**Date:** August 6, 2026  
**Script:** `tutorials/B_nonlinear/B1_bratu_newton.py`

### Observed Results

- **Newton residuals** (level 5, p1, lam=3): `1.42e-02, 1.66e-03, 3.08e-05, 8.04e-09, 1.17e-15` — 5 steps, clearly quadratic.
- **Spatial orders:** p1 `[2.353e-03, 5.898e-04, 1.476e-04]` -> orders `2.00, 2.00`; p2 `[2.056e-04, 2.574e-05, 3.219e-06]` -> orders `3.00, 3.00`.

Matches the docstring's EXPECTED RESULTS exactly — no bug.

### Explore (a)+(b) — Modified Newton vs. Picard vs. full Newton

Iteration counts to `tol=1e-12` (level 5, p1, lam=3):

| variant | iterations | convergence |
|---|---:|---|
| full Newton | 5 | quadratic (residual ~squares each step) |
| modified Newton (Jacobian frozen at 1st iterate) | 16 | linear, rate ~0.20/step |
| Picard (drop `-M_jac` entirely, `J=K`) | 22 | linear, rate ~0.32/step |

**Interpretation:** all three converge (the problem is well-posed at lam=3, far from the fold), but only full Newton is quadratic. Modified Newton still uses a real (if stale) linearization of the exponential term, giving a consistent linear rate; Picard uses none at all (just lags the whole nonlinear term one step behind), giving a slower linear rate. This is the textbook Newton/chord/Picard hierarchy, reproduced numerically: exact Jacobian -> quadratic, frozen Jacobian -> linear (fast), no Jacobian -> linear (slower).

### Explore (c) — Continuation toward lam* (physical Bratu, f=0, g=0)

Warm-started continuation from lam=0.5 up to lam=6.82 (level 5, p1), tracking the lower branch, reusing each converged solution as the next initial guess:

| lam | iterations | converged | max\|u\| |
|---:|---:|---|---:|
| 0.5 - 6.6 | 3-4 | yes | 0.038 -> 1.066 |
| 6.7 - 6.813 | 4-5 | yes | 1.150 -> 1.379 |
| 6.814 | 60 (capped) | **no** | diverges |
| 6.82 | 60 (capped) | **no** | diverges (max\|u\| spikes to 27+) |

**Interpretation:** the discrete critical point sits between `lam=6.813` (converges cleanly, 5 iterations) and `lam=6.814` (fails outright, never converges) — pinning the level-5 discrete lam*_h to about `6.813-6.814`, within ~0.1% of the literature continuum value `lam* ~ 6.808124` for the 2-D square. Contrary to a "gradually rising iteration count" expectation, the observed behavior is a sharp cliff, not a slow slide: iteration count stays flat (3-5) essentially all the way to the fold, then Newton fails completely one step later. This matches fold-bifurcation theory — the Jacobian only becomes singular exactly at lam*, so degradation is confined to a narrow neighborhood our lam-steps (0.001-1.0) mostly stepped over.

### Explore (d) — Host-assembly vs. solve timing (level 7), and a kernel sketch

| level 7 (n_free=16,641) | per-step time |
|---|---|
| host assembly (`gp_values` + `weighted_integrals`, numpy) | 0.055, 0.040, 0.037, 0.056, 0.013 s |
| splu solve | 0.049, 0.047, 0.043, 0.043 s |
| **totals** | assembly `0.201s`, solve `0.182s` (ratio `1.03x`) |

**Interpretation:** at level 7 the two costs are roughly tied, with host assembly a hair ahead. Checked the source directly: `BratuWorkspace.gp_values`/`weighted_integrals` are pure NumPy (`np.einsum`, `np.add.at`) — no `warp` import in this file at all, unlike the linear stiffness `K`, which *is* GPU-assembled once via `assemble_csr`. So the nonlinear term is the one piece of this pipeline never ported to GPU, and it's re-run from scratch on the host every single Newton step.

**Kernel sketch (not implemented, per the prompt):** the codebase already has the right template — `make_poisson_element_matrices_var` (`assembly/operators.py`) is a "closure-hook" stiffness kernel weighted by a per-Gauss-point scalar `kq[e*nqp+q]`, used elsewhere for spatially-varying coefficients. A `make_bratu_mass_var(nbf, nqp, dim)` kernel would follow the exact same shape but with a MASS-like integrand (`N_a * N_b` instead of `dN_a . dN_b`), fused with an on-device evaluation of `u_h` at each Gauss point (interpolate nodal `u` via the basis-value table `Ntab`, already resident on device from `DeviceMesh`) and `lam * exp(u_h)` as the weight. That eliminates the host round-trip entirely for the nonlinear term: nodal `u` goes GPU -> device kernel produces `Me[e,a,b]` -> only the small triplet arrays return to host for the existing COO->CSR step (or skip even that via the device-resident scatter path `DeviceScalarPoissonAssembler` already used elsewhere for the linear case).

## Task B2 - Bratu in 3-D + Continuation

**Date:** August 6, 2026
**Script:** `tutorials/B_nonlinear/B2_bratu_3d.py`

### Observed Results

- **3-D MMS (p1, lam=2):** errors `[2.057e-02, 5.189e-03, 1.301e-03]` (levels 2/3/4), orders `1.99, 2.00`.
- **Continuation (level 3, physical Bratu):** Newton iteration counts creep `2 -> 3 -> 3 -> 3 -> 3 -> 4 -> 4 -> 6` as `lam` goes `1 -> 3 -> 5 -> 7 -> 8 -> 9 -> 9.5 -> 10`; fails outright at `lam=11`.

Matches the docstring's EXPECTED RESULTS — no bug.

### Explore (a) — Bisect the fold across levels 2/3/4

Coarse continuation to bracket the fold, then bisected to `+-0.01` in `lam` (had to make the Newton call exception-safe first: a diverging step causes `exp` overflow and an exactly-singular `splu` factorization, which crashes rather than fails cleanly on the stock `Bratu3D.newton` — wrapped it to catch both and blow-up (`max|u|>1e4`) as a clean failure).

| level | n_free | discrete lam*_h bracket | midpoint |
|---:|---:|---|---:|
| 2 | 125 | [10.3594, 10.3672] | 10.363 |
| 3 | 729 | [10.0156, 10.0234] | 10.020 |
| 4 | 4,913 | [9.9219, 9.9297] | 9.926 |

**Interpretation:** the discrete fold converges monotonically toward the literature value `lam* ~ 9.9` as the mesh refines (10.363 -> 10.020 -> 9.926), landing within `~0.3%` of literature at level 4. Coarse meshes systematically *overestimate* the critical lam — makes sense: an under-resolved mesh can't fully represent the solution's blow-up curvature, so the discrete problem tolerates a slightly higher forcing before its Jacobian goes singular.

### Explore (b) — Pseudo-arclength augmented system (derivation only, per the prompt)

Full derivation is in the companion report. Summary: treat `z=(u,lam)` jointly, parametrize the solution curve by arclength `s` instead of `lam` directly, and add one scalar constraint `N(u,lam,s) = u_dot0.(u-u0) + lam_dot0*(lam-lam0) - ds = 0` to the residual `F(u,lam)=0`. The resulting bordered `(n+1)x(n+1)` Newton system stays nonsingular exactly at the fold (where the plain `dF/du` alone is singular), because the extra tangent row/column removes the rank deficiency — this is what lets continuation walk smoothly through and past a fold that defeats simple lam-stepping.

### Explore (c) — Performance: weighted-integrals vs solve at level 4, FLOP estimate

Measured mid-continuation at `lam=8`, level 4 (`n_free=4913`, `nbf=8`, `nqp=8`, `4096` elements):

| stage | per-iteration time |
|---|---:|
| `gp_values` | ~0.0005s |
| `weighted_integrals` (both calls combined) | ~0.028s |
| `splu` solve | ~0.09s |

**Finding 1 (corrects the prompt's premise):** at level 4, `splu` solve (~0.09s) is ~3x *more* expensive than the weighted-integrals host assembly (~0.028s), not the other way around — the prompt's "weighted-integrals is now the bottleneck" claim doesn't hold at this level. (It may hold at smaller levels where Python/einsum call overhead is proportionally larger relative to an already-tiny factorization; not measured here since the prompt specified level 4.)

**Finding 2 (redundant work, independent of #1):** confirmed directly — for the physical continuation path (`f_fn=None`), `w_rhs == w_exp`, so `Bratu3D.newton`'s two `weighted_integrals(...)` calls per iteration compute *bit-for-bit identical* `(Fv, M)` pairs (`np.allclose` true, sparse-difference `nnz=0`) and each throws away half the result. An easy 2x win on this specific (very common) code path.

**FLOP estimate for the `Me` einsum `"qa,qb,eq,q,e->eab"`:** counting 3 multiplies + 1 add per `(e,a,b,q)` term plus one final per-element scale gives `n_elems*nbf^2*nqp*4 + n_elems*nbf^2 = 8,650,752` FLOPs per call at level 4. Measured time per call (`0.0143s`) implies **~0.61 GFLOP/s** achieved. A practical peak on this machine (measured via a `2000x2000` BLAS `dgemm`) is **~77 GFLOP/s** — so the einsum runs at **~0.78% of practical peak**, over 100x off.

**Interpretation:** this is a textbook memory-bound kernel. The computation is a *batch* of 4,096 tiny (`8x8`) independent matrices, not one large matmul — numpy's einsum engine can't lower a batched contraction with a non-reduced batch index (`e`) onto a single efficient BLAS GEMM call, so it falls back to a generic, unblocked, poorly-cache-reused loop. Arithmetic intensity is low (~1 FLOP per byte moved: reads `w_gp[e,q]`, writes `Me[e,a,b]`, both streamed from/to memory with almost no data reuse across elements), which is exactly the regime where a CPU is bottlenecked on memory bandwidth, not FLOPs — matching the `<1%`-of-peak result.

## Task P1 - Cost Model and Scaling

**Date:** August 8, 2026
**Script:** `tutorials/P_performance/P1_cost_model_and_scaling.py`

### Observed Results

```
level    n     nnz     mesh   constraints  assemble  factorize  backsolve
5      1089    9409   0.0007    0.0001      0.0016    0.0020     0.0001
6      4225   37249   0.0007    0.0001      0.0023    0.0099     0.0004
7     16641  148225   0.0034    0.0001      0.0073    0.0512     0.0022
8     66049  591361   0.0094    0.0003      0.0308    0.3124     0.0107
```

Matches the docstring's "Act 2" story exactly: constraints is cheap (`0.0003s` at level 8, not the historical `222s`), and the crown returns to the textbook holder — `factorize (0.31s) > assemble (0.03s) > mesh (0.01s) >> constraints (0.0003s)`. No bug.

### Explore (a) — Add p=2 columns

| level | n(p2/p1) | mesh | constraints | assemble | factorize | backsolve |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 3.880 | 1.77x | 1.98x | 2.13x | 4.84x | 7.07x |
| 6 | 3.939 | 2.10x | 1.49x | 4.64x | 6.87x | 10.68x |
| 7 | 3.969 | 1.67x | 1.44x | 5.72x | 8.83x | 8.74x |
| 8 | 3.984 | 2.83x | 1.93x | 7.52x | 16.92x | 8.03x |

**Interpretation:** neither of the prompt's predicted constants (DOF ratio `~2.25x`, kernel-work ratio `~5x`) holds cleanly for any single stage. Measured DOF ratio is `~3.98x` (matches A4's independent finding almost exactly), not `2.25x`. `assemble`'s ratio trends *upward* with level (`2.13x -> 7.52x`) rather than sitting at a fixed `~5x` — it mixes a per-element GPU kernel cost (closer to the predicted `(9/4)^2=5.06x`) with host-side CSR/triplet work that scales with `nnz`, whose own p2/p1 ratio grows with level. `factorize`'s ratio (up to `16.92x` at level 8) is the largest and most level-dependent of all, consistent with sparse factorization's superlinear dependence on a DOF ratio that's already `~4x`, not `2.25x`. `mesh` and `constraints` stay roughly flat (`~1.5-2.8x`), closest to (but not exactly) the raw DOF/element-count ratio.

### Explore (b) — 3-D crossover (factorize vs. assemble)

Clean crossover determination (median of 5 reps/level, full multi-level warmup first):

| dim | level | n | assemble | factorize | ratio |
|---|---:|---:|---:|---:|---:|
| 2-D | 4 | 289 | 0.00091 | 0.00039 | 0.43 |
| 2-D | 5 | 1,089 | 0.00110 | 0.00173 | **1.57** |
| 3-D | 2 | 125 | 0.00147 | 0.00025 | 0.17 |
| 3-D | 3 | 729 | 0.00192 | 0.00305 | **1.59** |

**Interpretation:** 2-D crosses between level 4 and 5 (`n` 289->1089); 3-D crosses between level 2 and 3 (`n` 125->729) — two full refinement levels earlier. The crossover *DOF count* is actually similar in magnitude in both dimensions (order ~300-1000) — what changes is how fast you get there: 3-D's `8x`/level element growth reaches that same magnitude two levels sooner than 2-D's `4x`/level growth, which is exactly "comes much sooner" in the practical, mesh-refinement sense the prompt means. Measured exponents confirm the underlying cause: 3-D factorize exponent `1.82 -> 2.46` (approaching/exceeding `O(n^2)`) vs. assemble's `0.63 -> 0.99` (`~O(n)`) — this measured gap is P2's entire justification.

### Explore (c) — JIT tax for a first-ever kernel variant

Built a source-distinct-but-mathematically-identical copy of the p1 2-D stiffness kernel (avoids touching the shared `~/.cache/warp` disk cache — the prompt's own "careful" flag) to force a genuine first-ever compile:

```
cold (first-ever launch, log confirms "(compiled)" not "(cached)"): 0.5207s
warm (2nd launch, same compiled kernel object):                     0.0001s
warm (3rd launch):                                                  0.0001s

JIT tax = 0.5206s  (~7,841x the warm launch cost)
```

**Interpretation:** the JIT tax for this specific p1 2-D variant is essentially the *entire* cost of one cold call — over half a second for a kernel whose warm launches complete in ~100 microseconds. This confirms the docstring's rule (ii) directly and quantifies it: never benchmark a cold kernel cache, and always warm every variant you intend to time before measuring (exactly the discipline `main()`'s own `stages(warm_level)` line already practices).

### Explore (d) — FLOP ledger for the p1 2-D stiffness kernel, vs. GPU FP64 peak

Counting operations directly in `poisson_Ke` (`make_poisson_element_matrices`, p1 2-D: `nbf=4`, `nqp=4`, `dim=2`) — 3 multiplies + 1 add per `(e,a,b,q)` term from the two `fe_dN_s` calls and their product, plus 1 multiply for `dJxW`, plus the per-element `jac`/`dscale` setup:

```
flops/element = dim + 1 + nqp*(nbf^2*(4*dim+2) + 1) = 647
```

Timed the isolated kernel launch (not the whole `assemble_csr` stage, which also includes host-side CSR triplet work) at levels 6/7/8, warmed and repeated:

| level | n_elems | total FLOPs | achieved (min) |
|---:|---:|---:|---:|
| 6 | 4,096 | 2,650,112 | 46.1 GFLOP/s |
| 7 | 16,384 | 10,600,448 | 85.5 GFLOP/s |
| 8 | 65,536 | 42,401,792 | **98.0 GFLOP/s** |

GPU FP64 peak estimate (queried, not assumed): `sm_count=22`, `arch=sm_89` (Ada Lovelace) via Warp's own device query; combined with Ada's public architecture facts (128 FP32 cores/SM, FP64 = 1/64 of FP32 on non-datacenter Ada) and the card's `2130 MHz` application boost clock (`nvidia-smi -q`): `FP32 peak = 2*22*128*2.13e9 ~= 12.0 TFLOPS`, `FP64 peak = 12.0/64 ~= 187 GFLOPS`.

**Achieved efficiency: `98.0 / 187 ~= 52%` of estimated FP64 peak** — and rising with level (46% -> 46% -> 52%, still climbing at level 8, consistent with the launch-floor rule (i) diluting smaller runs).

**Interpretation:** unlike B2's CPU einsum kernel (which hit <1% of practical peak, memory-bound), this custom Warp kernel is doing well: arithmetic intensity (excluding the small, cache-resident `dNtab`/`wtab` tables, counting only per-element-unique traffic — `h[e]` read, `Ke[e,:,:]` written) is `647 flops / 136 bytes ~= 4.76 FLOPs/byte`. Even at a generous GDDR6 bandwidth estimate for this card (order ~100-250 GB/s), the roofline ridge point (`FP64 peak / bandwidth`) sits at `~0.8-1.9` FLOPs/byte — well below the kernel's own `4.76`, so this kernel reads as **compute-bound**, not bandwidth-bound, across the whole plausible bandwidth range. Reaching roughly half of estimated FP64 peak on a small, per-thread, table-lookup-heavy kernel is a genuinely good result for a first-pass, non-hand-tuned implementation.

## Task E0a - Thinking Differentiable

**Date:** August 9, 2026
**Script:** `tutorials/E_differentiable/E0a_thinking_differentiable.py`

### Bug fix: adjoint timing block paid a hidden JIT tax

The base script's own "COST TABLE" printed `FD/adjoint ratio = 0.4x` (FD looked *faster* than the adjoint!), flatly contradicting its own EXPECTED RESULTS block (`30-80x`). Root cause: `solve_poisson` assembles via `make_poisson_element_matrices_var` (kappa-weighted), but PATH A's sensitivity step separately calls the *constant*-kappa `make_poisson_element_matrices` for the first time ever, *inside* the timed block — so the adjoint timing silently included a one-time JIT compile. Fixed by adding a warm-up launch of that kernel before `t_a0 = time.time()` (mirrors P1's own launch-floor/JIT-tax lesson, applied here). Verified: adjoint time dropped `0.216s -> 0.001s`, FD/adjoint ratio corrected to `79.9x`, squarely in the expected range. Does not affect the three-way *correctness* check (already exact before the fix).

### Observed Results (post-fix)

```
J at eval kappa          : 8.347e-03  (expect ~8.35e-03)
adj  vs FD  max rel err  : 1.56e-08   (expect < 1e-6)
tape vs FD  max rel err  : 1.56e-08   (expect < 1e-6)
adj  vs tape max rel err : 3.41e-15   (expect < 1e-9)
adjoint time              : 0.001s    (expect < 0.05)
FD time                   : 0.086s    (expect < 1.5)
FD / adj cost ratio       : 79.9x     (expect 30-80x)
```

Matches the docstring's EXPECTED RESULTS exactly, post-fix.

### Explore (a) — Scale to level 4 and level 6

| level | n_params | FD time | adjoint time | ratio |
|---:|---:|---:|---:|---:|
| 3 | 64 | 0.086s | 0.001s | 79.9x |
| 4 | 256 | 0.624s | 0.002s | 299.4x |
| 6 | 4,096 | 123.9s | 0.022s | **5,632x** |

**Interpretation:** the ratio grows roughly in proportion to `n_params` (≈1.2-1.4x per parameter fairly consistently across all three levels), exactly as the O(N) vs O(1) theory predicts — FD cost scales linearly with parameter count, adjoint cost stays essentially flat (0.001s → 0.022s, a mere 22x increase for a 64x larger parameter count, mostly from the larger linear solves, not from any per-parameter cost). At level 6, FD took over two minutes for a gradient the adjoint computed in 22 milliseconds.

### Explore (b) — Volume-integrated QoI (`J = int u dV`)

Using `diffsim.sbm.adjoint.volume_qoi`: `J = 3.077e-01` at `kappa_eval`, adj vs FD max rel err `2.225e-08` — same order of magnitude as the probe-based `J`'s `1.56e-08`.

**Interpretation:** the three-way check holds for this different (linear, not quadratic-in-`u`) QoI without any change to the adjoint machinery itself — only `dJ/du` changes (here, simply the mass row-sums `m`, since `J` is linear in `u`); the adjoint equation `A^T lam = dJ/du` and the gradient contraction formula are completely QoI-agnostic.

### Explore (c) — Non-symmetric operator: why the adjoint always needs `A^T`

Built a small nonsymmetric perturbation of the Poisson stiffness (standing in for a convection term, `||A-A^T||_max = 0.233` vs. `0.0` for the base symmetric case) to test three solve variants against the same right-hand side:

```
trans='T' (reuse forward factorization) vs. fresh-factorize A^T: max diff 4.4e-16  (agree to machine precision)
trans='T' (correct adjoint) vs. plain lu.solve (no trans, WRONG): max diff 0.256   (substantially different)
```

**Interpretation:** the adjoint equation `A^T lam = dJ/du` always needs the transpose — this was true even in the symmetric Poisson case, just invisible, because `A=A^T` there makes a plain solve and a transposed solve identical by coincidence. Once symmetry is gone (convection breaks it), using plain `lu.solve` instead of `trans='T'` gives a measurably wrong answer. The LU-reuse argument survives fully intact regardless of symmetry: `splu`'s `trans='T'` option solves with the transposed system using the *same* factored `L`/`U` (confirmed to machine precision against a fresh `A^T` factorization) — reuse was never about symmetry, it was always about `trans='T'` being a built-in, nearly-free capability of LU-based solvers.

### Explore (d) — `J = sum_i exp(kappa_i * u_i)`

Analytic derivation: `dJ/dkappa_i = u_i * exp(kappa_i * u_i)`. Tape matched this exactly (`0.000e+00` error) across positive kappa, negative kappa, and a large-magnitude stress test (`kappa*u` up to `~30`, gradients ranging `~1e-12` to `~1e6`). Pushed further to find the real failure point: at `kappa*u=2000` (>> `~709`, FP64's `exp` overflow threshold), `J` and the corresponding gradient component both become `inf` — and the tape and the hand-derived analytic formula agree exactly even there (both `inf`, not one `inf` and one `NaN` or finite-but-wrong).

**Interpretation:** contrary to what the prompt's phrasing ("try making kappa negative") might suggest, negative kappa is completely safe here — `exp` of a very negative argument just smoothly underflows toward `0`, no error. The real numerical hazard is large *positive* `kappa*u`, which overflows `exp` — and when it does, the tape's automatic differentiation propagates the resulting `inf` through the chain rule exactly the same way plain IEEE-754 arithmetic would in the hand-derived formula, rather than silently producing a wrong finite number or a `NaN`.

## Task E0b - Anatomy of a Taped Brick

**Date:** August 9, 2026
**Script:** `tutorials/E_differentiable/E0b_anatomy_of_a_taped_brick.py`

### Observed Results

```
Dot-product rel err       : 9.36e-18  (expect < 1e-12)
Factorization reuse ratio : 2.33x    (expect > 1.2x)
misfit J_init             : 8.3467e-03  (expect 5e-3 - 5e-2)
misfit J_final            : 7.1454e-05  (expect < 5e-4)
misfit drop                : 116.8x   (expect >= 100x)
param error ratio         : 0.908  (expect 0.5 - 1.5)
```

Matches the docstring's EXPECTED RESULTS exactly — no bug this time.

### Explore (a) — Add a convection term, re-run the dot-product test

Built `C[i,j] = int N_i (v.grad N_j) dV`, `v=[1,0]`, added to the Poisson stiffness. (Two scratch-script bugs found and fixed along the way — both were *my* hand-rolled numpy gradient computations forgetting the `dscale=2/h` reference-to-physical scaling that the real kernels apply via `fe_dN_s`; the tutorial's own code was not affected.)

```
Pure convection term: ||C + C^T||_max = 8.333e-02  (NOT zero -- not skew-symmetric)
Full nonsymmetric operator: ||A - A^T||_max = 0.375
Dot-product test <Av,w> vs <v,A^Tw>: relative error 4.06e-17  (still machine precision)
WRONG check <Av,w> vs <v,Aw> (using A instead of A^T): relative "error" 7.03e-03  (measurably large)
```

**Interpretation:** `C^T != -C` — the prompt's implied "yes" doesn't hold, because the domain has boundaries where `v=[1,0]` has nonzero normal component (the `x=0,1` faces). Integration by parts gives `C + C^T = int_boundary N_i N_j (v.n) dS`, a genuine boundary-flux term, not zero — confirmed to be a real, mathematically-explainable quantity (not roundoff noise, which would be ~1e-15 not ~0.08). Separately: the dot-product test itself passes at machine precision on the nonsymmetric operator, unaffected by the loss of symmetry — direct, concrete confirmation that the test validates transpose-correctness, a property fully independent of symmetry.

### Explore (b) — Level 4 + Tikhonov regularization

| config | J_init | J_final | misfit drop | param err ratio |
|---|---:|---:|---:|---:|
| level 3, no Tikhonov (baseline) | 8.3e-3 | 7.0e-5 | 118.9x | 0.908 |
| level 4, no Tikhonov | 8.9e-3 | 2.0e-4 | 44.7x | 0.945 |
| level 4, eps=0.001, ALPHA=36 (unchanged) | 2.0e-2 | 4.5e-3 | 4.6x | 0.899 |
| level 4, eps=0.1, ALPHA=36 (unchanged) | 1.16 | 169 | **diverges** | 10.8 |
| level 4, eps=1.0, ALPHA=0.5 (re-tuned down) | 11.5 | 0.063 | **181.7x** | 1.120 |

**Interpretation:** level 4 (4x more parameters, same 5 probes) drops misfit less than level 3 at a fixed step budget (44.7x vs 118.9x) — confirms "more underdetermined" makes optimization harder, as the prompt hints. Naively adding Tikhonov regularization *without* re-tuning the step size is actively dangerous: at `eps=0.1-1.0` with the original `ALPHA=36`, the added quadratic term changes the objective's curvature enough that the same step size now overshoots catastrophically (`J_final` explodes to 169-2049, parameter error gets *worse* than doing nothing). Re-tuning `ALPHA` down for the regularized problem restores stable, even strong, convergence (up to 181.7x misfit drop) — but the *parameter*-space recovery still doesn't fundamentally improve (`param err ratio` stays >= 1.0 even at strong regularization and large misfit drop). Regularization fixes the optimization's conditioning/stability; it does not supply the missing information needed to actually pin down 256 unknowns from 5 probes.

### Explore (c) — Single-blob misplaced initial guess + step-size sweep

| init | ALPHA=36 | ALPHA=10 | ALPHA=2 | ALPHA=0.5 |
|---|---|---|---|---|
| flat (baseline) | drop 116.8x, ratio 0.908 | drop 64.0x, ratio 0.955 | drop 37.5x, ratio 0.977 | drop 3.2x, ratio 0.981 |
| single-blob (misplaced) | drop 254.9x, ratio 0.849 | drop 140.2x, ratio 0.895 | drop 93.0x, ratio 0.914 | drop 7.6x, ratio 0.934 |

No divergence at any tested `ALPHA`, including `2.0`.

**Interpretation:** the misplaced single-blob guess recovers *better* than the flat guess at every step size tested (bigger misfit drop, smaller parameter error) — a single, even wrongly-placed, blob of structure is a better prior than no structure at all, since it gives the descent a head start that happens to be structurally closer to the (two-blob) truth. `ALPHA=2.0` (the prompt's suggested test) doesn't diverge for either initial guess — it's simply much slower to converge within a fixed 40-step budget (drop 37.5x vs 116.8x for flat init at the tuned `ALPHA=36`), a graceful, not catastrophic, degradation.

### Explore (d) — Nonlinear, state-dependent `kappa(u) = 1 + alpha*u^2`

Full derivation (see companion report for the complete math). Verified numerically: built a Picard-iterated forward solve, the full nonlinear Jacobian (standard kappa-weighted stiffness **plus** an extra term from `d(kappa)/du = 2*alpha*u` via the chain rule), and the adjoint using that full Jacobian, for `J = int u dV` and a scalar parameter `alpha`.

```
adjoint dJ/dalpha = -4.453117e-02
FD dJ/dalpha       = -4.453117e-02
relative error      = 2.72e-09
```

(First attempt gave a 99.6% mismatch — traced to the same missing `dscale` bug as Explore (a); the full Jacobian itself was verified independently against a directly-FD'd residual, `~1e-10` agreement per column, before trusting the adjoint result above.)

**Interpretation:** the IFT/adjoint argument holds completely unchanged in form — `(dR/du)^T lambda = dJ/du`, solved once at the *converged* state, same as the linear case and same as E0b's own Newton-loop principle (§3(B)) — the only change is that `dR/du` now contains a genuine extra term (the residual is truly nonlinear in `u` once `kappa` depends on `u`), and the forward solve needs actual iteration (Picard here; Newton would work identically, since only the *converged* `u*` and the *relation* `R(u,m)=0` matter to the adjoint, not which algorithm reached it — the same "differentiate the relation, not the algorithm" principle from E0a/E0b §3, demonstrated concretely by using Picard, a different algorithm, and it not mattering at all).

## Task E0c - The Recipe for a New PDE

**Date:** August 9, 2026
**Script:** `tutorials/E_differentiable/E0c_recipe_new_pde.py`

### Observed Results

All base-task checks passed, matching the docstring's EXPECTED RESULTS exactly (no bug): heat-chain adjoint vs FD `6.41e-07` (<1e-5); store vs recompute agreement `0.00e+00` (<1e-10); backward/forward wall ratio `1.17x` (<2.5x); logsumexp/max jump ratio `0.128` (<0.5); signal/FD-floor ratio `4.63e8` (>1e3); flat-J contrast `98.8x` (>10x); Allen-Cahn three-way `8.01e-09` (<1e-6).

### Explore (a) — Scale to level 4 / 60 steps; sweep checkpoint budget

| level/steps | full-store | backward ratio | recompute (10 ckpt) | recompute (4 ckpt) | recompute (2 ckpt) |
|---|---|---:|---:|---:|---:|
| L3/30 steps | 19.6 KiB, 0.57x | — | 1.60x fwd | 2.42x fwd | 4.72x fwd |
| L4/60 steps | 137.7 KiB, 0.74x | — | 2.34x fwd | 4.47x fwd | 7.67x fwd |

**Interpretation:** the memory-vs-compute trade-off is exactly monotonic and matches the checklist's prediction: fewer checkpoints -> less memory, more recompute cost, cleanly. At this toy scale (level 4, 137.7 KiB) nothing is remotely "uncomfortable" yet — the docstring's own scaling note (8 GB at `1e6` dofs x `1e3` steps) is the regime where this actually bites; this explore just confirms the trend direction and rate are correct in miniature.

### Explore (b) — Time-integrated `J`, re-derived seed injection

Re-derived the adjoint seed recurrence for `J = (dt/2) sum_n ||W u^n - u_obs^n||^2`: every step now injects its own direct `dJ/du^n` term (`seed^n = P^T lam^{n+1} + dJ/du^n`), not just the final step.

```
correctly re-derived seed logic: adjoint vs FD rel err = 5.99e-07  (matches base task's 6.41e-07 precision)
WRONG (reusing final-time-only seed logic on this new J): rel err vs FD = 1.000e+00
```

**Interpretation:** the correct re-derivation matches FD as precisely as the original final-time-only case. Reusing the *old* seed logic unmodified on the *new* J gives a completely uncorrelated (100% relative error) gradient — a stark, unambiguous demonstration that the seed-injection logic must be re-derived per choice of `J`, not just structurally copied.

### Explore (c) — Sweep beta in the logsumexp relaxation

| beta | jump ratio (lse/max) | bias (mean abs error vs true max) |
|---:|---:|---:|
| 1 | 0.026 | 0.958 |
| 20 | 0.062 | 0.012 |
| 40 (tutorial's default) | 0.128 | 0.004 |
| 100 | 0.322 | 0.001 |
| 500 | 0.814 | 0.0001 |

**Interpretation:** bias drops steeply from beta=1 to ~20-40 (0.96 -> 0.004), while the jump ratio only creeps up slowly over that same range (0.026 -> 0.128) — most of the smoothness-preserving benefit is obtained cheaply by beta~20-40; beyond that, bias improvement diminishes sharply while the jump ratio climbs steeply back toward the nonsmooth limit. The tutorial's own default (`beta=40`) sits right at this practical knee.

### Explore (d) — Fully implicit (Newton) Allen-Cahn, IFT adjoint at the converged state

Rebuilt the AC march as fully-implicit BDF1 (Newton-solved each step, not the semi-implicit split), and built the adjoint using the CONVERGED Newton Jacobian at each step (never unrolling the Newton iterations themselves).

```
dJ/dM      adjoint=5.427483e-05  FD=5.427483e-05  rel_err=5.06e-09
dJ/dkappa  adjoint=1.280978e-02  FD=1.280978e-02  rel_err=5.97e-10
worst rel err: 5.06e-09  (compare to the semi-implicit scheme's own 8.01e-09 three-way check)
```

**Interpretation:** matches FD to the same precision class as the base task's semi-implicit three-way check, confirming the do-not-differentiate-Newton-loops principle (E0b §3B) generalizes cleanly to this genuinely nonlinear, fully-implicit scheme — exactly the pattern the docstring says M4 uses for learned phase-field thermodynamics.

### Explore (e) — Adam fit with Tikhonov, and a genuine (not quoted) conditioning measurement

```
Adam, single IC, no Tikhonov:      M=-0.975 kappa=0.0190   (TRUE M=1.0 kappa=0.01) -- unstable, wrong sign
Adam, single IC, WITH Tikhonov:    M=0.9999 kappa=0.0138   -- pulled to a near-correct, stable answer
Adam, THREE diverse ICs, no Tikhonov: M=1.684 kappa=0.0142  -- better than unregularized single-IC, imperfect (untuned Adam LR for the joint 3-IC objective)

Observation-Jacobian conditioning (own from-scratch computation, 2 parameters):
  single IC:   singular values [0.988, 1.28e-12]   cond = 7.69e+11  (essentially singular)
  diverse ICs: singular values [2.87, 0.00787]     cond = 3.65e+02
  improvement: 2.1 BILLION x
```

**Interpretation:** the single-IC unregularized fit genuinely diverges to an unstable, wrong-sign answer, directly reproducing (at this project's own small, tractable scale) the qualitative failure mode the base task's beyond-FH story describes — Tikhonov regularization pulls it to a stable, near-correct answer instead of letting it wander, exactly as advertised ("Tikhonov as science, not hack"). The conditioning result is the strongest, cleanest finding: with only one initial condition, `(M, kappa)` are almost perfectly *aliased* (second singular value `~1e-12`, i.e. two directions in parameter space are nearly indistinguishable from the data) — adding two more, differently-shaped initial conditions collapses that near-singularity by 9 orders of magnitude (`cond` `7.69e11 -> 3.65e2`). Notably, this from-scratch `3.65e2` lands right inside the range the quoted beyond-FH literature figures report for *their* diverse-protocol conditioning (`2e2-5e2`) — a satisfying, independent quantitative echo of the same mechanism in a different (much smaller) physical system.

## Task E1 - Shape Optimization (Find the Hidden Circle)

**Date:** August 9, 2026
**Script:** `tutorials/E_differentiable/E1_shape_optimization.py`

### Observed Results

```
recovered theta = [0.52004 0.4699  0.30984]
true      theta = [0.52    0.47    0.31   ]
|error|         = [4.10e-05 9.70e-05 1.56e-04]
```

Recovers the hidden disk to high accuracy (no stated docstring EXPECTED RESULTS block/asserts to check against, but errors are small and the loop converges cleanly) — no bug.

### Explore (a) — Delete probes; find degenerate configurations

| probe count | max param error | J_final |
|---:|---:|---:|
| 9 (default) | 0.00016 | 7.4e-08 |
| 5 | 0.00049 | 6.2e-08 |
| 4 | 0.00052 | 6.4e-08 |
| 3 (ring) | 0.00039 | 1.1e-08 |
| 2 | **0.064** | 4.9e-08 |
| 1 | **0.129** | 1.9e-12 |

| 3-probe configuration | max param error | J_final |
|---|---:|---:|
| standard ring (control) | 0.00039 | 1.1e-08 |
| collinear (all on one horizontal line) | **0.082** | 4.2e-07 |
| nearly coincident (tiny cluster) | **0.118** | 2.6e-08 |

**Interpretation:** recovery stays accurate down to 3 well-spread probes (generically enough to determine 3 unknowns), but fails sharply at 2 or 1 probes — and, crucially, **count alone is not sufficient**: 3 *collinear* or *nearly-coincident* probes fail just as badly as having too few probes at all, despite nominally satisfying "3 readings for 3 unknowns." In every failure case, `J_final` stays small (the optimizer still successfully minimizes the — impoverished — objective) while the recovered geometry is measurably wrong — a direct, geometric echo of E0b's "misfit drop without parameter recovery" lesson and E0c's diverse-data lesson, now in shape-parameter space instead of a field or a scalar pair.

### Explore (b) — Conductivity `kappa` as a 4th unknown

Verified `kappa_gradient` itself directly against FD (independent of any optimization): `adjoint = -3.358522e+00`, `FD = -3.358522e+00`, relative error `2.57e-09` — exact.

Joint 4-parameter optimization (`cx, cy, r, kappa` together, kappa starting at the wrong value 1.0 vs true 1.3):

```
recovered theta = [0.52228 0.45896 0.3424], kappa = 1.23129
true      theta = [0.52    0.47    0.31  ], kappa = 1.3
|error| theta    = [0.0023  0.0110  0.0324]   |error| kappa = 0.0687
```

**Interpretation:** the gradient formula itself is exact; the joint recovery is visibly less tight than the 3-parameter-only case (radius and kappa both off by several percent) — plausibly a genuine correlation between `kappa` and `r` (both influence the overall magnitude of `u` in a similar direction), compounded by an untuned hand-rolled Adam schedule for the new scalar parameter. This cleanly separates two different questions worth keeping separate whenever a gradient-based fit underperforms: is the *gradient* correct (yes, to 9 significant figures) vs. is the *optimization* well-tuned/well-conditioned for the *joint* parameter set (a separate, harder question, not yet fully resolved here).

### Explore (c) — GridSDF voxel-level (level-set topology) optimization

Voxel-level gradient verified directly against FD at the 5 highest-sensitivity voxels (32x32 grid, 1,089 parameters): all relative errors `1e-9` to `3e-8` — exact, matching `test_gradient_gridsdf_voxels`' own methodology.

Naive unregularized Adam on all 1,089 voxel values (from a wrong-circle initial guess, targeting the same hidden-disk probe data) made slow, non-monotonic progress and then **failed**: the closest-point Newton projection became inadmissible at 11 of 64 surrogate Gauss points around iteration 44, even with gradient clipping and a generous `max_fail_frac=0.15` fallback.

**Interpretation:** the core claim — the same adjoint machinery extends to thousands of voxel parameters with zero changes beyond swapping the oracle — is confirmed and exact at the gradient level. But raw, unregularized per-voxel gradient descent is numerically fragile in practice: `GridSDF` is only `C0` (piecewise multilinear, `near_eikonal=False`), so nothing prevents individual voxel updates from locally distorting the level set into a shape with an ill-defined or non-unique closest point (the same class of admissibility failure A3's Explore (c) hit with a sampled `GridSDF`). Production level-set topology optimization routinely adds explicit regularization (periodic reinitialization to a valid signed-distance field, or a smoothness/total-variation penalty on neighboring voxels) specifically to prevent this — machinery this tutorial deliberately doesn't build (consistent with the project's own YAGNI stance elsewhere, e.g. E0c's un-built REVOLVE checkpointing).

### Performance Corner — forward vs. adjoint vs. tape vs. re-carve timing, 3 params vs. thousands

| config | n_params | carve | forward solve | **adjoint solve** | tape sweep | (adj+tape)/forward |
|---|---:|---:|---:|---:|---:|---:|
| CSG circle | 3 | 2.44ms | 1.98ms | **0.133ms** | 1.52ms | 0.84x |
| GridSDF n=16 | 289 | 38.2ms | 2.15ms | **0.120ms** | 25.0ms | 11.70x |
| GridSDF n=32 | 1,089 | 14.8ms | 1.95ms | **0.119ms** | 14.2ms | 7.33x |
| GridSDF n=48 | 2,401 | 15.0ms | 2.61ms | **0.118ms** | 13.8ms | 5.31x |

**Interpretation:** the *adjoint linear solve* itself — the theoretically `O(1)` part, and the part that would cost `O(N)` separate solves under finite differences — stays essentially flat (`0.118-0.133ms`) across a `3` to `2,401` parameter range, exactly confirming the docstring's claim: this cost depends only on mesh size (fixed here), never on parameter count. The *tape sweep* (turning the mesh-level adjoint into per-parameter gradients) is NOT free — it costs meaningfully more for GridSDF (`14-25ms`) than for the 3-parameter CSG case (`1.5ms`), since it fundamentally has to touch each of the `(n+1)^2` voxels at least once. This is not a contradiction of the "adjoint is O(1)" claim — it's the correct, complete version of it: **zero extra *solves* regardless of parameter count, but still `O(N)` bookkeeping to distribute the sensitivity back to `N` individual parameters** — a cost that is real but, per-parameter, vastly cheaper than a full linear solve (at 2,401 params, the whole adjoint+tape pipeline still finishes in under 15ms combined, versus the multi-second cost `2,401` separate finite-difference solves would require).
