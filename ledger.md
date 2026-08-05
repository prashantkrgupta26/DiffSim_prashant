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
