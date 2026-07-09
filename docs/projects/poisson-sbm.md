# Poisson / Shifted-Boundary Method

**The foundation.** Steady diffusion on geometry that is *immersed* in a
structured octree grid rather than body-fitted. The domain boundary cuts
through cells arbitrarily; instead of remeshing to fit it, we impose boundary
conditions on the nearest grid-aligned **surrogate boundary** and correct for
the gap with a Taylor shift. This is what lets DiffSim run FEM on an INR or an
STL without ever generating a conforming mesh.

## The math

Solve −∇·(κ∇u) = f with the boundary shifted from the true surface Γ to the
surrogate Γ̃. Dirichlet data is carried across the gap by a first-order Taylor
expansion (Nitsche-consistent), Neumann data by the Eq.-21 flux with an area
correction. The penalty follows the unified law **α ≈ (1 + Pe/4)·p²**.

Full derivation: [`docs/theory/p2_band_study_for_students.md`](../theory/p2_band_study_for_students.md).
Tutorial chapter: `tutorials/A_foundations/A3_shifted_boundary.py`.

## Quickstart

```bash
# convergence ladder on an immersed sphere (dim=3, levels 4→5)
python benchmarks/poisson-sbm/band_study.py 3 4 5

# pick a clean radius and a GPU-direct solver
python benchmarks/poisson-sbm/band_study.py --radius 0.19 --solver cudss 3 4 5
```

## Validation

| Case | Observed order | Note |
|---|---|---|
| 2-D band, r = 0.19 | 2.0 | canonical clean radius |
| 3-D band, r = 0.15, L4→L7 | 1.97 / 1.94 / 2.03 | asymptotic 2nd order (Nova) |

**The dyadic-halo rule** (a real lesson): a radius that is a dyadic multiple of
`h` puts the surrogate boundary on cell faces and destroys convergence
(`r = 0.25` → L7 error 400× the clean value). Benchmark radii must avoid this.

## Differentiability

The SBM operators (surrogate extraction, Taylor shift, Nitsche penalty) are all
differentiable. `dQ/dκ` for a field diffusivity matches central finite
differences to **3e-9** (see `benchmarks/inverse-heroes/b1_dqdkappa_wip.py`) —
the correctness backbone for the inverse-design heroes.

## Learn more

- Author a brick: the fully-annotated `PoissonBrick`
  (`src/diffsim/api/example_bricks.py`) maps the weak form to code line by line —
  the template for extending DiffSim with new physics.
- Tutorials: A1 (MMS discipline), A3 (shifted boundary), A5 (3-D), A6 (carving
  geometry)
- Benchmarks: [`benchmarks/poisson-sbm/`](../../benchmarks/poisson-sbm/README.md)
- Provenance: `docs/dev/m0-deferred-findings.md`, the band-sweep verdicts in
  `docs/dev/nova-mirrors/band_sweep/`
