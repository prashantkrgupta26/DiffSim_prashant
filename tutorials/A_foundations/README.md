# Track A — Foundations — steady diffusion

Part of the [DiffSim curriculum](../README.md). Every chapter is a runnable,
self-contained script: *Learning outcome → Background → documented code →
Expected results (measured numbers you must reproduce) → Explore.*

| ch | script | outcome |
|---|---|---|
| A1 | `A1_mms_convergence.py` | MMS discipline; p1→order 2, p2→order 3 |
| A2 | `A2_boundary_conditions.py` | strong Dirichlet vs natural Neumann; the do-nothing condition |
| A3 | `A3_shifted_boundary.py` | immersed geometry: surrogate boundaries + the Taylor shift |
| A4 | `A4_mixed_elements.py` | mixed p1/p2 meshes, the minimum rule, where p2 pays |
| A5 | `A5_three_dimensions.py` | the same physics at k=3 — what changes (cost!) and what doesn't |
| A6 | `A6_complex_geometry.py` | carving domains: channel → sphere → STL |

Run a chapter from the repo root, e.g.:

```bash
python tutorials/A_foundations/A1_mms_convergence.py
```

Related solver project page: [`docs/projects/poisson-sbm.md`](../../docs/projects/poisson-sbm.md).
