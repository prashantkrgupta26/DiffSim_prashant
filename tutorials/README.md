# DiffSim tutorials

A pedagogical ladder. Each script is standalone, heavily commented, and runs
in seconds-to-minutes on one GPU. Read them in order — each introduces exactly
one new layer of the stack. (We assume you know basic FEM: weak forms, shape
functions, quadrature.)

| # | Script | What you learn | Runtime |
|---|---|---|---|
| 1 | `01_poisson_immersed_disk.py` | octree → SDF classification → surrogate boundary → SBM Nitsche solve → measured convergence order | ~30 s |
| 2 | `02_shape_optimization.py` | the differentiable loop: adjoint solve → shape gradient → Adam → re-carve; recovers a hidden circle from probe data | ~2 min |
| 3 | `03_lid_driven_cavity.py` | vector (u,p) Navier–Stokes with VMS stabilization, pseudo-time stepping, validation against Ghia et al. (1982) | ~30 s |

Suggested exercises are at the bottom of each script — they are real: each is
a small, well-posed extension whose answer you can check against the test
suite.

Coming with M1b completion / M1c: transient heat (write your own CEquation
brick), flow past an immersed cylinder (forces & the traction observable),
and shape optimization *in flow* (the hero problem).
