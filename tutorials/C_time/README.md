# Track C — Time integration

Part of the [DiffSim curriculum](../README.md). Every chapter is a runnable,
self-contained script: *Learning outcome → Background → documented code →
Expected results (measured numbers you must reproduce) → Explore.*

| ch | script | outcome |
|---|---|---|
| C1 | `C1_heat_bdf.py` | transient heat; BDF1 vs BDF2 measured orders 1 and 2 |
| C2 | `C2_advection_diffusion.py` | when Galerkin fails: SUPG stabilization, p1/p2 × BDF1/BDF2 |

Run a chapter from the repo root, e.g.:

```bash
python tutorials/C_time/C1_heat_bdf.py
```

Related solver project page: [`docs/projects/adaptivity.md`](../../docs/projects/adaptivity.md).
