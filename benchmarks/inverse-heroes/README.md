# inverse-heroes — differentiable inverse design

The hero ladder: recover a hidden geometry edit (a GENIE INR displacement) from
sparse flow/thermal observations, by differentiating end-to-end through the
octree-SBM solve. This is DiffSim's signature capability — the adjoint runs
through assembly, the linear solve, and the shifted-boundary operators back to
the shape parameters.

| Script | Hero | Result |
|---|---|---|
| `hero_h1_sphere_steady.py` | sphere INR, steady flow probes | recovery err **1.769e-3** |
| `hero_h2_sphere_transient.py` | sphere INR, transient probes | H2 |
| `hero_h3_bunny_steady.py` | Stanford-bunny INR, steady | H3 |
| `hero_h4_bunny_transient.py` | bunny INR, transient | H4 |
| `hero_d_bunny_drag.py` | bunny drag + 25×(u_x,u_y) probes | the drag/probe QoI |
| `hero_m3_bunny_continuation.py` | bunny via epoch/h-continuation | **headline 1.23e-3** |
| `b1_dqdkappa_wip.py` | field-κ adjoint vs central FD | dQ/dκ to 3e-9 |
| `b2_closure_recovery.py` | learn a closure field in-loop | closure recovery demo |
| `b3_retrain_demo.py` | retrain a closure over de Vahl Davis | S2 demo RMSE 0.049 |

## Run

```bash
python benchmarks/inverse-heroes/hero_h1_sphere_steady.py 4 1   # level 4, 1 epoch (smoke)
python benchmarks/inverse-heroes/hero_m3_bunny_continuation.py  # the headline
```

The hero scripts cross-import each other (h3→h1, d→h1+h3, m3→d) and load the
INR checkpoints from `assets/sdf/`. Imports resolve through
`_bench_bootstrap.py` regardless of the working directory.

## The lessons (each failure converted)

Sub-cell edits are invisible below the mesh scale (edit-scale vs h decides the
signal); rank-2 force aliasing needs a Gramian check before compute; the bunny
headline came only after switching to dense probes and h-continuation at the
parameter level. The full arc is in `docs/projects/differentiable.md` and the
M1/M3 findings logs.
