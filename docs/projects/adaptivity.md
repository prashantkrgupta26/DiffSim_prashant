# Adaptivity — Spatial & Temporal

**Refine where and when it matters.** DiffSim adapts the octree in space
(interface-band refinement with per-epoch re-carve and exact state transfer) and
the time step in space-time (BDF1/BDF2 with LTE-controlled dt). Both are built to
be *conservative* and *differentiable* — the adjoint runs cleanly across a
re-carve.

## Spatial adaptivity

Mark cells by a feature indicator (e.g. |∇φ| across a phase-field interface),
re-carve the octree, and transfer the solution through explicit operators:

- **shared nodes** → identity through the constraints,
- **interior nodes** → FE interpolation,
- **fresh-strip nodes** → nearest, O(h).

Nested refinement is **exact-conservative by construction** (the fine space
contains the coarse one) — measured **zero mass drift** across a re-carve. L2
projection is needed only for *coarsening*. The adjoint across the re-carve
matches to 1e-7, so inverse problems survive adaptivity.

## Temporal adaptivity

BDF1/BDF2 with a **relative-L2 local-truncation-error** controller. The lesson
recorded here matters: a max-norm LTE controller collapsed the step and took
16,580 steps / 54 min; switching to the relative-L2 measure gave **178× dt
growth** and finished in 6 minutes on the same problem.

For evaporation, the solvent flux is an **external clock invisible to the LTE
controller**, so the stepper carries an explicit cap
`dt ≤ tol·h_surf/(k_e·Δφ)` (Baskar's insight) — the plateau-then-growth dt
signature is confirmed in the film marches.

## Quickstart

```bash
python benchmarks/phase-field/adaptive_ch_opener.py    # re-mesh + transfer, zero drift
```

## Validation

| Property | Result |
|---|---|
| Mass drift across re-carve | 0 (nested exact-conservative transfer) |
| Adjoint across re-carve | 1e-7 |
| Temporal dt growth (relative-L2 LTE) | 178× vs fixed dt |

## Learn more

- Tutorials: C-track (time integration), F-track (adaptivity in phase field)
- Code: `src/diffsim/adaptivity/transfer.py`, `continuation.py`
- Provenance: `docs/dev/m3-*` findings, `docs/dev/m4-milestone-report.md`
