# Track E — Differentiable simulation

Part of the [DiffSim curriculum](../README.md). Every chapter is a runnable,
self-contained script: *Learning outcome → Background → documented code →
Expected results (measured numbers you must reproduce) → Explore.*

| ch | script | outcome |
|---|---|---|
| E0a | `E0a_thinking_differentiable.py` | three paths to a gradient (FD / tangent-linear / adjoint), naked `wp.Tape` toy, Poisson dJ/dκ verified to three-way 1e-9 |
| E0b | `E0b_anatomy_of_a_taped_brick.py` | open a real DiffSim kernel, do-not-differentiate list, compute-once patterns, dot-product test, probe-misfit drop >100× + the underdetermination lesson (5 probes ≠ 64 unknowns) |
| E0c | `E0c_recipe_new_pde.py` | the adjoint-readiness checklist applied end-to-end: transient heat chain, store-vs-recompute, choosing J, mini Allen-Cahn phase-field gradient |
| E1 | `E1_shape_optimization.py` | the adjoint loop: recover a hidden shape from probe data |
| E2 | `E2_genie_diffsbm.py` | GENIE INR + DiffSBM: the implicit-neural geometry path |

Run a chapter from the repo root, e.g.:

```bash
python tutorials/E_differentiable/E0a_thinking_differentiable.py
```

The normative adjoint-readiness checklist (what to tape, what to freeze,
the verification ladder, J-design rules) lives at:
[`docs/site_src/theory/adjoint_readiness_checklist.md`](../../docs/site_src/theory/adjoint_readiness_checklist.md).
Specs and reviews cite that page; E0c derives and demonstrates it.

Related solver project page: [`docs/projects/differentiable.md`](../../docs/projects/differentiable.md).

---

## Attribution note

The E0 mini-sequence (E0a–E0c) is adapted from the mathematical background
for **dolfin-adjoint / pyadjoint** developed by **Patrick E. Farrell**,
and from the following works:

- Farrell, Ham, Funke & Rognes (2013). *Automated Derivation of the
  Adjoint of High-Level Mathematical Programs.* SIAM J. Sci. Comput.
  35(4):C369–C393.
- Mitusch, Funke & Dokken (2019). *dolfin-adjoint 2018.1: automated
  adjoints for FEniCS and Firedrake.* J. Open Source Softw. 4(38):1292.

Reuse is adapted-with-attribution (example framings, the "mother problem"
and "learning reaction rates" lineage, the three-way comparison device),
not verbatim code blocks. DiffSim-native elements — the findings-4c taping
rules, the M1a/M2/M4 adjoint stack, the H4 signal-scale protocol, and the
beyond-FH gauge story — extend this foundation.
