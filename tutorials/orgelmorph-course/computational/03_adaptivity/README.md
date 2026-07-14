# Computational C3 — Octree refinement and temporal adaptivity

Resolution should follow the physics. This tutorial measures octree
refinement and temporal adaptivity — **honestly**:

- **Octree refinement + hanging-node constraints** — `build_adaptive`
  puts small elements only where the field is sharp. Measured ≈ 3.6× fewer
  nodes than a uniform mesh at the same finest $h$, and the CH brick runs
  on the hanging-node mesh. **This refines to a *static* geometric
  criterion — it is NOT solution-adaptive AMR.**
- **What true dynamic AMR needs** — estimate → mark → refine/coarsen →
  2:1 balance → rebuild → conservative transfer → BDF-history transfer →
  continue. We measure the correctness-critical piece: **conservative
  transfer** (naive injection loses ~20% of sub-cell mass; cell averaging
  is exact). Full dynamic AMR is a Phase-3 deliverable.
- **Temporal (LTE step ladder)** with **real cost accounting** —
  accepted/rejected steps, full+half solves, Newton iterations, wall time,
  and a **matched-accuracy fixed-dt sweep** (NOT the fictional
  horizon/min-dt ratio the old version reported).
- **Why variable-coefficient BDF2** — growing $\Delta t$ forces BDF2 to
  use coefficients built from the actual $(\Delta t, \Delta t_{\text{prev}})$;
  the constant-step coefficients collapse to order ≈ 1. **Both orders are
  MEASURED here** (variable ≈ 2.01; constant ≈ 0.93, via a tutorial-local
  forced-`r=1` march — no dependence on an inaccessible dev note).

**Read** the course document, Computational Chapter *"Octree refinement
and temporal adaptivity"* (start with `adaptivity.py`).

**Run:**
```bash
python run.py                 # octree + transfer + real cost + BDF2 order
```
`run.py` prints eight `PASS/FAIL` checks; compare with
[`EXPECTED.md`](EXPECTED.md).

**Regenerate the document's figures + numbers** (optional):
```bash
python gen_figures.py         # writes ../../latex/figures/c3_*.png + numbers/c3.tex
```

| file | role |
|------|------|
| `adaptivity.py` | the core: `octree_refinement`, `transfer_error`, `adaptive_cost`, `instrumented_march`, `bdf2_variable_order` — read this first |
| `run.py` | the driver you run; prints the four studies |
| `gen_figures.py` | regenerates the octree/transfer/ladder/cost figures and `numbers/c3.tex` |
| `EXPECTED.md` | reference numbers your run should reproduce |

The tutorial uses the production brick
`src/diffsim/physics/cahn_hilliard.py` and its `adaptive_march`. The
constant-coefficient BDF2 degradation is **measured locally** (a
`_const_march` that forces the coefficient ratio `r=1` on a varying
history), not cited — so the comparison is reproducible from this folder
alone.
