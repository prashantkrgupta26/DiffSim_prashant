# Computational C3 — Spatial and temporal adaptivity

Resolution should follow the physics. This tutorial measures both
adaptivities the simulator uses, plus the subtlety that makes the
temporal one correct:

- **Spatial (octree refinement)** — `build_adaptive` puts small elements
  only where the field is sharp (near an interface) and leaves the bulk
  coarse. Measured: 3.6× fewer nodes than a uniform mesh at the same
  finest $h$, and the CH brick runs on the hanging-node mesh.
- **Temporal (LTE step ladder)** — `adaptive_march` shrinks $\Delta t$
  during the violent onset and grows it through coarsening (measured
  ≈ 563× step savings over a fixed march).
- **Why variable-coefficient BDF2** — growing $\Delta t$ safely forces
  BDF2 to use coefficients built from the actual $(\Delta t, \Delta
  t_{\text{prev}})$; the constant-step coefficients collapse to order ≈ 1
  (measured 0.90/0.95 at the G3 retrofit). The current brick keeps order
  2 (measured 2.01).

**Read** the course document, Computational Chapter *"Spatial and
temporal adaptivity"* (start with `adaptivity.py`).

**Run:**
```bash
python run.py                 # octree + time-ladder + BDF2 order, ~1 min
```
`run.py` prints the three studies with a `PASS/FAIL` check; compare with
[`EXPECTED.md`](EXPECTED.md).

**Regenerate the document's figures + numbers** (optional):
```bash
python gen_figures.py         # writes ../../latex/figures/c3_*.png + numbers/c3.tex
```

| file | role |
|------|------|
| `adaptivity.py` | the core: `octree_refinement`, `adaptive_time_stepping`, `bdf2_variable_order` — read this first |
| `run.py` | the driver you run; prints the three studies |
| `gen_figures.py` | regenerates the octree + ladder figures and `numbers/c3.tex` |
| `EXPECTED.md` | reference numbers your run should reproduce |

The tutorial uses the production brick
`src/diffsim/physics/cahn_hilliard.py` and its `adaptive_march`; the
gates promoted are `test_ch_adaptive_dt` and
`test_ch_bdf2_variable_dt_order`. The constant-coefficient BDF2 baseline
(0.90/0.95) is cited from `docs/dev/2026-07-13-m0m5-retrofit-audit.md`
(G3), not re-run — the brick no longer exposes that degraded path.
