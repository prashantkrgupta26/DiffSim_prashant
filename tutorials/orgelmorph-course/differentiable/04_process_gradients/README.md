# Differentiable D4 — Process gradients

Material parameters (D1–D3) are what the blend *is*; **process**
parameters are what we *do* to it. The headline process control is the
temperature schedule $T(t)$ — the quench — applied while the film
crystallizes. This concept computes the gradient of a morphology
objective with respect to the *whole schedule at once*, as a **time
series** $dJ/dT_n$ (one number per step), from a single adjoint reverse
sweep, and finite-difference verifies every entry.

**Read** the course document, Chapter *"Process gradients"* (start with
`schedule.py`).

**Run:**
```bash
python run.py                 # dJ/dT_n time series, FD-verified per step
```

`run.py` prints the schedule, the crystallinity it produces, and the
step-by-step gradient with its finite-difference check; compare with
[`EXPECTED.md`](EXPECTED.md).

**Regenerate the document's figures + numbers** (optional):
```bash
python gen_figures.py         # writes ../../latex/figures/d4_schedule.png + numbers/d4.tex
```

| file | role |
|------|------|
| `schedule.py` | the core: `schedule_gradient`, `march` — read this first |
| `run.py` | the driver; prints and FD-checks the whole $dJ/dT_n$ series |
| `gen_figures.py` | regenerates `d4_schedule.png`, `numbers/d4.tex` |
| `EXPECTED.md` | reference schedule gradient your run should reproduce |

Uses the coupled crystallization adjoint
(`src/diffsim/adjoint/crystallization.py`), `CACHAdjoint.temperature_gradient`.
The evaporation-rate $k_e$ process channel is a documented frontier (it
needs the film-face flux infrastructure); the temperature schedule is the
delivered, verified process gradient.
