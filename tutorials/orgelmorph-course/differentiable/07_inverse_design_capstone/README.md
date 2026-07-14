# Differentiable D7 — Capstone: inverse-designing a process

The capstone closes the loop. Everything before this computed a gradient
or read one off; here we use it to **design**. This is PDE-constrained
optimization: find a temperature schedule $T(t)$ that drives the coupled
Cahn–Hilliard × Allen–Cahn crystallization to a **target crystalline
fraction**, by minimizing $J(T)=\tfrac12(\langle\psi_N\rangle-\phi^\star)^2$
over the whole schedule with the D4 schedule gradient and a bounded
L-BFGS-B. The simulator, run backward, returns the recipe.

**Read** the course document, Chapter *"Capstone: inverse-designing a
process"* (start with `design.py`).

**Run:**
```bash
python run.py                 # design T(t) to hit a target crystalline fraction
```

`run.py` FD-verifies the design gradient, runs the optimizer, and asserts
the achieved fraction reaches the target; compare with
[`EXPECTED.md`](EXPECTED.md).

**Regenerate the document's figures + numbers** (optional):
```bash
python gen_figures.py         # writes ../../latex/figures/d7_design.png + numbers/d7.tex
```

| file | role |
|------|------|
| `design.py` | the core: `design`, `obj_and_grad`, `fd_check` — read this first |
| `run.py` | the driver; gradient self-check + design, with assertions |
| `gen_figures.py` | regenerates `d7_design.png`, `numbers/d7.tex` |
| `EXPECTED.md` | reference design your run should reproduce |

Uses the coupled crystallization adjoint's schedule gradient
(`CACHAdjoint.temperature_gradient`) inside `scipy` L-BFGS-B — the same
G5 machinery in the merged `src/diffsim/adjoint/`.
