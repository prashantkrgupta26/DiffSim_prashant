# Differentiable D5 — Crystallinity sensitivity

Device performance often hinges on **crystallinity** — how much of the
film has ordered into the crystalline phase. This concept differentiates
a crystallinity metric through the coupled Cahn–Hilliard × Allen–Cahn
crystallization rollout with respect to the thermodynamic crystallization
parameters ($\Delta h$, $T_m$, $\Delta\sigma$, $\varepsilon^2$, $L$), all
five from one adjoint sweep, each finite-difference verified.

**Read** the course document, Chapter *"Crystallinity sensitivity"*
(start with `crystallinity.py`).

**Run:**
```bash
python run.py                 # dJ/d{dh,Tm,dsig,eps2,L}, FD-verified
```

`run.py` prints the five sensitivities with their finite-difference
checks; compare with [`EXPECTED.md`](EXPECTED.md).

**Regenerate the document's figures + numbers** (optional):
```bash
python gen_figures.py         # writes ../../latex/figures/d5_*.png + numbers/d5.tex
```

| file | role |
|------|------|
| `crystallinity.py` | the core: `sensitivity`, `march` — read this first |
| `run.py` | the driver; prints and FD-checks all five sensitivities |
| `gen_figures.py` | regenerates `d5_crystallinity.png`, `d5_sensitivity.png`, `numbers/d5.tex` |
| `EXPECTED.md` | reference sensitivities your run should reproduce |

Uses the coupled crystallization adjoint
(`src/diffsim/adjoint/crystallization.py`, `CACHAdjoint.gradient`).

**On noise:** sensitivity to the *nucleation-noise amplitude* is a
stochastic derivative (of an expectation), a documented frontier — this
deterministic gate differentiates the crystallization *drift* parameters
and says so, rather than fabricate a noise gradient.
