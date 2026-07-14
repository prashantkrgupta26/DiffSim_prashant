# Differentiable D2 — Morphology sensitivity

Now that the gradient is trustworthy (D1), we ask a physical question:
how sensitive is the **morphology** to the material parameters? This
concept differentiates a scalar morphology observable — the *demixing
amplitude* (spatial variance of the composition) — with respect to the
Flory interaction $\chi$ and the gradient penalty $\kappa$, through a
short Cahn–Hilliard rollout, and finite-difference verifies each.

**Read** the course document, Chapter *"Morphology sensitivity"* (start
with `sensitivity.py`).

**Run:**
```bash
python run.py                 # dJ/dchi, dJ/dkappa of the morphology
python run.py --steps 10      # longer rollout, more demixing
```

`run.py` prints the two sensitivities, checks them against finite
differences, and asserts the physical signs ($dJ/d\chi>0$,
$dJ/d\kappa<0$); compare with [`EXPECTED.md`](EXPECTED.md).

**Regenerate the document's figures + numbers** (optional):
```bash
python gen_figures.py         # writes ../../latex/figures/d2_*.png + numbers/d2.tex
```

| file | role |
|------|------|
| `sensitivity.py` | the core: `sensitivity`, `chi_sweep` — read this first |
| `run.py` | the driver; prints and FD-checks the sensitivities |
| `gen_figures.py` | regenerates `d2_morphology.png`, `d2_slope.png`, `numbers/d2.tex` |
| `EXPECTED.md` | reference values your run should reproduce |

Same production adjoint as D1 (`src/diffsim/adjoint/phasefield.py`).
