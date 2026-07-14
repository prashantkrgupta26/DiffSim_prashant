# C10 — Validation and uncertainty

**Why it matters.** "Is the answer right?" is not one question but six, and
conflating them is the most common way a computational study misleads. A code
can pass a convergence test (right equations) and still be the wrong *model* of
reality; a run can be beautifully converged and still be swamped by the fact
that a material input is known only to order-of-magnitude. Grad students who
cannot separate these ship confident wrong numbers. This chapter separates them
and closes with the deliverable that forces the separation: an **uncertainty
budget** that says which source of error actually dominates the result.

**The six questions.**
1. **Code verification** — am I solving the equations right? (order/MMS,
   conservation, energy monotonicity — a bug check against *math*.)
2. **Solution verification** — is the *discretization* error of *this* run small
   enough? (mesh refinement of the observable → a numerical uncertainty.)
3. **Model validation** — is it the right model of *reality*? (compare against an
   independent benchmark; here the linear-stability onset wavelength.)
4. **Parameter uncertainty** — how much does the observable move because a
   material input has a spread? (propagate χ's uncertainty from `materials.yaml`.)
5. **Stochastic variability** — how much does it move seed-to-seed at *fixed*
   parameters? (the C9/P8 seed ensemble.)
6. **Model–experiment discrepancy** — the residual gap once 2–5 are accounted
   (documented honestly: no lab data here; the structure is in place).

**Learning objectives.** Name and separately measure all six; propagate a
material parameter uncertainty from `materials.yaml` to a morphology observable
by finite-difference sensitivity; combine parametric, stochastic and numerical
standard deviations in quadrature into one budget; and read off the dominant
source (and therefore where to spend the next effort — a better χ, more seeds,
or a finer mesh).

**Prerequisites.** Computational C1 (code/solution verification — the MMS order
study this chapter cites), C9 (the seed ensemble + the χ sweep whose sensitivity
is reused), the `materials.yaml` schema + loader (the parameter uncertainty
source), and `diffsim.diagnostics.stochastic`.

**Expected cost.** Any CUDA GPU, 2-D, `<1 GB` device memory. The budget campaign
is ~13 real runs (a 5-seed ensemble, a 6-run sensitivity, a 3-level mesh
refinement, a linear-stability probe); ~2–3 min on an RTX 6000 Ada.

## The one material case

`materials.yaml` → `P3HT_PCBM` → `chi_polymer_fullerene = 0.86`, uncertainty
`{type: order_of_magnitude, value: 0.5}` (real OSC literature — a single "true"
χ does not exist for this blend). χ enters the binary Cahn–Hilliard free energy
`f = A[c ln c + (1-c)ln(1-c)] + B·c(1-c)` as the enthalpic `B ≡ χ` (the C9
engine). The morphology observable is `L_area`, the interfacial-area domain
length. We take the material's reported spread (`0.5`) as the 1σ parametric
uncertainty on χ, documented as such.

## Files

| file | role |
|------|------|
| `uncertainty.py` | the six facets + `build_budget()` (the deliverable) |
| `gen_figures.py` | runs the budget, renders `c10_*.png`, writes `numbers/c10.tex` |
| `config.yaml` | the immutable material case + frozen model operating point |
| `baseline.yaml` | tolerance gate on the budget (invariants, not bit-identity) |

## Run it

```bash
cd computational/10_validation_and_uncertainty
PYTHONPATH=<repo>/src python uncertainty.py     # print the budget
PYTHONPATH=<repo>/src python gen_figures.py     # figures + numbers/c10.tex + outputs/budget.json
```

## The result

The budget combines the three quantifiable spreads in quadrature. **Parameter
(χ) uncertainty dominates** — because P3HT:PCBM's χ is known only to
order-of-magnitude, the morphology's uncertainty is set by the *material input*,
not by the seed noise or the mesh. The lesson: to sharpen this prediction, first
pin down χ (a better measurement), not buy more GPU seeds. See `EXPECTED.md` for
the measured budget table, `INSTRUCTOR.md` for teaching notes, and
`grading_rubric.md` for the deliverable rubric.
