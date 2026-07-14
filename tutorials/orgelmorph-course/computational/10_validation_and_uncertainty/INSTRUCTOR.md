# C10 — Instructor notes

## The one idea
Verification, validation, and uncertainty are three different questions, and
"uncertainty" is itself several. The budget is the forcing function: you cannot
build it without separating parametric, stochastic and numerical spread, and
once built it tells you where the error actually lives.

## Where students go wrong
- **"It converged, so it's right."** Solution verification (facet 2) says the
  *discretization* error is small; it says nothing about whether χ = 0.86 is the
  right χ, or whether the binary CH model matches a real ternary drying film.
  The budget shows the discretization is the *smallest* contributor here — the
  convergence study, while necessary, was not where the uncertainty was.
- **Reporting a mean ± seed-sd as "the uncertainty."** That is facet 5 alone. It
  omits the parameter uncertainty, which here is nearly 2× larger. A paper that
  quotes only the seed spread understates its true uncertainty by ~40%.
- **Comparing the coarsened observable to the onset theory.** Facet 3 must use
  the *early-time* probe (linear_probe); comparing the fully-coarsened L_area to
  λ\* gives a bogus 200% "discrepancy". This is a teachable trap — run it both
  ways and discuss.

## Talking points
- Why quadrature: independent sources add in variance, not in σ. Show that the
  combined σ (0.068) is *not* the sum of the three σ's (0.109).
- The actionable conclusion: the dominant source names the next experiment. Here
  it says "measure χ," a lab task, not "refine the mesh," a compute task.
- The order-of-magnitude interpretation: we take the material's reported spread
  (0.5) as 1σ. Discuss how a log-normal reading of "order of magnitude" would
  change the parametric band — and why the *dominance* is robust to that choice.

## Grading
See `grading_rubric.md`. The deliverable is a budget for a different material
case from `materials.yaml`, with the dominant source identified and defended.
