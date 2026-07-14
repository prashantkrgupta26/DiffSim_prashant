# P10 — grading rubric (100 pts)

| # | criterion | pts |
|---|-----------|-----|
| 1 | **Loads through the validated loader.** No hand-copied numbers; the resolved provenance is archived (`materials.resolved.json`). | 10 |
| 2 | **Provenance audit.** Correctly classifies each parameter as measured / fitted / assumed and its uncertainty; identifies that the chosen system is representative/fitted, not measured. | 15 |
| 3 | **Reproduces a real trend on the de-confounded axis.** Morphology reported at MATCHED dryness, not at a common time; the spin-speed → drying-rate mapping is documented (order-preserving, order-of-magnitude Biot uncertainty acknowledged). | 15 |
| 4 | **Mesh convergence.** Matched-state metrics reported in mesh-independent form on ≥2 meshes; relative change quantified. | 12 |
| 5 | **Time convergence.** Matched-state metrics at ≥2 time steps; relative change quantified. | 10 |
| 6 | **Parameter-uncertainty sensitivity.** χ swept across its recorded uncertainty band; which conclusions move and which hold reported. | 13 |
| 7 | **Tolerance-checked, repeatable.** `results.json` gated against `baseline.yaml`; figures generated from saved data; run re-runs clean. | 10 |
| 8 | **Honest robustness verdict.** Explicitly separates the robust invariant (demixing degree vs dryness/χ, Bi collapse) from the scale-limited claim (wavelength ordering), and names what a quantitative reproduction would require (seed ensembles, larger box). | 15 |

**Automatic deductions:** presenting the noisy wavelength ordering as a clean
result without error bars (−10); hiding the polymer conservation residual
instead of reporting it (−5); a rung silently exiting `t_end` reported as
"dried" (−10).
