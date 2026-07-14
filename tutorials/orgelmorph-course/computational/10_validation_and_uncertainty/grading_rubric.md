# C10 — Grading rubric

**Deliverable.** An uncertainty budget for a material case of the student's
choice from `materials.yaml` (a *different* one from P3HT:PCBM), propagating a
parameter uncertainty and a seed ensemble through to a morphology observable,
with the dominant source identified and defended.

| criterion | points | what full marks looks like |
|-----------|--------|----------------------------|
| Six facets separated | 20 | code vs solution verification vs model validation vs parameter vs stochastic vs model–experiment, each named and (where possible) measured — not conflated |
| Material case from `materials.yaml` | 10 | χ *and its uncertainty* read from the loader, never hand-copied; provenance recorded |
| Parameter propagation | 20 | a real finite-difference sensitivity dL/dχ (noise-averaged over seeds) × the material's σ_χ |
| Stochastic variability | 10 | a seed ensemble at fixed parameters; σ_stoch from ≥3 seeds |
| Solution verification | 10 | a real mesh-refinement of the observable → a numerical uncertainty |
| The budget | 20 | variances combined in quadrature; each source's % reported; the dominant source correctly identified with the actionable conclusion |
| Honesty about the gap | 10 | model–experiment discrepancy documented as unquantified (no lab data) rather than faked; the onset-vs-coarsened trap avoided |

**Automatic fail.** Quoting a seed-sd as "the uncertainty" while a larger
parametric contribution is present and unreported — the one error the chapter
exists to prevent.
