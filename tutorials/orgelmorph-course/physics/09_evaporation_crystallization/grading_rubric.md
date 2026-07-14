# Grading rubric — P9 Evaporation-conditioned embryo growth

*The course rubric (`../../ASSESSMENT.md`) specialised to this chapter's
headline result. Weights are the course standard; the "P9 specifics"
column says what to look for.*

| Component | Weight | P9 specifics — full marks |
|---|---:|---|
| **Scientific correctness & interpretation** | 30% | `φ* = 1 − |drive|/χ_ca` correctly derived from the homogeneous free energy; the embryo-composition-sweep crossover correctly reported *below* `φ*` with the self-enrichment (P7 crystal-bulk channel) explanation, not treated as disagreement; the runs correctly described as **implanting** an embryo, never as "nucleation." |
| **Numerical verification** | 20% | The embryo composition sweep independently brackets the grow/dissolve crossover and it is compared quantitatively (bracket range, direction relative to `φ*`) against the derived threshold. |
| **Reproducibility** | 15% | `config.resolved.yaml` + `metadata.json` present; both figures regenerated from saved data; `φ*`, the crossover bracket, and the wet/dry terminal states reproduce within `baseline.yaml` tolerance. |
| **Software & CUDA** | 10% | `MultiPhaseStepper` (M=2,K=1, film mode, r14) with cuDSS used correctly; constant-Onsager-mobility simplification named explicitly (not presented as production); honest `TERMINATION` status reported for every march, not inferred as "drying time." |
| **Failure diagnosis** | 10% | A sub-critical embryo (small r0) at a super-solubility composition is shown redissolving despite favorable composition; diagnosed via the Gibbs–Thomson radius, not miscategorized as a composition failure. |
| **Exploration & research bridge** | 10% | One exploratory question answered with a plot (the implant-time sweep locating the φ_s crossover is recommended); a paragraph connecting the validated `φ*` and the wet/dry arc to solvent or blend-ratio choices in a real cast solar cell. |
| **Communication** | 5% | Labeled axes/units; `φ*` and the measured crossover reported together with the direction of the gap explained; `TERMINATION` status quoted verbatim, not paraphrased as a physical drying time. |

## Automatic zero-credit triggers (flag, don't fail silently)

- Describing the default runs as "nucleation" rather than embryo
  implantation/growth.
- Reporting a `time_horizon` stop as a measured "drying time."
- Treating the composition-sweep crossover falling below `φ*` as a
  derivation error rather than the expected self-enrichment effect.
- Tolerances in `baseline.yaml` loosened to make the gate pass without a
  documented physical justification.

## Partial-credit guidance

- Correct `φ*` derivation but no composition-sweep validation run → cap
  Numerical verification at half.
- Right wet-dissolves/dry-grows qualitative result but no sub/
  supercritical radius control → cap Scientific correctness at half.
- A clean arc figure with no explicit constant-Onsager-mobility caveat →
  Communication credit only; the science credit is in naming the
  simplification.
