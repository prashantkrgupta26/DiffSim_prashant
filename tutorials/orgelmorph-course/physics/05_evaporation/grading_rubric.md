# Grading rubric — P5 Evaporation (the drying film)

*The course rubric (`../../ASSESSMENT.md`) specialised to this chapter's
headline result. Weights are the course standard; the "P5 specifics"
column says what to look for.*

| Component | Weight | P5 specifics — full marks |
|---|---:|---|
| **Scientific correctness & interpretation** | 30% | Morphology is compared at **matched dryness** (`φ_s`), never at matched time; the demixing *degree* (contrast) is reported as dryness-set and rate-independent (agrees across rates to a stated %); the domain *size* is reported via the `Bi = k_e·h0/D_s` collapse, not a bare rate-vs-size claim; does **not** conclude "faster = finer" from a matched-time comparison. |
| **Numerical verification** | 20% | Per-solute conservation (`h(t)·∫φᵢ`) reported at machine precision with the height/solvent budget closing to the same order; the Bi-collapse checked quantitatively (spread stated, not eyeballed); wavelength reported as a real length (box-width fraction and cells `>1`), never a bare sub-1 "cells" number. |
| **Reproducibility** | 15% | `config.resolved.yaml` + `metadata.json` present; all four figures (`p5_drying`, `p5_matched`, `p5_morphology`, `p5_regime`) regenerated from saved `results.json`; invariants (conservation, Bi-collapse, contrast-vs-dryness trend) reproduce — not the exact noisy wavelength. |
| **Software & CUDA** | 10% | cuDSS used for the film stepper at the intended scale, `splu` only invoked as the documented small-run fallback; sensible `level`/mode choice; every run's `exit_reason` checked and only `phis_stop` runs called "dried". |
| **Failure diagnosis** | 10% | Re-run at level 4 (16×16) and report the per-solute drift rising from ~10⁻¹⁵ to ~10%, with the mechanism (under-resolved demixed interfaces / surface-enrichment layer) explained — not just "it got worse". |
| **Exploration & research bridge** | 10% | One exploratory question answered with evidence (a sweep or plot — e.g. widening the Biot range and checking whether the collapse breaks); a paragraph connecting the Bi-collapse to a real solvent/blend choice for casting a film. |
| **Communication** | 5% | Labeled axes/units (wavelength as a length, not bare cells); the collapse spread and contrast-rate-spread reported as percentages with the matched-`φ_s` level stated; honest about the wavelength's single-seed noise. |

## Automatic zero-credit triggers (flag, don't fail silently)

- Any morphology-vs-rate conclusion drawn from a matched-*time* (not
  matched-dryness) comparison.
- A wavelength reported as a bare "cells" number below 1 with no
  box-width-fraction cross-check.
- A run with `exit_reason: t_end` described as "dried".
- Hand-copied numbers that do not match the submitted `results.json`.
- Tolerances in `baseline.yaml` loosened to make the gate pass, without a
  documented physical justification.

## Partial-credit guidance

- Correct matched-dryness setup but no Bi-collapse check across
  different `(k_e, D_s)` pairs → cap Numerical verification at half (the
  chapter's core verification is the collapse, not just one rate pair).
- Right qualitative story ("dryness sets degree, Bi sets size") but no
  quantitative spread reported (contrast-rate-spread %, collapse spread
  %) → half credit on Scientific correctness.
- Conservation reported at level 6 only, no level-4 failure run attempted
  → Failure diagnosis credit capped at Communication-only (describing the
  mechanism without inducing it is not a diagnosis).
- A clean regime-map figure with no stated collapse spread number →
  Communication credit only; the science credit is in the quantitative
  invariant.
