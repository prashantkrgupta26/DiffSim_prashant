# Grading rubric — P1 Binary Cahn–Hilliard

*The course rubric (`../../ASSESSMENT.md`) specialised to this chapter's
headline result. Weights are the course standard; the "P1 specifics"
column says what to look for.*

| Component | Weight | P1 specifics — full marks |
|---|---:|---|
| **Scientific correctness & interpretation** | 30% | Correct GL free energy → CH flow; interface width quoted as `√(κ/W)` with the cell count; the three "energy decreases" statements kept apart; does **not** claim lower energy = more accurate. |
| **Numerical verification** | 20% | Dispersion overlay: measured fastest mode within ~1 FFT shell of predicted `k*`; largest positive stepwise energy increment reported (≈ 0 → discretely energy-stable); mass drift vs solver tolerance stated. |
| **Reproducibility** | 15% | `config.resolved.yaml` + `metadata.json` present; all four figures regenerated from saved data; invariants reproduce (not the morphology). |
| **Software & CUDA** | 10% | `splu` used (not raw-CH cuDSS); `fp64`; sensible level/mode; clean `exit_reason`. |
| **Failure diagnosis** | 10% | FH driven into the box projection; `proj_dofs` and mass drift shown rising together; mechanism (accepted-step clip moves mass) explained. |
| **Exploration & research bridge** | 10% | One exploratory question answered with a plot (e.g. `√(κ/W)` width scaling, or the ensemble spread); a paragraph on what `λ*`/exponent predict for a real blend. |
| **Communication** | 5% | Labeled axes/units; coarsening exponent quoted **with** its uncertainty and fit window; honest about seed-sensitivity. |

## Automatic zero-credit triggers (flag, don't fail silently)

- A coarsening exponent quoted as a bare "1/3" with no uncertainty or fit
  window.
- Any claim that the morphology is bit-reproducible across GPUs.
- Hand-copied numbers that do not match the submitted `results.json`.
- Tolerances in `baseline.yaml` loosened to make the gate pass, without a
  documented physical justification.

## Partial-credit guidance

- Right `n` value but no uncertainty → cap Numerical verification at half.
- Correct dispersion prediction but no measured overlay → half of the
  verification weight (prediction without measurement is the thing this
  chapter is trying to cure).
- A beautiful morphology figure with no invariant reported → Communication
  credit only; the science credit is in the invariants.
