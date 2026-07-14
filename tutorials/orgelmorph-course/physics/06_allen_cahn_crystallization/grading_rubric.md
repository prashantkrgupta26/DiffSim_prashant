# Grading rubric — P6 Allen–Cahn crystallization

*The course rubric (`../../ASSESSMENT.md`) specialised to this chapter's
headline result. Weights are the course standard; the "P6 specifics"
column says what to look for.*

| Component | Weight | P6 specifics — full marks |
|---|---:|---|
| **Scientific correctness & interpretation** | 30% | Correctly states `drive = Δh(T/Tm − 1)` changes sign at `Tm`; interface velocity `v(ΔT)` reported monotone with the constant-mobility caveat (no thermal maximum claimed); critical radius `r*` shrinks with undercooling and is explained via the interface-vs-bulk balance; Avrami `n` reported with its finite-seed interpretation, **not** asserted as the ideal 2 or 3. |
| **Numerical verification** | 20% | `v(ΔT)` shown monotone increasing across both undercoolings; `r*` bracketed by a grow/shrink radius sweep at two undercoolings, ratio compared (sense + order of magnitude, not exact) to Gibbs–Thomson `1/|drive|`; Avrami fit reports `R²` and a 95% CI on `n`, with the fit window stated. |
| **Reproducibility** | 15% | `config.resolved.yaml` + `metadata.json` present; all three figures regenerated from saved data; grow/melt, `v(ΔT)`, `r*`, and Avrami invariants reproduce within `baseline.yaml` tolerance. |
| **Software & CUDA** | 10% | `MultiPhaseStepper` with cuDSS (or documented `splu` fallback) used correctly; sensible level/mode; clean `exit_reason`; accelerated Avrami parameters (T=250K, L_psi=11) labeled as pedagogical, not physical. |
| **Failure diagnosis** | 10% | A sub-critical seed (radius below `r*` at the chosen undercooling) is shown redissolving instead of growing; the mechanism (Gibbs–Thomson floor, not a solver bug) is correctly diagnosed. |
| **Exploration & research bridge** | 10% | One exploratory question answered with a plot (activated-mobility `v(ΔT)` maximum, or the `ε²` interface-width scaling, are recommended); a paragraph connecting `v`, `r*`, `n` to a real PCBM-class cast-film crystallization. |
| **Communication** | 5% | Labeled axes/units; quadrature ⟨ψ⟩ reported as primary with the thresholded area as a secondary, cut-sensitive check; Avrami `n` quoted **with** CI and fit window. |

## Automatic zero-credit triggers (flag, don't fail silently)

- Reporting the Avrami exponent as a bare "2" or "3" with no CI or fit
  window, or without acknowledging the finite pre-placed-seed effect.
- Claiming the constant-mobility `v(ΔT)` curve reproduces a real
  thermal-transport maximum.
- Describing orientation θ as an evolved field in this chapter (it is a
  frozen label; evolving θ is P7).
- Tolerances in `baseline.yaml` loosened to make the gate pass without a
  documented physical justification.

## Partial-credit guidance

- Correct `r*` bracket but no comparison to the Gibbs–Thomson prediction
  → cap Numerical verification at half.
- Right Avrami exponent value but no CI/fit-window/seeding interpretation
  → cap Scientific correctness at half.
- A beautiful grow/melt figure with no quadrature ⟨ψ⟩ numbers reported →
  Communication credit only; the science credit is in the invariants.
