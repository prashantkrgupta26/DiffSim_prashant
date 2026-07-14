# Grading rubric — P3 Substrate / surface energy and boundary conditions

*The course rubric (`../../ASSESSMENT.md`) specialised to this chapter's
headline result. Weights are the course standard; the "P3 specifics"
column says what to look for.*

| Component | Weight | P3 specifics — full marks |
|---|---:|---|
| **Scientific correctness & interpretation** | 30% | Substrate `φ` correctly tracks `φ* = −g/(2h)` for attracting/repelling/opposing/confined/demixing cases; enrichment `Δφ` reported with sign and magnitude, not just "it enriched"; correctly states the wall condition never exchanges mass (it is a condition on `φ`, not on the mass flux) and contrasts this with a Dirichlet condition, which does. |
| **Numerical verification** | 20% | The quadrature mass `∫φ dV` tracked over the run for every case, drift `< 10⁻¹³` (not a nodal-mean proxy); the boundary-layer `δ(κ)` log–log fit reported with its slope *and* how it compares to the predicted ½; `projected_dofs` reported for the interior-`φ*` cases (expected 0). |
| **Reproducibility** | 15% | `config.resolved.yaml` + `metadata.json` present; all four figures (`p3_fields`, `p3_profiles`, `p3_energy`, `p3_bdlayer`) regenerated from saved `results.json`/`history.npz`; invariants (substrate `φ`, enrichment sign, mass drift, `δ` slope) reproduce within the stated tolerances, not bit-identical morphology. |
| **Software & CUDA fluency** | 10% | `splu` used (not raw-CH cuDSS on the indefinite saddle, same reasoning as P1); sensible mode/mesh choice for the question asked; clean `exit_reason`; boundary-layer sweep run on a stable sub-spinodal bulk, not the demixing case. |
| **Failure diagnosis** | 10% | A bounds-violating initial condition (raised fluctuation amplitude) is used to fire the projection honestly; `projected_dofs > 0` and the largest correction are reported, *and* the converged-state quadrature mass is shown still holding — the student explains this is a safeguard on iterates, not a conservation failure; a linear wall (`h=0`) is identified as the case that would break the equilibrium itself. |
| **Exploration & research bridge** | 10% | One exploratory question answered with evidence — a `φ*` sweep confirming `projected_dofs = 0` up to a stated bound, the opposing-wall top/bottom inversion check, or a 3-more-κ boundary-layer extension; a paragraph connecting `Δφ`/`δ` to which component reaches a real device's substrate vs top electrode. |
| **Communication** | 5% | Labeled axes/units on all four figures; the `δ ~ κ^s` slope quoted with its fit method (1/e decay vs exponential fit) and stated to be measured on a sub-spinodal bulk; honest about third-significant-figure card-to-card variation. |

## Automatic zero-credit triggers (flag, don't fail silently)

- Reporting mass conservation from a nodal-mean `φ` rather than the
  quadrature integral `∫φ dV` (this chapter explicitly documents the
  nodal mean as a spurious ~10⁻³ artifact near the wall).
- Claiming the wall energy "conserves mass because nothing crosses the
  boundary visually" without citing the actual quadrature drift number.
- Claiming a Dirichlet BC also conserves mass, or treating it as
  interchangeable with the wall-energy condition used in this chapter.
- Tolerances in `baseline.yaml` loosened to make the gate pass without a
  documented physical justification.
- Hand-copied numbers that do not match the submitted `results.json`.

## Partial-credit guidance

- Correct enrichment sign/trend but no quadrature mass-drift number
  reported → cap Numerical verification at half.
- Boundary-layer slope reported as a bare "0.5" with no fit or `κ` range
  stated → half of the verification weight for that item.
- Projection failure induced but the converged-state mass not checked
  (student stops at "projected_dofs > 0" without showing the mass still
  holds) → cap Failure diagnosis at half.
- A correct qualitative story (attract enriches, deplete depletes,
  neutral flat) with no numeric `Δφ` or `φ*` comparison → Communication
  credit only; the science credit is in the numbers.
