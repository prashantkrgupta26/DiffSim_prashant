# Grading rubric — P4 Ternary Cahn–Hilliard

*The course rubric (`../../ASSESSMENT.md`) specialised to this chapter's
headline result. Weights are the course standard; the "P4 specifics"
column says what to look for.*

| Component | Weight | P4 specifics — full marks |
|---|---:|---|
| **Scientific correctness & interpretation** | 30% | The two GMM coexisting-phase endpoints `(φ₁, φ₂)` and populations are reported *and* correctly contrasted with the spinodal `det H` sign at the initial blend — the student does **not** conflate the spinodal (linear instability) with the binodal (true equilibrium coexistence, common tangent); correctly reads that the simulated endpoints sit *inside* the predicted binodal (short quench, not a discrepancy). |
| **Numerical verification** | 20% | The lever-rule residual (conserved mean vs. population-weighted phase average) is reported and small (~6×10⁻⁴ reference); admissibility (simplex residual, machine precision) and per-solute quadrature-content drift (round-off) are stated as an independent cross-check that the clustering and conservation agree. |
| **Reproducibility** | 15% | `config.resolved.yaml` + `metadata.json` present; all five figures (`p4_landscape`, `p4_spinodal`, `p4_gibbs`, `p4_fields`, `p4_nshift`) regenerated from saved `results.json`/run data; invariants (det, `χ₁₂*`, GMM means, lever residual) reproduce within `baseline.yaml` tolerance — not bit-identical morphology. |
| **Software & CUDA** | 10% | Correct solver used and justified: cuDSS as the fast primary path here (the multiphase block is well-conditioned, unlike P1's raw binary saddle), `splu` as the documented small-run fallback; clean `exit_reason`; sensible level/mode choice. |
| **Failure diagnosis** | 10% | The asymmetric blend is pushed to `N₁ = 5` or `10` (Q3) and the rising Newton iteration count / reject-ladder activity is reported and explained as the entropic log barrier weakening under the stiffer quench — not just observed. |
| **Exploration & research bridge** | 10% | One "Explore on your own" question answered with evidence (e.g. Q1: a `χ₁₂` sweep across `χ₁₂*` showing `det H` change sign and the spread only growing above it); a paragraph connecting the spinodal/binodal comparison to a real blend (e.g. mapping `P3HT_PCBM` from `materials/ternary_p4.yaml`, Q5). |
| **Communication** | 5% | Labeled Gibbs-triangle axes; endpoints reported *with* the interface-excluded sensitivity as an uncertainty, not as bare numbers; honest about which mode (quick/reference/research) produced the reported figures. |

## Automatic zero-credit triggers (flag, don't fail silently)

- Reporting the spinodal (`det H` sign / `χ₁₂*`) as if it were the
  coexisting-phase composition, or vice versa — these are two different
  quantities in this chapter and conflating them is a conceptual error,
  not a rounding one.
- Coexisting-phase endpoints obtained by a median split (or any method
  other than the GMM) presented as the chapter's result.
- Hand-copied numbers that do not match the submitted `results.json`.
- Any claim that the GMM endpoints are bit-reproducible across GPUs, or
  that the reference-mode run has "reached" the predicted binodal without
  checking the actual endpoint-vs-binodal gap.
- Tolerances in `baseline.yaml` loosened to make the gate pass, without a
  documented physical justification.

## Partial-credit guidance

- Correct GMM endpoints but no interface-excluded sensitivity reported →
  cap Scientific correctness at half.
- Lever-rule residual computed but admissibility/conservation checks
  omitted (or vice versa) → half of the verification weight — the point
  is that all three agree, not any one alone.
- `N₁` pushed to 5 or 10 but the Newton-iteration/reject-ladder evidence
  not shown → Failure diagnosis credit for attempting the perturbation
  only, not for diagnosing it.
- A correct spinodal/binodal distinction stated in prose but never
  connected to the actual `p4_landscape.png` overlay → Communication
  credit only; the science credit is in the measured comparison.
