# Grading rubric — 04 Shifted Boundary Method

*The course rubric (`../ASSESSMENT.md`) specialised to this chapter.*

| Component | Weight | 04 specifics — full marks |
|---|---:|---|
| **Scientific correctness & interpretation** | 30% | Correct Taylor shift $S N_a = N_a + (\nabla N_a)\cdot d$, area correction, and orientation contract; "shift is data, not code" understood; $d=0$ recovers Chapter 03; the confinement caveat on absolute $C_d$ noted. |
| **Numerical verification** | 20% | Faithfulness: projection + shift matches the same-mesh-**with-shift** monolithic (rel ≈ 0.3%); $d_{\max}/h$ reported as evidence the shift is active. |
| **Reproducibility** | 15% | Resolved config + provenance; drag figure regenerated; invariants reproduce. |
| **Software & CUDA** | 10% | `splu`; the shift active in the reference run (`zero_shift: false`); clean `exit_reason`. |
| **Failure diagnosis** | 10% | `--zero-shift` run: the match against the true oracle breaks; mechanism (Taylor term + area correction removed) explained. |
| **Exploration & research bridge** | 10% | `offset` swept (kept $<h$) with $C_d$/rel-diff vs $d_{\max}/h$; a paragraph on the shift enabling 3-D bodies. |
| **Communication** | 5% | Labeled axes/units; rel-diffs quoted; honest about the first-order Taylor limit as $d\to h$. |

## Automatic zero-credit triggers (flag, don't fail silently)

- Claiming a good match proves the shift works *without* the zero-shift
  anti-vacuity break.
- Reporting a reference run with `zero_shift: true` as the headline.
- Hand-copied numbers that do not match `results.json`.
- Loosening `baseline.yaml` without a documented reason.

## Partial-credit guidance

- Good match reported but anti-vacuity not run → cap Numerical verification
  at half (the break is the proof).
- Taylor shift written but "data, not code" not grasped → cap Scientific at
  half.
