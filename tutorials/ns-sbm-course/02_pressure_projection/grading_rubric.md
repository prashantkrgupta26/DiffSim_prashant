# Grading rubric — 02 Pressure projection

*The course rubric (`../ASSESSMENT.md`) specialised to this chapter.*

| Component | Weight | 02 specifics — full marks |
|---|---:|---|
| **Scientific correctness & interpretation** | 30% | Correct Helmholtz–Leray split; the PPE identified as SPD and named as the 100M-DOF-capable operator; faithfulness (`proj−mono`) read as the validation, not the Ghia gap; does **not** claim the projection is "more accurate" than the oracle. |
| **Numerical verification** | 20% | `max|proj−mono|` reported with its station; the PPE-space solenoidality identity named as the right divergence gate (vs the finite pointwise norm). |
| **Reproducibility** | 15% | Resolved config + provenance; centerline figure regenerated; invariants reproduce. |
| **Software & CUDA** | 10% | `splu` for the predictor saddle at this size; the contrast (cuDSS/AMG *is* right for the SPD PPE) stated; clean `exit_reason`. |
| **Failure diagnosis** | 10% | The mechanism of the open-outflow instability without `consistent_projection` correctly named (not required to reproduce a blow-up on the pinned cavity). |
| **Exploration & research bridge** | 10% | The two engines' `||div u||` compared and explained; a paragraph on the SPD-PPE → 100M path. |
| **Communication** | 5% | Labeled axes; faithfulness vs benchmark named distinctly; honest that the coarse-mesh Ghia "win" is coincidental. |

## Automatic zero-credit triggers (flag, don't fail silently)

- Claiming the projection is more accurate than the monolithic oracle.
- Claiming the pointwise `||div u||` must be zero, or using it as the gate.
- Hand-copied numbers that do not match `results.json`.
- Loosening `baseline.yaml` without a documented reason.

## Partial-credit guidance

- Right numbers but SPD/AMG significance not explained → cap Scientific
  correctness at half (the *why projection exists* is the chapter).
- Pointwise divergence used as the gate → half of Numerical verification.
