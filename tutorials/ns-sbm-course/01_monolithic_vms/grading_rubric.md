# Grading rubric — 01 Monolithic VMS

*The course rubric (`../ASSESSMENT.md`) specialised to this chapter. Weights
are the course standard; the "01 specifics" column says what to look for.*

| Component | Weight | 01 specifics — full marks |
|---|---:|---|
| **Scientific correctness & interpretation** | 30% | Correct equal-order VMS weak form; PSPG identified as the term making equal-order legal (points at the C-block); `mono−Ghia` (discretization) and `proj−mono` (faithfulness) kept apart; does **not** claim `||div u||` must be zero. |
| **Numerical verification** | 20% | `max|mono−Ghia|` and `max|proj−mono|` reported with the station where the max occurs; faithfulness stated as the mesh-independent check. |
| **Reproducibility** | 15% | `config.resolved.yaml` + `metadata.json` present; centerline figure regenerated from saved data; invariants reproduce. |
| **Software & CUDA** | 10% | `splu` used (not cuDSS on the indefinite saddle); `fp64`; sensible level/mode; clean `exit_reason`. |
| **Failure diagnosis** | 10% | `--solver cudss` forced; the symptom (indefinite operator, no pivoting) diagnosed. |
| **Exploration & research bridge** | 10% | Refined to level 5 with `d_mono_ghia` shrinking shown; a paragraph on oracle-vs-scalable setting up Chapter 02. |
| **Communication** | 5% | Labeled axes/units; the two comparisons named distinctly; honest about coarse-mesh limits. |

## Automatic zero-credit triggers (flag, don't fail silently)

- Claiming the pointwise `||div u||` must be zero for this scheme.
- Reporting `mono−Ghia` as "the solver error" without distinguishing
  discretization from faithfulness.
- Hand-copied numbers that do not match the submitted `results.json`.
- Loosening `baseline.yaml` without a documented reason.

## Partial-credit guidance

- Right numbers but the two comparisons conflated → cap Scientific
  correctness at half.
- Faithfulness (`proj−mono`) omitted → half of Numerical verification (the
  same-mesh check is the point).
