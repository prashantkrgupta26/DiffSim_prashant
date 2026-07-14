# Grading rubric — P2 Adaptive time stepping

*The course rubric (`../../ASSESSMENT.md`) specialised to this chapter's
headline result. Weights are the course standard; the "P2 specifics"
column says what to look for.*

| Component | Weight | P2 specifics — full marks |
|---|---:|---|
| **Scientific correctness & interpretation** | 30% | Matched-accuracy speed-up reported as *both* a step-count ratio and a solve-count ratio (step-doubling costs three solves per attempt); accuracy defined as error vs. the tight reference, never read off the free energy; the stiffness-cliff mechanism (a fixed step must resolve the quench in advance) is stated correctly. |
| **Numerical verification** | 20% | Observed order on the smooth single-mode relaxation matches design (BDF1 ≈ 1.0, BDF2 ≈ 2.0, within the `baseline.yaml` gates); the fixed-`dt` quench sweep is shown converging near order ≈ 2 up to the cliff; the reference self-check (halved `ref_dt` changes `c(T)` by < 5×10⁻³ relative L2) is reported. |
| **Reproducibility** | 15% | `config.resolved.yaml` + `metadata.json` present; all five figures (`p2_order`, `p2_convergence`, `p2_dt_newton`, `p2_morphology`, `p2_energy`) regenerated from saved data; the `baseline.yaml` gate passes (or a documented deviation is given). |
| **Software & CUDA fluency** | 10% | `splu` used (not raw-CH cuDSS on the indefinite saddle); consistent precision; sensible mode/level choice; clean `exit_reason`; rejected steps and half-solves counted, not silently dropped. |
| **Failure diagnosis** | 10% | A fixed step is deliberately pushed above the stiffness cliff (or the controller is disabled per the last "Explore on your own" question); the diagnostic that flags it (relative L2, free-energy error, or length-scale error jumping by more than an order of magnitude) is identified and the mechanism explained. |
| **Exploration & research bridge** | 10% | One exploratory question answered with evidence (e.g. the tolerance-knee sweep, or the cliff-vs-`κ` relation to `λ_max ~ M/(4κ)`); a paragraph connecting the speed-up and `dt` span to the cost of a real, longer coarsening campaign. |
| **Communication** | 5% | Labeled axes/units; the speed-up is reported for *both* steps and solves with the step-doubling caveat stated; robust physics statistics are explicitly distinguished from the pixelwise trajectory-sensitivity floor. |

## Automatic zero-credit triggers (flag, don't fail silently)

- Citing the naive worst-case bound (a fixed step pinned at the
  quench's smallest adaptive step for the whole horizon) as "the"
  matched-accuracy speed-up.
- Any claim that the coarsened morphology (the pixel field) is
  bit-reproducible across runs, seeds, or GPUs.
- Hand-copied numbers that do not match the submitted `results.json`.
- Tolerances in `baseline.yaml` loosened to make the gate pass, without
  a documented physical justification.

## Partial-credit guidance

- Correct step-count speed-up but no solve-count speed-up reported →
  cap Scientific correctness at half (the whole point of "real cost"
  is that the two numbers differ).
- Observed BDF1/BDF2 order reported but never connected to the
  fixed-`dt` quench sweep or the stiffness cliff → cap Numerical
  verification at half.
- A tolerance sweep plotted with no knee identified (or tightened
  arbitrarily far past it with no comment on the flattening) → half of
  the verification weight.
- A correct adaptive run with no reference self-check (halved `ref_dt`
  comparison) → cap Reproducibility at half.
