# Grading rubric — P8 Thermal noise and nucleation

*The course rubric (`../../ASSESSMENT.md`) specialised to this chapter's
headline result. Weights are the course standard; the "P8 specifics"
column says what to look for.*

| Component | Weight | P8 specifics — full marks |
|---|---:|---|
| **Scientific correctness & interpretation** | 30% | Correctly derives/states the discrete FDT normalization `√(2L/(dt·wJ))` and why both factors are needed; nucleation reported as an ensemble (probability, distribution, CI), never a single seed; explicitly states this is a noise-amplitude sweep, **not** a temperature sweep. |
| **Numerical verification** | 20% | Equilibrium variance shown `dt`-independent (CoV ≈0.02) and `∝1/V_cell` across meshes (CoV ≈0.11) in a clip-free stable well; zero-noise control shows no nucleation (X≈0, P=0). |
| **Reproducibility** | 15% | `config.resolved.yaml` + `metadata.json` present; all three figures regenerated from saved data; FDT CoVs and ensemble statistics (probability, X mean±sd, nuclei density) reproduce within `baseline.yaml` tolerance across a re-run with a different `noise_seed`. |
| **Software & CUDA** | 10% | BDF1 used for all noise runs (no BDF2+noise attempted); FDT well correctly run on coarse CPU/splu, nucleation ensemble at level 5 on GPU; clean `exit_reason`; clipping (saturated-ψ fraction) quantified. |
| **Failure diagnosis** | 10% | The mesh-blind noise experiment (Exercise 2: remove `1/wJ`) is run and correctly diagnosed — the variance now depends on mesh, breaking equipartition; mechanism explained. |
| **Exploration & research bridge** | 10% | One exploratory question answered with a plot (the barrier-slope-from-ensemble question recommended); a paragraph connecting the FDT-verified noise and ensemble nucleation statistics to a real undercooled-melt nucleation rate. |
| **Communication** | 5% | Labeled axes/units; ensemble quantities reported as mean ± 95% CI (never a bare number from one seed); the noise-amplitude-vs-temperature distinction stated explicitly. |

## Automatic zero-credit triggers (flag, don't fail silently)

- Reporting nucleation from a single seed with no probability/CI/
  distribution.
- Calling this chapter's `noise_psi` sweep a "temperature sweep."
- Running BDF2 with nonzero `noise_psi` (should raise; if a student
  worked around the assertion, that is itself a red flag).
- Tolerances in `baseline.yaml` loosened to make the gate pass without a
  documented physical justification.

## Partial-credit guidance

- Correct FDT `dt`-independence shown but no mesh/`1/V_cell` check → cap
  Numerical verification at half (equipartition needs both).
- Right ensemble numbers but no zero-noise negative control → cap
  Scientific correctness at half.
- A clean sweep figure with no explicit noise-amplitude-vs-temperature
  caveat → Communication credit only; the science credit is in stating
  what was and wasn't varied.
