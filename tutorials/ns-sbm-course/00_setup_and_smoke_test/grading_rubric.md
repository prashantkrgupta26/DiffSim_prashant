# Grading rubric — 00 Setup & smoke test

*The course rubric (`../ASSESSMENT.md`) specialised to this chapter. Weights
are the course standard; the "00 specifics" column says what to look for.*

| Component | Weight | 00 specifics — full marks |
|---|---:|---|
| **Scientific correctness & interpretation** | 30% | Reads `||div u||` as finite/bounded (NOT a bug); explains why the two engines differ yet both respect the lid trace `|u|max ≈ 1`; does not equate the pointwise divergence with a broken solver. |
| **Numerical verification** | 20% | `all_finite == true` and `|u|max ≈ 1` shown; states that `max|proj−mono|` at 20 steps is a transient, not a disagreement. |
| **Reproducibility** | 15% | `config.resolved.yaml` + `metadata.json` present; the `baseline.yaml` gate green; a second run reproduces the invariants. |
| **Software & CUDA** | 10% | Correct device/solver (`splu` for the indefinite saddle); a clean `READY` verdict; sane `exit_reason`. |
| **Failure diagnosis** | 10% | Forces `cudss` on the saddle (or a diverging `dt`) and correctly diagnoses the symptom from the diagnostics. |
| **Exploration & research bridge** | 10% | Raises `nsteps` and shows `max|proj−mono|` shrinking; one sentence on why the course keeps two engines. |
| **Communication** | 5% | The verdict line and the self-check table reported verbatim; honest about card-to-card last-digit differences. |

## Automatic zero-credit triggers (flag, don't fail silently)

- Claiming the pointwise `||div u||` must be zero for this scheme.
- Hand-copied numbers that do not match the submitted `results.json`.
- Loosening `baseline.yaml` to make the gate pass without a documented
  reason.

## Partial-credit guidance

- `READY` reported but the live smoke section not shown → half of Software &
  CUDA (versions are not the toolchain).
- The engines' `||div u||` reported but not interpreted → Communication
  credit only.
