# Grading rubric — P7 Coupled Cahn–Hilliard + Allen–Cahn

*The course rubric (`../../ASSESSMENT.md`) specialised to this chapter's
headline result. Weights are the course standard; the "P7 specifics"
column says what to look for.*

| Component | Weight | P7 specifics — full marks |
|---|---:|---|
| **Scientific correctness & interpretation** | 30% | Correctly distinguishes p1 (absolute) vs r14 (increment) χ conventions; demixing correctly decomposed into crystal-bulk (dominant, ≈0.376) and χ-expulsion (smaller, ≈0.044) channels; does **not** claim the χ-contrast channel dominates or that the energy decreasing certifies physical correctness. |
| **Numerical verification** | 20% | `test_chi_limits.py` passes for both p1 and r14 pure limits against the production evaluator; the three commensurate controls (ch_only/coupled_nochi/full) scored on the same mask and metric; seed sensitivity of the full contrast reported (≈0.420 ± 0.001, 3 seeds). |
| **Reproducibility** | 15% | `config.resolved.yaml` + `metadata.json` present; both figures regenerated from saved data; contrast/channel/budget invariants reproduce within `baseline.yaml` tolerance. |
| **Software & CUDA** | 10% | `MultiPhaseStepper` with cuDSS (or documented `splu` fallback) used correctly; sensible level/mode; clean `exit_reason` for all three controls and both causality runs. |
| **Failure diagnosis** | 10% | A χ-convention mix-up (p1 absolute fed where r14 increment expected, or vice versa) is induced and correctly diagnosed via `test_chi_limits.py`'s pure-limit mismatch. |
| **Exploration & research bridge** | 10% | One exploratory question answered with a plot (χ_ca sweep vs χ-expulsion channel, or the p1→r14 conversion, are recommended); a paragraph connecting the channel decomposition to purity/network trade-offs in a real cast blend. |
| **Communication** | 5% | Labeled axes/units; contrasts reported as absolute values with channel increments (never a raw ratio against the ≈0 ch_only denominator); honest about seed sensitivity. |

## Automatic zero-credit triggers (flag, don't fail silently)

- Reporting a relative amplification computed against the `ch_only`
  denominator (divide-by-floor) instead of `coupled_nochi`.
- Confusing the p1 and r14 χ conventions without running
  `test_chi_limits.py` to check.
- Claiming the χ-expulsion channel is the dominant demixing mechanism
  when the crystal-bulk channel measures larger.
- Tolerances in `baseline.yaml` loosened to make the gate pass without a
  documented physical justification.

## Partial-credit guidance

- Correct channel decomposition but no seed-sensitivity check → cap
  Numerical verification at half.
- Right qualitative story (coupling causes demixing) but no causality
  control (frozen geometry or fixed-coupling/varied-kinetics) run → cap
  Scientific correctness at half.
- A clean demixing figure with no χ-limit verification against
  `test_chi_limits.py` → Communication credit only; the science credit
  is in the verified convention.
