# Rung 3 (adaptive-trajectory adjoint) — gap probe & deferral — 2026-08-07

Before building Rung 3 (differentiate the production adaptive-dt march instead of
the rung-1/2 fixed-step proxy), we measured the two gaps it would close, per
measure-then-lock. **Verdict: DEFERRED** — Rung 3 is a pure cost optimization
(0–10× by horizon), not an accuracy/fidelity improvement; revisit only if/when
gradient compute is a *measured* campaign bottleneck at long horizon.

## The framing that the probe settled

The fixed-step and adaptive-trajectory gradients are the *same* continuous
sensitivity dJ/dp in the dt→0 limit; the dt schedule is a numerical detail, not
a design parameter. So Rung 3 cannot give a "more correct" gradient — only the
same one, computed along a different (cheaper, where dt can grow) path.

## Probe (ternary, phi=(0.35,0.35), chi=(3.5,1.0,0.6), 32², bdf2, mob=0.2, kap=6e-4)

**(a) Gradient dt-convergence (fixed-step, interfacial-energy objective):**
| dt | steps | J | χ-grad rel diff vs dt/4 |
|---|---|---|---|
| 5e-4 | 40 | 1.660982e-05 | — |
| 1.25e-4 | 160 | 1.660300e-05 | **0.09%** (all three χ pairs) |

The fixed-step gradient at production dt is already dt-converged to <0.1%. An
adaptive gradient would not differ meaningfully → **no accuracy gap.**

**(b) Adaptive vs fixed step count (cost), by horizon:**
| t_end | adaptive steps | fixed@5e-4 | adaptive savings |
|---|---|---|---|
| 0.02 | 48 | 40 | 0.8× (worse) |
| 0.05 | 62 | 100 | 1.6× |
| 0.1 | 53 | 200 | 3.8× |
| 0.3 | 58 | 600 | **10.3×** |

Adaptive plateaus at ~50–60 steps (dt grows to dt_max=0.02 once coarsening
starts); fixed grows linearly. Savings are entirely a function of horizon.
Also: at this χ the interface-energy early-stop did **not** fire (ran to t_end),
so the early-stop mismatch (gap #3) is χ-dependent and absent for moderate
quenches.

## Decision

Rung 3 delivers ~0× at the short horizons (t_end ≤ 0.05) the inverse-design demo
uses, and ~10× fewer gradient steps only at long (coarsened, t_end≈0.3) horizons.
Since there is no accuracy benefit and the cost benefit is speculative until a
campaign's morphology horizon is fixed, building the adaptive adjoint now is
premature optimization. **Deferred**; the trigger to revisit is a measured
gradient-cost bottleneck in an actual long-horizon 256×128 campaign, where Rung 3
× the rung-2 cuDSS backend would cut GPU gradient cost ~10×.
