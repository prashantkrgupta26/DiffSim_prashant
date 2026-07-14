# Grading rubric — C2 Boundary conditions, numerically

*The course rubric (`../../ASSESSMENT.md`) specialised to this chapter's
headline result. Weights are the course standard; the "C2 specifics"
column says what to look for.*

| Component | Weight | C2 specifics — full marks |
|---|---:|---|
| **Scientific correctness & interpretation** | 30% | Correctly identifies the mixed system's *two* boundary terms (mass flux on `mu`, wetting term on `c`); correctly classifies natural/wall-energy as mass-conserving and Dirichlet/MMS as not; does **not** claim pinning both `c` and `mu` is a physical contact (recognizes it as an MMS device). |
| **Numerical verification** | 20% | Flux balance `d/dt Int c = -Int J.n` checked directly: no-flux net flux ~0 (machine precision), Dirichlet influx decaying from early to late; penalty boundary error's `1/beta` convergence checked against the strong (row-replace) reference error. |
| **Reproducibility** | 15% | Nine `[PASS]`/`ALL CHECKS: PASS` lines reproduced; all five figures (`c2_fields`, `c2_mass`, `c2_matrix`, `c2_penalty`, `c2_flux`) regenerated from saved run data via `gen_figures.py`; qualitative contrasts (conserves/does-not, symmetric/asymmetric) reproduce even though field values are seed-dependent. |
| **Software & CUDA fluency** | 10% | `splu` used on the `(c,mu)` saddle (not raw-CH cuDSS); correct use of `dirichlet=`, `gc_fn`/`gm_fn` on the production brick; sensible mesh level; clean run with no swallowed exceptions. |
| **Failure diagnosis** | 10% | Penalty `beta` pushed past `1e6`; the student reports where the boundary error stops improving / the solve degrades and explains the exactness-vs-conditioning trade (growing diagonal dominance swamping the O(1) stiffness entries in double precision). |
| **Exploration & research bridge** | 10% | One exploratory question answered with evidence (the `g=0`/`c=0` mass-drift IC-coincidence and the mixed L/R case are the recommended ones); a paragraph connecting the taxonomy here to a real device boundary (sealed box vs. composition-controlled contact vs. wetting substrate, and the wall-energy BC of Physics P3). |
| **Communication** | 5% | Reports mass drift and edge composition with units/context for *both* treatments side by side; states which contrasts are seed-dependent (field values) vs. which are not (conservation/pinning qualitative behavior). |

## Automatic zero-credit triggers (flag, don't fail silently)

- Claiming a Dirichlet or MMS boundary condition conserves mass in
  general (rather than reporting the measured drift and explaining why
  it is a reservoir).
- Claiming pinning both `c` and `mu` models a physical contact (rather
  than identifying it as an MMS device, per Chapter C1).
- Reading the `c=0.0` row of the BC test matrix as evidence that
  Dirichlet "can conserve mass" without noting it is an IC coincidence
  (mean composition near zero), not a boundary-condition property.
- Hand-copied numbers that do not match the submitted run's printed
  `[PASS]`/`[FAIL]` table or `numbers/c2.tex`.

## Partial-credit guidance

- Correctly reports no-flux conserves and Dirichlet does not, but skips
  the flux-balance overlay (`dm/dt` vs. the mass-history derivative) →
  cap Numerical verification at half.
- Reports the penalty error decreasing with `beta` but never pushes past
  `1e6` to find the conditioning crossover → half credit on Failure
  diagnosis.
- Correctly distinguishes row-replacement (asymmetric) from symmetric
  elimination (SPD) but does not connect this to why a Jacobian assembly
  should use the symmetric variant → Communication credit only, not full
  Scientific correctness.
- A complete BC test matrix reproduced with no discussion of *why*
  `c=0.0` looks anomalously well-conserved → Reproducibility credit only;
  the interpretation credit requires naming the IC coincidence.
