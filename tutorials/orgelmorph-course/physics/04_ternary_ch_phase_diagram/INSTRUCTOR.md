# Instructor companion — P4 Ternary Cahn–Hilliard

*Teaching notes for the instructor. Student-facing material is in
`README.md` and the course document chapter `p4.tex`. Grading is in
`grading_rubric.md`; hints and selected solutions in `solutions/`.*

## What this chapter is really teaching

The mechanics of the Gibbs triangle are a vehicle for three ideas that
generalize P1's binary lesson to a genuinely two-field problem:

1. **Clustering, not eyeballing, reads off a phase.** The composition
   cloud from a phase-separated run is not two clean deltas; a naive
   median split forces a 50/50 partition and lets interfacial nodes
   contaminate both means. The 2-component Gaussian mixture (EM) gives a
   mean, a covariance, *and* a population per phase, and the
   interface-excluded refit turns "trust me" into a measured sensitivity
   (≈0.13 here).
2. **Spinodal and binodal are different curves answering different
   questions.** The spinodal (`det H = 0`) is a *linear-instability*
   boundary at a *given* composition — it says whether noise grows or
   decays there. The binodal (common tangent, pinned by the lever rule)
   is the *true equilibrium coexistence* pair. A quench deep inside the
   spinodal does not imply the simulation has reached the binodal
   endpoints — a short run's GMM means sit *inside* the predicted
   binodal, not on it, which is exactly what `p4_landscape.png` shows.
3. **Three independent checks must jointly agree.** Admissibility (the
   Gibbs simplex, exact here because the solvent is eliminated
   algebraically), per-solute quadrature-content conservation, and the
   lever rule (conserved mean = population-weighted phase average) are
   three separate diagnostics computed by different code paths; when they
   agree it is a real cross-check, not a tautology.

## Common student misconceptions & typical incorrect conclusions

- **"A median split of `φ₁` gives the same phases as the GMM."** It
  doesn't — the median forces an exact 50/50 partition and lets
  high-`|∇φ|` interfacial nodes pull both means toward the middle. The
  chapter's whole point in "Reading the two phases" is that clustering
  removes this bias; ask students to compute both and compare.
- **"The spinodal and the binodal are the same curve."** The spinodal is
  `det H = 0` (linear instability at *this* composition); the binodal is
  the common-tangent equilibrium pair, always outside/enclosing the
  spinodal region. A student who reports "the phases match the spinodal"
  has not distinguished the two — the correct comparison is GMM endpoints
  vs. the *binodal* prediction (`predict_binodal`), not the spinodal
  curve.
- **"One run's GMM means are the converged coexisting compositions."**
  Check the interface-excluded sensitivity (should be small, ~0.13 here)
  *and* whether the endpoints have actually reached the predicted binodal
  — in the reference run they have not (short quench), which is a correct
  and expected reading, not a bug.
- **"`det H < 0` alone tells you where the two phases end up."** No — the
  Hessian only tests local (in)stability at a point; the actual
  coexistence composition requires solving the separate common-tangent
  system (`predict_binodal`), pinned by the lever rule.
- **"Unequal `N_i` shouldn't matter, it's just an entropy correction."**
  It moves the spinodal substantially: raising `N₁` from 1 to 2 drops
  `χ₁₂*` from 2.84 to 1.98 and deepens `det H` — the entropic `1/N_i`
  terms in the Hessian are not a small correction here.
- **"cuDSS always fails on Cahn–Hilliard-like systems, so avoid it here
  too."** That is P1/C4's lesson for the *raw* binary `(c, μ)` saddle,
  which is genuinely indefinite. The coupled multiphase block here is
  well-conditioned (Flory–Huggins bulk, `f'' ≥ 4A > 0`), so cuDSS is the
  *primary* fast path, with `splu` as the documented fallback for small
  runs without cuDSS — the opposite of P1's default. Students who
  over-generalize P1's "never cuDSS on CH" rule will unnecessarily force
  `splu` here.

## Expected runtime ranges (RTX 6000 Ada; scale by card)

| mode | mesh / horizon | wall | notes |
|---|---|---|---|
| quick | 33×33 (level 5), `t_end 0.3` | comparable to P1's **≈75 s** measured (no separate P4 quick-mode measurement recorded) | target < 2 min per the cost spec |
| reference | 65×65 (level 6), `t_end 0.6` | a few minutes, target 5–30 min | the `baseline.yaml`/`EXPECTED.md` numbers; includes the small asymmetric-N companion run (level 4, `t_end 0.1`) |
| research | 129×129 (level 7), `t_end 1.0` | tens of minutes (scales with mesh) | a single finer/longer run — no fixed sweep or ensemble count is specified in `configs/p4.yaml` |

First run of a session pays a one-time Warp kernel-compile cost (as in
every other chapter); subsequent runs are faster.

## Common CUDA / solver errors students hit

- **Forcing `--solver splu` "to be safe" (over-generalizing P1).** Not
  wrong, but slower and unnecessary — the multiphase block here is
  well-conditioned for cuDSS (see misconceptions above). Check the
  resolved config's `solver` field before assuming a problem.
- **`out of memory` at research level (129×129) on a small card.** Still
  a small 2-D problem, so OOM at this level usually means a much larger
  request than intended, or a forgotten `--mode` override — check
  `level` in `config.resolved.yaml`.
- **`nan`/`inf` in the log terms.** Almost always a `φ_i` (or the
  eliminated `φ_s = 1 - φ_1 - φ_2`) landing outside `(0, 1)`, e.g. from an
  initial blend or perturbation amplitude that is too large relative to
  `(1 - φ_1 - φ_2)` at the corner of the simplex.
- **Newton iteration count climbing / reject-ladder firing at `N₁ = 5` or
  `10`.** Expected, not a bug (Q3): a larger `N₁` weakens the entropic log
  barrier and stiffens the quench; this is precisely the failure mode
  item 6 of the required deliverable asks students to induce and explain.

## Discussion prompts

- Why must the spinodal and the binodal be *different* curves in
  general, and when (if ever) would they coincide?
- The simulated endpoints sit inside the predicted binodal rather than on
  it — is that a failure of the theory, the simulation, or neither? What
  would change your answer?
- If a student's interface-excluded sensitivity were large (say > 0.3),
  what would that tell you about the clustering, independent of whether
  the mean endpoint values "look right"?
