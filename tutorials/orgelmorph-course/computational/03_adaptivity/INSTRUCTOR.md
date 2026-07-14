# Instructor companion — C3 Octree refinement and temporal adaptivity

*Teaching notes for the instructor. Student-facing material is in
`README.md` and the course document chapter `c3.tex`. Grading is in
`grading_rubric.md`; hints and selected solutions in `solutions/`.*

## What this chapter is really teaching

The mechanics of octree meshing and step-doubling are a vehicle for the
course's central habit: **measure the claim, report the uncertainty** —
applied here to adaptivity itself, which is unusually easy to oversell.
Four ideas do the teaching:

1. **"Adaptive" is not one thing, and precision about which kind matters.**
   Static octree refinement to a known feature (a circle) is *not*
   solution-adaptive AMR. The chapter is deliberately built so students
   see the easy half (static refinement + hanging nodes) and then are
   told, explicitly, everything dynamic AMR additionally needs
   (estimate → mark → refine/coarsen → 2:1 balance → rebuild → transfer
   state *and* BDF history → continue) before measuring the one piece
   that decides correctness.
2. **Conservation is not automatic under mesh transfer.** Injection looks
   like the obvious way to restrict a fine field to a coarse grid; it is
   not conservative. Cell averaging is. This is the mechanism, not a
   footnote — it is *why* dynamic AMR cannot use naive injection.
3. **A cost claim is only as good as its accounting.** The old
   "horizon / smallest-step" ratio the chapter explicitly disavows is the
   single most instructive wrong number in this chapter: it is not
   fabricated maliciously, it is the kind of back-of-envelope reasoning
   that *looks* rigorous. The fix is to count real work (accepted +
   rejected steps, full + half solves, Newton iterations, wall time)
   against a matched-accuracy fixed-step competitor.
4. **A time-integration scheme's order is a property of its coefficients,
   not just its formula.** BDF2 is "second order" only if the
   coefficients are rebuilt from the actual step ratio `r`; forcing `r=1`
   on a genuinely varying history silently collapses the order. This
   directly extends C1's convergence-order verification habit into the
   adaptive setting.

## Common student misconceptions & typical incorrect conclusions

- **"The octree refinement in this chapter is adaptive mesh refinement."**
  The single most important distinction in the chapter. It is *not*
  solution-adaptive AMR: the refinement criterion is a static circle
  chosen in advance, and it never responds to the evolving field. Push
  students to name the missing pieces (estimate/mark/refine-coarsen/2:1
  balance/rebuild/transfer) rather than accept "it looks adaptive because
  the mesh isn't uniform."
- **"Injection is fine because the mesh is only coarsened a little."**
  False in general and false here: on sub-coarse-cell features (droplets
  smaller than the coarse cell), naive nodal injection loses ≈20% of the
  mass, while cell averaging conserves it exactly at every feature size
  (see the radius sweep in `transfer_error_sweep`). The lesson is that
  averaging's exactness *at every size* — not just in some favorable
  regime — is the property a real AMR restriction operator needs.
- **"Adaptivity gives a huge (100s of ×) speed-up."** This chapter
  explicitly measures and rejects that framing. The old
  "horizon / smallest accepted step" ratio is a *fictional* number — it
  assumes a fixed run would need the smallest step for the entire
  horizon, which no sane fixed-step run would use. The real,
  matched-accuracy comparison gives a *modest* ≈1.85× Newton-work and
  ≈1.79× wall-time saving to `t=0.6` — real, but far smaller than the
  naive ratio, and it *grows* with the horizon because the step-doubling
  overhead (three solves per accepted step, plus rejects) is amortized
  only once the coarsening tail dominates.
- **"BDF2 is second order, full stop."** Only if the coefficients are
  rebuilt from the real step ratio `r = Δt/Δt_prev` every step. On the
  *same* alternating-`dt` sequence, the brick's variable-coefficient form
  measures order ≈2.01 while a tutorial-local `_const_march` that forces
  `r≡1` on that same varying history measures order ≈0.93 — the order
  collapses toward 1, not toward 0; it doesn't "break", it silently
  degrades one order.
- **"The dt ladder makes long horizons free."** No — the LTE controller
  and the variable-coefficient BDF2 are a matched pair: the controller
  varies `dt`, and only the variable-coefficient scheme keeps the order 2
  guarantee from C1 valid while it does. Losing either half quietly
  undermines the other.

## Expected runtime ranges (RTX 6000 Ada; scale by card)

| study | wall | notes |
|---|---|---|
| octree refinement + hanging-node CH step | seconds | tiny 2-D mesh (≤4,225 nodes) |
| conservative-transfer sweep | seconds | pure NumPy, no marching |
| adaptive cost accounting (`adaptive_cost`, `t_end=0.6`) | adaptive ladder `~60 s`; matched fixed sweep `~108 s` | measured, from `EXPECTED.md` |
| BDF2 variable/constant order (`bdf2_variable_order`) | tens of seconds | three `dt` values × two march variants + reference |

Full-chapter live runs (`python run.py`) are comparable in order of
magnitude to physics P1's quick mode (`≈75 s` wall, RTX 6000 Ada, mostly
Warp kernel compile + Python start-up). First run of a session pays that
one-time compile cost; subsequent runs in the same environment are
faster.

## Common CUDA / solver errors students hit

- **`cudss` selected and the march blows up.** Same as P1/P2: the CH
  `(c,μ)` block is indefinite; `splu` is used for a reason. Forcing
  `--solver cudss` here should visibly misbehave — a good teachable
  moment, not a bug to silently work around.
- **Octree refinement predicate never fires / refines everything.**
  Check the sign and scale in `refine_fn` (band around `|centers - 0.5|
  - radius`); a bug here either produces a mesh identical to uniform (no
  savings) or refines the whole domain (savings collapse toward 1×).
- **`step_ok = False` on the adaptive mesh.** The hanging-node constraint
  operator `T` was not applied (recall `A = T^T K T` from C2) — check
  `build_constraints` is actually wired into the `DeviceMesh` before the
  stepper is built.
- **Adaptive cost accounting reports zero rejected steps.** Not
  necessarily wrong (a smooth-enough quench can accept every step), but
  double-check the LTE tolerance is not so loose that the controller
  never actually shrinks `dt` during onset — compare against the
  `c3_ladder` figure's span.
- **BDF2 order study gives a variable-coefficient order far from 2, or a
  constant-coefficient order far from 1.** Almost always a `dt_prev`
  bookkeeping bug — check that `_var_march`'s alternating sequence
  actually exercises a fresh `r` every step (not `r=1` by accident) and
  that `_const_march`'s forced overwrite happens *before* `st.step()`,
  not after.

## Discussion prompts

- Why is conservative transfer (step (f) of dynamic AMR) about the *BDF
  history*, not just the current field `c^n`? What would go wrong if only
  `c^n` were transferred and `c^{n-1}` were left on the old mesh's
  layout?
- The "old horizon/min-dt ratio" is disavowed in this chapter as
  fictional. What made it look plausible in the first place, and what
  question should you always ask before trusting a cost-saving ratio?
- If a student tightens the LTE tolerance and the adaptive advantage over
  the matched fixed sweep shrinks, is that evidence adaptivity is "not
  worth it," or evidence about *where* the fixed-dt sweep's coarsest
  matching step happened to land? How would you design a sweep that
  answers this cleanly?
- Static refinement to a known circle worked here because the feature was
  known in advance. What information would a *solution-adaptive* marker
  need that a static predicate does not?
