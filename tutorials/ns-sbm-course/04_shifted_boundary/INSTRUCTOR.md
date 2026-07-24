# Instructor companion — 04 Shifted Boundary Method

*Teaching notes. Student-facing material is in `README.md` and course
document Chapter 4. Grading is in `grading_rubric.md`; hints in
`solutions/`.*

## What this chapter is really teaching

1. **SBM is Nitsche with the shape functions Taylor-extrapolated to the true
   boundary.** The entire novelty is $S N_a = N_a + (\nabla N_a)\cdot d$ plus
   the area correction. Nothing in the *solver* changes from Chapter 03 — the
   shift is fixture *data* ($d$, $\text{corr}$). That "data, not code"
   framing is the punchline.
2. **The anti-vacuity break is the proof.** A tutorial that only shows a good
   match cannot distinguish "the shift works" from "the shift does nothing at
   this offset." Zeroing $d$ and $\text{corr}$ and watching the match break
   against the *true* oracle is what makes the claim falsifiable.
3. **Same-mesh-with-shift is the honest comparison.** Both engines carry the
   identical shifted geometry, so the 0.3% agreement is a pure
   split-vs-oracle statement, not confounded by the geometry.

## Common student misconceptions

- **"The shift changes the solver."** No — the predictor/PPE machinery is
  identical to Chapter 03. Only the fixture's $d$/$\text{corr}$ differ.
- **"A good match proves the shift is right."** Only with the anti-vacuity
  break. Without it, a small offset could be doing nothing and still match.
- **"offset can be anything."** It must stay $<h$: the Taylor extrapolation
  is first-order and degrades as $d\to h$. At $d_{\max}/h=0.80$ we are
  already fairly aggressive.

## Expected runtime ranges (RTX 6000 Ada; scale by card)

| mode | level / steps | wall | notes |
|---|---|---|---|
| quick | 4 / 300 | ~2–4 min | projection + monolithic to steady |
| reference | 4 / 600 | ~5–10 min | the EXPECTED numbers |
| research | 5 / 600 | ~20–40 min | finer mesh; watch $d_{\max}/h$ change |

The `--zero-shift` run is a second full march (the anti-vacuity leg); budget
accordingly.

## Common CUDA / solver errors students hit

- **`AssertionError: expected a genuine SBM shift (dmax>0)`** — the offset
  was too small or the carve landed cell-aligned. Increase `offset` (keep
  $<h$).
- **`--solver cudss`** on the predictor saddle — indefinite, same failure as
  earlier chapters.

## Discussion prompts

- Why is the shift "data, not code"? What does that buy you for re-meshing?
- How would you *know* the shift is doing real work if all you had was a good
  match? (The zero-shift break.)
- What happens to the first-order Taylor accuracy as the offset approaches a
  full cell?
