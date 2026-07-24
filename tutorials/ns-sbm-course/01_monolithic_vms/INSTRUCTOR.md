# Instructor companion — 01 Monolithic VMS

*Teaching notes. Student-facing material is in `README.md` and course
document Chapter 1. Grading is in `grading_rubric.md`; hints in
`solutions/`.*

## What this chapter is really teaching

The monolithic engine is a vehicle for two ideas:

1. **PSPG is what makes equal-order legal.** The nonzero pressure–pressure
   C-block ($\tau_M\,\nabla N_a\cdot\nabla N_b$) is the whole reason the
   equal-order saddle is nonsingular. Students should be able to point at
   it in `lin_ns_Ae` and say why removing it kills the pressure.
2. **Two different comparisons, kept apart.** `mono−Ghia` is a
   *discretization* comparison (coarse mesh vs a $129^2$ benchmark);
   `proj−mono` is a *faithfulness* comparison (two engines, same mesh). The
   course's central check is the second, and it is stronger because it is
   mesh-independent.

## Common student misconceptions

- **"The 16×16 mesh disagrees with Ghia by 0.06, so the solver is wrong."**
  No — that is coarse-mesh error; it shrinks under refinement. The right
  bar for *the solver* is the same-mesh faithfulness, not the benchmark.
- **"`||div u||` should be ~0."** The recurring trap. The scheme controls
  the *weak* divergence; 0.19 (mono) / 2.0 (proj) pointwise are both
  correct on this mesh.
- **"cuDSS is faster, use it here."** The saddle is indefinite; cuDSS has no
  partial pivoting. `splu` is the documented small-problem choice. Forcing
  cuDSS is the intended failure exercise.

## Expected runtime ranges (RTX 6000 Ada; scale by card)

| mode | level / steps | wall | notes |
|---|---|---|---|
| quick | 4 / 100 | ~1 min | mostly Warp compile + Python start-up |
| reference | 4 / 200 | ~2–4 min | the EXPECTED numbers |
| research | 5 / 400 | ~10–20 min | tighter to Ghia |

## Common CUDA / solver errors students hit

- **`--solver cudss` → the centerline is garbage / NaN.** Indefinite saddle,
  no pivoting. Expected; a good teachable moment (contrast with Chapter 02's
  SPD PPE, which *is* cuDSS/AMG-friendly).
- **OOM at level 6+** on a small card — the saddle has `ndof=3` per node;
  the factorization fill is larger than a scalar problem. Stay at level 4–5
  for this chapter.

## Discussion prompts

- Why is the C-block empty without PSPG, and what goes wrong then?
- `mono−Ghia` vs `proj−mono`: which one would you report to argue the
  *split* is correct, and why?
- The oracle is robust but does not scale. What, specifically, stops the
  indefinite saddle at 100M DOF? (Sets up Chapters 02 and 06.)
