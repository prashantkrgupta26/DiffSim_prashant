# 05 — hints & selected solutions (instructor-only)

## Hints for the exploratory questions

**Q (why does cuDSS buy one level, not many?).** The monolithic saddle at
level 4 is ~5k free nodes × 4 DOF; at level 5 it is ~35k nodes → ~143k DOF,
and the *direct-factorization fill* grows super-linearly. cuDSS is a GPU
*direct* solver — it lifts the wall by the GPU's memory over the host's, i.e.
roughly one refinement level, then hits the same capacity ceiling (the
15.15M-DOF hero cannot be factorized on a GH200 at all). That is precisely
why the 100M path must be *iterative* (SPD-PPE → AMG), not direct — the
Chapter 06 argument.

**Q (finite vs stable).** The projection composition's velocity/pressure
stay finite (no NaN) over a long march, but its surrogate-traction $C_d$
drifts monotonically without settling. "Finite" means bounded at each step;
"stable" means the marched quantity converges. The diagnostic is the
$C_d(t)$ trajectory: flat (monolithic) vs monotone drift (projection long
march).

## Full solution — the projection long-march divergence

The short `--pipeline` window (4 steps) reports the de-risk invariants and is
*not* the unstable regime. To see the R2 item, march the `LeraySBMStepper`
composition for many steps and plot $C_d(t)$: it diverges monotonically while
staying finite (documented in `tests/.../test_p2r0_projection_sbm.py`). The
correct student conclusion: the composition is *de-risked* (finite,
axisymmetric, BDF2, PPE divergence machine-zero) but its long-time 3-D drag
transient is an open R2 problem — so drag is taken from the monolithic.

## Full solution — level-5 cuDSS (if a GPU is available)

Build the fixture on `--device cuda:0` and run the monolithic solve through
`solver="cudss"`. The DOF count jumps to ~143k and the per-step time is
dominated by the GPU factorization; the steady $C_d$ should stay near the
level-4 value (physical convergence, not a new number). If cuDSS OOMs, that
*is* the wall — the lesson that motivates Chapter 06.
