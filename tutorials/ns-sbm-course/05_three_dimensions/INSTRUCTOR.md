# Instructor companion — 05 Three dimensions

*Teaching notes. Student-facing material is in `README.md` and course
document Chapter 5. Grading is in `grading_rubric.md`; hints in
`solutions/`.*

## What this chapter is really teaching

1. **An honest two-engine verdict.** This is where the course refuses to
   oversell. The monolithic is the working 3-D drag path (+0.381); the
   projection composition is *de-risked* (finite, axisymmetric, BDF2, PPE
   divergence machine-zero) but its long-time drag transient is unstable at
   feasible mesh. A student must learn to state that plainly, not paper over
   it.
2. **"Finite" ≠ "stable".** The projection composition never NaNs — yet its
   $C_d$ diverges monotonically over a long march. The distinction between a
   bounded-but-drifting quantity and a converged one is the transferable
   idea.
3. **The scaling contradiction, resolved.** The monolithic is the *drag*
   path today, but the *projection* is the 100M-DOF path (its SPD PPE takes
   AMG; the saddle cannot be factorized at the hero scale). Both statements
   are true; Chapter 06 is where the projection's scaling pays off even
   though its 3-D drag transient is still an open item.

## Common student misconceptions

- **"The projection is broken, drop it."** No — it is de-risked and is the
  *only* path to 100M DOF. Its 3-D long-time drag transient is a documented
  R2 item, not a dead end.
- **"$C_d = 0.38$ disagrees with the literature (~0.6)."** Confinement. The
  unit box inflates it; the bar is the M1b lock, which it reproduces.
- **"The startup negative $C_d$ is a bug."** It is the transient; the steady
  value is positive. Watch the step-by-step $C_d$ recover.

## Expected runtime ranges (RTX 6000 Ada / gpubox CPU splu)

| mode | level / steps | wall | notes |
|---|---|---|---|
| quick | 3 / 20 | ~1 min | small; smoke of the pipeline |
| reference | 4 / 60 | ~5–10 min (CPU splu ~seconds/step) | the +0.381 lock |
| research | 5 / 80 | needs cuDSS/AMGX | past the host-splu wall |

## Common CUDA / solver errors students hit

- **OOM / very slow at level 5 on splu** — the ~143k-DOF saddle exceeds the
  host direct wall. Switch to `--solver cudss` on a GPU (or `blockamgx`).
- **A negative reference $C_d$** — still in the startup transient; raise
  `max_steps`.
- **Forcing the projection *long* march** — it diverges (the R2 item); use
  the short `--pipeline` window for the invariants.

## Discussion prompts

- The projection is the scalable engine but not the 3-D drag path today. How
  do you hold both facts at once? (Chapter 06.)
- What diagnostic distinguishes "finite" from "stable" for the projection
  composition here?
- Why does the indefinite saddle stop scaling before the SPD PPE does?
