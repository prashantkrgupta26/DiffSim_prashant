# Instructor companion — 00 Setup & smoke test

*Teaching notes. Student-facing material is in `README.md` and the course
document Chapter 0. Grading is in `grading_rubric.md`; hints in
`solutions/`.*

## What this chapter is really teaching

Two things, both habits the whole course rests on:

1. **"Imports succeed" ≠ "the toolchain works."** The doctor deliberately
   compiles a Warp kernel and runs one real monolithic NS step. A student
   whose `import torch` works but whose CUDA driver is stale will pass the
   version checks and fail the live smoke — which is the point.
2. **The `||div u||` misreading, pre-empted on day one.** The equal-order
   VMS scheme does *not* drive the pointwise divergence to zero. Students
   arriving from a body-fitted Taylor–Hood background expect `||div u|| →
   0` and read the finite value as a bug. Planting the correct expectation
   here saves confusion in every later chapter.

## Common student misconceptions

- **"The projection `||div u||` is 1.3 — the solver is broken."** No. The
  weak/PPE-space divergence is the controlled quantity; the pointwise one is
  finite and bounded. The two engines even show *different* pointwise
  `||div u||` legitimately.
- **"max|proj−mono| = 0.87, so the engines disagree."** At 20 steps on 8×8
  they have not reached the common steady state. Raising `nsteps` collapses
  the gap — Chapter 01–02 show it at `0.049`.
- **"cuDSS is faster, so I should force it here."** The saddle is
  *indefinite*; cuDSS does no partial pivoting and can diverge on it. `splu`
  (host, exact pivoting) is the documented small-problem auto choice. Good
  teachable failure.

## Expected runtime ranges (RTX 6000 Ada; scale by card)

| step | wall | notes |
|---|---|---|
| `doctor.py` | ~30–60 s | mostly the one-time Warp kernel compile |
| smoke `run.py` | seconds after compile | 8×8, 20 steps, both engines |

## Common CUDA / solver errors students hit

- **`NOT READY: no CUDA device`** on a laptop with no NVIDIA GPU — the 2-D
  chapters still run on `--device cpu`; only Chapter 05 and the scaling path
  need CUDA. Point them at the CPU config.
- **`cudss` forced → the saddle solve returns garbage / NaN.** Expected:
  cuDSS on the indefinite monolithic block. `--solver splu` is correct here.
- **A stale Warp cache** after a driver upgrade — clear `~/.cache/warp/`.

## Discussion prompts

- Why does the course carry *two* engines rather than picking the "best"
  one? (Oracle vs scalability; the 100M-DOF path.)
- What is the difference between the *weak* and the *pointwise* divergence,
  and which one is the right steady-state gate?
