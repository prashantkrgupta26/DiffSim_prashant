# Finding: two-path parity gates vs GPU run-to-run nondeterminism

Date: 2026-07-09 (the release-gate full-suite runs). Status: measured,
fixed, rule extracted.

## What happened

The first-ever back-to-back FULL suite runs (351 tests) produced
different failure sets: run 1 failed {inr gates45, ns_shape drag, wodo
device parity}; run 2 (after the ns_shape fix) failed {wodo device
parity} only; targeted pairings then flipped verdicts on identical
commands (fused parity: fail standalone, pass in the pairing where it
had just failed). The order-dependence hypothesis was DISPROVED by that
inversion; the phenomenon is run-to-run nondeterminism.

## Measured rates and magnitudes (standalone, A100, warp 1.14)

| Gate | Flake rate | Failure magnitude | Passing margin |
|---|---|---|---|
| fused-Krylov vs splu trajectory (velocity tol 1e-7) | 4/10 | 1.9987e-7 (2.0x tol) | typical 2.3e-8 |
| wodo host/device march parity (identical reject ladder) | 1/6 | reject count 3 vs 2 | parity 1.3--1.9e-13 when ladders match |

Both magnitudes are ulp-amplification class, NOT race class (a real
handoff race would be O(1e-2)+; measured 2e-7). Mechanisms: (i) the
iterative BiCGStab stopping point wobbles with GPU reduction ordering;
(ii) cuDSS factorization is not run-to-run bitwise reproducible, and the
wodo spinodal quench-onset attempt sits exactly on the 50-Newton-
iteration convergence knife edge, so one reject can appear/disappear.

## Fixes (commit-paired with this note)

1. Fused gate: tolerances re-locked FROM THE MEASURED TAIL (velocity
   1e-6, pressure 1e-5 = 5x the observed 2e-7 worst case), not from a
   single lucky measurement.
2. Wodo gate: ladder mismatch triggers one full retry of both paths —
   a real parity break fails consistently, the cuDSS coin flip does not.
   Strict 1e-11 state parity still asserted on matched ladders.

## The rules extracted

- A parity tolerance is a DISTRIBUTION statement: lock it from repeated
  runs (the tail), never from one green measurement. The fused gate's
  1e-7 was set from a single 2.3e-8 observation; the actual tail is 2e-7.
- Any gate whose pass/fail hinges on an iteration-count threshold inside
  chaotic dynamics is a coin flip amplifier; either make the comparison
  conditional on matched discrete decisions (with retry) or move the
  comparison away from the knife edge.
- The two identical full-suite runs disagreeing is what exposed all of
  this: keep the full suite as the release gate, and treat a flake in it
  as a measurement opportunity, not noise. (The same runs also caught a
  REAL latent NameError in drag_shape_gradient, dead code since d5a79f8
  because tier5+ad is deselected in routine runs.)

## Open

- Latent question: why did these gates hold through the M1d era? Likely
  they were run far fewer times than believed (routine runs deselect
  tier5; test_device_assembly ran alone on a quiet GPU). No evidence of
  a code regression: git blame shows both test bodies unchanged since
  their introduction.
- If flakes reappear at the new tolerances, the next suspect is the
  warp-torch stream handoff (raw __dlpack__ without stream negotiation
  in device_csr) — no evidence tonight, magnitude says no, but it is the
  one unaudited seam.
