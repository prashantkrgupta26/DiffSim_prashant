# C6 — expected results (self-check)

`run.py --mode reference` should reproduce the following to a couple of
significant figures. Exact iteration counts and condition numbers vary a little
by card/BLAS; the **behaviours** (converged / not-converged / pinned /
ill-conditioned) are the load-bearing quantities. Nine `[PASS]` gates and
`ALL GATES: PASS` must print, and the baseline check must pass.

## 1. Anatomy of a healthy poly Newton solve (quadratic)

| iteration | ‖r‖ | ‖δx‖∞ |
|---|---|---|
| 1 | 5.6e-2 | 3.1e-3 |
| 2 | 2.0e-8 | 1.2e-6 |
| 3 | 2.6e-16 | 1.6e-15 |

Converged in **3** iterations; the residual is squared each step (correct
analytic Jacobian). No clamp fires.

## 2. Stopping criteria (tol 1e-8)

Residual, absolute-update, and relative-update tests all fire at iteration **3**
on the well-scaled field. Rescaling the field ×10⁻³ makes the absolute-update
test fire earlier — it is scale-blind.

## 3. Safeguards (deep FH quench, B=8)

Box projection moves **27** dofs (max correction 2.0); the trust clamp is
available; the step still **converges**. The safeguard absorbs the overshoot
that would otherwise drive f″ into the 1/ε = 1e4 cap.

## 4. Failures (each genuinely fails)

| case | measured | signature / action |
|---|---|---|
| Newton stagnation (Δt=0.05) | **not converged** in 15 iters, ‖r‖≈7.9e-1, clamp fraction 1.00 | clamped diverging direction → reduce Δt |
| → same quench, Δt=2e-3 | **converged** in 8 iters | proof it is a step-size problem |
| min-Δt | 8 steps, all at floor; LTE≈1.38 > tol 1e-9 | LTE tol unreachable → loosen tol / accept floor |
| ill-conditioned FH | cond(J): 1.3e3 (c=0.5) → 6.3e4 (c=0.003), ~49× | state at the wall → keep iterates off it |

A failure checkpoint is written to `checkpoints/stagnation_step.npz`.

## What must be true regardless of hardware

- Healthy poly Newton converges in ≤ 5 iterations, residual drops > 6 orders.
- The stagnation case does **not** converge (clamp fraction > 0.5); the small-Δt
  variant does.
- The adaptive controller stays pinned at `dt_min` with LTE > tol.
- cond(J) climbs > 10× from bulk to wall.
