# C6 — Nonlinear solver diagnostics

**Why it matters.** Every implicit phase-field step is a nonlinear (Newton)
solve. Whether a run is trustworthy, slow, or silent garbage is decided inside
that loop. This chapter teaches you to read the loop's trail — residual and
update norms, safeguard activity, condition number — recognize the canonical
failures, and act on each. The failures here are **run and genuinely fail**.

**Objectives.** Read `‖r‖` and `‖δx‖∞` per iteration and recognize quadratic
convergence; pick between absolute / relative / residual stopping and know when
each misleads; see the trust clamp and box projection act; diagnose Newton
stagnation, a min-Δt wall, an ill-conditioned Jacobian, and a diverging start
from their signatures, using the decision tree, leaving a failure checkpoint.

**Prerequisites.** Newton's method + quadratic convergence; the mixed
Cahn–Hilliard weak form and its analytic Jacobian (C0); adaptive stepping + LTE
(C3); Physics P1 (the brick solved here).

**Cost.** Any CUDA GPU, 2-D, <0.5 GB device memory, solver `splu`. ~20–30 s
wall (mostly Warp compile + Python start-up); single- and few-step studies.

## What it does

The production `CahnHilliardStepper` is driven **unchanged**. The per-iteration
Newton trajectory it does not log is reconstructed through the public API:
re-solve the same step capped at `newton_max = 1,2,3,…` and read `last_newton`
and the captured `last_system` (`capture_system=True`). A Newton solve from a
fixed state is deterministic, so the k-th entry is exactly the k-th iterate.

1. **Anatomy** — a healthy poly step converges quadratically (residual squared
   each step) in 3 iterations.
2. **Stopping** — absolute / relative / residual tests; they agree on a
   well-scaled O(1) field and diverge under a rescale (absolute is scale-blind).
3. **Safeguards** — a deep FH quench (`B=8`) fires the box projection (27 dofs
   moved) and the trust clamp, and still converges.
4. **Failures** (each genuinely fails as taught):
   - **Newton stagnation** — deep poly quench at Δt too large: the trust clamp
     fires every iteration, the residual crawls linearly, and after the full
     budget it is still far from tol. The same quench at small Δt converges.
   - **Min-Δt** — the adaptive controller is pinned at `dt_min`; measured
     LTE-at-floor exceeds the requested tolerance.
   - **Ill-conditioned** — an FH state near the wall inflates cond(J) ~49×.
   A **failure checkpoint** for the stagnation case is written to
   `checkpoints/`.

## Run it

```bash
export PYTHONPATH=<repo>/src
python run.py --config configs/c6.yaml --device cuda:0 --solver splu \
    --output outputs/c6 --overwrite --mode reference
python gen_figures.py --run-dir outputs/c6
```

Nine `PASS/FAIL` gates print (several assert a *failure* — the stagnation solve
must not converge, the controller must stay pinned); `results.json` is checked
against `baseline.yaml`. Compare with `EXPECTED.md`.

## Files

- `diagnostics_ch.py` — the diagnostic core (drives the production stepper).
- `run.py` — harness driver (config → provenance → results → baseline check).
- `gen_figures.py` — figures `c6_newton/conditioning/decision.png` +
  `numbers/c6.tex`.
- `configs/c6.yaml` — canonical run record; `baseline.yaml` — invariant gates.
