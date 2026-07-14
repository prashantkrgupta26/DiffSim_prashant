# D7 — expected results (self-check)

Running `python run.py` (defaults: 9×9 nodes, coupled Cahn–Hilliard ×
Allen–Cahn crystallization, 6 steps, BDF2, $dt=0.02$; schedule bounded
$T\in[0.2,2.0]$; target from a ground-truth cold ramp $T:0.5\to0.3$)
reproduces the design below.

**Gradient self-check** (design objective, adjoint vs finite diff, at the
flat guess $T=1$): worst `adj/fd` $\approx 5.6\mathrm{e}{-9}$ over the 6
steps — all `< 1e-6`.

**Design:**

| quantity | value |
|---|---|
| target crystalline fraction (cold ramp) | $0.23877$ |
| flat initial guess $T=1.0$ gives | $0.21909$ |
| designed schedule achieves | $0.23877$ |
| $\lvert$achieved $-$ target$\rvert$ | $4.2\mathrm{e}{-15}$ |
| objective $J$: initial $\to$ designed | $1.94\mathrm{e}{-4}\to 8.7\mathrm{e}{-30}$ |
| optimiser evaluations | 7 |
| designed $T(t)$ | $\approx[0.249,0.426,0.431,0.446,0.488,0.615]$ |

The designed schedule is **interior** (no step pinned at the $[0.2,2.0]$
bounds).

**What must be true regardless of hardware:**

- **The design-objective gradient matches finite differences** (`< 1e-6`)
  — the FD-verified gate the optimiser rests on.
- **The designed process hits the target** (to $\sim10^{-15}$). The target
  was produced by a reachable ground-truth schedule, so a solution exists;
  running the simulator *backward* finds a recipe that reaches it.
- **The objective collapses** by many orders of magnitude, in a handful of
  evaluations — the schedule gradient points straight downhill.
- **The recovered schedule need not equal the ground-truth ramp.** Many
  schedules can hit the *same scalar* crystalline fraction; the optimiser
  finds *a* valid recipe, not necessarily the one that generated the
  target. (Pin down more of the morphology — add a domain-size term — to
  constrain it further; see the questions.)

Exact schedule values and iteration counts depend on the optimiser/BLAS
build; the invariant is achieved $\to$ target with the objective collapsing.
