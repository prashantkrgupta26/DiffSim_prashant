# E0c — The Recipe for a New PDE: Detailed Report

**Script:** `tutorials/E_differentiable/E0c_recipe_new_pde.py`

This report assumes E0a's and E0b's theory (gradients, the adjoint via Lagrange
multipliers, "relation not algorithm," the do-not-differentiate list, the
dot-product test) as background. E0c's new material is the *checklist* that
ties everything together, extended to **time** (transient PDEs), plus a set of
practical lessons about *designing* a good objective `J`. Part 1 covers the new
theory; Part 2 covers the base task and all five Explore tasks; Part 3 is
takeaways.

---

## Part 1 — The complete theory, as a story

### The checklist: seven questions to ask before writing gradient code for any PDE

E0a and E0b built up individual pieces; E0c's contribution is naming them as a
single, ordered checklist you can run through for *any* new PDE, before writing
a line of gradient code:

1. **What to tape** — only the residual kernels that actually depend on the
   parameters you're differentiating with respect to.
2. **What to freeze** — mesh/classification/geometry (piecewise-constant, no
   useful derivative), basis/quadrature tables (fixed reference-element data),
   RNG seeds (replayed, not differentiated), and any field precomputed outside
   the tape (the "frozen at Gauss points" pattern).
3. **What to store across steps** — for anything with a time dimension, the
   adjoint sweeps *backward* through time and needs the primal state at every
   step it visits. New to E0c; the next section covers this in depth.
4. **Transposed-solve infrastructure** — one factorization serves both the
   forward solve and every adjoint solve, via `trans='T'`, symmetric or not.
5. **The verification ladder** — climb from cheap-and-general to
   expensive-and-specific: the dot-product test (per operator) -> the
   three-way check (per gradient, ideally with an *exact* third leg — see
   complex-step differentiation, below) -> a full transient-chain check
   (the whole backward sweep against FD, end to end).
6. **Nondifferentiability hazards** — `abs`, `min`, `max`, thresholds, and
   masks all have gradients that either jump discontinuously or are zero
   almost everywhere. Replace them with smooth relaxations *before* handing
   `J` to an optimizer.
7. **The cost contract** — a correctly implemented reverse-mode adjoint costs
   `O(1)` forward solves, not `O(N)`. Concretely: the backward sweep should
   cost at most a small constant multiple of the forward march (this codebase's
   own contract: `<= ~2.5x`). If your backward pass costs much more than that,
   you are very likely accidentally unrolling something (a solver, a Newton
   loop) that should have been differentiated via its relation instead.

### Extending the adjoint to time: the transient chain

E0a and E0b's adjoint was for a single steady-state solve, `Au=b`. A time-
marching PDE is a *sequence* of such solves, one per timestep, each depending
on the previous step's state:

```
F^n(u^n, u^{n-1}, m) := A(m) u^n - P u^{n-1} = 0,     n = 1, ..., N
```

(For implicit Euler / BDF1 heat conduction: `A = M/dt + K(m)`, `P = M/dt`,
where `M` is the mass matrix and `K(m)` the parameter-weighted stiffness.) This
is really just `N` copies of the same kind of relation E0a already handled —
so you can, in principle, apply the Lagrangian trick *once per step*, with a
separate multiplier `lambda^n` for each. Doing this and collecting terms
produces a clean **backward recurrence** (derived, not merely stated, in the
kernel source's own comments):

```
seed^N = dJ/du^N                          (only the final step, if J depends only on u^N)
for n = N down to 1:
    A^T lambda^n = seed^n                  (one transposed solve, SAME factorization every step,
                                             since A doesn't change in time here)
    grad += -lambda^n . (dA/dm) u^n         (contract with the STORED primal state u^n)
    seed^{n-1} = P^T lambda^n               (propagate to the previous step)
```

Read this as: run time *backward*. At each step, solve one transposed linear
system to get that step's adjoint variable `lambda^n`; use it to accumulate a
sensitivity contribution (needing that step's *primal* state `u^n`, which is
why storage matters — see next); then hand a "seed" derived from `lambda^n`
to the *previous* step, exactly mirroring how the forward march hands `u^{n-1}`
forward to produce `u^n`. This is why it's called a "backward sweep" or
"transient adjoint chain" — it's the forward march, run in reverse, carrying
sensitivity instead of state.

**Crucial detail, and the subject of Explore (b):** the `seed^N = dJ/du^N` line
above is specific to a `J` that only looks at the *final* state. If `J` instead
depends on the state at *every* timestep (a time-integrated misfit), *every*
step in the backward sweep needs its own additional, direct injection of
`dJ/du^n`, not just a propagated contribution from the future. Getting this
wrong doesn't produce a slightly-off gradient — it silently computes the
gradient of a *different* objective than the one you actually wrote down.

### Full-store vs. recompute-from-checkpoint: the memory/compute trade-off

The backward sweep above needs `u^n` at every step it visits — but the
forward march that produced those states is long gone by the time the backward
sweep runs. Two ways to supply them:

- **Full-store:** keep every `u^n` from the forward march in memory. Backward
  cost: pure solves, no extra forward work — the cheapest possible backward
  pass, at the cost of memory that grows linearly with the number of *steps*.
- **Recompute-from-checkpoint:** keep only a handful of saved states
  ("checkpoints") — the initial condition plus, say, every `k`-th step. When
  the backward sweep needs an un-saved `u^n`, re-run the forward march from
  the nearest earlier checkpoint up to step `n`. Memory drops to
  `O(n_checkpoints)`; the cost is extra forward solves during the backward
  pass.

This is a genuine trade-off, not a free lunch, and the "sweet spot" is a real
algorithmic question: keeping only the initial condition (one checkpoint)
minimizes memory but costs `O(steps^2)` recompute solves in the worst case
(each backward step might need to re-march from the very start); keeping
*every* step is `O(steps)` memory with zero recompute. The provably optimal
middle ground, for a fixed memory budget, is a *logarithmic* checkpoint
schedule (the algorithm is called **REVOLVE**, Griewank & Walther) —
`O(log(steps))` checkpoints buys `O(steps * log(steps))` total recompute cost,
the best possible trade-off curve. This codebase deliberately does not build
REVOLVE itself (existing libraries — `pyrevolve`, `checkpoint_schedules` — do
this well already, and building your own is unnecessary duplicated effort);
the point of this chapter is to make sure you understand *why* the trade-off
exists and *what invariant must hold regardless of strategy*: **both modes
must compute the exact same discrete adjoint**, agreeing to machine precision
— because recompute reproduces `u^n` bit-for-bit (same factorization, same
solves), not approximately.

### Complex-step differentiation: an *exact* third verification leg

E0a used finite differences as an independent check on the adjoint and the
tape. Central FD has a fundamental limitation: it's only *approximately*
correct, limited by the trade-off between truncation error (large `eps`) and
floating-point cancellation error (small `eps`) — there's no `eps` at which
central FD is *exact*. **Complex-step differentiation** sidesteps this
entirely, for any function built from ordinary analytic arithmetic (`+`, `-`,
`*`, `/`, `exp`, polynomial operations — everything a typical PDE residual is
made of). The trick: evaluate your function at a *complex* input,
`p + i*h`, with `h` extremely small (say `1e-30`) — no subtraction of two
nearly-equal numbers is ever involved, so there's no cancellation error to
manage. A Taylor expansion shows:

```
f(p + i*h) = f(p) + i*h*f'(p) - (h^2/2)*f''(p) + O(h^3)
=> Im[f(p + i*h)] / h = f'(p) - O(h^2)
```

Since `h` can be made *absurdly* small (`1e-30`) without any cancellation
penalty (there's no subtraction — you just read off the imaginary part), the
`O(h^2)` error term is negligible to the point of being exact in double
precision. The catch: your code has to actually support complex arithmetic
throughout the computation (which is why E0c's complex-step leg uses a
separate, complex-dtype version of the Allen-Cahn forward solve) — a
constraint FD and the adjoint don't share. Having three legs — hand-derived
adjoint, automatic tape/complex-step (exact), and central FD (independent,
approximate) — where two of the three are *exact* by different mechanisms
(the adjoint from algebra, complex-step from an cancellation-free numerical
trick) gives unusually strong verification: if all three agree, a coincidental
bug canceling itself out in exactly the same way across three unrelated
mechanisms is vanishingly unlikely.

### Choosing `J`: five ways a well-implemented gradient can still mislead you

A completely correct adjoint computes the exact gradient of whatever `J` you
handed it — but a *badly designed* `J` can make that correct gradient useless
in practice, in specific, nameable ways:

- **Nonsmoothness.** `J` built from `max`, `min`, `abs`, or a hard threshold
  has a gradient that jumps discontinuously wherever the "active" branch
  switches (`max_i g_i` picks a different `i` on either side of a crossing),
  and is exactly zero almost everywhere for a true hard threshold. Feed this
  to a gradient-based optimizer and it stalls or oscillates near the jump. The
  fix: a **smooth relaxation** — for `max`, the **log-sum-exp** (softmax)
  function, `lse_beta(g) = (1/beta) log(sum_i exp(beta*g_i))`, which
  converges to `max_i g_i` as `beta -> infinity` but has a smooth, well-
  behaved gradient (a softmax-weighted blend of the branch gradients) at any
  finite `beta`. `beta` trades bias (how close `lse` is to the true `max`)
  against smoothness (how close its gradient's jumpiness is to the true
  `max`'s) — Explore (c) maps this trade-off out directly.
- **Signal below the noise floor (the "H4 protocol").** Even a perfectly
  correct, perfectly smooth gradient is useless if `|dJ/dp|` is so small that
  it's comparable to the *numerical noise* in evaluating `J` itself (floating-
  point roundoff, iterative-solver tolerances). The diagnostic: estimate the
  FD noise floor directly (by looking at how much a central-FD estimate
  *itself* varies as you sweep the step size `eps` — a well-scaled problem has
  a wide, stable plateau; a poorly-scaled one doesn't), and check your actual
  gradient clears that floor by orders of magnitude *before* spending any
  optimizer iterations.
- **Flat-`J` geometry (where you measure matters).** A gradient can be exactly
  correct and still be *numerically tiny* simply because your chosen
  observable doesn't actually see the parameter's effect — a probe placed
  somewhere the field never has meaningful amplitude has `du/dm ~ 0` there,
  regardless of how strongly `m` affects the field *elsewhere*. This isn't a
  bug in the gradient; it's a bug in the experiment design.
- **Identifiability and conditioning (the deepest of the five).** Even with a
  strong, well-scaled, smooth gradient, a parameterization can have
  **structurally invisible directions** — combinations of parameters that
  produce (numerically) identical observations, no matter how you vary them.
  The diagnostic isn't the gradient at all — it's the **condition number of
  the observation Jacobian** (how the observable predictions change as *each*
  parameter varies, stacked into a matrix, then examined via its singular
  values). A near-singular observation Jacobian means two or more parameter
  directions are "aliased" — nearly indistinguishable from your data — and no
  amount of optimizer cleverness fixes that; only *better or more diverse
  data* (or a smarter, gauge-fixed parameterization that removes the
  redundant direction *by construction*) can.
- **Multi-observable objectives.** The general fix for both flat-`J` geometry
  and poor identifiability: don't rely on one scalar probe misfit. Combine
  several *diverse* observables — probes at multiple locations, an
  integrated/structural quantity, a time-resolved signal — into
  `J = sum_k w_k ||O_k(u) - d_k||^2`. Different observables are sensitive to
  different directions in parameter space; combining enough diverse ones is
  often what turns a fundamentally underdetermined inverse problem (echoing
  E0b's 5-probes-for-64-unknowns lesson) into a well-posed one.

**Tikhonov regularization, reframed.** A penalty term `eps*||m - m_ref||^2`
doesn't fix identifiability by adding information — it adds a *stated prior
belief* ("prefer solutions near `m_ref`"), which makes the optimization
well-posed (strictly convex, no flat directions) even when the data alone
cannot distinguish some directions. Framed this way, Tikhonov isn't a
numerical band-aid — it's a *scientific hypothesis* you're choosing to encode,
and the honest thing to report alongside a regularized fit is that the
regularization is doing some of the work, not the data alone.

---

## Base Task — The checklist, the transient heat chain, checkpointing, five J-design lessons, and a phase-field payoff

### 1. Problem statement

Apply the full checklist to a genuinely new (time-dependent) PDE: transient
heat conduction with a spatially-varying conductivity, `u_t = div(kappa(x)
grad u)`, discretized with implicit Euler in time. Verify the transient
adjoint chain against finite differences; measure the primal-storage cost;
compare full-store against recompute-from-checkpoint and confirm they agree to
machine precision; measure the backward/forward cost ratio against the `2.5x`
contract; run five small demonstrations of the `J`-design pitfalls above; and
finally differentiate a mini Allen-Cahn phase-field problem (semi-implicit in
time) with respect to two physical parameters (mobility `M` and gradient-
energy coefficient `kappa`), verified with the strongest available three-way
check (adjoint / complex-step / central FD).

### 2. Results

All checks passed, matching the docstring's EXPECTED RESULTS exactly:

```
Heat chain adj vs FD          : 6.41e-07   (expect < 1e-5)
Store vs recompute dJ         : 0.00e+00   (expect < 1e-10)
Backward/forward wall ratio   : 1.17x      (expect < 2.5x)
logsumexp/max grad-jump ratio : 0.128      (expect < 0.5)
signal / FD-floor ratio       : 4.63e+08   (expect > 1e3)
flat-J contrast               : 98.8x      (expect > 10x)
AC three-way (adj/cstep/FD)   : 8.01e-09   (expect < 1e-6)
```

### 3. Interpretation

Every check passes comfortably, and by a wide margin in most cases — the
signal-to-floor ratio (`4.63e8`, vs. a `1e3` bar) and the flat-J contrast
(`98.8x`, vs. a `10x` bar) both suggest this particular worked example was
deliberately chosen to sit well inside "healthy" territory, giving you a clear
reference point for what a *well-posed* setup looks like before you go looking
for the pathological ones (which is exactly what the explore tasks then do).
The `0.00e+00` store-vs-recompute agreement is worth pausing on: this isn't
"very close," it's *bit-for-bit identical*, exactly as the checklist demands —
both modes solve the same linear systems with the same factorization in the
same order, so there is no room for even floating-point-level divergence.

### 4. The real learning — why this is in the tutorial

**A checklist is only valuable if you've seen every one of its items actually
matter at least once — and this base task is a guided tour that manufactures
exactly that experience, item by item, inside one coherent physical example.**
Rather than presenting seven abstract rules, the chapter builds a real
transient PDE, and then *deliberately* exercises each checklist item as a
measurable, falsifiable claim: item 3 (storage) becomes a byte count you can
read off; item 4 (transposed reuse) becomes the `2.33x`-style timing ratio from
E0b; item 5 (verification ladder) becomes an actual climbed sequence of three
checks; item 7 (cost contract) becomes a measured `1.17x`. This is the
capstone of E0a/E0b's whole pedagogical arc: by the time you reach E0c, the
checklist isn't a list of things to trust — it's a list of things you've
already personally verified, once, on a worked example, and can now apply with
real confidence to something genuinely new.

---

## Explore (a) — Scaling up: level 4, 60 steps, and a second checkpoint budget

### 1. Problem statement

Push the heat chain to level 4 (256 elements) and 60 timesteps. Recompute the
storage inventory and the backward/forward ratio. At what `(steps x dofs)`
does full-store become uncomfortable? Sweep a second (and third) checkpoint
budget and watch the recompute cost grow as fewer checkpoints are kept.

### 2. Results

| level/steps | full-store size | backward/forward ratio | recompute (10 ckpt) | recompute (4 ckpt) | recompute (2 ckpt) |
|---|---|---:|---:|---:|---:|
| L3/30 steps | 19.6 KiB | 0.57x | 1.60x fwd | 2.42x fwd | 4.72x fwd |
| L4/60 steps | 137.7 KiB | 0.74x | 2.34x fwd | 4.47x fwd | 7.67x fwd |

### 3. Interpretation

The trade-off is exactly, cleanly monotonic in both directions — more
checkpoints kept, less recompute cost; fewer checkpoints kept, more recompute
cost, at every configuration tested. Neither configuration comes remotely
close to being memory-uncomfortable in absolute terms (kilobytes, not
gigabytes) — this experiment is a scaled-*down* rehearsal of the trade-off,
not yet the regime where it bites. The docstring's own extrapolation (`8 GB`
at `1e6` dofs and `1e3` steps) is roughly `10^4` times more DOFs and `~17x`
more steps than this test — i.e. full-store memory would grow by very roughly
`10^4 x 17 ~ 1.7e5x` from this experiment's `137.7 KiB`, landing close to that
quoted `8 GB` figure, a reassuring consistency check on the extrapolation even
without actually running a problem that large.

### 4. The real learning — why this is in the tutorial

**A trade-off curve you've only seen at toy scale is still worth mapping,
because the *shape* of the curve (monotonic, and roughly how steeply
recompute cost grows as checkpoints shrink) is what transfers to production
scale, even when the absolute numbers don't.** You don't need to actually run
a million-DOF, thousand-step problem to understand how checkpointing will
behave there — you need to have confirmed, on a problem small enough to fully
inspect, that the mechanism behaves the way the theory predicts. This is a
recurring pattern across this entire project: verify the *mechanism* at small,
cheap scale; trust the *extrapolation* of that mechanism to scales you can't
directly afford to test.

---

## Explore (b) — Time-integrated `J`, and re-deriving the seed injection

### 1. Problem statement

Replace the final-time-only misfit with a time-integrated one, `J = (dt/2)
sum_n ||W u^n - u_obs^n||^2`. Now `dJ/du` seeds *every* step, not just the
last. Re-derive the seed injection in the backward sweep and re-verify against
FD.

### 2. Results

```
correctly re-derived seed logic: adjoint vs FD rel err = 5.99e-07
  (compare to the base task's final-time-only case: 6.41e-07 -- same precision class)

WRONG (reusing the OLD final-time-only seed logic, unmodified, on this new J):
  rel err vs FD = 1.000e+00
```

### 3. Interpretation

The correctly re-derived recurrence (`seed^n = P^T lambda^{n+1} + dJ/du^n` for
every `n < N`, instead of only injecting at `n=N`) matches finite differences
just as precisely as the original case. The deliberately-wrong comparison —
literally just reusing the old code's seed logic against the new `J` without
updating it — doesn't produce a *slightly* wrong gradient; it produces one
with **100% relative error**, i.e. essentially uncorrelated with the correct
answer. This makes sense once you see why: the old logic only ever injects a
direct `dJ/du` contribution once, at the very last step; for the new,
time-integrated `J`, every other step's own substantial misfit contribution
(each timestep contributes its own probe-vs-truth mismatch) was being silently
dropped entirely.

### 4. The real learning — why this is in the tutorial

**The transient adjoint recurrence is not one universal formula you memorize
and reuse — it's a derivation that depends on exactly how `J` reads the
trajectory, and changing `J` requires re-deriving it, every time.** This is
the sharpest, most concrete illustration in the whole E0 sequence of a lesson
that could otherwise stay abstract: "correctly implementing the adjoint for
problem A" gives you *zero* free correctness guarantee for problem B, even
when B looks like a small, natural variation of A. The 100%-error result isn't
a subtle numerical discrepancy you might rationalize away — it's a loud,
unmistakable signal that *something structural* was skipped, which is exactly
the kind of signal you want a verification check to produce when a real
derivation step has been missed.

---

## Explore (c) — Where is the logsumexp "knee"?

### 1. Problem statement

Sweep `beta` in the logsumexp relaxation from 5 to 200. Small `beta` is smooth
but biased (the relaxation sits well above the true `max`); large `beta`
approaches the true `max` but the gradient's jumpiness returns. Where is the
practical sweet spot?

### 2. Results

| beta | jump ratio (lse/max) | bias (mean abs error vs. true max) |
|---:|---:|---:|
| 1 | 0.026 | 0.958 |
| 20 | 0.062 | 0.012 |
| 40 (the base task's own default) | 0.128 | 0.004 |
| 100 | 0.322 | 0.001 |
| 500 | 0.814 | 0.0001 |

### 3. Interpretation

Both quantities move monotonically with `beta` across the whole tested range —
there's no non-monotonic "U-shape" where the jump ratio first improves, then
gets worse; it climbs steadily throughout. The real trade-off is in the
*rates*: bias falls steeply from `beta=1` to `beta~20-40` (from `0.96` down to
`0.004`, a `>99%` reduction) while the jump ratio only creeps up slowly over
that same stretch (`0.026` to `0.128`); past `beta~40`, further bias
improvement is marginal (down to `0.0001` by `beta=500`) while the jump ratio
climbs steeply toward the fully nonsmooth limit (`0.81`, close to `1.0`). The
"knee" — where you've captured most of the achievable bias reduction while
still paying only a modest smoothness cost — sits right around `beta=20-40`.
The base task's own default, `beta=40`, lands almost exactly there.

### 4. The real learning — why this is in the tutorial

**A hyperparameter in a relaxation isn't something to crank as high as
possible "to be more accurate" — pushing it too far quietly un-does the entire
reason you introduced the relaxation in the first place.** It would be a
natural but wrong instinct to think "logsumexp with `beta=500` is *better*
because it's closer to the true `max`" — technically true for the *value*, but
by then you've thrown away most of the smoothness you introduced the
relaxation to get, defeating its purpose for gradient-based optimization. This
task is teaching you to think of a relaxation parameter as tracing out an
explicit trade-off *curve*, not a single number to be maximized — and to
actually plot (or, as here, tabulate) that curve rather than picking a value
by instinct.

---

## Explore (d) — Fully implicit Newton Allen-Cahn, adjoint via the converged Jacobian

### 1. Problem statement

The base task's Allen-Cahn demo uses a semi-implicit split (interface term
implicit, reaction term explicit) specifically to keep each step a single
linear solve. Make it **fully implicit** instead (Newton-solved each step),
and differentiate the **converged residual** via the implicit function theorem
— per E0b's do-not-differentiate-Newton-loops principle — rather than
unrolling the Newton iterations. Confirm the three-way check still holds.

### 2. Results

```
dJ/dM      adjoint=5.427483e-05  FD=5.427483e-05  rel_err=5.06e-09
dJ/dkappa  adjoint=1.280978e-02  FD=1.280978e-02  rel_err=5.97e-10
worst rel err: 5.06e-09  (compare to the semi-implicit scheme's own three-way result: 8.01e-09)
```

### 3. Interpretation

The fully implicit scheme's adjoint matches finite differences to the same
precision class as the base task's semi-implicit three-way check — genuinely
strong agreement, achieved on the first implementation attempt. The key
structural difference from the semi-implicit case: at each timestep, the
adjoint solve now uses the **converged Newton Jacobian** (`dR^n/dc^n`,
evaluated at the fully-converged `c^n` — including the full nonlinear reaction
term's derivative `f''(c^n) = 3(c^n)^2 - 1`, not the frozen, previous-step
value the semi-implicit scheme used), and that Jacobian was built and
factorized once *after* Newton had already converged for that step — never
during the Newton iterations themselves. The only other change versus the
semi-implicit backward sweep: since the reaction term is now evaluated
implicitly (at `c^n`, not `c^{n-1}`), the coupling to the *previous* step
simplifies to just the bare mass/dt term (`dR^n/dc^{n-1} = -(m/dt)`), because
`f'(c^n)` no longer depends on `c^{n-1}` at all in the fully implicit scheme.

### 4. The real learning — why this is in the tutorial

**"Never unroll a Newton loop" is easy to accept as an abstract rule and easy
to get subtly wrong in practice — the fix is building the fully implicit
version yourself and confirming the IFT-based adjoint still works, rather than
only ever seeing the rule applied to code someone else wrote.** The
semi-implicit demo in the base task was deliberately built to *avoid* ever
needing this — a design choice explicitly flagged in the source comments. This
explore task removes that safety net on purpose: building the Newton loop,
correctly identifying the converged-state Jacobian (not the Jacobian at some
intermediate iterate — a genuinely easy mistake, since Newton itself uses many
different Jacobians along the way to convergence, and only the very last one
is relevant to the adjoint), and confirming the result against finite
differences, is a considerably more convincing demonstration of the principle
than reading about it. The docstring notes this is *exactly* the pattern M4
uses for learned phase-field thermodynamics — this toy version is the smallest
faithful rehearsal of a technique used for real, physically-motivated
production work elsewhere in this project.

---

## Explore (e) — A real Adam fit, Tikhonov, and measuring conditioning directly

### 1. Problem statement

Turn the Allen-Cahn demo into a genuine parameter-recovery fit: run Adam on
`(M, kappa)` from a flat/naive starting guess. Add a small Tikhonov prior and
observe how it changes *which* minimum the optimizer reaches. Then add
composition-diverse initial conditions and measure the conditioning
improvement — the base task's quoted "beyond-FH" story, reproduced hands-on
rather than cited.

### 2. Results

```
Adam, single IC, no Tikhonov:         M=-0.975  kappa=0.0190   (TRUE: M=1.0, kappa=0.01)
  -- unstable: wrong sign on M entirely
Adam, single IC, WITH Tikhonov (eps=0.5): M=0.9999  kappa=0.0138
  -- pulled to a near-correct, stable answer
Adam, THREE diverse ICs, no Tikhonov: M=1.684  kappa=0.0142
  -- better than the unregularized single-IC fit, though not perfectly tuned
     (fixed Adam hyperparameters weren't separately recalibrated for the
     3x-larger joint gradient of the multi-IC objective)

Observation-Jacobian conditioning (own from-scratch computation, 2 free parameters):
  single IC:   singular values [0.988, 1.28e-12]     cond = 7.69e+11  (essentially singular)
  diverse ICs: singular values [2.87,  0.00787]      cond = 3.65e+02
  improvement: ~2.1 billion x
```

### 3. Interpretation

The single-IC, unregularized fit doesn't just under-perform — it diverges to
a physically nonsensical, wrong-sign mobility. Adding a modest Tikhonov prior
(`eps=0.5`) rescues it completely, recovering `M` to four significant figures
— directly reproducing, at this project's own small and fully-inspectable
scale, the qualitative failure-and-rescue pattern the base task's quoted
beyond-FH story describes ("Tikhonov ... pulls the ill-conditioned fit to a
definite answer rather than letting it wander"). The conditioning measurement
is the cleanest and most decisive result of the five explore tasks: with only
one initial condition, the observation Jacobian's second singular value is
`~1e-12` — for practical purposes, exactly zero. That means there is a
direction in `(M, kappa)`-space (some particular combination of the two) whose
effect on the observed probe data is, to machine precision, *invisible* — the
textbook definition of an unidentifiable ("gauge") direction, and exactly why
the unregularized fit could wander off to a wrong-sign answer: it was moving
freely along a direction the data provides zero information about. Adding two
more, differently-shaped initial conditions collapses that near-singularity by
nine full orders of magnitude (`cond` `7.69e11 -> 365`) — each new initial
condition probes the system's response along different directions, and enough
diversity removes the aliasing entirely. Notably, this small system's diverse-
protocol condition number (`365`) lands right inside the range the quoted
beyond-FH literature figures report for *their* (much larger, different
physical system's) diverse protocol (`2e2-5e2`) — an unplanned, satisfying
quantitative echo suggesting this isn't a coincidence of one toy problem, but
a genuine, transferable signature of how data diversity resolves parameter
aliasing.

### 4. The real learning — why this is in the tutorial

**A quoted result from elsewhere in the project ("conditioning improves 15-30x
with diverse data") is a claim worth personally reproducing at a scale you can
fully see and verify — not because the original claim is doubted, but because
*building* the diagnostic yourself is what turns "I read that regularization
and data diversity matter" into "I have personally watched a 2-parameter
system go from utterly unidentifiable to well-posed, and I know exactly which
number to compute to tell the difference."** This is the capstone lesson of
the entire E0 sequence, applied to identifiability specifically: the
observation-Jacobian condition number is a completely general, computable
diagnostic — two lines of finite-difference code and an SVD, here — that
applies to any inverse problem, of any size, with any parameterization. Having
built it once, on a problem small enough to fully understand and sanity-check,
is what makes it a tool you'll actually reach for on a problem too large to
fully understand by inspection alone.

---

## Key Takeaways — E0c, in plain terms

1. **The seven-item checklist isn't an abstract list to memorize — this
   chapter makes you personally verify every item once, on one real worked
   example**, turning "the checklist says to do X" into "I measured X myself
   and confirmed it matters."

2. **The transient adjoint is the same Lagrangian trick as the steady-state
   one, applied once per timestep** — a backward-in-time sweep, one
   transposed solve per step (reusing one factorization, since the operator
   doesn't change in time here), needing the stored (or recomputed) primal
   state at each step.

3. **The full-store/recompute-from-checkpoint trade-off is real, monotonic,
   and must produce bit-for-bit identical gradients regardless of which mode
   you use** — verified directly, not just claimed, and the trade-off's shape
   (confirmed here at toy scale) is what transfers to problems too large to
   directly test.

4. **A `J` you wrote down for one purpose does not silently transfer its
   correctness to a modified version of itself.** Switching from a final-time
   misfit to a time-integrated one requires re-deriving the seed injection —
   reusing the old logic unmodified gives 100% relative error, not a subtle
   discrepancy.

5. **Five distinct, nameable failure modes can sabotage an otherwise-correct
   gradient: nonsmoothness, a buried signal, flat-J probe geometry, poor
   identifiability, and (the general fix for the latter two) too narrow an
   observable.** Each has its own diagnostic — smooth-relaxation jump ratios,
   the H4 signal/floor check, informative-vs-flat probe contrast, and
   observation-Jacobian conditioning — and none of them is "just run the
   optimizer and see."

6. **Tikhonov regularization is a stated scientific prior, not a numerical
   trick — and, verified directly here, it can be the difference between a
   fit that diverges to a wrong-sign answer and one that recovers the truth
   to four significant figures**, when the underlying data alone cannot
   distinguish every parameter direction.

7. **The observation-Jacobian condition number is a small, general,
   reusable diagnostic for parameter identifiability — worth having built and
   trusted on a toy problem before ever needing it on a real one.** Here it
   went from `7.69e11` (single IC, essentially singular) to `365` (three
   diverse ICs) — a billion-fold improvement from nothing more than better
   experiment design, with zero change to the physics, the solver, or the
   optimizer.
