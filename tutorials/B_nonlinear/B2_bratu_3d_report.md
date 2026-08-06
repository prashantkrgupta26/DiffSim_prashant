# B2 — Bratu in 3-D + Continuation: Detailed Report

**Script:** `tutorials/B_nonlinear/B2_bratu_3d.py`
**Date:** August 6, 2026

This report covers the base task and all three Explore tasks in B2. For each: the
problem statement, what was measured, what it means, and the underlying lesson the
tutorial is teaching. A "Key Takeaways" section closes it out.

---

## Base Task — Carrying the Newton solver to 3-D, and naive continuation

### 1. Problem statement

B1 built a Newton solver for the Bratu problem in 2-D and proved it two ways: the
Newton residual squares each step, and the spatial MMS order matches A1 exactly.
B2 asks two things. First, a sanity check: does the *exact same* nonlinear solver
carry over to 3-D unchanged, with the same order-2 spatial accuracy? Second, and
the real point of the chapter: use the *physical* Bratu problem (`f=0`, `g=0` — no
manufactured solution) and walk a parameter `lam` upward from a small value,
re-using each converged solution as the starting guess for the next `lam`
("continuation" / warm-starting). The physical Bratu problem has two solution
branches for `lam` below a critical `lam*` (for the unit cube, literature puts
`lam* ~ 9.9`) and *no* solution above it — so continuation should walk the "lower"
branch until it runs into that wall. What happens to the Newton iteration count as
`lam` approaches `lam*`?

### 2. Results

```
3-D MMS p1: errors 2.057e-02  5.189e-03  1.301e-03   orders 1.99  2.00

continuation (physical Bratu, unit cube, level 3):
  lam    |u|_inf  newton its
  1.0     0.0600           2
  3.0     0.1973           3
  5.0     0.3684           3
  7.0     0.6004           3
  8.0     0.7612           3
  9.0     0.9910           4
  9.5     1.1693           4
 10.0     1.5893           6
 11.0          —      FAILED   <- past the fold?
```

Matches the docstring's EXPECTED RESULTS exactly.

### 3. Interpretation

The 3-D order matches 2-D's order-2 result exactly, confirming (again, in a new
dimension) that nonlinearity doesn't touch discretization accuracy — only the
solver loop changes. The continuation table is the interesting part: the iteration
count needed to converge is *flat* (2-3) for small `lam`, then visibly creeps
upward (4, then 4, then 6) as `lam` approaches 9-10, and finally fails outright at
`lam=11`. This creeping-then-failing pattern is the Jacobian *telling you* it is
heading toward singularity: as `lam -> lam*`, the linearized system becomes
increasingly ill-conditioned, so each Newton correction becomes less accurate,
needing more iterations to compensate — until, past the fold, no amount of
iterations helps because there's genuinely nothing there to converge to.

### 4. The real learning — why this is in the tutorial

This is the direct payoff of B1's Explore (c): there, tracking the 2-D fold with
fixed lam-steps produced a *sharp cliff* (converges fine right up to the edge, then
fails completely one step later) because the steps were too coarse to resolve the
narrow zone where degradation actually happens. Here in 3-D, the schedule
(`1,3,5,7,8,9,9.5,10,11`) happens to land *inside* that narrow zone at `9-10`,
so you get to see the gradual creep (4, 4, 6) that B1's coarser steps skipped over
entirely. The professor is teaching you that **naive parameter continuation is a
genuinely fragile technique** — its usefulness for diagnosing a bifurcation depends
entirely on how finely you happen to sample near it, which you don't know in
advance. That fragility is not a flaw to shrug off; it's the exact motivation for
Explore (b)'s pseudo-arclength method, which removes the dependence on lucky step
placement entirely.

---

## Explore (a) — Bisect the fold at levels 2, 3, 4

### 1. Problem statement

The base run found *some* value of `lam` between 10 and 11 where Newton stops
converging, but that's only known to within a whole unit. The task: **bisect**
`lam` between the last-converged value and the first-failing value to pin down the
discrete critical `lam*_h` precisely, and do this at three mesh levels (2, 3, 4) to
see how the discrete fold moves as the mesh refines — literature gives `lam* ~ 9.9`
for the true (continuum) problem on the unit cube.

### 2. Results

Coarse continuation to bracket the fold, then bisected to a `lam` resolution of
`0.01`, warm-starting every trial from the best known-converged solution:

| level | n_free | discrete lam*_h bracket | midpoint | vs. literature (9.9) |
|---:|---:|---|---:|---:|
| 2 | 125 | [10.3594, 10.3672] | 10.363 | +4.7% |
| 3 | 729 | [10.0156, 10.0234] | 10.020 | +1.2% |
| 4 | 4,913 | [9.9219, 9.9297] | 9.926 | +0.26% |

(A robustness note on the implementation: the stock `Bratu3D.newton` crashes rather
than fails cleanly when a step diverges badly enough — `exp(u)` overflows and the
resulting Jacobian becomes exactly singular, which `scipy.sparse.linalg.splu`
raises a hard `RuntimeError` for. Bisection deliberately probes right at the edge
of convergence, so this had to be fixed first: wrapped the Newton loop to catch the
overflow warning, catch the `RuntimeError`, and also bail out on a diverging
iterate (`max|u| > 1e4`) — all three now register as a clean "Newton failed" rather
than crashing the whole sweep.)

### 3. Interpretation

The discrete fold converges **monotonically downward** toward the literature value
as the mesh refines — `10.363 -> 10.020 -> 9.926` — landing within a quarter of a
percent of `9.9` at level 4. The direction of the error is not an accident: a
coarse mesh under-resolves the solution's curvature near blow-up, so the *discrete*
Jacobian doesn't go singular until you push `lam` a bit further than the true
continuum problem requires. In other words, **coarse meshes systematically
over-estimate how far you can push a nonlinear parameter before things break** —
worth remembering any time a coarse mesh "successfully" solves a case that later
fails to converge on a refined one; that isn't always a bug, it can be exactly this
effect.

### 4. The real learning — why this is in the tutorial

This ties mesh convergence (an idea you've been testing since A1 purely for
*accuracy*) to something new: **mesh convergence also applies to qualitative
solver *behavior*, not just to the numerical value of the solution.** The location
of a bifurcation is itself a quantity with a discretization error, and it converges
at some rate as `h -> 0`, exactly like an L2 error does. This is a genuinely
easy thing to miss if you've only ever thought of "does my mesh resolve the
solution" — here the question is "does my mesh resolve the *stability structure* of
the solution," which is a strictly harder and more subtle property. It's also a
very concrete, load-bearing warning for real nonlinear simulation work: if you're
hunting for a critical parameter value (an ignition threshold, a buckling load, a
flutter speed — anything with a Bratu-like fold), you cannot trust a coarse-mesh
answer for *where* the critical point is, even if every individual solve on that
coarse mesh converges cleanly.

---

## Explore (b) — The pseudo-arclength augmented system (derivation only)

### 1. Problem statement

Newton fails exactly *at* the fold because the Jacobian `J = dF/du` is singular
there — by definition, a fold is a point where the linearization loses full rank.
Straightforward lam-stepping (increase `lam`, hold it fixed, solve for `u`) cannot
walk through, or even up to, this point, because "solve for `du` given a `lam`
step" requires inverting an increasingly ill-conditioned (and, in the limit,
singular) matrix. **Pseudo-arclength continuation** fixes this by no longer
treating `lam` as an independent input you choose — instead, `u` and `lam` are
solved for *jointly*, parametrized by a new variable, the arclength `s`, along the
solution curve in `(u, lam)`-space. The task, explicitly *not* to be implemented
(a "week project," per the prompt) — just derive the augmented system on paper.

### 2. The derivation

Let `z = (u, lam) in R^(n+1)` and write the residual from B1/B2 as `F(u, lam) = K
u - b_nl(u, lam)` (the same `K` and weighted-integral machinery used throughout;
`b_nl` folds in the `lam * exp(u)` nonlinear term and any `f`). The solution set
`{F(u,lam) = 0}` is (generically) a smooth 1-D curve in `(u, lam)`-space — even
though this curve *turns back on itself* in the `lam` direction at a fold (so `lam`
alone cannot parametrize it there), the curve itself remains perfectly smooth as a
function of **arclength** `s` measured along it. This is the entire trick: stop
parametrizing by `lam`, start parametrizing by `s`.

**Given:** a known point on the curve `(u0, lam0)` with `F(u0, lam0) = 0`, and a
known unit tangent direction `(u_dot0, lam_dot0)` at that point (satisfying `du/ds,
dlam/ds` there — see the tangent update below for how it's obtained).

**Step 1 — predictor.** Take an explicit Euler step along the tangent to get an
initial guess for the next point, a target arclength step `Δs` ahead:

```
u_pred   = u0   + Δs * u_dot0
lam_pred = lam0 + Δs * lam_dot0
```

**Step 2 — augmented (bordered) system.** Solve for the *actual* next point `(u,
lam)` on the curve, `Δs` away from `(u0, lam0)` in arclength, via the augmented
system of `n+1` equations in the `n+1` unknowns `(u, lam)`:

```
F(u, lam) = 0                                                     (n equations)
N(u, lam) := u_dot0 . (u - u0) + lam_dot0 * (lam - lam0) - Δs = 0    (1 equation)
```

The second equation is the **arclength constraint**: it says "the projection of the
step `(u - u0, lam - lam0)` onto the known tangent direction equals `Δs`" — a
linear condition that pins down a unique point near the predictor, without ever
asking "what is `lam`" the way plain continuation does.

**Step 3 — Newton on the bordered system.** Linearizing `F` and `N` around the
current iterate gives the bordered (augmented) Jacobian system for the Newton
correction `(du, dlam)`:

```
[ J_u        J_lam  ] [ du   ]   [ -F ]
[ u_dot0^T   lam_dot0 ] [ dlam ] = [ -N ]
```

where `J_u = dF/du` is exactly the same Jacobian `(K - M_jac)` used throughout
B1/B2, and `J_lam = dF/dlam` is a length-`n` vector: differentiating `b_nl`'s
`lam*exp(u)` term with respect to `lam` gives `J_lam = -(weighted load vector with
weight exp(u_h) instead of lam*exp(u_h))` — i.e. exactly one more call to the
existing `weighted_integrals` machinery, just with a different weight field.

**The key property.** The plain Jacobian `J_u` alone is singular exactly at the
fold — that's the whole problem. But the **bordered** `(n+1)x(n+1)` matrix above is
generically *nonsingular* even there, because the extra tangent row/column
(`u_dot0`, `lam_dot0`) breaks the rank deficiency: at a simple fold, the null
vector of `J_u` is (to leading order) parallel to `u_dot0` itself, so the bordering
row picks up exactly the missing information `J_u` lost. This is *why* pseudo-
arclength continuation can walk straight through a fold that stops plain Newton
cold — and, if you keep stepping, it will naturally continue onto the **other**
solution branch (`lam` decreasing again) without any special-casing, because `s`
keeps increasing monotonically along the curve even where `lam` turns around.

**Step 4 — tangent update.** After the augmented Newton solve converges to a new
point `(u, lam)`, the tangent for the *next* predictor step is obtained by solving
one more linear system with the same bordered matrix (now evaluated at the new
point), forcing the tangent to have unit projection onto itself:

```
[ J_u        J_lam  ] [ u_dot ]   [ 0 ]
[ u_dot0^T   lam_dot0 ] [ lam_dot ] = [ 1 ]
```

then normalize `(u_dot, lam_dot)` to unit length (`||u_dot||^2 + lam_dot^2 = 1`),
choosing the sign that keeps `s` increasing consistently (i.e. that has positive
projection onto the previous tangent).

### 3. Interpretation

The whole construction is one idea, applied twice: **augment the unknowns with the
parameter, and augment the equations with a constraint that stays well-posed
exactly where the original system doesn't.** Everything numerically expensive in
this system (`J_u`, i.e. `K - M_jac`) is *already implemented* in B1/B2's
`weighted_integrals` machinery — the only genuinely new piece is `J_lam` (a call to
the same routine with a different weight field) and the one extra scalar row/column
for the arclength constraint. That's a much smaller lift than it might first sound,
which is exactly why the prompt frames it as "a week project," not "a rewrite."

### 4. The real learning — why this is in the tutorial

This explore task is deliberately a **derivation, not an implementation**, and
that's the point: the professor wants you to understand *why* the fix works before
you ever write a line of code for it — because the payoff of pseudo-arclength
continuation isn't "it's a different numerical trick," it's "it removes B2's
fragility (Explore (a)'s finding that the fold location itself needs mesh
convergence, and the base task's finding that naive step size determines whether
you *see* the slowdown at all) by making the method robust *by construction* near
the exact point where the naive method is guaranteed to fail." Seeing the algebra
first — the bordered matrix, the tangent, why it's nonsingular at the fold — means
that when you (or B3, or a future chapter) eventually implement it, you're
implementing something you already understand structurally, rather than
transcribing an unfamiliar formula. This is also a preview of a much bigger idea in
numerical continuation and bifurcation theory generally (used constantly in fluid
stability, structural buckling, chemical kinetics, etc.): **any time your method's
natural parametrization goes singular, look for a different parametrization of the
*same underlying object* (here, arclength instead of lam) under which it doesn't.**

---

## Explore (c) — Performance: weighted-integrals vs. solve, and a FLOP estimate

### 1. Problem statement

The prompt asserts that "the weighted-integrals host assembly is now the
per-iteration bottleneck" and asks you to measure it, at level 4, against the
`splu` linear solve — then go one level deeper: estimate the FLOP count of the
specific einsum used to build the weighted mass matrix
(`"qa,qb,eq,q,e->eab"`), compare the achieved GFLOP/s against the machine's
practical peak, and explain the gap (hint: "memory-bound einsum").

### 2. Results

Measured mid-continuation at `lam=8`, level 4 (`n_free=4,913`, `nbf=8`, `nqp=8`,
`4,096` elements):

| stage | per-iteration time |
|---|---:|
| `gp_values` (interpolate `u_h` to Gauss points) | ~0.0005s |
| `weighted_integrals` (both calls, combined) | ~0.028s |
| `splu` solve (factorize + backsolve) | ~0.09s |

**FLOP estimate for the `Me` einsum:** counting 3 multiplies + 1 running-sum add
per `(e, a, b, q)` term, plus one final per-`(e,a,b)` scale by the element
Jacobian, gives:

```
FLOPs = n_elems * nbf^2 * nqp * 4  +  n_elems * nbf^2
      = 4096 * 64 * 8 * 4  +  4096 * 64
      = 8,650,752 FLOPs per call
```

Measured time per call: `0.0143s` -> **achieved ~0.61 GFLOP/s**.
Practical peak on this machine, measured via a `2000x2000` BLAS `dgemm`
(`numpy`'s `@`): **~77.4 GFLOP/s**.

**Efficiency: ~0.78% of practical peak** — over 100x below what this machine's own
BLAS library achieves on a same-sized dense matrix multiply.

### 3. Interpretation

Two separate, both-real findings came out of this, and they don't contradict each
other:

**Finding 1 — the prompt's premise doesn't hold at level 4.** `splu` solve
(~0.09s) is about **3x more expensive** than the weighted-integrals assembly
(~0.028s) here, not the other way around. The assembly may well have been the
bottleneck at a *smaller* level (where Python/einsum call overhead is a bigger
fraction of an already-tiny factorization), but at the specific level the prompt
asks about, the claim as stated is simply wrong — worth catching rather than just
reporting the number the prompt implied.

**Finding 2 — a genuine, easy-to-fix redundancy, independent of Finding 1.** For
the physical continuation path (`f_fn=None`), the code computes `w_rhs = w_exp`
(no separate forcing term), which means `Bratu3D.newton`'s two
`weighted_integrals(...)` calls per iteration — one for `b_nl`, one for `M_jac` —
are called with **identical inputs**. Checked directly: the two calls' outputs are
bit-for-bit identical (`np.allclose` true; sparse difference has zero nonzeros).
Half of every `weighted_integrals` call in this code path is pure waste — an easy
2x win with zero algorithmic change, just reusing the first call's discarded
matrix output instead of recomputing it.

**Finding 3 — the FLOP-efficiency question, answered as asked.** At `<1%` of this
machine's practical BLAS peak, the einsum is unambiguously **memory-bound**. The
reason is structural: this computation is a *batch* of 4,096 independent, tiny
(`8x8`) element matrices — not one large matrix product. A single big matrix
multiply achieves high efficiency because it reuses each loaded value many times
(classic `O(n^3)` compute over `O(n^2)` data — high "arithmetic intensity"). This
einsum's batch index `e` is *not* contracted (it survives to the output), so numpy
cannot lower the whole operation onto one efficient BLAS call; it falls back to a
generic, unblocked loop that streams the large per-element arrays
(`w_gp[e,q]` in, `Me[e,a,b]` out) through memory doing barely more than one FLOP
per byte moved — while the small, reusable basis tables (`N[q,a]`) are the only
thing with any real cache locality. That imbalance — huge memory traffic, tiny
compute per byte — is the textbook definition of a memory-bound kernel, and it's
exactly why the achieved rate sits nowhere near the compute-bound BLAS peak.

### 4. The real learning — why this is in the tutorial

This explore task is teaching **rigor in the face of a plausible-sounding
performance claim** — the same discipline shown throughout this whole tutorial
series (A5's stale "constraints dominate" finding, A6's "verify order 2" claim
that didn't hold): *always measure before trusting a stated bottleneck*, even one
written by the person who built the code. But it's also teaching something
sharper and more specific: **not all slowness is the same kind of slowness.** A
sparse `splu` factorization being slow is (largely) an *inherent algorithmic* cost
(fill-in, as A5's whole chapter was about). This einsum being slow is an
*implementation-pattern* cost — the underlying math (a batch of tiny weighted Gram
matrices) is cheap, but the way it's expressed (a generic `numpy.einsum` call with
a non-contracted batch index) prevents the hardware from ever getting close to its
own peak. Recognizing that distinction — "is this fundamentally expensive, or just
expressed in a way that stops the hardware from doing its job?" — is precisely the
diagnostic skill that later motivates B1 Explore (d)'s GPU-kernel sketch: the
`*_var`-style closure-hook kernels exist in this codebase *because* someone already
made exactly this observation about this exact pattern, and fixed it structurally
by moving the batched per-element work onto the GPU, where thousands of small,
independent element computations are precisely what the hardware is built to do
well in parallel — the opposite of what a single CPU thread's generic einsum loop
is good at.

---

## Key Takeaways — B2, in plain terms

1. **The Newton machinery from B1 needed zero changes to work in 3-D** — same
   residual, same Jacobian pattern, same code, just `dim=3`. Confirms (again) that
   nonlinearity is a solver-loop concern, orthogonal to spatial dimension.

2. **A bifurcation's location is itself a mesh-dependent, convergent quantity.**
   The discrete fold `lam*_h` converged monotonically toward the literature value
   as the mesh refined (`10.36 -> 10.02 -> 9.93` vs. literature `9.9`) — just like
   an L2 error converges, except what's converging here is a *qualitative solver
   threshold*, not a solution value. A coarse mesh will confidently "succeed" past
   where a fine mesh (or the true continuum problem) would fail.

3. **Naive parameter continuation is fragile in a way that depends on luck.**
   Whether you *see* the Newton-iteration-count creep as you approach a fold
   depends entirely on how finely your parameter schedule happens to sample near
   it — B1's coarse steps saw a sharp cliff; B2's steps (by chance) landed inside
   the slowdown zone and saw the creep clearly. This fragility motivates pseudo-
   arclength continuation, whose whole design goal is to be robust regardless of
   step placement.

4. **Pseudo-arclength continuation is one idea (augment + reparametrize), not new
   physics.** Add the parameter to the unknowns, add one arclength constraint to
   the equations, and the bordered Jacobian stays invertible exactly where the
   plain one doesn't — because the tangent direction supplies exactly the
   information the singular Jacobian is missing. Nearly everything needed
   (`J_u`, the weighted-integral machinery) already exists in this codebase; only
   `J_lam` and the bordering row are new.

5. **Not every performance claim in a docstring is correct, and not every
   slowdown has the same cause.** The "weighted-integrals is the bottleneck" claim
   didn't hold at the level tested (solve dominated, 3x over); a real, separate,
   easily-fixed 2x redundancy was found instead; and the einsum's <1%-of-peak
   result showed that a mathematically cheap operation can still run 100x below
   hardware peak purely because of *how* it's expressed (batched-not-contracted,
   memory-bound) — a distinct problem from "this algorithm is inherently
   expensive," and one that maps directly onto why the codebase's GPU-kernel
   closure-hook pattern (from B1 Explore (d)) exists in the first place.
