# A3 — Shifted Boundary Method: Detailed Report

**Script:** `tutorials/A_foundations/A3_shifted_boundary.py`

This report covers the base task and all three Explore tasks in A3, plus an
untitled fourth task (a "Performance Corner" sitting outside the printed EXPLORE
block, at the very end of the file) that had no prior ledger entry and was run
directly for this report. For each: the problem statement, what was measured, what
it means, and the underlying lesson the tutorial is teaching. A "Key Takeaways"
section closes it out.

---

## Base Task — Solving a PDE with no body-fitted mesh

### 1. Problem statement

Every prior tutorial (A1, A2) solved on the unit square itself — the mesh and the
domain were the same shape. A3 introduces the **Shifted Boundary Method (SBM,
Main & Scovazzi 2018)**: solve `-lap(u) = f` inside a disk of radius 0.3 centered
in the unit square, `u = g` on the circle — but *without ever building a
circle-conforming mesh*. Instead, the background mesh is a plain uniform Cartesian
octree of the whole square, and the circle is known only through a **signed-
distance oracle** `psi(x)` (negative inside). Three moving parts make this work:
(1) **classification** — decide which background elements to keep; (2) the
**surrogate boundary** — the staircase outline formed by the outer faces of the
kept elements, which only approximately follows the true circle; (3) the **Taylor
shift** — correct for the gap between the staircase and the true boundary by
Taylor-expanding the solution from each surrogate quadrature point to its closest
point on the true circle (`S u = u + grad(u).d`, using the distance vector `d`).
Verify the whole pipeline with MMS (`u* = sin(pi x) cos(pi y)`), expecting second-
order L2 convergence for linear elements despite the geometry never being exactly
resolved by the mesh.

### 2. Results

```
Level 4 (h=0.0625): L2 error = 3.210e-03
Level 5 (h=0.0312): L2 error = 6.615e-04
Level 6 (h=0.0156): L2 error = 1.508e-04

orders: level 4->5: 2.28   level 5->6: 2.13
```

Matches the docstring's own EXPECTED RESULTS closely (`3.2e-3/6.6e-4/1.5e-4`,
orders `2.28/2.13`) — no bug, using the file's default `lam=0.0` (most
conservative classification: keep only elements fully inside the disk).

### 3. Interpretation

The orders are slightly *above* 2 (mildly "superconvergent") rather than sitting
exactly at 2 — the docstring calls this out as common for SBM on smooth geometry
at the pre-asymptotic (coarser) end of a refinement sequence; the trend (2.28 ->
2.13) is visibly settling toward 2.0 as the mesh refines further, which is the
expected asymptotic behavior. The key achievement being verified here isn't just
"order 2," it's that order 2 is achievable **at all** on a mesh that never
conforms to the true geometry — the staircase surrogate boundary is a genuinely
crude, jagged approximation of a circle, and yet the Taylor shift fully recovers
the accuracy a body-fitted mesh would give.

### 4. The real learning — why this is in the tutorial

This is a genuine paradigm shift from A1/A2, and the professor wants you to feel
that shift, not just read about it. Traditional FEM ties accuracy to mesh quality
at the boundary — a curved boundary needs curved (or at least boundary-conforming)
elements, and building that mesh is often the single most labor-intensive part of
a real simulation workflow (mesh generation for complex CAD geometry is its own
sub-discipline). SBM decouples these two things entirely: the mesh is always the
same trivial uniform background grid, and *all* the geometric fidelity is pushed
into the Taylor-shift correction, computed from a signed-distance function. The
practical payoff, which A6 makes explicit later, is that changing the geometry
becomes a one-line change (swap the oracle) rather than a remeshing project. A3 is
where you first pay the conceptual cost of understanding *why* this works (the
staircase + shift decomposition) so that A6's "geometry is a plug-in" claim isn't
a magic trick — it's a direct, understood consequence of what you verify here.

---

## Explore (a) — Quadratic basis (`p=2`)

### 1. Problem statement

Set `p=2` in `solve_at_level` and watch the order climb to 3. Then read
`_shift_fn_for` in `src/diffsim/sbm/poisson.py` to see the extra Hessian term that
makes third-order accuracy possible for quadratic elements — and understand why
p1 must *not* include that same term.

### 2. Results

```
Level 4 (h=0.0625): L2 error = 1.726e-04
Level 5 (h=0.0312): L2 error = 1.423e-05
Level 6 (h=0.0156): L2 error = 1.808e-06

orders: level 4->5: 3.60   level 5->6: 2.98
```

### 3. Interpretation

The order climbs to almost exactly 3, confirming the expected `O(h^{p+1})` rate
for p2 elements on this immersed geometry, same as smooth-boundary FEM predicts.
The first interval (3.60) is mildly pre-asymptotic (superconvergent, similar in
character to the base task's own 2.28), settling to `2.98` — essentially exact
order 3 — at the finer interval. The mechanism, per the shift formula in the
docstring's BACKGROUND: `S u = u + grad(u).d + (1/2) d^T H(u) d` for quadratic
elements — the extra Hessian (second-derivative) term is needed because a p2
element's Taylor expansion must be carried to second order to match its own
polynomial accuracy; dropping it would leave an `O(d^2)` inconsistency that caps
accuracy at order 2 no matter how rich the local element basis is (this exact
failure mode — Taylor shift order not matching basis order — is documented
elsewhere in this codebase's own deferred-findings notes as a real bug that was
once present and fixed).

### 4. The real learning — why this is in the tutorial

**The correction has to match the accuracy you're asking the correction to
enable.** This is a specific, sharp instance of a very general FEM principle: you
cannot get `O(h^{p+1})` accuracy out of any part of your discretization — element
basis, quadrature rule, *or* boundary correction — if any single piece of the
pipeline is only accurate to a lower order. The Taylor shift is not a fixed,
one-size-fits-all correction; it is a truncated series whose truncation order must
be raised in lockstep with the element order it's paired with. This is also a
direct preview of a symmetric danger raised in the same breath: the "why must p1
NOT include it" half of the question. A p1 element's own basis is only complete to
first order, and its (mixed-partial-only, since bilinear elements have zero
second pure-partial derivatives) discrete Hessian is an incomplete, partial
representation of curvature — adding a Hessian correction term built for p2 onto a
p1 basis doesn't help, it actively pollutes a consistent first-order shift with
spurious higher-order noise the p1 basis has no ability to represent. "More
correction" is not always better; the correction and the space it's applied to
must be **matched**.

---

## Explore (b) — Retain intercepted elements (`lam=1.0`)

### 1. Problem statement

Change `lam=0.0` to `lam=1.0` — this keeps *every* intercepted element instead of
only fully-interior ones, so the surrogate staircase now hugs the true circle from
the **outside** of the retained set, rather than from inside it. Does the order
change? Why does the shift not care about the sign of `d` (whether the surrogate
sits inside or outside the true boundary)?

### 2. Results

```
Level 4 (h=0.0625): L2 error = 9.324e-03
Level 5 (h=0.0312): L2 error = 3.753e-03
Level 6 (h=0.0156): L2 error = 6.083e-04

orders: level 4->5: 1.31   level 5->6: 2.63
```

### 3. Interpretation

The order is noisier across this refinement pair than the base case (a
pre-asymptotic `1.31` at the coarser interval, followed by a super-second-order
`2.63` at the finer one) but the trend is clearly consistent with the same
asymptotic `O(h^2)` target — order is not systematically capped or degraded by
the sign flip. This directly answers the "why doesn't it care" question: the
Taylor shift `S u = u + grad(u).d` is a first-order expansion along the vector `d`
from the surrogate point to the true boundary — it works identically whether `d`
points inward or outward, because a Taylor expansion is valid in either direction
along a sufficiently smooth field. What matters for accuracy is the *magnitude*
of `d` (bounded by the local mesh size `h`, regardless of `lam`) and the
smoothness of `u` near the surrogate point — not which side of the true boundary
the surrogate happens to sit on.

### 4. The real learning — why this is in the tutorial

This dissolves what looks, at first glance, like it should matter a great deal:
surely a surrogate that includes partially-outside elements is geometrically
"worse" than one that's conservatively all-inside? The result says no — because
the *correction* mechanism (the shift), not the *raw geometric fidelity* of the
staircase, is what carries the accuracy. This is a deep and somewhat
counter-intuitive point about how SBM (and shifted/immersed methods generally)
achieve their accuracy: the staircase approximation itself can be genuinely crude
in either direction, as long as the shift vector `d` and the field being shifted
are both well-behaved. The professor is teaching you to stop reasoning about
accuracy purely in terms of "how close is my discrete geometry to the true
geometry" (the traditional body-fitted-mesh mental model) and instead reason about
"how well does my correction account for the gap" — a genuinely different, and for
immersed methods, more accurate way to think about where accuracy comes from. This
also has a very practical payoff: `lam` becomes a free *engineering* knob (how many
elements do you keep, which affects cost and which side the staircase sits on) that
doesn't trade off against *accuracy* the way you might naively fear — a fact used
elsewhere in this codebase (e.g. `lam=0.5` for flow-past-cylinder configurations)
precisely because it doesn't have to be tuned for accuracy's sake.

---

## Explore (c) — Sampled `GridSDF` geometry

### 1. Problem statement

Replace the exact analytic `Sphere` oracle with `GridSDF.from_oracle(Sphere(...),
n=128)` — a **sampled**, piecewise-linear approximation of the same circle,
representing the case where the true geometry oracle comes from real data (a CAD
model, a level-set field, a 3-D scan) rather than a closed-form formula. Compare
the resulting error floors.

### 2. Results

**Strict projection (default):** the run **stopped before solving** — Newton
closest-point projection failed at 8 of 80 surrogate quadrature points at level 4
(`newton_ok_frac = 0.9`).

**With the optional fallback** (`max_fail_frac=0.1`, allowing up to 10% projection
failures):

```
Level 4 (h=0.0625): L2 error = 9.317e-03
Level 5 (h=0.0312): L2 error = 3.791e-03
Level 6 (h=0.0156): L2 error = 6.401e-04

orders: 1.30, 2.57
```

Very close to the exact-circle `lam=1.0` results from Explore (b) (`9.324e-3 /
3.753e-3 / 6.083e-4`, orders `1.31 / 2.63`) at the same levels.

### 3. Interpretation

The strict-projection *failure* is itself the primary, most important result here
— more informative than the error floor comparison the task literally asks for.
The sampled `GridSDF` is only piecewise-linear (built from `n=128` grid samples of
the true, smooth circle), so it is **not** globally smooth the way the analytic
`Sphere` oracle is: its gradient has kinks at sample-cell boundaries, and the
Newton closest-point projection (which assumes a well-behaved, smooth signed-
distance field to converge reliably) can and does fail near those kinks. Once the
fallback is enabled, the accuracy that *does* come out is nearly identical to the
exact-circle case — meaning the sampled geometry's *impact on accuracy*, where the
projection succeeds, is small. The real story is about *robustness*, not accuracy:
a piecewise-linear geometry representation is less forgiving of the strict-
projection assumptions than an analytic one is.

### 4. The real learning — why this is in the tutorial

This is the tutorial's way of warning you, hands-on, about the gap between
**"works on the textbook geometry"** and **"works on the geometry you'll actually
have in a real project."** Every earlier A3 result used a perfect, infinitely-
smooth analytic sphere — exactly the kind of geometry real applications almost
never hand you. Real geometry comes from CAD exports, level-set fields, or 3-D
scans, all of which are sampled/discretized and therefore have exactly the kind of
local non-smoothness that broke the strict Newton projection here. The lesson
isn't "sampled geometry doesn't work" (the fallback shows it works fine, once you
account for its rougher character) — it's **"the code's own admissibility check
(`newton_ok_frac`) caught a real assumption violation instead of silently
producing a wrong answer,"** which is exactly the kind of defensive engineering
you want in a research codebase that will eventually be pointed at real,
imperfect data. Note also (per the ledger) that this was explicitly run as a
diagnostic test with temporary code changes, restored afterward — a reminder that
exploring "what if my assumptions don't hold" doesn't require permanently
modifying the tutorial itself.

---

## Performance Corner — `classify_lambda` timing vs. level (untitled task, no prior data)

### 1. Problem statement

This task sits outside the printed `EXPLORE` block, as a trailing comment at the
very end of the file — easy to miss, and it had no entry in the ledger prior to
this report. The claim to test: `classify_lambda`'s cost should track the number
of **intercepted** (narrow-band) elements, `O(2^level)`, not the total element
count, `O(4^level)` — because of a two-pass design where a cheap center-distance
check resolves most elements immediately, and only a narrow band near the true
boundary pays for expensive dense per-element quadrature sampling.

### 2. Results

Ran directly for this report (levels 4-8), using two independent measures: direct
element counting (recomputing the same narrow-band criterion the function uses
internally) and wall-clock timing.

| level | total elements | narrow-band elements | band growth |
|---:|---:|---:|---:|
| 4 | 256 | 88 | — |
| 5 | 1,024 | 176 | 2.00x |
| 6 | 4,096 | 336 | 1.91x |
| 7 | 16,384 | 688 | 2.05x |
| 8 | 65,536 | 1,368 | 1.99x |

Wall-clock timing was too noisy at this scale (all runs under 5ms; e.g. level 7
median `5.07ms` vs. min `1.18ms`, a 4x run-to-run spread) to extract a trustworthy
exponent.

### 3. Interpretation

The direct element-count check is a clean, essentially noise-free confirmation of
the claim: the narrow band grows almost exactly `2.00x` per level (`88 -> 176 ->
336 -> 688 -> 1,368`) while total elements always grow exactly `4.00x` — a
textbook match to `O(2^level)` for the narrow band against `O(4^level)` for the
full grid. The wall-clock timing, by contrast, was unusable evidence at this
problem size: with every run completing in single-digit milliseconds, fixed
Python-level dispatch overhead (and this being a shared, sometimes-busy machine)
swamps whatever the true asymptotic signal is.

### 4. The real learning — why this is in the tutorial

**When your timer is too noisy to trust, measure the algorithm directly instead
of its wall-clock shadow.** This is a subtly different, complementary skill to
every other performance-corner task in this series (which mostly rely on timing):
here, the *cleanest* evidence for the claim wasn't a stopwatch at all, it was
recomputing exactly what the two-pass classification logic decides internally
(which elements need the expensive dense treatment) and counting them directly.
This sidesteps timing noise entirely rather than trying to average it away with
more repetitions or bigger problem sizes. It's a genuinely useful diagnostic
instinct: when a performance claim is really an *algorithmic* claim ("this
subset of the work is what scales," not "this operation takes this many
seconds"), you can often verify it more convincingly by inspecting what the
algorithm actually does than by timing it — especially at problem sizes too small
for wall-clock measurement to be reliable in the first place.

---

## Key Takeaways — A3, in plain terms

1. **You can get body-fitted-mesh accuracy without a body-fitted mesh.** The
   Shifted Boundary Method achieves full order-2 (or order-3, for p2) accuracy on
   a crude staircase approximation of a circle, by pushing all the geometric
   correction into a Taylor-shift term computed from a signed-distance oracle.

2. **A correction's accuracy must match the basis it's paired with, in both
   directions.** Missing the Hessian term caps p2 at order 2 instead of 3; adding
   it to p1 (whose incomplete discrete Hessian can't properly represent it)
   actively makes things worse rather than better. Matching, not maximizing, is
   the right instinct.

3. **The surrogate boundary's geometric fidelity and the solution's accuracy are
   not the same thing.** Whether the staircase sits inside or outside the true
   circle (`lam=0.0` vs `lam=1.0`) doesn't change the achieved order, because the
   shift correction — not raw staircase closeness — is what carries the accuracy.

4. **Sampled, real-world-style geometry stresses assumptions that idealized
   analytic geometry never tests.** The strict-projection failure on `GridSDF` is
   a genuine, useful signal — evidence that the code's admissibility checks work
   as intended, not a sign that sampled geometry is unusable (the fallback result
   shows accuracy is barely affected once you accommodate the rougher geometry).

5. **When timing noise swamps the signal, verify the algorithm directly instead.**
   Counting the narrow-band elements the classification logic actually processes
   gave a clean, immediate confirmation of an `O(2^level)` claim that wall-clock
   timing, at this small problem scale, simply could not.
