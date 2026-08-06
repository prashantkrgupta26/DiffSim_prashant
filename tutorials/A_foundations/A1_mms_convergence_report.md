# A1 — MMS Convergence: Detailed Report

**Script:** `tutorials/A_foundations/A1_mms_convergence.py`

This report covers the base task and all three Explore tasks in A1. For each: the
problem statement, what was measured, what it means, and the underlying lesson the
tutorial is teaching. A "Key Takeaways" section closes it out. Results are drawn
from `ledger.md`; two small gaps (Explore (b) had no prior entry) were filled in
directly rather than left blank or guessed at.

**A note on which manufactured solution is "the" base result.** The docstring's
BACKGROUND section and its own EXPECTED RESULTS block describe the original MMS
setup, `u* = sin(pi x) sin(pi y)`. But the file's current module-level `u_star`/
`f_star` actually implement `u* = x^2 + y^2` — which is exactly Explore (a)'s
prescribed patch-test case. So the numbers recorded under "Observed Convergence
Rates" in the ledger are really Explore (a)'s answer, running as the file's
default. This report treats that transparently: the base task is described using
the general MMS-discipline concept (which applies to either choice of `u*`), and
Explore (a) below points back to those same already-recorded numbers rather than
re-deriving them.

---

## Base Task — The MMS discipline

### 1. Problem statement

Solve `-lap(u) = f` on the unit square with strong Dirichlet boundary conditions
(`u = g` on the boundary, imposed as identity rows in the linear system). Rather
than solving a problem with an unknown answer and hoping the solver is right, the
**Method of Manufactured Solutions (MMS)** flips the process: *pick* the answer
`u*` first, algebraically derive the `f` and `g` that make `u*` the exact solution
(`f = -lap(u*)`, `g = u*` on the boundary), then measure how fast the discrete
solution `u_h` approaches `u*` as the mesh is refined. Finite element theory
predicts a specific convergence *order* in the L2 norm: `O(h^{p+1})` — order 2 for
linear (p1) elements, order 3 for quadratic (p2) elements. The task is to confirm
this numerically, not just cite the theorem.

### 2. Results

As actually run (`u* = x^2 + y^2`, the file's current default — see the note
above):

```
p=1: errors [1.302e-03, 3.255e-04, 8.138e-05]   orders 2.00, 2.00
p=2: errors [1.985e-15, 5.943e-15, 2.943e-14]   orders (machine-precision noise, not a real "order")
```

### 3. Interpretation

p1 gives a clean, textbook order-2 result. p2's errors are all at the ~1e-15
level — machine precision, not a meaningfully decreasing sequence — because
`x^2 + y^2` happens to lie **exactly** inside the space of functions a p2
(biquadratic) element can represent exactly (see Explore (a) for why). This is
covered in detail below since it's the direct subject of Explore (a); the point
here is just that the numbers match theory cleanly either way you read them.

### 4. The real learning — why this is in the tutorial

This is the methodological foundation for *every subsequent tutorial in this
entire series*. Without MMS, "does my solver work?" has no crisp answer — you'd be
staring at a solution field with no ground truth to compare against. MMS converts
correctness into a falsifiable, quantitative claim: *the error must shrink at a
specific, predicted rate, or something is wrong.* The professor is teaching you a
discipline, not just a technique — "you distrust any solver claim that isn't a
measured order" (the docstring's own words). This single habit is what let every
later tutorial in this series (A2 through A6, B1, B2) catch real, sometimes
nontrivial bugs (A4's mislocated p2 region) or stale assumptions (A5's outdated
"constraints dominate" claim) — because each one, first and foremost, checked a
measured order against a predicted one.

---

## Explore (a) — The patch test (`u* = x^2 + y^2`)

### 1. Problem statement

Set `u* = x^2 + y^2` (so `f = -4`, a constant). At p=2, the docstring predicts the
error should be **machine zero** at every mesh level — not just small, not just
converging quickly, but essentially exact regardless of `h`. Confirm this, and
explain *why*. Also: what does p1 give, and why is p1 still only second order
(not exact) for this same `u*`?

### 2. Results

(Same numbers as the base task above, since the file's default already implements
this case.)

```
p=1: errors [1.302e-03, 3.255e-04, 8.138e-05]   orders 2.00, 2.00
p=2: errors [1.985e-15, 5.943e-15, 2.943e-14]   ~ machine precision at every level
```

### 3. Interpretation

**Why p2 is exact:** this is a **patch test**. A p2 (biquadratic, tensor-product
Lagrange) element's local basis spans every polynomial of the form
`sum_{i,j<=2} c_ij x^i y^j` — which *includes* `x^2` and `y^2` individually. Since
the true solution `u* = x^2 + y^2` is itself a member of that exact function space,
the finite element method doesn't have to *approximate* it at all — the best
approximation in that space *is* the exact function, for every mesh, at every
level. The tiny residual (`~1e-15`) is pure floating-point roundoff, not
discretization error; that's why the "order" computed from these numbers is noise,
not a real convergence rate.

**Why p1 is still only second order, not exact:** a p1 (bilinear) element's local
basis is `a + bx + cy + dxy` — it has no `x^2` or `y^2` terms at all. `u* = x^2 +
y^2` is *not* in that space, so p1 can only ever produce the *best bilinear
approximation* to it on each element, not the function itself. That approximation
error is exactly what standard finite element theory predicts: `O(h^{p+1}) =
O(h^2)` for p1 — which is precisely the order-2 result measured. The "patch test"
name comes from exactly this contrast: it's a test that isolates whether a given
element's polynomial space is rich enough to reproduce a target field exactly, with
zero contamination from any other discretization effect.

### 4. The real learning — why this is in the tutorial

**A patch test is a sharper, more diagnostic MMS than a generic one.** A generic
"pick some smooth `u*`" test (like `sin(pi x) sin(pi y)`) confirms the *rate* is
right, but a patch test additionally confirms *exactness where exactness is
theoretically guaranteed* — a much stronger, more specific claim, and a classic
verification-and-validation technique used throughout computational mechanics (the
patch test is a named, standard part of the FEM literature, not something specific
to this codebase). If your p2 basis were subtly wrong — a missing term, an
off-by-one in the local-to-global node mapping, an incorrectly integrated
quadrature rule for polynomial degree 2 — this exact test would catch it
immediately as a *nonzero* floor, rather than requiring you to notice a slightly
wrong order-3 (instead of order-2.99) exponent buried in three significant
figures. The professor is teaching you that verification has *levels* — "does the
order match" is good, but "does this specific, theoretically-exact case actually
come out exact" is a strictly stronger and more targeted check, worth having in
your toolkit alongside the general MMS habit from the base task.

---

## Explore (b) — Breaking the MMS on purpose (`g = 0` everywhere)

### 1. Problem statement

Keep the correct `f`, but set the boundary data `g = 0` everywhere instead of the
correct `g = u*`. What convergence order do you measure now, and what does that
tell you about how boundary errors pollute interior accuracy?

### 2. Results

No prior ledger entry existed for this task; ran it directly (two variants, since
the choice of `u*` turns out to matter — see interpretation):

```
u* = sin(pi x) sin(pi y), f correct, g=0 (nominally "wrong"):
  errors [1.606e-03, 4.015e-04, 1.004e-04]   orders 2.00, 2.00
  -- IDENTICAL to the correct-g control run.

u* = x^2 + y^2 (the file's current default), f correct, g=0 (genuinely wrong):
  errors [9.045e-01, 9.051e-01, 9.052e-01]   orders -0.001, -0.000
  -- error plateaus around 0.905, does not shrink at all under refinement.
```

### 3. Interpretation

The two results tell different halves of the same story. With `u* = x^2 + y^2`
(nonzero on most of the boundary — it ranges from 0 at the origin corner up to 2 at
the far corner), setting `g=0` is a genuinely wrong boundary condition, and the
result is exactly what the task expects: the error **does not decrease at all**
under mesh refinement (order effectively `0`), plateauing at a large, roughly
constant value (`~0.905`). No amount of interior mesh refinement can fix a wrong
boundary condition — the boundary error simply doesn't have anywhere else to go.

But the `sin(pi x) sin(pi y)` run reveals something the task doesn't warn you
about: that particular `u*` is *already exactly zero* on all four sides of the
unit square (`sin(0)=sin(pi)=0`), so setting `g=0` there isn't actually breaking
anything — it's still exactly correct. The order stays a perfect 2.00, identical
to the control run to the last printed digit, because nothing was ever actually
wrong.

### 4. The real learning — why this is in the tutorial

**Boundary-condition errors are non-local and non-vanishing** — this is the
headline lesson, and it recurs (explicitly) in A2's Explore (a): a wrong `g` (or,
in A2's case, a silently-wrong natural/Neumann condition) doesn't get "washed out"
by refining the interior mesh, because the discrete system couples every DOF
together through the stiffness matrix; a persistently wrong constraint at the
boundary propagates its error into the whole domain at every mesh size. This is
one of the most important debugging instincts in FEM: **if refining your mesh
doesn't reduce your error, suspect the data (boundary conditions, forcing,
material properties), not the discretization.**

But the accidental `sin*sin` non-result is arguably the *more valuable* lesson
here, precisely because it wasn't the one asked for: **a test is only as good as
its ability to actually detect the failure mode it's designed to catch.** If you'd
run only the `sin*sin` case (matching the docstring's own default background
problem) and stopped there, you would have concluded — wrongly — that this whole
class of bug ("wrong boundary data") is harmless, purely because of an unlucky
(or, from a test-design perspective, poorly chosen) coincidence between the
manufactured solution's symmetry and the specific mistake being tested. This is
exactly the kind of thing a careful verification engineer has to watch for:
picking test cases that don't just exercise the code path, but are actually
*sensitive* to the specific error you're trying to catch.

---

## Explore (c) — Performance corner: staged cost timing

### 1. Problem statement

Time the three pipeline stages — mesh+constraints, assembly, and the sparse
solve — separately, at levels 4 through 8, for p1. Which stage grows fastest?
Fit the exponent of each stage from consecutive time ratios. Predict *before*
measuring: the reference document (`FEM_computational_cost.pdf`) suggests `splu`
on a 2-D mesh should scale close to `O(n^1.5)`, and assembly close to `O(n)`.

### 2. Results

Two runs are recorded: an initial CPU-only run (no CUDA driver visible in that
environment) and a GPU-visible rerun.

**CPU-only run:**

| Level | DOFs (n) | Mesh+constraints (s) | Assembly (s) | Solve (s) |
|---:|---:|---:|---:|---:|
| 4 | 289 | 0.0009 | 0.0010 | 0.0004 |
| 5 | 1,089 | 0.0009 | 0.0022 | 0.0019 |
| 6 | 4,225 | 0.0014 | 0.0082 | 0.0083 |
| 7 | 16,641 | 0.0055 | 0.0244 | 0.0414 |
| 8 | 66,049 | 0.0105 | 0.1451 | 0.2428 |

Measured exponents: mesh `0.06, 0.33, 0.97, 0.47` (noisy — fixed overhead
dominates at small sizes); assembly `0.56, 0.97, 0.80, 1.29` (~linear); solve
`1.19, 1.08, 1.17, 1.28` (superlinear, trending toward the predicted `O(n^1.5)`).

**GPU-visible rerun** (`cuda:0`, RTX 2000 Ada): essentially the same pattern —
solve remains the largest stage at level 8 (`0.2433s`), assembly next (`0.1501s`),
mesh cheapest. Note: the `splu` solve itself is still CPU-based in this pipeline
even when Warp kernels run on GPU, so these solve timings are not GPU numbers
either way.

### 3. Interpretation

The prediction holds up: assembly's measured exponent (~0.8-1.3, centering near 1)
is consistent with the predicted `O(n)`, and solve's exponent (~1.1-1.3, trending
up toward `1.5` at the largest, most reliable levels) is consistent with the
predicted `O(n^1.5)` 2-D sparse-factorization scaling. By level 8, solve is the
clear bottleneck (`0.24s`, roughly 1.6x assembly's `0.15s`), and that gap will only
widen at larger levels, since solve's exponent is measured *higher* than
assembly's. Small-level exponents (level 4->5, 5->6) are noisy for all three
stages — at these tiny DOF counts, fixed Python/dispatch overhead dominates
whatever the "true" asymptotic scaling is, a recurring theme across this whole
tutorial series (the same launch-floor effect shows up explicitly in later A5/A6
timing work).

### 4. The real learning — why this is in the tutorial

This is your **first controlled introduction to the cost model** that the rest of
the performance-focused tutorials (P1, and the "Performance Corner" sections
threaded through A4-A6 and B1-B2) build on. The specific numbers matter less than
the *method*: state a prediction from theory first, then measure, then compare —
exactly the MMS discipline from the base task, applied to *performance* instead of
*accuracy*. The professor is also quietly pre-loading a fact you'll need later: the
direct sparse solver is the fastest-growing cost even in 2-D, and its growth rate
only gets worse in 3-D (as A5 measures explicitly, finding an even steeper
exponent and an eventual out-of-memory failure) — this A1 measurement is the
*baseline* against which that later, much more dramatic 3-D result is meant to be
read and understood.

---

## Key Takeaways — A1, in plain terms

1. **Manufactured solutions turn "does it work?" into a number you can check.**
   Pick the answer, derive the data that makes it exact, measure the convergence
   rate. Every later tutorial in this series builds on this one habit.

2. **A patch test is MMS's sharpest tool.** Choosing a `u*` that lies exactly
   inside your basis's polynomial space (like `x^2+y^2` for p2 elements) turns
   "close to the right order" into "exactly right, at every mesh size" — a much
   stronger, more specific check that catches subtle basis/quadrature bugs a
   generic smooth `u*` might miss.

3. **Boundary-condition errors don't wash out under refinement — and your test
   case must actually be *sensitive* to the mistake you're testing for.** A wrong
   `g` pollutes the whole domain regardless of mesh size (confirmed with `x^2+y^2`)
   — but the same "wrong" `g=0` was silently harmless for `sin(pi x)sin(pi y)`,
   because that field already vanishes on the boundary. Test design matters as
   much as the test itself.

4. **State the predicted cost scaling before you measure it.** A1's performance
   corner isn't really about the specific numbers — it's the first rehearsal of
   "predict, then measure, then compare," the same discipline as MMS itself,
   applied to timing instead of accuracy. It sets the 2-D baseline (`~O(n^1.5)`
   solve) that A5's 3-D result is later measured against.

5. **The meta-lesson threading through this whole tutorial:** distrust any claim —
   about accuracy *or* performance — that isn't backed by a fresh measurement.
   That single habit is what caught every interesting finding in the tutorials
   that follow this one.
