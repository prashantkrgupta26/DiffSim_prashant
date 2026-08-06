# A2 — Boundary Conditions: Detailed Report

**Script:** `tutorials/A_foundations/A2_boundary_conditions.py`

This report covers the base task and all three Explore tasks in A2. For each: the
problem statement, what was measured, what it means, and the underlying lesson the
tutorial is teaching. A "Key Takeaways" section closes it out.

**A note on the base case, mirroring A1.** The docstring's BACKGROUND describes
`u* = cos(pi x) cos(pi y)`, whose normal derivative is exactly zero on all four
sides of the unit square — the choice that makes "Dirichlet on x-faces only,
nothing on y-faces" a *legitimate* boundary condition (the omitted term really is
zero). But the file's current module-level `u_star` implements `sin(pi x) sin(pi
y)` instead — Explore (a)'s prescribed change, already applied as the default. So,
exactly as in A1, what's recorded as the "base" result in the ledger is really
Explore (a)'s answer. This report is transparent about that rather than presenting
numbers under the wrong label.

---

## Base Task — Where boundary conditions actually live in the discrete system

### 1. Problem statement

Starting from the weak form of `-lap(u) = f` (multiply by a test function `w`,
integrate by parts): `int grad(w).grad(u) dV - oint w (grad(u).n) dS = int w f
dV`. Whatever you do with that boundary surface integral *is* your boundary
condition. Two cases: (1) **Dirichlet** — pin `u` directly on a face (identity rows
in the linear system; the corresponding test functions vanish there so the surface
term drops out of the weak form by construction); (2) **Neumann** — leave a face
alone entirely, which the weak form silently interprets as `grad(u).n = 0`. The
task: verify with MMS that omitting the surface term on a face where the *true*
flux really is zero still gives full order-2 accuracy — and understand that
"doing nothing" is itself a specific, meaningful boundary condition, not an
absence of one.

### 2. Results

As actually run (`u* = sin(pi x) sin(pi y)`, per the note above):

```
Dirichlet    all: errors [1.606e-03, 4.015e-04, 1.004e-04]   orders 2.00, 2.00
Dirichlet x-only: errors [4.673e-01, 4.692e-01, 4.697e-01]   orders -0.01, -0.00
```

### 3. Interpretation

These numbers are actually Explore (a)'s result (see below) — the x-only case
*collapses* rather than converging, because `sin(pi x) sin(pi y)`'s normal
derivative is **not** zero on the y-faces (`d/dy = pi sin(pi x) cos(pi y)`, which
is nonzero except exactly at `y=0.5`). Leaving the y-faces alone silently imposes
zero flux there, which is *inconsistent* with this particular `u*`. For the
tutorial's intended baseline case, `cos(pi x) cos(pi y)` (whose flux genuinely is
zero everywhere), both the all-Dirichlet and x-only cases would converge at order
2 — that's the theoretical claim the docstring's EXPECTED RESULTS block describes,
though it was not the case actually re-run and recorded here (see the note at the
top of this report).

### 4. The real learning — why this is in the tutorial

**"Do nothing" is a choice, not a null option.** This is one of the most
consequential and easy-to-miss facts in weak-form FEM: a face with no explicitly
assembled boundary term does not mean "no boundary condition was applied" — it
means "zero-flux Neumann was applied, whether you intended it or not." Any code
that only assembles the volume (stiffness) term is making this choice on *every*
non-Dirichlet face, silently. This matters enormously in real applications: an
open outflow boundary, an unmodeled far-field, a face you simply forgot to
constrain — all of these get the *same* implicit treatment (zero flux) unless you
deliberately do something else. The professor is teaching you to read your own
discrete system precisely enough to know exactly what boundary condition every
face has, at all times, rather than treating "boundary conditions" as something
you only think about for faces you explicitly touch.

---

## Explore (a) — Break it: `sin(pi x) sin(pi y)`, x-only Dirichlet

### 1. Problem statement

Change `u*` to `sin(pi x) sin(pi y)` and rerun the x-only Dirichlet case *without
changing anything else*. The order should collapse. What flux is the do-nothing
(y-face) side now silently asserting, and what is this `u*`'s true flux there?

### 2. Results

(Same numbers as the base task, since the file's current default already
implements this case — this is the direct, already-answered result.)

```
all:    errors [1.606e-03, 4.015e-04, 1.004e-04]   orders 2.00, 2.00
x-only: errors [4.673e-01, 4.692e-01, 4.697e-01]   orders -0.01, -0.00 (collapsed)
```

### 3. Interpretation

The do-nothing y-faces silently assert `grad(u).n = 0` there. This `u*`'s true
flux on those faces is `d(u*)/dy = pi sin(pi x) cos(pi y)`, which is **not** zero
(it's zero only along the single line `y=0.5`, not on the boundary faces
`y=0,1`). So the discrete system is being asked to solve a problem with a boundary
condition that doesn't match the manufactured data — a fixed, `O(1)` inconsistency
that refining the mesh cannot fix, hence the flat, non-decreasing error (order
`~0`, plateauing near `0.47`). This is the same underlying mechanism as A1's
Explore (b) (a wrong `g` pollutes the whole domain without vanishing under
refinement) — here the wrongness lives in the *implicit* flux condition instead of
an explicit Dirichlet value, but the consequence is identical.

### 4. The real learning — why this is in the tutorial

This closes the loop the base task opened: it's one thing to be *told* that
omitting a term imposes zero flux; it's another to *watch* a solver silently fail
in a very specific, diagnosable way (a flat, `O(1)` error plateau, not a crash, not
even an obviously "wrong-looking" result) because of it. This is exactly the kind
of bug that's dangerous in real work: the solver runs, produces a smooth-looking
field, and gives no error message — the only way to know something is wrong is to
already be doing the MMS-order check from A1. The professor is fusing A1's
discipline ("always measure the order") with A2's conceptual content ("do-nothing
means zero-flux") into one concrete, memorable failure mode.

---

## Explore (b) — The pure-Neumann trap

### 1. Problem statement

Make *all* faces natural (no Dirichlet anywhere) — legal in principle for a `u*`
whose flux is genuinely zero everywhere (the intended `cos(pi x) cos(pi y)`
baseline). Pure-Neumann Poisson problems have a well-known pathology: the system
matrix is **singular**, because `u + constant` solves the same PDE with the same
(zero) flux everywhere — there's no way to pin down the additive constant from the
PDE alone. The fix used throughout track D's pressure solves: pin one node to a
known value, removing exactly the one degree of freedom the system can't
otherwise determine.

### 2. Results

Recorded in the ledger, but with an important caveat (see interpretation):

```
pin: errors [2.192e+01, 2.544e+01, 2.896e+01]   orders -0.21, -0.19
```

### 3. Interpretation

**This result does not validly test what Explore (b) is actually asking.** It was
run using the same `sin(pi x) sin(pi y)` `u*` as Explore (a) — which, as
established above, does **not** have zero flux on the y-faces. Making *all* faces
natural for this `u*` isn't just "the pinning trick tested on a hard case" — it's
an internally **inconsistent** problem. Pure-Neumann problems are only solvable at
all when the data satisfies a compatibility condition (`int_Omega f dV = oint_{d
Omega} q dS`, from the divergence theorem — net source must equal net boundary
flux); using `f` derived for `sin*sin` while the assembled system silently imposes
zero flux everywhere violates that condition outright. The large, *growing* error
under refinement (not just large — actively getting worse, order `-0.2`) is the
signature of exactly this kind of inconsistency, not a real test of whether the
pinning fix works. The pinning technique itself did do its mechanical job — the
matrix was no longer singular, and `splu` returned *a* solution rather than
crashing — but "did the solve complete" and "is the solve accurate" are different
questions, and only the first one was actually validated here. A meaningful test
would need to restore the true `cos(pi x) cos(pi y)` baseline (whose flux really
is zero on every face) and rerun all three cases; that rerun was not performed.

### 4. The real learning — why this is in the tutorial

**Not every "the code ran and gave numbers" outcome is a validated result** — and
this is worth learning as vividly as any successful result in this whole series.
Pure-Neumann singularity is a real, common trap (track D's own pressure solves hit
it constantly), and pinning one node is the standard, correct fix for the
*singularity*. But this particular test run happened to conflate two different
things: the well-posedness fix (pinning) and the data consistency requirement
(compatible `f` and flux). Getting numbers out of a solve is not the same as
having answered the question you set out to ask — a distinction that matters far
more in real research than in a classroom exercise, where a wrong-but-plausible-
looking number can otherwise sit unquestioned in a results table. Recognizing
*which* question a given run actually answers (and flagging clearly when it
doesn't) is itself the skill being exercised here, as much as the Neumann-pinning
mechanics.

---

## Explore (c) — Performance corner: Dirichlet row-replacement timing

### 1. Problem statement

The Dirichlet boundary condition is implemented as a **host-side Python loop**
over boundary nodes, replacing each corresponding matrix row with an identity row.
Time this loop at levels 5 through 8: when does it stop being "free" (negligible)?
(Flagged in the docstring as m1b findings item 4: production code replaces this
loop with a precomputed masked assembly instead.)

### 2. Results

| Level | DOFs (n) | Boundary nodes | Row-replacement time (s) |
|---:|---:|---:|---:|
| 5 | 1,089 | 128 | 0.000223 |
| 6 | 4,225 | 256 | 0.000458 |
| 7 | 16,641 | 512 | 0.000911 |
| 8 | 66,049 | 1,024 | 0.001792 |

Measured exponents (vs. total DOFs `n`): `0.53, 0.50, 0.49`.

### 3. Interpretation

The exponent is almost exactly `0.5` at every step — the row-replacement time
scales like `sqrt(n)`, which makes complete geometric sense: in 2-D, the number of
*boundary* nodes grows like the perimeter of the domain, `O(sqrt(n))`, while total
DOFs `n` grow like the *area*. The loop is genuinely cheap at every level tested
here (under 2 milliseconds even at 66k DOFs), so at this scale it truly is
"free" relative to assembly or solve. But the scaling law itself tells you the
limit of that free ride: this is a **host-side, per-node Python loop**, and even
though its complexity exponent (`0.5`) is favorable, a fixed per-iteration Python
overhead means it will eventually become the dominant *constant-factor* cost at
large enough boundary-node counts, purely from interpreter overhead, not
algorithmic complexity — which is exactly the m1b finding this docstring points
to.

### 4. The real learning — why this is in the tutorial

**A "cheap" operation and a "well-scaling" operation are not the same claim, and
conflating them is a common performance-tuning mistake.** `O(sqrt(n))` is an
excellent complexity exponent — far better than assembly's `O(n)` or solve's
`O(n^1.5)` from A1's Explore (c) — and yet this exact operation is *still*
eventually worth replacing in production, because it's implemented as a Python
`for` loop rather than a vectorized/precomputed masked operation. The professor is
teaching you to separate two independent performance questions that beginners
often merge: **"how does the cost scale with problem size?"** (answered here:
favorably, `O(sqrt(n))`) and **"what is the constant factor, and where does it come
from?"** (a Python-interpreter-bound loop, which vectorization would shrink by
orders of magnitude at the *same* algorithmic exponent). Both questions matter,
and a good scaling exponent alone doesn't mean an implementation is done being
optimized.

---

## Key Takeaways — A2, in plain terms

1. **"Doing nothing" to a boundary face is itself a specific boundary condition**
   (zero flux), not an absence of one — a fact that's invisible until you either
   read the weak form carefully or watch a solver silently fail because of it.

2. **A wrong implicit (Neumann) boundary condition pollutes the whole domain,
   exactly like a wrong explicit (Dirichlet) one does** — non-decreasing error
   under refinement is the diagnostic signature either way (this mirrors A1's
   Explore (b) finding precisely).

3. **Pure-Neumann problems are singular by construction, and pinning one node is
   the standard fix — but the fix for singularity and the requirement for data
   consistency are two separate things.** A solve that completes after pinning
   isn't automatically an accurate solve; this tutorial's own recorded pin-case
   result is a real example of a run that answered the wrong question, flagged
   honestly rather than reported as if it were valid.

4. **A favorable complexity exponent doesn't mean an implementation is finished.**
   The row-replacement loop scales beautifully (`O(sqrt(n))`) yet is still
   flagged for replacement in production — because its *constant factor* (a
   per-node Python loop) is the actual problem, a completely separate axis from
   its asymptotic scaling.

5. **The throughline from A1 continues:** every one of this chapter's four
   findings was only knowable because something was actually measured — the
   collapsed order, the inconsistent pin-case numbers, the sqrt(n) exponent. None
   of these would be visible from reading the code alone.
