# E0b — Anatomy of a Taped Brick: Detailed Report

**Script:** `tutorials/E_differentiable/E0b_anatomy_of_a_taped_brick.py`

This report assumes E0a's theory story (gradients, finite differences, the
tangent-linear method, the adjoint via Lagrange multipliers, the implicit
function theorem, "differentiate the relation, not the algorithm") as
background — read that report first if you haven't. This one covers the *new*
theory E0b introduces: what a tape mechanically records at the kernel level,
the full "do-not-differentiate list," the dot-product test, compute-once
patterns, and the mathematics of underdetermined inverse problems. Then: the
base task and all four Explore tasks, each with a problem statement, results,
interpretation, and the underlying lesson, followed by takeaways.

---

## Part 1 — The complete theory, as a story

### From "the adjoint works" to "here is exactly what code implements it"

E0a proved the adjoint method works, end to end, on one worked example. E0b's
job is different: it opens up a *real* kernel from this codebase (the Poisson
matrix-vector-product kernel) line by line, and asks you to be able to answer,
for any kernel you might ever look at: **what does taping this actually
record, and which of its inputs would sensibly be differentiated?**

### What a `wp.Tape` launch graph actually is

Picture `wp.Tape()` as a notebook that, for every `wp.launch(kernel, dim=N,
inputs=[...])` call made while it's "open," writes down exactly one line:
*which kernel, how many threads, which arrays went in, which came out.*
Nothing about the kernel's internal loop structure, its temporary variables,
or its control flow gets recorded separately — the tape's unit of memory is
"one kernel launch," not "one line of kernel code." When you call
`tape.backward()`, it reads that notebook from the bottom up (last launch
first), and for each entry, calls that *specific kernel's* backward
counterpart — a second piece of code (either hand-written or automatically
generated from the forward kernel's source) that knows how to turn "the
gradient of everything downstream of this kernel's outputs" into "the
gradient of everything downstream of this kernel's inputs," and accumulates
that into `tape.gradients[array]` for every array flagged `requires_grad=True`.

Three details of *this specific kernel* (Poisson's element matvec) matter for
understanding what does and doesn't get taped:

- **Register-local variables are invisible to the tape.** Things like the
  small `FEMElm` struct (holding the element index, quadrature-point index,
  and element size) or the local gradient accumulator `g` inside the kernel
  live in GPU/CPU registers for the duration of one thread's execution — they
  are never `wp.array`s, so there's nothing for the tape to track about them.
  The tape only ever sees *array* traffic in and out of a kernel launch, never
  what happens to scalars and small fixed-size buffers *inside* it.
- **Precomputed scalars (the Jacobian, the derivative scale) are baked in, not
  taped.** `jac = (h/2)^dim` and `dscale = 2/h` are computed once per thread
  from the input array `h`; if `h` isn't marked `requires_grad=True`, these
  are just numbers, structurally no different from writing a numeric literal.
- **`enable_backward=False` means "this kernel is structurally excluded from
  ever being taped."** It's not merely "don't tape it this time" — it tells
  Warp not to even generate the backward code for this kernel, saving real
  compile time (the docstring cites measured savings of minutes per module for
  higher-order, higher-dimension variants). This is a deliberate performance
  choice for kernels this codebase has already decided will *never* be
  differentiated directly (because their gradient comes from elsewhere — see
  next).

### Custom VJP vs. unrolled tape: two different ways to supply a gradient

A "VJP" (vector-Jacobian product) is just the technical name for exactly the
operation `tape.backward()` performs at each recorded kernel: given an
incoming cotangent (a "vector"), multiply by the kernel's own local Jacobian
(the "J") to produce the outgoing cotangent. There are two ways to obtain
this:

- **Unrolled tape (automatic):** let Warp generate the backward kernel from
  the forward kernel's source, mechanically. This is what happens for a
  kernel marked `enable_backward=True` (E0a's `kappa_residual` kernel, taped
  only in `kappa`).
- **Custom VJP (manual, supplied by you):** skip taping the kernel entirely,
  and instead supply the gradient using a hand-derived formula — exactly what
  E0a's adjoint solve is. `assemble_csr`'s stiffness-matrix kernel has
  `enable_backward=False` for exactly this reason: its gradient contribution
  doesn't come from differentiating the matrix-assembly kernel's arithmetic at
  all, it comes from the *transposed linear solve* (`A^T lambda = dJ/du`),
  which has nothing to do with unrolling the assembly loop.

Both are completely legitimate; the choice is an engineering one (custom VJPs
are cheaper to compile and can exploit problem structure the automatic path
can't see, like "this whole operation is just a matrix solve, whose transpose
I already know how to compute cheaply") — but you have to know which one a
given kernel is using before you can reason about its gradient path at all.

### The full "do-not-differentiate list"

E0a's `splu` example generalizes into four categories, each an *algorithm*
implementing some clean underlying *relation* — the same "relation, not
algorithm" principle, applied broadly:

1. **Linear solvers** (`splu`, `bicgstab`, `GMRES`, `cuDSS`, ...). Relation:
   `Au=b`. Never unroll the solver's internal iterations or factorization
   steps; use `A^T lambda = dJ/du` instead, reusing the forward factorization.
2. **Newton / nonlinear iterations.** Relation: `R(u*,m)=0` at the *converged*
   state `u*`. The adjoint equation `(dR/du)^T lambda = dJ/du` is evaluated
   once, at convergence — never inside the Newton loop, and never by
   differentiating through the (typically 5-20) individual Newton steps. This
   is exactly the implicit function theorem again, just applied to a
   nonlinear (rather than linear) `R`.
3. **Mesh/geometry construction** (octree build, element classification,
   preflight checks). These produce *combinatorial* objects — connectivity
   tables, in/out element flags — which are piecewise-constant functions of
   any underlying geometric parameter (a cell is either kept or discarded; a
   threshold crossing is a discontinuity). A derivative of a piecewise-constant
   function is zero almost everywhere and undefined exactly at the jumps —
   not useful. The fix: **freeze the mesh before opening the tape.** Any real
   *shape* sensitivity you want has to be computed from the boundary
   parametrization directly (a proper shape derivative), not from
   differentiating the mesh-building code — this is exactly E1's subject.
4. **RNG seeds and tabulated basis functions.** A random draw, once made
   inside a tape scope, is replayed with the *same* value on the backward
   pass (the tape doesn't re-randomize) — it's treated as a fixed input, not
   as something with a derivative of its own (getting a derivative *of* a
   distribution's parameters needs a different technique, "reparametrization,"
   not naive taping through the random draw). Precomputed basis tables
   (`N`, `dN`, `w`) are compile-time constants from the tape's perspective —
   they never carry `requires_grad=True`, because there's no parameter they
   represent a *sensitivity to*; they're geometry of the reference element,
   fixed once `p` and `dim` are chosen.

### The dot-product test: the cheapest possible adjoint sanity check

For *any* linear operator `A` and *any* two vectors `v, w`, transposition
means, by definition:

```
<Av, w> = <v, A^T w>
```

(`<.,.>` is the ordinary dot product.) This isn't a property that needs proof
for your *specific* `A` — it's true for every linear operator, always, by the
very definition of "transpose." That makes it an extremely cheap, extremely
general test: **pick two random vectors, apply the forward operator to one,
the adjoint operator to the other, and check the two inner products agree.**
If your adjoint implementation is buggy — a sign error, a forgotten term, a
transposed index — this test will almost always catch it, because getting
`<Av,w>` and `<v,A^Tw>` to match by *accident*, despite a real bug, is
numerically very unlikely. Crucially, the test **never needs the explicit
matrix `A`** — only its forward action (`matvec`) and its adjoint action
(`rmatvec`) — so it works identically for matrix-free / operator-only
implementations (this codebase's M1b track), not just assembled sparse
matrices. Run it on every new operator you ever add; it costs two matrix-
vector products and a subtraction.

### Compute-once-and-store: three patterns, one idea

"Don't recompute something that hasn't changed" sounds obvious, but three
specific instances of it recur constantly in this codebase and are worth
naming individually:

- **One LU factorization serves both the forward and the adjoint solve.**
  `splu(A)` doesn't just store `L` and `U`; scipy's underlying SuperLU
  representation retains enough information to solve *both* `Ax=b`
  (`lu.solve(b)`) and `A^T y = c` (`lu.solve(c, trans='T')`) from the *same*
  factorization — because solving a transposed triangular system is cheap
  (a different order of back-substitution), not a new factorization. This
  holds whether or not `A` is symmetric (Explore (a), and E0a's own Explore
  (c), both confirm this directly).
- **Tabulated basis functions are computed once, referenced everywhere.**
  `basis_tables(p, dim)` evaluates Gauss-Legendre quadrature and shape
  functions on the *reference* element exactly once; every element in the
  entire mesh, at every timestep or optimization iteration, reuses the same
  small table (only the physical mapping — via `h`, hence `jac` and
  `dscale` — differs per element).
- **"Frozen at Gauss points"** — when a coefficient field comes from some
  outer closure (a neural network, a spline, another optimization loop), you
  evaluate it *before* opening the tape and pass the result in as a frozen
  (`requires_grad=False`) array. The tape then only differentiates through
  the PDE solve itself, never through the closure's internals — the "M2
  one-way coupler" pattern, useful whenever the closure changes on a slower
  timescale than the PDE solve inside an outer loop.

### Underdetermined inverse problems, and what a "misfit drop" does and doesn't tell you

The chapter's payoff example tries to *recover* a spatially-varying `kappa`
field (64 unknowns) from just 5 point measurements. This is a classic
**underdetermined inverse problem**: far fewer independent pieces of
information (5 numbers) than unknowns (64). A PDE like Poisson's equation
acts as a smoothing, "low-pass" operator — high-frequency variations in
`kappa` barely register in the smooth field `u` that a handful of probes can
actually see. Gradient descent on the misfit `J = (1/2) sum (u(x_i) -
u_obs_i)^2` will happily, correctly, minimize `J` — but minimizing `J` only
constrains `kappa` in the directions the probes are actually sensitive to;
it leaves every other direction in the 64-dimensional parameter space
essentially unconstrained, free to sit wherever gradient descent's implicit
bias (starting point + step history) happens to leave it. That's why
"misfit dropped 100x" and "the parameter field was recovered" are *different
claims* — the first is guaranteed by a correctly-implemented gradient; the
second additionally requires enough *independent, informative* data to
constrain every degree of freedom, which 5 probes for 64 (or 256) unknowns
structurally cannot provide.

**Tikhonov regularization** is the classical fix: add a penalty term, `eps *
||kappa - kappa_ref||^2`, to the objective. This doesn't add new information
about the true field — it adds a *prior belief* ("prefer solutions close to
`kappa_ref` unless the data strongly says otherwise"), which resolves the
mathematical ill-posedness (the optimization landscape becomes strictly
convex once `eps>0`, removing flat/degenerate directions) at the cost of
biasing the result toward that prior. It's a stabilizer for the *optimization
problem*, not a way to manufacture missing measurements.

---

## Base Task — Opening a real kernel, and the underdetermined-recovery payoff

### 1. Problem statement

Walk through the annotated source of a real DiffSim kernel (Poisson's
matrix-vector product) and identify exactly what gets taped and what
doesn't. Run the dot-product test on the assembled Poisson operator. Time one
LU factorization serving both a forward and an adjoint solve against two
separate factorizations. Then run the full payoff: recover an unknown `kappa`
field from 5 probes via gradient descent, and understand precisely what
"the misfit dropped over 100x" does and does not prove.

### 2. Results

```
Dot-product rel err       : 9.36e-18  (expect < 1e-12)
Factorization reuse ratio : 2.33x    (expect > 1.2x)
misfit J_init             : 8.3467e-03  (expect 5e-3 - 5e-2)
misfit J_final            : 7.1454e-05  (expect < 5e-4)
misfit drop                : 116.8x   (expect >= 100x)
param error ratio         : 0.908  (expect 0.5 - 1.5)
```

Matches the docstring's EXPECTED RESULTS exactly — no bug this time.

### 3. Interpretation

The dot-product test passes to essentially machine precision (`9.36e-18`,
smaller than FP64's own `~2e-16` unit roundoff would suggest is even
meaningful — a sign the symmetric Poisson case makes this test *exact*
algebraically, with only floating-point noise entering). Reusing one LU
factorization for both solves is `2.33x` faster than factorizing twice, even
at this tiny problem size (81 free nodes) where factorization cost is
relatively small — the docstring notes this ratio only grows favorably as
problems scale, since factorization (`O(n^1.5)` in 2-D, per P1's own
measurements) grows faster than a triangular solve (`O(n log n)`). The
recovery result is the chapter's central, carefully-worded lesson: misfit
dropped `116.8x` (comfortably past the `100x` bar), but the parameter error
ratio (`0.908`) says the recovered `kappa` field is still, in norm, almost as
far from the true field as the flat initial guess was — the gradient
correctly minimized what it was asked to minimize; it was never possible for
it to do more than that with only 5 probes for 64 unknowns.

### 4. The real learning — why this is in the tutorial

**A large, correctly-computed misfit drop is not the same claim as "the
parameter was recovered," and the gap between those two claims is not a bug —
it's a fact about how much information the data actually contains.** This is
easy to misread if you only look at the misfit curve (which looks like an
unambiguous success — over 100x smaller!) without also checking what happened
in parameter space. The professor is teaching you to hold both diagnostics
simultaneously whenever you run an inverse problem: is the *objective* being
minimized correctly (verified independently via the dot-product test and
E0a's three-way check — a pure correctness question), and does minimizing
that specific objective, with this specific data, actually constrain the
thing you *care about* (a completely separate, data-sufficiency question). A
correct gradient computed on an underdetermined problem will still converge
smoothly and look successful by its own metric — which is exactly why you
need a second, independent diagnostic (parameter-space error, when you have
access to ground truth; more generally, methods like posterior covariance or
resolution analysis when you don't) to know whether "the misfit dropped" also
means "I learned what I wanted to learn."

---

## Explore (a) — A nonsymmetric operator, and whether pure convection is skew-symmetric

### 1. Problem statement

The Poisson operator is symmetric, so the dot-product test passes somewhat
trivially (it would even pass with a bug that only broke the *asymmetric*
part of a correct implementation, since there is no asymmetric part to break).
Add a convection term, `C[i,j] = int N_i (v.grad N_j) dV` with `v=[1,0]`, and
re-run the test on the genuinely nonsymmetric `A + C`. Separately: is `C`
itself skew-symmetric (`C^T = -C`)?

### 2. Results

(Two bugs in my own scratch implementation were found and fixed along the
way — both were the same mistake: using the *reference*-element gradient
table `dN` directly, without the `dscale = 2/h` reference-to-physical scaling
the real kernels apply via `fe_dN_s`. Not a tutorial bug — my own hand-rolled
numpy reproduction of the weak form. Final, corrected numbers:)

```
Pure convection term: ||C + C^T||_max = 8.333e-02  (nonzero -- not skew-symmetric)
Full nonsymmetric operator: ||A - A^T||_max = 0.375
Dot-product test <Av,w> vs <v,A^Tw>: relative error 4.06e-17
WRONG check <Av,w> vs <v,Aw> (using A, not A^T): relative "error" 7.03e-03
```

### 3. Interpretation

`C^T != -C` — the intuitive-sounding "convection should be skew-symmetric"
does not hold for this literal, unmodified advective form on a bounded
domain. The reason is a standard integration-by-parts identity: for a
divergence-free velocity `v` (true here, trivially, since `v=[1,0]` is
constant), `C_ij + C_ji = int_{boundary} N_i N_j (v.n) dS` — a genuine
*boundary flux* term, zero only where the domain has no boundary (periodic)
or where `v` is everywhere tangent to the boundary (`v.n=0`). Neither holds
here: `v=[1,0]` has full normal component on the left/right (`x=0,1`) faces of
the unit square. The measured residual (`0.083`) is exactly this boundary
term, not numerical noise (noise would be `~1e-15`, four orders of magnitude
smaller). Separately, and more importantly for the chapter's actual point:
the dot-product test on the *full* nonsymmetric operator still passes at
machine precision — direct, concrete proof that the test's validity has
nothing to do with whether `A` happens to be symmetric. And the deliberately
"wrong" comparison (`<Av,w>` vs. `<v,Aw>`, silently dropping the transpose)
gives a measurably large discrepancy (`0.7%`, not machine-precision) —
exactly the signature the real test is designed to catch if an adjoint
implementation ever forgets to transpose.

### 4. The real learning — why this is in the tutorial

**An intuitive-sounding mathematical claim ("convection should be
antisymmetric") deserves the same skepticism as any other unverified
assertion in this whole project — and working it out reveals a genuinely
useful fact (the boundary-flux identity) rather than just confirming or
denying a yes/no.** This mirrors E0a's Explore (c) almost exactly, but from
the opposite direction: there, symmetry accidentally held and hid a
requirement; here, skew-symmetry was *expected* to hold and didn't, for a
reason (a nonzero boundary flux) that's directly useful to know if you ever
implement a real convection-diffusion solver in this codebase — the
"skew-symmetric form" of the convective term, commonly used in stabilized
finite-element methods specifically to avoid this exact boundary-coupling
behavior, is a real, standard alternative formulation, and understanding why
the naive form doesn't have that property is what tells you when you'd want
the alternative.

---

## Explore (b) — Level 4 and Tikhonov regularization

### 1. Problem statement

Increase to level 4 (256 elements/parameters, still only 5 probes). Does the
misfit drop more or less than at level 3? Add a Tikhonov term `eps *
||kappa-1||^2` to `J` and tune `eps` to stabilize recovery.

### 2. Results

| config | J_init | J_final | misfit drop | param err ratio |
|---|---:|---:|---:|---:|
| level 3, no Tikhonov (baseline) | 8.3e-3 | 7.0e-5 | 118.9x | 0.908 |
| level 4, no Tikhonov | 8.9e-3 | 2.0e-4 | 44.7x | 0.945 |
| level 4, eps=0.001, ALPHA=36 (unchanged) | 2.0e-2 | 4.5e-3 | 4.6x | 0.899 |
| level 4, eps=0.1, ALPHA=36 (unchanged) | 1.16 | 169 | **diverges** | 10.8 |
| level 4, eps=1.0, ALPHA=0.5 (re-tuned) | 11.5 | 0.063 | **181.7x** | 1.120 |

### 3. Interpretation

At a fixed step budget (40 steps), level 4 drops misfit noticeably less than
level 3 (`44.7x` vs `118.9x`) — exactly the prompt's hint: 4x more unknowns
against the same 5 probes makes the problem more underdetermined and the
optimization landscape flatter in more directions, so the same number of
gradient steps makes less relative progress. The Tikhonov sweep produced a
sharper, more useful lesson than "tune `eps` until it works": *naively*
increasing `eps` while keeping the same step size (`ALPHA=36`, tuned for the
unregularized problem) makes things dramatically *worse*, not better — at
`eps=0.1` the misfit explodes to `169` (from an already-larger `J_init=1.16`)
and the parameter error ratio blows past `10`, far worse than doing nothing.
This isn't regularization failing — it's a step-size mismatch: adding `eps *
I` to the effective Hessian raises its eigenvalues, so a step size well-tuned
for the *un*regularized landscape overshoots badly once regularization
stiffens the problem. Re-tuning the step size down (`ALPHA=0.5` for
`eps=1.0`) restores stable convergence and even a strong misfit drop
(`181.7x`) — but the parameter-space error still doesn't improve
(`ratio=1.120`, if anything slightly worse than the unregularized baseline).

### 4. The real learning — why this is in the tutorial

**Regularization and step-size tuning are two separate knobs that interact,
and forgetting that is a realistic, easy way to conclude "regularization made
things worse" when the real problem is an untouched, now-mismatched learning
rate.** This is a genuinely common trap in real optimization work, well
beyond this toy problem — anyone who has added an L2 penalty to a loss
function and watched training diverge, without re-checking the step size, has
hit exactly this. The second half of the finding is just as important: even
once the optimization is stabilized and looks successful by its own misfit
metric, the parameter-recovery diagnosis (Explore (a)'s companion lesson from
the base task) still shows no real improvement — regularization is a
*well-posedness* fix, not an *information* fix, and conflating the two is a
much subtler mistake than the step-size one, because a stabilized, converged-
looking optimization run gives no obvious external signal that it's still not
answering the question you actually wanted answered.

---

## Explore (c) — A misplaced initial guess, and step-size sensitivity

### 1. Problem statement

Replace the flat initial guess with a single, misplaced Gaussian blob (only
one, not matching either of the true field's two blobs). Measure the error
drop. What happens at `ALPHA=2.0`?

### 2. Results

| init | ALPHA=36 | ALPHA=10 | ALPHA=2 | ALPHA=0.5 |
|---|---|---|---|---|
| flat (baseline) | drop 116.8x, ratio 0.908 | drop 64.0x, ratio 0.955 | drop 37.5x, ratio 0.977 | drop 3.2x, ratio 0.981 |
| single-blob (misplaced) | drop 254.9x, ratio 0.849 | drop 140.2x, ratio 0.895 | drop 93.0x, ratio 0.914 | drop 7.6x, ratio 0.934 |

No divergence at any tested `ALPHA`, including `2.0`, for either initial guess.

### 3. Interpretation

The misplaced single-blob guess outperforms the flat guess at *every* step
size tested — bigger misfit drop, smaller parameter error, consistently.
Even though the blob is in the wrong place, it's *structurally* closer to the
truth (localized, elevated conductivity, roughly the right shape and
magnitude) than a perfectly flat field is — gradient descent gets a genuine
head start from that partial structural correctness, even though the initial
"location" information is wrong. `ALPHA=2.0` produces no drama at all: both
initial guesses continue to converge, just more slowly within the fixed
40-step budget (`37.5x`/`93.0x` drop, vs `116.8x`/`254.9x` at the tuned
`ALPHA=36`) — a graceful, monotonic slowdown as the step size shrinks, all
the way down to `ALPHA=0.5`, not a threshold beyond which things suddenly
break.

### 4. The real learning — why this is in the tutorial

**Not every parameter has a hard, need-to-search-for threshold — some just
trade off speed for stability smoothly, and telling the two situations apart
by simply trying a range of values (rather than assuming a story in advance)
is itself the useful habit.** Explore (b)'s Tikhonov `eps` had a sharp,
dangerous failure mode at a mismatched step size; this task's `ALPHA` sweep,
on the *same underlying problem*, shows no such cliff at all across the
entire tested range — smaller step, slower convergence, nothing more
dramatic. The contrast is the point: you cannot know in advance which kind of
parameter you're tuning (one with a stability cliff, one with a smooth
speed/robustness trade-off) without actually sweeping it and looking, and
assuming either behavior by analogy from a different parameter (or a
different problem) is exactly the kind of unverified assumption this whole
project consistently warns against making.

---

## Explore (d) — A nonlinear, state-dependent coefficient `kappa(u) = 1 + alpha*u^2`

### 1. Problem statement

The compute-once sensitivity matrix `Ke` was built at `kappa=1`, exploiting
`kappa` being a fixed, `u`-independent parameter. Write the *exact* element
sensitivity for a nonlinear, state-dependent coefficient — `kappa` now
depends on the unknown solution `u` itself. What changes in the gradient
formula, and does the implicit function theorem argument still hold?

### 2. The derivation

With `kappa(u,x) = 1 + alpha*u(x)^2` (`alpha` now the true free parameter,
since `kappa` is no longer independently adjustable — it's entirely
determined by `u` and `alpha`), the weak-form residual becomes genuinely
nonlinear in `u`:

```
R_a(u) = int [ kappa(u) grad(N_a).grad(u) ] dV - int [N_a f] dV
```

Differentiating with respect to a nodal value `u_c` (needed for the Newton/
adjoint Jacobian) now picks up a term that didn't exist when `kappa` was a
fixed, independent parameter — because `kappa` itself changes as `u` changes:

```
dR_a/du_c = int [ kappa(u) grad(N_a).grad(N_c) ] dV     <- the "old" term (same as before)
          + int [ (dkappa/du_c) grad(N_a).grad(u) ] dV   <- NEW: kappa reacting to u
```

Using `dkappa(x)/du_c = 2*alpha*u(x)*N_c(x)` (chain rule through `kappa(u(x))
= 1 + alpha*u(x)^2`, with `u(x) = sum_c u_c N_c(x)`), the new term is an
extra, non-symmetric element matrix: `Extra_ac = int [grad(N_a).grad(u)] *
2*alpha*u(x) * N_c(x) dV`. The **full nonlinear Jacobian** is `J_full =
K(kappa(u)) + Extra` (the standard kappa-weighted stiffness, plus this new
term), and it is *not* symmetric even though the standard stiffness part
alone would be — a second, independent illustration of Explore (a)'s lesson
that symmetry is a special case, not a default.

The adjoint recipe from the do-not-differentiate list's "Newton loops" entry
applies completely unchanged: solve `(dR/du)^T lambda = dJ/du` using this
*full* Jacobian, evaluated at the **converged** solution `u*` (not at any
intermediate iterate) — one linear solve, regardless of how the forward
nonlinear solve was reached — then `dJ/dalpha = -lambda^T (dR/dalpha)`, using
`dkappa(x)/dalpha = u(x)^2` in place of the earlier constant weight.

### 3. Results (numerical verification)

Built a Picard-iterated forward solve (a deliberately *different* nonlinear
solver than Newton, chosen specifically to test whether the algorithm choice
matters — per the theory), the full nonlinear Jacobian above, and the
adjoint using it, for `J = int u dV`:

```
adjoint dJ/dalpha = -4.453117e-02
FD dJ/dalpha       = -4.453117e-02
relative error      = 2.72e-09
```

(First attempt gave a 99.6% mismatch, traced to the exact same `dscale`
scaling bug as Explore (a) in my own hand-rolled gradient code — fixed, and
the full Jacobian was separately verified column-by-column against a
directly finite-differenced residual, `~1e-10` agreement, before trusting the
adjoint number above.)

### 4. Interpretation

The IFT/adjoint argument holds completely unchanged **in form** — same
equation, `(dR/du)^T lambda = dJ/du`, same "solve once at convergence" rule —
because nothing about the derivation of the adjoint ever assumed `R` was
linear in `u`; it only ever assumed `R(u,m)=0` defines `u` implicitly near a
point where `dR/du` is invertible. What changes is entirely on the *forward*
side: `dR/du` now genuinely depends on `u` (it's a true nonlinear problem,
needing real iteration to solve, unlike the earlier fixed-`kappa` case which
was exactly linear), and building the correct adjoint requires the *full*
Jacobian, extra term included — using only the "old" (partial) stiffness term
would silently give a wrong gradient, since it would be missing exactly the
part of the sensitivity that flows through `kappa`'s own dependence on `u`.
Using Picard iteration (not Newton) for the forward solve, and getting the
identical, correct answer, is a deliberate, direct demonstration — not just
an assertion — of "differentiate the relation, not the algorithm": the
adjoint doesn't know or care that Picard, not Newton, produced `u*`; it only
needs `u*` itself and the residual's own derivatives there.

### 5. The real learning — why this is in the tutorial

**Generalizing the adjoint to a new, harder case is not "learn a new
formula" — it's "re-derive the same equation with the correct Jacobian for
the new residual," and the whole value of having internalized the derivation
(rather than memorizing the linear-case formula) is being able to do exactly
that.** This explore task is the chapter's hardest, and it's positioned last
deliberately: it requires combining the do-not-differentiate list's Newton-
loop entry (E0b §3(B)) with the boundary-zeroing mechanics (E0a) and the
compute-once discipline (E0b §4) into a genuinely new derivation, not a
substitution into a memorized template. It's also a direct, explicit forward
pointer to E0c (which formalizes "the adjoint-readiness checklist" this whole
chapter has been informally building) and to M4 (temperature-dependent
material properties in a real phase-field thermodynamics setting) — this toy
`kappa=1+alpha*u^2` is the smallest possible instance of a pattern
(state-dependent coefficients) that shows up in real, physically-motivated
nonlinear PDEs throughout this project.

---

## Key Takeaways — E0b, in plain terms

1. **A tape's unit of memory is "one kernel launch," not "one line of
   code."** Register-local variables, precomputed scalars, and anything not
   passed as a `wp.array` with `requires_grad=True` are invisible to it —
   understanding this is what lets you read any kernel and know in advance
   what taping it would and wouldn't capture.

2. **The do-not-differentiate list is one idea (relation, not algorithm),
   applied to four recurring situations:** linear solvers, Newton loops,
   mesh/geometry construction (freeze before taping), and RNG/tabulated
   inputs (frozen constants, not variables).

3. **The dot-product test validates transpose-correctness, a property
   completely independent of symmetry** — confirmed twice over, once by a
   symmetric operator passing "for free," once by a genuinely nonsymmetric
   one still passing at machine precision, and once by a deliberately wrong
   version of the test failing measurably.

4. **A large, correctly-computed misfit drop and "the parameter was
   recovered" are different claims, and only the data's actual information
   content — not the correctness of your gradient — determines whether the
   second one follows from the first.** Tikhonov regularization stabilizes
   the *optimization*; it does not supply missing *information*, and
   conflating the two is easy because a stabilized run still looks
   successful by its own metric.

5. **Not every hyperparameter has a sharp failure cliff — some trade off
   speed for stability smoothly, and you can only tell which kind you have by
   actually sweeping it**, not by assuming behavior from a differently-shaped
   parameter (Tikhonov's `eps` vs. gradient descent's `ALPHA`, on the exact
   same problem, behaved completely differently).

6. **Generalizing the adjoint to a genuinely nonlinear, state-dependent
   coefficient is a re-derivation exercise, not a lookup** — and correctly
   doing it (verified here to `2.7e-9` against finite differences, after
   catching and fixing a real bug in the process) is the clearest possible
   demonstration that the IFT-based adjoint recipe generalizes precisely
   because its derivation never depended on linearity in the first place.
