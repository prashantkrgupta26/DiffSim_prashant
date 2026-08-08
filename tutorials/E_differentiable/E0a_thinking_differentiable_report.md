# E0a — Thinking Differentiable: Detailed Report

**Script:** `tutorials/E_differentiable/E0a_thinking_differentiable.py`

This report has three parts: (1) a from-scratch theory story covering every
mathematical idea this tutorial assumes — gradients of simulation outputs,
finite differences, the tangent-linear method, the adjoint method derived by
hand via Lagrange multipliers, the implicit function theorem, and how automatic
differentiation ("taping") realizes all of this in code; (2) the base task and
all four Explore tasks, each with a problem statement, results, interpretation,
and the underlying lesson; (3) a closing takeaways section.

---

## Part 1 — The complete theory, as a story

### The question this whole chapter answers

Every earlier tutorial (A1 through A6, B1, B2, P1) asked "does the solver get
`u` right?" This chapter asks a new question: **if `u` depends on some
parameters `m` (a material property, a shape, a coefficient field), how does a
scalar summary of `u` — call it `J` — change as `m` changes?** That's a
*gradient*, `dJ/dm`, and computing it well (accurately, and cheaply enough that
you can actually use it) is the entire subject of "differentiable simulation."
Why would you want this? Because a gradient is the engine behind every
optimization loop: shrink `J` (an error against measured data, a design cost, a
misfit) by nudging `m` downhill. E1 and E2, later in this track, are exactly
this: use a gradient, computed the way E0a teaches, to *recover* an unknown
parameter or shape from data.

### Setting the stage: the "mother problem"

This chapter, and the two after it, keep returning to one running example (a
device borrowed, with attribution, from the dolfin-adjoint teaching materials):
solve `-div(kappa(x) grad(u)) = f` on the unit square, `u = g` on the boundary.
`kappa` is a spatially-varying conductivity — think of it as "how easily heat
flows at each point." Someone hands you noisy temperature *measurements* at a
few probe locations, `u_obs`. You don't know the true `kappa`, but you want to
know: **if I nudge my current guess of `kappa` at element `e`, how much does my
mismatch with the data change?** That's `dJ/dkappa_e`, for every element `e`
simultaneously — and this chapter is about the cheapest correct way to get it.

### Method 1 — Finite differences (the obvious way, and why it doesn't scale)

The most naive way to estimate a derivative is the one you learned in first-year
calculus: nudge one input, see how much the output moves.

```
dJ/dm_i ≈ [J(m + eps*e_i) - J(m - eps*e_i)] / (2*eps)
```

(`e_i` is the unit vector in direction `i` — "nudge only parameter `i`.") This
is *central finite differences*, and it needs **two full forward solves per
parameter** — one with `m_i` nudged up, one nudged down. If you have `N`
parameters (one `kappa` value per element, say), you need `2N` solves total.
It's also never exact — `eps` has to be small enough that the linear
approximation of `J` is good, but not so small that floating-point subtraction
of two nearly-equal numbers destroys your precision (this tension has a sweet
spot around `eps ~ sqrt(machine epsilon)`, and even at the sweet spot you're
still only approximately right). For a handful of parameters, `2N` solves is
completely fine. For a real spatially-varying field with thousands of
parameters, it becomes the single most expensive part of your entire workflow —
by far.

### Method 2 — The tangent-linear method (half the cost, still doesn't scale)

Instead of approximating the derivative numerically, you can compute it
*exactly* by differentiating the equation that defines `u` itself. Every
forward solve you've done in this whole project satisfies some *residual*
equation `R(u, m) = 0` (for a linear problem, `R(u,m) = A(m)u - b`). Since
`u` is *implicitly* a function of `m` through this equation, you can
differentiate the whole equation with respect to one parameter `m_i`, using the
chain rule:

```
dR/du * (du/dm_i) + dR/dm_i = 0
=>  A * (du/dm_i) = -dA/dm_i * u          (one linear solve, for each i)
```

This gives you `du/dm_i` exactly (no `eps` needed, no discretization error
beyond what's already in your original mesh), and then `dJ/dm_i = (dJ/du) *
(du/dm_i)` is a cheap follow-up. This is the **tangent-linear method (TLM)**,
also called forward-mode differentiation. It costs `N` solves instead of `2N` —
twice as cheap as finite differences, and exact instead of approximate — but
it's still `O(N)`: one new linear solve *per parameter*. For a big parameter
field, that's still prohibitive.

### Method 3 — The adjoint method (the one that doesn't care how many parameters you have)

Here's the trick that makes the adjoint method different in kind, not just
degree: **flip which variable you solve the linear system for.** Instead of
asking "how does `u` respond to a nudge in parameter `i`" (which needs one
solve per `i`), ask a single, different question: **"what linear combination of
the residual equations, when combined with the output `J`, makes the whole
thing stationary with respect to `u`?"**

This is a classical technique from constrained optimization: the method of
**Lagrange multipliers**. You're trying to compute a gradient of `J`, subject
to the constraint that `u` actually solves `R(u,m)=0`. Form the **Lagrangian**:

```
L(u, m, lambda) = J(u, m) - lambda^T R(u, m)
```

`lambda` is a new, free vector of the same size as `u` — the **adjoint
variable** (or "Lagrange multiplier"). Because `R(u,m)=0` exactly whenever `u`
is the true solution, subtracting `lambda^T R(u,m)` changes *nothing* about the
value of `L` at that point — `L = J` there, for *any* choice of `lambda`. That
freedom is the whole trick: you get to *choose* `lambda` to make the rest of
the derivation as convenient as possible.

Differentiate `L` with respect to `m_i`, using the chain rule (remembering that
`u` itself secretly depends on `m_i`, since it solves the PDE):

```
dL/dm_i = dJ/du * (du/dm_i) + dJ/dm_i - lambda^T [dR/du * (du/dm_i) + dR/dm_i]
        = [dJ/du - lambda^T dR/du] * (du/dm_i) + dJ/dm_i - lambda^T dR/dm_i
```

That first bracketed term has a `du/dm_i` in it — exactly the expensive,
per-parameter quantity we're trying to avoid ever computing. So: **choose
`lambda` specifically to make that bracket vanish.** Set

```
dJ/du - lambda^T dR/du = 0     =>     (dR/du)^T lambda = (dJ/du)^T
```

This is the **adjoint equation** — one single linear system, in the *same*
size as the original forward solve, and it does not depend on which parameter
`i` you eventually care about. Solve it *once*. Then, for every parameter `i`
simultaneously:

```
dJ/dm_i = dL/dm_i = dJ/dm_i - lambda^T (dR/dm_i)
```

— a cheap algebraic expression (no more solves), evaluated for every `i` at
once from the *same* `lambda`. Total cost: **one forward solve (to get `u`) +
one adjoint solve (to get `lambda`)**, full stop, regardless of whether you
have 10 parameters or 10 million. This is the entire reason the adjoint method
exists, and the entire reason this whole track of the project is built around
it.

### The "information flow" intuition (a physical way to picture the algebra above)

The forward solve pushes information *forward*: from the parameters `m`,
through the PDE, to the state `u`, to the output `J`. The adjoint solve pushes
information *backward*: it starts at the scalar output `J` and asks "how
sensitive is `J` to a nudge at *every point* in the domain, simultaneously,"
propagating that sensitivity backward through the PDE operator (`A^T` instead
of `A`) until it lands as a sensitivity field `lambda` covering the whole
domain in one shot. That's *why* one adjoint solve can stand in for what would
otherwise be thousands of individual forward-mode solves: it isn't computing
the same information more cleverly, it's computing a completely different
(and, for gradients, exactly sufficient) quantity — the sensitivity of one
scalar to everything, rather than the sensitivity of everything to one thing at
a time.

### Discrete vs. continuous adjoint (a choice this codebase makes explicitly)

There are two philosophically different ways to build an adjoint for a PDE:
**differentiate the continuous PDE first, then discretize the resulting
adjoint PDE** ("optimize-then-discretize"), or **discretize the PDE first,
then differentiate the resulting discrete linear system** ("discretize-then-
optimize" — exactly the derivation above, applied to the assembled matrix
equation `Au=b`, not to the continuous PDE). This codebase deliberately commits
to the second option (spec S4.3). The practical payoff: the discrete adjoint is
*exact* for the discrete forward problem — there's no separate discretization
error sneaking in from solving a different (adjoint) PDE with a different mesh
or basis. Whatever error your forward solve has (from A1's MMS convergence
studies, you already know how to measure that), your gradient inherits — no
more, no less.

### A detail that matters a lot in practice: boundary conditions and the adjoint

Strong Dirichlet boundary conditions are enforced, throughout this codebase, by
literally replacing rows of the assembled matrix with identity rows (`A[i,:] =
e_i^T`) — you've seen this since A1. Those rows don't depend on the parameter
field `m` at all. So `dA/dm_e` is *exactly zero* in every boundary row, for
every element `e`. Consequence: when you build the adjoint seed `dJ/du`, you
must **zero it out at every boundary node** before solving the adjoint
equation — otherwise you'd be asking the adjoint solve to account for
sensitivity through rows that structurally can't carry any `m`-dependence,
which would corrupt the gradient. This isn't a numerical nicety, it's a direct
algebraic consequence of how Dirichlet conditions are implemented, and getting
it wrong is a classic, easy-to-make adjoint bug.

### What "taping" actually is (automatic differentiation, concretely)

Everything above is pure math — it doesn't care how you compute `lambda^T
dR/dm_e` in code. This codebase (via NVIDIA Warp) offers a second, complementary
route: **automatic differentiation**, done by literally recording what a GPU
kernel computed and mechanically reversing it. A `wp.Tape()` context records
every kernel *launch* that happens inside it — which kernel, with which input
and output arrays. Calling `tape.backward()` walks that recorded launch graph
in *reverse*, applying the chain rule to each recorded operation automatically,
accumulating **cotangents** (the technical name for "the gradient of the final
output with respect to this intermediate array") into `tape.gradients[array]`
for every array you marked `requires_grad=True`. You never write `d(...)/d(...)`
by hand for the kernel itself — the system derives it from the kernel's own
source code, term by term, the same mechanical process used by every modern
deep-learning framework, applied here to FEM kernels instead of neural network
layers.

Two things make this genuinely powerful, and both show up directly in this
chapter's results:

- **Only arrays marked `requires_grad=True` accumulate gradients.** Anything
  else (the frozen solution `u`, say) is treated as a constant during the
  backward pass, even though it appears in the same kernel. This is exactly
  how you tell the tape "differentiate the *relation*, holding this other thing
  fixed" — the mechanical embodiment of the Lagrangian trick above, where `u`
  and `m` are differentiated separately.
- **The tape doesn't need you to derive anything by hand.** For a simple
  algebraic kernel (like `J = sum kappa_i * u_i^2`), the tape's answer and a
  hand-derived formula (`dJ/dkappa_i = u_i^2`) match to the last bit — not
  approximately, exactly, because both are computing the same exact chain-rule
  expression, one by algebra, one by mechanically walking a recorded graph.

### Why you should never try to "differentiate through" a linear solver

Section §6 of the base script states this explicitly, and it's worth
understanding *why* it's true, not just accepting it: `splu` (sparse LU
factorization) is an iterative, branchy numerical algorithm — Gaussian
elimination with pivoting. In principle you *could* write an automatic-
differentiation rule that walks through every arithmetic operation inside LU
factorization and differentiates it step by step, but this would be enormously
expensive and completely unnecessary. The key realization: **you don't need
the derivative of the *algorithm* that solves `Au=b` — you only need the
derivative of the *relation* `Au=b` itself.** This is exactly the implicit
function theorem at work: `u(m)` is defined implicitly by `R(u,m)=0`, and its
derivative can be extracted directly from `dR/du` and `dR/dm` (both cheap,
explicit, algebraic quantities — no factorization required to write them down)
without ever needing to differentiate whatever black-box procedure happens to
solve that equation numerically. This "relation, not algorithm" principle is
the single most load-bearing idea in this whole track — E0b picks it up
directly and generalizes it into a full "do-not-differentiate list" (Krylov
iterations, Newton loops, random seeds, and more, all treated the same way).

### The verification habit: "three-way agreement"

If you have three independent ways to compute the same gradient (finite
differences, the adjoint, and automatic differentiation through the residual),
and all three agree to high precision, that's very strong evidence all three
are implemented correctly — a wrong sign, a missing boundary-zeroing step, a
transposed matrix, would each break the agreement in a specific, diagnosable
way. This is the exact same discipline as A1's MMS convergence checks, just
applied to a different quantity (a gradient, not a solution field) — verify
against something you trust, and disagree loudly if the check fails, exactly
as this chapter's own `assert` statements do at the end of the script.

---

## Base Task — Three paths to a gradient, verified against each other

### 1. Problem statement

Implement all three gradient methods above (finite differences, an adjoint
solve, and automatic differentiation via `wp.Tape` through the discrete
residual) for the "mother problem" — recover `dJ/dkappa` where `J` is a
least-squares misfit against 5 probe measurements, `kappa` has 64 per-element
values (a level-3, `8x8` grid) — and check that all three agree. Separately,
run a minimal, non-FEM `wp.Tape` toy example (`J = sum kappa_i * u_i^2`) just
to see the taping mechanics with nothing else going on.

### 2. Results (after the timing fix described below)

```
Naked tape toy: max error vs hand-derived gradient = 0.00e+00 (exact)

J at eval kappa          : 8.347e-03  (expect ~8.35e-03)
adj  vs FD  max rel err  : 1.56e-08   (expect < 1e-6)
tape vs FD  max rel err  : 1.56e-08   (expect < 1e-6)
adj  vs tape max rel err : 3.41e-15   (expect < 1e-9)
adjoint time              : 0.001s    (expect < 0.05)
FD time                   : 0.086s    (expect < 1.5)
FD / adj cost ratio       : 79.9x     (expect 30-80x)
```

### 3. Interpretation

The correctness story was clean from the very first run: all three gradient
methods agree to within `1.56e-08` of each other against finite differences
(limited by FD's own `eps`-truncation error, not by any bug), and the adjoint
and tape agree with each other to `3.41e-15` — essentially machine precision,
since both are computing the *exact* same chain-rule expression by two
different mechanical routes. The *performance* story, however, initially did
not match — see the bug below.

**Bug found and fixed:** the script's own printed "COST TABLE" showed FD
running *faster* than the adjoint (a `0.4x` ratio), directly contradicting its
own EXPECTED RESULTS block (`30-80x`, matching the `O(N)` vs. `O(1)` theory
above). Root cause: `solve_poisson` (used for every forward solve, including
inside the FD loop) always uses the kappa-*weighted* variant kernel
(`make_poisson_element_matrices_var`), but the adjoint's sensitivity-assembly
step (building the per-element `Ke` matrices at `kappa=1` for the gradient
contraction) uses the *plain*, constant-kappa kernel
(`make_poisson_element_matrices`) — and that kernel had never been launched
anywhere else in the script before that exact point. The adjoint's timed block
was silently paying a one-time JIT compilation tax (P1's own lesson, landing
here) for a kernel nothing else in the script had warmed. Fixed by adding one
warm-up launch of that kernel before the timer starts; verified the corrected
ratio (`79.9x`) now matches the docstring.

### 4. The real learning — why this is in the tutorial

**The three-way check protects you from a correctness bug; it does nothing to
protect you from a performance-measurement bug — and this chapter's own base
script had exactly the second kind, hiding behind a passing first kind.** This
is a genuinely valuable, slightly humbling thing to have found directly: the
gradient math was flawless from the start (three independent methods agreeing
to 15 significant digits), while the very next section of the same file — whose
entire purpose is to demonstrate *why* the adjoint matters (it's O(1)!) — was
reporting a number that said the opposite. The lesson generalizes past this one
file: a verification suite that checks *value* correctness tells you nothing
about whether your *performance claims* about that same code are trustworthy.
Both need independent scrutiny, and P1's warm-the-cache discipline turns out to
matter just as much here, in a completely different chapter, as it did in its
own.

---

## Explore (a) — Scaling the FD/adjoint ratio with parameter count

### 1. Problem statement

Change the mesh level from 3 (64 elements/parameters) to 4 (256), and predict
what would happen at level 6 (4,096). The theory says FD cost grows with
`n_params` while adjoint cost stays roughly constant — measure whether that
holds.

### 2. Results

| level | n_params | FD time | adjoint time | ratio |
|---:|---:|---:|---:|---:|
| 3 | 64 | 0.086s | 0.001s | 79.9x |
| 4 | 256 | 0.624s | 0.002s | 299.4x |
| 6 | 4,096 | 123.9s | 0.022s | **5,632x** |

### 3. Interpretation

The scaling matches theory closely: FD time grows roughly linearly with
`n_params` (as expected — twice as many parameters means twice as many `2N`
solves), while adjoint time stays close to flat (0.001s -> 0.022s, a mere `22x`
increase for a `64x` larger parameter count — and even that growth is coming
from the underlying linear solves themselves getting bigger, not from any
per-parameter cost, since the adjoint is still exactly "one forward + one
adjoint solve" regardless of `N`). The ratio itself (`FD time / adjoint time`)
consequently grows almost linearly with `n_params` too — `79.9x` at 64 params,
`5,632x` at 4,096 params, a factor of `64x` more params producing a factor of
roughly `70x` more ratio. At level 6, FD needed over two minutes to compute a
gradient the adjoint produced in 22 *milliseconds*.

### 4. The real learning — why this is in the tutorial

**A constant-factor speedup is nice; an asymptotically different scaling law is
a different category of result entirely, and only measuring across a real range
of problem sizes reveals which one you have.** At level 3, "80x faster" might
read as "a nice optimization." At level 6, "5,632x faster" reads as "the only
approach that's remotely usable." The task is deliberately structured to make
you watch the *ratio itself* grow with problem size, not just confirm the
adjoint is faster once — because the entire point of the adjoint method is
that its advantage isn't a fixed multiplier, it's a fundamentally different
growth curve, and a single data point can never distinguish "10x faster" from
"10x faster, and getting more so every time you refine."

---

## Explore (b) — A different quantity of interest: volume-integrated `J`

### 1. Problem statement

Replace the probe-misfit `J` with a volume-integrated one, `J = int_Omega u
dV`, using `diffsim.sbm.adjoint.volume_qoi`. Does the three-way agreement still
hold?

### 2. Results

```
J = int u dV at kappa_eval: 3.077e-01
adj vs FD max rel err: 2.225e-08   (compare to the probe-J's 1.56e-08)
```

### 3. Interpretation

Yes, the check holds, at essentially the same precision as before. This is a
direct, concrete confirmation of a structural fact from the theory story above:
the adjoint *equation* (`A^T lambda = dJ/du`) and the gradient *formula*
(`dJ/dm_e = -lambda^T dA/dm_e u`) never referenced the specific *shape* of `J`
— they only ever needed `dJ/du`, computed once, up front. Swapping a quadratic
probe-misfit for a linear volume integral only changes that one input vector
(here, from `W^T(Wu - u_obs)` to the mass row-sums `m`); everything downstream
— the adjoint solve, the sensitivity contraction, the reuse of the forward
factorization — is completely unaffected. `volume_qoi`'s implementation is
even simpler than the probe case's, since `J` is *linear* in `u` here (`dJ/du`
is just a fixed vector, not something that depends on the current `u` at all).

### 4. The real learning — why this is in the tutorial

**The adjoint machinery is decoupled from the choice of objective, and seeing
that decoupling hold for a genuinely different `J` is more convincing than any
amount of reading the derivation.** This modularity is precisely why the
adjoint method scales to real applications: you write the expensive part (the
adjoint solve, tied to your PDE) once, and then any number of different
objectives — data misfits, integrated quantities, weighted combinations — plug
into the *same* machinery by supplying nothing more than a `dJ/du` vector. E1's
shape-recovery optimization loop, and E2's implicit-geometry work, both lean on
this decoupling directly: neither would be tractable if the adjoint solve
itself had to be rederived for every new objective function.

---

## Explore (c) — Why the adjoint needs `A^T`, even when the operator isn't symmetric

### 1. Problem statement

The Poisson operator used throughout this chapter is symmetric (`A = A^T`).
Find (or construct) a problem where `A` is *not* symmetric — the docstring
suggests adding a convection term, `-div(kappa grad u) + v.grad(u) = f` — and
explain why the adjoint solve still needs `A^T`, never `A`. What does this
mean for the "reuse the forward factorization" claim?

### 2. Results

Built a small nonsymmetric perturbation of the Poisson stiffness matrix
(standing in for a convection-like term) and tested three solves against the
same right-hand side:

```
base Poisson A: ||A - A^T|| = 0.0 (exactly symmetric)
convection-like A: ||A - A^T||_max = 0.233 (genuinely nonsymmetric)

trans='T' (reusing the FORWARD factorization) vs. fresh factorize of A^T:
  max difference = 4.4e-16  (agree to machine precision)
trans='T' (correct adjoint) vs. plain lu.solve (no trans, WRONG):
  max difference = 0.256    (substantially, measurably different)
```

### 3. Interpretation

The adjoint equation, `(dR/du)^T lambda = (dJ/du)^T`, has that transpose in it
by construction — it comes directly from the Lagrangian derivation in the
theory story above (`dJ/du - lambda^T dR/du = 0`, rearranged), and nothing
about that derivation ever assumed `dR/du` was symmetric. The transpose was
*always* required; the symmetric Poisson case just happens to make `A^T` and
`A` numerically identical, which silently hides the requirement — a plain
`lu.solve(rhs)` gives the exact same answer as `lu.solve(rhs, trans='T')` when
`A=A^T`, so nothing would appear to break if you (incorrectly) used the plain
solve on a symmetric problem. This experiment breaks that coincidence
deliberately: once `A != A^T`, using the wrong one gives a genuinely wrong
answer (`0.256` off, not a rounding-level discrepancy). Crucially, the "reuse
the forward factorization" argument survives completely intact even here —
`splu`'s `trans='T'` option solves the transposed system using the *same*
already-computed `L` and `U` factors (confirmed to machine precision against
independently re-factorizing `A^T` from scratch), because solving a
transposed triangular system is cheap ("back-substitute in the other order"),
not because the matrix happens to be symmetric. Symmetry was never what made
factorization reuse free — `trans='T'` being a native capability of LU-based
solvers is what did.

### 4. The real learning — why this is in the tutorial

**A convenient coincidence in your first test case can quietly hide a
requirement that will break the moment you generalize — and the fix is to
deliberately go looking for the case where the coincidence stops holding.**
Every prior result in this chapter's own Poisson example technically didn't
*need* the `A^T` requirement to be handled correctly, purely by accident of
symmetry — if the base script had used a plain `lu.solve()` instead of
`lu_fwd.solve(..., trans='T')`, every single number in this whole chapter would
still have come out exactly right, for the wrong reason. Explore (c) is
deliberately constructed to unmask that: it's not enough to verify your adjoint
code on the case that happens to be forgiving. This is a durable, general
research habit — when a piece of code passes its tests, ask specifically
whether those tests could pass *even if a conceptually important step were
silently skipped*, and if so, go build the test case that would actually catch
it.

---

## Explore (d) — `J = sum_i exp(kappa_i * u_i)`, derived by hand and stress-tested

### 1. Problem statement

Replace the toy tape kernel's energy function with `J = sum_i exp(kappa_i *
u_i)`. Derive `dJ/dkappa_i` analytically, verify the tape reproduces it
exactly, then explore what happens numerically when `kappa` is made negative.

### 2. Results

Analytic derivation: since only the `i`-th term of the sum depends on
`kappa_i`, `dJ/dkappa_i = u_i * exp(kappa_i * u_i)` directly from the chain
rule. Tape matched this **exactly** (`0.000e+00` error, every case):

```
positive kappa (baseline):        exact match
negative kappa:                   exact match, no numerical issue at all
large-magnitude stress test (kappa*u up to ~30): exact match
                                   (gradients ranging ~1e-12 to ~1e6, all finite)
genuine overflow test (kappa*u=2000, >> FP64's exp threshold ~709):
  J = inf, gradient for that term = inf — tape and analytic formula
  agree exactly even here (both go to inf together)
```

### 3. Interpretation

The tape reproduces the hand-derived analytic gradient exactly in every regime
tested, including the genuinely pathological one — when `exp` overflows to
`inf`, the tape's mechanically-derived gradient becomes `inf` too, in lockstep
with what the hand-derived formula would also give (since `u_i * inf = inf`
for `u_i > 0`). This is worth pausing on: the tape isn't "usually right" or
"right when things are well-behaved" — it's implementing the *exact same*
IEEE-754 floating-point arithmetic the hand derivation would, so wherever the
hand derivation would produce `inf`, so does the tape, faithfully, not
silently or with a different (wrong) failure mode. The more interesting finding
is a **correction to the prompt's own framing**: "try making kappa negative"
implies negative values are the risky direction to test — but they aren't.
`exp` of a very negative argument smoothly *underflows toward zero* (a
perfectly well-defined, safe floating-point operation), never producing an
error. The actual numerical hazard lives on the *positive* side: large
positive `kappa_i * u_i` is what overflows `exp`, at `kappa*u` a bit above
`709` in FP64.

### 4. The real learning — why this is in the tutorial

**Automatic differentiation doesn't just compute the right formula — it
inherits your forward computation's numerical failure modes exactly, which is
usually a feature, but only if you know what those failure modes actually are.**
If you'd only tested negative kappa (following the prompt's literal wording)
you would have concluded, incorrectly, that this kernel has no numerical
danger zone — and missed the real one entirely. This directly foreshadows the
much higher-stakes version of the exact same issue: the physical Bratu problem
elsewhere in this project (`B1`, `B2`) has a term of the identical shape,
`lambda * exp(u)`, and its Newton iteration genuinely does diverge once `u`
grows too large — not a toy-kernel curiosity, but the literal mechanism behind
a real nonlinear solver failing at a fold point. Recognizing "large positive
argument to `exp` is the danger zone, not the sign of the coefficient in front
of it" here, on a three-line toy kernel with nothing else going on, is
directly transferable to correctly reasoning about *why* that later, more
complicated solver fails exactly where and how it does.

---

## Key Takeaways — E0a, in plain terms

1. **Three gradient methods, three completely different cost profiles for the
   exact same answer.** Finite differences (`2N` solves, approximate),
   tangent-linear (`N` solves, exact), and the adjoint (`~2` solves total,
   exact, cost independent of `N`) all compute the same `dJ/dm` — the adjoint
   wins not by being a clever trick, but by asking a structurally different
   question (sensitivity of one output to everything, instead of everything to
   one input at a time).

2. **The adjoint equation always needs the transpose, whether or not your
   operator happens to be symmetric.** The symmetric Poisson case hides this
   fact by coincidence; Explore (c) deliberately breaks that coincidence to
   show it, and confirms LU-factorization reuse survives regardless, via
   `trans='T'`, not via symmetry.

3. **"Differentiate the relation, not the algorithm" is the single idea that
   makes all of this tractable.** You never need the derivative of `splu`
   itself — only of the equation `Au=b` it solves — via the implicit function
   theorem. This principle is the seed for E0b's entire "do-not-differentiate
   list."

4. **A correctness check (three-way gradient agreement) and a performance
   check (is the adjoint actually fast) are separate concerns, and this
   chapter's own base script had a real bug in the second one hiding behind a
   perfect result in the first.** Both need independent, explicit
   verification — passing one tells you nothing about the other.

5. **Automatic differentiation is exact, not approximate — it inherits your
   forward computation's numerical behavior precisely, including its failure
   modes.** Understanding *where* a computation can overflow or underflow
   (large positive exponent, not negative coefficient, for `exp(kappa*u)`) is
   necessary to correctly interpret what the tape gives you, and this exact
   pattern reappears with real consequences in the Bratu problem later in this
   project.
