# B1 — Bratu / Newton: Detailed Report

**Script:** `tutorials/B_nonlinear/B1_bratu_newton.py`
**Date:** August 6, 2026

This report covers the base task and all four Explore tasks in B1. For each: the
problem statement, what was measured, what it means, and the underlying lesson
the tutorial is teaching. A "Key Takeaways" section closes it out.

---

## Base Task — Newton's method on the Bratu problem

### 1. Problem statement

The Bratu (solid-fuel ignition) equation is a **nonlinear** Poisson problem:

```
-lap(u) = lam * exp(u) + f,     u = g on the boundary
```

Every solver used in the A-tutorials (A1-A6) was for a *linear* PDE — assemble one
matrix, solve one linear system, done. Here `exp(u)` makes the equation nonlinear:
the "stiffness" you'd need depends on the very solution you're trying to find. The
task is to solve this by hand-writing a **Newton iteration**: at each step, linearize
around the current guess, solve a linear system for the correction, and repeat until
the residual is machine-zero. The weak residual and Jacobian are:

```
R_a(u) = (grad N_a, grad u_h) - (N_a, lam*exp(u_h) + f)
J_ab   = (grad N_a, grad N_b) - (N_a, lam*exp(u_h)*N_b)
```

`J` is just the familiar stiffness matrix minus an "exponentially-weighted mass
matrix" — a new pattern (weighted integrals over Gauss points, assembled with numpy
`einsum`) that recurs constantly in nonlinear FEM. The problem is verified two
independent ways: the manufactured-solution (MMS) spatial order should be unchanged
from A1 (order 2 for p1, order 3 for p2) — proving nonlinearity didn't corrupt
*discretization* accuracy — and the Newton residual should shrink **quadratically**
— proving the *iteration* itself is correct.

### 2. Results

```
Newton |R|_inf per step (level 5, p1, lam=3):
  1.42e-02  1.66e-03  3.08e-05  8.04e-09  1.17e-15        (5 steps)

p=1: errors 2.353e-03  5.898e-04  1.476e-04   orders 2.00  2.00
p=2: errors 2.056e-04  2.574e-05  3.219e-06   orders 3.00  3.00
```

This matches the file's own EXPECTED RESULTS exactly.

### 3. Interpretation

The residual sequence is a clean demonstration of **quadratic convergence**: each
step's residual is roughly the *square* of the previous one (e.g. `(1.66e-3)^2 ~
2.8e-6`, close to the next residual `3.08e-5`; the effect gets even sharper as the
iterate approaches the root). In practice this means the number of *correct digits
roughly doubles every step* — 2 digits, then 4, then 8, then full machine precision
in just 5 steps total. Meanwhile the spatial convergence orders are untouched from
the purely-linear A1 tutorial: p1 still gives order 2, p2 still gives order 3. Two
completely independent checks (iteration behavior, spatial accuracy) both come out
exactly as theory predicts — strong evidence the residual/Jacobian derivation and
the numpy assembly code are both correct.

### 4. The real learning — why this is in the tutorial

**Nonlinearity is not a discretization problem, it's an *iteration* problem.** A
huge, common misconception when moving from linear to nonlinear PDEs is to think you
need a fundamentally different finite-element method. You don't — the *basis
functions, quadrature, and mesh are exactly the same as before*. What changes is
that you can no longer assemble-and-solve once; you assemble a *sequence* of linear
problems (the Newton linearizations) and solve each one with the same linear solver
you already trust. The professor is teaching you to cleanly separate two concerns
that beginners conflate: **(1) does my discretization approximate the true PDE
well?** (answered by the spatial order test, unchanged from A1) and **(2) does my
iterative solution process actually converge to the discrete solution?** (answered
by the residual-squaring test). Verifying both *independently* is the actual
professional habit being taught — if you only checked the final spatial order, a
bug that made your Newton loop converge to the *wrong* fixed point could still slip
through with a misleadingly "correct-looking" order plot.

---

## Explore (a) — Modified Newton (frozen Jacobian)

### 1. Problem statement

Full Newton recomputes and re-factorizes the Jacobian `J` at *every* iteration —
expensive, since factorizing a sparse matrix is one of the costliest steps in the
whole pipeline (as later borne out by A5's cost-table lesson). The question: what if
you compute `J` only **once**, at the first iterate, and reuse that same (frozen)
matrix/factorization for every subsequent Newton correction, using only the current
residual `R(u)` to keep updating `u`? This is called the **modified Newton method**
(or "Newton with a stale/frozen Jacobian"). Does it still converge? At what rate,
and how many more iterations does it need to reach the same tight tolerance
(`1e-12`)?

### 2. Results

| variant | iterations to `1e-12` | residual pattern |
|---|---:|---|
| full Newton | 5 | quadratic (`1.4e-2 -> 1.7e-3 -> 3.1e-5 -> 8.0e-9 -> 1.2e-15`) |
| modified Newton (frozen at 1st iterate) | 16 | linear, ratio ≈ 0.20 per step |

### 3. Interpretation

Modified Newton **does** still converge — but linearly, not quadratically. Once the
Jacobian is frozen, each step's residual is reduced by roughly a *constant factor*
(~0.20, i.e. losing about one "5" of accuracy per step) instead of squaring. This
takes over 3x more iterations (16 vs 5) to reach the same tolerance. The mechanism:
Newton's quadratic convergence specifically requires the Jacobian to track the
*current* iterate; once you freeze it at an earlier (less accurate) point, you're
really running a **chord method** (fixed-slope Newton), whose convergence rate is
governed by how far the frozen Jacobian has drifted from the true, iterate-dependent
one — a fixed, geometric ("linear") contraction rather than the doubling-digits
behavior of true Newton.

### 4. The real learning — why this is in the tutorial

This is a **direct preview of a real engineering trade-off**, not just a math
curiosity. In A5 you already measured that factorizing a sparse matrix is one of
the most expensive operations in the whole pipeline (superlinear in the number of
DOFs, and the dominant 3-D cost). If refactorizing 16 times (cheap-but-slow) still
costs *less total wall-clock time* than refactorizing 5 times (expensive-but-fast)
— because each of the 5 full-Newton factorizations is itself much more costly than
reusing one frozen factorization — modified Newton can be the *faster* choice in
practice, despite needing more iterations. This is exactly the kind of
iterations-vs-cost-per-iteration trade-off that dominates real nonlinear-solver
engineering (it's also precisely the mechanism behind quasi-Newton / BFGS-family
methods, and behind why many production nonlinear solvers only refresh the
Jacobian every few steps instead of every step). The professor wants you to see the
quadratic-vs-linear distinction not as an abstract theorem, but as something with a
direct, measurable cost consequence you already have the tools to evaluate.

---

## Explore (b) — Picard / fixed-point iteration

### 1. Problem statement

Push the "how much Jacobian do I really need" question to its extreme: drop the
`-M_jac` term entirely, so the linear solve at every step just uses `J = K` (the
plain, unweighted linear stiffness matrix — no information about the nonlinear term
at all). This is the classical **Picard / fixed-point iteration**: `u_{n+1} =
K^{-1} (nonlinear right-hand side evaluated at u_n)`. What's the convergence rate
now?

### 2. Results

| variant | iterations to `1e-12` | residual pattern |
|---|---:|---|
| full Newton | 5 | quadratic |
| modified Newton | 16 | linear, ratio ≈ 0.20 |
| Picard (`J = K`, no nonlinear-term linearization at all) | 22 | linear, ratio ≈ 0.32 |

### 3. Interpretation

Picard also converges (the problem is well-behaved at `lam=3`), but at a **slower**
linear rate than modified Newton (0.32 vs 0.20 contraction per step), needing 22
iterations instead of 16. This produces a clean three-tier hierarchy: exact,
iterate-tracking Jacobian → quadratic (5 steps); frozen-but-real Jacobian → linear,
fast (16 steps); no Jacobian information at all → linear, slower (22 steps). Picard
is cheaper *per step* than either Newton variant (`K` is fixed for the *entire*
problem, so it only needs factorizing once, ever — even cheaper than modified
Newton, which still needed to build `M_jac` once) but needs the most total steps.

### 4. The real learning — why this is in the tutorial

This completes the "convergence-rate vs. per-step-cost" spectrum that Explore (a)
opened, and it's teaching you to **recognize a whole family of methods as one
dial**, not three unrelated algorithms: how much of the true nonlinear structure do
you inject into your linear solve? Full information → fastest convergence, priciest
step. Zero information → slowest convergence, cheapest step. Real solvers
(SNES in PETSc, Newton-Krylov methods, etc.) let you tune exactly this dial, and
practitioners routinely start an iteration with cheap Picard steps (to get roughly
into the basin of attraction) before switching to full Newton (to finish with
quadratic precision) — a hybrid strategy that only makes sense once you've seen,
numerically, that both ends of the dial are legitimate, working methods with very
different cost profiles.

---

## Explore (c) — Continuation to the fold point (physical Bratu, f=0, g=0)

### 1. Problem statement

Switch from the manufactured (MMS) problem to the **physical** Bratu problem: no
forcing (`f=0`), homogeneous boundary conditions (`g=0`). This equation is famous in
nonlinear PDE theory for having **two solutions** for `lam` below a critical value
`lam* ≈ 6.808` (on the 2-D unit square) and **no solution at all** above it — a
textbook *fold* (saddle-node) bifurcation. The task: track the **lower branch** by
**continuation** — solve at `lam=1`, use that solution as the initial guess for
`lam=2`, use *that* solution as the guess for the next `lam`, and so on, walking
`lam` up toward the critical value. What happens to the Newton iteration count as
`lam` approaches `lam*`?

### 2. Results

Continuation from `lam=0.5` up to `lam=6.82` (level 5, p1), warm-starting each
solve from the previous converged solution:

| lam range | iterations | converged? | max\|u\| |
|---|---:|---|---:|
| 0.5 – 6.6 | 3–4 | yes | 0.038 → 1.066 |
| 6.7 – 6.813 | 4–5 | yes | 1.150 → 1.379 |
| 6.814 | 60 (capped) | **no** | diverges |
| 6.82 | 60 (capped) | **no** | diverges (max\|u\| spikes to 27+) |

Zooming in with step size `0.001` pinned the transition precisely: `lam=6.813`
converges cleanly in 5 iterations; `lam=6.814` fails to converge at all (residual
never drops below `1e-4` even after 60 iterations).

### 3. Interpretation

The discrete critical value is `lam*_h ≈ 6.813–6.814` — within about **0.1%** of the
literature continuum value `lam* ≈ 6.808124`, an excellent validation of the whole
pipeline on a genuinely different (non-manufactured) problem. The more interesting,
and slightly counter-intuitive, finding: the iteration count does **not** creep up
gradually as `lam` approaches the fold — it stays flat at 3–5 iterations essentially
all the way to `6.813`, then Newton fails *completely* one step later. This is
consistent with fold-bifurcation theory: the Jacobian is only exactly singular *at*
`lam*` itself, so convergence degradation is confined to a very narrow neighborhood
of the fold; the step sizes used here (as coarse as `1.0`, as fine as `0.001`) mostly
stepped clean over that narrow zone rather than resolving a gradual slowdown within
it.

### 4. The real learning — why this is in the tutorial

This is the professor showing you that **Newton's method is not just a numerical
trick — it's a diagnostic instrument for the underlying mathematics.** The reason
Newton fails at `lam=6.814` isn't a bug or a tolerance problem: it's telling you,
correctly, that **no solution exists there** on the branch you're tracking (both
branches of the physical Bratu problem literally collide and vanish at the fold).
The solver's failure mode is physically meaningful, not just numerically
inconvenient — and being able to read "Newton stopped converging" as "you've hit a
bifurcation" rather than "something is broken" is a core skill for anyone doing
nonlinear PDE research (multiple equilibria, criticality, blow-up, etc. all show up
this way). This explore task is also explicitly a stepping stone: the docstring
notes "B2 turns this into proper continuation" — meaning what you did here by hand
(naive lam-stepping with a fixed step size, which necessarily fails catastrophically
right at the fold) is exactly the naive approach that real **arclength/pseudo-
continuation** methods (covered next in B2) are built to fix, by parametrizing the
solution path by arclength instead of by `lam` directly, so the solver can smoothly
walk *around* the fold instead of walking straight into a wall.

---

## Explore (d) — Performance: host assembly vs. solve, and a kernel sketch

### 1. Problem statement

Every earlier A-tutorial performance corner cared about the linear-solve cost
(`splu` factorize + backsolve). But B1's Newton loop does something new every
iteration: it re-assembles the **nonlinear weighted-mass term** from scratch, on the
*host* (CPU), using plain numpy — not on the GPU like the rest of the pipeline. The
task: at level 7, measure how much time goes to this host-side re-assembly versus
the `splu` linear solve, per Newton step — which dominates? Then, **without
implementing it**, sketch what a GPU (`warp`) kernel for this exponentially-weighted
mass matrix would look like, referencing the "brick API" pattern already used
elsewhere in the codebase for closure-hook-style variable coefficients.

### 2. Results

Level 7 (`n_free = 16,641`), 5 Newton steps:

| stage | per-step times (s) | total |
|---|---|---:|
| host assembly (`gp_values` + `weighted_integrals`, numpy) | 0.055, 0.040, 0.037, 0.056, 0.013 | 0.201s |
| `splu` solve | 0.049, 0.047, 0.043, 0.043 | 0.182s |

Assembly/solve ratio: **1.03x** — essentially tied, assembly a hair ahead.

Checked the source directly: `B1_bratu_newton.py` has **no `import warp`
statement at all**. `BratuWorkspace.gp_values` and `weighted_integrals` are pure
`numpy` (`np.einsum`, `np.add.at`, `scipy.sparse.coo_matrix`) — 100% host/CPU. This
is in contrast to the linear stiffness `K`, which *is* GPU-assembled once, up
front, via `assemble_csr(self.dm)` (a `warp`-kernel path).

### 3. Interpretation

At this DOF count the two costs are roughly balanced, but that balance is fragile:
`splu`'s factorize cost scales *superlinearly* in DOFs (A5 measured close to
`O(n^1.5)` in 2-D), while the host numpy assembly here is doing per-element,
per-Gauss-point work that should scale close to *linearly* in element/DOF count —
and it's running on a single CPU thread with Python-level `einsum` dispatch
overhead per call, *every single Newton step*, on *every* mesh refinement. As the
mesh gets finer (or in 3-D, per A5's lesson), the linear solve will increasingly
dominate — but at *this* level, the host assembly is already costing just as much
as the "expensive" sparse factorization, purely because it was never moved to the
GPU like everything else in the pipeline was.

**Kernel sketch (per the prompt: sketch only, not implemented).** The codebase
already contains the right template for this: `make_poisson_element_matrices_var`
in `assembly/operators.py` is a "closure-hook" stiffness kernel that accepts a
per-Gauss-point scalar coefficient array `kq[e*nqp+q]` and weights the usual
`dN_a . dN_b` stiffness integrand by it — used elsewhere for spatially-varying
material coefficients. A `make_bratu_mass_var(nbf, nqp, dim)` kernel would follow
the *exact same shape*, but with:
- a **mass-like** integrand (`N_a * N_b`, the basis *values* table, instead of the
  gradient table `dN_a . dN_b`), and
- the per-Gauss-point weight `kq` computed **on-device**, by first interpolating the
  current nodal solution `u_h` to each Gauss point via the existing basis-value
  table (`uq = sum_a N_a * u_a`, a small additional device kernel or a fused step
  inside the same launch), then evaluating `lam * exp(uq)`.

This would eliminate the host round-trip for the nonlinear term completely: nodal
`u` (already resident on the GPU from the linear solve's constrained free-DOF
vector) goes in, per-element `Me[e,a,b]` comes out, and only the small COO triplet
arrays need to return to host for the existing `coo_matrix -> csr` assembly step (or
even that could be avoided using the device-resident scatter path
`DeviceScalarPoissonAssembler`, already used for the *linear* case elsewhere in the
codebase, per its docstring).

### 4. The real learning — why this is in the tutorial

This is the professor closing the loop on a pattern that's been building since A1's
own performance corner and P1's cost-model chapter: **"getting the right answer" and
"getting it efficiently" are separate engineering problems, and a working tutorial
implementation is often deliberately the *simple, slow* version first.** B1's Newton
loop was written entirely in host numpy so that the *mathematics* (residual,
Jacobian, weighted integrals) would be transparent and easy to verify — not because
numpy is the right long-term choice. The explore task is teaching you to *notice*
that gap yourself (nothing in the file's docstring hides it — it says outright "this
Newton loop re-assembles the weighted mass on the HOST every step") and, crucially,
to recognize that the fix isn't a novel idea you have to invent from scratch: the
codebase's "closure-hook" variable-coefficient kernel pattern (`*_var` kernels
taking a per-Gauss-point `kq` array) was *already built for exactly this kind of
problem*, just for a different physics case (spatially-varying material properties)
first. Learning to read an unfamiliar function's docstring/pattern and recognize
"this is the brick I need for a different nonlinear term" is a far more durable
skill than any specific kernel implementation — which is exactly why the task says
to *sketch*, not implement, it.

---

## Key Takeaways — B1, in plain terms

1. **Nonlinear PDEs don't need a new method, they need a loop around the old one.**
   Newton's method turns "solve one nonlinear system" into "solve a sequence of
   linear systems you already trust." Everything from A1–A6 (mesh, basis functions,
   quadrature, linear solve) is reused completely unchanged inside that loop.

2. **Verify two independent things, not one.** Spatial accuracy (does the
   discretization approximate the PDE well?) and iterative convergence (does the
   solve process actually reach the discrete answer?) are separate questions that
   can each hide the other's bugs. B1 checks both, on purpose.

3. **The "exact Jacobian → frozen Jacobian → no Jacobian" spectrum is one dial, not
   three algorithms.** Full Newton (quadratic, 5 steps), modified Newton (linear,
   16 steps), Picard (linear but slower, 22 steps) — more information per step buys
   faster convergence at a higher per-step cost. Real solvers exploit this trade-off
   deliberately (e.g. starting with cheap Picard steps, finishing with Newton).

4. **A solver's failure can be physically meaningful, not just numerically
   inconvenient.** Newton refusing to converge past `lam ≈ 6.813` isn't a bug — it's
   telling you a genuine mathematical fact (no solution exists there). Learning to
   read solver failure as *information* is a core research skill, and this exercise
   was deliberately set up as a preview of the real continuation machinery in B2.

5. **A tutorial's "slow but clear" implementation is a deliberate teaching choice,
   not the final answer.** B1's host-numpy nonlinear-term assembly was written for
   transparency, not speed — and the codebase already had the right GPU-kernel
   pattern (`*_var` closure-hook kernels) sitting nearby for exactly this kind of
   problem, waiting to be recognized and reused rather than reinvented.

6. **Across the whole B1 chapter, the recurring meta-lesson is: measure, don't
   assume.** Every explore task replaced a plausible-sounding guess with an actual
   number — convergence *rate* (not just "does it converge"), the *exact* discrete
   fold location (not just "somewhere near 6.81"), and the *real* assembly/solve
   cost balance (not just "the solve is probably the bottleneck"). That habit is the
   throughline connecting B1 back to every A-tutorial before it.
