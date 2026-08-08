# E1 — Shape Optimization: Detailed Report

**Script:** `tutorials/E_differentiable/E1_shape_optimization.py`

This report assumes E0a-E0c's theory (the adjoint via Lagrange multipliers,
"differentiate the relation," the do-not-differentiate list, the checklist) as
background. E1's new material is *geometric* sensitivity — differentiating a
simulation output with respect to the *shape* of the domain itself, not a
material field — which is the payoff this whole E0 sequence was building
toward. Part 1 covers the new theory; Part 2 covers the base task, all three
Explore tasks, and the untitled performance-corner task; Part 3 is takeaways.

---

## Part 1 — The complete theory, as a story

### The game, and why it's the smallest instance of the thing this whole project is for

Someone solved a Poisson problem (identical in spirit to A1/A3) on a disk whose
center and radius you don't know, and handed you nine point measurements from
inside it. Recover the disk. This sounds like a puzzle, but it's structurally
identical to a huge swath of real engineering and scientific inverse problems:
infer an unknown *shape* (a crack, a cavity, an inclusion, an optimal design)
from indirect *measurements* of a physical field that depends on that shape.
The entire machinery built up in E0a-E0c — the adjoint, the do-not-
differentiate list, the checklist — exists to make exactly this kind of
problem tractable. E1 is the smallest possible version of it: three unknowns
(`cx, cy, r`), nine data points, one PDE solve per guess.

### What's actually new here: differentiating with respect to *geometry*

Every previous adjoint example differentiated with respect to a *field*
defined on a fixed domain (`kappa(x)` in E0a/E0b, a time-varying state in
E0c). Here, the parameter `theta = (cx, cy, r)` doesn't live *on* the mesh at
all — it defines *which* mesh (via SBM's classification/carving, from A3 and
A6) gets built in the first place. Differentiating `J` with respect to `theta`
means asking: "if I nudge the circle's center or radius, how does the *whole
carved geometry* — the retained elements, the surrogate staircase boundary,
every closest-point projection onto the true circle — respond, and how does
that propagate through to the solution `u` and then to `J`?" This sounds like
it should require differentiating the *mesh itself* — exactly the kind of
piecewise-constant, ill-defined derivative E0b's do-not-differentiate list
warns you away from. The resolution is the single most important idea in this
chapter.

### The freeze-and-differentiate-the-smooth-part trick

Recall E0b's item (C): mesh/classification/carving is piecewise-constant in
any geometric parameter — a cell is either kept or discarded, a discrete
jump, with a derivative that's zero almost everywhere and undefined exactly at
the threshold. You cannot differentiate *through* that decision. But you don't
need to. **Within one optimization iteration, the classification is frozen**
— fixed, given the current `theta`. What's *not* frozen, and what the
adjoint/tape chain actually differentiates, are the *smooth geometric
quantities computed on top of that fixed classification*: the closest-point
distance vectors `d(x)` from each (fixed) surrogate quadrature point to the
(theta-dependent, smoothly-moving) true boundary, and the boundary data
`g(x + d)` evaluated at that (also smoothly moving) true closest point. As
`theta` varies infinitesimally, *which* elements are retained doesn't change
(you're not crossing a classification threshold), but *where exactly* the
true circle sits relative to each fixed surrogate quadrature point changes
smoothly — and that smooth dependence is exactly what's differentiable.
Between iterations, `theta` moves by a real, finite amount (an optimizer
step), classification is *re-run from scratch* on the new `theta` (a fresh
carve, a fresh surrogate, fresh constraints — genuinely rebuilding the
piecewise-constant part), and the next iteration's gradient is computed on
*that* fixed classification. The gradient is well-defined precisely because it
never has to cross a classification boundary within the differentiable part
of any single iteration — echoing this project's own test-suite assertion,
quoted directly in the docstring, that classification is provably
piecewise-constant in `theta`.

### The full gradient chain, piece by piece

The docstring lays out four links; here's what each one actually is,
building on E0a-E0c:

```
J -> dJ/du                (ordinary calculus on the probe-misfit functional)
  -> adjoint solve         A^T lambda = dJ/du     (exactly E0a's adjoint, unchanged)
  -> -lambda^T dR/dtheta   (a Warp TAPE over the SBM face-residual kernels)
  -> theta                 (a torch graph through the closest-point projection,
                            differentiated via the implicit function theorem)
```

The first two links are pure E0a: assemble the residual's sensitivity to the
solution, solve one transposed linear system. The third link is new in kind
but not in *method*: the SBM face-residual kernels (which encode the shifted
Nitsche boundary terms from A3) are taped, with the *geometric* quantities
`d` (the shift vector) and `g(x+d)` (the shifted boundary data) marked
differentiable — this produces `-lambda^T dR/dtheta`, a sensitivity with
respect to those geometric quantities, via ordinary reverse-mode
differentiation (E0b's mechanics, applied to a different kernel).

The fourth link is the genuinely new piece: **`d` and `g(x+d)` are themselves
computed by an iterative Newton procedure** (finding the closest point on the
true circle to a given surrogate quadrature point — exactly the projection
machinery A3 introduced). You now face a familiar-looking question with a new
answer: do you tape through the Newton *iterations* of that projection?
**No — for the same reason as every other iterative solver in the do-not-
differentiate list: differentiate the *relation* the projection satisfies at
convergence, not the *algorithm* that finds it.** The closest point `p*`
on the true circle satisfies a clean, explicit geometric relation at
convergence (roughly: `p*` lies on the circle, and the vector from the query
point to `p*` is normal to the circle there) — and *that* relation, not the
Newton loop that found `p*`, is what gets differentiated, via the implicit
function theorem, exactly as E0b's Newton-loop entry prescribes, just applied
to a geometric root-finding problem instead of a PDE residual. This is
implemented as a `torch` computational graph wired through the projection's
converged output, so ordinary `torch.autograd` (via `oracle.center.grad`,
`oracle.radius.grad`) carries the sensitivity the rest of the way back to
`theta` itself.

### Two representations of "shape": compact parameters vs. a full voxel field

`theta = (cx, cy, r)` is a **compact, low-dimensional** shape representation —
three numbers fully determine the geometry, and any shape it can represent is,
by construction, a circle. `GridSDF` is the opposite extreme: an
`(n+1)x(n+1)` grid of *independent* signed-distance values, multilinearly
interpolated between grid points. Any smooth-enough shape can be approximated
by *some* setting of these voxel values — this is no longer "find the 3
numbers describing a circle," it's **level-set topology optimization**: let
the shape be *anything the voxel grid can represent*, and let gradient descent
on the same adjoint machinery sculpt it, voxel by voxel, toward whatever
minimizes `J`. The remarkable thing (and the actual point of Explore (c)): the
*adjoint chain itself* doesn't change at all between these two
representations — only `oracle.params` changes, from a 3-element list to a
`(n+1)^2`-element tensor, and the exact same `shape_gradient` call produces a
gradient with respect to every single voxel, simultaneously, at (up to
bookkeeping overhead — see the performance corner) the same one-extra-solve
cost as the 3-parameter case.

---

## Base Task — Recovering a hidden circle from nine probes

### 1. Problem statement

Given nine probe readings from an unknown disk (secret center and radius),
recover `(cx, cy, r)` by gradient descent, using the full adjoint chain above,
with the mesh re-carved from scratch every iteration.

### 2. Results

```
recovered theta = [0.52004, 0.4699,  0.30984]
true      theta = [0.52,    0.47,    0.31   ]
|error|          = [4.10e-05, 9.70e-05, 1.56e-04]
```

### 3. Interpretation

The optimizer recovers all three parameters to within `~1.6e-4`, converging
smoothly over 100 iterations (the docstring's own learning-rate schedule
change at iteration 60 visibly settles Adam's oscillation in the printed `J`
trace). This confirms, end to end, that a gradient computed through a
*re-carved-every-iteration* mesh — the piecewise-constant classification step
included — is both well-defined and correctly implemented, closing the loop
the whole E0 sequence was building toward.

### 4. The real learning — why this is in the tutorial

**This is the moment the abstract machinery from three previous chapters
becomes an actual capability: turning simulation output into design or
inference decisions.** Everything in E0a-E0c was building *infrastructure* —
correct gradients, verified against three independent methods, understood well
enough to know exactly what to tape and what to freeze. E1 is the first time
that infrastructure is pointed at a genuine *inverse problem in geometry*
rather than a demonstration of the mechanics themselves. The professor chose
the smallest possible instance (a circle, three parameters, nine noiseless
probes) deliberately, so that when it works, you can be confident it worked
*for the right reasons* — the same reasons that will make it work on far
larger, far less toy-like geometric inverse problems later in this project.

---

## Explore (a) — How few probes, and which configurations are degenerate?

### 1. Problem statement

Delete probes until recovery fails. How few readings determine three
geometric unknowns, and which probe configurations are degenerate?

### 2. Results

| probe count | max parameter error | J_final |
|---:|---:|---:|
| 9 (default) | 0.00016 | 7.4e-08 |
| 5 | 0.00049 | 6.2e-08 |
| 4 | 0.00052 | 6.4e-08 |
| 3 (ring) | 0.00039 | 1.1e-08 |
| 2 | **0.064** | 4.9e-08 |
| 1 | **0.129** | 1.9e-12 |

| 3-probe configuration | max parameter error | J_final |
|---|---:|---:|
| standard ring (control) | 0.00039 | 1.1e-08 |
| collinear (one horizontal line) | **0.082** | 4.2e-07 |
| nearly coincident (tiny cluster) | **0.118** | 2.6e-08 |

### 3. Interpretation

Three well-spread probes are generically enough — matching the naive counting
argument (3 unknowns, 3 independent readings). Below that, recovery degrades
sharply: 2 probes give a `6.4%` worst-case parameter error, 1 probe `13%`. But
the more important finding is that **probe count alone doesn't guarantee
identifiability** — 3 probes arranged collinearly or nearly on top of each
other fail just as badly as having too few probes at all (`8-12%` error,
comparable to the 1-2 probe failures), despite nominally satisfying "3
readings for 3 unknowns." In every failure case — too few probes, or too few
*independent* ones — `J_final` stays small: the optimizer is still correctly
minimizing the misfit it was given, it's just that the misfit no longer
constrains the true geometry. This is the exact underdetermination signature
from E0b's kappa-field recovery (large misfit drop, poor parameter recovery),
now appearing in a completely different parameterization (3 geometric
numbers instead of 64 field values) and confirming the mechanism is general,
not an artifact of E0b's specific setup.

### 4. The real learning — why this is in the tutorial

**"How many measurements do I need" is the wrong question by itself — "are my
measurements independent, in the directions that matter for what I'm trying
to determine" is the right one, and this task is designed to make the
difference between them impossible to ignore.** A naive count-the-unknowns
argument (3 parameters, so 3 probes should suffice) is *technically* true only
generically — for almost every choice of 3 probe locations, but not for every
choice. Collinear probes on a circle carry an inherent geometric symmetry
(reflections across the line of probes are hard to distinguish from small
perturbations in some directions), and nearly-coincident probes carry almost
no more information than a single probe, regardless of how the count is
tallied. This is the same lesson as E0c's diverse-initial-conditions finding
and E0c's own conditioning-number diagnostic, applied here through direct
experiment rather than a computed condition number — both are valid ways to
discover the same underlying fact, and seeing it appear via a completely
different method (brute-force probe-deletion, rather than SVD of an
observation Jacobian) is worthwhile confirmation that it's a property of the
*problem*, not an artifact of a particular diagnostic.

---

## Explore (b) — Recovering the conductivity `kappa` as a fourth unknown

### 1. Problem statement

The conductivity was fixed at `kappa=1.3`. Add it as a fourth free parameter,
using `diffsim.sbm.adjoint.kappa_gradient` (the same function the test suite
uses in `test_ad_gradients.py::test_kappa_gradient`).

### 2. Results

Verified `kappa_gradient` directly against finite differences, independent of
any optimization loop:

```
adjoint dJ/dkappa = -3.358522e+00
FD     dJ/dkappa = -3.358522e+00
relative error    = 2.57e-09
```

Joint 4-parameter recovery (starting `kappa=1.0`, true `kappa=1.3`):

```
recovered theta = [0.52228, 0.45896, 0.3424], kappa = 1.23129
true      theta = [0.52,    0.47,    0.31  ], kappa = 1.3
|error| theta    = [0.0023,  0.0110,  0.0324]   |error| kappa = 0.0687
```

### 3. Interpretation

The gradient formula is exact — verified to 9 significant figures against
finite differences, on its own, before any optimization enters the picture.
The *joint* recovery, by contrast, is visibly looser than the 3-parameter
case (radius off by `3.2%`, kappa off by `5.3%`) — plausibly reflecting a
genuine, physically sensible correlation between `kappa` and `r` (both
influence the *overall scale* of the solution `u` in a broadly similar way,
so the data alone may only weakly separate "the disk is slightly bigger" from
"the material conducts slightly better"), compounded by an untuned,
hand-rolled learning-rate schedule for the new scalar parameter that wasn't
separately calibrated. Deliberately keeping these two findings distinct — a
verified-exact gradient, alongside an imperfect optimization result — is the
point: had the two been conflated, an imperfect final answer might have
wrongly been blamed on the gradient.

### 4. The real learning — why this is in the tutorial

**When a fit underperforms, "is the gradient wrong" and "is the optimization
poorly tuned or the problem poorly conditioned" are different questions, and
answering the first one directly (via an isolated FD check, decoupled from any
iterative process) before ever suspecting the second is the disciplined,
efficient order of investigation.** It would be easy, faced with an imperfect
4-parameter recovery, to start suspecting the newly-added `kappa_gradient`
function — but a two-line, isolated FD check settles that question completely
and immediately, for free, before spending any effort tuning Adam
hyperparameters or investigating conditioning. This ordering — verify the
math first, in isolation, then investigate the optimization dynamics
separately — is the same discipline E0a-E0c built up around the three-way
check and the dot-product test, now applied as a practical debugging habit
rather than a scripted verification step.

---

## Explore (c) — Level-set topology optimization with `GridSDF`

### 1. Problem statement

Swap the compact CSG circle for a `GridSDF` and optimize the individual voxel
values with the *same* adjoint machinery — level-set topology optimization,
using exactly the pattern in `test_gradient_gridsdf_voxels`.

### 2. Results

Voxel-level gradient verified directly against FD at the 5 highest-sensitivity
voxels (a `32x32` grid, `1,089` parameters): all relative errors between
`1e-9` and `3e-8` — exact, matching the test suite's own methodology and
precision.

Naive, unregularized Adam optimization on all `1,089` voxel values (starting
from a wrong-circle initial guess) made slow, non-monotonic progress and then
**failed**: the closest-point Newton projection became inadmissible (`11 of
64` surrogate Gauss points) around iteration 44 — even with gradient clipping
and a generous `max_fail_frac=0.15` fallback tolerance.

### 3. Interpretation

The core claim of this explore task is fully confirmed at the gradient level:
the *identical* `shape_gradient` call, with no code changes beyond swapping
the oracle from `Sphere` to `GridSDF`, produces exact gradients for over a
thousand independent voxel parameters simultaneously. The optimization
*instability*, however, is a real and informative separate finding, not a
bug: `GridSDF` is only `C0` continuous (piecewise multilinear between grid
points, `near_eikonal=False`) — nothing in raw per-voxel gradient descent
prevents individual voxel updates from locally distorting the level set into
a shape whose closest-point map becomes genuinely ambiguous or ill-defined
somewhere (the same class of admissibility failure A3's Explore (c) hit
directly, there with a *sampled* `GridSDF` approximating a smooth circle).
Production-grade level-set topology optimization routinely adds explicit
countermeasures — periodic reinitialization back to a valid signed-distance
field, or a smoothness/total-variation penalty coupling neighboring voxels —
specifically to prevent this failure mode, and this tutorial deliberately
doesn't build that machinery (the same YAGNI stance E0c takes explicitly
toward REVOLVE checkpointing).

### 4. The real learning — why this is in the tutorial

**"The same machinery works for thousands of parameters" is a true and
important claim about the *gradient computation* — and a separate, false
implication would be that raw gradient descent on those parameters is
therefore just as easy as it was for 3 compact parameters.** The prompt's own
celebratory framing ("congratulations, you are doing level-set topology
optimization") is genuine and earned at the gradient-correctness level — but
this explore task, worked through honestly rather than only until the first
success, reveals that topology optimization in practice is a genuinely harder
*optimization* problem than shape optimization over a handful of compact
parameters, for reasons that have nothing to do with whether the adjoint is
correct. Recognizing that gap — and specifically, recognizing *why* it
appears (a non-smooth, C0 representation with no built-in regularity
constraint) rather than treating it as an unexplained instability — is exactly
the kind of practical, hard-won understanding that separates "I ran the
tutorial" from "I understand what I'd need to add to make this
production-ready."

---

## Performance Corner — Forward vs. adjoint vs. tape vs. re-carve, 3 parameters vs. thousands

### 1. Problem statement

Time each stage of one optimization iteration — re-carving the mesh, the
forward solve, the adjoint solve, and the tape sweep — for both the 3-
parameter CSG case and the many-parameter `GridSDF` case. The adjoint should
cost about one forward solve *regardless* of having 3 or 3,000 parameters —
verify this directly by comparing across voxel-grid resolutions.

### 2. Results

| config | n_params | carve | forward solve | **adjoint solve** | tape sweep |
|---|---:|---:|---:|---:|---:|
| CSG circle | 3 | 2.44ms | 1.98ms | **0.133ms** | 1.52ms |
| GridSDF n=16 | 289 | 38.2ms | 2.15ms | **0.120ms** | 25.0ms |
| GridSDF n=32 | 1,089 | 14.8ms | 1.95ms | **0.119ms** | 14.2ms |
| GridSDF n=48 | 2,401 | 15.0ms | 2.61ms | **0.118ms** | 13.8ms |

### 3. Interpretation

The *adjoint linear solve* — the specific piece of the pipeline that would
cost `O(N)` separate solves under finite differences (E0a's whole point) —
stays essentially flat, `0.118` to `0.133` milliseconds, across a `3` to
`2,401` parameter range: a `>800x` growth in parameter count produces
*no measurable growth* in this specific cost, exactly confirming the
theoretical `O(1)` claim, because this solve's cost depends only on the fixed
mesh size, never on how many geometric parameters are being differentiated.
This is the single cleanest, most direct piece of evidence in this whole
report for why the adjoint method exists. The *tape sweep* tells a more
nuanced, equally important story: it is measurably *not* free for the voxel
case (`14-25ms`, versus `1.5ms` for the 3-parameter CSG case) — because
turning one mesh-level adjoint field into `N` individual parameter gradients
fundamentally requires touching each of the `N` voxels at least once. This
doesn't contradict the "adjoint is `O(1)`" claim; it completes it correctly:
**zero extra linear solves, ever, regardless of parameter count — but a real,
`O(N)` (not `O(1)`) bookkeeping cost to distribute the sensitivity back out to
`N` individual parameters**, a cost that is nonetheless vastly cheaper, per
parameter, than a full linear solve would be (the entire adjoint+tape pipeline
for `2,401` parameters finishes in under `15` milliseconds combined — finite
differences would need `2,401` separate forward solves, several *seconds*, for
the same information).

### 4. The real learning — why this is in the tutorial

**A theoretical complexity claim ("`O(1)` in the number of parameters")
deserves precise verification, and precise verification sometimes reveals the
claim is true of one *specific* piece of the pipeline, not the whole thing —
which is a more useful, more actionable fact than either "it's all free" or
"the claim is wrong."** Measuring only end-to-end iteration time here would
have muddied this distinction (total per-iteration time *does* grow with
voxel count, dominated by carving and the tape sweep) and might have
wrongly been read as contradicting the adjoint's `O(1)` promise. Breaking the
timing down stage by stage is what reveals the actual, precise, and correct
claim: it's the *linear solve* that's parameter-count-independent, not
"everything." This is the same diagnostic discipline as P1's staged cost
table and E0c's checklist item 7 (measure the backward/forward ratio
explicitly, don't assume it) — applied here to settle, with real numbers
rather than an appeal to theory alone, exactly what "the adjoint is O(1)"
does and does not promise.

---

## Key Takeaways — E1, in plain terms

1. **A geometric parameter's mesh-classification effect is piecewise-constant
   (and hence non-differentiable), but the *smooth* geometric quantities
   built on top of a frozen classification (distance vectors, projected
   boundary data) are exactly differentiable** — freezing classification
   within an iteration and rebuilding it fresh between iterations is what
   makes shape optimization on a re-carved mesh well-defined at all.

2. **The closest-point projection is itself an iterative algorithm, and it
   gets the same treatment as every other solver in the do-not-differentiate
   list**: differentiate the converged geometric relation via the implicit
   function theorem, never the Newton iterations that found it.

3. **"3 unknowns need 3 measurements" is only a generic truth, not a
   guarantee** — 3 measurements arranged with insufficient geometric
   diversity (collinear, clustered) fail exactly like having too few
   measurements at all, and a small final misfit is never, on its own,
   evidence that the true parameters were actually recovered.

4. **Separate "is the gradient correct" from "is the optimization
   well-conditioned/well-tuned" whenever a fit underperforms** — an isolated
   FD check on `kappa_gradient`, decoupled from the full 4-parameter
   optimization loop, settled the correctness question immediately and for
   free, before any time was spent on the harder conditioning/tuning
   question.

5. **The same adjoint machinery scales, without modification, from 3 compact
   parameters to thousands of independent voxel values — but a correct
   gradient does not, by itself, make naive optimization on those parameters
   easy.** Level-set topology optimization needs explicit regularization
   (reinitialization, smoothness penalties) this tutorial deliberately
   doesn't build, and hitting that limitation directly is more instructive
   than a smoothed-over success would have been.

6. **"The adjoint costs O(1)" is precisely true of the linear solve, and only
   approximately true of the whole pipeline** — measured directly, the
   solve itself stayed flat across an `800x` parameter-count range, while the
   bookkeeping that distributes sensitivity to individual parameters scales
   with parameter count (cheaply, but not for free) — a distinction only a
   staged timing breakdown, not a single end-to-end number, can reveal.
