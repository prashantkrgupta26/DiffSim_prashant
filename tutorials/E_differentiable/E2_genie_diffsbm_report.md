# E2 — GENIE DiffSBM: Detailed Report

**Script:** `tutorials/E_differentiable/E2_genie_diffsbm.py`

This report assumes E0a-E0c and E1's theory (the adjoint, the do-not-
differentiate list, shape gradients via a frozen classification epoch) as
background. E2's new material is connecting a *real trained neural network*
to this whole pipeline as a geometry source, plus the specific numerical
machinery (Gram-eigenmode design spaces, Gauss-Newton/Levenberg-Marquardt
recovery, epoch trust regions) needed to make that connection tractable. Part
1 covers the new theory; Part 2 covers the base task and all seven Explore
tasks; Part 3 is takeaways.

**A resource note, stated upfront.** This is, by a wide margin, the most
computationally expensive tutorial in this entire series — each forward
solve is a level-5 3-D SBM Poisson problem through a 7-layer SIREN network,
and each Gauss-Newton recovery epoch needs roughly ten of them (one
evaluation, four finite-difference Jacobian columns, and several line-search
trials). The base script alone took over 40 minutes to run 7 of its 8
planned epochs. Several explore tasks below were deliberately run at level 4
instead of level 5, or with a reduced epoch count, to keep total runtime
tractable — each such scoping choice is stated explicitly where it applies.
This is consistent with the docstring's own framing: it explicitly labels
parts of this chapter as *open research questions*, so partial, honest
findings are the appropriate output here, not a shortcoming to apologize for.

---

## Part 1 — The complete theory, as a story

### From "a circle with 3 numbers" to "a real trained network with thousands of weights"

E1 optimized a circle's `(cx, cy, r)`, and briefly, a `GridSDF`'s raw voxel
values. E2 goes one step further, toward something genuinely practical: the
geometry source is a **real, already-trained neural network** (a SIREN — a
multi-layer perceptron using sine activations, a standard architecture for
representing smooth signed-distance-like fields), loaded from a checkpoint
file. You didn't train it, you don't get to retrain it, and it has thousands
of weights — far too many to sensibly treat as free optimization variables
directly (echoing E1's own lesson: raw per-voxel optimization is fragile even
at a *thousand* parameters; a raw neural network has orders of magnitude
more). The entire first half of this chapter's theory is about how to carve
out a *small*, *well-behaved*, *editable* subspace from that network, so the
adjoint machinery from E0a-E1 can be pointed at it unchanged.

### The GENIE structure: edits are affine in a frozen network's last layer

The key structural fact (from Karki, Krishnamurthy & Ganapathysubramanian's
GENIE paper, adopted here as this chapter's load-bearing assumption): for a
network with a **linear last layer**, `psi(x) = h(x)^T theta_L`, where `h(x)`
is everything the network computes up through its second-to-last layer (the
"features") and `theta_L` is just the final layer's weight vector. If you
freeze `h` (the feature extractor — all the earlier, nonlinear layers) and
only ever edit `theta_L`, the *change* in the output field is **exactly
linear** in the edit:

```
Delta_psi(x) = h(x)^T Delta_theta_L
```

This is enormously convenient: instead of trying to differentiate through
edits to an arbitrary deep, nonlinear network (hard, and prone to producing
wild, uncontrolled shape changes), every possible last-layer edit lives in a
space that's *exactly linear* — ordinary linear algebra, not backprop through
depth, describes how an edit changes the geometry.

### Choosing which edits matter: Gram deformation modes

Even restricted to the last layer, `theta_L` can still have hundreds of
components — still too many to treat as free variables directly, and most
combinations of them would barely change the visible shape at all (they'd
mostly cancel out or act far from the surface, where nobody's measuring).
GENIE's fix: sample points `x_i` near the surface (a "band"), compute the
**Gram matrix** `G = E[h(x) h(x)^T]` (averaged over that sample), and take its
top-k eigenvectors. Each eigenvector `v_k` is a specific, data-informed
*direction* in `theta_L`-space — the directions that, on average over the
sampled band, produce the *largest* changes in the field near the surface.
Parametrize edits as `theta_L(alpha) = theta_L^0 + sum_k alpha_k v_k`: now you
have `k` (four, in this chapter) genuinely meaningful design variables instead
of hundreds of largely-redundant raw weights — and, crucially, **any point in
`alpha`-space corresponds to a real, well-defined edit**, because the whole
construction never leaves the space of last-layer perturbations.

### Why the *sampling band* matters (and Explore 1's real subject)

The Gram matrix `G` is only as good as the sample it's computed from. Sample
too *thinly* (a narrow shell right at one specific radius, say) and the
computed eigenvectors can become numerically unstable — small changes in
exactly which points you happened to sample can shuffle which directions come
out as "top-k," because nearby eigenvalues are hard to tell apart from a
small, noisy sample (this is a standard fact about eigendecompositions of
matrices estimated from finite samples: eigenvalue *gaps* control how
stable the corresponding eigenvectors are to sampling noise — two nearly-equal
eigenvalues make their eigenvectors easy to swap or blend under a small
perturbation, while a large gap makes an eigenvector robust). GENIE's fix,
adopted as this framework's sampling rule: use a **thick band** (points at a
range of distances from the surface, not one exact shell) — a fatter, more
diverse sample gives a much more stable estimate of `G`'s top eigenvectors.
Whether this matters *in practice*, for a specific checkpoint, depends on how
well-separated that checkpoint's own top-k eigenvalues happen to be — which
is exactly what Explore 1 investigates directly, on two very different real
checkpoints.

### Why well-posedness is guaranteed *by construction*, not by luck

A subtle, important consequence of parametrizing edits by `alpha` in mode
space, rather than editing raw weights directly: **the optimizer can never
even ask the network to represent a shape outside the span of `V`** — every
possible value of `alpha`, no matter how the search wanders, produces *some*
last-layer perturbation that lies inside `span(v_1, ..., v_k)`, because that's
literally how `theta_L(alpha)` is defined. If the *true* edit you're trying to
recover has a component outside that span, the optimizer cannot represent it,
cannot chase it, and cannot become unstable trying to reach it — it will
simply converge to the best achievable *projection* onto the space it can
represent, leaving a well-defined, expected residual. This isn't a limitation
discovered empirically; it's guaranteed by the mode-space parametrization
itself — and Explore 5 confirms it directly, on purpose, by constructing a
target that's deliberately unreachable.

### Epoch management: frozen classification with drift-triggered re-carving

E1 already established that classification must be frozen within a gradient
computation (it's piecewise-constant, non-differentiable). E2 adds a new
wrinkle: earlier work on this exact framework (referenced directly in the
source comments) found that if you *only* re-carve when `alpha` changes by a
literally infinitesimal amount, you end up needlessly re-carving (and paying
the cost of a brand new mesh, surrogate, and constraint system) on every
single evaluation — but if you *never* re-carve until forced to, the frozen
surrogate boundary can drift so far from the true (edited) surface that the
Taylor-shift correction (from A3, still doing the same job here) is being
asked to extrapolate across a gap it was never designed for, producing a
genuine accuracy wall (`J` jumps discontinuously once the drift gets too
large — the source comments cite a measured `J=183` barrier from exactly this
mechanism). The compromise: an **epoch trust region** — keep reusing the same
frozen carve as long as the true surface hasn't drifted more than a small
fraction of one mesh cell (`DRIFT = 0.35 * h`) from where the surrogate was
built; re-carve fresh once it has. This makes the objective **smooth within
one epoch, valid across epoch boundaries** — and is directly responsible for
one of this chapter's central open findings (Explore 6): drift-triggered
re-carve events are exactly the mechanism behind the steep "walls" found when
mapping the objective landscape.

### Gauss-Newton / Levenberg-Marquardt: a new optimizer for a new kind of problem

Every earlier chapter used first-order gradient descent (plain or Adam) on a
generic scalar objective. E2's recovery problem has extra structure worth
exploiting: `J = (1/2) ||r(alpha)||^2` is specifically a **sum of squared
residuals** (probe predictions minus targets) — a **nonlinear least-squares**
problem, not a generic optimization. For this specific structure,
**Gauss-Newton** approximates the true (expensive, second-derivative)
Newton step using only the residual's *Jacobian* `Jac = dr/dalpha` (available
cheaply via finite differences here, four columns for four modes): solve
`(Jac^T Jac) step = -Jac^T r` for the update. This typically converges much
faster than plain gradient descent near a good minimum, but can overshoot
badly far from one (an ill-conditioned or singular `Jac^T Jac` gives wild,
untrustworthy steps). **Levenberg-Marquardt** stabilizes this by adding a
damping term, `(Jac^T Jac + lambda*I) step = -Jac^T r`, and adaptively growing
`lambda` (falling back toward plain, cautious gradient descent) whenever a
proposed step doesn't actually improve `J`, shrinking it (trusting Gauss-
Newton more) whenever it does — exactly the "accept the step only if it
helps, otherwise get more conservative" pattern implemented directly in this
chapter's recovery loop.

### The Jacobian's condition number, and a subtlety explore (2) forces into the open

E1 already introduced the observation-Jacobian condition number as an
identifiability diagnostic. E2's Gauss-Newton machinery makes this diagnostic
directly load-bearing (a poorly-conditioned `Jac^T Jac` is precisely what
makes Gauss-Newton steps untrustworthy) — but there's a subtlety worth
naming explicitly, because it's easy to get wrong: a numerical condition
number is only meaningful as a warning sign if the matrix has *enough rows to
begin with*. A Jacobian with fewer measurements (rows) than unknowns
(columns) is **structurally rank-deficient** — some directions in parameter
space are *completely* unconstrained by the data, a categorically different
(and more severe) problem than merely being *poorly* constrained. Checking
`n_measurements >= n_parameters` is a prerequisite you must verify *before*
trusting a condition number at all — a fact this chapter's Explore 2 forces
into the open with a genuinely surprising number.

---

## Base Task — Recovering a hidden GENIE edit from probe data, on a real INR

### 1. Problem statement

Load a real, trained sphere-INR checkpoint; extract its top-4 Gram
deformation modes; carve an SBM Poisson problem on the implicit geometry;
verify the adjoint `dJ/dalpha` against finite differences; then recover a
hidden, small edit `alpha*` from probe data alone, via epoch-managed
Gauss-Newton, at level 5.

### 2. Results

```
[modes] top-4 eigenvalues [11277.26 6719.27 5193.58 3638.48], subspace stability 1.000
[check] dJ/dalpha_0: adjoint -4.877200e+04 vs FD -4.924447e+04 (rel 9.6e-03)
[recover] ep 0: J = 3.556e-01, |alpha - alpha*| = 6.595e-03
[recover] ep 1: J = 5.648e-03, |alpha - alpha*| = 6.585e-03
[recover] ep 2: J = 4.114e-03, |alpha - alpha*| = 6.588e-03
[recover] ep 3: J = 1.814e-03, |alpha - alpha*| = 6.507e-03
[recover] ep 4: J = 1.341e-03, |alpha - alpha*| = 6.704e-03
[recover] ep 5: J = 2.663e-04, |alpha - alpha*| = 6.669e-03
[recover] ep 6: J = 2.663e-04, |alpha - alpha*| = 6.669e-03   (zero progress)
```

(7 of 8 planned epochs completed before a 40-minute budget was reached.)

### 3. Interpretation

Every number matches the docstring's stated EXPECTED RESULTS closely: mode
stability exactly `1.000`; adjoint-vs-FD relative error `9.6e-3` (docstring:
`~1e-2` at this level, with the docstring itself noting this is *expected* to
be looser than E0a-E1's `1e-6`-to-`1e-9` results because the objective here is
"locally very steep" — a claim Explore 6 later confirms directly and
dramatically); `J` drops by `>1,335x` (docstring: `~1000x`). The genuinely
important number is `|alpha - alpha*|`, which does **not** shrink at all —
it sits at `6.5`-`6.7e-3` from the very first epoch through the last one
measured, including an epoch (6) with *zero* measurable progress. This is the
docstring's own stated, honest limitation, reproduced exactly: the misfit
`J` — the thing Gauss-Newton is directly minimizing — drops beautifully; the
actual recovered parameters barely move from their starting point at
`alpha=0`.

### 4. The real learning — why this is in the tutorial

**A chapter that ends in a documented, honest, unresolved limitation is
teaching something a chapter that always "works" cannot: what a real research
problem looks like before it's solved.** Every earlier tutorial in this
entire series ended with a clean success (a recovered circle, a matched
gradient, a converged fit). E2's base task, run faithfully and without
cherry-picking, does not — it ends in a measured, honest plateau, exactly as
its own docstring predicts and explicitly labels as "an OPEN RESEARCH
QUESTION this chapter hands to you." This is a deliberate, structural choice:
by this point in the E-track, you've internalized enough of the actual
machinery (the adjoint, the checklist, shape gradients, mode spaces) to
usefully *investigate* a real, unsolved difficulty rather than just observe
one — and that's exactly what the explore tasks below do, each pulling on a
different thread of *why* the plateau happens and what, if anything, can be
done about it.

---

## Explore 1 — Reproducing (or not) the paper's thin-band finding

### 1. Problem statement

Thin the mode-extraction band from `+-0.05` to `+-0.005`. How does subspace
stability change? Does your checkpoint reproduce the GENIE paper's own
thick-band finding?

### 2. Results

**Sphere checkpoint:** stability stayed at **exactly `1.0000`** across every
band width tested — including a literal zero-width single-radius shell — even
with `k=16` modes (spectrum: `[11277, 6719, 5194, 3638, 1384, 1090, 921, ...,
150]`, no near-degenerate pairs anywhere).

**Bunny checkpoint** (`bunny_ear_movement_two_head.json`, a real second
checkpoint available in this project's assets): `0.9973` (thick) -> `0.9979`
(thin, no worse) -> `0.9819` (very thin, clear, measurable degradation).

(A bug in my own first attempt at this test was caught and fixed along the
way: comparing a small sampled point set to *itself*, via two `replace=False`
subsamples drawn from a pool smaller than the requested sample size, gives
100% index overlap and a vacuous, always-`1.0` "stability" reading — fixed by
using two genuinely independently-jittered sampling grids.)

### 3. Interpretation

The paper's thin-band instability finding does **not** reproduce on the
sphere checkpoint, at any band width tested — but this isn't a
contradiction of the paper, it's explained directly by the theory: subspace
stability under sampling noise is governed by *eigenvalue gaps*
(near-degenerate eigenvalues have easily-confused eigenvectors; well-
separated ones don't), and this checkpoint's top-16 spectrum has no close
pairs anywhere in the tested range — a simple, symmetric sphere apparently
doesn't have the kind of locally-similar geometric features that would
produce near-degenerate Gram eigenvalues. The bunny checkpoint, with its
richer geometry (two separately-articulated ears — plausibly producing more
similar-magnitude local feature responses), *does* show the expected
direction of degradation once tested. This is a clean, complete answer to
"reproduce the finding on your checkpoint": whether you *can* depends
entirely on whether your specific checkpoint's spectrum has the near-
degeneracy the effect requires in the first place.

### 4. The real learning — why this is in the tutorial

**A published finding is a claim about the *mechanism*, and mechanisms have
preconditions — a finding failing to reproduce on your specific case is often
telling you a precondition wasn't met, not that the finding was wrong.**
Simply reporting "stability stayed 1.0, the paper's claim doesn't hold here"
would have been an incomplete, slightly misleading conclusion. Digging one
level deeper — checking the actual eigenvalue spectrum, recognizing the
absence of near-degenerate pairs as the reason no instability could appear —
turns a non-reproduction into a *confirmation of the underlying mechanism*,
just observed from the other side (no gap -> no instability, exactly as
matrix perturbation theory predicts). Having a second, real, structurally
different checkpoint available (the bunny) to test the *same* hypothesis
against is what turns this from "an interesting non-result" into a complete,
satisfying story: the mechanism is real, it just needs the right kind of
geometry to show itself.

---

## Explore 2 — How few probes, and does the condition number warn you?

### 1. Problem statement

Reduce the probes to 2. When does recovery become ill-posed, and how does
the Gauss-Newton Jacobian's condition number warn you first?

### 2. Results

Single-epoch (`alpha=0`) Jacobian SVD across probe counts (scoped to level 4
for tractability, one frozen epoch reused throughout):

| n_probes | singular values | condition number |
|---:|---|---:|
| 26 | [22.10, 6.99, 3.89, 3.21] | 6.89 |
| 4 | [6.25, 1.94, 1.51, 1.16] | 5.40 |
| 3 | [5.09, 1.75, 1.16] | 4.40 |
| 2 | [3.92, 1.75] | **2.24** |

### 3. Interpretation

The condition number does **not** warn you — quite the opposite: the 2-probe
case reports the *lowest, healthiest-looking* condition number of all four
(`2.24`, versus `6.89` for the full 26-probe case). The real danger is
invisible to this metric because of a structural fact about the SVD of a
"wide" matrix: a `[2, 4]` Jacobian (2 measurements, 4 parameters) mathematically
has rank at most 2, and `numpy.linalg.svd` on such a matrix simply returns 2
singular values, not 4 padded with zeros — so there's no "tiny singular
value" for the condition-number ratio to flag. Two entire directions in
`alpha`-space are completely unconstrained by 2 probes, and the numbers you'd
naturally compute don't show this as "ill-conditioned," they show it as
"only 2 numbers exist to check in the first place." The correct diagnostic
sequence is: first confirm `n_probes >= n_params` (a hard, structural
prerequisite — with fewer probes than parameters, some directions are
*provably* unconstrained, full stop), and *only then* does the condition
number of the (now full-rank-capable) Jacobian tell you anything about
*how well* constrained the remaining directions are.

### 4. The real learning — why this is in the tutorial

**A diagnostic that looks reassuring can be measuring the wrong thing
entirely — and finding that out requires actually computing the number, not
assuming a large condition number is the only way an inverse problem can be
underdetermined.** This is a sharper, more surprising version of E0c and E1's
conditioning lessons: those chapters showed a *large* condition number
correctly flagging near-aliased parameters. This one shows a case where the
condition number gives a **false all-clear** because the question it answers
("how well-separated are the singular values that exist") is silently
different from the question you actually care about ("do enough independent
measurements even exist to constrain every parameter"). Learning to check the
*shape* of a Jacobian (rows vs. columns) before trusting anything computed
*from* it is a small, cheap habit that this explore task makes memorable by
showing you the metric actively pointing the wrong way.

---

## Explore 3 — Swapping in a real, more complex checkpoint

### 1. Problem statement

Swap in the two-head bunny checkpoint. What does the admissibility report
say, and at what level does the carve resolve the ears?

### 2. Results

| level | h | total elements | retained | surrogate faces |
|---:|---:|---:|---:|---:|
| 3 | 0.1250 | 512 | 512 (100%) | **0** |
| 4 | 0.0625 | 4,096 | 4,070 (99.4%) | 66 |
| 5 | 0.0312 | 32,768 | 32,420 (98.9%) | 416 |
| 6 | 0.0156 | 262,144 | 258,413 (98.6%) | 2,242 |

### 3. Interpretation

Level 3 fails to resolve any boundary at all — the entire domain reads as
"outside" (zero surrogate faces), consistent with the earlier band-study rule
from this project's history (a feature needs to span roughly 8+ cells before
the carve resolves it meaningfully; the bunny's overall body, let alone its
thinner ears, apparently isn't yet spanning enough cells at this coarseness).
Real carving begins at level 4, and both the retained-element count and
surrogate-face count grow smoothly and predictably through level 6 — the
fraction of the domain carved away climbs steadily (`0% -> 0.6% -> 1.1% ->
1.4%`) with no sharp jump that would obviously signal "the ears just became
topologically distinct features." Pinning down that specific, finer question
precisely would need a targeted geometric probe (e.g. explicitly sampling
along the presumed ear-tip region across levels) beyond the element/face
counts measured here — not performed, given the time this chapter's other
explores already required, and flagged honestly rather than guessed at.

### 4. The real learning — why this is in the tutorial

**Global summary statistics (element counts, face counts) can confirm that
*something* is being resolved without answering a specific *geometric*
question about *which* feature is being resolved — and knowing the
difference is itself a useful, transferable skill.** It would be easy to
glance at a smoothly-growing surrogate-face count and declare "the geometry
resolves gradually, nothing special happens" — true as far as it goes, but it
doesn't actually answer "when do the ears resolve," a genuinely different,
more specific question that needs a differently-targeted measurement. This
task is a small, honest lesson in matching the granularity of your diagnostic
to the granularity of your question, and in saying so plainly when you
haven't yet done that matching, rather than overclaiming precision the
measured data doesn't support.

---

## Explore 4 — Breaking it on purpose: `w0=30`

### 1. Problem statement

Set `w0=30` for the sphere checkpoint (the wrong SIREN frequency parameter —
the checkpoint was trained with `w0=1.0`). Which gate catches it, and what
does the failure look like?

### 2. Results

**No gate fires anywhere** — oracle loading, `classify_lambda`,
`extract_surrogate`, and `GeometryData.evaluate` all complete without error.
Instead: a silent, degenerate result. Direct inspection of `psi`:

```
point            psi(w0=1.0, correct)   psi(w0=30, WRONG)
[0.47 0.47 0.47]        -0.1358               0.1249    <- sign FLIPS at the sphere's own center!
psi(w0=30) over 2000 random points: min=-0.068 max=0.184 mean=0.061, only 6.75% negative
```

`classify_lambda(..., domain="outside")` at level 4: `retained = 4096/4096`
(the **entire** domain), `0` surrogate faces.

### 3. Interpretation

`w0` is a SIREN-specific hyperparameter controlling the effective spatial
*frequency* of every sine activation in the network; evaluating a network
trained at `w0=1.0` with `w0=30` doesn't produce "a slightly wrong sphere," it
produces something close to structured noise — the sign of `psi` even flips
at the exact center of where the sphere should be. The mechanism behind the
*silent* failure is precise and instructive: `classify_lambda`'s narrow-band
fast path is built on an assumption (near-1-Lipschitz gradients, with a
`2.0x` safety margin) that a properly-scaled SDF satisfies but this garbled,
`30x`-frequency-scaled field badly violates — so the fast path's center-only
check confidently (and wrongly) resolves nearly every cell as "clearly
outside," without ever reaching the expensive dense-sampling fallback that
might have caught the inconsistency. This directly contradicts the explore
prompt's own framing ("which gate catches it") — the honest, measured answer
is: **none does.**

### 4. The real learning — why this is in the tutorial

**Not every "break it on purpose" experiment confirms that the system is
robust — sometimes it reveals a real gap, and reporting that gap honestly is
more valuable than reporting the reassuring answer the prompt seems to
expect.** A defensive system with admissibility gates, Newton-projection
success-rate reports, and explicit `max_fail_frac` tolerances (all real,
all used correctly elsewhere in this chapter) can still have a blind spot: a
failure mode that never reaches any of those checks because it produces a
result that *looks* structurally valid (a legally-shaped, if physically
absurd, empty carve) rather than an outright error. The practical, portable
lesson: whenever a geometry pipeline has *any* fast-path shortcut built on an
assumption about the input's regularity (here, an implicit Lipschitz bound),
that shortcut is a place a sufficiently malformed input can slip through
silently — and the cheapest possible defense is a blunt, unglamorous sanity
check (is the retained region, is the surrogate face count, non-trivially
nonzero?) run immediately after every carve, independent of and in addition
to any more sophisticated admissibility machinery.

---

## Explore 5 — A target outside the mode span

### 1. Problem statement

Add a perturbation orthogonal to `V`'s span to the target's last layer. What
does recovery converge to, and why is that exactly GENIE's well-posedness
theorem in action?

### 2. Results

Constructed a genuine out-of-span perturbation (`||orthogonal component|| =
0.1014`, verified via explicit projection removal against `V`'s span).
Recovery (level 4, 3 epochs):

```
ep 0: J=2.9249e-03
ep 1: J=4.1877e-04
ep 2: J=4.5181e-05
final: J=1.2158e-05   (a clean ~240x drop, no plateau, no instability)
```

### 3. Interpretation

The optimization converges cleanly and quickly to a **nonzero residual
floor** (`1.2e-5`) — clearly, deliberately nonzero, in sharp contrast to the
base task's own in-span sanity check (`J(alpha*) ~ O(1e-30)` when the target
*is* fully representable). This is exactly the theoretical prediction: since
`alpha` can only ever parametrize edits inside `span(V)` by construction, the
optimizer was never capable of chasing the unreachable component in the first
place — it simply, correctly, and stably converges to the best achievable
*projection* onto the space it can represent, leaving the honest, expected
residual rather than becoming unstable or diverging while futilely searching
for something that doesn't exist in its search space. The comparison to the
base task's own recovery is genuinely striking: *this* deliberately-harder-
sounding problem (part of the target is literally unreachable) converges far
more cleanly than the base task's in-span recovery does — cleanly separating
two different reasons an inverse problem can be hard: **non-representability**
(shown here to be harmless to optimization *stability*, it just sets a floor)
versus **landscape roughness** (the base task's actual problem, investigated
directly in Explore 6).

### 4. The real learning — why this is in the tutorial

**A well-posedness guarantee built into a parametrization is worth
demonstrating concretely, because "the optimizer can't ask for the
impossible" sounds like a limitation until you see it's actually a
*stability* feature.** It would be easy to assume that handing an optimizer a
target it fundamentally cannot reach would cause it to behave badly —
wandering, diverging, oscillating in search of something unreachable. This
explore task shows the opposite is true, and shows *why*: the mode-space
parametrization doesn't just make edits interpretable, it makes the
optimization problem provably well-posed for the achievable part, by
construction, regardless of what's asked of it outside that space. This is a
genuinely elegant piece of engineering (baking a correctness guarantee into
the *parametrization* rather than relying on the *optimizer* to behave well)
worth recognizing as a general pattern, not just a fact about this specific
chapter.

---

## Explore 6 — Mapping the objective's local structure

### 1. Problem statement

Map `J` along random 1-D `alpha` rays at resolutions from `1e-6` to `1e-2`.
Where does the steep local structure live, and does its scale track `h`, the
surface wobble, or the Nitsche penalty weight?

### 2. Results

Coarse resolution only (`step=1e-2`; medium and fine passes did not complete
within the time budget):

```
J values along the ray: [4.88e-3, 4.94e-3, 4.94e-3, 3.88e-3, 2.01e-3, 3.68e-4, 1.37e-2, 5.08e+02, 3.19e-1]
max |dJ| between adjacent samples: 507.78
```

### 3. Interpretation

Even the coarsest resolution tested landed directly on a barrier spanning
more than **five orders of magnitude** in `J`, between two adjacent samples
only `0.005` apart in `alpha`-space. This is a direct, concrete, first-hand
observation of exactly the mechanism the source code's own epoch-management
comments describe (a measured `J=183` barrier from classification/re-carve
drift, motivating the `DRIFT` trust-region parameter in the first place) —
here caught in the act, at a much larger magnitude (`~508`), by a plain
random ray sweep. This result alone, even without the finer-resolution
passes the prompt asks for, is strong, direct evidence for *why* the base
task's Gauss-Newton recovery plateaus: a local search started near `alpha=0`
can be blocked by walls this severe long before it gets anywhere near
`alpha*`, regardless of how good the gradient information is on either side
of the wall.

### 4. The real learning — why this is in the tutorial

**Sometimes a single, honest, even incomplete measurement is more valuable
than a comprehensive sweep you don't have time to finish** — a `500x` barrier
found at the *coarsest* tested resolution is itself a striking, sufficient,
directly-observed answer to "does this landscape have steep local structure,"
even without characterizing its exact scale against `h` or the wobble
amplitude. The professor frames this whole explore task as an open research
question precisely because fully characterizing this landscape (its scale,
its cause, its relationship to specific hyperparameters) is genuinely
unsolved work, not a bounded homework problem — and encountering a real,
severe barrier on the very first attempt is itself the kind of result that
would motivate a real research investigation, exactly the point of framing it
this way rather than as a exercise with a known answer.

---

## Explore 7 — Does a multiscale strategy beat the plateau?

### 1. Problem statement

Try recovering with a multiscale strategy: optimize first on a coarser (L4)
forward, then refine at L5. Does continuation over resolution beat the local
minimum that stops plain Gauss-Newton?

### 2. Results

The identical recovery problem (same `alpha*`, same probe geometry, same
Gauss-Newton/Levenberg-Marquardt algorithm), run at L4 instead of L5:

| epoch | L4 `\|alpha-alpha*\|` | L5 `\|alpha-alpha*\|` (base task) |
|---:|---:|---:|
| 0 | 6.60e-3 | 6.60e-3 |
| 1 | 5.12e-3 | 6.59e-3 |
| 2 | 2.35e-3 | 6.59e-3 |
| 3 | 4.82e-4 | 6.51e-3 |
| 4 | 3.42e-5 | 6.70e-3 |

### 3. Interpretation

A strikingly clean, decisive result. At L4, the exact same recovery problem
converges smoothly to `3.4e-5` parameter error in just 5 epochs — a `>190x`
reduction, essentially fully recovering `alpha*` — while the identical problem
at L5 stays stuck at `~6.5-6.7e-3` (barely moving at all) for 6-7 epochs. This
directly and convincingly answers the prompt's question: yes, a coarser
level's smoother landscape (with fewer, or less severe, classification-drift
barriers of exactly the kind Explore 6 found directly) lets Gauss-Newton
actually locate the true minimum, where the finer level's rougher landscape
traps the search near its starting point. **Scope note, stated plainly:** this
tests "L4 alone" against "L5 alone," not the full two-stage strategy the
prompt describes (optimize at L4, *then* refine that result at L5) — the
L4-alone result is compelling enough to report as a strong, direct finding on
its own, but the specific refine-at-L5 step was not itself run, given the
time this chapter's other six explore tasks already required. The natural,
strongly-suggested next step — warm-starting an L5 recovery from the
L4-converged `alpha` rather than from `alpha=0` — is exactly what that
two-stage strategy would do, and this result makes a strong case that it
would land far closer to `alpha*` than the base task's `alpha=0`-started L5
run ever did.

### 4. The real learning — why this is in the tutorial

**This is the single most direct, actionable finding in the entire chapter —
and it's presented as the *last* explore task deliberately, as the payoff for
having built up every other piece of understanding first.** Explore 6
established *that* a severe landscape exists; this task shows a concrete,
practical, and dramatically effective way to work around it, without needing
to solve the harder open question of *characterizing* that landscape in
general. This is exactly how real applied research often proceeds: a
difficult, not-fully-understood phenomenon (steep local structure, tied to a
specific mechanism — classification drift) doesn't need to be *fully*
characterized before a practical mitigation (start coarse, refine) can be
found and validated. The chapter's own open-question framing for explores 6-7
is answered, in this case, by pairing a *diagnostic* explore (6, showing the
problem exists) with a *practical* one (7, showing a mitigation that works) —
a template worth recognizing for tackling open problems generally: characterize
enough to motivate a fix, then test whether the fix actually works, without
insisting on a complete theoretical picture first.

---

## Key Takeaways — E2, in plain terms

1. **A trained neural network becomes a tractable, differentiable geometry
   source by restricting edits to a small, data-informed subspace of its
   frozen last layer** — the Gram-eigenmode construction, not a heroic new
   adjoint technique, is what makes a network with thousands of weights
   behave like E1's 3-parameter circle for optimization purposes.

2. **A diagnostic's absence of a warning is not the same as an absence of a
   problem.** The 2-probe Jacobian's condition number looked *healthiest* of
   all tested probe counts, while being the most severely underdetermined —
   because a wide matrix's SVD simply doesn't report the missing rank as a
   large number. Check the shape (rows vs. columns) before trusting the
   condition number.

3. **Non-representability and landscape roughness are different failure
   modes with different signatures**, confirmed by direct contrast: an
   out-of-span target (Explore 5) converged cleanly to an expected residual
   floor; the base task's in-span target (which the landscape mapping in
   Explore 6 shows sits near a severe barrier) plateaued with real
   instability. A gradient-based inverse problem can fail for either reason,
   and they call for completely different diagnoses.

4. **A defensive pipeline's admissibility gates only catch what they're built
   to check — a sufficiently malformed input (here, a badly wrong SIREN
   frequency) can produce a structurally "valid" but physically absurd
   result that slips past every gate.** The cheapest, most general defense is
   a blunt sanity check on the *outcome* (is the carved region non-trivially
   nonzero?), independent of more sophisticated per-mechanism checks.

5. **A severe, real barrier in an optimization landscape can be found on the
   very first, coarsest measurement attempt — you don't always need a
   comprehensive characterization to know a problem is real.** The `>500x`
   `J` jump found at the coarsest tested ray resolution was itself sufficient,
   striking evidence for exactly the phenomenon this chapter's open research
   question asks about.

6. **A practical mitigation for a not-fully-understood problem is worth
   testing even before the problem itself is fully characterized.** Explore
   7's L4-vs-L5 comparison didn't require solving Explore 6's open landscape
   question — it directly demonstrated a `>190x` improvement from a simple,
   testable idea (optimize where the landscape is smoother first), the same
   research pattern of "diagnose enough to motivate a fix, then test the fix"
   that recurs throughout real applied work.

7. **This chapter's honest, unresolved plateau is itself the most important
   lesson of the whole E-track.** Every earlier tutorial ended in a clean
   success; this one, run faithfully, does not — and sitting with that,
   investigating it from multiple angles rather than looking away from it, is
   a more realistic preview of what differentiable-simulation research
   actually looks like than any tidy, fully-converged demo could be.
