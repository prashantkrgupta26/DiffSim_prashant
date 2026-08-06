# A4 — Mixed p1/p2 Elements and the Minimum Rule: Detailed Report

**Script:** `tutorials/A_foundations/A4_mixed_elements.py`

This report covers the base task and all four Explore tasks in A4. For each: the
problem statement, what was measured, what it means, and the underlying lesson the
tutorial is teaching. A "Key Takeaways" section closes it out. A real bug was found
and fixed in the base script before any of this data was collected — covered
first, since it changes the numbers for everything downstream.

---

## The bug: p2 was a thin column, not a half-domain

Before any of the results below were trustworthy, `solve()`'s p2/p1 mesh split had
to be fixed. It computed the split as:

```python
G = 1 << level
xs = tree.anchors()[:, 0] / float(G)
p_elem = np.where(xs < 0.5, 2, 1).astype(np.int8)
```

`tree.anchors()` lives on a **fixed** grid (`lmax(dim)=31`, independent of
`level`), so dividing by `1 << level` doesn't normalize to `[0,1)`. Measured
directly: instead of a true left half, p2 covered only the single leftmost
element column — `6.25%` of the mesh at level 4, shrinking to `1.56%` at level 6
(`2^-level` of the mesh). The fix: the function had already computed the
*correctly*-normalized `anchors = tree.anchors() / tree.anchors().max()` one line
above and simply never used it — `xs` now reads that instead of recomputing from
`G`. Verified this gives exactly `50.00%` p2 at every level. This mattered a great
deal: the pre-fix numbers showed rich-L only `1.5x` more accurate than rich-R,
with that margin visibly *shrinking* under refinement — a compelling-looking but
entirely spurious story, caused by the p2 fraction itself shrinking every level,
not by any real numerical-analysis effect. All results below use the corrected
mesh.

---

## Base Task — Mixing element orders in one mesh

### 1. Problem statement

Build a single mesh where the left half (`x < 0.5`) uses quadratic (p2) elements
and the right half uses linear (p1) elements. Solve the A1 Poisson MMS on it,
under two manufactured solutions that are mirror images of each other: **case
rich-L**, where the field has high curvature near `x=0` (inside the p2 region) and
decays toward `x=1`; **case rich-R**, the same field mirrored, so the curvature
sits near `x=1` (inside the p1 region) instead. The question the chapter is
built around: does it matter *where* you spend a fixed p-refinement budget?

### 2. Results (corrected mesh, true 50/50 split)

```
rich-L (p2 on left): errors [1.251e-03, 3.108e-04, 7.751e-05]   orders 2.01, 2.00
rich-R (p2 on left): errors [6.426e-03, 1.614e-03, 4.039e-04]   orders 1.99, 2.00

rich-R / rich-L error ratio: 5.14x (level 4), 5.19x (level 5), 5.21x (level 6)
```

### 3. Interpretation

Both cases converge at order **~2, not 3** — the p1 half caps the *global rate*
no matter where the p2 budget sits ("a chain is as slow as its weakest link," per
the docstring). But the *constant* tells a much richer story: rich-L is a stable,
even mildly *growing*, ~5.2x more accurate than rich-R at every level. With a true
half-domain split, most of rich-L's high-curvature region (near `x=0`) sits well
clear of the p2/p1 interface at `x=0.5`, so it keeps genuine p2 accuracy; only a
fixed-width band near the interface itself gets taxed down to p1 accuracy (this
band mechanism is exactly what Explore (a) measures directly).

### 4. The real learning — why this is in the tutorial

**p-refinement buys you a better constant everywhere it covers, but the global
convergence *rate* is set by the weakest element order present, full stop.** This
is a genuinely important practical lesson for adaptive/local refinement strategy:
you cannot "hide" a coarse region behind a locally-refined one and expect the
overall accuracy to reflect the refined region — the coarse region's rate
dominates asymptotically no matter how small it is. The corollary, which the
rich-L vs. rich-R comparison drives home, is that this doesn't mean local
refinement is useless — spending your p2 budget in the right place still buys a
substantial, durable multiplicative accuracy improvement (5x here). The chapter is
teaching you to hold both facts at once: rate is global and set by the weakest
link, but *where you place* your local refinement still matters enormously for the
constant in front of that rate.

---

## Explore (a) — Constraint counts at the p2/p1 interface

### 1. Problem statement

Inspect `build_constraints`'s output directly: how many constrained ("hanging")
degrees of freedom does the p2/p1 interface actually produce, and what do their
interpolation rows look like? The chapter frames this as drawing the "constrained
edge mode" by hand.

### 2. Results

Level-4 mesh (128 p2 elements, 128 p1 elements): total nodes `Nn=697`, free
nodes `681`, hanging nodes `16`. All 16 hanging nodes sit exactly on the interface
line `x=0.5`; zero hanging nodes anywhere else. There are 16 p2 elements whose
right face borders the interface (one per row of the mesh) — giving **exactly one
hanging node per interface element**. Sample interpolation row, for the hanging
node at `(0.5, 0.03125)`:

```
(0.5, 0.03125) = 0.5 * (0.5, 0.0) + 0.5 * (0.5, 0.0625)
```

### 3. Interpretation

The minimum rule doesn't discard any p2 degrees of freedom wholesale — it
**reclassifies exactly one specific mode per interface element**: the quadratic
edge's mid-face node, and pins it to the plain linear average of the edge's two
corner nodes. That's precisely a hat (linear) function riding on top of what would
otherwise be an independent quadratic bump mode. In other words, the p2 element's
third degree of freedom along that one shared face is *forced* to reproduce
exactly what a p1 element would compute there — the p2 element keeps its full
richness everywhere else (its interior node, its other three edges), but that one
face is surgically constrained down to p1 behavior.

### 4. The real learning — why this is in the tutorial

**"The minimum rule" stops being an abstract phrase and becomes a countable,
literal fact about the linear algebra once you inspect `build_constraints`
directly.** This is the mechanistic heart of the whole chapter: everything else in
A4 (the capped global order, the interface-placement sensitivity in Explore (b),
the connection to the m1a Neumann-band finding in Explore (c)) is a downstream
*consequence* of this one precise fact — one hanging DOF per interface element,
linearly interpolated. The professor wants you to have actually looked at the
constraint matrix's nonzero pattern with your own eyes, not just accepted "p2
elements next to p1 elements lose their quadratic face modes" as a slogan. This
is also a template for a broader habit: when a piece of machinery (constraints,
in this case) is doing something conceptually important, go read its actual
output on a small, hand-checkable case before reasoning about its consequences at
scale.

---

## Explore (b) — Strip `|x-0.5|<0.25` vs. half-domain, same p2 budget

### 1. Problem statement

Move the p2 region from the left half to a symmetric strip straddling the
interface, `|x-0.5| < 0.25` — the same total p2 element count (`50%`), just placed
differently. Does half the p2 budget still buy most of the benefit, regardless of
where it's placed?

### 2. Results

| region | rich-L errors (lv 4/5/6) | rich-R errors (lv 4/5/6) | p2 frac |
|---|---|---|---|
| p1 (baseline) | 6.537e-3 / 1.642e-3 / 4.110e-4 | same | 0.0 |
| half (`x<0.5`) | 1.251e-3 / 3.108e-4 / 7.751e-5 | 6.426e-3 / 1.614e-3 / 4.039e-4 | 0.5 |
| strip (`\|x-0.5\|<0.25`) | 5.811e-3 / 1.463e-3 / 3.663e-4 | same (symmetric) | 0.5 |
| p2 (ceiling) | 2.183e-4 / 2.733e-5 / 3.418e-6 | same | 1.0 |

### 3. Interpretation

No — half the budget does **not** buy most of the benefit; placement dominates
budget. The interface-centered strip gives identical, modest gains (~11% over p1)
for *both* rich-L and rich-R, since it's symmetric and never reaches either
field's peak-curvature edge. `half`, at the exact same budget, swings from ~5.2x
better than p1 (rich-L, well-placed) to just ~1.7% better (rich-R, misplaced) — a
more than 4x outcome swing at identical cost. A second, subtler effect also shows
up: `half`'s gap to the full-p2 ceiling *widens* under refinement for rich-L
(5.7x at level 4 -> 22.7x at level 6), because full p2 sustains order 3 while the
interface-capped `half` case is permanently stuck at order 2 — a genuinely
different, and worse, long-run trend than a fixed constant-factor gap.

### 4. The real learning — why this is in the tutorial

**The question "how much refinement budget do I have" is secondary to "where does
the solution actually need it."** This is the practical, actionable version of the
base task's lesson: two meshes with *identical* cost (same p2 element count) can
differ by a factor of 4-5x in accuracy purely based on placement. In real adaptive
FEM, this is exactly the argument for *error-driven* (not uniform or naively
centered) refinement — you have to know where the solution is actually rich
before spending your budget there, or you risk landing in the "strip" regime:
correctly-sized, plausibly-placed (centered on the geometric feature of interest —
the interface — rather than the solution feature of interest), and yet capturing
almost none of the available benefit. The widening full-p2 gap under refinement is
an additional, sharper warning: a locally-capped rate isn't just "somewhat worse,"
it's a *permanently growing* gap relative to what full refinement would have
given you, which only gets more expensive to ignore as you refine further.

---

## Explore (c) — Connection to the m1a Neumann-band finding (4b)

### 1. Problem statement

An earlier, unrelated finding in this project's development history (documented in
`docs/dev/m1a-deferred-findings.md`, finding 4b) studied a **different** chapter's
problem: an SBM Neumann boundary condition where a band of p2 elements next to the
true (immersed) boundary needs to carry Hessian information for the boundary
condition to be accurate, and that band's *outer* face borders ordinary p1
elements. The task: re-read that finding and restate it using this chapter's now-
concrete vocabulary (interface constraints, the minimum rule).

### 2. Results (a restatement, not a new numerical run)

Finding 4b(c), verbatim: *"the real mechanism: band elements carrying the Hessian
must be decoupled from the minimum-rule p1 trace constraints — face-neighbor rings
of 1-2 layers cap at order ~1 (0.85/1.04), 4 layers restore 2; NODE-adjacent growth
... is clean at >=3 layers (2.01/2.04), marginal at 2 (1.78/1.37)."*

### 3. Interpretation

Restated in A4's vocabulary: that finding is describing the *exact same*
minimum-rule interface constraint measured directly in Explore (a) — just in a
different geometric setting (a curved SBM band boundary rather than a flat p2/p1
line) and for a different reason for needing accuracy (a Hessian-dependent
boundary condition, rather than a manufactured solution's curvature). If the p2
band is too thin, the minimum-rule-constrained trace strip (the same "one hanging
DOF per interface element, linearly interpolated" mechanism from Explore (a))
*overlaps* the very elements that are supposed to be carrying the Hessian
information the boundary condition needs — so the correction and its own
constraint cancel each other out, capping order at ~1. A thick-enough band (>=3-4
layers, depending on the exact adjacency definition used) leaves a "clean" core of
p2 elements between the true boundary and the constrained rim, restoring order 2.
Explore (b) is the *steady-state, no-boundary-condition* version of this identical
mechanism: `half`'s rich region sits several elements clear of the interface
(effectively "thick enough"), while `strip` puts the constrained interface band
directly on top of the region that would have benefited (effectively "too thin"),
and the payoff nearly vanishes either way.

### 4. The real learning — why this is in the tutorial

**The same underlying mechanism shows up in completely different-looking
problems, and recognizing it is a transferable skill, not a one-off fact about one
chapter.** This explore task is deliberately designed to make you connect two
findings that, on the surface, look unrelated — one about accuracy near a curved
immersed boundary, one about a flat interior mixed-order interface in a steady-
state MMS problem. The unifying insight ("the minimum rule taxes a band near any
p2/p1 interface, and that band must not overlap the region that needs the extra
accuracy") is far more valuable than either individual finding, because it's the
kind of pattern-recognition that lets you *predict* a new failure mode in a
problem you haven't yet tested, rather than only explaining ones you've already
observed. This is precisely how the deferred-findings document itself is meant to
be used across the project — not as a one-time bug report, but as an accumulating
vocabulary of mechanisms that recur.

---

## Explore (d) — Performance corner: assembly time, p1 vs. p2 vs. mixed

### 1. Problem statement

p2 elements cost more DOFs and more per-element work than p1 elements. Measure
assembly time for a pure-p1 mesh, a pure-p2 mesh, and this chapter's 50/50 mixed
mesh, all at level 6. Is the mixed mesh's assembly time the mean of the two pure
cases, and why would per-bin (per-polynomial-degree) kernel launches make that
true?

### 2. Results

| config | DOFs | nnz | assemble median/min (ms) |
|---|---:|---:|---:|
| p1-uniform | 4,225 | 37,249 | 1.083 / 1.028 |
| p2-uniform | 16,641 | 263,169 | 6.694 / 5.766 |
| mixed 50/50 | 10,401 | 149,281 | 3.768 / 3.635 |

### 3. Interpretation

The measured DOF ratio (p2/p1) is `3.94x`, **not** the docstring's own predicted
`(3/2)^2 = 2.25x` — same-level p2 node density actually approaches `4x` p1
density asymptotically (`(2n+1)^2 / n^2 -> 4`), so the docstring's own stated
prediction doesn't hold as a same-level DOF-count ratio (worth flagging rather
than trusting silently). Assembly time ratio (`6.18x`) is closer to the
predicted per-element `(9/4)^2 = 5.06x` (matching the `nbf^2` scatter-size ratio,
`81/16`). Most interestingly: mixed assembly time (`3.768ms`) matches
`mean(p1, p2) = 3.889ms` to within `3%` — essentially exact agreement. This is
because `volume_triplets` (called by `assemble_csr`) loops `for pv, b in
dm.bins.items(): ... wp.launch(...)` — one specialized GPU kernel launch **per
polynomial-degree bin**, each processing only that bin's own disjoint element
subset. A Warp kernel needs uniform per-thread work (fixed `nbf`, `nqp`), so a
mixed-p mesh literally cannot share one kernel launch across both element types;
the total cost is therefore *exactly* `(p1 elements x p1 cost) + (p2 elements x p2
cost)`, with no cross term — which collapses to the mean of the two uniform costs
precisely because this mesh happens to split 50/50.

### 4. The real learning — why this is in the tutorial

**A performance model is only trustworthy once you understand the actual code path
producing the cost, not just the abstract element-count arithmetic.** The
"mixed assembly = mean of the two pure costs" result isn't a coincidence or an
approximation — it's an exact structural consequence of how the assembly loop is
written (one Warp kernel per bin), and the professor wants you to trace that
connection yourself rather than just observe the numeric coincidence. This also
quietly reinforces the meta-lesson threading through the whole tutorial series:
the docstring's own `(3/2)^d` DOF-scaling prediction turned out to be measurably
wrong (`3.94x` vs. predicted `2.25x`) — a reminder, even within a single
performance-corner task, that a written-down prediction is a *hypothesis* to test,
not a fact to assume, even when it comes from the same source you're otherwise
trusting.

---

## Key Takeaways — A4, in plain terms

1. **Global convergence rate is set by the slowest (lowest-order) element present,
   no matter how small a fraction of the mesh it covers.** p-refinement in one
   region cannot rescue the overall *rate* if any part of the mesh stays coarse —
   though it still substantially improves the *constant*.

2. **The "minimum rule" is a precise, countable fact, not a slogan:** exactly one
   hanging DOF per p2/p1 interface element, pinned to the linear average of its
   two corners. Every other finding in this chapter is a direct consequence of
   this one mechanism.

3. **Placement beats budget.** Identical p2 element counts produced a 4-5x swing
   in accuracy purely from where the p2 region sat relative to the solution's
   curvature — and a misplaced budget's disadvantage *grows* under refinement,
   not just stays constant.

4. **The same mechanism recurs across unrelated-looking problems.** The
   interface-constraint band that caps accuracy in this chapter's flat p1/p2 split
   is the identical mechanism documented in a completely different context (a
   curved SBM Neumann boundary) elsewhere in this project's history — recognizing
   that connection is a more durable skill than either finding alone.

5. **A stated performance prediction is a hypothesis, not a given — even one
   written directly in the same file you're testing.** The docstring's own DOF-
   scaling estimate didn't survive measurement; only the mixed/mean assembly-time
   relationship, traced back to the actual per-bin kernel-launch structure, turned
   out to be exactly right.

6. **This chapter's bug is itself a lesson.** The corrected mesh told a materially
   different, cleaner story (a stable 5x margin) than the buggy one did (a
   spuriously shrinking 1.5x margin) — a reminder that a plausible-looking, even
   internally-consistent result can still be silently wrong, and the only defense
   is checking that the setup code actually does what it claims to.
