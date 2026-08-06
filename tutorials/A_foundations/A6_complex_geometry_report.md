# A6 — Complex Geometry: Detailed Report

**Script:** `tutorials/A_foundations/A6_complex_geometry.py`

This report covers the base task and all four Explore tasks in A6. For each: the
problem statement, what was measured, what it means, and the underlying lesson the
tutorial is teaching. A "Key Takeaways" section closes it out.

---

## Base Task — Carving domains by composition

### 1. Problem statement

Solve the same A1 Poisson MMS on three different "carved" domains, all built by
combining signed-distance oracles (CSG operations) rather than hand-building a
conforming mesh for each shape: (1) a **channel** — the region inside a rectangular
box, the first non-square domain in this series, and the shape of every later
flow problem; (2) a **plate with a hole** — the region *outside* a disk (the
`domain="outside"` flag, the standard setup for obstacle/exterior problems); (3) an
**icosphere** — a genuinely triangulated 3-D surface, run through the same
`TriMesh`/BVH code path a real STL file from CAD would take. The chapter's central
claim: geometry in this stack is a plug-in, not a mesh — swapping any of these
three in requires zero changes to the solver.

### 2. Results

```
channel (inside a box):        errors [2.522e-04, 1.373e-04]   order 0.88
plate minus disk (outside):    errors [1.024e-03, 2.493e-04]   order 2.04
icosphere STL-path (outside):  errors [1.961e-02, 4.138e-03]   order 2.25
```

Matches the docstring's own EXPECTED RESULTS closely (`0.88 / 2.04 / 2.25`) — no
bug.

### 3. Interpretation

The two smooth-boundary carves (disk, icosphere) both recover the expected order-2
SBM accuracy this whole tutorial series has been building toward. The channel is
the deliberate outlier, at order `0.88` — a box has **corners**, where the signed-
distance function is not differentiable (the gradient jumps discontinuously) and
the closest-point projection is *set-valued* (multiple equally-close points exist
right at a corner), which breaks the smoothness assumption the Taylor shift (from
A3) fundamentally relies on. This is flagged explicitly in the docstring as the
single most important number in the whole file — "READ THIS ONE CAREFULLY" — and
it's the thread the rest of the chapter's explore tasks pull on.

### 4. The real learning — why this is in the tutorial

**Geometric composability (CSG) is powerful, but it inherits the worst smoothness
property of any primitive you compose.** This is the base task's real message,
and it's a direct extension of A3's Taylor-shift lesson: A3 taught you that the
shift correction needs the geometry to be locally smooth to work; A6 shows you
what happens when it isn't — not a crash, not an obviously wrong answer, just a
quietly degraded convergence order. The professor picked a box specifically
because it's the simplest possible shape with a smoothness defect (a corner),
making the failure mode as isolated and easy to reason about as possible before
the later explore tasks push into messier real-world geometry (a real 3-D scan).

---

## Explore (a) — Compose channel MINUS disk

### 1. Problem statement

Build `Intersection(Box(channel), Complement(Sphere))` — a channel with a
cylindrical obstacle removed, i.e. exactly D-track's flow-past-a-cylinder
geometry. The prompt says to "verify order 2" on it.

### 2. Results

```
channel minus disk: errors [3.182e-04, 1.387e-04]   order 1.20   (levels 5/6)
```

A control check (running plain `Box` alone through the identical harness)
reproduced the base script's `0.88` exactly, confirming the implementation itself
is correct — the `1.20` is a genuine measurement, not a bug in the exploration
code.

### 3. Interpretation

The prompt's expectation does not hold: measured order is `1.20`, not 2.
Removing a disk from the channel's interior doesn't touch the channel's own four
corners — the same corner-degradation mechanism from the base task persists in
the composed shape essentially unchanged. The disk removal *does* help somewhat —
`1.20` is meaningfully better than the plain channel's `0.88` — but it improves
the *constant*, not the fundamental rate-capping mechanism, because the actual
cause (the box's corners) was never addressed by adding a smooth obstacle
elsewhere in the domain.

### 4. The real learning — why this is in the tutorial

**Composing a smooth shape into a domain does not "fix" a smoothness defect that
was already present elsewhere in that domain.** This is a natural, easy
mental-model trap: flow-past-a-cylinder *looks* like a "nicer," more
realistic, more interesting geometry than a bare rectangular channel, and it's
tempting to assume its added geometric richness would improve accuracy generally.
It doesn't — the box corners are exactly as non-smooth after the disk is removed
as before, and the measured order proves it. This task is deliberately testing
whether you'll trust the prompt's stated expectation or actually check it — a
direct continuation of A5's lesson about citing a claim versus measuring it,
applied here to a *geometric* prediction instead of a *performance* one.

---

## Explore (b) — Corner-effect diagnosis (mask near-corner elements)

### 1. Problem statement

The base task's order-0.88 channel result is attributed to corner effects.
Verify that diagnosis directly: re-measure the L2 error, but exclude all
quadrature points within `k` element-widths of any of the box's four corners.
The prompt's expectation: smooth order-2 behavior should reappear once the
corners are excluded from the measurement.

### 2. Results

| exclusion radius | order (levels 5/6) |
|---|---:|
| none (full domain) | 0.88 |
| 1h | 0.88 |
| 2h | 0.87 |
| 3h | 0.86 |
| 5h | 0.82 |
| 10h | 0.49 |
| 20h | degenerate — mask empties almost the entire (small) channel domain |

### 3. Interpretation

Contrary to the prompt's expectation, masking away from the corners does **not**
recover order 2 — order barely moves at all out to `5h`, and gets actively
*worse* at `10h` (before the measurement becomes physically meaningless at `20h`,
since the exclusion zones start overlapping and consuming the whole small
channel). This means the corners' damage is **not** simply a local contribution to
the error norm that can be filtered out after the fact by ignoring nearby
quadrature points. Through the *coupled linear system* — the same stiffness
matrix that ties every degree of freedom together — a badly-conditioned corner
region pollutes the discrete solution more globally than its immediate
neighborhood; elliptic PDEs offer no guarantee that a local numerical defect
stays local in its effect on the rest of the field. The corner is still,
unambiguously, the right *root cause* (the smooth-only cases from the base task
converge cleanly) — but the appropriate fix has to attack the corner geometry
itself, not the way the error is measured afterward. Two concrete fixes follow
naturally: round the corner (blend the box's SDF with a small-radius fillet via a
smooth-min `Union`), or add local h-refinement concentrated at the corners to
better resolve the region where the closest-point projection is ambiguous.

### 4. The real learning — why this is in the tutorial

**A numerical defect's *cause* can be perfectly local while its *effect* is
global — and conflating the two leads to fixing the wrong thing.** This is a
subtle and genuinely important point about how finite element (and more broadly,
any globally-coupled discretization) accuracy works: you cannot generally "opt
out" of a bad region's influence just by not looking at it in your error metric,
because the linear solve doesn't respect your choice of measurement region — it
solves the whole coupled system at once. This task is deliberately set up so that
the "obvious," plausible-sounding fix (just don't count the bad part) fails, in
order to force you toward the *actually correct* category of fix (change the
geometry or the discretization near the defect itself). This is a much sharper
and more memorable lesson than simply being told "elliptic problems have global
coupling" as an abstract fact.

---

## Explore (c) — The Stanford bunny (real STL-path test)

### 1. Problem statement

Get a real triangulated mesh from the Stanford 3D Scanning Repository
(`bun_zipper.ply`, the classic benchmark scan), load its vertices and triangles,
wrap them in `TriMeshOracle`, scale into `[0,1]^3`, and rerun the icosphere case's
solve pipeline on it instead. Report the resulting error floor and explain it
using the **sagitta (faceting) bound** — the geometric error inherent in
approximating a curved surface with flat triangular facets.

### 2. Results

Downloaded the real `bun_zipper.ply` (`graphics.stanford.edu/pub/3Dscanrep/`;
35,947 vertices, 69,451 triangles), scaled to fit `[0,1]^3` (max extent `0.6`,
centered at `0.5`):

| level | DOFs | error |
|---:|---:|---:|
| 3 | 707 | 1.499e-02 |
| 4 | 4,744 | 3.053e-03 |
| 5 | 34,546 | 7.498e-04 |
| 6 | 263,518 | 1.781e-04 |

Orders: `2.30, 2.03, 2.07` — no plateau through level 6 (which alone cost `785s`
to solve). Mean/max triangle edge length after scaling: `0.0057 / 0.0189`.
Sagitta estimate (`s^2/8R`): roughly `4e-5` for smooth body regions (`R~0.1`) up
to `~2.2e-3` for thin, sharply-curved features like the ears (`R~0.02`, using the
larger max edge length).

### 3. Interpretation

No error floor was actually observed through level 6 — order stays clean at
roughly `2.0-2.3` the whole way. The reason connects directly back to the sagitta
concept the task asks about: the octree element size `h` at level 6 (`0.0156`) is
still *larger* than the mesh's own mean facet size (`0.0057`) — the background
solver mesh hasn't yet been refined finer than the geometry file's own resolution,
so discretization error (which keeps shrinking with `h`) is still the dominant
term, not the fixed geometric faceting error. That said, the level-6 error
(`1.78e-4`) already sits *inside* the broad sagitta estimate range computed above,
which hints that thin, highly-curved features (the ears) may already be
facet-limited even while the smooth body of the bunny is still octree-limited —
but confirming a genuine, unambiguous global floor would require pushing to level
7 or 8, which was judged too expensive/risky to attempt (level 6 alone consumed
13 minutes, and the level 5->6 solve-time growth was already steep enough to risk
the same kind of OOM outcome A5 hit at its own level 6).

### 4. The real learning — why this is in the tutorial

**Your solver cannot be more accurate than the geometry you feed it, and finding
that ceiling in practice is itself expensive.** This is the tutorial's most
direct confrontation with what real, non-idealized geometry data (a CAD export, a
3-D scan) actually looks like, after five prior chapters of clean analytic shapes
(spheres, boxes, icospheres). The professor deliberately picked a shape (the
bunny) with both large smooth regions (the body) *and* thin, sharply-curved
features (the ears) at very different local curvature scales, specifically so
that "the error floor" isn't a single clean number but a genuinely complicated
question about *which part* of the geometry becomes facet-limited first. That
this report couldn't fully answer "where exactly is the floor" within a
reasonable compute budget is itself an honest, representative outcome — real
geometry-limited accuracy studies often do require exactly this kind of
expensive, patience-testing mesh refinement to fully characterize, and knowing
when to stop (having gathered enough evidence to reason about the mechanism, even
without pinning the exact number) is a practical research skill in its own right.

---

## Explore (d) — Performance corner: `GeometryData.evaluate` timing vs. level

### 1. Problem statement

`GeometryData.evaluate` queries a BVH (bounding volume hierarchy) once per
surrogate-face Gauss point, to find each point's closest point on the true
geometry. What computational complexity exponent would you expect from that
description, and what do you actually measure, as the mesh (and hence the number
of query points) is refined?

### 2. Results

Isolated `GeometryData.evaluate` for the icosphere case, timed with repeated calls
(median/min of 7-9 repetitions per level) to reduce measurement noise:

| level | n surrogate faces | median | min |
|---:|---:|---:|---:|
| 3 | 192 | 0.001s | 0.001s |
| 4 | 528 | 0.009s | 0.006s |
| 5 | 1,992 | 0.014s | 0.009s |
| 6 | 7,320 | 0.075s | 0.070s |
| 7 | 28,512 | 0.148s | 0.102s |
| 8 | 112,344 | 2.506s | 2.009s |

### 3. Interpretation

Expected exponent, from the description alone, is roughly `1`: each Gauss point
is one independent BVH query against a *fixed-size* triangle mesh (the icosphere's
own triangle count never changes as the octree refines), so total cost should
scale linearly with the number of query points. Levels 3-6 are unreliable
evidence either way — repeated measurements of the *same* level transition gave
wildly different apparent exponents (`0.18` to `2.3`) run to run on this shared
machine, a clear symptom of small-workload, launch-floor-dominated noise (the same
effect documented elsewhere in this project's own performance-corner tutorials).
At the largest, most trustworthy scale measured (level 7 -> 8, comparing minimum
times to reduce noise further): the number of faces grew `3.94x`, but the time
grew `~19.7x` — an exponent of roughly `2.17`. That is clearly super-linear, not
the ~1 the "independent per-point query" mental model predicts. Something in
`GeometryData.evaluate` — plausibly the FP64 torch closest-point/region-masking
machinery, or host-device synchronization overhead — stops scaling cleanly once
the query count gets large enough.

### 4. The real learning — why this is in the tutorial

**A component's algorithmic description ("one independent query per point") is
not automatically its measured scaling behavior, and the gap only shows up at
scale.** This is a specific, sharp instance of a lesson repeated in different
forms throughout this whole tutorial series: what a piece of code is *supposed*
to do, based on reading its docstring or its high-level design, is a hypothesis
about performance, not a guarantee — exactly like A4's DOF-scaling prediction
that didn't hold, and A5's stale bottleneck citation. The specific twist here is
that the deviation only becomes visible at the *largest* tested scale (level 8) —
at smaller scales the measurement is simply too noisy to see it at all. This is a
practical, generalizable warning: a performance regression or an unexpected
scaling exponent can hide comfortably inside "probably just noise" at small
problem sizes, and only reading strongly enough at real scale reveals whether it's
a genuine, worth-investigating effect (as it turned out to be here) or actually
was noise all along.

---

## Key Takeaways — A6, in plain terms

1. **Geometry really is a plug-in in this stack** — swapping a box for a disk for
   a real 3-D scanned mesh required zero solver changes across all six results in
   this chapter, confirming the core architectural claim the tutorial opens with.

2. **A composed shape inherits the worst smoothness defect of any primitive
   inside it, and adding a smooth feature elsewhere doesn't fix that.** The
   channel's corners degrade accuracy in the base case, and still degrade it
   almost as much after composing in a smooth cylindrical obstacle.

3. **A numerical defect's cause can be local while its effect on the solution is
   global.** Masking the error measurement near the corners didn't recover
   accuracy, because the coupled linear system spreads a local defect's influence
   across the whole domain — the fix has to change the geometry or discretization
   itself, not the measurement.

4. **Real-world geometry (a 3-D scan) introduces a genuine accuracy ceiling — the
   sagitta/faceting floor — that idealized analytic shapes never reveal**, and
   different parts of the same real shape (a bunny's smooth body vs. its thin
   ears) can hit that ceiling at very different mesh resolutions.

5. **Both a stated correctness expectation ("verify order 2," "order should
   reappear") and a stated performance model ("one query per point should scale
   linearly") turned out to be wrong when actually measured in this chapter** —
   reinforcing, twice more, the throughline of this entire tutorial series:
   measure before you trust, whether the claim is about accuracy or about cost.
