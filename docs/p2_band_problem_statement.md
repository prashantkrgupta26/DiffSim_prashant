# Local p-Refinement Bands for the Shifted Boundary Method: Problem Statement, Measured Results, and Open Questions

*Self-contained research brief. Everything needed to reproduce or extend
this study is defined in this document — no access to our codebase is
required. All numbers quoted are measured (July 2026, GPU-native octree
FEM implementation, float64 throughout).*

---

## 1. Setting and motivation

### 1.1 The Shifted Boundary Method in one page

Let Ω be a domain with smooth boundary Γ, immersed in an axis-aligned
Cartesian octree mesh (unit box, elements of size h at refinement level
L, i.e. h = 2⁻ᴸ). Elements are classified inside/outside; the retained
mesh's boundary defines the **surrogate boundary** Γ̃ — a staircase of
axis-aligned element faces near Γ.

For each surrogate quadrature point x ∈ Γ̃, define the **distance vector**
d(x) = p(x) − x, where p(x) is the closest-point projection onto Γ.
Boundary data given on Γ is transferred to Γ̃ through the Taylor
**shift operator**

    S u (x) = u(x) + ∇u(x)·d + ½ dᵀ H(u)(x) d + ...        (1)

truncated at the order the local polynomial space can represent
(gradient term for p1, +Hessian term for p2).

**Neumann conditions** (the case studied here). With prescribed flux
q = ∇u·n on Γ (n the true normal), the SBM Neumann face contribution on
Γ̃ is, per the local-p-refinement formulation:

    ∫_Γ̃ w [ −(S∇u)·ñ + a q(p(x)) ] dS̃,     a = n·ñ,          (2)

where ñ is the surrogate (axis-aligned) face normal and the shift is
applied to the **normal flux component only**. The scalar a = n·ñ is the
**area correction**: it accounts for the staircase surface measure
exceeding |Γ| (for a circle rasterized to axis-aligned faces the excess
is exactly 4/π − 1 ≈ 27%). Both ingredients are load-bearing; see §6.

### 1.2 The order barrier and the band idea

With p1 elements the element Hessian vanishes, S is exact only to first
order, and the *boundary* consistency error caps global convergence at
**O(h)** regardless of interior accuracy. Elevating only a **band** of
elements adjacent to Γ̃ to p2 lets those elements carry the ½dᵀHd term,
restoring **O(h²)** — at a cost that scales with the boundary, not the
volume.

**The question**: how must this band be constructed (which adjacency
relation? how many layers?) for second order to actually materialize?
The empirical answer, developed below, is that band construction is
**load-bearing, not a free parameter**, and the mechanism is the
interaction between the band and the p1/p2 interface constraints.

---

## 2. Discrete setting (precise definitions)

- **Mesh**: 2:1-balanced octree over the unit box; in this study the
  background mesh is *uniform* at level L (so the band study isolates
  p-adaptivity from h-adaptivity). Element size h = 2⁻ᴸ.
- **Classification**: an element is retained iff it lies in the exterior
  of the immersed ball (domain = box ∖ ball). We use the "λ = 1
  keep-all" rule: retain every element not fully inside the ball (no
  small-cut trimming), so Γ̃ ⊂ Γ's exterior side.
- **Spaces**: Q1 (trilinear/bilinear) on p1 elements, Q2 on p2 elements,
  C⁰ across interfaces.
- **Minimum rule** (the standard p-FEM interface constraint): on a face
  shared by a p2 and a p1 element, the trace of the p2 side is
  constrained to the p1 space of the face. Concretely, every p2 node on
  such a face (edge midpoints, face centers) is an affine combination of
  the p1 vertex values. The same applies at hanging (h-)interfaces via
  the usual conforming interpolation; chains (a p-constrained node whose
  masters are h-constrained) are resolved by substitution to free nodes.

### 2.1 Band constructions under test

Let F₀ = the set of retained elements owning at least one surrogate
face. Define two dilation operators on element sets:

- **Face-adjacent dilation** 𝒟_f(S): S plus every retained element
  sharing a **face** (codim-1 entity) with an element of S.
- **Node-adjacent dilation** 𝒟_n(S): S plus every retained element
  sharing **any mesh node** (face, edge, or corner contact) with an
  element of S. (This is what "cells touching" means; in 2-D it adds the
  4 diagonal neighbors; in 3-D it adds 12 edge- and 8 corner-neighbors
  over 𝒟_f.)

A **band of N layers** is 𝒟ᴺ(F₀) (N applications), with every element in
the band elevated to p2 and everything else p1.

---

## 3. Experimental protocol

- **Geometry**: circle (2-D) / sphere (3-D), radius r = 0.25, centered
  in the unit box. Exterior problem.
- **Manufactured solution** (smooth, non-vanishing normal derivative on
  Γ):
  - 2-D: u\* = sin(πx) cos(πy), f = 2π²u\*
  - 3-D: u\* = sin(πx) cos(πy) sin(πz), f = 3π²u\*
- **Boundary conditions**: strong Dirichlet from u\* on the outer box;
  SBM Neumann (2) on the ball with exact q = ∇u\*·n evaluated at the
  *true* boundary foot points p(x).
- **Primary observable**: L2(Ω) error against u\* (masked to the true
  exterior). Orders from consecutive-level ratios.
- **Contrast baseline**: identical pipeline with **no band** (p1
  everywhere).

A remark on observables: L2(Ω) is primary. Flux observables require a
*flux-carrying* manufactured solution — the sin·cos family above has
**zero net flux** through the ball (∮ q dS ≈ 10⁻¹⁶) and can neither test
the area correction nor distinguish flux errors (§6).

---

## 4. Measured results

### 4.1 The adjacency × thickness sweep (2-D, levels 5–7)

| Band construction | Layers N | Measured L2 orders | Verdict |
|---|---|---|---|
| face-adjacent 𝒟_f | 1–2 | 0.85 / 1.04 | capped at ~1 |
| face-adjacent 𝒟_f | 4 | ≈ 2 | recovers, wastefully |
| node-adjacent 𝒟_n | 2 | 1.78 / 1.37 | marginal, non-monotone |
| **node-adjacent 𝒟_n** | **3** | **2.01 / 2.04** | **clean second order** |

**Mechanism (working explanation).** The Hessian-carrying elements at Γ̃
must be *decoupled from the minimum-rule p1 trace constraints*. A p2
element whose nodes are pinned to p1 traces on too many sides cannot
represent the quadratic it exists for — the ½dᵀHd term in (1) is
silently crippled and first-order boundary error survives. Node
adjacency closes the diagonal gaps that face rings leave, so the
constraint front is pushed 3 genuine layers back; face rings need ~4
layers to achieve the same separation.

**Consequences for the write-up of any local-p method**: (i) the
adjacency semantics must be stated explicitly ("cells touching" =
node-adjacent); (ii) claims that results are "insensitive to layer
count" hold only *above the threshold* — below it, layer count is
decisive and the failure is a silent order-1 cap, not a blow-up.

### 4.2 Ruling out the formulation first (method point)

Before the sweep, two formulation-level suspects were eliminated:

1. **Is (2) itself consistent?** Isolation test: feed *exact* data
   (u\*, q at true foot points) through the identical pipeline at p1.
   Result: clean 2.00/2.00. The formulation is exonerated; the defect
   had to be discrete.
2. **Would shifting the tangential flux component too help?** No —
   measured *worse*. Shifting only on the surrogate-normal side
   introduces a one-sided O(d) inconsistency. The asymmetric-looking
   form (shift the normal component only) is the consistent one.

We record these dead ends deliberately: at order-of-accuracy debugging,
*isolate formulation from discretization before believing either is at
fault*, and beware "improvements" that symmetrize a formula at the cost
of its consistency order.

### 4.3 Full 2-D convergence (node-band(3) vs p1-only)

Exterior circle, levels 5–8 (h = 1/32 … 1/256):

| Level | band L2 error | order | p1-only L2 error | order |
|---|---|---|---|---|
| 5 | 6.62e-4 | — | 6.78e-3 | — |
| 6 | 1.75e-4 | 1.92 | 4.84e-3 | 0.49 |
| 7 | 4.53e-5 | 1.95 | 2.16e-3 | 1.16 |
| 8 | 1.09e-5 | 2.06 | 1.13e-3 | 0.93 |

Second order sustained to 58k DOFs; p1-only pinned at ~1 throughout;
**104× accuracy gap at level 8**.

**Cost scaling**: band element count grows like the boundary
(~2×/level: 188 → 1416 over the sweep) while the mesh grows like the
area (4×/level). The band's *relative* cost vanishes under refinement:
a boundary-dimensional price for a domain-dimensional accuracy gain.

### 4.4 3-D results and the preasymptotic trap

Exterior sphere, node-band(3), levels 4–6:

| Level | elements across sphere | band L2 | p1-only L2 | ratio |
|---|---|---|---|---|
| 4 | ~8 | 1.99e-2 | 4.68e-3 | band **4.3× worse** |
| 5 | ~16 | 5.70e-4 | 3.52e-3 | band 6.2× better |
| 6 | ~32 | 1.38e-3 | — | see §4.5 |

At level 4 the band is *substantially worse than doing nothing*: with
the sphere spanning ~8 elements, the shift machinery performs a
second-order Taylor extension of an unresolved field — it amplifies
geometry-resolution error, at p2 prices.

> **Engagement rule (measured): activate the p2 band only when the
> geometric feature spans ≳15 elements.** Below that, p1 is both cheaper
> and more accurate. Any adaptive driver should gate band activation on
> a feature-resolution criterion, not apply it unconditionally.

### 4.5 The level-5 dip — an open anomaly

The 3-D error sequence 1.99e-2 → 5.70e-4 → 1.38e-3 is **non-monotone**:
the implied L4→L5 order is ~5.1 (impossible for the scheme) and L5→L6 is
negative. Working hypothesis: **error cancellation** — during the
preasymptotic-to-asymptotic transition the dominant error component
changes sign, and level 5 happens to sit near the zero crossing, making
it fortuitously accurate. Under this hypothesis the L6 value is on the
true asymptotic curve and the L6→L7 pair would show ~2.

The L7 verification (~2.8M DOFs) is currently blocked by linear algebra,
not discretization: the nonsymmetric SBM system at 356k DOFs (level 6)
already defeats Jacobi-preconditioned BiCGStab (divergence) and stock
classical/aggregation AMG (best relative residual 1.2e-1 after 4000
iterations); sparse direct on GPU solves level 6 in ~14 s but exhausts a
48 GB card at level 7. See open question Q6.

---

## 5. Summary of established facts

1. **Node adjacency ("cells touching") with ≥3 layers** is the correct
   band construction: measured orders 2.01/2.04 (2-D), sustained to 58k
   DOFs (2.06 at level 8).
2. **Face-adjacent rings of 1–2 layers silently cap at order ~1**
   (0.85/1.04) — the failure mode is quiet degradation, not blow-up.
3. The mechanism is **minimum-rule trace-constraint contamination** of
   the Hessian-carrying elements, not the SBM Neumann formulation
   (exonerated by an exact-data isolation test).
4. **Preasymptotic engagement rule** (3-D, measured): the band must not
   be engaged below ~15 elements across the feature; at ~8 it is 4.3×
   worse than p1.
5. Band cost scales with the boundary; its relative cost vanishes under
   refinement.
6. Flux observables need a flux-carrying manufactured solution; the area
   correction a = n·ñ removes a ~27% (= 4/π − 1) staircase inflation for
   the circle.

---

## 6. Traps we hit (recorded so you don't)

1. **Zero-net-flux MMS.** sin(πx)cos(πy) has ∮_Γ q dS ≈ 3e-16 on the
   centered circle. Any flux-based verification against it is vacuous —
   including an apparent "discrete conservation identity" we briefly
   believed. With a flux-carrying quadratic MMS: correction-on flux
   error ~7.6%|F\*| at level 5 (consistent with the scheme's order);
   correction-off adds the full 27% staircase inflation.
2. **Symmetrizing the shift.** See §4.2 — consistency order beats
   aesthetic symmetry.
3. **Preasymptotic p2.** See §4.4 — higher order on unresolved geometry
   is actively harmful, and the crossover is quantifiable.

---

## 7. Open questions (research-grade)

**Q1 — Sharp threshold, empirically.** Map order vs (adjacency, N) at
finer granularity. Is node-adjacent N=3 sharp, or does the threshold
depend on the feature's surface-to-volume ratio (ellipse aspect sweep;
channel walls vs closed curves)? Deliverable: the order-vs-N figure for
both adjacencies — this figure is the caveat any local-p paper needs.

**Q2 — Threshold, theoretically.** Conjecture: second order requires
that every surrogate-face element possess a fully unconstrained Q2
sub-basis in the surrogate-normal direction, and 𝒟_n³ is the minimal
dilation guaranteeing this on a 2:1-balanced octree. Can this be proven
as a rank condition on the minimum-rule constraint operator restricted
to the band? A counting argument on constrained vs free p2 nodes per
boundary element may suffice for the lower bound.

**Q3 — Local predictor.** For a 1-layer band, does the per-element count
of p1-constrained p2 nodes predict the *local* error distribution? If
yes, band construction could be driven by the constraint graph directly
(elevate until every boundary element's constraint count clears the
threshold) instead of uniform N-layer dilation — potentially thinner
bands in convex regions.

**Q4 — 3-D adjacency refinement.** Node adjacency in 3-D adds edge- and
corner-neighbors over face adjacency. Which carry the coupling? Test an
*edge-adjacent* intermediate (share ≥ an edge). If edge adjacency at N=3
suffices, the band is meaningfully smaller than the node-adjacent one;
if corners matter, that is itself a noteworthy fact about constraint
percolation on octrees.

**Q5 — The dip.** Test the error-cancellation hypothesis of §4.5 without
level 7: sweep the radius r (hence elements-across) at fixed level 6.
Under the hypothesis, the error as a function of resolution should show
a smooth dip near ~16 elements across, with sign structure visible in a
signed error functional (e.g. ∫(u−u\*)).

**Q6 — The solver wall (separate but blocking).** The nonsymmetric SBM
Poisson system (Nitsche/shift face terms on a symmetric diffusion core)
at ≥350k DOFs defeats stock AMG. What preconditioner respects the
structure — e.g., AMG on the symmetric core with the face terms treated
as a low-rank-ish correction, or field-split between band and interior?
Until solved, 3-D convergence studies stop one level short of
comfortable asymptotics on a single GPU.

**Q7 — p3 extension.** With a third-order shift (add ⅙ dᵀ(∇H)d²), what
is the required band for a p3 layer nested inside the p2 band? The
minimum rule now acts on two interfaces (p3/p2 and p2/p1); does the
threshold compose (3+3), or interact?

---

## Appendix A — Reproduction checklist

Any immersed-octree FEM code with (i) 2:1 balance + hanging-node
constraints, (ii) mixed Q1/Q2 with minimum-rule interfaces, and (iii)
SBM Neumann per (2) can reproduce §4 directly from the definitions
above. Checklist: unit box; ball r = 0.25 centered; exterior retained
(keep-all); uniform level L background; strong outer Dirichlet from u\*;
exact q at true foot points; L2(Ω) masked to the true exterior; band =
𝒟ᴺ(F₀) with the adjacency under test; report consecutive-level orders
for band and p1-only. The 2-D level-8 case is ~58k DOFs (workstation
scale); 3-D level 6 is ~356k DOFs (needs a direct solver or a working
preconditioner, per Q6).

## POSTSCRIPT (2026-07-06): §4.5's anomaly resolved — benchmark degeneracy

The non-monotone L4–L6 sequence was an **alignment artifact of r = 0.25**:
the sphere's cardinal tangent planes coincide with mesh face planes at
every refinement level (dyadic radius in a unit box), producing
knife-edge classifications whose error impact varies erratically with
level. A radius sweep at fixed L6 (the Q5 experiment) exposed it: errors
are a clean ~1.6–2.2e-4 for r ∈ [0.15, 0.27] and the degenerate r = 0.25
sits at 1.38e-3. The corrected ladder at **r = 0.27** is monotone —
1.95e-3 / 5.68e-4 / 1.91e-4, orders 1.78 / 1.57 — with L7 as the
asymptotic confirmation. **Add to §6's trap list: never choose feature
dimensions that are dyadic multiples of the mesh spacing.** Q5 is hereby
answered; the other open questions stand.
