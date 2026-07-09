# The p2 Band Around a Shifted Boundary: Adjacency, Thickness, and When It Pays Off

*A pedagogical record of what we tried, what failed, what worked, and the
measured 2-D/3-D results. Everything below is reproducible with
`benchmarks/band_study.py`; the raw findings live in
`docs/superpowers/m1a-deferred-findings.md` (entries 4b, 4b(e), 4b(f)).*

---

## 1. The setting: why a p2 band at all?

We solve Poisson/Neumann problems on octree meshes with the **Shifted
Boundary Method (SBM)**: the true boundary Γ is replaced by a *surrogate*
boundary Γ̃ made of mesh faces (a staircase), and boundary data is
transferred from Γ to Γ̃ by a Taylor expansion along the distance vector
**d** (the "shift"):

```
S u = u + ∇u·d + ½ dᵀ (Hess u) d + ...
```

- With **p1 elements**, the element Hessian is zero: the shift is
  first-order only, and the boundary error caps the whole solution at
  **order 1** no matter how accurate the interior is.
- With **p2 elements near the boundary**, the element can carry the
  ½dᵀHd term, and second-order convergence becomes possible — *without*
  paying for p2 everywhere.

So the idea (from the group's local-p-refinement draft) is a **band of p2
elements** hugging the surrogate boundary, p1 everywhere else. The
question the student is exploring: **how do you grow that band, and how
thick must it be?** This turns out to be *load-bearing, not a free
parameter*.

---

## 2. The two adjacency semantics (the crux)

When you grow a band "N layers" outward from the surrogate faces, there
are two different meanings of "neighbor":

| Semantics | Definition | Other names |
|---|---|---|
| **Face-adjacent** | element B is a neighbor of A if they share a **face** (2D: an edge; 3D: a quadrilateral face) | face rings, von Neumann neighborhood |
| **Node-adjacent** | element B is a neighbor of A if they share **any node** — face, edge, *or corner* contact | "cells touching", Moore neighborhood |

In 2D the difference is the 4 diagonal (corner) neighbors; in 3D it is the
12 edge-neighbors + 8 corner-neighbors that face-adjacency misses. The
local-p draft's language ("cells touching") means **node adjacency** — and
the distinction is not cosmetic. It decides whether the method converges
at order 2.

### Why it matters: the minimum rule contaminates thin bands

At a p2/p1 interface, hanging/interface constraints follow the **minimum
rule**: the trace on the shared face is constrained to the *lower*-order
side (p1). A p2 band element whose nodes are pinned by p1 traces on too
many sides cannot actually represent the quadratic it exists to carry.
The Hessian-carrying elements must be **decoupled** from the p1 trace
constraints by enough interior p2 neighbors. Thin bands — especially
face-adjacent ones, which leave diagonal gaps — keep the boundary
elements handcuffed to p1 traces, and the ½dᵀHd term is silently
crippled.

---

## 3. What we tried, in order (the diagnosis)

This section is the honest history — including the two dead ends —
because the *process* is the lesson.

**Attempt 0 — suspect the formulation.** Convergence stalled at order ~1
with a thin band, so the first suspect was the Neumann shift formula
itself (the draft's Eq. 21: shift only the *normal* flux component, area
correction a = n·ñ on the data term). We tested it in isolation by
feeding *exact* surrogate-point data through the pipeline at p1:
**clean order 2.00/2.00.** Eq. 21 exonerated. *Lesson: isolate the
formulation from the discretization before blaming either.*

**Attempt 1 — "improve" the formulation anyway.** We tried fully
shifting the tangential term too (an S-grad on the surrogate-normal side
only). This made things **worse** — it introduces a one-sided O(d)
inconsistency. Reverted. *Lesson: a formula that looks less symmetric on
paper (shift only the normal component) can be the consistent one; check
consistency order, not aesthetics.*

**Attempt 2 — the real mechanism: band thickness × adjacency.**
Systematic sweep (exterior-disk MMS, levels 5–7):

| Band construction | Layers | Measured orders | Verdict |
|---|---|---|---|
| face-adjacent rings | 1–2 | 0.85 / 1.04 | capped at ~1 |
| face-adjacent rings | 4 | ~2 | recovers, but wasteful |
| **node-adjacent** | 2 | 1.78 / 1.37 | marginal |
| **node-adjacent** | **≥ 3** | **2.01 / 2.04** | **clean order 2** |

Two conclusions the student should internalize:

1. **Node adjacency is the right growth semantics.** Face-adjacent rings
   need ~4 layers to do what node-adjacent does at 3, because the
   diagonal gaps re-expose boundary elements to p1 traces.
2. **The draft's "insensitive to layer count" claim needs a caveat**: it
   is insensitive *once thick enough*, and the adjacency definition must
   be stated explicitly. Below the threshold, layer count is decisive.

---

## 4. Comprehensive 2-D results (exterior circle, r = 0.25)

Configuration: exterior-disk MMS, λ = 1 keep-all classification,
node-adjacent band of 3 layers, versus a p1-only baseline. L2 error on Ω.

| Level | band L2 error | band order | p1-only error | p1 order |
|---|---|---|---|---|
| 5 | 6.62e-4 | — | 6.78e-3 | — |
| 6 | 1.75e-4 | **1.92** | 4.84e-3 | 0.49 |
| 7 | 4.53e-5 | **1.95** | 2.16e-3 | 1.16 |
| 8 | 1.09e-5 | **2.06** | 1.13e-3 | 0.93 |

- The band sustains **second order to 58k DOFs**; p1-only is pinned at
  ~order 1 at every scale. At level 8 the accuracy gap is **104×**.
- **Cost scaling** (the practical argument for the band): band elements
  grow like the *boundary* (~2×/level: 188 → 1416) while the mesh grows
  like the *area* (4×/level). The band's relative cost **vanishes under
  refinement** — you pay a boundary-dimensional price for a
  domain-dimensional accuracy upgrade.

---

## 5. 3-D results (exterior sphere, r = 0.25) — and the preasymptotic trap

Same configuration, node-band(3), levels 4–6:

| Level | elems across sphere | band L2 error | p1-only error | band vs p1 |
|---|---|---|---|---|
| 4 | ~8 | 1.99e-2 | 4.68e-3 | **4.3× WORSE** |
| 5 | ~16 | 5.70e-4 | 3.52e-3 | 6.2× better |
| 6 | ~32 | 1.38e-3 | — | (see dip note) |

Two big lessons here:

### 5a. The preasymptotic rule

At level 4 the sphere spans only ~8 elements, and the band is **4.3×
worse than plain p1**. The shift machinery *amplifies* error when the
geometry itself is unresolved: a second-order Taylor extension of a
poorly-resolved field is extrapolating garbage, expensively.

> **RULE (measured): engage the p2 band only once the feature is
> ~15+ elements across.** Below that, p1 is both cheaper and more
> accurate.

### 5b. The L5 dip — an open question for you

The 3-D sequence 1.99e-2 → 5.70e-4 → 1.38e-3 is **non-monotone**: the
implied L4→L5 order is ~5.1 (impossibly high) and L5→L6 is *negative*.
Our working hypothesis: **error cancellation at L5** — the dominant error
component changes sign during the preasymptotic-to-asymptotic
transition, and L5 happens to sit near the zero crossing, making it look
fortuitously accurate. The honest asymptotic rate needs the L6→L7 pair.

L7 status (~2.8M DOFs): the *mesh/constraint build* now takes minutes
(after the 2026-07-05 lookup vectorization — 656× on constraint
construction), but the **linear solve is the wall**: the nonsymmetric
SBM system defeats both our fused Jacobi-BiCGStab (diverges) and stock
AMG configurations (best relres 1.19e-1 after 4000 iterations), and the
GPU direct solver (cuDSS) runs out of memory at 2.8M 3-D DOFs on a 48 GB
card. The L7 point — and with it the dip verdict — awaits a multi-GPU
solve or a bespoke preconditioner. *This is a genuinely open thread.*

---

## 6. Two traps we fell into (so you don't)

1. **The zero-flux MMS trap.** Our first flux observable used the
   sin·cos MMS — which has *zero net flux* through the disk
   (F\* ≈ 3e-16). It can neither test the area correction a = n·ñ nor
   distinguish flux errors, and an apparent "conservation identity" we
   celebrated was the same artifact. Use a **flux-carrying MMS** for any
   flux claim. With one: correction-on gives ~7.6% |F\*| flux error at
   level 5 (discretization order); correction-off adds the ~27%
   (4/π − 1) staircase area inflation — which is exactly the geometric
   series you'd predict for a circle rasterized to axis-aligned faces.
2. **Observable choice.** The primary observable is **L2 error on Ω**
   (this matches the draft's own Fig. 6 evidence). Flux is a *secondary*
   observable that needs the care above.

---

## 7. Reproducing everything

```bash
# 2-D sweep, levels 5-8 (minutes)
python benchmarks/band_study.py 2 5 6 7 8

# 3-D sweep, levels 4-6 (L6 ~20 min: 356k DOFs, direct GPU solve)
python benchmarks/band_study.py 3 4 5 6
```

Results append to `band_study_results.txt`. The band construction lives
in `p2_band(...)` (node-adjacent growth, `n_layers=3` default); the
adjacency semantics are exactly the ones in §2.

---

## 8. Questions to explore

1. **Sweep adjacency directly**: modify `p2_band` to face-adjacent
   growth and reproduce the order-1 cap at 1–2 layers, then the recovery
   at 4. Plot order vs layers for both semantics on one figure — that
   figure *is* the paper's caveat.
2. **Where exactly is the threshold?** Node-adjacent 2 layers gave
   1.78/1.37 (marginal). Is the transition sharp at 3, or does it depend
   on the surface-to-volume ratio of the feature (circle vs ellipse vs
   channel)?
3. **The minimum-rule mechanism, visualized**: for a 1-layer band, count
   how many of each boundary element's p2 nodes are constrained by p1
   traces. Does that count predict the local error?
4. **3-D edge vs face vs corner**: in 3-D, node adjacency adds 12 edge +
   8 corner neighbors over face adjacency. Which matter? Try an
   *edge-adjacent* intermediate (share an edge or face, not just a
   corner) — is it enough at 3 layers, or do corners carry real coupling?
5. **The L5 dip** (§5b): if you can get the L7 point (or an L6.5 via a
   different radius r that shifts the elements-across count), does the
   error-cancellation hypothesis hold? A radius sweep at fixed level is
   the cheap version of a level sweep at fixed radius.
6. **Cost accounting**: measure band-element count vs level in 3-D and
   confirm the boundary-scaling argument (should be ~4×/level in 3-D
   against 8×/level for the volume).
