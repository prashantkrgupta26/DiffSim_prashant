# A5 — Going 3-D: Detailed Report

**Script:** `tutorials/A_foundations/A5_three_dimensions.py`

This report covers the base task and all three Explore tasks in A5. For each: the
problem statement, what was measured, what it means, and the underlying lesson the
tutorial is teaching. A "Key Takeaways" section closes it out.

---

## Base Task — The same code, a different economy

### 1. Problem statement

Rerun the A1 box MMS and the A3 immersed-sphere SBM solve, changing only
`dim=2` to `dim=3`. The code is claimed to be dimension-generic — verify that
claim (order 2 for p1, order 3 for p2 on the box; order ~2 for the sphere once
resolved), while starting to notice what the docstring calls the real point of the
chapter: **"dimension-independence of the CODE is a design property;
dimension-independence of the COST is a myth."** Elements per refinement level grow
`8x` (not `4x` like in 2-D), a p2 element's local DOF count goes from 9 to 27, and
the direct solver's fill-in gets dramatically worse — none of which shows up by
reading the source code, only by running it.

### 2. Results

```
box  p1: errors [4.548e-03, 1.136e-03, 2.840e-04]   orders 2.00, 2.00
box  p2: errors [1.386e-03, 1.772e-04, 2.227e-05]   orders 2.97, 2.99
sphere p1, immersed: errors [3.014e-03, 1.190e-03]   order 1.34 (levels 3/4)
```

Matches the docstring's own EXPECTED RESULTS exactly — no bug.

### 3. Interpretation

The box case confirms the code is genuinely dimension-generic — no algorithmic
change was needed, order 2 and order 3 both carry over unchanged from A1's 2-D
result. The sphere's order-1.34 result is deliberately *not* yet the expected
order-2 SBM accuracy — the docstring flags this as pre-asymptotic: at level 3 the
sphere is only ~5 elements across, so the surrogate staircase is still dominated
by unresolved *geometry*, not by the underlying discretization order. This is a
planted example, not a bug: the resolved pair (levels 4-5) does give ~2.0 (per
the project's own test suite), but running that pair here would already cost
minutes — which is exactly the point Explore (c) makes explicit.

### 4. The real learning — why this is in the tutorial

**Code portability across dimensions and computational tractability across
dimensions are two completely different claims, and conflating them is a trap.**
It would be very easy to read "the code just works in 3-D, same orders as 2-D" and
conclude that going to 3-D is basically free — the whole rest of this chapter
exists to correct that conclusion with hard numbers. The professor picked this
specific pairing deliberately: the *reassuring* result (code correctness, order
match) comes first, immediately followed by explore tasks that reveal the *cost*
story is nothing like as reassuring. This ordering matters pedagogically — it's
much more memorable to first trust the dimension-generic claim and then watch it
get complicated, than to be told upfront "3-D is expensive" as an abstract
warning.

---

## Explore (a) — Staged cost table, levels 3-6

### 1. Problem statement

Extend the cost table from A1's 2-D performance corner into 3-D: time mesh-
building/constraints, assembly, and solve separately across levels 3 through 6.
In 2-D, `splu` scales close to `O(n^1.5)`; in 3-D it should be closer to `O(n^2)`,
with `O(n^{4/3})` memory. Find the level where solve overtakes assembly, and
compare that crossover point to 2-D's own crossover from A1.

### 2. Results

| level | DOFs | solve time | solve/assemble |
|---:|---:|---:|---:|
| 3 | 729 | 0.003s | 1.3x |
| 4 | 4,913 | 0.105s | 13.6x |
| 5 | 35,937 | 13.40s | 220x |
| 6 | 274,625 | **OOM-killed** | — |

Level 6 was attempted directly and the process was killed by the operating
system's out-of-memory killer mid-factorization — `anon-rss` measured at `14.6GB`
on this machine's `15GiB` of RAM, before the factorization ever finished.

### 3. Interpretation

The solve-time scaling exponent measured `1.83` (level 3->4) then `2.44` (level
4->5) — consistent with, and by the second interval trending *above*, the
predicted `~O(n^2)`, plausibly reflecting early swap/memory pressure setting in
even before the outright OOM at level 6. The crossover itself is dramatic: solve
already exceeds assembly by level 3 (`n=729`, ratio `1.3x`) in 3-D, whereas A1's
own 2-D performance-corner table doesn't cross that same threshold until somewhere
between level 5 and 6 (`n` in the `1,089`-`4,225` range) — a difference of roughly
2-3 refinement levels, i.e. one to two orders of magnitude in DOF count, purely
from the dimension change. Extrapolating the measured `O(n^2)` trend from level 5
predicts a ~13-minute level-6 solve — but that number never gets tested, because
the *real* limit hits first: **memory**, not time. The direct solver's 3-D fill-in
(`O(n^{4/3})`) exhausts available RAM before the factorization gets slow enough to
simply "take a while."

### 4. The real learning — why this is in the tutorial

**In 3-D, a direct solver doesn't fail by getting annoyingly slow — it fails by
running out of memory, at a scale you can hit surprisingly easily on ordinary
hardware.** This is presented as *the* concrete justification for why an entire
other track of this project (iterative solvers) exists at all: it's not a
theoretical nicety about asymptotic complexity classes, it's a direct, measured
wall that a plausible, moderate-sized 3-D problem (274,625 DOFs — not an
enormous mesh by any standard) hits on real, commodity-scale RAM. The professor
wants this experienced directly, as an actual crash with a real memory-usage
number attached, rather than accepted as an abstract "`O(n^{4/3})` memory" fact
in a slide. The A1-vs-A5 crossover-point comparison is the quantitative anchor
that makes "3-D is worse" a specific, falsifiable claim (2-3 levels earlier, not
just "somewhat worse") rather than a vague intuition.

---

## Explore (b) — Estimate the k=4 cost (no run)

### 1. Problem statement

This library also runs on 4-D octrees (for future space-time work), and the test
suite exercises this at small scale. Without running anything, estimate the
level-4 element count and p2 DOF count for `k=4`, and estimate at what level a
`k=4` p2 problem would fit inside a 16GB GPU's memory.

### 2. Results (pure calculation)

| level | elements | p2 DOFs | nnz (~`5^4`/row) | assembled matrix (~16B/nnz) |
|---:|---:|---:|---:|---:|
| 2 | 256 | 6,561 | 4.1M | 0.07 GB |
| 3 | 4,096 | 83,521 | 52.2M | 0.84 GB |
| 4 | 65,536 | 1,185,921 | 741.2M | **11.9 GB** |

### 3. Interpretation

Level 4, `k=4`, p2 needs `65,536` elements and `1,185,921` DOFs — and just the
assembled sparse matrix (ignoring solver fill-in or host-side triplet staging
entirely) already needs `~11.9GB`, right at the edge of, or past, a 16GB GPU's
budget on its own. Level 3 fits comfortably (`~0.84GB`). This lines up with an
independent reference point elsewhere in this project's development history (the
m1a deferred-findings document mentions `~7,105` nodes as the practical scale
actually used for 4-D p2 testing) — which sits almost exactly between this
table's level 2 and level 3, consistent with level 3-ish being roughly the
practical ceiling that was actually used in practice, not just estimated here.

### 4. The real learning — why this is in the tutorial

**Each additional spatial dimension doesn't just add cost, it compounds the cost
of *every earlier finding* in this chapter simultaneously.** The 8x-per-level
element growth (base task) and the `O(n^2)`/`O(n^{4/3})` solve-and-memory scaling
(Explore (a)) both get *worse* again going from 3-D to 4-D — element growth
becomes `16x` per level, and the memory wall arrives even earlier in absolute
mesh-level terms. The professor is deliberately asking for a *pen-and-paper*
estimate here, not a run, because the actual lesson is that you should be able to
predict "this will not fit" *before* burning GPU time and risking exactly the kind
of OOM crash Explore (a) just produced. Being able to do this arithmetic quickly —
elements-per-axis to the power of the dimension, DOFs similarly, nnz from the
local stencil width — is a genuinely practical skill for anyone about to attempt a
high-dimensional (or even just "one dimension higher than they're used to")
simulation for the first time.

---

## Explore (c) — Profile the sphere solve at level 5

### 1. Problem statement

Rerun the immersed-sphere solve at level 5 and profile it by stage: classification,
constraint-building, assembly, or solve — which dominates? The prompt explicitly
hints at an expected answer, citing an earlier finding ("the m0.5 findings say
constraints") — but says to profile *before* answering, not to just cite the
finding.

### 2. Results

| stage | time | share |
|---|---:|---:|
| solve | 0.102s | 42.9% |
| classify | 0.076s | 31.9% |
| geometry_data | 0.021s | 8.9% |
| extract_surrogate | 0.012s | 5.1% |
| build_constraints | 0.012s | 5.0% |
| build_mesh | 0.009s | 3.8% |
| build_uniform | 0.003s | 1.2% |
| sbm_setup | 0.002s | 0.7% |
| device_mesh | 0.001s | 0.5% |

(`n_elems=2,968`, `n_dofs=3,791`, total wall `0.27s`.)

### 3. Interpretation

The prompt's cited expectation does **not** hold at this level: `build_constraints`
is only `5.0%` of the total time — nowhere near dominant. `solve` (`42.9%`) and
`classify` (`31.9%`, the host-side work deciding which octree cells are inside,
outside, or cut by the sphere) dominate instead. Tracing the referenced finding
directly explains the mismatch: it measured `build_constraints` taking `48s` for a
4-D p2 mesh, using the **old, unvectorized** implementation (one `LeafLookup.find`
call per node per probe, in a Python loop). The *current* `build_constraints` was
vectorized in a later pass (batching all probes into a single call) and was
separately confirmed elsewhere in this project (the P1 performance-corner
tutorial) to drop from `222s` to `0.71s` at 66k 2-D DOFs — a 300x speedup. At this
sphere mesh's small scale (`3,791` DOFs), that fix is more than enough to make
constraints a non-issue; the finding being cited describes code that, in this
form, no longer exists.

### 4. The real learning — why this is in the tutorial

**A performance finding has a shelf life, and citing one without re-measuring is a
trap the tutorial sets on purpose.** This explore task is explicitly constructed
to test whether you'll profile first or just repeat the hinted answer — and the
"correct" behavior (measure, then discover the hint is stale) is the actual
lesson, not the specific percentages. This is a direct, deliberate echo of A1's
founding MMS discipline ("you distrust any solver claim that isn't a measured
order") applied to *performance* claims instead of *accuracy* claims — and it's
made concrete with a real, dated example: a genuine historical bottleneck that was
fixed by a later optimization pass, whose old cost profile is still sitting in a
findings document that a careless reader (or a rushed AI assistant, or anyone
working from memory instead of a fresh measurement) could easily cite as if it
still applied.

---

## Key Takeaways — A5, in plain terms

1. **The code being dimension-generic and the cost being dimension-generic are
   two entirely separate claims.** Every A1 result carried over unchanged to 3-D
   with zero code changes — but every A1 *cost* number got dramatically worse,
   often by orders of magnitude, for the exact same problem sizes.

2. **In 3-D, a direct solver's practical failure mode is running out of memory,
   not just getting slow.** Level 6 (274,625 DOFs — not an unusually large mesh)
   was OOM-killed on a 15GiB machine before its factorization ever finished. This
   is the concrete reason iterative solvers exist as a separate track of this
   project.

3. **You can, and should, predict "this won't fit" with pencil-and-paper
   arithmetic before spending compute to discover it the hard way.** Explore (b)'s
   whole point is that the level-4 k=4 memory estimate is knowable in advance,
   without ever touching a GPU.

4. **Every additional dimension compounds every earlier cost finding
   simultaneously** — worse element growth, worse solve scaling, an earlier
   memory wall, all at once. There's no dimension where "just one more" is free.

5. **A cited performance finding is a snapshot, not a permanent fact — always
   re-measure before trusting it, especially your own project's documentation.**
   The specific claim tested here ("constraints dominate") was true once, for
   code that has since been rewritten; profiling instead of citing is what
   catches that a claim has gone stale.
