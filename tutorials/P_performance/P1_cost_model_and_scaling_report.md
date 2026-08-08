# P1 — The Cost Model: Detailed Report

**Script:** `tutorials/P_performance/P1_cost_model_and_scaling.py`

This report has three parts: (1) a from-scratch, plain-language story explaining
every concept this tutorial assumes you already know, so the rest of the report
is actually readable; (2) the base task and all four Explore tasks, each with a
problem statement, results, interpretation, and the underlying lesson; (3) a
closing takeaways section.

---

## Part 1 — The complete theory, as a story

### Where we are in the story so far

Every tutorial before this one (A1 through A6, B1, B2) was about one question:
**is the answer correct?** You picked a manufactured solution, measured an error,
and checked that the error shrank at the right rate as the mesh got finer. That's
it — that's the entire concern of "correctness."

P1 asks a completely different question, about the *exact same code*: **how
expensive is it, and why?** Not "is the number right," but "how long did it take
to get the number, how does that time grow as the problem gets bigger, and can you
*predict* that growth before you measure it?" This is the discipline of
**computational cost modeling**, and it's just as rigorous a discipline as MMS
convergence testing — you're going to state a hypothesis (a predicted growth
rate), measure it, and hold yourself accountable to the disagreement, exactly like
every earlier chapter did for accuracy.

### What actually happens when you "solve a PDE," broken into stages

Picture the whole pipeline, from "I want to solve `-lap(u)=f`" to "here is `u`,"
as an assembly line with five stations. This tutorial's whole job is to time each
station separately, because a beginner's instinct — "solving the equation" is one
big lump of cost — hides five *very* differently-behaved costs inside it.

**Station 1 — Mesh.** Before you can do anything, you need a grid: a set of
points (nodes) and a set of small pieces (elements) connecting them, covering
your domain. In this codebase that's an *octree* — a uniform grid built by
repeatedly halving a square (or cube) — plus the bookkeeping (`build_mesh`) that
turns "here is a grid of boxes" into "here is a numbered list of nodes and which
nodes belong to which element."

**Station 2 — Constraints.** Not every mesh is perfectly uniform (you saw this
directly in A4, with mixed p1/p2 elements) — sometimes one node's value has to be
*expressed in terms of* other nodes' values rather than being a free unknown of
its own (a "hanging node"). `build_constraints` figures out which nodes are truly
free and which are tied down, and builds the matrix that expresses that
relationship.

**Station 3 — Assembly.** This is where the actual physics gets turned into
numbers. For every element, you compute a small local matrix (the "element
stiffness matrix," `Ke`) that says how that one element's corners/nodes interact
according to the PDE. Then you scatter all those small local matrices into one
giant global matrix `A` — most of whose entries are exactly zero, because most
pairs of nodes in a big mesh have nothing to do with each other (they're not in a
shared element). Assembly has two very different halves: the GPU computes all the
small element matrices in parallel (fast, one thread per element), and then the
*host* (regular CPU/Python) has to stitch (`scatter`) all those small pieces into
one big sparse matrix — which, as later chapters find out, is not automatically
fast just because the GPU part was.

**Station 4 — Factorize.** Now you have a giant matrix equation `Au=b` and you
need to solve for `u`. The classic, always-correct-if-you-can-afford-it way is
**direct solve**: factor `A` into a product of a lower-triangular matrix `L` and
an upper-triangular matrix `U` (`A=LU`), the matrix version of "long division."
Once you have `L` and `U`, solving `Au=b` becomes two cheap triangular solves
instead of one expensive general one. Computing that `L` and `U` factorization is
called **factorize**, and it is, across almost every result in this entire
project, the single most expensive stage once the mesh gets large — because of
something called **fill-in** (next section).

**Station 5 — Backsolve.** Once you have `L` and `U`, actually solving for `u`
(two cheap triangular solves) is the fast, "free" last step.

### Why is factorize the expensive one? ("Fill-in")

Here's the key idea that explains almost every number in this whole tutorial.
Your original matrix `A` is **sparse** — most entries are zero, because node `i`
and node `j` only interact if they share an element, and most pairs of nodes in a
big mesh don't. But when you factor a sparse matrix into `L` and `U`, some of
those zero entries turn into *nonzero* entries — this is fill-in, and how much of
it happens depends on the matrix's structure. For a 2-D mesh, fill-in makes the
factorization cost grow like `n^1.5` (n = number of unknowns/DOFs) — noticeably
faster than the `O(n)` cost of just building the matrix in the first place. For a
3-D mesh, it's much worse — closer to `n^2` in time and `n^{4/3}` in memory. This
single fact (2-D fill-in is bad, 3-D fill-in is *much* worse) is the entire
reason this project has a separate "iterative solvers" track (P2) — at some
problem size, direct factorization simply stops being an option, not because it's
slow, but because it runs out of memory (you saw this exact failure mode directly
in A5's level-6 3-D solve, which was killed by the operating system for using too
much RAM).

### What does "O(n)" even mean, in plain words?

You'll see notation like `O(n)`, `O(n^1.5)`, `O(n log n)` everywhere in this
chapter. All it means is: **if you double the problem size `n`, how many times
bigger does the cost get?** `O(n)` (linear): double the problem, double the cost.
`O(n^1.5)`: double the problem, cost goes up by `2^1.5 ≈ 2.83x`. `O(n^2)`: double
the problem, cost *quadruples*. The whole point of measuring an "exponent" in this
tutorial is figuring out, from real timing data, which of these growth curves a
given stage actually follows — because that exponent tells you how bad things get
as you push to bigger, more realistic problems, long before you can afford to
just try it and find out.

### Why a GPU makes timing *harder*, not just faster

A CPU runs one (or a few dozen) instructions at a time, in order, so timing tends
to behave the way your intuition expects: do more work, take proportionally more
time. A GPU is different: it runs *thousands* of small, identical pieces of work
(here: one thread per mesh element) simultaneously. This introduces three
GPU-specific effects that this tutorial's docstring calls out explicitly, and that
distort naive timing if you don't know to watch for them:

- **The launch floor.** Starting a GPU kernel at all — even one that does almost
  no work — costs a small, fixed amount of time (getting the GPU's attention,
  setting up the launch). If your actual per-element work is tiny (a small mesh),
  this fixed "hello, GPU" cost can be *bigger* than the real work, and your
  timing measures the overhead, not the algorithm. This is why small-level
  measurements in this whole project are consistently noisy and untrustworthy —
  you're measuring launch overhead, not the thing you meant to measure.

- **JIT compilation.** This codebase doesn't ship pre-compiled GPU machine code —
  it writes GPU kernels in Python-like source and compiles them the *first time*
  they're actually needed ("Just-In-Time" compilation), then caches the compiled
  result on disk so every later run is instant. The very first time any specific
  kernel "shape" (a specific element order, dimension, etc.) is used, you pay a
  real, seconds-scale compilation tax — and if you don't know this is happening,
  you might think your algorithm itself is that slow.

- **Host-device sync stalls.** The GPU and the regular CPU (the "host") work
  somewhat independently — the CPU can fire off GPU work and keep going without
  waiting. But the moment you ask to actually *look at* a GPU result (pulling
  data back with something like `.numpy()`), everything has to stop and wait for
  the GPU to finish and hand the data over. This is called a "sync," and doing it
  more often than necessary is a classic, easy-to-introduce performance bug.

### FLOPs, GFLOP/s, and "how close to the hardware's limit am I?"

A **FLOP** is one floating-point operation — one multiply, or one add. If you
count up exactly how many multiplies and adds a piece of code does, and divide by
how long it took, you get a rate: **FLOP/s** (or, since modern hardware does
billions of these, **GFLOP/s** — billions of FLOPs per second). Every chip has a
theoretical **peak** GFLOP/s it could achieve if every single cycle did useful
math with zero waste — in practice you never hit that exactly, but the
*percentage of peak* you achieve tells you how well-suited your specific
computation is to the hardware.

There are two fundamentally different reasons a real computation falls short of
peak, and telling them apart is the actual skill this chapter (and its
predecessor, B2) is teaching:

- **Compute-bound:** the hardware is doing math as fast as it can; the bottleneck
  is literally how many multiply/add units exist and how fast they run.
- **Bandwidth-bound (memory-bound):** the hardware is mostly *waiting* — not for
  math, but for data to arrive from memory. Modern chips can do vastly more
  arithmetic per second than they can move bytes per second, so any computation
  that does only a *little* math per byte it touches (low "arithmetic intensity")
  will sit idle waiting on memory, no matter how fast its math units are.

This distinction — compute-bound vs. bandwidth-bound — is called the **roofline
model**, and it's the single most useful mental tool in this whole chapter for
answering "why isn't my code faster," because the *fix* is completely different
depending on which one you have (compute-bound: reduce the math; bandwidth-bound:
reduce the data movement, or reuse data you've already loaded).

### Putting it all together: what this whole tutorial is actually asking you to do

Every explore task in P1 is a variation on the same move: **state a prediction
from the theory above, then measure it, then explain the gap.** Sometimes the
prediction survives (Explore (c)'s JIT tax matches the docstring's warning almost
exactly). Sometimes it doesn't (Explore (a)'s DOF-ratio prediction is measurably
wrong). Either outcome is a valid result — the discipline being taught is the
*habit* of checking, not the specific numbers you happen to get on this machine,
today. That's the story. Everything below is that story applied to five specific
questions.

---

## Base Task — Measuring the analytical cost table

### 1. Problem statement

The docstring lays out an *analytical* prediction table (derived from theory, the
same way you'd predict on paper before running anything): mesh/constraints
`O(n)`, element kernels `O(n)` work, CSR assembly `O(n)`, `splu` factorization
`O(n^1.5)` in 2-D, triangular solves `O(n log n)`. The task: build the actual
per-stage timing table (mesh, constraints, assemble, factorize, backsolve) at
levels 5 through 8, fit a measured exponent for each stage from consecutive time
ratios, and compare against this table. The docstring frames this as a two-act
historical story: **Act 1** (before a since-landed optimization), constraints was
*so* slow (a raw Python loop, `222s` at level 8) that it beat even the `O(n^1.5)`
factorize stage — the wrong bottleneck, for a boring, fixable reason (bad
constant, not bad complexity). **Act 2** (after that fix), the ranking returns to
what the analytical table predicts.

### 2. Results

```
level    n     nnz     mesh   constraints  assemble  factorize  backsolve
5      1089    9409   0.0007    0.0001      0.0016    0.0020     0.0001
6      4225   37249   0.0007    0.0001      0.0023    0.0099     0.0004
7     16641  148225   0.0034    0.0001      0.0073    0.0512     0.0022
8     66049  591361   0.0094    0.0003      0.0308    0.3124     0.0107
```

Matches the docstring's own "Act 2" story exactly: `factorize (0.31s) > assemble
(0.03s) > mesh (0.01s) >> constraints (0.0003s)`. No bug.

### 3. Interpretation

The bottleneck ranking is exactly what the textbook analytical table predicts —
constraints, once the fastest-growing stage in this project's own history (per
Act 1), is now the *cheapest* stage by a wide margin, and factorize is
unambiguously dominant by level 8. This isn't a coincidence to shrug off — it's
the direct, measured payoff of a specific optimization (vectorizing the
constraint-building loop) that happened earlier in this project's history. The
memory model check (`sparse CSR: ~9MB` vs. `DENSE would be 8n^2 = 34.9GB` at
level 8) makes the sparse-vs-dense stakes concrete: representing this exact same
matrix densely would need nearly 35 gigabytes, for a problem whose *sparse*
representation fits in single-digit megabytes.

### 4. The real learning — why this is in the tutorial

**A bottleneck ranking is an empirical, versioned fact about the code, not a
permanent truth about the algorithm.** This is stated explicitly in the
docstring, and it's the single most important idea in this entire chapter: the
*same* analytical prediction (factorize should dominate) was **true**, then
became **false** (constraints' bad Python-loop constant took over), then became
**true again** (after the fix). If you had only ever measured during Act 1, you
would have walked away with "constraints is the bottleneck in this codebase" as a
durable fact — and you'd have been right, until someone fixed it, at which point
you'd be wrong and not know it. (This exact trap is what A5's Explore (c) walks
directly into on purpose — citing this project's own Act-1-era finding as if it
still applied.) The lesson isn't "measure once," it's "re-measure after every
change that could plausibly matter, because your mental model of the bottleneck
has a shelf life."

---

## Explore (a) — Add p=2 columns to the table

### 1. Problem statement

Extend the cost table to include p=2 (quadratic) elements alongside p=1. The
question: which stages' costs grow with the *DOF ratio* (p2 has roughly `2.25x`
as many degrees of freedom as p1 in 2-D, from `((2n+1)/(n+1))^2 -> 4` asymptotically,
though the docstring's own quoted figure is `~2.25x`), and which grow with the
*kernel-work ratio* instead (a p2 element's local stiffness matrix has `9x9=81`
entries vs. p1's `4x4=16`, a `~5.06x` ratio, since more work happens per element,
not just more DOFs overall)?

### 2. Results

| level | n(p2/p1) | mesh | constraints | assemble | factorize | backsolve |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 3.880 | 1.77x | 1.98x | 2.13x | 4.84x | 7.07x |
| 6 | 3.939 | 2.10x | 1.49x | 4.64x | 6.87x | 10.68x |
| 7 | 3.969 | 1.67x | 1.44x | 5.72x | 8.83x | 8.74x |
| 8 | 3.984 | 2.83x | 1.93x | 7.52x | 16.92x | 8.03x |

### 3. Interpretation

Neither predicted constant survives contact with measurement, for any single
stage. The measured DOF ratio itself is `~3.98x`, not the docstring's own quoted
`~2.25x` (this independently reproduces A4's finding of the same fact, to within
measurement noise — a nice cross-chapter consistency check). `assemble`'s ratio
*climbs* with level (`2.13x -> 7.52x`) instead of sitting near the predicted
`~5x` kernel-work constant — makes sense once you remember `assemble` is really
two different costs bundled together (a GPU element-kernel cost, which *should*
track the `~5.06x` per-element work ratio, plus a host-side CSR-triplet-building
cost, which tracks `nnz`, and `nnz`'s own p2/p1 ratio grows with level just like
the DOF ratio does). `factorize`'s ratio is the largest and most level-dependent
of all (up to `16.92x`) — unsurprising given factorization is *already*
superlinear in DOF count, so feeding it a DOF ratio that's actually `~4x` (not
the smaller `2.25x`) compounds through that superlinearity. `mesh` and
`constraints` stay closest to flat, in the `1.5x-2.8x` range — noisy at these
small absolute times, but roughly tracking simple per-node/per-element costs.

### 4. The real learning — why this is in the tutorial

**"Which stages follow which ratio" turns out to be the wrong question — almost
no stage follows either ratio cleanly, and the reason why is itself the
lesson.** Real pipeline stages are rarely a single, pure cost; `assemble` here is
visibly *two* different costs (device-kernel work and host-triplet work) riding
on top of each other, each with its own scaling behavior, and the blended result
tracks neither cleanly. The professor is teaching you to distrust tidy, single-
number performance summaries and instead go find the actual sub-costs hiding
inside a stage that "should" behave one simple way — a habit that pays off
directly a few chapters later, when B2's Explore (c) needs exactly this kind of
decomposition to correctly diagnose a different bottleneck.

---

## Explore (b) — The 3-D crossover, and why it comes "much sooner"

### 1. Problem statement

Repeat the staged cost table in 3-D, levels 3 through 5. Find the level where
`factorize` overtakes `assemble` — the prediction is that this happens *much
sooner* than in 2-D, because 3-D fill-in is worse (`O(n^2)` time vs. 2-D's
`O(n^1.5)`, `O(n^{4/3})` memory vs. presumably better in 2-D). This measured fact
is stated as the entire justification for the next chapter, P2 (iterative
solvers).

### 2. Results

Clean crossover determination (median of 5 repetitions per level, with a full
multi-level warmup pass first to eliminate any first-touch/JIT bias):

| dim | level | n | assemble | factorize | ratio |
|---|---:|---:|---:|---:|---:|
| 2-D | 4 | 289 | 0.00091 | 0.00039 | 0.43 |
| 2-D | 5 | 1,089 | 0.00110 | 0.00173 | **1.57** |
| 3-D | 2 | 125 | 0.00147 | 0.00025 | 0.17 |
| 3-D | 3 | 729 | 0.00192 | 0.00305 | **1.59** |

Measured exponents: 3-D `factorize` climbs from `1.82` to `2.46` across levels
3-5 (approaching, then exceeding, the predicted `O(n^2)`); 3-D `assemble` stays
close to `0.63-0.99` (`~O(n)`, as predicted).

### 3. Interpretation

2-D crosses between level 4 and 5 (`n` going from 289 to 1,089); 3-D crosses
between level 2 and 3 (`n` going from 125 to 729) — a full two refinement levels
earlier. What's interesting is that the crossover *DOF count itself* is
similar in both dimensions (order of a few hundred to about a thousand) — the
real difference isn't "3-D crosses at a smaller n," it's that 3-D's `8x`-per-level
element growth (vs. 2-D's `4x`) reaches that same crossover magnitude in far
fewer refinement steps. In other words: the same "danger zone" in absolute
problem size exists in both dimensions, but 3-D drives you into it much faster as
you refine, simply because each 3-D refinement step adds so much more work than
each 2-D step does.

### 4. The real learning — why this is in the tutorial

**"It gets worse in 3-D" is a vague, easy-to-nod-along-with claim; "it gets worse
two refinement levels sooner, and here's the exponent that proves it" is a
falsifiable, load-bearing fact you can act on.** This explore task exists to turn
a plausible-sounding assertion in the docstring into something you've personally
verified with your own numbers, on your own machine — and the specific mechanism
uncovered here (similar crossover *n*, very different crossover *level*, because
of differing per-level growth rates) is a more precise and more useful
understanding than "3-D costs more" alone. This is also a direct, concrete
instance of A5's own finding (which measured 3-D's factorize exponent trending
toward `O(n^2)` and eventually hitting an out-of-memory wall) — P1's Explore (b)
is where that A5 story's *root cause* (fill-in's dimension-dependence) is first
isolated and measured in its cleanest, smallest-scale form, before A5 shows you
what happens when you push it to a scale that actually breaks.

---

## Explore (c) — The JIT tax for a first-ever kernel variant

### 1. Problem statement

Time the very first-ever launch of a genuinely new GPU kernel variant (one the
disk cache has never seen before) and compare it to a subsequent warm launch of
the same, now-compiled kernel. The docstring's own rule (ii) warns that this cold
compile costs "seconds-to-minutes" — quantify it precisely for the p1 2-D
stiffness kernel. (The prompt suggests deleting `~/.cache/warp` to force this,
with an explicit "careful" flag — a safer route was used instead, described
below.)

### 2. Results

Rather than deleting the shared on-disk kernel cache (which could affect other
work relying on that cache, and isn't reversible without recompiling everything),
a small, source-distinct-but-mathematically-identical copy of the p1 2-D
stiffness kernel was written — different enough in source text to guarantee the
disk cache has never seen it, without touching any files:

```
cold (first-ever launch, log confirms "(compiled)" not "(cached)"): 0.5207s
warm (2nd launch, same compiled kernel object):                     0.0001s
warm (3rd launch):                                                  0.0001s

JIT tax = 0.5206s  (~7,841x the warm launch cost)
```

### 3. Interpretation

The JIT tax is, for practical purposes, the *entire* cost of that first call —
half a second of compilation sitting in front of a kernel whose actual per-launch
work, once compiled, takes about 100 *microseconds*. That's a difference of
roughly four orders of magnitude between a cold and a warm call for this specific
kernel. This directly and precisely confirms the docstring's own warning, and
gives it a hard number for the p1 2-D case specifically (other variants — 3-D,
p2, larger element counts — would presumably pay a different, likely larger, tax,
though that wasn't separately measured here).

### 4. The real learning — why this is in the tutorial

**Every single timing table in this entire tutorial series is only meaningful
because of one unglamorous line of code: `stages(warm_level)` before the real
measurement loop.** This explore task exists to make you personally feel *why*
that line is there — not as a boilerplate ritual, but as the difference between a
measurement that reflects your algorithm and one that reflects a one-time,
unrepeatable compilation cost masquerading as your algorithm's speed. If you
skipped warming and timed a "cold" run, you'd conclude this kernel is
catastrophically, unfixably slow — a completely wrong conclusion, since the real,
steady-state cost is four orders of magnitude smaller. This is a sharp, specific
example of a much more general principle that shows up constantly in performance
engineering: **know what your first measurement actually includes**, because a
startup cost hiding inside a "hot loop" measurement will send you chasing a
problem that doesn't exist.

---

## Explore (d) — A FLOP ledger for one element matrix, vs. GPU peak

### 1. Problem statement

Reproduce, for this GPU 2-D stack, the same "flop-ledger" discipline the
reference document applies to a 1-D CPU solver: count the actual arithmetic
operations in the p1 2-D stiffness kernel (`make_poisson_element_matrices`) by
reading its source, measure its achieved FLOP/s, and compare against this
machine's GPU FP64 peak. What fraction of peak is reached, and is the kernel
compute-bound or bandwidth-bound?

### 2. Results

Counting operations directly in `poisson_Ke`'s source (p1 2-D: `nbf=4`, `nqp=4`,
`dim=2` — 3 multiplies + 1 add per `(element, a, b, quadrature-point)` term from
the two `fe_dN_s` table lookups and their product, plus 1 multiply for the
quadrature-weight/Jacobian term, plus the small per-element setup cost):

```
flops/element = dim + 1 + nqp*(nbf^2*(4*dim+2) + 1) = 647
```

Timed the *isolated* kernel launch (not the whole `assemble_csr` stage, which
also includes host-side CSR triplet work) at levels 6-8, fully warmed, minimum
of repeated launches:

| level | n_elems | total FLOPs | achieved (min) |
|---:|---:|---:|---:|
| 6 | 4,096 | 2,650,112 | 46.1 GFLOP/s |
| 7 | 16,384 | 10,600,448 | 85.5 GFLOP/s |
| 8 | 65,536 | 42,401,792 | **98.0 GFLOP/s** |

GPU FP64 peak estimate: this machine's GPU properties were *queried* directly
(`sm_count=22`, `arch=sm_89`/Ada Lovelace, via Warp's own device object, and a
`2130 MHz` application boost clock via `nvidia-smi -q`) rather than assumed,
combined with two well-documented Ada-architecture facts (128 FP32 cores per SM;
FP64 throughput is `1/64` of FP32 on non-datacenter Ada chips): `FP32 peak = 2 *
22 * 128 * 2.13e9 ~= 12.0 TFLOPS`, so `FP64 peak ~= 187 GFLOPS`.

**Achieved efficiency: `98.0 / 187 ~= 52%` of estimated FP64 peak**, and still
climbing at the largest level tested.

### 3. Interpretation

This kernel is compute-bound, not bandwidth-bound. The argument: counting only
the memory traffic that's genuinely *unique per element* (reading `h[e]`, writing
the `4x4` output `Ke[e,:,:]` — the small `dNtab`/`wtab` lookup tables are shared
and cache-resident across all 65,536 elements, not re-fetched from scratch each
time), arithmetic intensity works out to `647 flops / 136 bytes ~= 4.76
FLOPs/byte`. Even using a generous range for this card's real GDDR6 bandwidth
(order `~100-250 GB/s` — not separately measured here, so treated as a range
rather than a single trusted number), the roofline "ridge point" (`FP64 peak /
bandwidth`) works out to roughly `0.8-1.9 FLOPs/byte` — comfortably below this
kernel's own `4.76`, across the *entire* plausible bandwidth range. That means
the kernel's bottleneck is compute capacity, not data movement, and reaching
roughly half of estimated FP64 peak — for a small, per-thread, table-lookup-heavy
kernel that was never explicitly hand-tuned for peak throughput — is a genuinely
strong result. Compare this directly to B2's Explore (c), which measured an
analogous but very differently-structured computation (a batched contraction over
many tiny matrices, expressed as a generic CPU `numpy.einsum` call) achieving
under 1% of *its* practical peak — that kernel was memory-bound (arithmetic
intensity around `~1 FLOP/byte`, and worse, running on a generic, unblocked CPU
loop that couldn't lower onto an efficient BLAS call at all). Same underlying
idea (element-local, per-Gauss-point weighted computation), radically different
outcome, purely because of *how* each one is expressed and *where* it runs.

### 4. The real learning — why this is in the tutorial

**"Compute-bound vs. bandwidth-bound" isn't an abstract classification exercise —
it tells you exactly what kind of optimization would and wouldn't help**, and
this task is designed to be read side-by-side with B2's opposite result. If this
p1 2-D kernel were somehow slow, the fix would be to reduce arithmetic (fewer
FLOPs per element, e.g. a cheaper quadrature rule) — reducing data movement
wouldn't help, because data movement was never the bottleneck. B2's einsum
kernel, by contrast, needed the *opposite* fix: restructuring the batched
computation (e.g., onto a GPU, where thousands of small independent element
computations are precisely what the hardware is built to do well in parallel) to
improve data reuse and cache behavior, not to shave arithmetic operations that
were already cheap relative to the memory traffic. The professor is teaching you
that "make it faster" is not one instruction — it's a diagnosis-then-treatment
process, and the roofline argument (arithmetic intensity vs. the hardware's own
ridge point) is the specific, quantitative tool that tells you which treatment
applies *before* you spend time trying the wrong one.

---

## Key Takeaways — P1, in plain terms

1. **A bottleneck ranking is a measured, versioned fact about a specific version
   of the code — not a permanent truth about the algorithm.** This project's own
   history (constraints dominating, then being fixed, then factorize retaking the
   crown) is direct, first-party proof; A5's Explore (c) shows what happens when
   someone trusts a stale version of that ranking instead of re-measuring.

2. **A single "stage" of a pipeline can secretly be two or more different costs
   glued together, each scaling differently.** `assemble`'s p2/p1 ratio didn't
   match either simple prediction because it's really a GPU-kernel cost and a
   host-side CSR-building cost riding together — decomposing further, not
   accepting the blended number, is what actually explains the data.

3. **Dimension doesn't just make things "worse" — it changes *how fast* you reach
   the same danger zone.** The 3-D and 2-D factorize/assemble crossovers happen at
   similar absolute DOF counts, but 3-D's faster per-level element growth gets
   you there two refinement levels sooner — a precise, actionable version of "3-D
   is more expensive."

4. **Warming the JIT cache isn't boilerplate — it's the difference between
   measuring your algorithm and measuring a one-time compilation tax.** A cold
   kernel launch here was ~7,800x slower than a warm one; every trustworthy
   timing table in this whole project depends on getting this right first.

5. **"Is it slow?" is the wrong question — "is it compute-bound or
   bandwidth-bound?" is the right one, because the answer determines what fix
   would actually help.** This p1 2-D kernel reached ~52% of FP64 peak and is
   compute-bound; B2's einsum kernel reached <1% of peak and was memory-bound —
   same general kind of per-element computation, opposite diagnosis, opposite
   fix.

6. **The throughline connecting this entire tutorial series continues here,
   applied to a new axis:** every earlier chapter taught you to distrust an
   accuracy claim that isn't backed by a measured order; P1 teaches you to
   distrust a performance claim — your own predicted exponent, a stated
   DOF-scaling ratio, even this project's own prior findings — that isn't backed
   by a fresh measurement, taken on the code as it exists today.
