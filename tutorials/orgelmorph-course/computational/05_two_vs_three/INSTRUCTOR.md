# Instructor companion — C5 Two dimensions versus three

*Teaching notes for the instructor. Student-facing material is in
`README.md` and the course document chapter `c5.tex`. Grading is in
`grading_rubric.md`; hints and selected solutions in `solutions/`.*

## What this chapter is really teaching

C5 is the capstone of the computational track: it explains *why* every
piece of machinery in Chapter C4 (device assembly, `blockch`, the
matrix-free frontier) exists. Four ideas do the teaching:

1. **The growth law, not a constant factor.** Dofs scale $2^{\dim}$ per
   refinement level — $\times4$ in 2-D, $\times8$ in 3-D. Four levels is
   $256\times$ in 2-D but $4096\times$ in 3-D. This is arithmetic
   students can derive, not a number to memorise.
2. **Sparsity comes from the FE model, not a stencil guess.** nnz/dof =
   coupled nodes $\times$ fields = $(2p+1)^{\dim}\times n_{\text{fields}}$
   — 18 in 2-D, 54 in 3-D for $p=1$. Students who reach for a
   finite-difference "$3^{\dim}-1$" count get 8/26 — wrong, because it
   omits the node itself and the $(c,\mu)$ field block.
3. **"Will it fit?" is a complete-accounting question.** The CSR values
   array is one of nine buffers `estimate_capacity.py` counts, and the
   *solver workspace* (factorization fill-in or preconditioner blocks)
   usually dominates, not the matrix values.
4. **2-D is not a cheap 3-D.** At *equal* resolution the physics differs
   — interfacial-area density, the peak $S(q)$ wavelength, and only 3-D
   admits a bicontinuous (both-phases-percolating) morphology. Cost is
   not the only reason to run in 3-D; the question being asked is
   different.

## Common student misconceptions & typical incorrect conclusions

- **"nnz/dof should be $3^{\dim}-1$ — a finite-difference stencil."**
  The single most common error in this chapter. The correct count is
  $(2p+1)^{\dim}\times n_{\text{fields}}$: $(2p+1)^{\dim}$ counts every
  node sharing a finite *element* with the node in question (including
  itself), and each such node pair carries a dense $n_{\text{fields}}
  \times n_{\text{fields}}$ block because the mixed $(c,\mu)$ system
  couples both fields at every shared node. A "$3^d-1$" FD-style count
  omits the node itself *and* the field block, so it always undercounts
  (8 vs. the correct 18 in 2-D; 26 vs. 54 in 3-D). Push students to
  `sparsity_breakdown()` in `scaling.py`, which reports both numbers
  side by side precisely so the wrong count is visible as wrong.
- **"Memory is the CSR values array."** It is one line among nine in
  `estimate_capacity.py`'s breakdown: CSR values, column indices, row
  pointers, a device index *mirror* the resident assembler keeps,
  Newton solution/RHS + BDF history vectors, the local-to-global
  connectivity map, quadrature/geometry tables, the solver's workspace
  (factorization fill-in *or* preconditioner blocks — usually the
  biggest term), and the output snapshot. A student who reports only
  the values-array size has under-counted by a large factor, especially
  in 3-D where the solver workspace dominates.
- **"int32 indices are a hardware limit."** They are not — they are
  *this build's* choice of CSR index width. The overflow is a real
  failure mode once nnz exceeds $2^{31}\approx2.1\times10^9$, but an
  int64 build (`--idx-bytes 8`, at $2\times$ the index memory) or a
  distributed/block-masked pattern lifts the wall entirely. Watch for
  students describing the ceiling as something the GPU imposes.
  It doesn't; the code's `idx_bytes` argument does.
- **"2-D is a cheap stand-in for 3-D — same physics, less cost."** False
  at equal resolution: interfaces are curves in 2-D, surfaces in 3-D, so
  the interfacial-area density and $S(q)$ wavelength differ, and a 3-D
  blend can be *bicontinuous* (both phases percolating simultaneously),
  which is topologically impossible for two phases in 2-D. 2-D remains
  a legitimate, cheaper *model problem* for convergence/BC/solver work —
  but its morphology is not a thin slice of the 3-D answer. The
  `physics_comparison()` output makes both points at once: same solver,
  same `h`, different numbers.

## Expected runtime ranges (RTX 6000 Ada; scale by card)

| mode | what runs | wall | notes |
|---|---|---|---|
| live scaling + sparsity + physics (`run.py`) | 2-D levels 5–7, 3-D levels 3–5 | on the order of physics P1's quick mode | anchor: P1 quick $\approx75$ s wall (measured), mostly Warp compile + Python start-up |
| `estimate_capacity.py` (any `--dim`/`--n`) | pure arithmetic, no solve | effectively instant | it is a closed-form estimator, not a run |

The big 3-D cases ($128^3$, $256^3$, and the entire cited device-scale
ladder) are **estimated or cited, never re-run** in this chapter — do
not fabricate a wall-clock time for them; only the memory/dof numbers
from `estimate_capacity.py` or the dev notes are reportable.

First run of a session pays a one-time Warp kernel-compile cost;
subsequent runs in the same environment are faster.

## Common CUDA / solver errors students hit

- **int32 nnz overflow past $2^{31}$.** This is a *real, teachable*
  failure mode at scale: the cited `mk32 128x128x48 (M=3,K=2)` case
  ($8{,}028{,}160$ dofs, $2.17\times10^9$ nnz) and the Nova-class
  `256x256x128` case both overflow the 32-bit CSR ceiling. If a student
  runs `estimate_capacity.py` at a large `--n` with the default
  `--idx-bytes 4` and sees `int32_overflow: True` (or `OVERFLOW` in the
  headroom line), that is the expected, correct behavior — the fix is
  `--idx-bytes 8` or a block-masked/distributed pattern, not a bug.
- **Out-of-memory if a student actually tries to *run* a big 3-D case**
  instead of just estimating it. `estimate_capacity.py` flags $256^3$ as
  $\sim63.4$ GB, over a 48 GB card — a student who ignores the estimate
  and launches the equivalent live solve will hit a real device OOM.
  That is the point: the estimator exists so you don't have to find
  this out the hard way.
- **Confusing "estimated" with "measured."** A student who writes
  "we ran the $256^3$ case and it took X seconds" has fabricated a
  number — this chapter never launches that solve. Only dofs/nnz/memory
  are reportable for the big cases; step times are reported only for
  the live 2-D/tiny-3-D rows.

## Discussion prompts

- Why does the "$3^d-1$" stencil intuition come from finite differences
  specifically, and why does it fail for a finite-element mixed
  $(c,\mu)$ system? What would make it correct (e.g. a scalar FD
  Laplacian)?
- If the solver workspace dominates total memory (as it does at
  $128^3$/$256^3$ blockch), what does that imply about where to spend
  engineering effort: the matrix format, or the solver?
- The int32 ceiling is "just" an implementation choice, yet it forced a
  block-masked pattern for the $(M{=}3,K{=}2)$ films. When is fixing an
  implementation choice (switch to int64) enough, and when does it force
  a deeper redesign (block-masking, matrix-free)?
- A student claims 2-D results "generalize" to 3-D because the solver
  and discretization are the same code path. What, specifically, in
  `physics_comparison()`'s output contradicts that claim?
