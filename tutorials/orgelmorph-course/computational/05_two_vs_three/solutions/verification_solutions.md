# Selected full solutions — C5 verification exercises

*Full worked solutions for the two **verification** exercises only (the
int32-vs-int64 memory delta, and the sparsity count vs. the wrong
finite-difference-style count). Hints for all "Explore on your own"
questions are in `hints.md`. Instructor-only — do not distribute before
the deadline.*

---

## V1 — int64 vs int32 CSR indices: the memory delta at $256^3$ (Q1)

**Claim.** Switching `estimate_capacity.py`'s `--idx-bytes` from 4
(int32) to 8 (int64) grows every index-bearing buffer — but not the
float64 buffers — and, because the `blockch` solver workspace itself
embeds index-sized block copies, the *dominant* term grows too. Whether
int64 alone clears the 48 GB card is a separate question from whether it
lifts the int32 overflow ceiling.

**Procedure.**
```bash
python estimate_capacity.py --dim 3 --n 256 --solver blockch --idx-bytes 4
python estimate_capacity.py --dim 3 --n 256 --solver blockch --idx-bytes 8
```
Compare the two `TOTAL` lines and the breakdown by component.

**Expected result.** At $256^3$ (blockch), dofs $=33{,}949{,}186$, nnz
$=1{,}833{,}256{,}044$ (54 nnz/dof, matching the FE model):

| idx\_bytes | total | biggest term | fits 48 GB? |
|---|---|---|---|
| 4 (int32) | **63.41 GB** (matches `EXPECTED.md`'s $\sim63.4$ GB) | solver workspace, 33.20 GB (52.4%) | NO |
| 8 (int64) | **86.36 GB** | solver workspace, 41.74 GB (48.3%) | NO |

The delta is $+22.95$ GB ($+36\%$). Every index-bearing buffer grows:
CSR column indices $6.83\to13.66$ GB, the device index mirror
$6.96\to13.91$ GB, the local-to-global map $0.5\to1.0$ GB, row pointers
$0.13\to0.26$ GB — but the CSR *values* (float64, unchanged at 13.66 GB),
the solution/RHS/BDF-history vectors, and the output snapshot do not
move, because they are not index arrays. The *solver workspace* term
also grows ($33.20\to41.74$ GB) because `blockch`'s preconditioner-block
estimate itself has an `(8 + idx_bytes)` factor — the workspace is not
index-free. **int64 does not make $256^3$ fit a 48 GB card** — the total
goes from 63.4 GB to 86.4 GB, both far over 48 GB; the solver workspace
remains the dominant term (and grows) either way. int64 only matters for
the *int32 overflow* question (nnz vs. $2^{31}$), which at this
particular $(n{=}256, \text{blockch})$ case is not even triggered:
nnz $=1.83\times10^9<2^{31}\approx2.15\times10^9$ (headroom 0.854, "ok")
— the case that *does* overflow int32 is the cited `mk32
256x256x128 (Nova)` ladder point ($2.28\times10^{10}$ nnz), which has
more fields/species than this 2-field CH system.

**Common wrong conclusion.** "int64 fixes the $256^3$ memory problem
since it removes the int32 overflow risk." No — int64 *widens* the
overflow ceiling (useful when nnz $>2^{31}$), but it makes the *total*
memory footprint larger, not smaller, and the $256^3$ case was already
memory-infeasible (over 48 GB) at int32 for reasons unrelated to
overflow — the solver workspace alone (33–42 GB) is the wall. Fixing
"will it fit" at $256^3$ requires a matrix-free or multi-GPU approach,
not a wider index type.

---

## V2 — Sparsity is $(2p+1)^{\dim}\times n_{\text{fields}}$, not
$3^{\dim}-1$ (from `sparsity_breakdown()`)

**Claim.** For the degree-$p=1$ mixed $(c,\mu)$ Cahn–Hilliard system,
nnz/dof = (coupled nodes) $\times$ (fields) = $(2p+1)^{\dim}\times
n_{\text{fields}}$: 9 coupled nodes $\times$ 2 fields $=18$ in 2-D, 27
$\times$ 2 $=54$ in 3-D. A naive "$3^{\dim}-1$ finite-difference
stencil" count — 8 in 2-D, 26 in 3-D — is *wrong* for this
discretization.

**Procedure.**
1. Call `sparsity_breakdown(2)` and `sparsity_breakdown(3)` (in
   `scaling.py`) — it returns `coupled_nodes`, `nnz_per_dof`, and
   `fd_stencil_would_say` side by side.
2. Compare both counts against the *measured* nnz/dof from
   `measure_scaling()`'s live captured Jacobians.

**Expected result.** From `EXPECTED.md`:

| dim | coupled nodes $(2p+1)^{\dim}$ | $\times$ fields | = nnz/dof (correct) | "$3^d-1$" (wrong) |
|---|---|---|---|---|
| 2-D | 9 | 2 | **18** | 8 |
| 3-D | 27 | 2 | **54** | 26 |

Measured live averages sit just below these interior maxima —
$\approx18$ in 2-D, $\approx50$ in 3-D — because boundary nodes have
fewer neighbours than an interior node's full $(2p+1)^{\dim}$ block.

**Why the FD-style count is wrong for this FE discretization.** A
"$3^{\dim}-1$" count is the right answer to a *different* question: how
many *distinct neighboring grid points* does a scalar 5-point/7-point
finite-difference Laplacian stencil touch, excluding the point itself
($3^2-1=8$ in 2-D, $3^3-1=26$ in 3-D). It fails here for two independent
reasons, both visible in `stencil_nodes()`/`sparsity_breakdown()`:
1. **It omits the node itself.** The finite-element mass/stiffness
   contribution couples every node to itself too (the diagonal is part
   of the $(2p+1)^{\dim}$ coupled-node block, not a separate
   afterthought) — so the FD count is off by one node before fields
   are even considered.
2. **It omits the field block.** This is a *mixed* $(c,\mu)$ system: two
   fields are solved simultaneously, and within each pair of coupled
   nodes the $2\times2$ field block is fully dense (both $c$ and $\mu$
   equations reference both fields at a shared node). The FD count is a
   *scalar* stencil; multiplying by $n_{\text{fields}}=2$ is required to
   get nnz/dof for this vector-valued problem, which the naive count
   never does.
Both omissions push the FD-style number down, so it always
*undercounts*: $8$ vs. the correct $18$ in 2-D (a $2.25\times$
undercount), $26$ vs. $54$ in 3-D (a $2.08\times$ undercount).

**Common wrong conclusion.** "The finite-difference stencil formula
$3^{\dim}-1$ is a reasonable estimate of matrix density for any mesh
discretization." It is specific to a *scalar*, node-centered FD
Laplacian and even there it excludes the diagonal by convention — it
does not generalize to a finite-element method (which couples every
node sharing an *element*, self included) or to a multi-field system
(which needs the extra $n_{\text{fields}}$ factor). Using it here
understates memory and solver cost by roughly a factor of two, in both
dimensions.
