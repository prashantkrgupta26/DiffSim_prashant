# Computational C5 — Two dimensions versus three

Going from 2-D to 3-D changes the growth *law*, the storage, **and the
physics**. This tutorial measures all three live on small meshes.

- **dofs per level** scale as $2^{\dim}$: ×4 in 2-D, ×8 in 3-D.
- **sparsity, correctly explained**: nnz/dof = coupled nodes × fields =
  $(2p+1)^{\dim}\times n_{\text{fields}}$ = ≈ 18 (2-D) / ≈ 54 (3-D) — from
  element **connectivity / order / fields / block / constraints / dim**,
  NOT a "$3^d-1$ finite-difference stencil" (which undercounts).
- **complete memory accounting** + an `estimate_capacity.py` CLI: every
  buffer (row ptrs, index mirror, vectors, history, l2g, quadrature,
  solver fill/preconditioner, output), not just CSR values. 128³ ≈ 8 GB;
  256³ ≈ 63 GB (exceeds a 48 GB card).
- **int32 CSR is an implementation choice**: nnz > 2³¹ overflows *this*
  build; an int64 build or a distributed/block-masked pattern lifts it.
- **physics at equal resolution**: 2-D is *not* a cheap 3-D — interfaces
  are curves vs surfaces, blends can be bicontinuous in 3-D only, so
  interfacial-area density, S(q), and percolation differ.

**Read** the course document, Computational Chapter *"Two dimensions
versus three"* (start with `scaling.py`).

**Run:**
```bash
python run.py                 # scaling + sparsity + memory + physics + ladder
python estimate_capacity.py --dim 3 --n 256   # full memory of a big run
```
Compare with [`EXPECTED.md`](EXPECTED.md).

**Regenerate the document's figures + numbers** (optional):
```bash
python gen_figures.py         # writes ../../latex/figures/c5_*.png + numbers/c5.tex
```

| file | role |
|------|------|
| `scaling.py` | the core: `measure_scaling`, `sparsity_breakdown`, `physics_comparison`, `memory_accounting` — read this first |
| `estimate_capacity.py` | CLI: complete memory footprint at any (dim, n, solver, index width) |
| `run.py` | the driver you run; prints scaling + sparsity + memory + physics |
| `gen_figures.py` | regenerates the figures and `numbers/c5.tex` |
| `EXPECTED.md` | reference numbers your run should reproduce (counts exact) |

The **2-D and tiny-3-D rows are live** (real captured CH Jacobians). The
**device-scale ladder is cited** from
`docs/dev/2026-07-13-m5-device-assembly.md` (slab64 nnz) and
`docs/dev/2026-07-13-blockch-mpf.md` (B4 film128, B5 mk32 int32 study) —
those take minutes per step, so we do not re-run them.

## Learning objectives

By the end you can:

- Explain why dofs scale as $2^{\dim}$ per refinement level — $\times4$
  in 2-D versus $\times8$ in 3-D (measured: 2-D $\approx\times3.9$, 3-D
  $\approx\times7.0$) — and why four levels is $256\times$ the dofs in
  2-D but $4096\times$ in 3-D.
- Derive the correct sparsity count nnz/dof = coupled nodes $\times$
  fields = $(2p+1)^{\dim}\times n_{\text{fields}}$ from element
  connectivity, basis order, number of fields, the field-block
  structure, and dimension — 9 coupled nodes $\times$ 2 fields = 18
  nnz/dof in 2-D, 27 $\times$ 2 = 54 in 3-D — and explain why the naive
  "$3^{\dim}-1$ finite-difference stencil" count (8 in 2-D, 26 in 3-D)
  is *wrong* for this FE discretization: it omits the node itself and
  the field block, so it undercounts.
- Produce a complete device-memory accounting for a Cahn–Hilliard
  solve — not just the CSR values array, but column indices, row
  pointers, the device index mirror, the Newton solution/RHS + BDF
  history vectors, the local-to-global connectivity map, quadrature/
  geometry tables, the solver's factorization fill-in or
  preconditioner-block workspace, and the saved output snapshot — using
  `estimate_capacity.py`.
- Explain why int32 CSR indices are an *implementation choice* with a
  genuine overflow risk once nnz exceeds $2^{31}\approx2.1\times10^9$,
  not a hardware limit, and connect this to the `--idx-bytes` flag
  (`--idx-bytes 8` = int64, at $2\times$ the index memory, lifts the
  wall).
- Demonstrate that 2-D is not a cheap stand-in for 3-D at equal
  resolution: interfacial-area density and the peak $S(q)$ wavelength
  differ between the two, and only a 3-D blend can be *bicontinuous*
  (both phases percolating simultaneously) — topologically impossible
  for two phases in 2-D.
- Read the cited device-scale ladder (`3d_slab64`, `3d_film128`, the
  `mk32` $(M{=}3,K{=}2)$ cases) and identify which cases overflow int32
  and which need a matrix-free or multi-GPU path.

## Prerequisites

- **Concepts:** finite-element connectivity and sparsity patterns, CSR
  matrix storage, basic GPU device-memory accounting.
- **Chapters:** Chapter 00 (`00_setup_and_smoke_test`, environment
  green); Chapter C1 (this track); Chapter C4 (device assembly /
  `blockch` solver — this chapter is the capstone that explains *why*
  C4's machinery exists).

## Expected cost

- **Device:** any CUDA GPU. The taught runs are 2-D and tiny-3-D live
  captures using well under 1 GB of device memory — an 8 GB card is
  ample for the LIVE runs in this chapter.
- **Solver:** `estimate_capacity.py` assumes `blockch` (preconditioner
  blocks + Krylov vectors) by default, or direct fill-in via
  `splu`/`cudss` if requested; the live `_capture_nnz` runs use `cudss`
  when available, else `splu`.
- The big 3-D memory numbers are **ESTIMATES, not runs**: at $128^3$ the
  complete accounting is $\sim8.0$ GB (fits a 48 GB card); at $256^3$ it
  is $\sim63.4$ GB, which **exceeds a 48 GB card**. These come from
  `estimate_capacity.py`'s capacity-planning model — this chapter does
  *not* actually launch a $256^3$ solve, so do not confuse "estimated to
  not fit" with "we ran it and it OOM'd."
- **Quick/run target:** the live 2-D/tiny-3-D runs in `run.py` are on
  the order of 30 s–2 min; anchor to physics P1's quick mode, measured
  at $\approx75$ s wall on an RTX 6000 Ada (mostly Warp kernel compile +
  Python start-up) — this chapter's live captures are the same order of
  magnitude, not separately re-measured here. Expect the same first-run
  Warp-compile overhead on a fresh session.

## Required deliverable

Submit the eight-item report of `../../ASSESSMENT.md`, specialised, with:

1. The `config.resolved.yaml` + `metadata.json` from a reference run of
   `run.py`.
2. The harness's five `[PASS]` checks green (`ALL CHECKS: PASS`), or a
   documented deviation.
3. The four figures (`c5_dofgrowth.png`, `c5_nnz.png`, `c5_memory.png`,
   `c5_physics.png`) regenerated from your saved run via
   `gen_figures.py`.
4. **Headline:** the dof/sparsity/memory accounting — report the
   *correct* sparsity count $(2p+1)^{\dim}\times n_{\text{fields}} = 18$
   (2-D) / $54$ (3-D) against the *wrong* "$3^{\dim}-1$" count ($8$ /
   $26$) and explain why the latter undercounts; report the complete
   memory total at $128^3$ ($\sim8.0$ GB) and $256^3$ ($\sim63.4$ GB,
   exceeds 48 GB) — **and** the physics-at-equal-resolution difference
   between 2-D and 3-D (interfacial-area density and $S(q)$ wavelength).
5. **Verification:** the int32-vs-int64 (`--idx-bytes 8`) memory delta
   at $256^3$ — how much the total grows, and whether int64 alone makes
   it fit a 48 GB card or the solver workspace is still the wall.
6. **Failure:** identify a cited case that overflows int32 (e.g. `mk32
   128x128x48 (M=3,K=2)`, $8{,}028{,}160$ dofs, $2.17\times10^9$ nnz) and
   explain why (nnz $>2^{31}$) and what fixes it (an int64 build or a
   distributed/block-masked pattern).
7. **Exploration:** answer one "Explore on your own" question with
   evidence (a sweep, a plot, or a fitted trend), not prose alone.
8. **Research bridge:** one paragraph connecting the dof/sparsity/memory
   growth law measured here to why C4's device assembler, `blockch`
   solver, and matrix-free frontier exist.
