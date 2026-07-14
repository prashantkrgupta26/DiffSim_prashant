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
