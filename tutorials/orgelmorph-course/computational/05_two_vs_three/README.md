# Computational C5 — Two dimensions versus three

Going from 2-D to 3-D is not a constant factor — it changes the growth
*law*. This tutorial measures the change live on small meshes and
connects it to the device-scale ladders from the dev notes.

- **dofs per level** scale as $2^{\dim}$: ×4 in 2-D, ×8 in 3-D.
- **nnz per dof** reflects the stencil ($3^{\dim}-1$ neighbors): ≈ 18 in
  2-D (9-point), ≈ 50 in 3-D (27-point). The matrix is bigger *and*
  denser.
- **int32 ceiling**: nnz $< 2^{31}$. At $(M{=}3,K{=}2)$ production
  physics the 128×128×48 superset already overflows → the block-masked
  pattern.
- **direct-solver fill** is far worse in 3-D → the cuDSS memory wall of
  C4, which is why 3-D forces the device assembly path + `blockch`.

**Read** the course document, Computational Chapter *"Two dimensions
versus three"* (start with `scaling.py`).

**Run:**
```bash
python run.py                 # live 2-D vs tiny-3-D scaling + cited ladder
```
The 3-D level-5 step takes ~20 s (a real 72k-dof CH step on the box); the
whole run is ~30–40 s. Compare with [`EXPECTED.md`](EXPECTED.md).

**Regenerate the document's figures + numbers** (optional):
```bash
python gen_figures.py         # writes ../../latex/figures/c5_*.png + numbers/c5.tex
```

| file | role |
|------|------|
| `scaling.py` | the core: `measure_scaling`, `scaling_row`, `CITED_LADDER`, `int32_headroom` — read this first |
| `run.py` | the driver you run; prints the scaling tables + cited ladder |
| `gen_figures.py` | regenerates the dof-growth + nnz figures and `numbers/c5.tex` |
| `EXPECTED.md` | reference numbers your run should reproduce (counts exact) |

The **2-D and tiny-3-D rows are live** (real captured CH Jacobians). The
**device-scale ladder is cited** from
`docs/dev/2026-07-13-m5-device-assembly.md` (slab64 nnz) and
`docs/dev/2026-07-13-blockch-mpf.md` (B4 film128, B5 mk32 int32 study) —
those take minutes per step, so we do not re-run them.
