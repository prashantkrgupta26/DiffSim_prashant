# C5 — expected results (self-check)

Running `python run.py` measures the 2-D-vs-3-D scaling live and echoes
the cited device-scale ladder. The dof and nnz counts are
**deterministic** and should match exactly; the step times vary by card.

## Live scaling (real Cahn–Hilliard systems)

| dim | level | dofs | nnz | nnz/dof | mem (MB) |
|---|---|---|---|---|---|
| 2-D | 5 | 2,178 | 37,636 | 17.3 | 0.5 |
| 2-D | 6 | 8,450 | 148,996 | 17.6 | 1.8 |
| 2-D | 7 | 33,282 | 592,900 | 17.8 | 7.1 |
| 3-D | 3 | 1,458 | 62,500 | 42.9 | 0.8 |
| 3-D | 4 | 9,826 | 470,596 | 47.9 | 5.6 |
| 3-D | 5 | 71,874 | 3,650,692 | 50.8 | 43.8 |

- **dofs per level:** 2-D ×3.9, 3-D ×7.0 (theory ×4 vs ×8).
- **nnz per dof:** 2-D ≈ 18, 3-D ≈ 47 (9-point vs 27-point stencil for
  the mixed $(c,\mu)$ system).
- At comparable dofs, 3-D carries ≈ 6× the nnz and ≈ 6–7× the step time.

## Cited device-scale ladder (dev notes — NOT re-run)

| case | dofs | nnz | int32 |
|---|---|---|---|
| 3d_slab64 | 417,792 | 65,028,096 | 3% |
| 3d_film128 (128³×64) | 6,389,760 | 1.02×10⁹ | 47% |
| mk32 128×128×48 (M=3,K=2) | 8,028,160 | 2.17×10⁹ | **OVERFLOW** |
| mk32 256×256×128 (Nova) | 84,541,440 | 2.28×10¹⁰ | **OVERFLOW** |

- The int32 CSR ceiling is nnz $< 2^{31} \approx 2.1\times10^9$. At
  $(M{=}3,K{=}2)$ production physics, the 128×128×48 superset pattern
  already overflows — which is why the block-masked pattern exists.
- 256×256×128 has no single-card stored CSR at all → matrix-free or
  multi-GPU.

**What must be true regardless of hardware:**

- **3-D grows faster in *both* dofs and nnz/dof** — the `CHECK ... PASS`
  line asserts exactly this. The matrix is bigger *and* denser.
- **dofs scale as $2^{\dim}$ per level** (×4 vs ×8) and **nnz/dof
  reflects the stencil** ($3^{\dim}-1$ neighbors).
- **The counts are exact** (uniform meshes, fixed): 2-D L7 has 33,282
  dofs, 3-D L5 has 71,874 dofs, etc. If they differ, something changed
  in the mesh or the constraint handling.

This compounding growth — more dofs, denser matrix, worse direct-solver
fill — is the whole reason 3-D forces the device assembly path and the
`blockch` solver of C4.
