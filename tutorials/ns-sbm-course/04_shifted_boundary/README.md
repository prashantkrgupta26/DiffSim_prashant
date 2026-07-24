# 04 — The Shifted Boundary Method (a genuine shift, $d\neq0$)

The body sits **off** the mesh grid. A square whose center is offset by a
sub-cell fraction ($\text{offset}=0.05$), so the grid-aligned surrogate no
longer coincides with the true `Box` face: $0<d_{\max}<h$. Now the Taylor
$(\nabla N)\cdot d$ term in the Nitsche block **and** the area correction
$\text{corr}=\tilde n\cdot n$ in the traction do real work. We compare drag
$C_d$ against the same-mesh monolithic oracle carrying the **identical**
shifted geometry.

**Read** the course document, Chapter 4 (*The Shifted Boundary Method*).
Start with `shifted.py` — the importable core.

**Run:**
```bash
python run.py                                       # level 4, Re=40, offset=0.05
python run.py --config configs/shift.yaml --mode reference --output outputs/sh
python run.py --zero-shift                           # anti-vacuity: break the match
```

`run.py` prints a drag self-check table; compare with
[`EXPECTED.md`](EXPECTED.md).

| file | role |
|------|------|
| `shifted.py` | the core: `run_shifted`, `zero_shift` — curates the ladder fixtures with a sub-cell offset; read this first |
| `run.py` | the driver you run; harness + drag self-check table |
| `gen_figures.py` | regenerates the drag figure + `numbers/c4.tex` |
| `EXPECTED.md` | reference numbers your run should reproduce |
| `baseline.yaml` | the tolerance baseline the harness checks |

The chapter calls the real SBM bricks: `sbm_vector_dirichlet` (shifted
Nitsche) and `surrogate_traction` (area-corrected drag) in
`src/diffsim/sbm/vector.py`, and the surrogate/geometry machinery in
`src/diffsim/sbm/surrogate.py` — via the validated ladder marchers. No toy
solver. The core self-locates the repo `tests/` directory.

## Learning objectives

By the end of this chapter you can:

- Explain the surrogate boundary $\tilde\Gamma$, the distance vector $d$,
  the outward normal $n$, and the area correction $\text{corr}=\tilde n\cdot
  n$, and where each is produced (`GeometryData.evaluate`).
- Write the Taylor shift $S N_a = N_a + (\nabla N_a)\cdot d$ and say why
  $d=0$ recovers standard Nitsche (Chapter 03).
- Read the area-corrected surrogate traction and the orientation contract
  ($\hat n = -\text{geo.n}$).
- Verify projection + SBM end-to-end: the split with the genuine shift
  matches the same-mesh-with-shift monolithic in drag.
- Demonstrate the shift is load-bearing via the zero-shift anti-vacuity
  break.

## Prerequisites

- **Concepts:** Nitsche's method (Chapter 03), Taylor expansion / level-set
  distance fields, immersed-boundary ideas.
- **Chapters:** 00–03.

## Expected cost

- **Device:** CPU runs level 4 in a few minutes (`splu`); a GPU is optional.
- **Quick** (300 steps) / **reference** (600 steps, the EXPECTED numbers) /
  **research** (level 5).

## Required deliverable

Submit the eight-item report of `../ASSESSMENT.md`, specialised:

1. The `config.resolved.yaml` + `metadata.json` from a reference run.
2. The `baseline.yaml` check green.
3. The drag figure (`gen_figures.py`).
4. **Headline:** $d_{\max}$, $d_{\max}/h$, and the projection vs monolithic
   $C_d$ with the rel-diff (≈0.3%).
5. **Verification:** faithfulness — projection + shift matches the
   same-mesh-**with-shift** monolithic.
6. **Failure:** run `--zero-shift` and show the match against the *true*
   shifted oracle breaks — the shift is load-bearing.
7. **Exploration:** vary `offset` (keeping $<h$) and plot $C_d$ or the
   rel-diff vs $d_{\max}/h$.
8. **Research bridge:** one paragraph on why the mesh-free shift is the
   enabler for 3-D immersed bodies (Chapter 05).
