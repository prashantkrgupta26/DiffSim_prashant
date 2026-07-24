# 01 — expected results (self-check)

The automated gate is **`baseline.yaml`** (tolerance-based scientific
invariants, not bit-identity), checked by the harness:

```bash
python run.py --config configs/ldc.yaml --mode reference --output outputs/ldc --overwrite
python gen_figures.py --run-dir outputs/ldc     # centerline figure + numbers/c1.tex
```

Reference mode (level 4 = 16×16, Re=100, dt=0.05, 200 steps) reproduces the
following to a few significant figures. Different cards/BLAS differ in the
last digits — the tolerances in `baseline.yaml` absorb that.

## Centerline $u(y)$ at $x=0.5$

| station $y$ | projection $u$ | monolithic $u$ | Ghia $u$ |
|---|---|---|---|
| 0.9766 | +0.8376 | +0.8529 | +0.84123 |
| 0.5000 | −0.1894 | −0.1462 | −0.20581 |

$$
\max|\text{proj}-\text{mono}| = 0.0486,\quad
\max|\text{proj}-\text{Ghia}| = 0.0164,\quad
\max|\text{mono}-\text{Ghia}| = 0.0620.
$$

$\|\nabla\!\cdot u\|$: monolithic **0.194**, projection **2.035**.

## What must be true regardless of hardware

- **The monolithic oracle tracks Ghia** within the coarse-mesh band
  (`max|mono−Ghia| = 0.062`). The level-4 (16×16) mesh cannot resolve
  Ghia's $129^2$ table to the third digit — that is expected. Refining to
  level 5 tightens it.
- **The projection tracks the same-mesh monolithic** (`max|proj−mono| =
  0.049`) — the course's central *faithfulness* check. This is a stronger,
  mesh-independent statement than the Ghia comparison: the two engines
  should agree on the *same* mesh whatever the mesh resolves.
- **The pointwise $\|\nabla\!\cdot u\|$ is finite and bounded, not zero.**
  For the equal-order VMS scheme this is correct — the scheme controls the
  *weak* divergence. The two engines even show *different* pointwise
  $\|\nabla\!\cdot u\|$ (0.19 vs 2.0) legitimately; it is the wrong
  steady-state gate.

If your monolithic centerline drifts from Ghia by more than ~0.09, or the
field does not settle (rising $\|\nabla\!\cdot u\|$), re-read the
walkthrough — something differs from the tutorial.
