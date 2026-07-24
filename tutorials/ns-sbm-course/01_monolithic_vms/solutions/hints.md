# 01 — hints & selected solutions (instructor-only)

## Hints for the exploratory questions

**Q (why is $-\nu\Delta u_h$ zero at $P_1$ but needed at $P_2$?).** The
viscous strong-residual term is $-\nu\,\nabla\!\cdot(\nabla u_h)$. For $Q1$
(bilinear) shape functions the *pure* second derivatives $\partial_{ii}N$
vanish on axis-aligned cubes, so the term is identically zero; only the
mixed $\partial_{ij}$ survive and they do not enter the Laplacian. At $P_2$
(biquadratic) the second derivatives are nonzero and the term is required
for consistency (`ns_bricks.py:406-407`).

**Q (cuDSS on this cavity?).** It *hurts*. The monolithic block
$\begin{bmatrix}F&G\\D&C\end{bmatrix}$ is indefinite; cuDSS factorizes
without partial pivoting, which is safe only for (quasi-)SPD systems. Expect
divergence / NaN. `splu` pivots and is exact. Contrast Chapter 02: the
projection's PPE *is* SPD, and there cuDSS/AMG is exactly the right tool.

## Full solution — the mesh-refinement study

```bash
python run.py --config configs/ldc.yaml --mode research --output outputs/ldc5 --overwrite
```

Level 5 (32×32) tightens `max|mono−Ghia|` from ~0.062 toward ~0.03–0.04
(card-dependent), while `max|proj−mono|` (faithfulness) stays small and
mesh-largely-independent. The lesson: refinement reduces *discretization*
error (the Ghia gap); it does not need to change the *faithfulness* gap,
because the two engines share the mesh. Report both curves.

## Full solution — the cuDSS failure

```bash
python run.py --config configs/ldc.yaml --solver cudss --output outputs/bad --overwrite
```

Either a solver exception or a centerline that violates the
`u_mono_lid`/`d_mono_ghia` bounds in `baseline.yaml` — the gate catches it.
Mechanism: indefinite operator + no pivoting. This is the same lesson as
Chapter 00's smoke, now on a real steady solve.
