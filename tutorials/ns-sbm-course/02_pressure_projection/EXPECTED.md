# 02 — expected results (self-check)

The automated gate is **`baseline.yaml`** (tolerance-based scientific
invariants), checked by the harness:

```bash
python run.py --config configs/ldc.yaml --mode reference --output outputs/ldc --overwrite
python gen_figures.py --run-dir outputs/ldc     # centerline figure + numbers/c2.tex
```

Reference mode (level 4, Re=100, dt=0.05, 200 steps) reproduces the same
cavity as Chapter 01 — the two engines run on identical fixtures.

## Centerline $u(y)$ at $x=0.5$

$$
\max|\text{proj}-\text{mono}| = 0.0486,\quad
\max|\text{proj}-\text{Ghia}| = 0.0164,\quad
\max|\text{mono}-\text{Ghia}| = 0.0620.
$$

$\|\nabla\!\cdot u\|$: projection **2.035**, monolithic **0.194**.

## What must be true regardless of hardware

- **The projection reproduces the same-mesh monolithic** — `max|proj−mono|
  = 0.049`. This is the whole point of the chapter: the scalable engine is
  *faithful* to the oracle on the same mesh. It is a stronger statement than
  the Ghia comparison because it is mesh-independent.
- **The projection also tracks Ghia** within the coarse-mesh band
  (`max|proj−Ghia| = 0.016`), in fact slightly closer than the monolithic
  here — a coincidence of the coarse mesh, not a claim of superiority.
- **The pointwise projection $\|\nabla\!\cdot u\| = 2.0$ is finite and
  bounded, not zero.** For the equal-order VMS scheme the *weak* / PPE-space
  divergence is the controlled quantity. The right divergence gate is the
  PPE-space solenoidality identity $\|\sigma B^\top\hat u - K_p\phi\|\approx
  0$, which is machine-zero at the fixed point — *not* the pointwise
  $\|\nabla\!\cdot u\|$.

**On `consistent_projection` and the outflow.** The enclosed cavity is
stabilised by a single pressure-DOF pin (free-node 0), so the base split is
already faithful here. The `consistent_projection` machinery becomes
*load-bearing* at an **open outflow** (Chapters 03–04), where the naïve
split fabricates a spurious boundary term and can go pressure-unstable. This
chapter establishes the engine; Chapter 03 is where the consistency items
earn their keep.

If your projection centerline drifts from the monolithic by more than ~0.08,
re-read the walkthrough.
