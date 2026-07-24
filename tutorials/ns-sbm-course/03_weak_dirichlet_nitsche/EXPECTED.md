# 03 — expected results (self-check)

The automated gate is **`baseline.yaml`**, checked by the harness:

```bash
python run.py --config configs/square.yaml --mode reference --output outputs/sq --overwrite
python gen_figures.py --run-dir outputs/sq     # drag figure + numbers/c3.tex
```

Reference mode (level 4, Re=40, dt=0.02, weak Nitsche) reproduces:

| | projection | monolithic | rel-diff |
|---|---|---|---|
| $C_d$ | **+1.3529** | **+1.3941** | 2.96% |
| mean $|u|$ | 0.9057 | 0.8550 | 5.93% |

$d_{\max}=0$ (cell-aligned ⇒ standard Nitsche); the run does **not** blow up.

## What must be true regardless of hardware

- **The weak-Nitsche projection matches the same-mesh weak-Nitsche
  monolithic** in both drag (rel 3.0%) and velocity (rel 5.9%) — the
  faithfulness check, now with an immersed body imposed *weakly*. The
  Nitsche layer preserves faithfulness on the working projection.
- **$d_{\max}=0$**: the square is cell-aligned, so the SBM shift is inert
  and this is exactly standard Nitsche. (Chapter 04 turns the shift on.)
- **The bar is faithfulness, not the literature drag.** The confined
  unit-box channel inflates the absolute $C_d$ above the unconfined square
  value; both engines see the same confinement, so the meaningful comparison
  is between them.

## Notes on the modes

- **`--mode-noslip strong`** imposes no-slip by row replacement. On this
  cell-aligned square it gives a slightly different $C_d$ (a coarse-mesh
  difference in how the obstacle trace is represented); both are legitimate,
  and refinement narrows the gap.
- **The base "strong" single-pass split on the *open* outflow** (without the
  consistent-projection machinery) is pressure-unstable and can blow up — a
  known rung-A finding. The validated, robust path is the weak
  `consistent_projection` one, which is the default here.
- **`--alpha 0`** removes the penalty; the obstacle stops being felt and
  $C_d \to 0$ — the anti-vacuity check that the penalty is load-bearing.

If your run reports `blew_up: true` in the default weak mode, or the rel-diff
exceeds ~15%, re-read the walkthrough.
