# 04 — expected results (self-check)

The automated gate is **`baseline.yaml`**, checked by the harness:

```bash
python run.py --config configs/shift.yaml --mode reference --output outputs/sh --overwrite
python gen_figures.py --run-dir outputs/sh     # drag figure + numbers/c4.tex
```

Reference mode (level 4, Re=40, offset=0.05, dt=0.02) reproduces:

$$
d_{\max}=0.0500\ \ (0<d_{\max}<h=0.0625,\ d_{\max}/h=0.80),\quad
\text{area-corrected GPs}=4.
$$

| | projection | monolithic (true shift) | rel-diff |
|---|---|---|---|
| $C_d$ | **+1.5407** | **+1.5457** | **0.32%** |
| mean $|u|$ | 0.9090 | 0.8527 | 6.60% |

## What must be true regardless of hardware

- **The shift is genuinely active:** $0<d_{\max}<h$, $d_{\max}/h=0.80$ — a
  real sub-cell shift, with 4 area-corrected surrogate Gauss points. The
  square is *not* cell-aligned (contrast Chapter 03's $d_{\max}=0$).
- **Projection + SBM works end-to-end in 2-D:** the consistent-projection
  split with the genuine shift active matches the same-mesh-**with-shift**
  monolithic to **0.32%** in drag. The monolithic oracle carries the
  *identical* shifted geometry ($d$, $\text{corr}$), so this is an
  apples-to-apples same-mesh comparison.
- **The shift is load-bearing (anti-vacuity).** Running `--zero-shift`
  ($\text{geo.d}=0$, $\text{geo.corr}=1$) while comparing against the *true*
  shifted oracle **breaks** the match — proof the Taylor $(\nabla N)\cdot d$
  term and the area correction are doing real work. The `zero_shift: false`
  gate ensures the reference run is *not* the zeroed one.

If your reference `dmax` is 0 (the offset was too small / the carve was
cell-aligned), or the rel-diff exceeds ~15%, re-read the walkthrough — the
shift is not active.
