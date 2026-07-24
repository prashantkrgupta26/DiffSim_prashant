# 03 — hints & selected solutions (instructor-only)

## Hints for the exploratory questions

**Q ($\alpha$ sweep).** Very small $\alpha$: the penalty cannot balance the
consistency term, the no-slip condition *leaks*, and $C_d$ drifts low. Very
large $\alpha$: the boundary block dominates, conditioning degrades, and the
solve slows / loses accuracy. The useful window is broad (the examples use
$\alpha=10$); the plot should be roughly flat across $\alpha\in[5,50]$ with
$C_d$ falling toward 0 as $\alpha\to0$ and the iteration count rising as
$\alpha\to100+$.

**Q (strong vs weak at level 4).** Neither is "more correct" a priori on a
coarse mesh — they represent the obstacle trace differently. The way to
decide is to refine (level 5): both should converge toward a common $C_d$.
Report the level-4 gap and the direction of convergence, not a verdict.

## Full solution — the anti-vacuity ($\alpha=0$) break

```bash
python run.py --alpha 0 --output outputs/noalpha --overwrite
```

With the penalty removed, the Nitsche block loses coercivity and the no-slip
condition is no longer enforced; the flow passes through the "obstacle" and
$C_d$ collapses toward 0 (or the run is far off the oracle). The baseline
`cd_proj: reference 1.3529 atol 0.05` would FAIL — which is the point: the
penalty is load-bearing. This is the direct analogue of Chapter 04's
zero-shift anti-vacuity break.

## Full solution — the strong-split open-outflow instability

The base single-pass "strong" projection (mode `strong` *without* the
consistent machinery) on the open channel outflow fabricates a spurious
boundary pressure term and can go unstable (`blew_up: true`). The validated,
robust path is the weak `consistent_projection` one (the default). If a
student sees a blow-up, check they are on the weak path; the finding itself
is the rung-A lesson (see the course document Chapter 3, "Note").
