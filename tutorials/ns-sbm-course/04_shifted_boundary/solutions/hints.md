# 04 — hints & selected solutions (instructor-only)

## Hints for the exploratory questions

**Q (Taylor accuracy as offset → h).** The shift $S N_a = N_a + (\nabla
N_a)\cdot d$ is a *first-order* extrapolation: its error is
$\mathcal{O}(|d|^2\,\|\nabla^2 N\|)$. As $|d|\to h$ the extrapolation reaches
across a full cell and the neglected second-order term grows; accuracy
degrades and eventually the surrogate is a poor stand-in for $\Gamma$. That
is why `offset < h` is enforced. At $d_{\max}/h=0.80$ we are already probing
the aggressive end and the match is still 0.3% — a good result.

**Q (which GPs are area-corrected?).** `corr = ñ·n` differs from 1 only where
the surrogate face normal $\tilde n$ (grid-aligned) is not parallel to the
true-boundary normal $n$ — i.e. at the *corners* of the offset square, where
the true face and the grid face point in different directions. On a
face-aligned edge $\tilde n = n$ and corr = 1. Hence only a handful (4 here).

## Full solution — the zero-shift anti-vacuity break

```bash
python run.py --zero-shift --output outputs/zero --overwrite
```

The projection now runs with `geo.d=0`, `geo.corr=1` (the shift terms
neutralised) while the monolithic oracle still carries the *true* shifted
geometry. The comparison is now zeroed-projection vs true-shift-oracle, and
the drag rel-diff jumps well above the 0.3% of the matched run (tens of
percent at this offset). The baseline's `cd_rel: max 0.15` would FAIL under
the true-oracle comparison — which is exactly the point: the Taylor term and
area correction are load-bearing, not decorative. (The reference run is
gated with `zero_shift: false` so a student cannot accidentally submit the
zeroed run as the headline.)

## Full solution — offset sweep

Vary `offset ∈ {0.02, 0.03, 0.05}` (all < h = 0.0625). `dmax/h` rises with
the offset; the matched projection tracks the shifted oracle across the
range (rel-diff stays small), while the *zeroed* projection's error grows
with `dmax/h` — a clean demonstration that the shift's contribution scales
with how far the body sits off the grid.
