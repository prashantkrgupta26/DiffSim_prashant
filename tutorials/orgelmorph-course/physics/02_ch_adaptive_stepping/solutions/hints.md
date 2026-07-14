# Hints — P2 exploratory questions

*Hints for every "Explore on your own" question (course document
`p2.tex`, §"Explore on your own"). These are nudges, not answers — full
worked solutions for two verification exercises are in
`verification_solutions.md`. Do not distribute the solutions file to
students before the deadline.*

**Q1 — Move the tolerance across the knee (`work_tol`); confirm the
pixelwise error stops falling while the solve count keeps rising.**
Hint: sweep `tols` past the reference value and re-plot the left panel
of Fig.~p2conv. Below the knee, `c(T)` relative L2 flattens (it has hit
the trajectory-sensitivity floor described in the "robust statistics vs
the pixel field" keybox — of order 10⁻³) while the solve count keeps
climbing linearly-ish with `1/tol^{1/(p+1)}`. The free-energy error,
being a robust statistic, keeps falling a little further — contrast the
two curves.

**Q2 — Widen the interface (increase `κ`). Does the stiffness cliff
move to a larger safe fixed `Δt`?**
Hint: `λ_max ~ M/(4κ)` is the fastest spinodal growth rate; the step
that resolves it scales like `1/λ_max ~ 4κ/M`. Larger `κ` → smaller
`λ_max` → a *larger* safe fixed step, i.e. the cliff moves right.
Re-run the fixed-`dt` sweep at a larger `κ` (e.g. the quick-mode value)
and confirm the cliff (the `dt` where relative L2 jumps by more than an
order of magnitude) sits further out than at the sharp reference
`κ = 2×10⁻³`. See `verification_solutions.md` for the full worked
version.

**Q3 — Lengthen the horizon (`t_end`) and raise `dt_max`; does the
matched-accuracy speed-up grow?**
Hint: the adaptive step is already small during the fixed-duration
quench and only gets to grow during the (now longer) coarsening tail.
A fixed step must stay pinned at the quench's small scale for the
*whole* horizon regardless of length, so its cost grows linearly with
`t_end` while the adaptive run's cost barely changes once past the
quench — the ratio (and hence the speed-up) grows with the horizon.
This is the general argument for why adaptivity pays off more the more
scale-separated (quench vs. tail) the problem is.

**Q4 — Change the RNG seed and rerun; which reported quantities move
and which do not?**
Hint: exactly the seed-sensitivity story from Chapter P1. The
pixel field and the exact coarsened pattern move (which domain merges
when is chaotic); the robust statistics — free energy, structure-factor
length scale, phase fractions — do not, within their stated tolerances.
This is also why the pixelwise `c(T)` error has a floor that no amount
of tightening `tol` can push below.

**Q5 — Turn the controller off using the largest fixed step above the
cliff; what does the morphology look like, and which diagnostic flags
the failure most clearly?**
Hint: run the fixed-`dt` sweep at (or above) `PtwoCliffdt`. The
morphology is visibly destroyed (Fig.~p2morph-style comparison), but
the *quantitative* tell is the relative L2 of `c(T)` jumping by more
than an order of magnitude relative to a safe step — check whether the
free-energy error or length-scale error move as sharply, or whether the
pixelwise metric is the more sensitive flag here.
