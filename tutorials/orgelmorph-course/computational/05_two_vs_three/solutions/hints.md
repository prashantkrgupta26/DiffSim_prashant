# Hints — C5 exploratory questions

*Hints for every "Explore on your own" question (course document `c5.tex`,
§"Explore on your own"). These are nudges, not answers — full worked
solutions for the two verification exercises are in
`verification_solutions.md`. Do not distribute the solutions file to
students before the deadline.*

**Q1 — `estimate_capacity.py --idx-bytes 8` at $256^3$: how much does
the total grow, and does int64 alone make it fit 48 GB?**
Hint: only the *index-bearing* buffers grow (column indices, row
pointers, the device index mirror, the local-to-global map, and the
index share of the solver workspace) — the CSR *values*, the
solution/RHS/history vectors, and the output snapshot are float64
already and do not change. Compute the delta component-by-component
before summing; then compare the new total against 48 GB and against
the *solver workspace* line alone — which one is still over the wall?

**Q2 — Coarsen further in 2-D and 3-D; do the coarsening rates
differ?**
Hint: run `physics_comparison()` (or a modified `_spinodal_to`) at a
larger `steps` and track `interfacial_area_density` over time in both
dimensions, not just the one-shot equal-resolution snapshot the chapter
reports. Because a 3-D interface is a surface (not a curve), the same
"interface shrinks as domains merge" mechanism operates on a different
geometric quantity — think about how area-to-volume scaling changes the
rate at which interfacial area is paid down as $L(t)$ grows.

**Q3 — Fit a power law to the live 3-D step times.**
Hint: `scaling_row()` already returns `step_s` for each level; fit
$\text{step}\sim\text{dofs}^\alpha$ in log–log across the three live
3-D levels. To separate "which stage drives $\alpha$", you will need to
time host assembly and the solve separately (instrument `_capture_nnz`
or `_spinodal_to`, or use the timers C4 already exposes) rather than
just the total step time — then compare the two exponents to the C4
device-assembly story.

**Q4 — Interface-only octree refinement changes the 3-D growth law.**
Hint: a *volume*-refined uniform grid scales dofs as (cells per
side)$^3$; a *surface*-refined mesh (refining only where an interface
sits) scales the refined region as (cells per side)$^2$ times a
constant interface thickness in cells. Compare that estimate against
the uniform $\CfiveThreeDdofs$-style dof count at the same resolution,
and think about why the gap between "volume" and "area" scaling only
opens up once $\dim=3$ (in 2-D, "surface" is already 1-D, so adaptivity
buys comparatively less).

**Q5 — A diagnostic that distinguishes percolating from droplet
morphologies, and why it is degenerate in 2-D.**
Hint: look at `diffsim.diagnostics.morphology` for a connected-component
/ labeling-style function — a percolating phase spans the domain along
every axis (a spanning cluster), a droplet phase does not. Then ask: in
2-D, can *both* phases simultaneously form a spanning cluster without
one of them cutting the domain and blocking the other? (Recall the
`c5.tex` claim that bicontinuity is topologically impossible for two
phases in 2-D — that is exactly why the same diagnostic collapses to a
trivial answer there.)
