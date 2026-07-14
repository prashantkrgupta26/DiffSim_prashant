# Hints — C2 exploratory questions

*Hints for every "Explore on your own" question (course document `c2.tex`,
Sec. "Explore on your own"). These are nudges, not answers — full worked
solutions for the two verification exercises are in
`verification_solutions.md`. Do not distribute the solutions file to
students before the deadline.*

**Q1 — Sweep Dirichlet wall value g in {-0.9, 0, +0.9}; explain why g=0
nearly conserves mass.**
Hint: the measured drifts are 0.95, 0.09, 0.94 respectively (`bc_test_matrix`,
`EXPECTED.md` §3) — `g=0` is an order of magnitude smaller than the other
two, not zero. Look at the initial condition in `run_bc`/`_run_generic`:
the blend is seeded as `0.05 * rng.standard_normal(...)`, i.e. mean ≈ 0.
A Dirichlet wall at `g=0` is pinning the boundary close to the *interior's
own mean*, so very little material has to flow to satisfy it — that is an
accident of this particular IC, not a property of the boundary condition.
Change the IC mean (e.g. `c_avg + 0.05*rng...` with `c_avg != 0`) and watch
the `g=0` drift stop being small.

**Q2 — Mixed BC (pin two opposite edges only): compare to all-edges
Dirichlet.**
Hint: the mixed row of the BC test matrix pins only 66 of 128 boundary
free-nodes (left/right edges) and leaves top/bottom natural. Compare its
mass drift and edge composition to the all-edges `c=+0.9` row — is the
drift roughly half, or does it scale differently? Look at the boundary
layer near the *unpinned* edges in the field image: does it look like the
no-flux case there, or is it still perturbed by diffusion from the pinned
edges? This geometry (contacts on two sides, sealed on the others) is the
common device configuration — keep it in mind for later device chapters.

**Q3 — What should mu be at a composition-pinned contact?**
Hint: think about what over-determining the boundary buys you. C1 pinned
both `c` and `mu` specifically so the discrete solution could be forced
to match a *known* manufactured field — that is only possible/useful
because the exact solution's `mu` was already known from the manufactured
`c`. At a real composition-controlled contact, `mu` is *not* a quantity
you get to choose — it must be whatever `f'(c) - kappa lapl(c)` evaluates
to for the (unknown) interior solution near the wall. A `c`-only Dirichlet
condition therefore needs the solver to treat `mu` at that node as a free
unknown determined by the interior physics, not a boundary datum — check
what changes in `CahnHilliardStepper`'s `dirichlet=`/`gc_fn`/`gm_fn`
signature if you only supply `gc_fn`.

**Q4 — Push penalty beta past 1e6; find the conditioning crossover.**
Hint: extend the `betas` argument of `strong_vs_weak_dirichlet` well past
`1e6` (try steps of 10x: `1e7, 1e8, ..., 1e14, 1e16`) and track both
`bc_err` *and* `np.linalg.cond(A_p)` (or just watch `bc_err` stop
improving / start growing). The mechanism: `A[i,i] += beta` makes that
diagonal entry many orders of magnitude larger than the O(1/h) stiffness
entries elsewhere in the row; in double precision (~16 significant
digits), once `beta` approaches the reciprocal of machine epsilon relative
to the matrix's intrinsic scale, the solve can no longer resolve the rest
of the row accurately. See `verification_solutions.md` for the full
worked version.

**Q5 — Does the no-flux mass drift (~1e-16) grow with mesh refinement?**
Hint: re-run `run_bc("noflux", level=level+1, ...)` and compare
`mass_drift` at the two levels. If conservation here came from
accumulated floating-point round-off across more quadrature points, you
would expect the drift to grow (roughly as sqrt(number of operations)
for random rounding, or linearly for correlated error) as the mesh is
refined. If it instead stays pinned at machine epsilon regardless of
level, that tells you conservation is enforced *structurally* (the
discrete divergence theorem is exact by construction of the weak form),
not merely "small because nothing much happens" — re-read the flux
balance section (`c2.tex`, "The flux balance ties conservation to the
wall") for why the discrete scheme makes `dm/dt` and the net boundary
flux the *same* quantity by construction.
