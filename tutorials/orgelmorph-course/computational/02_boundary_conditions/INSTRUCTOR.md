# Instructor companion — C2 Boundary conditions, numerically

*Teaching notes for the instructor. Student-facing material is in
`README.md` and the course document chapter `c2.tex`. Grading is in
`grading_rubric.md`; hints and selected solutions in `solutions/`.*

## What this chapter is really teaching

The mechanics are a vehicle for the course's central habit applied to
boundary conditions specifically: **a boundary condition is a modelling
choice with a measurable consequence, not a detail to accept from a
default.** Four ideas do the teaching:

1. **The mixed system has two boundary conditions, not one.** Because
   the fourth-order PDE is split into the mixed `(c,mu)` form, integrating
   by parts twice produces *two* surface integrals — the mass flux on `mu`
   and the interfacial (wetting) term on `c`. A complete BC must say what
   happens to both; students who only think about "the" boundary condition
   have missed one of them.
2. **Conservation is a consequence of imposition, not of the PDE alone.**
   Natural and wall-energy BCs conserve mass exactly; Dirichlet and MMS do
   not, by construction (they are reservoirs / devices). This is testable,
   not a matter of opinion — the flux balance makes it an identity.
3. **How you impose a constraint is a linear-algebra decision with
   consequences.** Row replacement, symmetric elimination, and penalty all
   impose "the same" Dirichlet value but leave the matrix in different
   states (asymmetric / SPD / SPD-but-approximate). This recurs in later
   solver chapters (C4).
4. **Manufactured vs. physical boundary conditions are different
   objects.** Pinning both `c` and `mu` (as C1's MMS study did) is a
   device for generating an exact reference solution; a physical
   composition-controlled contact pins only `c`. Conflating the two is a
   category error, not a detail.

## Common student misconceptions & typical incorrect conclusions

- **"Pinning both `c` and `mu` is what a real electrode/contact does."**
  No — pinning both fields is an **MMS device** (over-determines the
  boundary so a discrete solution can be forced to match a known exact
  field). A physical composition-pinned contact pins only `c`; what `mu`
  should be there follows from the interior physics, not from the modeler
  choosing it. This tutorial's Dirichlet runs pin `mu=0` at the wall only
  as a convenience of the binary brick — read that as a modelling
  shortcut, not a law (see the `mu` Explore-on-your-own question).
- **"Dirichlet conserves mass because the field looks like it
  equilibrates."** No — Dirichlet BCs do *not* conserve mass; they are a
  reservoir by construction. The measured mass drift under the wall
  `c=0.9` is 0.95 — nearly the full possible drift — while no-flux stays
  at machine precision (1.3e-16). Natural (no-flux) and wall-energy
  (wetting/Robin) BCs conserve mass; Dirichlet and MMS do not.
- **"The `c=0.0` Dirichlet run conserves mass, so Dirichlet can
  conserve."** This is the trap in the BC test matrix: `c=0.0` shows a
  drift of only 0.094 (an order of magnitude smaller than `c=\pm0.9`'s
  ~0.95) purely because the initial mean composition is close to zero —
  an **initial-condition coincidence**, not a property of the boundary
  condition. Push students to re-run with a different IC mean and watch
  the "near-conservation" disappear.
- **"A zero-flux (`g=0`) case that conserves mass proves the BC
  conserves in general."** Same trap in a different guise: a flux/BC
  configuration that happens to conserve on *one* run (because the
  boundary value matches the bulk state) is not evidence that the BC
  *class* conserves mass — only natural and wall-energy BCs do that
  unconditionally. Always check the flux-balance identity, not a single
  run's mass-drift number.
- **"The penalty method is strictly worse than strong imposition."**
  Missing the point: penalty preserves symmetry (useful on hanging-node
  / immersed boundaries where a clean row does not exist) at the cost of
  only approximate ($O(1/\beta)$) satisfaction. Students should report the
  trade, not declare a winner.

## Expected runtime ranges (RTX 6000 Ada; scale by card)

| stage | content | wall | notes |
|---|---|---|---|
| `run.py` (default) | natural vs Dirichlet + flux balance + BC test matrix + weak/strong demo | **~30-60 s** | comparable to Physics P1 quick mode (~75 s measured); most of it is Warp kernel compile + Python start-up |
| `gen_figures.py` | 5 figures + `numbers/c2.tex` | comparable to `run.py` | reruns the same computations |

First run of a session pays a one-time Warp kernel-compile cost; the
weak-vs-strong Dirichlet demo (`strong_vs_weak_dirichlet`) is a tiny dense
NumPy solve and is effectively instantaneous — it needs no GPU at all.

## Common CUDA / solver errors students hit

- **Dirichlet run looks identical to no-flux.** Usually means `dirichlet=`
  was not actually passed to `CahnHilliardStepper`, or `bidx` (the
  boundary free-node indices from `boundary_free_nodes`) is empty — check
  that `cons.free_nodes` and the coordinate tolerance (`1e-12`) match the
  mesh actually built.
- **Field difference (`c_nf` vs `c_dir`) is near zero.** The two runs must
  share the identical fixed seed and initial condition (`seed=3` by
  default) — if a student changes the seed for one run but not the other,
  the "boundary effect" is confounded with a different IC.
- **`ALL CHECKS: FAIL` on the symmetry checks.** `np.allclose(A_rr, A_rr.T)`
  should be `False` for row-replacement and `True` for symmetric
  elimination; if both come back `True`, the student likely edited
  `dirichlet_strong` and lost the asymmetric row-only branch.
- **Penalty error does not shrink with `beta`.** Check that `beta` is
  actually reaching the diagonal (`A[i,i] += beta`) and not overwriting it;
  also check the `betas` sweep passed to `strong_vs_weak_dirichlet` is
  increasing, not a single fixed value.
- **`nan`/`inf` at very large `beta`.** Past some `beta` the added diagonal
  term is many orders of magnitude larger than the rest of the (order-one)
  stiffness entries; in double precision that starts to swamp the other
  terms in the row and the dense solve loses accuracy — the conditioning
  crossover the last Explore-on-your-own question asks about.

## Discussion prompts

- Why does the *mixed* `(c,mu)` formulation force two boundary
  conditions where a naive scalar diffusion equation only needs one? What
  would go wrong if you only specified one of them?
- The wall-energy (wetting/Robin) BC conserves mass exactly, just like
  natural — why? What is actually different between the two physically?
- If row replacement is "fast and exact," why would anyone choose the
  slower, only-approximate penalty method? Where does the natural-row
  argument (hanging nodes, immersed boundaries) actually bite?
- Is "conserves mass to machine precision" the same claim as "the
  discretization is accurate"? Connect this back to Physics P1's three
  "energy decreases" statements — what is the analogous confusion here?
