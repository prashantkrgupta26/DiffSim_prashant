# Instructor companion — P3 Substrate / surface energy and boundary conditions

*Teaching notes for the instructor. Student-facing material is in
`README.md` and the course document chapter `p3.tex`. Grading is in
`grading_rubric.md`; hints and selected solutions in `solutions/`.*

## What this chapter is really teaching

Mixed Cahn–Hilliard carries *two* natural boundary integrals, and this
chapter's whole job is to make students stop treating "boundary
condition" as one undifferentiated idea. Three ideas do the teaching:

1. **A boundary condition is a choice about a specific equation's
   integral, not a blanket property of the domain edge.** The mass-flux
   condition lives on the φ (mass-balance) row; the wall condition lives
   on the μ row. You can turn one on without touching the other — that
   is exactly what the wall energy does (no-flux φ, wall-energy μ).
2. **Conservation is a statement about the flux row, not about "did the
   boundary do something."** The wall energy visibly changes the
   morphology (enrichment/depletion) while leaving `∫φ dV` untouched,
   because it never appears on the φ rows. Students who see a
   stratified morphology and assume mass moved have conflated
   *redistribution* with *transport*.
3. **Only the quadrature integral is the conserved quantity.** A nodal
   mean near a Neumann-type wall boundary picks up an `O(h)` weighting
   artifact that looks like a ~10⁻³ "drift" — this is not a leak, it is
   a wrong way to measure the right thing. The lesson generalizes: always
   ask which quantity a conservation law is stated for before trusting a
   "drift" number.

## Common student misconceptions & typical incorrect conclusions

- **"The wall enriches the substrate, so it must be adding mass there."**
  No — the wall condition is a condition on `φ` (via `grad φ · n`), not on
  the mass flux `grad μ · n`, which stays homogeneous Neumann throughout.
  Enrichment is a *redistribution* of the same total mass toward the
  boundary, not an injection.
- **"The nodal-mean φ near the wall should also be conserved."** False —
  `EXPECTED.md` explicitly flags that a nodal-mean check against the
  nominal 0.5 shows a spurious ~10⁻³ drift that is a boundary-weighting
  artifact. Only the quadrature integral `∫φ dV` (the same Gauss points
  the assembly uses) is the conserved quantity; push students to `
  diffsim.diagnostics.conservation.quadrature_mass`.
- **"A Dirichlet BC would behave the same way, just pinned instead of a
  wall energy."** No — Dirichlet pins `φ` or `μ` to a prescribed value,
  which *is* a reservoir and *does* exchange mass; the wall energy is a
  Neumann-type natural condition and conserves mass exactly. This is the
  taxonomy keybox in `p3.tex` — make sure it lands.
- **"projected_dofs = 0 means conservation is guaranteed by the code, full
  stop."** It means conservation holds *for this bounded well*. A
  bounds-violating IC or a linear wall (`h=0`, no interior minimum) can
  fire the projection or drive the equilibrium itself outside `(0,1)`.
  The honest claim is narrower: interior `φ*` ⇒ unconditional
  conservation; the projection firing on transient iterates does not
  corrupt the *converged* answer, but a genuinely inadmissible
  equilibrium would.
- **"The boundary-layer width scales the same way as the bulk interface
  width (Chapter P1's `ℓ ~ √(κ/W)`)."** They look similar but are
  different objects: P1's `ℓ` is the interior phase-boundary width set by
  the ratio `κ/W`; this chapter's `δ` is the *wall* decay length measured
  on a *stable, sub-spinodal* bulk (so there is no competing interior
  structure), and it scales as `δ ~ √κ` at fixed `f''` — students who
  quench into the demixing regime to measure `δ` are conflating the two
  length scales and will get a confounded number.

## Expected runtime ranges (RTX 6000 Ada; scale by card)

| mode | mesh | wall | notes |
|---|---|---|---|
| quick | 32×32 | comparable to P1's measured ≈ 75 s | CI smoke tier; most of it Warp compile + Python start-up |
| reference | 64×64, χ=2.2, quench to t=0.4 | a few minutes (target 5–30 min) | the `EXPECTED.md` numbers |
| research | 128×128, denser κ sweep | tens of minutes | finer boundary-layer resolution |

First run of a session pays a one-time Warp kernel-compile cost;
subsequent runs in the same environment are faster.

## Common CUDA / solver errors students hit

- **`cudss` selected on the wall-energy assembly.** Same story as P1: the
  `(φ,μ)` block is indefinite even with the extra wall face term, so
  cuDSS's no-pivoting factorization is unsuitable. `run_harness.py` names
  `splu` as the documented default solver for exactly this reason.
- **Treating `projected_dofs = 0` as evidence the run is fine, when
  `exit_reason` shows the FH log-barrier stiffened instead.** Pushing
  `φ*` too close to 0/1, or switching to a linear wall, can collapse the
  time step (non-termination) *without* ever tripping the box projection
  — check `exit_reason`, not just `projected_dofs`.
- **Measuring "mass conservation" from a nodal mean.** This produces a
  spurious ~10⁻³ drift near the wall (a boundary-quadrature-weight
  artifact) that some students mistake for a solver bug. Point them at
  the quadrature mass diagnostic.
- **Confusing the opposing-wall and confined-lateral cases' boundary
  setup with a periodicity bug.** The confined case intentionally
  replaces the periodic lateral sides with no-flux walls; students who
  expect periodic images may misread the field plot as a bug.

## Discussion prompts

- If a wall enriches the substrate with `φ ≈ 0.75` while the bulk mean
  stays at `φ̄ = 0.5`, where did the depleted material go, and how do you
  know from the diagnostics rather than the picture?
- The chapter distinguishes natural no-flux, wall-energy Neumann, and
  Dirichlet as three distinct boundary types. Give a physical device
  scenario where you would *want* the Dirichlet (mass-exchanging)
  condition instead of the wall energy used here.
- Why does the boundary-layer scaling study deliberately use a
  sub-spinodal bulk instead of the demixing χ=2.7 case used elsewhere in
  the chapter? What would go wrong if you measured `δ(κ)` during
  demixing?
