# E0 — "Thinking Differentiable": the adjoint on-ramp for the curriculum

**Design spec, ratified 2026-07-18.** Adds a three-chapter mini-sequence
(E0a/E0b/E0c) at the front of Track E, plus a normative reference page and a
LaTeX companion. Audience: the research group — fluent in FEM theory and
practice, new to differentiable simulation. Requested by Baskar with explicit
content mandates (see §2). E1/E2 are untouched and become the payoff chapters.

## 1. Sources and attribution

Structure and example lineage adapted from **dolfin-adjoint/pyadjoint**
(mathematical background by Patrick E. Farrell; the "mother problem" and
"learning reaction rates" examples), with attribution in every chapter's
Background section and the README: cite Farrell, Ham, Funke & Rognes (2013)
and Mitusch, Funke & Dokken (2019). Reuse is adapted-with-attribution
(framings, example ideas, the three-way comparison device), not verbatim
blocks. DiffSim-native sources: the findings-4c taping rules, the M1c/M2/M4
adjoint stack, the H4 signal-scale protocol, the beyond-FH gauge story.

## 2. Content mandates (Baskar, 2026-07-18)

Step-by-step how-to-think; detailed code walkthrough; the recipe for ANY new
PDE (what to tape); how Warp abstracts it; simple runnable examples; **what
NOT to differentiate through** (e.g. the linear solver); **what can be
computed once and stored**; **compute-vs-memory trade-offs**; **how to pick
good J's**. All land as first-class sections (map in §3).

## 3. The chapters

House chapter contract throughout: *Learning outcome → Background →
documented code → Expected results (measured numbers to reproduce) →
Explore.* All scripts CPU-runnable in minutes at small sizes.

### E0a — Thinking differentiable: from residual to gradient
- Foreword (Farrell-adapted): the reduced functional J(m) = J(u(m), m);
  **information-flow reversal** as the central image (adjoints carry
  sensitivity backward); why the adjoint is LINEAR even when the PDE is not.
- **Three ways to a gradient** — finite differences, tangent-linear, adjoint —
  comparative table with the cost argument (n_params solves vs ONE transposed
  solve; when each wins).
- **Naked `wp.Tape` toy**: a five-line kernel, `tape.backward()`, inspect
  gradients — the mechanics with no FEM in the way.
- **Poisson dJ/dκ** (the mother problem, adapted): assemble → solve → J(u);
  adjoint = *the transposed solve you already know*; **the three-way check
  taught as a first-class concept** (adjoint vs tape vs FD — the house
  verification contract as pedagogy).
- First seed of the deep principle: *we did NOT differentiate through
  `splu`* — we differentiated the relation Au = b. Named and flagged
  "developed fully in E0b."
- Math call-outs (styled asides, frequent per Baskar): the Lagrangian view;
  discretize-then-differentiate vs differentiate-then-discretize and why
  DiffSim is discrete-adjoint.

### E0b — Anatomy of a taped brick
- Line-by-line walkthrough of a real DiffSim kernel + its VJP: what
  `wp.Tape` records (launch graph), `requires_grad` arrays, custom VJP vs
  unrolled tape, where cotangents enter, tape scope discipline; how the
  kernel-factory/cache pattern coexists with taping.
- **The do-not-differentiate list** (mandate): differentiate the RELATION,
  not the algorithm —
  * linear solver: never unroll Krylov/LU; the gradient path is the
    transposed solve (same factorization!);
  * Newton loops: implicit function theorem — differentiate the converged
    residual, not the iteration;
  * mesh build, octree carve, element classification, preflight checks:
    FROZEN by design (piecewise-constant in parameters; the E1
    classification-freeze is the worked instance);
  * RNG seeds, basis tabulation: inputs, not tape content.
- **Compute-once-and-store** (mandate): one factorization serves A and Aᵀ
  solves; tabulated basis/quadrature reused; symbolic patterns and GP-field
  precomputation outside the tape; what "frozen at GPs" means for gradients
  (the M2 one-way-coupler convention).
- Verification depth: the **⟨Av,w⟩ = ⟨v,Aᵀw⟩ dot-product test** per operator.
- Payoff example: **κ(x)-field recovery** — a small inverse crime, gradient
  descent, converging before your eyes.

### E0c — The recipe: differentiating any new PDE
- **The checklist** (the chapter's spine; normative twin lives on the
  reference page, §4): what to tape (which residual kernels w.r.t. which
  inputs) · what to freeze · what to store across steps · transposed-solve
  infrastructure · the verification ladder (dot-product per operator →
  three-way per gradient → transient-chain check) · nondifferentiability
  hazards (abs/min/max/thresholds/masks → relaxed forms) · the cost
  contract (backward ≤ 2.5× forward).
- **Transient heat chain**: taped BDF1 march, the backward sweep, what the
  primal must store.
- **Compute-vs-memory trade-offs** (mandate, full section): full-store vs
  recompute-from-checkpoints on the transient chain; a worked memory budget
  (bytes per step per dof, when the tape outgrows the GPU); revolve-style
  checkpointing named as the scaling path; measuring the 2.5× contract.
- **Choosing your J** (mandate, full section): smoothness hazards (min/abs/
  thresholds — and their relaxed replacements); **signal scale before
  optimizing** (the H4 protocol lesson: check dJ magnitude vs noise floor
  first); flat-J / vanishing-gradient geometries; identifiability and
  conditioning — the **beyond-FH gauge story as case study** (aliased
  directions, anchored bases, conditioning numbers as the diagnostic);
  regularization as science, not hack (the Tikhonov lesson); multi-observable
  / instrument-space J's (why diverse protocols make inverse problems
  well-posed — the bridge to SP-1).
- **Mini phase-field gradient** (AC/CH on a tiny grid): the group's own
  physics; dolfin-adjoint's "learning reaction rates" cited as lineage;
  forward pointer to F-track, M4, and the learned-closures program.

## 4. The reference page (normative twin)

`docs/site_src/theory/adjoint_readiness_checklist.md` (nav under Theory):
the terse, citable statement of the E0c checklist + the do-not-differentiate
list + the J-design rules. Specs, code reviews, and task briefs link HERE.
Sync rule: the page owns the normative list; E0c derives and references it
(the `p2_band_problem_statement` ↔ `for_students` pairing, applied again).
Absorbs/retires the scattered "findings-4c rules" citations going forward
(dev docs keep their history; new documents cite the page).

## 5. LaTeX companion

`docs/course/e0_differentiable_tutorial.tex` (F-track precedent): the
Farrell-style mathematical background adapted to DiffSim notation, the three
chapters' theory consolidated, checklist as an appendix. Built with tectonic
(gpubox toolchain). Not load-bearing for the chapters — the scripts stand
alone.

## 6. Delivery & gates

- Files: `tutorials/E_differentiable/E0a_thinking_differentiable.py`,
  `E0b_anatomy_of_a_taped_brick.py`, `E0c_recipe_new_pde.py`; README/table
  updates (tutorials/README.md, E_differentiable/README.md, site nav via
  build_site.py); the reference page; the LaTeX doc.
- Every chapter prints Expected-results numbers (three-way check magnitudes,
  recovery errors, order-of-cost ratios) reproducible on CPU in ≤ ~3 min.
- E0a and E0c join the tutorial-smoke CPU tier (small sizes); E0b if runtime
  permits.
- Gates: scripts run clean from repo root; printed numbers match the
  documented Expected results (tolerance bands, ≥2× headroom); three-way
  checks at the 1e-6 class; site builds with the new nav; LaTeX compiles.

## 7. Out of scope (YAGNI)

Continuous-adjoint derivations beyond asides; shape/topology-optimization
theory (E1/E2 carry the geometry story); revolve implementation (named, not
built); porting existing dev docs wholesale into the site; any solver-code
changes (this is pedagogy over the existing stack — if a walkthrough exposes
a genuine wart, record it, don't refactor here).
