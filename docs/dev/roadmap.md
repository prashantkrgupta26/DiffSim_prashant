# DiffSim roadmap

Checkmarks = gated (a test or benchmark measures it) and committed. All numbers
are measured and locked in `tests/baselines/*.json` or the findings logs in this
folder.

- ✅ **M0 — octree foundations**: space-filling-curve build, 2:1 balance, node
  dedup, hanging constraints, k-generic kernels (k = 2, 3, 4).
- ✅ **M0.5 — k-generic refactor**: per-axis periodic topology, mixed p1/p2
  constraints (one-knob rule), machine-precision patch/trace/fuzz batteries.
- ✅ **M1a — SBM foundations**: surrogate boundary + Taylor shift, Nitsche
  Dirichlet, Eq.-21 Neumann with area correction (π/4 locks), P4 rotated-patch
  keystone (k = 2/3/4), the p2 Neumann band (node-band(3): 2-D orders 1.9–2.1;
  **3-D asymptotic 2nd order to L7 on Nova — 1.97/1.94/2.03**; the dyadic-radius
  halo rule, canonical r = 0.19).
- ✅ **M1b — incompressible NS**: VMS-stabilized p1, both steppers (Leray +
  monolithic), BDF1/2; cavity Re 100/1000, cylinder Re 20 (C_d = 1.352) + Re 100
  Strouhal, sphere Re 300; GPU solves (fused Krylov, cuDSS, AMGX).
- ✅ **M1c — adjoints + neural-SDF geometry**: taped kernels (findings-4c
  rules), transient chains (1e-8-class), Leray adjoint, shape + constitutive
  gradients; provided INR checkpoints (GENIE edit modes, window contract,
  projection hardening); heroes: **sphere steady 1.8e-3, sphere transient
  2.6e-4**.
- ✅ **M1d — device migration** (GH200 coherent-capacity probe pending
  external): cuDSS 300× over host splu; device CSR assembly 4–40×; zero-copy
  assemble→solve; steppers `use_device_assembly=True` (2.2× end-to-end,
  solver-bound); cuDSS mtlayer (~40× on refactorization loops). Formally closed
  2026-07-08 — device GP-field kernels (last per-step host compute migrated); D3
  trace host work 0.81 % / 0.04 % / 0.03 % of step; H1 hero epoch **44.3×**.
- ✅ **M2 — Heat/Mass + closures + p2-NS**: scalar brick (orders 2.00/3.00
  exact, VMS-complete residual), coupler (de Vahl Davis Nu **0.05 % / 0.02 %**),
  SBM-thermal composition (Péclet-aware Nitsche, consistent-flux extraction),
  closures-in-the-loop (S2 retrain demo, RMSE 0.049), pure-p2 framework (cylinder
  C_d = 1.334; **unified penalty law α ~ Pe × p²**).
- ✅ **M3 rungs 1–2 — differentiable adaptivity**: explicit transfer operators
  (adjoint exact across a re-carve), epoch continuation (exact at 2.7× trust
  region where plain GN stalls); **the bunny headline: cell-scale GENIE ear-edit
  recovered from transient flow, err 1.23e-3** (six-run mechanism ladder). Rung 3
  (relaxed classification) = research brief `docs/theory/n6_relaxed_classification_brief.md`.
- ✅ **M4 — differentiable phase-field & learned thermodynamics**: AC + CH bricks
  (orders exact; mass 1.9e-15), spatial adaptivity (zero-drift nested transfer),
  temporal adaptivity (178× Δt growth; evaporation dt-cap), ternary Onsager CH,
  **Wodo CMS-2012 evaporating films — all 14 cases at 250×100 in ~11 min on one
  A100** (≈100× the 2012 cost), and the punchline: **the first learned free
  energy** (Flory–Huggins χ_pf/χ_ps/χ_fs/k_e recovered to 1e-7), with its
  gauge/identifiability structure mapped. F-series tutorial + LaTeX course doc.
  One pending external: the H200 3-D stretch case.

## Forward

- Learning rung 2: instrument-space observables (S(q,t), film height h(t), PSF)
  + composition-diverse protocols → genuinely beyond-Flory–Huggins recovery.
- CH block preconditioner for full-resolution 3-D films (15.2 M dofs).
- PNP electrokinetics (production DendrIon physics); crystallization (η fields).
- Space-time (k = 4), multi-GPU, THB refinement.

Milestone reports (the full story per milestone) live alongside this file:
`m1-`, `m2-`, `m4-`, `m1d-milestone-report.md`.
