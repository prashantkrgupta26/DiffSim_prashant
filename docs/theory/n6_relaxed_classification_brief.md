# Relaxed Classification for Differentiable Immersed Optimization
## A self-contained research brief (N6 fork / M3 rung 3)

*Companion to `docs/p2_band_problem_statement.md` (same format: everything
needed is in this document). All numbers are measured (July 2026,
DiffSim prototype, float64).*

## 1. The problem in one paragraph

Immersed (SBM) discretizations make a DISCRETE decision per element —
retained or carved — from the sign structure of a level-set function
psi(x; alpha), where alpha parameterizes the geometry (for us: GENIE
edit-mode coefficients of a provided neural SDF). Gradient-based
optimization over alpha therefore differentiates a function that is
smooth WITHIN a fixed classification ("epoch") and JUMPS when any
element flips. We have measured both regimes precisely; the open
research question is whether RELAXING the classification (smoothing the
decision itself) yields optimization landscapes that beat the
epoch-continuation protocol we ship today.

## 2. What is measured (your evidence base)

- **Within-epoch smoothness vs wobble/h** (findings 4f-DATA;
  `benchmarks/data/landscape_4f.txt`): on a wobbly provided INR, the
  objective is quadratic down to alpha ~ 1e-5 at L4 but SIX ORDERS
  off-quadratic at L5 — micro-jump density tracks (surface wobble)/h,
  not the stabilization. Any relaxation scheme must beat this baseline
  at the SAME resolution.
- **Cross-epoch jumps** (tutorial arc): J = 183 spike at a mid-path
  re-carve; the carve-difference is the dominant landscape feature at
  cell-scale edits.
- **The failure of independent per-alpha carves** (M2-D v2): with each
  trial alpha carving its own mesh, GN escaped to a spurious J~0 well
  (|alpha| 6.6 vs 0.03) or stalled — the landscape is
  carve-jump-dominated.
- **The cure that works** (M3 rung 2, gated): epoch continuation —
  optimize within a trust region (0.35h of boundary drift), re-carve at
  epoch boundaries, transfer state through an explicit operator P
  (rung 1, adjoint-exact). Beyond-trust-region recovery: EXACT where
  plain GN stalls at 42% (`tests/test_adaptivity.py`).
- **The remaining gap** (bunny attempts): even with continuation, a
  4-mode cell-scale recovery on the bunny wanders between wells —
  identifiability AND landscape roughness interact.

## 3. The research question

Replace the hard classification chi_e = [psi_e < 0] with a relaxed
indicator chi_e(psi; tau) (e.g. sigmoid(psi_bar_e / (tau*h))) entering
(a) the volume integrals as a density/ersatz weight and/or (b) the face
set as blended surrogate contributions. Then:

Q1. Does the relaxed objective's gradient CONVERGE to the epoch-wise
    gradient as tau -> 0, and at what rate? (The adjoint through the
    relaxed form is straightforwardly exact for tau > 0 — that is its
    appeal.)
Q2. At what tau does the landscape become descent-navigable at
    cell-scale edits — and does the required tau destroy accuracy
    (ersatz-material error scales like tau*h?) worse than the
    continuation protocol's epoch management?
Q3. Head-to-head on the SAME task (the sphere rung-2 gate config, then
    the bunny ear edit): relaxed-tau descent vs epoch continuation —
    wall-clock, robustness (multi-start success rate), final error.
Q4. Hybrid: relaxed-tau for the FIRST epochs (global navigation), hard
    classification + continuation for the endgame (accuracy)?

## 4. Starting points in the codebase

- Classification: `diffsim.sbm.surrogate.classify_lambda` (the lambda
  rule is already a threshold PARAMETER — requirement 4 of the
  adjoint-readiness memo anticipated this study).
- The rung-2 gate config to reuse verbatim:
  `tests/test_adaptivity.py::test_rung2_beyond_trust_region`.
- Landscape probes: `scratchpad`-style alpha-ray scans, pattern in
  `benchmarks/data/landscape_4f.txt` generation.
- Adjoints for tau > 0: the volume kernels take per-GP coefficient
  fields (`kq`) — an ersatz-weighted volume term is a field-kappa
  problem, and `field_kappa_gradient` (gated 1e-6) is your gradient.

## 5. Deliverables (publishable unit)

1. The tau-convergence study (Q1) with the gradient-consistency plot.
2. The navigability-vs-accuracy trade curve (Q2).
3. The head-to-head table (Q3) on sphere + bunny configs.
4. A recommendation with measured evidence (Q4's hybrid is the
   expected winner — prove or refute it).

Rules of the road (inherited, load-bearing): non-dyadic feature
dimensions ONLY (findings 4b(h)); signal-scale check before any
optimization run (J0 vs the 4f floor); FD verification only along
branch-stable directions; report failures with mechanisms, not just
successes.
