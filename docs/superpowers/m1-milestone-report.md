# M1 Milestone Report — Differentiable Octree-SBM NS with Neural-SDF Geometry

Date: 2026-07-06 (night). Status: **COMPLETE** under the amended
contract (rulings 2026-07-06: p2-NS deferred to M2; hero bar =
converged H1-sphere; H2-H4 slot into M2).

## 1. Contract vs delivered

| Contract item | Delivered | Evidence |
|---|---|---|
| Octree-SBM incompressible NS, 3-D, static complex geometry | YES | M1b: cavity Re100/1000, cylinder Re20 (Cd 1.352 vs lit 1.33), Re100 Strouhal, sphere Re300 smoke; immersed-cylinder SBM+NS composition |
| VMS-stabilized | YES | s=1/2 skew forms, tau per production conventions; timestab sig2tau |
| p1 + p2 bases | AMENDED | p1 NS + p2 Poisson/band (validated: band study, orders ~2). p2-NS -> M2 |
| Both steppers (Leray + monolithic) | YES | S17 both-steppers gate on the cavity; Leray-vs-monolithic locked |
| BDF1/BDF2 | YES | order gates in the transient tier |
| Adjoint w.r.t. constitutive (nu) | YES | ns_volume/load_cotangents; transient chain 4.4e-10; Leray adjoint (predictor + finescale, gates green); dJ/dnu at hero config in H1 deliverables |
| Adjoint w.r.t. geometry | YES | steady drag shape gradient (adjoint-Picard, 1.25e-4); TransientShapeAdjoint (1.1e-8); alpha-gradients through GENIE modes |
| Neural-SDF geometry backend | YES | ProvidedINROracle: real checkpoints (sphere .pt w0=1.0; bunny two-head GENIE json), SIREN conventions, window contract, GENIE edit modes (displacement-normalized, orthonormal V + scales) |
| All four backends through the oracle suite | YES | CSG (three-way AD), TriMesh (vertices AD), GridSDF (voxels AD), INR (alpha gates); cross-backend SBM solve |
| Hero demo on a neural SDF | **YES — CONVERGED** | H1: hidden GENIE edit of the sphere INR recovered from steady 3-D flow probes, err 1.769e-3 < 2e-3 bar (tests/baselines/m1_hero_h1.json). BONUS: H2 transient recovery 2.56e-4 (m2_hero_h2.json) |
| Single GPU | YES | plus M1d (in progress): device assembly 4-40x, zero-copy solve chain, cuDSS 300x over host splu |

## 2. The hero results

- **H1 (M1 bar)**: sphere INR, steady sigma=0 Picard NS, 16-probe
  plane, k=4 displacement-normalized Gram modes, frozen epoch +
  anchored feet + marginal-GP borrow, Gauss-Newton: err 2.31e-2 ->
  1.77e-3 in 8 epochs; all components correct sign+magnitude;
  adjoint-Picard consistency cos up to 0.995.
- **H2 (M2 opener, bonus)**: same geometry, TRANSIENT (8 BDF2 steps):
  recovered to 2.56e-4; adjoint-vs-Jacobian cos 0.9994-1.0000. The
  transient inverse problem converges where the steady POISSON analogue
  plateaus (findings 4f) — advective measurement enrichment,
  demonstrated end-to-end.

## 3. Verification coverage

docs/superpowers/m1-verification-coverage.md: P4 keystone (rotated, all
lambda, k=2/3/4, exterior, vector), pi/4 area-correction locks, p2-band
acceptance (13.3.3), AD three-way on CSG+INR, Galerkin-duality Nitsche
lock, SBM4 lambda-optimality MEASURED (d_RMS halves at lambda=0.5),
SBM7b pathological classification. Remaining recorded gaps: SBM5
I_2lambda optimality lock; SBM6 complex-geometry convergence ladder
(bunny is admissibility/carve/modes + hero-flow only); two-sided shells
(post-M1 by design).

## 4. Notable science en route (findings index)

- 4c/4e: taped-kernel rules (NO struct mutation — silent zeros; warp
  issue paste-ready), residual-form compile fix (79 min -> 107 s)
- 4d: tau-frozen adjoints leak in transient chains (3.8e-4/link)
- 4f: SBM-probe objectives over wrinkly INRs have steep sub-edit-scale
  structure (open; landscape dataset in progress); foot continuation,
  epoch trust regions (drift > 0.35h re-carve), displacement-unit modes
- 4b(h): the 3-D "L5 dip" = benchmark degeneracy (dyadic radius r=0.25
  aligns tangent planes with mesh faces at every level; intrinsically
  3-D by contact dimension; 2-D control identical at both radii).
  Corrected ladder r=0.27 MONOTONE (1.78/1.57, L7 = asymptotic confirm)
- 8f-i: task #6 resolved — exact-F block preconditioner = 2-3 FGMRES
  its at sigma=0 (solver="blocktri"); the AMG F-cycle was the culprit
- 8j: mixed-precision fp32+IR — 2-8 refinements, structure-bound at
  prototype scale (Blackwell-relevant only at FLOP-bound factors)
- M1d D5: pure-device is the profile (cuFEM adaptivity is device-native
  per Baskar); coherent = capacity question (Nova GH200 measures it)

## 5. Open items (Baskar)

1. License + CITATION choice (evaluation finding; blocks public
   release, nothing else).
2. Nova allocation submissions (kits ready: cluster/; partitions need
   sinfo).
3. NVIDIA/warp issue filing (docs/superpowers/warp_issue_draft.md).
4. M3 placement ratification (differentiable adaptivity; task #15 memo).
5. w0=1.0 sphere-checkpoint convention confirmation (sentinel-guarded).

## 6. What M2 inherits

Bunny hero ladder (H3 running tonight: wake-probe configuration; H4
next), p2-NS, Heat/Mass bricks + closures, the 4f landscape question
(with tonight's dataset), SBM5/SBM6 gap tests, M1d finishing touches
(GP-field kernels; coherent-profile verdict from Nova).

## Decisions taken (2026-07-07 morning)

1. M3 RATIFIED (M2 -> M3 -> ... -> THB). 2. cuFEM memo held for
Baskar's review (spec file, second half). 3. License: Apache-2.0 +
CITATION.cff (added). 4. Nova: all three cards to be requested
(kits ready; partitions need sinfo on arrival).
