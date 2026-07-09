# M1 Verification Coverage Audit — SBM1–SBM7 + Tier AD

**Date:** 2026-07-06. **Sources:** spec `docs/superpowers/specs/2026-07-02-diffsim-design.md` (§9.1 adopts the cuFEM plan wholesale; §13.3 batteries; §10 M1 gate row), tier definitions from `MyPapers/cuFEM_Verification_Plan_Revised.tex:495-590`, Tier AD from spec §9.1 line 365. Backends: **CSG** = AnalyticCSG, **TM** = TriMesh, **GS** = GridSDF, **INR** = NeuralSDF/ProvidedINR.

## Requirements (extracted)

- **SBM1** distance/closest-point correctness (analytic reference 1e-12; face/edge/vertex projection branches; accelerated-vs-brute-force nearest triangle).
- **SBM2** Interior/Intercepted/FalseIntercepted/Exterior λ-classification (count vs independent reference; λ=0 ⊂ Ω; λ=1 ⊃ Ω; λ=0.5 half-retention).
- **SBM3** surrogate-boundary validity (whole faces, no opposite-face cycles, single closed cycle).
- **SBM4** d_RMS(λ) minimized at λ=0.5, rotated squares, multiple θ/levels.
- **SBM5** disk MMS: order 2 (L2) for all λ ∈ {0, 0.25, 0.5, 0.75, 1}; I_2λ minimized at λ=0.5, ≤0.4.
- **SBM6** manufactured convergence on complex STL geometry (Bunny/Moai/Armadillo), full pipeline.
- **SBM7** pathological geometries: sub-h thin feature (resolve or fail gracefully), interior hole (both surrogate boundaries), sharp corner aligned vs 1° misaligned.
- **P4 keystone** (§9.1): SBM linear patch, rotated geometries, all λ, machine precision; plus §13.3.3 p2-band Neumann acceptance and the π/4 area-correction lock; two-sided Γ̃⁺/Γ̃⁻ shell classification appears in the carve pipeline (spec line 228).
- **Tier AD** (spec line 365): three-way gradient checks per Tier-2 VJP/brick (adjoint vs unrolled tape vs central FD, rel < 1e-6); ⟨Av,w⟩=⟨v,Aᵀw⟩ per operator pair; Nitsche adjoint-consistency check; epoch-crossing gradient continuity.

## Coverage table

| Req | Tests (file:line) | Backends | Verdict |
|---|---|---|---|
| SBM1 | `test_geometry_oracles.py:103,115,126,225`; `test_backends.py:33` (GS), `:61,85` (TM); `test_inr_proxy.py:35`; `test_provided_inr.py:49` | CSG, TM, GS, INR | **partial** — no minimal-triangle face/edge/vertex branch unit tests; no BVH-vs-brute-force cross-check |
| SBM2 | `test_o_build.py:38` (O8 count vs independent classification, 1%); `test_sbm_surrogate.py:24` (ordering, λ=0 fully-interior), `:135,150,162` (volume-fraction accuracy, narrowband exactness, k=4) | CSG (TM/GS/INR carve exercised via SBM6-row solves) | **covered** (λ=1 superset / half-count asserted only via monotone ordering — acceptable) |
| SBM3 | `test_sbm_surrogate.py:33` (GPs near Γ), `:73` (partially-exposed face raises), `:92` (fine-vs-coarse legality), `:162` at 181–187 (vector-area closure ≡ watertight, exact) | CSG | **partial** — no single-cycle / opposite-face cycle-rejection test |
| SBM4 | — (`test_sbm_surrogate.py:24` retention ordering only) | — | **GAP** |
| SBM5 | `test_sbm_poisson.py:173` (λ=0, order 2), `:180` (λ=0.5), `:189` (p2 order 3), `:201` (exterior), `:211` (3D) | CSG | **partial** — λ ∈ {0, 0.5} only; no λ=0.25/0.75/1 ladder; no I_2λ optimality lock |
| SBM6 | `test_backends.py:147` (same solve through CSG/TM/GS vs CSG baseline); `test_provided_inr.py:69,195` (INR sphere forward; Bunny admissibility+carve) | CSG, TM, GS, INR | **partial** — cross-backend geometry is a sphere; Bunny is carve-only, no convergence ladder on a genuinely complex STL/INR |
| SBM7 | — (nearest: `test_sbm_surrogate.py:73` graceful-failure diagnostic) | — | **GAP** |
| P4 keystone | `test_sbm_poisson.py:58,63,69,73,77,83,89,95,126`; vector: `test_sbm_vector.py:32`; Neumann patch `test_sbm_neumann.py:102,119` | CSG (k=2,3,4; rotated boxes; exterior; α-insensitivity) | **covered** (approximate-geometry patch bound folded into `test_backends.py:158` tolerance instead of the planned 50·ε̂∞ patch) |
| π/4 area correction | `test_sbm_surrogate.py:57` (geometric lock); `test_sbm_neumann.py:143` (solve-level convergence-vs-stall lock) | CSG | **covered** |
| §13.3.3 p2-band Neumann | `test_sbm_neumann.py:193` (p1 degrades, band restores order 2), `:215` (layer sweep + hard rule GP∈p2); batteries `test_verification_batteries.py:65,73,81` | CSG | **covered** |
| Two-sided Γ̃⁺/Γ̃⁻ shells | — (no implementation hits in `src/`) | — | **GAP** (spec line 228 pipeline item; ThinShell brick is post-M1 per §10 — deferred, not M1-blocking) |
| AD: three-way | `test_ad_gradients.py:89` (shape), `:120` (κ); projection VJP `test_geometry_oracles.py:144,197`; `test_inr_proxy.py:115,169`; `test_provided_inr.py:114`; NS: `test_ns_adjoint.py:63,152`, `test_leray_adjoint.py:44`, `test_transient_adjoint.py:44,62,110`, `test_ns_shape_gradient.py:107`, `test_neumann_shape_gradient.py:62,85` | CSG (full three-way); GS `test_ad_gradients.py:138`, TM `:177` (adjoint-vs-FD two-way); INR three-way | **covered** — GS/TM legs lack the torch-twin leg (spec asks three-way per VJP/brick, satisfied on CSG/INR) |
| AD: dot-product per operator pair | `test_ad_gradients.py:69` (Poisson A only) | CSG | **partial** — NS/Leray operators rely on tape-vs-kernel checks (`test_ns_adjoint.py:116,196,271`), no ⟨Av,w⟩ identities |
| AD: Nitsche adjoint-consistency | `test_sbm_poisson.py:102` (documents nonsymmetry only) | CSG | **GAP** (indirectly guarded by three-way gates) |
| AD: epoch-crossing continuity | `test_ad_gradients.py:78` (frozen-classification trust region); `:225` (re-carve inverse loop) | CSG | **partial** — remesh+conservative-transfer continuity is M4 machinery; M1 analogue covered |

## Gap list — suggested minimal tests

1. **SBM4:** tier-4 geometry-only sweep — rotated square θ ∈ {10°,…,40°}, λ ∈ {0,…,1}, levels 4–6; assert argmin d_RMS = 0.5. Cheap (no solves).
2. **SBM5 completion:** extend `_mms_ladder` to λ ∈ {0.25, 0.75, 1}; lock I_2λ(0.5) minimal and ≤ 0.4 at the finest level.
3. **SBM6 completion:** one 2-level MMS ladder on a nonconvex TriMesh (e.g. two blended spheres STL, or the Bunny window) asserting order ~2; promotes the Bunny from carve-only.
4. **SBM7:** (a) thin plate < h → expect the `partially exposed`/diagnostic path, not corruption; (b) annulus oracle → assert two disjoint surrogate face sets, patch passes; (c) square corner 0° vs 1° → bounded L2 error ratio.
5. **SBM1 branches:** unit test one triangle with query points hitting face/edge/vertex projections; BVH vs O(N) brute force on a random 1k-triangle mesh.
6. **AD:** dot-product identity on the assembled NS Jacobian and Leray PPE operator; a dedicated Nitsche adjoint-consistency test (adjoint solve of the SBM form converges at primal order on the disk MMS).
7. **Two-sided shells:** structural-only extraction test (sign of mean n·ñ splits Γ̃⁺/Γ̃⁻ on a plate) — or record explicitly as post-M1 in the M1 gate review.

**Bottom line:** the M1 additions (P4 keystone, area correction, p2-band, three-way gradients, four backends at SBM1/SBM6 level) are well covered; the inherited cuFEM tiers SBM4 and SBM7 are unimplemented, and SBM5's λ sweep is truncated — none are hard to close with the suggested tests.
