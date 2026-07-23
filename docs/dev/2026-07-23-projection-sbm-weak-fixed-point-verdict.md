# Projection+SBM: why the split under-develops the 3-D flow — root verdict

**Date:** 2026-07-23
**Status:** root cause identified & quantified; flow-development remains an open
formulation decision (two candidate paths). Validates the R2a monolithic pivot.

This closes the diagnostic phase of the projection+SBM research track opened by
R2a (`2026-07-22-p2-r2a-projection-3d-findings.md`). Five focused experiments
(sphere, L4, Re=100; monolithic same-mesh oracle = mean|u|≈1.13, Cd≈+1.06)
took the problem from "diverges, cause unknown" to a named structural mechanism.

## The diagnostic arc

1. **No-penetration term** — the shifted SBM no-penetration coupling must be
   *extracted* into the PPE (`−σ⟨q, n̄·(S(ũ)−u_D)⟩`, surrogate normal). Verified
   correct (reconciliation contract ~1e-17) but **not sufficient** — flow still weak.
2. **ν-loop (inner_iterate)** — the within-step predictor↔PPE fixed-point iteration
   is an **unstable accelerant**: its contraction factor climbs 0.25→0.30→0.85→>1
   as the flow develops; at step 5 ρ>1 → blow-up. Single-pass is stable but weak.
3. **Viscous / stress form** — REFUTED as a cause. The monolithic and the
   projection predictor use the **identical** viscous discretization (Laplacian
   `ν∫∇w:∇u` + component-diagonal Laplacian SBM Nitsche). A symmetric-gradient
   (`2νε:ε` + `∇uᵀ` stress-transpose) knob was implemented and tested: moves the
   flow ~5% and *downward*. Exonerated.
4. **Pressure instability → SOLVED.** Root cause of unbounded ‖p‖: the PPE uses the
   direct FE Laplacian `K_p`, but a discretely divergence-free correction needs
   `L = GᵀM⁻¹G` (they differ 128%; shipped predict-correct kills only ~60% of a
   planted divergence). The SBM boundary flux is exonerated (a consistent operator
   kills the residual to 1e-10 regardless of the SBM block). The *textbook* operator
   fix (`GᵀM⁻¹G`) diverges faster live (idempotency ≠ coupled-loop stability); the
   **practical stabilizer is the PSPG fine-scale in the PPE source** (`ppe_fine_scale`),
   which IS the consistent equal-order pressure stabilization: ‖p‖ 1792→108, and
   `rotational + fine_scale + outflow-gauge` → ‖p‖≈17, |u.n|≈0.012.
5. **Flow under-development → the root defect (this verdict).**

## The named mechanism (experiment 5)

With pressure stability solved, the flow still plateaus at mean|u|≈0.1 vs the
monolithic 1.13. Localization:

| mean\|u\| | step 1 | step 5 | monolithic |
|---|---|---|---|
| predictor u_hat (pre-correction) | 0.0174 (2.0%) | 0.0316 (2.8%) | 0.867 / 1.115 |
| corrected uⁿ⁺¹ | 0.0085 | 0.0840 | — |

- **The momentum is lost in the PREDICTOR** — u_hat is ~2% of monolithic and never
  exceeds ~2.9%. The correction is not over-damping (uⁿ⁺¹ ≥ u_hat from step 2).
- **Not any single term:** SBM penalty 10→100→1000 moves mean|u| <1%; ‖p*‖ builds
  to ≈17 (≈ monolithic ‖p‖=21) yet u_hat stays pinned.
- **Decisive test:** seeding p* with the *true monolithic pressure* triples step-1
  u_hat (0.017→0.055, Cd→+0.21) — the pressure gradient genuinely drives the flow —
  **but the within-step pressure update overwrites the seed with the pressure
  consistent with the weak u_hat, and it re-converges to the same weak fixed point.**

**Mechanism:** the lagged-pressure split cannot self-consistently build the driving
(stagnation) pressure **from rest**. The monolithic couples continuity into the same
solve, so from rest (a=0) incompressibility instantly propagates the inflow through
the domain (a Stokes solve → mean|u|=0.87 in step 1). The split predictor pins the
pressure DOFs to the *known* p* (=0 at step 1) → a pure advection–diffusion solve
with no incompressibility coupling → the inflow only diffuses into a thin boundary
layer (0.017). The PPE then projects, but projection can only *remove* divergence
from the weak predicted field — it cannot *manufacture* bulk momentum. Each step the
PPE rebuilds only the pressure consistent with the current weak u_hat, so p* and u*
co-plateau at ~3%/8% of monolithic. **The plain lagged-p* single-pass split is
structurally incapable of reaching the strong branch on this external flow.**

## Consequences

- **Validates the R2a monolithic pivot** at a fundamental level: the monolithic gets
  the from-rest Stokes propagation "for free" via coupled continuity; the split does
  not. R2's 3-D engine correctly stays monolithic.
- **The open formulation decision (two paths) for the scalable projection engine:**
  - **(a) Stabilized inner predictor↔PPE iteration.** Iterate to a within-step
    fixed point (update p*, re-predict) — the intended vehicle — but the naive ν-loop
    is unstable (exp. 2). Needs damping / Anderson acceleration / a contraction-
    guaranteed relaxation, not abandonment. Preserves the SPD-PPE 100M-scalability path.
  - **(b) Pressure-coupled predictor.** Solve the momentum predictor with the
    continuity/PSPG pressure block live (Uzawa / segregated-but-coupled), so the
    from-rest Stokes propagation is present in the first solve. More robust; partially
    reintroduces coupling (weakens the pure-projection scalability argument).

## Task-4 addendum (2026-07-23, ladder Rung A) — the fix was implemented; the
## split is OPERATOR-INCONSISTENT (Lane 1c); neither path (a) nor a consistent
## operator flips Rung A.

Rung A (body-fitted square, STRONG Dirichlet, `dmax==0`, OPEN outflow, Re=40,
level 5) is the clean 2-D restatement of the sphere defect: same-mesh monolithic
oracle Cd = **+4.18**, mean|u| = **1.04**, ‖div‖ = 1.2 (steady, physical). The
stabilized inner predictor<->PPE iteration (path (a)) and the consistent PPE
operator were both implemented in `leray.py` (default-OFF, bit-for-bit) and run
head-to-head:

- **Inner iteration (`inner_iterate` + `inner_relax` ω + `inner_accel="anderson"`
  + divergence guard).** The within-step predictor<->PPE map is NON-CONTRACTIVE
  on this open-outflow external flow: the inner residual ‖p_hat − p*‖ never
  converges at any ω (1.0 / 0.3 / 0.1) or with Anderson — it plateaus/grows
  (~2 → 130). It DOES break the from-rest weak pin (mean|u| develops off the ~5%
  plateau) but Cd runs strongly NEGATIVE (−9 … −63) and ‖div‖ grows (30 → 68).
- **Consistent PPE `L = GᵀM⁻¹G` (`consistent_ppe`).** DECISIVE seeded-fixed-point
  diagnostic: seed the projection with the EXACT monolithic (u, p) and take ONE
  step. The predictor REPRODUCES it (‖u_hat − u_mono‖ = 3.6e-6, ‖div‖ = 1.22 —
  faithful). The PPE+correction then DESTROYS it: with the shipped `K_p`, Cd
  +4.18 → **−55** and ‖div‖ **1.22 → 6.0** (the "projection" RAISES divergence).
  Root: `K_p` ≠ `L`; measured ‖L − K_p‖/‖K_p‖ = **0.67** on this mesh. With the
  consistent `L`, a STATIC seeded projection is idempotent (‖Bᵀu_corr‖ 0.175 →
  8e-15), but LIVE it diverges FASTER (mean|u| overshoots to 2.5, Cd → −19),
  even paired with the damped/Anderson inner iteration (Cd → −7.6, mean|u| 1.66,
  ‖div‖ 54) — **idempotency ≠ coupled-loop stability** (confirms exp. 4).

**Verdict:** the monolithic steady state is NOT a fixed point of the lagged-p*
split on Rung A; the split is operator-inconsistent. Neither the stabilized
inner iteration nor the consistent PPE operator (alone or together) makes the
projection faithful to the same-mesh monolithic. This is the reportable Lane-1c
result and validates the R2a monolithic pivot at the 2-D body-fitted level. The
knobs ship default-OFF; the base single-pass Rung-A regression is unchanged.
Experiments: `tests/rungA_inner_experiment.py`, `tests/rungA_outflow_diag.py`;
regression `tests/test_ladder_rungA.py`; unit gates `tests/test_leray_inner_stab.py`.

## Artifacts
- Knobs + probes on branch `projection-sbm` (unmerged, default-off, tests green):
  `viscous_form` (`1b418ec`), `consistent_ppe` + Leray/idempotency probes (`3f836d4`),
  `sbm_no_penetration`, `inner_iterate`, `pressure_update`, `ppe_fine_scale`,
  `pressure_outflow_nodes`. Kept as reproducible measured proofs.
- Ladder branch `projection-ladder` (Task 4): `inner_iterate`/`inner_relax`/
  `inner_accel`/divergence-guard + `consistent_ppe` re-implemented on the CURRENT
  base stepper with unit gates and the Rung-A head-to-head (this addendum).
- Predecessor: `2026-07-22-p2-r2a-projection-3d-findings.md`; spec
  `docs/dev/specs/2026-07-22-projection-sbm-formulation-research-design.md`.
