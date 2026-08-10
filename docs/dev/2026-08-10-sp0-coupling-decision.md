# SP-0 Coupling Decision Memo — Monolithic vs Staggered CHNS

**Date:** 2026-08-10
**Status:** Evidence complete — awaiting Baskar's decision (Task 8 human gate)
**Decision:** MONOLITHIC PRIMARY (Baskar, 2026-08-10) — staggered retained as documented forward-only fast mode. Quartic-bulk ruling ratified.

## What was built (both reviewed and approved on `feat/sp0-chns`)

- **Staggered** (`CHNSStaggeredStepper`, c2e07fc): per step, CH solve (MultiPhaseStepper, new `bulk="quartic"` mode, `adv_gp`=uⁿ) → GP coefficients (mix_props ρ/η, capillary μ∇φ, gravity) → variable-coefficient NS predictor + PPE + velocity update (Task 3 assembly, Leray-v1 pressure convention).
- **Monolithic** (`CHNSMonolithicStepper` + `make_chns_newton`, 64d35b4): one Newton per step on the coupled (u,p,φ,μ) system, exact Jacobian including differentiated τ; **machine-precision parity vs the numpy mirror** (φ 3.3e-16, u 1.3e-14, p 5.7e-16) and exactly-zero mass drift (conservative CH advection).

Mid-build ruling to surface at the gate: MultiPhaseStepper lacked the [−1,1] quartic double-well, so a static-gated `bulk="quartic"` kernel mode was added (option 1 of 3) to keep both prototypes on identical CH physics and the same Warp tier. Veto point: this mode is also the natural host for SP-1 deposition and Task 10 CAC work.

Known twin differences (documented, relevant when reading the evidence): staggered NS viscous term is the Task 3 kernel's grad-grad form vs the mirror/monolithic symmetric D:D; staggered CH advection is convective-form (Task 2 term) vs monolithic conservative-form — hence staggered's nonzero mass drift is *structural split evidence, not a bug*.

## Axis 1 — Forward robustness (measured, level-6 64², bubble rise; commit 0b0e616)

| stepper | ρ-ratio | wall/step | Newton mean/max | max mass drift | survived | centroid agreement |
|---|---|---|---|---|---|---|
| staggered | 10 | 0.430 s (2400/2400 steps) | 4.00/4 | 5.0e-6 | yes | max Δcy 1.2e-3 |
| monolithic | 10 | 2.055 s (496 steps, 25-min cap) | 3.01/4 | 1.1e-15 | yes | (same window) |
| staggered | 100 | 0.432 s (2400/2400) | 4.00/4 | 5.4e-6 | yes | max Δcy 1.4e-3 |
| monolithic | 100 | 2.054 s (499, capped) | 3.01/4 | 8.9e-16 | yes | (same window) |
| staggered | 1000 | 0.436 s (50-step probe) | 4.00/4 | 4.2e-6 | yes | max Δcy 5.2e-5 |
| monolithic | 1000 | 2.136 s (50-step probe) | 3.08/4 | 7.8e-16 | yes | (same window) |

Readings:
- **Conservation:** monolithic is machine-exact (≤1e-15); staggered's drift is secular, ~2e-9/step — over an AM-scale march (10⁴–10⁵ steps) that extrapolates to 1e-4–1e-3 of phase mass, material when deposited mass is the quantity of interest.
- **Cost:** staggered ~4.8× faster per step on this CPU/splu prototype tier. The gap is dominated by the monolithic (dim+3)n splu factorization per Newton iterate — precisely the piece device assembly + a blockch-family preconditioner would attack.
- **Newton behavior:** monolithic converges in fewer iterations (3.01 vs 4.00 mean) — the exact coupled Jacobian pays off.
- **Stress at 10³:** both survive the probe; no clamps, no Newton failures anywhere in the resolvable-IC matrix.
- **Guard finding (first-pass matrix, archived `results/spike_underresolved_ic/`):** with the case's own Cn=0.01 (under-resolved at h=1/64), the monolithic clamp-guard *correctly killed* the run at r100/r1000 while staggered silently tolerated the under-resolution. Fail-loud vs fail-silent: an argument *for* the monolithic guard discipline, and a flag that Task 9 benchmarks must resolve Cn ≳ h. Trajectories at this resolution are relative evidence only (unit-square domain, thick interface); reference-quality runs are Task 9's job.

## Axis 2 — Adjoint tractability

**Monolithic — concrete and low-risk.** The per-step discrete-IFT adjoint is exactly the proven crystallization pattern: solve Jₙᵀλₙ = ∂j/∂xₙ − (∂Rₙ₊₁/∂xₙ)ᵀλₙ₊₁ backward using the *same converged Jacobian the forward already assembles* — and that Jacobian is exact (differentiated τ, clamp-Jacobian convention) with machine-verified parity, which is precisely what makes JᵀΛ trustworthy. The cross-step block ∂Rₙ₊₁/∂xₙ comes only from BDF1 history (u,φ rows: −ρⁿ⁺¹M/dt-type mass terms, mirrored in SUPG); parameter cotangents ∂R/∂p for (ρ-ratio, η-ratio, We, mobility, Fr) are single-term re-assemblies. New work: one adjoint class + torch-twin step; no new mathematical seams.

**Staggered — concrete but three seams, high surface area.** The reverse chain per step transposes three sub-solves plus two hand-offs: (i) CH Newton adjoint — exists for K=0 in `adjoint/multiphase.py`, but the quartic bulk needs new `dmu_dparam` support and the `adv_gp` term adds a (φ-row → uⁿ) cross-step cotangent; (ii) NS predictor transpose — needs new VJPs of the assembled operator w.r.t. (φ, μ) through mix_props/capillary/τ at Gauss points; (iii) PPE + velocity update — `leray_adjoint.py` covers the PPE core, but the split as composed here (Leray-v1 pinned p*) is untaped. Each coefficient hand-off (uⁿ→adv_gp; φ,μ→ρ/η/f at GPs) is a bespoke VJP. All doable; roughly 3× the seam count of the monolithic path, on less-proven ground.

## Axis 3 — GPU path

**Monolithic.** DOF layout is node-major (dim+3) — structurally identical to what `assembly="device"` (slot-map CSR, uniform meshes) + cuDSS handle today for the (2M+2K) CH system; the SP-0 3-D gate (dam break, level ≤5, <1M DOF) fits inside the proven cuDSS envelope. Beyond that, the `blockch` G4 two-factor Schur is per-(φ,μ)-pair; the natural CHNS extension is a 2×2 outer split — field block [u,φ,μ] (CH-pair preconditioner + velocity block) with a pressure Schur complement that is PPE-like/SPD (existing `gpu_cg` applies). Plausible, deferred engineering; documented fallback per the spec risk register: cuDSS at reduced size, preconditioner work becomes SP-1-adjacent.

**Staggered.** Best story on paper today: every sub-solve maps to an existing proven device solver (CH: cuDSS/`blockch_dev` at 10M+ DOF; PPE: SPD `gpu_cg`; predictor: cuDSS). The real cost is residency: per-step GP-coefficient hand-offs between sub-solves are currently host-side numpy and would need device-resident plumbing to be competitive at scale.

## Recommendation

**Monolithic as primary (adjoint-bearing) mode; staggered retained as a documented forward-only fast mode.**

The case: (1) the adjoint — SP-0's defining deliverable — is one proven low-risk pattern on the monolithic path vs three bespoke seams on the split, and the machine-verified exact Jacobian is already in hand; (2) machine-exact mass conservation vs secular drift matters for AM marches where deposited mass is the observable; (3) fewer Newton iterations and fail-loud guard discipline. The 4.8× per-step cost gap is real but is a solvable engineering property of the prototype tier (splu per iterate), with a concrete attack path (device assembly + 2×2 field/pressure Schur extension of blockch), whereas the split's adjoint seams and conservation gap are structural. Keeping staggered (already built, reviewed, 4.8× faster) gives the research doc's anticipated fast mode for quasi-steady/forward-only sweeps at zero extra cost.

Per the plan, Tasks 9–13 proceed on this choice once Baskar confirms: Task 9 builds the `CHNSStepper` facade around the monolithic mode; Task 11's adjoint plan applies as written (no re-plan needed).
