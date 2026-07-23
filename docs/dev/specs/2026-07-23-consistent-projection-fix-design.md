# Consistent Projection Fix — design spec

**Goal.** Make the equal-order (P1-P1, PSPG) pressure-projection stepper FAITHFUL to the
monolithic same-mesh oracle on open-outflow external flow, by implementing the group's
own VMS-stabilized Helmholtz-Leray projection scheme (ns_projection_vms_paper, ref [26])
plus the standard outflow-consistency refinements. Gate: **rung A flips to a monolithic
match** (steady Re=40 Cd/mean|u|; Re=100 shedding mean Cd + Strouhal).

## Root cause (established by the validation ladder)
The split's fixed point ≠ the monolithic solution because of **two consistency gaps**:
(i) the PPE uses the bare FE Laplacian `K_p` instead of the PSPG-consistent pressure
operator (measured ‖L−K_p‖/‖K_p‖=0.67); (ii) the outflow enforces a frozen `∂p/∂n=0`
that contradicts the momentum natural traction (GMS 2006). Both appear only with an OPEN
outflow (the closed cavity, rung 0, passes). Refs: `docs/dev/2026-07-23-projection-ladder-verdict.md`,
`docs/dev/2026-07-23-projection-sbm-weak-fixed-point-verdict.md`.

## Authoritative derivation sources
- `local_code_old/ns_projection_vms_paper (1).pdf` — Khara/Murugaiyan/Khanwale/BG,
  VMS-stabilized Helmholtz-Leray projection. **Eq 44b (VMS PPE), 45–49 (split≡monolithic
  operator), 44c/Algorithm 1 (correction), 67d–67e + Fig 8 / Tables 2–3 (P1/P1 cylinder,
  open outflow, VALIDATED — our exact failing case, working there).**
- `local_code_old/main.pdf` — Suresh projection+VMS: §3 PPE weak form with `(τᵘRᵘ,∇q)`
  fine-scale (solves for the pressure CORRECTION p′, so pressure BCs apply to p′), §4
  correction, §5 ν-loop flowchart.
- `local_code_old/Suresh_Pressure_Projection_Octree_SBM_...pdf` — Remark 3.9 (SBM
  surrogate: Neumann ∇p′·n̂=0; outflow: Dirichlet p′=0). TODO-draft, defers to [26].

## The 7 changes (all on the projection stepper; the monolithic is unchanged)

**Baskar's BC rule (authoritative):** the two outflow BCs are DISJOINT by sub-solve —
velocity `∇u·n=0` in the **predictor only** (p is known data there, no pressure BC);
`p′=0` Dirichlet on the pressure **correction** in the **PPE only**. Solve for `p′`,
then `p = p* + p′`.

1. **PSPG-consistent PPE operator.** Replace bare `K_p=∫∇q·∇p` with the VMS/PSPG
   pressure-Poisson: add `−σΣ_K(τ_m r̃_m^h,∇q)_K` (ns_projection Eq 44b; main.pdf §3
   `(τᵘRᵘ,∇q)`). This makes the split's discrete incompressibility identical to the
   monolithic PSPG block (Eq 47–49) ⇒ the monolithic steady state becomes a fixed point.
   Verification: the "seed exact monolithic (u,p), one step" test must now PRESERVE it
   (‖div‖ stays ≈ its seeded value instead of rising 1.22→6.0).
2. **Fine-scale `ũ'=−τ_m r̃_m` in BOTH the PPE source and the L² correction**, with the
   CONSISTENT mass matrix (D=Gᵀ), so the divergence measured in the PPE and the field
   produced by the correction use the same operators (Eq 44b/44c, Algorithm 1 Steps 2–3).
3. **Disjoint outflow BCs** (Baskar's rule / Poux et al. 2011 split-across-substeps):
   predictor — natural traction-free `∇u·n=0` on velocity only (the `(n·ν∇ũ,v)_Γ`
   boundary term present, no traction imposed); PPE — Dirichlet `p′=0` on ALL outflow
   nodes (increment form). Remove the current single-node absolute-`p` gauge on the open
   outflow.
4. **Rotational-incremental pressure update** `p = p* + φ − ν∇·ũ` (Timmermans 1996) —
   makes the wall pressure BC consistent (curl-curl form; Guermond-Shen O(Δt^{3/2})).
   Knob already exists (`pressure_update="rotational"`); wire it as the default for the
   consistent scheme. (Valid for constant ν — our case.)
5. **P1 boundary-vorticity stabilization** `δ(∇q×n, ν∇×u)_Γ` (Pacheco et al. 2021,
   nme.6615). For P1 the PSPG viscous residual `νΔu` vanishes elementwise and standard
   PSPG drops this boundary integral, fabricating a spurious `∂p/∂n≈0`; retaining it makes
   the monolithic PSPG and the split AGREE at the outflow. Apply consistently in the
   monolithic AND the projection so the same-mesh oracle stays exact.
6. **Backflow stabilization** `−βρ∫_{Γ_out}(u·n)₋(u·v)dΓ`, β∈[0.5,1] (Bazilevs 2009 /
   Moghadam 2011; = directional-do-nothing Braack-Mucha). Needed for the Re=100 shedding
   wake (reverse flow through the outlet injects energy → blow-up). Active only on
   backflow (`(u·n)₋=min(u·n,0)`), so it doesn't corrupt the shedding physics.
7. **From-rest startup.** Do NOT start `p⁰=0`: compute a consistent initial pressure
   (GMS: `∇²p(0)=∇·f(0)`, `∂p/∂n=(f(0)+νΔu₀)·n`) and ramp the inflow smoothly. Reduces
   the from-rest under-development the split showed.

## Implementation surface
`src/diffsim/steppers/leray.py` (PPE assembly + operator, correction, pressure update,
outflow BC, startup) and `leray_sbm.py` (SBM surrogate: Neumann p′ on the immersed
boundary per Remark 3.9). Existing knobs to reuse/extend: `ppe_fine_scale`,
`pressure_update` (rotational), `consistent_ppe`, `pressure_outflow_nodes`. The corrected
scheme is gated behind a single mode (e.g. `consistent_projection=True`) that turns on the
coherent set; defaults stay bit-for-bit. The monolithic gets the P1 boundary-vorticity
term (#5) so the same-mesh oracle remains exact.

## Validation (gate)
- **Rung A, Re=40 steady (PRIMARY):** projection Cd == monolithic same-mesh Cd within R0
  tol AND mean|u| → monolithic magnitude (~1.0, not the 5% weak plateau). Flip the rung-A
  characterization test to the pass-assertion.
- **Rung A, Re=100 shedding:** mean Cd over ≥1 period + Strouhal, projection vs monolithic
  (oracle) + directional literature; backflow (#6) keeps it stable.
- **Consistency probe:** the seed-monolithic-one-step test preserves the state (fixed
  point), and ‖L−K_p‖ gap closed by the PSPG operator.
- Then unblock rungs B (weak Nitsche) and C (SBM shift).

## Process
Subagent-driven on branch `projection-ladder`. Implementers read the papers above for the
exact discrete forms (τ_m definition, the elementwise residual r̃_m, the boundary terms).
Same-mesh monolithic oracle is the primary bar. Anti-vacuity: each term load-bearing
(turning it off must move the result). Bit-for-bit defaults preserved.
