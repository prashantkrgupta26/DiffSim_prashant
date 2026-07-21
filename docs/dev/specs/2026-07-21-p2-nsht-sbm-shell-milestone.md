# P2 — NSHT-SBM-Shell hero: milestone design (umbrella)

**Milestone design spec.** P2 of the reconciled roadmap
(`docs/dev/2026-07-20-roadmap-reconciliation-strawman.md`). Ratified +
brainstormed 2026-07-20/21. This is the **umbrella** spec: it fixes the
architecture, deliverables, rung ladder, and reference anchors for the
whole milestone. Each rung (starting P2-R0) gets its own implementation
spec + plan as we reach it.

## 1. Goal

A GPU-native, octree-SBM Navier–Stokes capability for **flow over
slender / thin-shell structures**, scaling to **100M+ dofs**, with a
**steady-state differentiable** path for design. Two scientific targets:
external-aerodynamic transient flows (truck + boat-tails) and **maize
canopy ideotype design** (steady mean-flow sensitivities to plant
architecture).

## 2. Deliverables

The forward solver runs in two modes (transient march / steady via
pseudo-transient continuation); the design gradient adds a steady adjoint.

| | Deliverable | Solve | Differentiable | Purpose / validation |
|---|---|---|---|---|
| **A** | 2-D differentiable framework | steady + transient | ✓ both adjoint modes | prove the adjoint machinery cheaply; gradient-gated |
| **B1** | Truck + boat-tails hero | **transient** 3-D | ✗ | external-aero drag; transient shell + carve-out compose (ThinShell Fig 29) |
| **B2** | Maize ideotype dataset | **steady** 3-D, 100M+ | ✗ | steady mean-flow data generation |
| **C** | Maize ideotype design engine | **steady** 3-D, 100M+ | ✓ steady adjoint | design gradients ∂(objective)/∂(geometry); **shares B2's forward solve** + one transposed solve |

**The steady-adjoint insight (why C is tractable at 100M):** a transient
adjoint would need O(steps) full-state checkpoints — infeasible at 100M.
A **steady** adjoint has no trajectory: at the converged fixed point the
gradient is `dJ/dp = ∂J/∂p − λᵀ ∂R/∂p` with a single transposed solve
`Aᵀλ = ∂J/∂u` (A = the converged Jacobian already in hand). Memory =
one state + one matrix + one solve. So differentiability at hero scale
costs ~one extra solve, not a trajectory. C is feasible precisely because
the maize *design* objective (mean drag / gas-exchange) is a steady /
time-averaged quantity (confirmed with Baskar 2026-07-21).

## 3. Reference anchors (parity targets)

All group-authored (in `local_code_old/`), giving strong parity anchors:

- **VMS-projection** — *"A Helmholtz-Leray projection method with VMS
  stabilization for the NSE"* (Khara, Murugaiyan, Khanwale,
  Ganapathysubramanian), `ns_projection_vms_paper.pdf`. **Defines the
  forward stepper.**
- **Solution-transfer** — *"A Pressure-Robust Solution-Transfer Strategy
  for Incompressible Flow on Dynamically Adapted Octree Meshes with SBM"*
  (Murugaiyan, Karki, Ganapathysubramanian),
  `Suresh_Pressure_Projection_Octree_SBM_Moving_Rigid_Body.pdf`. **Defines
  the AMR / moving-body transfer.**
- **ThinShell** — *"Flow Simulations Around Thin Shell Structures via
  Octree-based SBM"* (Chen/Hua et al.), `ThinShell.pdf`. **Defines the
  co-dim-1 two-sided shell surrogate; supplies the truck (Fig 28–29),
  blocked-channel, and 2-D thin-plate Re126/250 vs carved-out benchmarks.**
- **C++/MPI reference code** — `chenghauy-nshtsbm_shell-*.zip` (Dendrite-kt
  / PETSc), the executable parity oracle for the shell + coupling.
- **M1b (in-repo)** — cylinder Re100 (Cd=1.352), sphere Re300 volumetric
  SBM NS, the R0 stepper-validation anchor.

## 4. The forward stepper (VMS-incremental-projection)

Per the VMS-projection paper: a VMS-stabilized *incremental* Helmholtz-
Leray projection, **equal-order velocity–pressure** (fine scales supply
both advective and pressure stability → no inf-sup constraint, fits the
octree/node-major machinery natively). Per time step (BDF-r, r∈{1,2}):

1. **VMS momentum predictor** — solve for the predicted velocity `ũⁿ`
   using the incremental pressure `pⁿ⁻¹`; VMS fine-scale model (residual-
   based, the Bazilevs metric-tensor τ_M already in `vms.py`) supplies
   advective stability.
2. **Pressure-Poisson (the Leray projection)** — SPD elliptic solve for
   the pressure increment enforcing discrete solenoidality. **SPD → AMG /
   AMGX scalable to 100M** (the reason projection was chosen over the
   monolithic saddle).
3. **Velocity correction** — project `ũⁿ` onto the divergence-free space.

Incremental (previous-step pressure in the predictor) → O(Δt²) pressure
accuracy. BDF1/BDF2 via the existing cross-matrix history rule.

**Reuse:** the linearized RBVMS-NS brick (`api/ns_bricks.py`, `vms.py`)
supplies the predictor's assembly; the M1b Leray stepper is the structural
precedent; the SPD pressure-Poisson uses `solvers/` (fgmres + gamg/AMGX).

## 5. The co-dim-1 two-sided shell surrogate (staged)

Per ThinShell: a slender open surface has fluid on **both sides**. The
surrogate boundary is a **two-sided, non-overlapping decomposition** with
**distinct outward normals** and **two-sided traces** — enabling
well-defined pressure/traction **jumps** across the shell and preventing
cancellation of opposing normal contributions. The Nitsche BC uses the
Taylor-shift (`N_a + ∇w·d` test, `u + ∇u·d` trace) already in
`sbm/poisson.py`/`sbm/vector.py` (matches the reference incl. the p2
Hessian).

**What's new** (vs DiffSim's entirely volumetric SBM): a **non-carve-out
surrogate extraction** producing the two-sided face set; an **open-surface
signed distance** with per-side sign; relaxation of the
`SurrogateDotTrueNormal<0` / `corr>0` assertions that the slender band
triggers.

**Staging** (introduce one new subsystem at a time — the projection
stepper is *also* new in this context):
- **R0** — projection+VMS+BDF2 on **volumetric** SBM only; validate
  cylinder Re100 / sphere Re300 (M1b). Stepper proven, shell deferred.
- **R1** — the two-sided shell surrogate + deliverable A; validate
  blocked-channel + 2-D thin-plate Re126/250 vs the carved-out reference
  (ThinShell Fig 9–10).
- **R2** — 3-D single slender object; the steady-adjoint foundation
  (small-3-D C).
- **R3** — the heroes: B1 (truck, transient, hybrid carve-out + shell),
  B2 (maize, steady forward), C (maize, steady differentiable).

**Composition note:** B1's truck is *hybrid* — carved-out volumetric body
+ co-dim-1 shell boat-tails. So the volumetric SBM (R0) and shell SBM (R1)
must compose; they are not either/or.

## 6. Adaptive meshing / moving bodies (solution-transfer)

Both heroes use adaptive octrees (wake/feature tracking; B1 optionally
moving). Per the solution-transfer paper, standard history transfer
destroys discrete solenoidality → 1/Δt-amplified pressure/load spikes.
**Requirement (R3, and any adaptive rung):** apply **one approximate Leray
projection of the transferred BDF history per adaptation event, with
surrogate-consistent BCs**. This provably removes the leading
perturbation, preserves BDF2 order + energy stability, and preserves the
no-penetration (blockage) property, at <1% overhead. Validate against the
paper's Taylor–Green scaling, oscillating cylinder, and (B1) pitching/
bluff-body loads.

## 7. Steady path & differentiability

- **Steady forward (B2):** pseudo-transient continuation — march the
  projection stepper to its fixed point `R(u;p)=0`.
- **Steady adjoint (C):** at the fixed point the transient adjoint
  recursion *itself* converges to a fixed point (all states equal at
  steady state) → march the adjoint recursion to *its* fixed point, O(1)
  memory. Equivalent to the implicit-function-theorem adjoint: one
  transposed solve of the converged (projection fixed-point) Jacobian.
  Reuses the projection machinery — no new solver family.
- **2-D differentiable framework (A):** both adjoint modes (steady +
  taped-transient frozen-dt) gradient-verified in 2-D — three-way check
  ≤1e-6, dot-product test. Shares the adjoint discipline with SP-1 R1
  (P1) and the existing `sbm/*_adjoint` lineage.

**Design objectives for C (scoped honestly):**
- **Drag / mean-flow** — pure NS; ships first. QoI = mean drag (traction
  integral, via the shell two-sided traction) w.r.t. plate/plant geometry.
- **Gas exchange** — needs a **scalar-transport block** (CO₂/H₂O
  advection–diffusion coupled to u); an additive coupling, a later rung
  when that objective is targeted (structurally the reference's HT block).
- **Light interception** — a separate geometric/radiation model, NOT a
  flow quantity; out of P2's flow scope.

## 8. 100M-DOF scaling-pathway declaration (mandatory)

- **Stage residency:** octree/narrowband build + surrogate extraction +
  distance `d` — host, per-epoch (acceptable ONLY for static geometry;
  moving/optimized geometry ⇒ per-step ⇒ must be device-resident). Volume
  RBVMS assembly, shell Nitsche assembly, all three projection solves,
  observables, steady adjoint — device (Warp + `solvers/` +
  `device_assembly`).
- **Budget:** 3-D, ndof=4, p1 hex 27-node stencil → nnz/dof ≈ 108; at
  100M dofs nnz ≈ 1.1e10 ⇒ **nnz ≥ 2³¹ ⇒ ChunkedCSR (#38) mandatory**.
  Matrix ≈ 1.1e10 × (8B val + 4B col) ≈ **130 GB** ⇒ beyond one 80 GB GPU
  ⇒ forces one of {**matrix-free**, **fp32 values** (#36 fp32-IR halves
  value bytes — load-bearing), **multi-GPU (S4)**}. The SPD pressure-
  Poisson is the AMG/AMGX-scalable win of the projection choice.
- **Deliverables B/C's hero rungs (R3) are the hard dependency on S4
  (multi-GPU) or matrix-free+fp32 single big node.** A and R0/R1 run
  single-GPU at 2-D / small-3-D.
- **Deployment tiers:** workstation single-GPU (A, R0, R1 gates);
  single big node GH200/NVL4 (matrix-free+fp32 3-D hero); multi-node
  Horizon / AWS (true 100M via S4).

## 9. Rung ladder & gates (summary)

| Rung | Content | Gates / validation |
|---|---|---|
| **R0** | VMS-projection + BDF2 on volumetric SBM | NS MMS (±0.10/field), BDF2 order, cylinder Re100 Cd=1.352, sphere Re300 (M1b) |
| **R1** | Two-sided shell surrogate + 2-D differentiable (A) | blocked-channel blockage; 2-D thin-plate Re126/250 vs carved-out (ThinShell); shell MMS; three-way gradient ≤1e-6 + dot-product (A) |
| **R2** | 3-D single slender object; steady-adjoint foundation | 3-D thin-plate; steady-adjoint gradient check vs FD (small 3-D) |
| **R3** | Heroes: B1 / B2 / C | B1 truck Cd vs ThinShell Fig 29 + transfer-projection load cleanliness; B2 steady 100M forward; C steady-adjoint design gradient at scale; AMR transfer-projection (§6) |

Every rung carries the standing contracts: MMS ±0.10, three-way gradient
≤1e-6 (differentiable rungs), dot-product test, honest measured perf.

## 10. Process & compute placement

Subagent-driven, spec→plan→gated-rungs per rung (this umbrella first, then
P2-R0's own spec+plan). REPO-IDENTITY GUARD in all agent prompts. Compute
on **gpubox** (2-D / small-3-D gates) and **Nova GH200 / multi-GPU** (hero
rungs); the Mac only for file-level CPU checks. The C++/Dendrite reference
is the executable parity oracle — unzip + consult per rung, never commit
(`local_code_old/` is gitignored).

## 11. What P2 explicitly is NOT (scope fences)

- Not monolithic-saddle (projection chosen); the reference code's
  monolithic RBVMS is a *physics* parity target, not a stepper to match
  line-by-line.
- Not a transient 100M differentiable hero (steady adjoint only at hero
  scale; transient adjoint stays 2-D per deliverable A).
- No HT/Boussinesq or scalar-transport (gas exchange) in R0–R2; additive
  later rung when the gas-exchange objective is targeted.
- No light/radiation model (separate, non-flow).
- Multi-GPU (S4) is a *dependency*, developed on Track S, not inside P2.
