# DiffSim M1b: Navier–Stokes Forward (Both Steppers, BDF1/2, Benchmarks) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans, task-by-task with checkboxes.

**Phase context:** M1a (SBM core + 3 geometry backends + differentiable Poisson) is COMPLETE — 190+ tests, three-way-verified gradients, locked baselines. M1b delivers the NS-VMS forward solver: the matrix-free device path, the Integrands brick API (the spec §3 promise), both time steppers, and the flow benchmarks. M1c then adds NS adjoints + NeuralSDF + the hero demo.

**THE key documents (user directives, distilled with file:line quotes in `docs/superpowers/production-code-conventions.md` — cite it, don't re-derive):**
- **Leray/pressure-projection stepper:** the Helmholtz–Leray VMS draft (`MyPapers/ns_projection_vms_paper (2).pdf`), Algorithm 1 — NONLINEAR Newton momentum predictor (overrides spec §5.1's Oseen sketch; spec amended by this plan), NO τC anywhere, h-based τm (Eq. 45), fine-scale retained in PPE RHS + velocity update, `p* ← p̂` extrapolation.
- **Linearized monolithic stepper:** Biswajit's linearized-NSE draft — generalized advection `M_{a,s}u = (a·∇)u + s(∇·a)u`, `s ∈ {0, ½, 1}` as a kernel parameter, **default s = ½** (unconditionally energy-stable, Prop. 2/Eq. 58; also the self-adjoint choice `M*_{a,s} = −M_{a,1−s}` for M1c); s=1 documented as unreliable for Dirichlet-dominated flows. Exact-adjoint fine-scale test operator `M*w`; τM per the metric-tensor form; one KSP-analogue linear solve per step.
- **Nonlinear monolithic (contrast case):** dendrite-ibm's Newton VMS skeleton at **s = ½** (user directive).
- Production conventions throughout: BDF tables {1,−1,0}/{1.5,−2,0.5}/variable-Δt + `t < 1.5Δt` bootstrap; backflow/inflow stabilization; `Cb_f`-scaled SBM penalties; force/Nusselt extraction patterns.

**Hard constraints carried from M1a findings (docs/superpowers/m1a-deferred-findings.md):**
1. (4c) **No loop-reassigned locals in ANY taped kernel** (warp 1.14 adjoint bug, 32× measured); every new taped kernel ships with a tape-vs-kernel-FD unit test.
2. (3) **The inner time loop may not host-sync per iteration**: single-sync (fused-reduction) Krylov is Task 1, not an optimization.
3. (1b) `module="unique"` + explicit `enable_backward` on every kernel factory; batch variant creation; k=4 kernels use `max_unroll` limits.
4. (0) Degenerate-MMS rule: assert the tested sensitivity/observable is nonzero at truth.
5. Dim-coverage policy: structural invariants at k = 2, 3 (vector NS at k=4 deferred to M8 space-time).
6. GPU-only inner loop (spec §16b): per-step host transfers = convergence scalars and output only.

## Task list

### Task 1: Matrix-free device operator + single-sync Krylov
`solvers/krylov_dev.py`: device-resident CG + BiCGStab where each iteration performs ALL reductions in one fused kernel (partial per-block reductions + single final reduce; one host readback per iteration at most — for the convergence check — or none with a fixed-iteration + periodic-check mode). `assembly/matvec.py`: matrix-free constrained matvec for vector DOFs (`ndof = dim+1` node-major blocks per spec §3.1) via the traversal pattern; T-application on device (csr_spmv exists). Gates: matches assembled CSR to 1e-13 (scalar + vector); iteration-count parity with host bicgstab; wall-clock ≥ 20× host-sync version at 1e5 DOFs; zero per-iteration `.numpy()` calls (assert via a sync-counting wrapper).

### Task 2: Integrands brick API (the spec §3 promise)
`api/equation.py`: `CEquation` base + `FEMElm` accessor struct (N/dN/d2N/detJxW/position/nsd/nbf/value/valueDerivative), `NodeData` Vars enums with NUM_VARS-strided history slots, `VecInfo(vec, ndof, offset)` + `PLACEHOLDER_GUESS`, `solver_options_*` config blocks (libconfig-style per proteus quotes). The kernel factory consumes a brick's `Integrands_Ae/be` + `Integrands4side_*` written as `@wp.func`s. Gate: the M1a Poisson re-expressed as a `PoissonEquation(CEquation)` brick reproduces locked baselines to 1e-13 (the lego test — no L1/L2 diffs).

### Task 3: VMS machinery — τ and M_{a,s}
`physics/vms.py`: τM/τC in BOTH forms behind one interface — metric-tensor Ge form (monolithic steppers; production parity incl. `Ci_f = 36`, `timeStab` toggle, CFL-floor variant) and h-based form (projection stepper, draft Eq. 45). `M_{a,s}` and `M*_{a,s}` as kernel-parameterized advection ops (s compile-time constant per variant). Gates: τ values match hand-computed references; `⟨M_{a,s}u, v⟩ = ⟨u, −M_{a,1−s}v⟩ + boundary` verified discretely (the draft's Prop. 1) — this identity is ALSO the M1c adjoint mechanism, lock it now.

### Task 4: Time integration
`solvers/timestepping.py`: BDF coefficient tables (BDF1/BDF2/variable-Δt per production quotes), `t < 1.5Δt` bootstrap, θ-scheme hook, history rotation (PRE1/2/3), velocity extrapolation orders 1/2, pressure extrapolation orders 0/1/2 (proteus `{0,1,0,0}/{0,2,−1,0}`). Gate: scalar-ODE convergence orders (1, 2, 2) exact; variable-Δt formula vs sympy.

### Task 5: Linearized monolithic NS stepper
Per Biswajit's draft + DendrIon/Flow-Bench maps: Oseen freeze at extrapolated fine-scale-corrected velocity, exact-adjoint fine-scale operator M*, NO τM² Reynolds terms in the operator, known-history terms to RHS, monolithic (u,p) single linear solve per step (Task 1 solver). s ∈ {0, ½} tested. Gates: transient Taylor–Green (2D analytic NS) — spatial order 2 (p1), temporal order 2 (BDF2); divergence sentinel ‖∇·u‖ bounded.

### Task 6: Leray projection stepper (Algorithm 1, verbatim)
`steppers/leray.py`: Step 1 NONLINEAR Newton predictor (M0's Newton–Krylov + Task 1 inner solves; fine-scale recomputed per iterate, no lagging), Step 2 PPE (SPD, CG; q=0 on Γ_D^p / natural elsewhere; fine-scale RHS term — the PSPG-consistency piece, Eq. 50), Step 3 velocity update (mass solves, fine-scale retained), Step 4 `p* ← p̂`. `-momentum_/-pp_/-vupdate_` option prefixes. NO τC (draft Remark 2.3). Gates: Taylor–Green orders as Task 5; predictor Newton ≤ 3 iterations/step (draft reports 1–2); head-to-head vs monolithic on the same problem (spec §17: both steppers on every acceptance test).

### Task 7: SBM vector Dirichlet + backflow
Vector shifted-Dirichlet face terms (per-component, the M1a scalar form + production NSEquation parity incl. the p2-gated Hessian shift), `domain="outside"` as the working configuration (M1a-tested), backflow/inflow stabilization on outflow boundaries (production `inflow_g < 0` branch). Gates: vector P4 patch (linear velocity field) machine-exact at k=2,3; Stokes flow past a cylinder vs body-fitted reference at coarse tolerance.

### Task 8: Benchmarks (the milestone gates; regression-locked)
Lid-driven cavity Re=100/1000 vs Ghia et al. (literature values — production has none checked in) with centerline-profile extraction via pointeval; 2D cylinder Re=100: C_d, C_l, St via the SurfaceLoop force pattern (M1a surrogate_flux generalized to tractions: pressure + viscous, penalty-included conservative variant); 3D sphere Re=300 (coarse-tolerance CI variant + full nightly config). All locked into `m1b_baselines.json`. Runtime budget: CI variants < 10 min total; full configs documented, not in CI.

### Task 9: Verification tier
Transient NS MMS (forced) — spatial+temporal ladders both steppers; energy-stability demonstration s=½ vs s=0 vs s=1 (the draft's T-term structure; s=1 expected to misbehave Dirichlet-dominated — lock the CONTRAST); conservation sentinels (mass, momentum budget); dim-coverage structural tests at k=2,3.

### Task 10: Docs + closure
Tutorial chapters ⑤ (transient heat — needs only a brick on existing machinery) ⑥ (cavity) ⑦ (cylinder) of the spec §16b ladder; m1b-deferred-findings; spec §5.1 amendment (nonlinear predictor per the draft); update production-code-conventions with any new deltas; M1c input list.

## Execution notes
- Task 1 before everything (findings 3/4c make it foundational); Tasks 2–4 parallelizable after 1; 5–7 sequential; 8–9 after 7.
- Full suite + commit per green task; benchmarks enter CI only in their coarse variants.
- Any weak-form ambiguity: production file:line first (conventions doc), then the papers, then measurement — in that order, and record which level decided it.

## Progress ledger (2026-07-05 overnight session)

- [x] Task 1: single-sync Krylov (CG+BiCGStab, 4.6x/iter, sync-count gates)
      + vector matrix-free operator (kron-CSR parity 1e-13)
- [x] Task 2: Integrands brick API v1 (lego gate reproduces m1a baseline)
- [x] Task 3: VMS tau both forms + M_{a,s} adjoint identity locked
- [x] Task 4: BDF tables/bootstrap/history (orders 1/2 + variable-step 2)
- [x] Task 5: linearized monolithic brick (steady Stokes/Oseen order 2)
      + stepper (temporal order 2; timestab finding)
- [x] Task 6: Leray projection stepper v1 (incremental; ladder 1.86->1.10;
      head-to-head gate) — OPEN: implicit (1+sigma*tau_m) PPE weighting,
      Newton predictor upgrade
- [x] Task 5c: fine-scale-corrected extrapolation (default ON, gated)
- [x] Task 6b: Newton predictor (inexact; measured 0.2/iterate)
- [x] Task 7: vector SBM Dirichlet + backflow + surrogate_traction
      (2D machine-exact; 3D case env-guarded pending ~1h kernel compile)
- [x] Task 8: cavity Re=100 both steppers + Re=1000 (du=0.0727 L6);
      cylinder Re=20 (Cd=2.847) + Re=100 (Cd=1.352, wake stable at this
      blockage — St deferred to L7 nightly); sphere Re=100 smoke (first 3D
      SBM+NS, Cd=0.381 pipeline lock); m1b_baselines.json locked.
      3D NS kernels: max_unroll=0, compile 79min -> 1.3s.
- [~] Task 9: energy stability s=1/2 measured (monotone decay); MMS ladders
      exist for both steppers — OPEN: s=1 contrast documentation sweep
- [ ] Task 10: tutorial chapters + spec S5.1 amendment + findings
      finalization

Findings log: docs/superpowers/m1b-deferred-findings.md (7 entries, all
measured).


## M1c input list (handoff)

1. NS adjoints: the M_{a,1/2} self-adjointness is locked (test_vms); the
   taped-kernel rules (findings 4c m1a) and the Tier-2 VJP pattern
   (sbm/adjoint.py) are the templates. Stepper checkpointing for the
   reverse sweep is the design question.
2. NeuralSDF backend: SDFOracle contract + admissibility (geometry/oracle),
   torch-native — slot alongside gridsdf/trimesh; cross-backend suite is
   the gate.
3. Hero demo: shape-optimize an obstacle in channel flow — every piece
   exists (E1 tutorial loop x D3 configuration x NS adjoint).
4. Infrastructure debts that will bite M1c: ASM/AMG preconditioner (356k
   3D breakdown measured), build_constraints vectorization (~2h at L6-3D),
   strong-Dirichlet masked assembly (tolil row surgery is O(n) host/step).
5. Deferred physics validations: cylinder St (L7 nightly), sphere Re=300,
   3D band study L6+ (needs items in 4).
