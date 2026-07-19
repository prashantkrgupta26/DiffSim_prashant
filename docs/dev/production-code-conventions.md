# Production-code conventions (Dendrite / DendrIon / FlowBench) — M1a cross-checks + M1b inputs

Distilled 2026-07-04 from Explore-agent maps of the group's repos (shallow clones,
scratchpad). Sources: github.com/baskargroup/{Dendrite, DendrIon, FlowBench_NSHT,
Flow-Bench-Dendrite}. This file is the persistent record; re-clone to re-verify.

## SBM (Dendrite: projects/Poisson-SBM, ns_fpo_sbm, Elasticity-SBM)

1. **Dirichlet Nitsche form (Poisson-SBM/include/PoissonEquation.h:407-439):**
   consistency `-N_a (grad_u . n~)`; adjoint-consistency `-(grad_w . n~)(N_b + grad_u . d)`;
   penalty `(Cb_f/h) (N_a + grad_w . d)(N_b + grad_u . d)` — **penalty tests against the
   SHIFTED test function** (matches DiffSim; the octree-SBM paper's Eq. 7 shows plain w
   but the shipped code uses Sw).
2. **Second-order Hessian shift exists in production** (ns_fpo_sbm NSEquation.h:333-459,
   Elasticity ifHessian): `secondOrderTerm = 1/2 d^T H d`, gated on `elemOrder == 2` —
   identical to DiffSim's p2 fix (shift order = basis order).
3. **Penalty parameter:** `alpha = Cb_f / h`, h = element-volume-based
   `(2^DIM * vol_jacc)^(1/DIM)`; Cb_f = 3.0 (Poisson config), 1e2 * (1/Re) (NS).
   No explicit p-scaling of the coefficient.
4. **Lambda criterion (SBMMarker.h:51-123): `RatioGPSBM` is a DROP threshold on the
   INACTIVE Gauss-point fraction** — intercepted element dropped iff
   `(1 - ActiveGP/nGP) > RatioGPSBM`. So RatioGPSBM=1 keeps ALL intercepted elements
   (Poisson/Elasticity/thermal default; surrogate outside, Omega~ superset), 0.5 for
   flow-past-cylinder. **DiffSim's original lam was inverted (retain iff frac_in >= lam);
   flipped 2026-07-04 to production semantics.** Sampling: Gauss points at RelativeOrder.
5. **Surrogate faces (CheckSurface.h:48-114):** face flagged iff ALL its corner nodes are
   (inactive OR on a dropped element) — the "all-nodes-flagged" rule; plus
   `correctCycles()` removes elements with BOTH faces of one axis flagged (sandwich
   rule). DiffSim's sub-face-probe rule is the same predicate evaluated geometrically;
   **the sandwich/cycle removal is NOT yet in DiffSim** (needed for keep-all-intercepted
   surrogates; recorded in m1a deferred findings).
6. **Distance vectors:** KD-tree (nanoflann) closest primitive on STL/MSH + analytic
   projection with edge/vertex fallback; no Newton, no SDF. NormalofGeo: n = +-d/|d|
   with inside/outside sign flip. NS truncates + warns when |d| > 2h. (DiffSim's TriMesh
   backend mirrors this; the Newton/IFT path is DiffSim's SDF-family addition, justified
   by the neural-geometry paper's Assumption 3.1.)
7. **Integrands API:** `Integrands_Ae/be(fe, Ae/be, h)`, `Integrands4side_Ae/be(fe,
   side_idx, id, ...)`, `ibm_Integrands4side_*` for immersed GPs; SBM wired via
   `SBMCalc(fe, idata, imga, kd_trees, geomID).Dist2Geo(d)`.

## Linearized NS (DendrIon include/NSEquationLinear.h — the M1b momentum solver)

1. **Oseen semi-implicit:** advecting velocity u* frozen per solve; RHS uses
   second-order extrapolation `2 u^{n-1} - u^{n-2}`; one LINEAR solve per step (KSP,
   no Newton). Monolithic (u,p), NS_DOF = dim+1.
2. **Skew parameter `s_linearity`** (config; 0.0 in shipped configs; spec S5.1 wants
   s = 1/2 default for energy stability — DiffSim decision pending M1b): trial
   convection `L u = s (div u*) u + u* . grad u`; adjoint/test operator
   `L* w = -(1-s)(div u*) w - u* . grad w`.
3. **Fine-scale VMS via the FULL adjoint momentum operator**
   `M* w = K1 sigma w + K2 L* w + K4 lap(w)` (TIME_ADJOINT/DIFFUSION_ADJOINT toggles);
   terms `-M*_a tauM R(u,p)_b`, tauC grad-div, PSPG continuity.
4. **tauM/tauC (calc_tau_alt, NSEquationLinear.h:332-403):**
   `tauM = 1/sqrt((2 K1 b0/dt)^2 + (K2 u)G(K2 u) + Ci_f K4^2 G:G)`, `Ci_f = 36`
   (hardcoded), `tauC = 1/(tauM g.g)`; G from inverse-Jacobian cofactors.
   Nondimensionalization: K1 = K2 = 1/Schmidt, K3 = 1, K4 = -1 (DendrIon EC scaling);
   plain-NS variant folds Coe_diff = 1/Re into the diffusion term.
5. **BDF (NSPNPInputData.h:163-201):** BDF1 {1,-1,0}; BDF2 {1.5,-2,0.5}; variable-dt
   BDF2 {(2 dt+dt_p)/(dt+dt_p), -(dt+dt_p)/dt_p, dt^2/(dt_p (dt+dt_p))}; first step or
   restart-without-history falls back to BDF1. sigma = b0/dt; history RHS
   (b1 u^{n-1} + b2 u^{n-2})/dt.
6. **Stepper reality check:** production is MONOLITHIC only; the projection/fractional
   stepper exists only as an unused dendrite-kt example. M1's Leray-projection stepper
   comes from the Helmholtz-Leray VMS paper (ns_projection_vms_paper), not from
   production code.

## PNP (DendrIon — M2+/hero inputs, recorded for later)

- Three modes: monolithic Newton PNP (c+, c-, phi); split linear NP + Poisson;
  mixed. No (rho, s) reformulation in the solver (only in post-processing) — the
  spec S15 charge-relaxation split is a DESIGN item, not existing code.
- NP tau: advecting velocity augmented by electromigration `Vk = u - D z grad(phi)`,
  reaction term `D z lap(phi)`, `tau = tau_scale/sqrt((2A/dt + |sig|)^2 + uGu + GG)`.
- Block driver: fixed-point Gauss-Seidel, under-relaxation `sumRatio`, fixed-beta
  extrapolation predictor `x* = (1+beta) x1 - beta x2` with residual-spike safeguard;
  NOT Aitken. Config: NSPNPBlockIter/PNPBlockIter/NSBlockIter, loopATol/loopRTol.
- BCs: production DendrIon uses STRONG Dirichlet only; the weak-BC CMAME machinery
  exists as inactive templates (PNPEquation.h:1169-1560).
- Solver blocks: `-ns_`, `-pnp_`, `-poisson_`, `-np_` prefixes; default
  ksp_type=bcgs, pc_type=asm, sub_pc_type=lu.

## Linearized NS, Flow-Bench-Dendrite variant (lin_ns_lid_driven — closest to M1b)

1. **One KSP linear solve per step** (bcgs + asm; no Newton). Advecting velocity:
   u* = extrapolation (order 1: u^n; order 2: 2u^n - u^{n-1}) of the FINE-SCALE-
   CORRECTED velocity `u_pre - tauM * res_M_pre` — the previous-step momentum
   residual res_M_pre is fully explicit (built from PRE1/2/3 only).
2. **tau frozen at the extrapolated velocity** (calc_tau(fe, u_extrapolated)).
3. **Skew split with fixed weights:** convection = `u_i du_j/dx_j + 0.5 u_j du_i/dx_j`
   (divergence form weight 1, gradient form 0.5) — NOT an (s, 1-s) parameterization;
   the operator's divergence weight is frozen at `0.5 * div(u_pre1)`.
4. **All tauM^2 (Reynolds-stress) fine-scale products are DROPPED from the
   linearized operator**; cross-term-2's off-diagonal reduces to diffusion-only.
   Known-history BDF terms move to the RHS: `res_M_RHS = (b1 u^{n-1} + b2 u^{n-2})/dt - f`.
5. **BDF bootstrap:** `t < 1.5*dt` forces BDF1 {1,-1,0}; else BDF2 {1.5,-2,0.5}.
   The PRE2-level residual's temporal term always uses BE "(u_pre2 - u_pre3)/dt" to
   avoid storing another history slot.
6. **linearns_pspg_pnp_kt staggering:** per step exactly ONE momentum KSP solve then
   ONE PNP SNES solve, rotate history — no outer sweeps or convergence check (contrast
   DendrIon's block Gauss-Seidel with relaxation). PNP->NS body force -rho_e grad(phi)
   enters via the frozen residuals. NP weak-BC boundary terms use Tezduyar SUPG
   weighting.
7. **No Ghia/cavity reference values exist in production** — only a drag/force
   SurfaceLoop integrator (ns_lid_driven/include/NSPost.h: Force = (-p n_j + viscous)
   dS, reduced globally). M1b's cavity gate must take reference values from the
   literature (Ghia et al.), and the C_d/C_l machinery should mirror NSPost's
   SurfaceLoop pattern.

## FlowBench_NSHT (SBM NS + Heat — Task 8 / M1b / M2 primary reference)

1. **Classification (SBMMarker.h):** node-count decides IN/OUT/INTERCEPTED; then GP
   inactive-fraction vs RatioGPSBM (default 1.0; strict > so lambda=1 retains ALL
   intercepted). Single shared lambda for NS and Heat. A DISTANCE_CHECK alternative
   demotes on mean normalized distance (2D-only, 3D unfinished). After marking,
   `generateNeighborsOfFalseIntercepted` propagates flags so the surrogate closes
   across level jumps.
2. **Heat SBM Dirichlet (HTEquation.h:1830-35):** identical structure to DiffSim
   (kappa-scaled, penalty Cb_e*Coe_diff/h on shifted-test x shifted-trial, second-order
   Hessian shift gated elemOrder==2 — THIRD independent confirmation of
   shift-order = basis order).
3. **Neumann SBM (HTEquation.h:1928-37) — MATCHES the mixed-p paper Eq. 21 exactly:**
   +Coe_diff N_a (grad_u.n_true + d.H.n_true)(n.n_tilde) - Coe_diff N_a grad_u.n_tilde,
   RHS -= Coe_diff N_a q_bar (n.n_tilde). PLUS an optional Neumann penalty
   beta (grad_w.n + d.Hw.n)(grad_u.n + d.Hu.n)(n.n_tilde)^2, default beta = 0
   (BetaForSBMNeumann). `WithoutAreaCorrection` toggle exists (default false).
4. **Nusselt/flux extraction — two variants:** (a) true-boundary 2-probe linear
   extrapolation of grad(theta) (probe 2 stepped 0.05h inside), direct-gradient;
   (b) CONSERVATIVE Surrogate2True: Nu_0 = shifted-flux x (n.n_tilde) over surrogate
   GPs + Nu_1 = penalty term -(Cb_e/h)(S theta - T_g)(n.n_tilde). DiffSim Task 8
   implements (b) including the penalty term (the spec S14 penalty-included form).
5. **Known edge case:** SurrogateDotTrueNormal < 0 warnings at lambda=0.5 + high
   refinement (their debug guard) — DiffSim's corr>0 assertions must tolerate this
   for keep-most surrogates.
6. **NSHT NS:** semi-implicit extrapolated advection (u* = 2u1 - u2), RBVMS with
   inflow (backflow) stabilization when u.n < 0 on boundaries; tauM CFL-floor variant
   dt* = max(dt, h/|u|); buoyancy body force Gr/Re^2 * theta (two-way coupling knob).
   Adjoint-consistency toggle IfAdjointConsistency default TRUE; NonSymmetricNitcheNS
   default FALSE.
7. **Benchmark geometry enum** includes CYLINDER, SPHERE, BUNNY, AIRFOIL, GYROID etc.;
   no configs/reference values checked in (supplied at runtime).

## NAVIER-STOKES pressure-projection (Leray) TIME STEPPER: THE key document is the
## Helmholtz-Leray VMS draft (Khara, Murugaiyan, Khanwale, Ganapathysubramanian —
## MyPapers/ns_projection_vms_paper (2).pdf; user directive 2026-07-04)

Scope: this directive governs the NS fractional-step solver (spec S5.1's
"LerayProjection" stepper, built in M1b) ONLY. It does not touch: M1a's Poisson
solves, the geometric Newton closest-point projection (Tier-2 VJP #3), or the
S4.4 pressure-robust Leray correction for epoch transfer (M4, Suresh's paper).
M1b's projection stepper implements the draft's Algorithm 1 (p. 12), NOT the spec
S5.1 sketch where they differ:

1. **Step 1 — momentum predictor is NONLINEAR, solved by Newton** (Remark 2.4: the
   fine-scale closure u' = -tau_m(u_h) r_m(u_h, p*) is recomputed consistently at
   every Newton iterate, NO lagging/extrapolation of u' or the advecting velocity;
   1-2 Newton iterations/step in practice; inner solves BiCGStab + ASM). This
   OVERRIDES spec S5.1's "Oseen-linearized predictor (a_hat = 2u^n - u^{n-1}); all
   linear" — spec to be amended at M1b planning. Advection in conservative/IBP form
   -(u (x) u, grad v); full VMS cross terms both ways + Reynolds -tau_m^2 r (x) r;
   outflow boundary term (n.[u(x)u - nu grad u], v)_GammaN.
2. **Step 2 — pressure Poisson (linear, SPD):** (grad p_hat, grad q) =
   (grad p*, grad q) - sigma (div u~, q) - sigma (tau_m r_m, grad q)  — the last
   term is the VMS fine-scale RHS that recovers PSPG-like consistency (Eq. 50).
   BCs: q = 0 on Gamma_D^p, n.grad = 0 on Gamma_N^p (boundary term vanishes).
   **NO tau_C anywhere** (Remark 2.3): pressure is not scale-decomposed — the
   projection removes the saddle point, PPE is coercive, equal-order spaces fine
   with no inf-sup machinery.
3. **Step 3 — velocity update (linear, mass matrix on ALL of H^1, no BC):**
   (u_hat, w) = (u~, w) - (tau_m r_m, w)_K - 1/sigma (grad(p_hat - p*), w) — note
   the fine-scale term is RETAINED in the projection (consistency, not stability).
4. **Step 4 — pressure extrapolation: p* <- p_hat** (first-order/incremental).
   (proteus additionally has the order-2 table {0,2,-1,0}; draft uses order 1.)
5. **tau_m (Eq. 45): h-based form** [sigma^2 + c1 |u_h|^2/h^2 + c2 C_I nu^2/h^4]^(-1/2)
   with sigma = bdf0/dt — NOT the metric-tensor Ge form used in the monolithic
   production codes. Both forms recorded; the projection stepper uses the draft's.
6. sigma from BDF; the scheme is defined per-step tn -> tn+1 with previous fields +
   extrapolated pressure as inputs — clean epoch/checkpoint boundaries.
7. The spec S4.4 "pressure-robust Leray correction" for epoch transfer (Suresh's
   moving-body paper) is a SEPARATE mechanism from this stepper; unchanged.

## Heritage repos (bitbucket: proteus, dendrite-ibm) — monolithic NS + nomenclature

1. **Advection forms as shipped:** both codes use the bare CONVECTIVE form (u.grad)u —
   i.e. s = 0 in the linearized-NSE draft's parameterization (the draft's s=1 is the
   divergence form). No skew switch exists. proteus has the conservative/adjoint twin
   `convConsv = (grad_w . u_avg) 0.5 N_b` COMMENTED OUT
   (CHNSIntegrandsGenForm.hpp:907-928) — re-enabling it against the 0.5-weighted
   advective term is exactly the s = 1/2 skew pair. M1b implements M_{a,s} fresh with
   s as a kernel parameter (decision above), so this is lineage, not a porting target.
2. **proteus = the CHNS projection stepper** (velocity-prediction KSP -> pressure-
   Poisson KSP (fgmres+gamg) -> velocity-update KSP; CH via SNES), per-solver PETSc
   prefixes `-momentum_`, `-pp_`, `-vupdate_`, `-ch_` — the spec S3.1 canonical
   sub-solve names. Pressure extrapolation table p* coefficients: order 1 {0,1,0,0},
   order 2 {0,2,-1,0} (incremental/rotational, Guermond). BDF bootstrap: step 0
   forces order 1. tauM has an extra CH mass-flux term J.G.u inside the radical and
   a separate tauPhi.
3. **dendrite-ibm = the monolithic (u,p) Newton skeleton** (SNES `-ns_`,
   NS_DOF = nsd+1, generalized-theta stepping incl. a separate theta for boundary
   terms, Bazilevs tau with `timeStab` toggle and per-side tauM_scale4Side near cut
   cells). Its immersed machinery is TRUE-boundary IMGA Nitsche (penalty
   Cb_f/hb^orderOfCb with wall-normal hb) — NO SBM precursor (no Taylor shift, no
   surrogate boundary).
4. **Nomenclature quotes captured** (agent report, this session): CEquation adapter
   delegating to a shared integrands object; NodeData Vars enum with NUM_VARS-strided
   history slots (VEL_X_PRE1 = NUM_VARS + 0, ...); VecInfo(vec, ndof, nodeDataOffset)
   with PLACEHOLDER_GUESS for the SNES iterate; solver_options_* libconfig blocks
   (ksp_* only for linear blocks, snes_* added for Newton blocks). M1b's TimeLoop /
   config layer mirrors these.

## Open reconciliation items

- Advection s-form: RESOLVED 2026-07-04 (user directive + Biswajit's linearized-NSE
  draft, MyPapers/linearized_nse_paper (1).pdf):
    * **Nonlinear monolithic NS: s = 1/2** (skew-symmetric form).
    * **Linearized monolithic NS: s = 1/2 or s = 0** — the draft's Prop. 2/Eq. 58
      proves s = 1/2 unconditionally energy-stable (T1 = T2 = 0 identically,
      regardless of div(a_h), mesh, or dt), and its conclusions find s = 0 and
      s = 1/2 "consistently accurate and robust" while s = 1 (divergence form) is
      less reliable in Dirichlet-dominated problems. DiffSim M1b default: s = 1/2,
      with s exposed as a parameter {0, 1/2, 1}.
    * The draft's Table 2 exact-adjoint identity M*_{a,s} = -M_{a,1-s} is the spec
      S5.2 adjoint-Oseen mechanism: at s = 1/2 the operator is (skew-)self-adjoint —
      the adjoint solve reuses the same integrand with a sign flip, which is also
      why s = 1/2 is the right default for the DIFFERENTIABLE stepper (M1c).
  The generalized operator M_{a,s} u = (a.grad)u + s (div a) u should be implemented
  with s as a compile-time-constant kernel parameter.
- Penalty h-definition: element-volume-based in production vs face-element h in DiffSim
  (identical on uniform cubes; differs under anisotropy — revisit if/when anisotropic).
- correctCycles sandwich removal: add to extract_surrogate when keep-all-intercepted
  (lam=1) surrogates produce slivers.
- FlowBench_NSHT + Flow-Bench-Dendrite agent reports pending (flux/Nusselt extraction,
  lin-NS lid-driven variants) — append here when they land.


## M1b closure addenda (2026-07-05)

- Inexact-Newton delta (Leray predictor): cross-term Galerkin-only; SUPG /
  tau' / (div du) a linearizations stay Picard-level. Measured contraction
  0.2/iterate (vs draft's consistent-Newton 1-2 iterations claim).
- HARD RULE: dim>=3 fat element kernels declare module_options
  {"max_unroll": 0} (compile 79 min -> 1.3 s measured, zero accuracy delta).
- Benchmark anchors now locked in tests/baselines/m1b_baselines.json:
  cavity Re=100 (mono du=0.0324/leray 0.0035 L5) and Re=1000 (du=0.0727 L6)
  vs Ghia; cylinder Re=20 Cd=2.847 / Re=100 Cd=1.352 (lit ~1.33; wake
  numerically stable at 14% blockage, D/h=9 — St is the L7 nightly);
  sphere Re=100 Cd=0.381 (L4 pipeline lock, preasymptotic).
- Jacobi-BiCGStab breaks down on the 356k-DOF 3-D SBM system: ASM/AMG
  preconditioning is the M1c-adjacent infrastructure item (amgx.py stub).


## M1b delta log (2026-07-05)

- Leray stepper implemented per Algorithm 1 with two measured deltas
  documented in code: Picard-for-Newton (v1) and incremental-PPE form
  (the draft's tau_m grad(p_hat) implicit treatment implemented behind
  ppe_finescale; findings 5).
- tau timeStab toggle: REQUIRED off for temporal-order studies (findings
  1); production default stays on.
- Solver stack beyond production's PETSc map: cuDSS default, AMGX for SPD
  subsystems, fused device Krylov; production bcgs+asm has no analogue yet
  (block preconditioning tracked, findings 8f).

## Dendro-KT / Dendro-5.01 insights (2026-07-05 exploration; file:line in repo clones)

1. CONSTRAINTS WITHOUT POINT LOCATION (the build_constraints fix): Dendro-KT
   never per-node-searches. Generate all 2^k-duplicated node instances,
   Morton-SORT, then one segment-count sweep: a node on a k'-cell is
   non-hanging iff it appears exactly 2^(dim-cdim) times at its location
   (SFC_NodeSort::resolveInterface_lowOrder, nsort.tcc:1266,1302-1308);
   mixed-level => coarser wins; scatter map produced in the same pass.
   Dendro-5.01 does hanging classification ON DEVICE as a pure gather:
   compare owner node's octant level to element level (GPU mesh_gpu.cuh
   is_node_hanging). Sort + segment-count + gather — all warp-friendly.
2. MATRIX-FREE T OPERATOR: constraints applied inside the matvec as
   per-axis tensor-product 1D interpolation selected by child bits
   (RefElement IKD_Parent2Child / Child2Parent, matvec.h:450,502;
   KroneckerProduct<dim>, refel.h:215) — null non-hanging contributions
   before the transpose (matvec.h:497-499). Dimension-generic k=2,3,4.
3. TREESORT/BALANCE: breadth-first counting-sort bucketing on Morton child
   index (tsort.cpp:229,318-324) — comparison-free, ideal for GPU
   segmented sort + prefix sums; 2:1 via bottom-up auxiliary octants.
4. 4D LESSON: everything is 1<<dim bucketing + tensor-product 1D operators
   — the 4D element op is (p+1) 1D applies, never a dense (p+1)^4 matrix;
   decompose interpolation by k-face dimension (matvec.h:487-495). Directly
   applicable to our M8 space-time and today's dim-4 compile monsters.
5. GPU PARALLELISM: parallelize over BLOCKS/segments, not elements —
   element parallelism races at hanging boundaries (Dendro-5.01
   findings/unzip_openmp.txt); per-thread interp scratch mandatory;
   reindex + SIMD tensor kernels gave 3.5-5x.

## cuFEM exploration (github.com/mshadkhah/cuFEM, 2026-07-06; file:line in clone)

Single-GPU CUDA/C++ linear-Morton octree FEM (uint64 Morton<<5|level,
DMAX=19), element-parallel flat kernels throughout. Verdicts for us:
1. VALIDATES our lookup: their "traversal" is sorted-Morton
   thrust::lower_bound + ancestor classification (NeighborTopology.cu:
   129-181) — the same algorithm as our per-level searchsorted; nothing
   faster found.
2. ADOPT: dense precomputed FaceNeighbor[oct*2dim+face] table built once
   (NeighborTopology.cuh:41-50) instead of re-searching per use.
3. M2 REFERENCE: on-device hanging/Dirichlet constraint ELIMINATION
   folded into assembly (<=4 masters w/ weights, master-DOF-only system;
   HangingNodeConstraints.cuh:7-26, FiniteElementSpace.cuh:46-52) — the
   GPU-native constraints design when we migrate off host T-matrices.
4. VALIDATES interleaved node-major DOFs (row = node*ndof + dof,
   Assembler.inl:56).
5. AVOID: their CSR numeric assembly = per-element atomicAdd with
   binary-search-per-column (AssemblyUtils.cuh:11-68) — atomic-contention
   anti-pattern at scale; coloring/matrix-free stays our plan.
6. Solver = AMGX aggregation-CG only; geometry = STL ray-cast cut-cell
   (no SBM, no differentiability, no VMS) — no overlap with our edge.

## Scaling-pathway declaration (standing rule, ratified 2026-07-19)

Every new physics development (NS, NS-SBM, thin-shell-NS, NS-PNP, PNP, XDD,
AM, and successors) must declare its 100M+-DOF pathway AT THE SPEC STAGE:

1. **Stage residency table** — assembly / solve / closures / marching /
   observables, each marked `host | device | either`. "Mixed" requires an
   explicit Amdahl budget ("stage X is host at Y s/step; acceptable at target
   scale because Z") — never mixed-by-default. Motivation: the 2026-07-19
   forensics found a ~61 s/step host floor (assembly/scatter/GP fields)
   common to Ada/A100/GH200 — a silent Amdahl ceiling inside a nominally
   device-resident stack (efficiencies capped at 55–75% of bandwidth-predicted).
2. **100M-DOF budget line** — bytes/dof, nnz/dof, index width (mixed-width
   CSR per the P0-2 templating), multi-GPU comms pattern if any stage halos.
3. **Deployment tiers** — workstation single-GPU (2-D production) · single
   big node (GH200 / Horizon NVL4 hero runs) · multi-node (Horizon gb-large /
   AWS on-demand clusters).

CPU-only code = tooling and parity references only; production compute
remains GPU-only per the standing program rule.
