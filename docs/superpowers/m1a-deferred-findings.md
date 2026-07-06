# M1a Deferred Findings (FINALIZED 2026-07-05 — M1a complete)

## 0. Test-design care points: the degenerate-MMS trilogy (one night, three
## bites — write this into the verification-tier docs)

Manufactured solutions/QoIs can be silently BLIND to the mechanism under
test: (a) sin*cos on a centered disk has ZERO net flux — cannot exercise the
Neumann area correction; (b) the same zero made "flux == F*" look like a
conservation identity (it was 0 == 0); (c) a LINEAR patch problem has u == g
on ANY domain (the P4 exactness property!), so dJ/d(geometry) is identically
zero and gradient tests compare noise to noise. RULE: before locking a test,
verify the sensitivity/observable it measures is NONZERO at the truth
(assert the magnitude, not just the match).

## Performance findings from Task 1 (2026-07-04, RTX 6000 Ada, warp 1.14, WSL2)

1. **Warp module compile cost is the dominant test-suite cost, and backward
   codegen doubles-to-10x's it.** `diffsim.assembly.operators` with dim=4
   kernels took ~280 s/compile with `enable_backward` on (warp default), ~16 s
   with it off. Fixed in Task 1 for all existing kernel modules (none are
   taped; adjoints go through CSR transpose per spec S5.2). **Rule going
   forward:** every new kernel module declares `enable_backward` explicitly —
   off unless the module's kernels are differentiated under `wp.Tape` (M1a:
   only the SBM face-residual kernels; M1b matrix-free paths revisit).
   Corollary: module hashes change whenever a kernel factory adds a variant,
   recompiling the whole module — batch kernel-variant creation where possible.
1b. **Warp kernel-module isolation adopted (module="unique") + the dim=4 compile
   monster.** Follow-up to finding 1: per-variant kernel modules (all 12 factories,
   2026-07-04 overnight) eliminate the module-hash churn — each variant compiles
   once, content-keyed, immune to unrelated edits. Remaining cost: the dim=4
   element-matrix kernels (poisson_Ke/Ke_var, 16 qp x 16x16 basis pairs x 4 dims
   fully unrolled) take tens of minutes of nvrtc compile EACH, one time. Mitigation
   for M8 (k=4 space-time): warp's max_unroll module option / rolled loops for k=4
   kernel variants — unrolling is a 3D-hot-path optimization that k=4 does not need.
   Watch: first-ever run on a fresh machine pays the 4D compile; document in CI.
2. **Closest-point projection: the simple update is not IFT-consistent
   (Task 3 finding, blueprint-relevant for spec S4.1).** The iteration
   y <- y - psi grad_psi/|grad_psi|^2 reaches {psi = 0} but, for non-eikonal
   fields (CSG blends, grid/neural SDFs), NOT the closest point (measured
   tangential misalignment ~8e-4 on a k=0.05 blend), and its fixed point is
   path-dependent — so it is not a well-defined differentiable map and IFT
   gradients computed against the augmented system disagree with what the
   forward actually computed. Resolution (implemented): gradient projection
   is only the warm start; convergence is driven by full Newton on the
   augmented system F(y, s) = [y - x + s grad_psi; psi] = 0 — the same system
   the IFT backward differentiates. The blueprint chapter should state the
   augmented system as THE projection definition, with the simple update as
   warm start only.
3. **Krylov host-sync latency is fatal on WSL2 at prototype sizes (Task 6
   finding; hard requirement for M1b matrix-free).** bicgstab does ~6
   host-synchronizing `blas.dot` calls per iteration (~10 ms round-trip each
   on WSL2); nonsymmetric SBM-Nitsche systems at tol 1e-14 need thousands of
   iterations => the 16-test P4 battery took 71 min via bicgstab vs 89 s via
   scipy splu (SBMPoisson default solver="direct", the cuDSS-analogue of the
   spec S5.4 reuse map; solver choice is invisible to gradients — Tier-2 VJP
   boundary). M1b's matrix-free NS path CANNOT use direct solves: it needs
   fused/batched reductions (single-sync iterations), pipelined bicgstab, or
   CUDA-graph capture — this is the concrete instantiation of the M0
   "dot() sync-per-call" deferred item, now with numbers.
4. **Taylor shift order MUST equal basis order — in BOTH directions (Task 7
   finding; spec S4.2 update needed).** (a) p2 with the first-order shift:
   L2 capped at order ~2 (pairwise 2.60/2.12) — the O(d^2) boundary
   consistency error dominates. Adding 1/2 d^T H(u) d restored order 3
   (measured pairwise 3.44/2.98, superconvergent early). (b) p1 WITH the
   Hessian term: order DEGRADED 2.00 -> 1.85 — Q1's Hessian is incomplete
   (mixed partials nonzero, diagonals identically zero) and the mixed-only
   correction pollutes the consistent first-order shift. Production gates on
   elemOrder==2 for the same reason. Implemented: per-p shift selection in
   the kernel factories (_shift_fn_for). Blueprint chapter: "Taylor shift at
   basis order — never above it."
4b. **SBM Neumann p2-band: thickness/adjacency is LOAD-BEARING, not a free
   parameter (Task 8 overnight diagnosis; for the local-p draft's authors).**
   Measured, exterior-disk MMS, levels 5-7, lam=0: (a) Eq. 21 as written is
   consistent and EXONERATED — with exact surrogate-point data the pipeline
   is clean order 2.00/2.00 at p1; (b) fully-shifting the tangential term
   (Atallah S-grad on the surrogate-normal side only) makes things WORSE
   (one-sided O(d) inconsistency) — reverted; (c) the real mechanism: band
   elements carrying the Hessian must be decoupled from the minimum-rule p1
   trace constraints — face-neighbor rings of 1-2 layers cap at order ~1
   (0.85/1.04), 4 layers restore 2; NODE-adjacent growth ('cells touching',
   the draft's semantics) is clean at >= 3 layers (2.01/2.04), marginal at 2
   (1.78/1.37). p2_band now grows by node adjacency; the draft's layer-sweep
   claim needs the caveat "insensitive once thick enough" + explicit
   adjacency definition. (d) TOTAL-FLUX observable care points, twice burned: the sin*cos MMS has
   ZERO net flux through the disk (F* = 3e-16) — it can neither test the
   area correction nor distinguish flux errors; and the apparent
   "conservation identity" (flux == F* to machine zero) was the same
   artifact. With a flux-carrying quadratic MMS: corr-on flux error ~7.6%
   |F*| at level 5 (discretization order), corr-off adds the ~27% (4/pi-1)
   staircase inflation. Locks now measurement-based; primary observable is
   L2 on Omega (the draft's own Fig. 6 evidence).
   (e) COMPREHENSIVE VALIDATION (2026-07-05, exterior sphere r=0.25,
   lambda=1 keep-all, node-band(3) vs p1-only):
   2-D levels 5-8: band errors 6.62e-4 / 1.75e-4 / 4.53e-5 / 1.09e-5,
   orders 1.92 / 1.95 / 2.06 — sustained second order to 58k DOFs; p1-only
   6.78e-3 / 4.84e-3 / 2.16e-3 / 1.13e-3, orders 0.49 / 1.16 / 0.93 —
   pinned at ~1 at every scale (104x accuracy gap at L8). Band elements
   grow like the boundary (~2x/level, 188 -> 1416) vs mesh 4x/level: the
   band's relative cost VANISHES under refinement.
   3-D levels 4-5: at L4 (sphere ~8 elems across) the band is 4.3x WORSE
   than p1 (1.99e-2 vs 4.68e-3) — preasymptotic: the shift machinery
   amplifies unresolved-geometry error. At L5 it snaps in: 5.70e-4 vs
   3.52e-3 (6.2x better). RULE: engage the p2 band only once the feature
   is ~15+ elements across. L6 3-D bounded by the build_constraints host
   loop (finding 6 / m0.5) — vectorizing it is the unlock for 3-D band
   studies at scale. Driver: benchmarks/band_study.py.
4c. **WARP 1.14 ADJOINT BUG (Task 10; report upstream; HARD RULE for all
   taped kernels incl. M1b matrix-free).** The loop-reassignment pattern
   `jacS = wp.float64(1.0); for _ in range(dim-1): jacS = jacS * half`
   inside a kernel differentiated by wp.Tape produces gradients scaled by
   EXACTLY 1/jacS for ALL differentiable inputs (measured 32.000x at
   h=1/16, dim=2, uniformly — even inputs entering linearly far from jacS;
   verified by kernel-level FD, and the torch dense twin + pipeline FD
   agreed with each other to 7 digits against it). Forward values are
   correct — only the adjoint is wrong. Fix: compute such factors without
   reassignment (wp.pow). RULE: no overwritten-local accumulation patterns
   in ANY enable_backward=True kernel; add a tape-vs-kernel-FD unit test for
   every new taped kernel. (This is also a testament to the three-way gate:
   adjoint-vs-twin caught a plausible 32x-wrong gradient that FD-only
   spot-checks at loose tolerance might have rationalized.)
5. **Weak-form conventions verified against the group's papers (2026-07-04,
   MyPapers/).** (a) Dirichlet: matches the octree-SBM paper (Mehdi et al.,
   Eq. 7) in every sign; ONE deliberate delta — our penalty tests against the
   shifted Sw (Main & Scovazzi original; also the group's own elasticity
   form Eq. 9), theirs against plain w for Poisson. Both are patch-exact;
   revisit at X2 cross-code comparison. (b) Neumann: plan updated to the
   local-p-refinement draft's Eq. 21 verbatim (shift only the NORMAL flux
   component; a = n.n_tilde on the data term) — the earlier transcription
   had an extra tangential Hessian-shift term (same order, different
   convention). (c) The neural-geometry paper's Assumption 3.1 / Lemma 3.4
   (|grad psi| >= c0 tubular band, Newton projection success masks) is
   exactly the oracle admissibility contract implemented in Task 3.
6. **`build_constraints` host probe loop:** 48 s for 4D p2 (7105 nodes), from
   calling `LeafLookup.find` once per (node, probe) instead of batching all
   2^dim probes across all nodes into one `find` call. Vectorize in a
   deliberate pass with the S13.3 batteries as the safety net — not a drive-by
   edit (M0.5-validated code). Now the largest single suite cost at dim=4.
   (f) 3-D band, levels 4-6 (overnight campaign; L6 unblocked by findings
   m1b-9 + solved by cuDSS after fused-Jacobi DIVERGED and AMGX classical
   returned not_converged at 356k nonsym SBM DOFs): errors 1.99e-2 /
   5.70e-4 / 1.38e-3 — NON-MONOTONE. Reading: the L4->L5 'order 5.1' snap
   and the L5->L6 rise are consistent with an error-cancellation DIP at L5
   (dominant error component crossing zero in preasymptotic recovery);
   the honest 3-D asymptotic rate needs L7 (~2.8M DOFs — cuDSS-feasible on
   48 GB, attempt queued). L6 build: 18.6 min end-to-end (was hours).
   (g) 3-D L7 (~2.8M DOFs): the BUILD now completes (findings m1b-9 +
   the dense-broadcast fix) but cuDSS factorization ALLOC_FAILED on the
   48 GB card — the single-GPU direct ceiling measured exactly where the
   capacity estimate placed it (3-D L6-L7 boundary). The L6->L7 order pair
   (dip-hypothesis verdict) awaits either a tuned AMGX config for the
   nonsym SBM system or multi-GPU AMGX — both queued. The band study
   record stands at L4-L6.

4c-AMENDMENT (2026-07-05 overnight, M1c kernel work; repro scripts in
docs/superpowers/warp_adjoint_repros/): the warp 1.14 taped-kernel bug
FAMILY, measured by term-masked bisection:
  BUG #1 (original 4c): loop-reassigned locals scale gradients 1/jacS.
  BUG #2 (NEW, repro nan_micro2.py): scalar accumulators initialized
  inside SECOND-level unrolled loops poison the ENTIRE tape with NaN —
  whether the accumulator is used or dead. Depth-1 accumulators (the m1a
  kernel shape) are safe.
  BUG #3 — RESOLVED, root cause found (the night's ten-falsification
  hunt): THE FEMElm STRUCT. Mutating struct fields inside kernel loops
  (fe.q = q per q-iteration) silently breaks warp 1.14's backward replay
  and poisons the tape with NaN. Every NaN formulation tonight (Ae-style,
  func-based, whole-value algebra, full-scalar) used FEMElm; the m1a
  kernels predate it and pass; stripping FEMElm for direct table indexing
  turned the 4c contract GREEN with the tape matching kernel-FD to 8
  digits. Falsified along the way (all repro-clean, scripts committed):
  vec/mat op adjoints, mat-vec, loop-accumulated mats, tau funcs,
  max_unroll dynamic loops (the generated .cu was fully unrolled),
  uninitialized grad buffers (verified zeroed). RULES FOR TAPED KERNELS
  (4c final): (1) no loop-reassigned locals (bug #1); (2) no scalar
  accumulators above loop depth 1 (bug #2); (3) NO STRUCT FIELD MUTATION
  — use plain locals and direct table indexing (bug #3, the big one);
  wp.pow for jacobians. Upstream report: FEMElm minimal repro TODO (the
  struct + loop mutation pair), plus the bug #2 nan_micro2 repro.

4d. **TAU-FROZEN adjoints: right for frozen-field partials, LEAKS in
   transient chains (measured).** The production tau-frozen pattern
   (stabilization parameter's advecting-field dependence not
   differentiated) is exact for shape/parameter gradients at a FROZEN
   advecting field. In the transient chain the advecting field varies
   with earlier states, and FD sees dtau/daq: measured leak 3.8e-4 per
   BDF1 chain link, 4.4e-3 under BDF2 coefficients. The volume residual
   kernel now has both variants (tau_frozen flag); the transient chain
   uses tau_frozen=False and lands at rel 4.4e-10 over 4 BDF2 steps.
   Diagnosis path preserved in the N=1/2/3 isolation ladder — N=1 exact
   at 1.9e-8 pinned the leak to the chains in one shot.
