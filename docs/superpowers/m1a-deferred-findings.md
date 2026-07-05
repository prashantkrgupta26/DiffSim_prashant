# M1a Deferred Findings (running log; finalized in Task 11)

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
