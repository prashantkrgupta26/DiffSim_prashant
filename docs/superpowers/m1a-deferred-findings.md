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
3. **`build_constraints` host probe loop:** 48 s for 4D p2 (7105 nodes), from
   calling `LeafLookup.find` once per (node, probe) instead of batching all
   2^dim probes across all nodes into one `find` call. Vectorize in a
   deliberate pass with the S13.3 batteries as the safety net — not a drive-by
   edit (M0.5-validated code). Now the largest single suite cost at dim=4.
