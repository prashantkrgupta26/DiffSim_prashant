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
2. **`build_constraints` host probe loop:** 48 s for 4D p2 (7105 nodes), from
   calling `LeafLookup.find` once per (node, probe) instead of batching all
   2^dim probes across all nodes into one `find` call. Vectorize in a
   deliberate pass with the S13.3 batteries as the safety net — not a drive-by
   edit (M0.5-validated code). Now the largest single suite cost at dim=4.
