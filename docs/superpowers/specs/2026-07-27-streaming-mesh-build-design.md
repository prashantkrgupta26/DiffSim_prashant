# Streaming Adaptive Mesh Build — Design

**Date:** 2026-07-27. **Approved direction:** Baskar (brainstorm 2026-07-27):
the design must serve (eventual) adaptive REMESHING and mixed p1/p2 meshes.

## Problem

The host adaptive mesh build is a measured wall on the 100M-DOF path:
**209.7 GB RSS** building the r7b9 3-D adaptive mesh (~10–15M-DOF-class target;
GH200 ladder, SLURM oom_kill at the 200G cgroup with the GPU idle). The
readiness study (`docs/dev/2026-07-27-100m-gh200-readiness.md`, WP1) sets the
requirement: **≤ ~150 GB host RSS for a 100M-DOF adaptive build (~1.5
GB/M-DOF)** — roughly a **14× memory reduction**. The closed-form uniform
build (merged; 17×/22× time) covers uniform-complete trees ONLY; adaptive
trees fall through to the general path (`np.unique`-based global dedup).
A per-stage memory attribution probe (`tests/mesh_build_memprofile.py`) is
running; its numbers prioritize the work packages below but do not change
the architecture.

## PROBE FINDING (2026-07-27, revises the problem statement)

The Mac ladder (tests/mesh_build_memprofile.py) measured the CURRENT builder at
**0.86-1.13 GB/M-DOF total peak** (L5/r7 0.20M-DOF/0.22GB; L6/r8 1.28/1.11;
L6/r9 1.80/2.04) — extrapolating to ~100 GB at 100M DOF, ALREADY under the WP1
budget. The GH200's 209.7 GB at r7b9 (~10-15M-DOF-class) is therefore ~14x
ABOVE this rate: the wall is a SUPERLINEAR ANOMALY that activates between
L6-base and L7-base 3-D trees, not a linear cost. Consequences:
1. **WP0 (new, first): anomaly diagnosis** — reproduce r7b9 on Grace
   (sbatch --mem=400G) with per-stage subprocess/tracemalloc attribution
   (the RSS-sampler attribution failed: allocator retention shows cumulative
   values); identify the superlinear stage; it may admit a targeted fix.
2. The streaming architecture below REMAINS the design for remeshing +
   mixed-p + headroom, but its urgency depends on WP0: if the anomaly is a
   targeted fix, streaming becomes the medium-term architecture rather than
   the 100M blocker-remover.

**WP0 RESOLUTION (2026-07-27, Grace job 11771926): the adaptive mesh build is EXONERATED.** The exact r7b9 case (base L7/band r9, 2.31M nodes / 9.24 M-DOF) builds in **3.71 GB peak, 94 s** on Grace (0.40 GB/M-DOF; L6/r9: 0.89 — sublinear per-DOF). The 209.7 GB OOM was CROSS-LEG ACCUMULATION in the single-process GH200 ladder (prior legs' assembled CSRs + the failed L7-uniform cuDSS factorization's host staging), mis-attributed to the build because the job died during that leg with the GPU idle. Mesh build extrapolates to ~40-90 GB at 100M DOF — inside Grace. The streaming build reclassifies to the medium-term remeshing/mixed-p architecture; the 100M critical path is the solver (R2b saddle preconditioner / matrix-free).

## Architecture: streaming skeleton + closed-form chunk producer

**Foundational fact (M0.5):** every constraint type in this framework is
neighbor-local — 2:1 hanging-node, p-interface, and transitive chains of
depth ≤ 2. Therefore a **bounded halo** (2 cells at the finest local level)
around any Morton-contiguous chunk contains everything needed to produce that
chunk's nodes, connectivity, and constraint rows **exactly**.

### Components

1. **Chunk planner** — partitions the (already-refined, balanced) leaf array
   into Morton-contiguous chunks with a bounded working-set target (config:
   `chunk_max_cells`, default sized so working set ≤ ~2 GB). Each chunk knows
   its halo leaf set. Pure indexing over the leaf array (which itself is
   O(cells·keysize) — the leaf array at 100M-DOF scale ≈ 25M cells ≈ small).

2. **Chunk producer** — per chunk, dispatch:
   - **Closed-form fast path** when the chunk (incl. halo) is
     uniform-complete at one (level, p): the merged closed-form generator
     emits nodes/connectivity/identity-constraints directly (no dedup
     structures at all).
   - **General path** otherwise (bands, level interfaces, p interfaces):
     the EXISTING build_mesh/build_constraints logic applied to the
     chunk+halo subtree, producing chunk-local nodes, connectivity, and
     constraint rows (incl. hanging + p-interface + chain closure — all
     resolvable inside the halo by the locality theorem).

3. **Boundary stitcher** — dedups nodes on chunk interfaces ONLY (the
   halo-shared faces): canonical global node ids via the node's integer
   coordinate key (`icoords`), so chunk-local ids map to global ids with a
   per-interface sorted-merge — never a global `np.unique` over all nodes.
   Output arrays are written append-only per chunk into preallocated (or
   grown-in-blocks) global buffers.

4. **Streaming 2:1 balance** (pre-pass, before chunking): balance ripples can
   cross chunk boundaries, so balance runs as an iterate-to-stable sweep over
   the leaf array with a frontier queue — the standard distributed-AMR
   pattern, single-node here. Working set = the frontier, not the tree.
   (The refine loop itself already iterates; the memory question is its
   intermediates — see probe.)

5. **Remeshing hook (design-for, do-not-build):** the chunk planner takes an
   optional dirty-range list; unchanged chunks' outputs are reusable
   verbatim in a future epoch-incremental rebuild. WP-scope here is ONLY to
   keep the chunk outputs individually addressable (stable chunk manifest) so
   incremental reuse is possible later without re-architecture.

### Mixed p1/p2

The (level, p) owner rule and p-interface constraints are neighbor-local; the
chunk producer's general path handles mixed-p chunks with the existing M0.5
machinery (incl. transitive closure, chain ≤ 2 — inside the halo). The
closed-form fast path applies per-(level,p)-uniform chunk; a p2-uniform chunk
uses the p2 closed-form lattice. Chunk planning treats a p-interface exactly
like a level interface (general path).

## Memory contract

- Peak host RSS ≤ `output_arrays + max_chunk_working_set + O(leaf_array)`.
- Output arrays at 100M DOF (25M nodes, p1 hex): coords+conn+constraints
  ≈ 20–40 GB (dense unavoidable payload) — well under 150 GB.
- Acceptance: measured ladder (see gates) demonstrating ≤ 1.5 GB/M-DOF
  with margin, extrapolating under the WP1 budget at 100M.

## Gates (binding)

1. **Identity:** streaming build ≡ current builder (node icoords,
   connectivity, boundary sets, constraint matrix — bit-identical after
   canonical node ordering) on: uniform 2-D/3-D, adaptive plate-band 2-D/3-D,
   mixed p1/p2 (the M0.5 fuzz configs), periodic variants. The existing
   builder is the oracle.
2. **Memory ladder:** per-stage RSS at 3+ sizes vs the current builder —
   the reduction curve recorded; the biggest current-builder stage (per the
   probe) must show the designed collapse.
3. **No regression:** full existing mesh/constraint test suite green;
   the closed-form fast-path tests remain green (the generator is reused,
   not modified).
4. **Chunk-size invariance:** results identical across `chunk_max_cells`
   choices (the streaming must not leak discretization into the output).

## Out of scope

On-the-fly/implicit consumers (cuFEM-native, later); GPU-side build; the
incremental-remeshing implementation itself (only the manifest hook);
distributed multi-node build.

## Open dependency

The probe's stage attribution decides WP ordering within the plan (e.g. if
`build_mesh`'s global dedup dominates, the stitcher+producer land first and
the balance pre-pass may initially remain the current implementation if its
memory is acceptable at target scale).
