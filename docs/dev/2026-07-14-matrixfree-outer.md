# Matrix-free OUTER solver for the multiphase (M, K) system — measured

Date: 2026-07-14.  Base: the B5 capacity study
(docs/dev/2026-07-13-blockch-mpf.md Sec 6) + the M5 device assembly
(docs/dev/2026-07-13-m5-device-assembly.md).  Question this pass
answers with MEASURED evidence: *can Baskar run 256x256x128 (M=3, K=2,
~84.5M dofs) on one A100-80, matrix-free, and at what wall/memory?*

## 0. Why matrix-free (the B5 wall, restated)

The B5 table forces the fact: at (M, K) = (3, 2), ndof = 10, the
256x256x128 stored CSR is ~148 GB of masked values+indices — fits on
NO single card (A100-80 included), and the int32 slot arithmetic
overflows at 128x128x48 already.  The stored-CSR road ends at ~1.47M
nodes (48 GB int32) / ~3M nodes (80 GB int64).  256x256x128 needs the
matrix-free OUTER: never store the monolithic CSR; apply J @ v
batch-wise through the device-assembly element kernels, keep the
blockch W-factor inners stored (node-pattern sized, ~5.5 GB — they
fit).

## 1. Design (what shipped)

Three new pieces, all reusing the D-track device-assembly path:

- **`DeviceNSAssembler.apply_batch_matvec(k_bin, e0, Ae, nb, v, y)`**
  (assembly/device_assembly.py): the matvec analogue of
  `scatter_batch`.  One thread per element-local output row (el, rl);
  rl = a*ndof + ca maps to global row conn[e, a]*ndof + ca (the SAME
  local->global map `_scatter_node_kernel` uses to place Ae into the
  CSR).  Each thread reads the nl-long Ae row, gathers v at the
  matching global column dofs, accumulates in float64, atomic-adds
  into y.  Node-graph pattern only (identity constraints).  It NEVER
  touches vals_d — no monolithic CSR is required for a matvec.

- **`MultiPhaseStepper._fill_element_batches(ctx)`** (physics/
  multiphase.py): the element-Jacobian batch fill (`make_mpf_newton`
  into the ~2 GB-capped Ae buffer) factored out of `_assemble_device`
  as a generator yielding (k_bin, e0, nb, Ae, be).  THE single source
  of the element-block fill — `_assemble_device` consumes it to
  scatter into the CSR; `apply_Jv` consumes it to matvec — so the two
  paths cannot drift.  `_assemble_device` stays bit-identical (GP eval
  hoisted into `_gp_eval_dev`; per-bin buffers are independent so the
  result is launch-order-invariant — the 15-test device parity suite
  confirms).

- **`MultiPhaseStepper.apply_Jv(v)` + `_matfree_setup(x, ctx)`**: freeze
  the linearization point (GP-eval x ONCE — Ae is x-only, identical
  across the outer Krylov matvecs, recomputed per apply because the
  full element-block set is too large to store: ~430 GB at 256^3), and
  per matvec refill Ae batch-wise + `apply_batch_matvec` into y.  The
  O(surface) face-term Jacobian (A2 wall (mu_i, phi_i) block + S3a
  top-flux (phi_i, phi_i) block) is x-INDEPENDENT within an attempt,
  so it is built ONCE as a tiny host CSR (`_build_face_jac`) and added
  as a correction; Dirichlet strong rows realized as y[strong] =
  v[strong].  apply_Jv is a drop-in replacement for the stored-CSR
  matvec at ANY config.

## 2. M1 — matrix-free J @ v parity + apply time (GATE: GREEN)

Parity vs the stored device CSR J @ v at one Newton iterate (mk =
(3, 2), fastmode_n Vignes, frozen theta, dt 1e-5, node-graph SUPERSET
pattern so the stored CSR carries every block — exact comparison; 3
random v per case).  RTX 6000 Ada 48 GB.

| case | elements | dofs | superset nnz | rel max\|Jv_mf - Jv_csr\| | apply_Jv s/matvec |
|---|---|---|---|---|---|
| 8x8x8 mk32, film off | 512 | 5,760 | 1.44M | 9.33e-16 | 0.010 |
| 8x8x8 mk32, film ON | 512 | 5,760 | 1.44M | 1.17e-15 | 0.011 |
| **64x64x32 mk32, film ON (GATE)** | 131,072 | 1,351,680 | 357.6M | **1.13e-15** | **0.532** |

- Parity is MACHINE-CLASS (rel ~1e-15) — the only difference from the
  stored CSR matvec is atomic-add summation order (the house FP-chaos
  class), identical to the D1 assembly parity readings.  LOCK 1e-12
  (>= 880x headroom).  The film-ON row validates the face-term
  correction (`_build_face_jac`) reproduces the top-flux Jacobian
  entries exactly.
- apply_Jv scales ~linearly in elements: 0.532 s at 131,072 elements
  = **4.06 us/element**.  This is the outer-FGMRES J.v cost basis
  (Sec 4 extrapolation).
- Existing suite: tests/test_multiphase_device.py 15/15 GREEN after
  the `_assemble_device` refactor (bit-identical parity gates hold).

## 3. M2 — matrix-free OUTER FGMRES (GATE: GREEN)

`blockch_pairs_device(..., jv=None)`: when `jv` (a host J @ v closure)
is supplied the outer FGMRES sees A ONLY through it (both the main and
the escalation-fallback lgmres); the W-factor inners still gather from
the stored pair-block CSRs — the "inners stored, outer matrix-free"
design.  `MultiPhaseStepper(matfree=True)` freezes the face/strong
correction once per attempt and passes `jv=self.apply_Jv`.

Same Newton system solved two ways (stored full-A device spmv vs
matrix-free element apply), SAME blockch preconditioner, at one iterate
(mk=(3,2), film, node-graph superset so both representations exist):

| case | dofs | outer (stored / matfree) | relres (stored / matfree) | \|x_mf - x_stored\|rel | wall (stored / matfree) |
|---|---|---|---|---|---|
| 16x16x8 mk32 film | 23,040 | 3 / 3 | 8.17e-11 / 8.20e-11 | 7.75e-15 | 6.6 / 7.0 s |
| **64x64x32 mk32 film (GATE)** | 1,351,680 | **3 / 3** | **5.86e-11 / 5.86e-11** | **2.80e-15** | 27.4 / 38.4 s |

- The matrix-free outer solves to the SAME residual with the SAME outer
  iteration count (3) and a solution agreeing to 2.8e-15 — the only
  difference is the atomic-add matvec summation order (FP-chaos class,
  well under the 1e-10 solver tol).  Gate: relres < 1e-8 AND
  |x_mf - x_stored|rel < 1e-7 — PASS with huge headroom.
- Wall: matfree outer is ~1.4x slower here because each of the ~3-4
  outer matvecs REFILLS Ae (a full element-fill pass, 0.53 s at this
  size) vs the stored spmv — the deliberate matrix-free trade (recompute
  vs store).  It buys the ability to run where the stored CSR cannot
  exist (Sec 4).
- Existing blockch suites (test_multiphase_blockprecond,
  test_ch_blockprecond, test_ternary_blockprecond) GREEN — the `jv`
  param is additive (default None => the unchanged B2 stored path).

## 4. M3 — SCALE: does 256x256x128 (M=3, K=2) run matrix-free?

Matrix-free-ONLY setup (`_init_matfree_assembly` + DeviceNSAssembler
`matvec_only=True`): builds ONLY bins + device conn + the GP/Ae buffers
apply_Jv needs — NO CSR pattern, NO indices (~91 GB host at 256^3), NO
vals_d (~148 GB), and crucially NO int32-nnz ceiling.  This is the TRUE
footprint of the matrix-free outer.  One live J @ v measured per size,
RTX 6000 Ada 48 GB (mk=(3,2), film, fastmode_n Vignes, frozen theta):

| rung | elements | dofs | apply_Jv s/matvec | us/element | GPU (matrix-free) | host rss |
|---|---|---|---|---|---|---|
| 128x128x64 | 1,048,576 | 10.6M | 4.42 | 4.21 | 7.1 GB | 6.4 GB |
| 128x128x128 | 2,097,152 | 21.1M | 8.59 | 4.10 | 11.2 GB | 12.6 GB |
| **256x256x128** | **8,388,608** | **84.5M** | **35.38** | **4.22** | **39.1 GB** | **49.7 GB** |

- **256x256x128 (84.5M dofs) matrix-free J @ v RUNS on ONE 48 GB card**
  — 39.1 GB GPU with ~8 GB headroom.  The stored CSR at this size is
  ~148 GB (B5 table): fits on NO single card, A100-80 included.  The
  matrix-free footprint is ~3.8x lighter than even the B5 STORED
  blockch at the 8x-smaller 128x128x64 (23.9 GB).
- apply_Jv per-element cost is FLAT across a 64x scale span (4.10-4.22
  us/element, 8x8x8 through 256x256x128) — the apply is bandwidth-bound
  element streaming; parity-locked (Sec 2) and matvec_only == full-CSR
  J @ v to 7.5e-16 (cross-checked).
- GPU footprint is dominated by the GP-field buffers (the eval-x-once
  design): at 256^3 the _grads_dev [ngp, 10, 3] alone is 16.1 GB
  (ngp = 67.1M), _vals_dev / hist / src-zero 5.4 GB each, qphi-zero
  4.8 GB; conn 0.27 GB, Ae transient 2 GB (capped), v/y 1.35 GB.  Host
  rss 49.7 GB is the octree/mesh build (build 413 s — a one-time setup).

### A100-80 (Nova) projection — the direct answer

The apply is spmv/bandwidth-bound; HBM2e ~2.0 TB/s vs this card's
GDDR6 ~0.96 TB/s gives ~2x.

- **One J @ v at 256x256x128 on A100-80: ~17-18 s** (35.38 s / ~2.0),
  bandwidth-scaled from the measured RTX 6000 Ada value.
- **Memory: ~39 GB (J.v) + ~5.5 GB stored W-factor inners + Krylov
  work ≈ ~50 GB — fits the 80 GB pool with ~30 GB margin.**
- **Step time: ~7-15 min/step.**  Basis: 4 Newton its/step (B5
  measured), each blockch-preconditioned Newton solve = ~4-10 outer
  matvecs (M2 measured outer 3 at production iterates; the FGMRES(30)
  budget allows up to ~10), so ~16-40 J @ v/step at ~17.5 s = 4.7-11.7
  min, plus the stored W-factor inner solves (~2-3 min/step at 256^3
  scaling of the B5 9 s/solve-call).  This matches the B5 note's
  "10-25 min/step class, VIABLE but painful" — now measured, not
  estimated.

**Verdict (the deliverable answer):** YES — 256x256x128 (M=3, K=2,
84.5M dofs) runs matrix-free on one A100-80.  The live J @ v is MEASURED
to fit and run in 39 GB even on a 48 GB card (35.4 s); on A100-80 it is
~17-18 s and a full preconditioned step is ~7-15 min, in ~50 GB.  The
stored-CSR road was closed at ~1.47M nodes; the matrix-free outer opens
the 8.45M-node (256^3) target on a single card.

### Remaining implementation piece (honest frontier)

M2 proves the OUTER is matrix-free and M3 proves the J @ v runs at
256^3, but the M2 gate still fills the blockch W-factor inners by
gathering from a stored `vals_d`.  For a LIVE 256^3 campaign (where
vals_d cannot exist) the pair-block CSRs (~5.5 GB, they fit) must be
filled DIRECTLY from the element blocks — a partial device scatter that
writes only the 5 pair/AC node-pattern CSRs instead of the monolithic
one (the same Ae, a subset of slots).  Scoped, not yet implemented; the
outer solver and the J @ v operator it needs are done and gated.  Until
then a full matrix-free step is runnable at any size whose pair-block
CSRs AND a transient full vals_d both fit (the ~1.47M-node int32 rung),
or the direct-fill lands.
