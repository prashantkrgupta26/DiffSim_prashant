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
