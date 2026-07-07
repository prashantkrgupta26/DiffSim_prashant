# Blueprint Chapter: Device Assembly, Constraints, and the Migration Rule

*For cuFEM (CUDA C++). Every design here is validated in the DiffSim
prototype with measured numbers (2026-07-06; RTX 6000 Ada, CUDA 12.9,
cuDSS via nvmath). Source: src/diffsim/assembly/device_assembly.py,
benchmarks/profile_stages.py, tests/test_device_assembly.py.*

## 1. The migration rule: profile first, migrate two things

Measured per-step profile (NS, 2-D L6-L8 + 3-D L4-L5) BEFORE migration:

| stage | 3-D L5 | verdict |
|---|---|---|
| direct factorization (host splu) | 686 s | replace with cuDSS (2.3 s) — 300x, zero code |
| numeric assembly (host COO->CSR) | 2.4 s | MIGRATE (this chapter) |
| strong-row LIL surgery | 1.4 s | MIGRATE (fold into assembly) |
| constraints build (vectorized) | 0.23 s | do NOT migrate |
| GP field eval / interp | 0.02 s | do NOT migrate (until it shows) |
| adjoint cotangent sweeps | 0.01 s | do NOT migrate |

Rule: the element-kernel work was ALREADY device-side; the host cost was
the values round-trip + scatter. Migrate the scatter, keep the symbolic
work host-side (it runs once per mesh epoch).

## 2. Slot-map scatter (the core design)

Per mesh epoch (host, once):
1. Build the CSR pattern from element connectivity (include every
   (row,col) block pair). Sort indices.
2. SLOT MAP: for each element e and local pair (a,b), the index into the
   CSR values array. Vectorized host build: give the pattern matrix
   data = arange(nnz) and fancy-index K2[rows, cols] — 2.9 s at 3-D L5
   (a per-entry loop was 56 s; do not write one).
3. Upload slot maps (int32) once.

Per step (device):
- element kernel fills Ae blocks [ne, (nbf*ndof)^2] (already device)
- scatter kernel: atomicAdd(vals[slot[i]], Ae_flat[i])
- rhs scatter: atomicAdd(F[gdof[i]], be_flat[i])

Measured: 4-40x vs host scatter (40x 2-D L6, 30x 3-D L5, 14x 2-D L8);
symbolic cost amortizes in ~20 assemblies (one Picard loop).

**Coloring verdict: do not bother.** Greedy element coloring (atomic-
free, per-color launches) was prototyped and benchmarked — never
consistently faster than atomicAdd slot maps (FP64 atomics on modern
NVIDIA are cheap; extra launches are not). Keep the simple variant.
If bit-reproducibility across runs is ever REQUIRED, coloring is the
fallback — it is deterministic by construction.

## 3. Constraint-aware scatter (hanging + p-interface constraints)

Do NOT assemble full then project (T^T K T on host was the old cost).
Expand element entries THROUGH the constraints at slot-map build time:
- node n -> masters {m_i} with weights {w_i} (from the constraint T;
  <= 4 masters per node in 3-D p1, more for p2 chains — resolve chains
  to free nodes first, as the constraint builder already does)
- element pair (a,b) -> all (master_ra, master_cb) pairs with weight
  w_ra*w_cb; slots point into the CONSTRAINED pattern
- device kernel: atomicAdd(vals[slot[i]], w[i] * Ae_flat[src[i]])
  (src = which element-pair value feeds this expanded entry)

Identity-constraint meshes keep the fast path (no weights). Gate used:
device constrained assembly == host T^T K T at 1e-12 on an adapted
mesh with hanging nodes. This is the cuFEM constraint-elimination
design; the prototype validates it end-to-end.

## 4. Strong rows (Dirichlet + pressure pin)

Precompute per epoch: slot spans (indptr[r]:indptr[r+1]) of the strong
rows + the diagonal slot per row. Per step, after scatter: one kernel
zeroes the spans, one writes diag=1 and F[r]=g(t). No row surgery, no
format conversion. (Symmetric elimination — zeroing the COLUMNS too and
moving g to the rhs — changes the pattern treatment; if cuFEM wants
symmetric systems, fold the column part into the expansion weights.)

## 5. Device-resident solve chain

Keep values on device end-to-end: the scatter buffer is exposed
zero-copy (dlpack in the prototype; a raw pointer + CSR descriptor in
C++) and handed to cuDSS directly. Measured: agreement 1e-15 with the
host path; the host round-trip was ~25% of solve time at 2-D L7.
Refactorize per step (pattern fixed): cuDSS handles this efficiently;
reuse the analysis phase.

## 6. What this buys at step level

Linearized NS stepper, per step, 2-D L7 (host path -> device path):
assembly 110 ms -> 3 ms; strong rows ~800 ms LIL -> in-kernel; solve
unchanged (cuDSS). The step becomes SOLVER-BOUND — further speedup is a
solver/preconditioner question (see the blocktri note: FGMRES +
exact-F block preconditioner = 2-3 iterations at sigma=0).

## 7. Adjoint-readiness (do not lose it)

The taped/adjoint kernels consume the SAME element tables and GP data
as the forward kernels. Two rules from hard-won prototype findings:
(1) NO struct-field mutation inside differentiated kernels (silent zero
gradients — warp; C++ AD frameworks have analogous aliasing traps);
(2) keep every interpolation/transfer/constraint operator EXPLICIT
(sparse object or slot map) — transposing an explicit operator is
mechanical; differentiating a fused-and-discarded one is impossible.
See the M3 adjoint-readiness memo for the adaptivity-specific
requirements.
