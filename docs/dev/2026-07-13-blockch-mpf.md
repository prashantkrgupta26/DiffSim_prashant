# B-track: blockch for the multiphase (M, K) system — measured

Date: 2026-07-13.  Base: the G1/G4/G5 blockch design laws
(docs/dev/2026-07-09-blockch-preconditioner.md) + the M5 device
assembly (docs/dev/2026-07-13-m5-device-assembly.md).  Target: the
binding constraint for 3-D M5 production — cuDSS's measured 48 GB
factorization ceiling at 812k dofs; Baskar's explicit 128x128x64 film
ambition (~6.4M dofs at ndof = 6) plus the B5 (M=3, K=2) capacity
question (256x256x128, 84.5M dofs).

## 1. Design (what shipped)

`linsolver="blockch" | "blockch_dev"` on `MultiPhaseStepper`, both
assembly modes:

- **M CH pairs** ride the G4 two-factor recipe unchanged
  (`_blockch_pairs` meta `pairs`): per pair (phi_i, mu_i) at offsets
  (2i, 2i+1), W1/W2 from the extracted Acc/Acm/Amc/Amm with the SIGNED
  curvature (laws 1-4).  The pair scalar m_i is the committed-mean
  mirror of the kernel mobility diagonal (`_mob_ref`: const/fastmode/
  fastmode_n/slowmode_n incl. the Eq. 13 drop and D(T) Arrhenius) —
  the wodo precedent (constant reference M11/M22 under var_mob); the
  TRUE variable mobility rides in through the extracted blocks.
  kappa_i from the constructor.  sigma is attempt-frozen by
  `_attempt_ctx` (BDF1 sigma = 1/dt; BDF2 sigma = a/dt — the recipe
  needs no other change).
- **K AC (psi, theta) blocks** (meta `ac`, new): extracted
  diagonal-block solves, lower-triangular within the pair —
  z_s = Ass^{-1} r_s; z_t = Att^{-1}(r_t - Ats z_s).  Ass is
  mass-dominated at production dt (sigma M + L_psi(f'' M + eps2 K) +
  film advection: possibly indefinite/nonsymmetric -> GMRES inner on
  host, BiCGStab on device — NOT CG).  Att: frozen theta = sigma M
  (+ marker advection) — a consistent-mass solve, exact at the inner
  tolerance; KWC = the (p+pf)-weighted SPD row.  Ats is the KWC
  p'(psi) torque column (empty/absent under frozen theta — handled).
  The phi/mu <-> psi couplings are DROPPED in the preconditioner; the
  outer FGMRES carries them.
- **Outer**: FGMRES (lgmres) over the full system; escalation on
  stall = per-pair exact Schur (G4 semantics; AC blocks are already
  exact-block solves).  Failure of both signals divergence to the
  Appendix-A reject ladder (nan contract, matching cudss).
- **Device-resident** (`assembly="device"`): `blockch_pairs_device`
  extended with `ac` — value GATHERS into ss/ts/tt buffers on the
  shared node pattern (no host matrix), device BiCGStab inners; the
  full-A device index pair is REUSED from the assembler
  (`idx_dev=asm._op_idx` — avoids a duplicate 4-8 GB indices copy at
  B4/B5 sizes).
- **Zero-rhs inner guard** (measured necessity): frozen-theta rows
  carry an exactly-zero residual; the device BiCGStab 0/0-breaks down
  (relres nan at 4000 its).  All device inner helpers short-circuit
  `not np.any(y)` to the exact zero solution (scipy inners already
  do).
- **B5-i block-masked pattern**: `DeviceNSAssembler(...,
  blockmask=...)` builds pattern = kron(G, mask) in closed form
  (masked indptr/indices/scatter kernels; node-graph mode only; the
  diagonal is forced live for strong rows).
  `MultiPhaseStepper(block_sparse=True)` supplies the compile-time
  (M, K, mob, theta) live-block mask (`_block_mask` — exactly the
  blocks the kernel/face terms write).  MEASURED exactness (film
  config, L5 2-D): masked nnz ratio 0.611 (22/36 live), rel max|dA|
  3.4e-16 vs the superset values, |dr| 5.6e-17.  Opt-in; the superset
  stays the gated default.

## 2. B1 — host-class correctness (gates: tests/test_multiphase_blockprecond.py)

One-solve parity on the identical assembled film system (L5, S3b-class
(M=2, K=1) film + FDT noise, frozen theta): rel |dx| 3.1e-11 vs splu,
relres 4.5e-11, outer its 3.

Measured march panel (L5 2-D, 10 steps splu vs blockch HOST inners,
same seeds; parity = max over the ndof field slices of rel max|dx|;
walls are host-inner walls — B2 is the performance path):

| config | parity | rej s/b | solves | outer max/mean | fallbacks | inner tot | wall splu/blockch |
|---|---|---|---|---|---|---|---|
| film (M2, K1) frozen + FDT noise, 10 steps | 5.3e-14 | 0/0 | 60 | 4 / 3.0 | 0 | 289,530 | 22.5 / 19.9 s |
| quench16390 (M2, K2, fig6 pars) frozen + noise, 10 steps | 2.3e-14 | 0/0 | 32 | 33 / 15.9 | 0 | 3,552,418 | 29.5 / 253 s |
| kwc_grains (M2, K2 + KWC theta, 15 crystallites), 6 steps | KWC-ROW | | | | | | |
| bdf2 film deterministic, 10 steps | 3.0e-14 | 0/0 | 38 | 4 / 3.0 | 0 | 116,403 | 5.5 / 8.6 s |
| fig4 annealing (M1, K1), 10 steps | 3.4e-14 | 0/0 | 31 | 4 / 3.0 | 0 | 77,764 | 3.7 / 7.3 s |
| deepq chi12=3 (M2, K0, marginal, M12=-0.2), 6 steps | 1.2e-12 | 0/0 | 20 | 5 / 3.5 | 0 | 214,656 | 2.4 / 17.8 s |

Readings so far:
- Production-class configs (film): outer <= 4, zero fallbacks, parity
  machine-class — the G1/G4 iteration flatness carries to (M, K).
- The 16390 quench ONSET works the outer harder (max 33, mean 16, no
  escalation): the dropped d12-class cross curvature + phi-psi
  couplings are marginal-but-convergent there.  Host-inner wall is
  8.6x splu at L5 — expected (2-D is splu/cudss territory; the
  preconditioner exists for the 3-D sizes where direct solvers die).
- MEASURED TRAP (recorded as a module-docstring warning in the
  gates): an IC whose retained fractions sum to 1 pins the eliminated
  solvent at the log regularization floor — the 1/phi_s CROSS
  curvature (1e4-class) lands in the dropped d12 block and the
  two-factor form stalls into the exact-Schur escalation (> 20 min at
  L5 before the run was killed).  Every production config keeps a
  trace fraction; the S2 gates use 2%.

Deep-quench boundary carry-over (G4), measured at (M, K): the
chi12 = 6.0 config (M=2, K=0; the d12-driven quench where per-pair
preconditioning is structurally blind) marched blockch-only through
the reject-ladder contract exactly as the G4 sigma-scaling predicts —
3 rejects, dt collapse 5e-3 -> 1.53e-4 (the d12/(2 sqrt(kap sigma))
~ 1 threshold), then ACCEPTED steps at outer its within budget, ZERO
exact-Schur fallbacks, t advanced to 3.0e-4 in 6 ladder steps.  The
boundary carries over 1:1; the failure mode is dt-penalty (not crash),
matching cudss divergence semantics.  Gate:
test_mpf_blockch_deepq_boundary_ladder.

## 3. B2 — device-resident path

TABLE-B2

## 4. B3 — AMGX verdict: measured NO for the inners, NO for the raw J

AMGX IS available on this box (extern/libamgxsh.so, sm_89 build from
the G-ladder era; pyamgx installed in the venv; loads via
solvers/amgx.py `_preload_libamgx`).  Evaluation on the SAME iterate
the production solver sees (3d_l5 film bench system, 202,752 dofs,
nnz 32.2M, sigma = 1e5, device assembly; scratchpad/b3_amgx.py; RTX
6000 Ada):

| target | production inner | AMGX (repo classical-AMG configs) |
|---|---|---|
| W1 (SPD) tol 1e-8 | Jacobi-cg_dev: **50 its, 0.013-0.030 s** (setup 4 ms) | PCG-AMG: 1626 its, ~1.2 s (setup 41 ms) |
| W2 (signed F) tol 1e-8 | Jacobi-bicgstab_dev: **50 its, 0.042-0.069 s** | BiCGStab-AMG: 2260 its, ~4.5 s |
| mass tol 1e-10 | Jacobi-cg_dev: **50 its, 0.020-0.039 s** | PCG-AMG: setup **21.4 s**, then 1 it / 0.015 s |
| raw Jacobian tol 1e-10 | (blockch outer: 1-4 its) | BiCGStab-AMG: **NOT CONVERGED**, relres 6.3e+03 after 1000 its / 9.6 s |

Readings:
- **Raw-Jacobian AMG fails as predicted** (saddle-class rows + FH log
  f''): residual grows 3 decades.  The honest-framing claim is
  demonstrated cheaply.
- **AMG on the W factors is the right TARGET but the wrong TOOL at
  production dt**: W1/W2/mass are MASS-DOMINATED (sigma = a/dt =
  1e4-1e5 against eps2/kappa-scaled stiffness), so Jacobi-Krylov is
  spectrally flat (50 its at 203k dofs — the same 50-its class as the
  2-D gates) and there is no O(h^-1) growth for AMG to flatten.
  Classical-AMG strength-of-connection degenerates on these operators
  (1626-2260 its = worse than plain Jacobi; the consistent-mass AMG
  setup is pathological at 21 s).  cuDSS-class direct factorization
  is not the comparison here — the inners never were the bottleneck.
- Verdict: **device Jacobi-Krylov inners stay**; AMGX adds no value
  for fully-device blockch operation at production parameters.  The
  one regime where an AMG-class W2 preconditioner could matter is the
  recorded G3 open item (FH quench-onset W2 at 3-D scale, f'' -> 1e4
  with sigma small after ladder collapse) — out of the production
  contract, unresolved by these default configs, and the G4-era tuned
  configs (scratchpad amgx_tune.py) were already measured weaker than
  the fused stack on SBM systems.  Nova needs no AMGX install for
  this pipeline.

## 5. B4 — the 128x128x64 target

TABLE-B4

## 6. B5 — (M=3, K=2) capacity study (ndof = 10)

Size arithmetic (27-point node graph; mask = 54/100 live blocks at
fastmode_n + frozen theta — computed by `_block_mask`, exactness
gated):

| rung | nodes | dofs | nnz superset | nnz masked | CSR GB (vals+idx) |
|---|---|---|---|---|---|
| 64x64x32 | 135,168 | 1.35M | 0.36B | 0.20B | 2.4 |
| 128x128x48 | 802,816 | 8.03M | 2.17B **int32 OVERFLOW** | 1.17B | 14.1 |
| 128x128x64 | 1,064,960 | 10.6M | 2.88B overflow | 1.55B | 18.6 |
| 128x128x88 | 1,458,176 | 14.6M | 3.94B overflow | 2.13B (99% of int32) | 25.5 |
| 256x256x128 (Nova ambition) | 8,454,144 | 84.5M | 22.8B | **12.3B — int32 OVERFLOW; ~148 GB CSR** | — |

Standing facts this table forces:
- The mk32 128-class rungs REQUIRE the block-masked pattern (the
  superset overflows the int32 slot arithmetic at 128x128x48
  already); the masked int32 ceiling is ~1.47M nodes (128x128x88 sits
  at 99% of it — the assert raises loudly beyond).
- 256x256x128 at ndof = 10 cannot exist as a stored CSR on ANY single
  card (masked values+indices ~148 GB > A100-80): the stored-matrix
  road ends at ~1.4M nodes (48 GB, this box, int32) / ~2-3M nodes
  (80 GB with an int64 kernel variant).  The honest paths beyond:
  (i) MATRIX-FREE OUTER — J.v batch-wise through the device-assembly
  element kernels (compute Ae per batch, apply Ae @ v_e, scatter;
  never store the global CSR).  Cost basis = the MEASURED assembly
  fill (D3: 1.4e-6 s/element at ndof 6; the mk32 ladder measures the
  ndof-10 constant): one J.v at 256x256x128 (8.39M elements) is an
  assembly-pass-class cost, and an FGMRES(30)-preconditioned Newton
  solve needs ~30-100 J.v per iterate — VIABILITY-ASSESSMENT-B5
  (filled from the measured mk32 asm s/call).  The W-factor inners
  can stay stored (pair blocks are node-pattern-sized: ~1.8 GB/pair
  at full res — they fit).
  (ii) multi-GPU domain decomposition (out of scope here).

TABLE-B5 (measured ladder)
