# M0 Deferred Findings (input for M1 planning)

Final whole-branch review verdict: READY TO MERGE (37/37 tests). All items below are
deferred, non-blocking; address opportunistically in M1 tasks that touch them.

## Architecture notes M1 must internalize
1. **Host/device `.numpy()` boundaries** (matvec_numpy, Newton/Bratu action pattern) —
   the Warp tape cannot see through them; M1 differentiability work must route around.
2. **ConstrainedOperator scratch buffers (x_full/y_full) are not reentrant** — document
   or make per-call before any GPU-async/streamed execution.
3. **_kernel_cache keyed by (name, nbf, nqp)** — include an integrand tag in the key
   when M1 adds operators sharing (nbf, nqp).
4. **build_adaptive passes h as (N,1)** to refine_fn (broadcast-ready); document in
   signature for M1 SBM refinement predicates.

## Deferred minors
- Frozen dataclasses don't freeze numpy contents (consider setflags(write=False)).
- lookup.py probe overshoot at coarse size<=2 (levels ~LMAX-1; unreachable in M0) — guard when k-ring lands (M4).
- bicgstab: add rho / (rhat.v) breakdown guards when M1 stresses nonsymmetric systems.
- Formalize the operator protocol (.matvec/.n_free/.device) as a Protocol/ABC.
- blas.scale/copy in spec interface but unimplemented (unused).
- operator_diagonal re-assembles CSR (perf); dot() sync-per-call (perf, GPU-relevant).
- Unused kernel args (Ntab in poisson_mv, fe.e) — spec-mandated Integrands scaffolding, keep.
- Type annotations missing on build.py/gauss_1d/lagrange_1d; assorted dead test helpers/comments.
- cg tol semantics at bnorm~0 (atol floor governs — intentional; document).
