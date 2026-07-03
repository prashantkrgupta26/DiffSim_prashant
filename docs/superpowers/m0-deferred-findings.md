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

## M0.5 deferred findings (final whole-branch review, 2026-07-03 — carry to M1)

1. **Dead code removal:** `fe_dN` / `fe_detJxW` in `src/diffsim/assembly/femelm.py` are superseded by `fe_dN_s`/`fe_detJxW_s` and have zero callers (survive only via `assembly/__init__.py` re-export and an unused import in `operators.py`). Remove in M1.
2. **p_elem guard:** `assert set(np.unique(p_elem)) <= {1, 2}` in `src/diffsim/mesh/nodes.py` is user-facing input validation — convert to `ValueError` (parity with the one-knob validator; bare asserts vanish under `python -O`).
3. **Uniform-mesh-only docstrings:** `integrate_volume` (operators.py) and `BratuProblem` (bratu.py) lack an explicit "uniform mesh only" docstring line; also note the deliberate back-compat property asymmetry (`dm.tables` no-assert vs `conn/h/N/dN/w` assert) in the DeviceMesh class docstring. A mixed-mesh BratuProblem constructs silently and fails only at first `dm.conn` access.
4. **(Informational)** m05 baseline mismatch error message is a raw `(errs, ref)` tuple — fine for debugging, could be formatted.
