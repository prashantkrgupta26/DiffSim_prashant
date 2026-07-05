# M1b Deferred Findings (running log)

1. **tau's transient term floors temporal-convergence studies (Task 5b,
   measured).** With timeStab on, tau_M carries (2 b0/dt)^2, so each dt in a
   ladder solves a slightly DIFFERENT spatially-stabilized problem; the
   ladder floored at ~2.4e-3 (rates 0.79 then -0.09) on the level-4 vortex
   MMS. With the production timeStab toggle off in the ladder runs: clean
   order 2. Rule: temporal-order gates run timestab=False; physics runs keep
   the production default (on). This is presumably WHY production carries
   the toggle — record in the verification-tier docs.

2. **WSL2 idle-queue syncs are cheap; the pathology needs queued load (Task
   1, measured).** A 27-iteration 15 ms CG microbench showed host-sync ~=
   fused (syncs ~0.1 ms on an empty queue); at 200k DOFs the fused path is
   4.6x per-iteration (0.321 vs 1.485 ms). Speedup assertions in CI must be
   per-iteration at meaningful size, or they flap.

3. **Warp closure-name typing:** `wp.float64(dim)` on a bare closure int
   coerces that name kernel-wide (breaks `range(dim)`); use a separately
   named float constant for exponents. (Cousin of m1a finding 4c — warp
   codegen care points now have their own section in the tutorial plan.)

4. **v1 deltas to close before the benchmark task (5c/8):** production
   fine-scale-corrected extrapolation (u_pre - tauM res_M_pre) not yet in
   the advecting field; strong-Dirichlet row surgery via tolil per step is
   O(n) host work — move to the precomputed-row masked-assembly pattern;
   splu per step -> matrix-free + fused BiCGStab (Task 1 machinery) once
   sizes demand.

5. **Leray stepper v1 (Task 6, all measured on the vortex MMS):** (a) the
   pressure INCREMENT trap: the PPE with RHS sigma(u_hat, grad q) solves for
   phi = p_hat - p*, not the total pressure — treating it as total
   double-counts p* every step (compounding blowup ~7e5). (b) Explicit
   fine-scale in the PPE RHS is unstable at sigma*tau_m ~ 0.8 (r_m contains
   sigma*u_hat); the draft keeps tau_m grad(p_hat) implicit. CLOSED same
   night: the (1/sigma + tau_m)-weighted PPE operator (phi-part of r_m on
   the LHS, RHS flux u_hat - tau_m r_m_expl with bounded coefficients) is
   STABLE and within 2x incremental accuracy on the ladder point (measured;
   test_leray_implicit_finescale_stable). Default stays ppe_finescale=False
   until the benchmark task picks per-case; the draft-faithful path exists
   and is gated.
   (c) Temporal ladder 1.86 -> 1.10: BDF2 regime decaying to the O(dt)
   splitting floor — classic incremental behavior; rotational correction /
   implicit fine scale raise the floor (benchmark task). (d) At
   splitting-dominated dt a second Picard pass does not reduce error —
   direction-asserting it was another degenerate-observable trap.
   (e) Head-to-head at level 4, n=16: Leray within 4x monolithic accuracy,
   divergence sentinel within 2x — the spec S17 both-steppers gate.

6. **3D NS element-kernel compile economics (Task 7, measured in flight):**
   the dim=3 lin_ns_Ae kernel (nbf=8, nqp=8, ndof=4 node-major blocks,
   quadruple-nested a/b/i/j unroll) is a >53-min one-time nvrtc compile —
   the M1a dim-4 pattern (finding 1b) recurring one dimension earlier
   because ndof multiplies the block. Disk-cached content-keyed afterward.
   Mitigations for the 3D benchmark task: warp max_unroll module option /
   rolled dof-loops for dim>=3 NS kernels; document the first-run cost in
   CI notes. 2D NS kernels compile in seconds.

7. **Immersed-cylinder Re=20 smoke config + measured drag (Task 8b):**
   [0,1]^2, cylinder r=0.07 at (0.3, 0.5) (blockage 14%), uniform inflow,
   no-slip walls, free outflow, level 5, lambda=0.5, BDF1 pseudo-time:
   steady in 50 steps, Cd = 2.847, Cl = -3e-5 (machine-symmetric wake).
   Confined-cylinder literature at this blockage ~2.0-2.8 — coarse-grid
   SBM lands at the top of the band; lock the MEASURED value in
   m1b_baselines at the benchmark task and compare against a wall-resolved
   reference there. Traction ORIENTATION CONTRACT (measured, was flipped):
   geo.n is domain-outward = INTO the obstacle; F_obstacle integrand is
   +p*geo.n - nu (S grad u).geo.n.

6b. **Finding 6 CLOSED: max_unroll=0 takes the dim-3 NS element kernel from
   >79 min to 1.3 s of compile** (decorator-level module_options on the
   unique module; measured end-to-end incl. first launch). RULE: every
   dim>=3 fat element kernel declares max_unroll=0 unless a measured hot
   path justifies unrolling. 3-D flow is unblocked.

7b. **Leray Newton predictor is INEXACT Newton (measured contraction
   ~0.2/iterate vs Picard's ~0.5)**: the (du.grad)a cross-block is
   Galerkin-only; SUPG/tau'/(div du)a linearizations stay Picard-level
   (standard practice). The draft's '1-2 iterations' presumes consistent
   linearization — recorded as the remaining delta. RHS partner (a.grad)a
   folds into f_eff so it inherits SUPG/PSPG consistency for free.

8. **Benchmark measurements (level-6 CI variants, 2026-07-05):** cavity
   Re=1000 max|du| vs Ghia = 0.0727 (65^2 grid, 900 pseudo-steps); cylinder
   Re=100 converges to Cd = 1.351 (unbounded literature ~1.33!) but a
   STEADY inflow tilt does not trigger shedding in 28 time units at this
   blockage/dissipation — a TRANSIENT kick run is the St config. Jacobi
   BiCGStab breaks down (NaN) on the 3-D SBM system at 356k DOFs — the
   measured case for the ASM/multigrid preconditioner item.

8. **GPU solver backends in the steppers (Task 5/8 follow-on; user-directed
   pyamgx + cuDSS adoption; all measured):** both steppers now dispatch
   solver in {splu, fused, cudss, amgx}; 11/11 parity matrix green.
   (a) Fused Jacobi-BiCGStab is UNCOMPETITIVE on the stabilized monolithic
   (u,p) block: ~36k iterations at cavity-L6 (~7 s/step vs splu 0.33 s) —
   Jacobi cannot precondition the saddle-point coupling; this is the
   measured case for AMG/direct GPU backends. The SPD sub-solves (PPE,
   mass) are fine fused.
   (b) bicgstab_dev hardening, three measured bugs: rho-breakdown must
   FREEZE the device state (guards in every update kernel) or NaNs churn
   until the periodic host check; breakdown thresholds must be relative to
   the CURRENT residual (scal[5]) not bnorm^2, and scal[5] must be SEEDED
   at bnorm^2 (zero disarms the guard); the host check must test
   convergence BEFORE breakdown (near convergence rho is legitimately
   tiny — 'breakdown' fired at relres 1e-27). Restart-on-breakdown
   (rhat <- r) with a cap replaces failure.
   (c) AMGX integration lessons: use the BUNDLED validated configs
   (src/configs/*.json; hand-rolled config dicts abort the process at
   Solver.create); AMGX objects are process-global SINGLETONS (multiple
   live Resources sets segfault); do NOT register pyamgx.finalize at
   atexit in a process holding warp/torch CUDA state — teardown-order
   SIGABRT after green tests (exit 134). Configs packaged in
   solvers/amgx_configs; libamgxsh.so staged in extern/ (gitignored),
   self-loaded via ctypes.
   (d) cuDSS (nvmath-python DirectSolver) = GPU splu: pip-installable,
   parity-exact, right default for per-step-new-matrix stepping at
   prototype scale; AMGX is the production/large-scale path (setup 3.7 ms
   + solve 1.2 ms on the probe; iteration counts flat in h by design).
   (e) MEASURED per-step timings, cavity monolithic stepper (RTX 6000 Ada):
        level    n        splu       cudss      amgx          fused
        L6       12675    308 ms     147 ms     FAIL(intern)  FAIL(stall)
        L7       49923    2684 ms    589 ms     FAIL(noconv)  FAIL
        L8       198147   16938 ms   2401 ms    FAIL(noconv)  FAIL
   cuDSS wins the monolithic block outright (7.1x over splu at L8) — the
   GPU-direct default. Classical AMG NOT converging on the coupled (u,p)
   VMS block is textbook (saddle-point structure needs block/physics-aware
   preconditioning, not blackbox AMG); AMGX's role here is the SPD
   subsystems (Leray PPE at scale) where its parity gates pass. OPEN
   (production path): (i) block-preconditioned monolithic solves — AMG on
   the velocity block + Schur approximation, via AMGX as preconditioner
   inside our fused Krylov; (ii) true matrix-free NS matvec to kill the
   per-step host CSR assembly (now the largest remaining host cost).
   (f) Block preconditioner v1 (AMGX-in-GMRES on the monolithic block):
   MEASURED NOT EFFECTIVE YET — recorded so nobody retraces it blind.
   Two probes on the step-4 cavity-L6 system (n=38k): (i) with solver-grade
   AMGX configs as the "preconditioner" each application ran a near-full
   solve — 19+ min vs cuDSS 147 ms (rule: preconditioner applications must
   be hard-capped cycles; _AMGXCycle now bakes max_iters into the config);
   (ii) with proper 1/3-cycle applications the cost moved to the outer
   GMRES: ~10k applications in 15 min WITHOUT convergence — the first-cut
   Cahouet-Chabard Schur (sigma Kp^-1 + nu Mp^-1) + single-V-cycle F does
   not cluster the spectrum of the PSPG/SUPG-stabilized block. Refinement
   candidates, in payoff order: fold the assembled PSPG block C into S~
   (our C IS tau-scaled Kp — relative scaling is the suspect); audit G's
   SUPG contribution sign/scale; 2-3 cycles on F; FGMRES instead of GMRES
   (the preconditioner is nonlinear across applications). CLASSIFIED as a
   research-grade follow-up with a proper study harness (iteration counts
   vs {Schur variant x cycles x Re x dt}); until it lands, the production
   position stands: cuDSS to the memory ceiling, AMGX for SPD subsystems.
   src/diffsim/solvers/block_precond.py kept as the experiment harness,
   marked EXPERIMENTAL, not in the dispatch.
