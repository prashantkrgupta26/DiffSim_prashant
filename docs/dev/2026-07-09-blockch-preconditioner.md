# Finding: the CH block preconditioner (G1) — design laws, measured

Date: 2026-07-09/10 (overnight). Status: G1 CLOSED — parity + iteration
flatness on Newton-converged regimes, zero-fallback robustness through
full adaptive marches, both bulk energies. Solver: `blockch` in
solvers/linsolve.py; gates: tests/test_ch_blockprecond.py.

## The system and the final design

Mixed CH Newton system J = [[sigma M, m K], [-(F + kap K), M]],
F = Int f''(c) N N (the bulk curvature, SIGN-INDEFINITE and, for the
Flory-Huggins log, reaching the regularization cap 1/eps = 1e4 on
Newton transients).

    S~ = W1 M^{-1} W2,   W1 = sqrt(sigma) M + sqrt(m kap) K
                          W2 = W1 + (m/sqrt(sigma)) F      (SIGNED)
    apply: a = W1^{-1}(r_c - mK M^{-1} r_mu); z_c = W2^{-1}(M a);
           z_mu = M^{-1}(r_mu + (F + kap K) z_c)
    inners: Jacobi-CG (W1, M), Jacobi-GMRES (W2, indefinite);
    outer: FGMRES; escalation on stall: inner GMRES on the EXACT Schur
    (matrix-free), W1-form preconditioned.

## The design laws (each learned from a dumped offender system)

1. **The classical square-root form (F dropped; the Pearson-Wathen /
   Boyanova-Neytcheva [1/2,1]-bound preconditioner) DIVERGES for the
   logarithmic potential** — its built-in cross term supplies an
   effective constant curvature 2 sqrt(sigma m kap) (= 2.0 at the gate
   parameters), fine for poly's f'' in [-1,2], hopeless against FH's
   10..1e4. (Why the literature never hit this: Bergermann et al.
   CiCP 2023 — the published preconditioner for THIS OSC system —
   treats the bulk term semi-implicitly (no F in the operator) AND
   replaces the log with a polynomial. We keep fully-implicit Newton
   on the true C1-regularized log.)
2. **Multiplicative absorption of f'' into the sqrt factor self-destructs
   quadratically**: gamma-weighted stiffness in W adds a spurious
   (m^2 f''^2 / 4 sigma) biharmonic, ~2e3 x the true one at the FH
   offender. Additive absorption in ONE factor (W2 = W1 + curvature) is
   the correct move.
3. **Carry F SIGNED.** Clipping to F_+ diverges on large-curvature
   Newton transients (poly c ~ 37 gives f'' ~ 4100: clipped 200 its,
   signed 88). W2 becomes indefinite where f'' < 0: GMRES inner, not CG.
4. **Consistent-mass solves are load-bearing.** Lumped mass in the
   pre/back-substitution diverges at large f'' (the lumping error is
   amplified by F). Jacobi-CG on M costs ~15 its.
5. **Fixed large dt at quench onset is OUT of the parity contract**:
   Newton itself does not converge there (15-cap, chaotic wells), so
   splu-vs-blockch trajectory parity is ill-posed (measured 2.7e-2
   'parity' between two exact-tolerance solvers) — the wodo
   device-parity lesson at the nonlinear level. Newton globalization
   added regardless: trust clamp |dc| <= 2/iterate (unclamped onset
   Newton reaches |c| ~ 37), FH projection to [1e-3, 1-1e-3],
   consistent mu-init (closes the recorded deferred item; kills the
   step-0 transient E 0.25 -> 173 class).

## Measured verdicts

Offender panel (dumped systems; splu-exact factor solves):

| system | v1 (F dropped) | W2 signed | exact-Schur |
|---|---|---|---|
| FH quench sigma=500, f'' to 9984 | DIVERGES | 32 | 2 |
| poly sigma=50, biharmonic-dominated | 50 | 53 | 2 |
| poly sigma=50, transient f'' ~ 4100 | 57 | 88 | 2 |

Production ladder (L5/L6 x dt 2e-3, 8-step quenches): poly 3/1 its,
FH 1/1 its; parity 1.0e-10 / 2.2e-10 / 5.5e-13 vs splu.

**The G1 robustness run** — LTE-adaptive marches t=0..0.5 (quench
through coarsening, sigma sweeping ~two decades):
poly 1335 accepted steps / 4026 solves / max 3 outer / 0 fallbacks;
FH 653 steps / 1983 solves / max 4 outer / 0 fallbacks, field
binodal-confined [0.068, 0.934].

## G3 ladder addendum (2026-07-10)

- 3-D marches (poly, blockch_dev): L5 (72k dofs) 4 steps at 1 outer it;
  L6 (549k dofs) 4 steps at 1 outer it, ~45 s/step (HOST-assembly
  dominated — the recorded device-assembly follow-up).
- **THE RESURRECTION, measured**: cuDSS direct on the L7 system
  (4.29M dofs, 228M nnz) = ALLOC_FAILED after 195 s on the 48 GB card;
  the SAME system marches with blockch_dev at **1 outer iteration per
  step** (2 steps, 580/712 s wall — host assembly + host preconditioner
  setup dominated; the solve itself is immediate). G3 CLOSED: the
  preconditioner solves a size class 8x beyond where direct
  factorization died on this card.
- **OPEN — FH at 3-D scale**: the FH L6 quench-onset step stalls BOTH
  inner paths (device BiCGStab > 1 h; host GMRES > 1.5 h, killed).
  2-D FH is 1-iteration clean, so the mechanism is scale- or
  spectrum-dependent (candidates: W2 inner Krylov without AMG-class
  preconditioning at 275k rows; the sqrt(sigma) balance at this
  kappa(h) scaling). Needs a dedicated dumped-system lab like the 2-D
  offenders. Recorded, not diagnosed.

## Open (G2+)

- One regime defeats BOTH the two-factor form and the W1-preconditioned
  exact-Schur inner (FH at fixed dt=2e-2 slammed into onset, f'' ~ 1e3
  + sigma=50): out of production contract, reachable only by forcing
  non-adaptive large steps; the escalation raises loudly there.
- Device port: inners are scipy CG/GMRES on host; the fused device CG
  carries W1/M (SPD); W2 needs a device GMRES or a CG-friendly
  symmetrization. Then G3: 3-D L6 on the 48 GB card where cuDSS
  ALLOC-failed; G4 ternary; G5 the 15.2M-dof full-res film.
