# LTE / PI(D) adaptive time-step controller (2026-07-14)

User-switchable local-truncation-error controller with PI(D) step
selection to a user tolerance, on top of the variable-coefficient BDF2
retrofit (A4b) and the reject/rewind ladder.  OFF by default — the
existing grow/shrink ladders stay the fallback.

Scope (Baskar spec): CH (binary), ternary_ch, AC, multiphase phase-field
steppers.  Deliverable: PI + the noise contract solid; PID exposed and
gated on the coupled config.

## Where it lives

- `src/diffsim/physics/lte.py` — the whole controller (new module; nothing
  imports it, so OFF-path parity is structural):
  - `StepController` — multiplicative PI(D) digital-filter step-size
    controller, normalized by the error order `1/(p+1)`.
  - `lte_march(stepper, t_end, tol=, controller=, estimator=, ...)` — the
    adaptive driver (step-doubling **or** embedded predictor-corrector
    estimate), reject/rewind with bit-exact snapshot, noise contract.
  - `march(stepper, t_end, adapt="ladder"|"lte", tol=, ...)` — the one-knob
    user switch (`adapt="ladder"` = OFF, dispatches to the native ladder
    unchanged; `adapt="lte"` = controller ON).
  - `_MixedAdapter` / `_MultiphaseAdapter` — snapshot/restore/advance/
    solution adapters so the loop is basis- AND tstep-agnostic.
- `src/diffsim/film/preflight.py` — the `lte-noise-incompatible` rule
  (fires only when `adapt="lte"`; the noise notice in the RunLog preflight).
- `src/diffsim/film/params.py` — `adapt` (default `"ladder"`) + `lte_tol`.

## The controller

For a scheme of order p the local error scales as `dt**(p+1)`, so the
integral exponent base is `1/(p+1)` — exactly the deadbeat rule the
existing ladders already use.  The PI(D) form keeps the last accepted
error(s):

    fac = safety * (tol/e)**(kI/pe)
                 * (e_prev/e)**(kP/pe)               [P: memory]
                 * (e_prev^2/(e*e_prev2))**(kD/pe)   [D: 2nd memory]
    pe = p+1,  clamped to [fac_min, fac_max]

Gains (multipliers on `1/pe`), measured-then-locked defaults:
`deadbeat=(1,0,0)`, `pi=(0.7,0.4,0)`, `pid=(0.7,0.4,0.1)`.
`kP=kD=0` reproduces the existing deadbeat ladder rule exactly.

## Estimators

- **doubling** (DEFAULT, scheme-agnostic genuine LTE — THREE solves/step):
  step-doubling Richardson estimate
  `err = ||x_full - x_half|| / ||x_half|| / (2^p-1)` (one dt vs two dt/2
  from the same state; accepted state is the more accurate two-half-step
  solution).  This is the codebase's established production-march pattern
  and the estimate that TRACKS the tolerance on stiff phase-field quenches
  (measured L1 gerr/tol 5..13; dt grows through coarsening).
- **embedded** (CHEAP — ONE solve/step): predictor-corrector
  ("BDF1-vs-BDF2") indicator.  The BDF-p corrector `x_{n+1}` is compared
  to the linear extrapolant `x_pred = x_n + (dt/dt_prev)(x_n - x_{n-1})`
  read from the committed history in the snapshot.  `||x_new - x_pred||`
  is the classic Milne-device local-error indicator (scales as dt^2 — the
  predictor error dominates), so the controller steps on the low-order
  estimate while marching the high-order solution.  Control order 1
  (pe=2).  MEASURED CAVEAT: on stiff CH quenches the dt^2 increment
  indicator is dominated by the fast physics and OVER-resolves (achieved
  error << tol; L1 embedded 973 solves vs doubling's 348 at tol=1e-4).
  On a genuine 2-level scheme (tstep=bdf2, or AC) it is an efficient
  genuine predictor-corrector LTE — used for the L2 coupled table
  (multiphase bdf2: 77-182 solves, error tracks tol, dt spans decades).
  Multiphase BDF1 keeps no x_{n-1}, so its embedded estimate degrades to
  increment-control — prefer doubling or tstep=bdf2 there.

Norm: relative-L2 over the CONSERVED fields (CH c; ternary phi1,phi2; AC c;
multiphase phi_i,psi_k) — the retrofit ladder audit's measured-good norm
(max-norm is hostage to the sharpest interface node).

## NOISE CONTRACT (non-negotiable)

Strong/pathwise LTE is ill-posed under FDT forcing: the step-doubling /
predictor-corrector difference of two noisy realizations is dominated by
the independent noise draws, not the truncation error, so `tol` has no
deterministic meaning.  `lte_march` therefore, when noise is on:
1. emits the explicit notice (printed, stashed on `stepper._lte_notice`,
   logged to a RunLog if passed, and surfaced by the preflight rule);
2. sets `_lte_active=False`, `_lte_fell_back=True`;
3. falls back to the stepper's existing fixed-order BDF ladder (byte-
   identical to a plain non-LTE `.march()`), keeping the noise.

RULING recorded for Baskar review: the spec says "fall back to BDF2".
BDF2 is the deterministic recommendation, but the steppers forbid
BDF2+noise at construction (the FDT weak order under BDF2 is out of scope,
A4b).  A run that KEEPS its noise (no silent downgrade) therefore falls
back to the noise-valid **BDF1** ladder, and the notice states this
explicitly.  Deterministic runs get the full LTE controller.

## Measured gates

### L3 — noise contract (multiphase p1 M=1,K=1, level 5)

| check | result |
|---|---|
| notice fires + stashed on stepper | PASS |
| `_lte_fell_back=True`, `_lte_active=False` under noise | PASS |
| fallback march completes (BDF1+noise) to t_end | PASS |
| fallback == plain native `.march()` (bit-parity) | max\|df\| = 0.0 |
| deterministic LTE unaffected (388 steps, no fallback) | PASS |
| preflight `lte-noise-incompatible` = WARN (noise) / PASS (det) / absent (OFF) | PASS |

### L1 — CH binary (poly, stiff quench, level 5, BDF2, t_end 0.04)

Estimator = step-doubling; reference = fixed dt 1.25e-4 BDF2.
achieved global error vs a fine reference, accepted/rejected steps,
solve count, dt range:

| ctrl | tol | acc | rej | solves | gerr | gerr/tol | dt range |
|---|---|---|---|---|---|---|---|
| deadbeat | 1e-3 | 50 | 7 | 171 | 6.4e-3 | 6.4 | 1.9e-7 → 3.4e-3 |
| deadbeat | 3e-4 | 73 | 7 | 240 | 2.9e-3 | 9.5 | 1.0e-7 → 2.2e-3 |
| deadbeat | 1e-4 | 102 | 6 | 324 | 1.3e-3 | 13.4 | 1.0e-7 → 1.5e-3 |
| **pi** | 1e-3 | 57 | 8 | 195 | 5.3e-3 | 5.3 | 1.9e-7 → 3.1e-3 |
| **pi** | 3e-4 | 79 | 7 | 258 | 2.5e-3 | 8.2 | 1.0e-7 → 2.1e-3 |
| **pi** | 1e-4 | 110 | 6 | 348 | 1.2e-3 | 11.7 | 1.0e-7 → 1.4e-3 |
| pid | 1e-3 | 57 | 8 | 195 | 5.3e-3 | 5.3 | 1.9e-7 → 3.1e-3 |
| pid | 3e-4 | 80 | 7 | 261 | 2.4e-3 | 8.1 | 1.0e-7 → 2.1e-3 |
| pid | 1e-4 | 110 | 6 | 348 | 1.2e-3 | 11.7 | 1.0e-7 → 1.4e-3 |

- (i) achieved error TRACKS tol: gerr/tol in 5.3..13.4, monotone in tol.
  PI tracks slightly tighter than deadbeat (5.3 vs 6.4, 11.7 vs 13.4).
  Locked: gerr < 40*tol (>= 2x headroom) + monotone.
- (ii) dt spans ~4 decades (1e-7 → 3e-3) — the dynamic range that makes
  long coarsening horizons affordable.  On THIS short onset-dominated
  window a fixed dt is competitive on raw solves (fixed dt 5e-4 → 80
  solves, gerr 2.1e-3); the adaptive payoff is automatic error-to-tol
  control + the decade-spanning dt range (the codebase's established
  production-march premise).  The `embedded` estimator here OVER-resolves
  (increment-control dominated by the fast quench) — doubling is default.
- (iii) OFF-parity: adapt="ladder" == adaptive_march, max|dc| = 0.0.
- (iv) reject/rewind: full+2half+restore == pre-step state, max|dc| = 0.0.

### L2 — coupled (multiphase p1 M=1,K=1: phi CH + psi AC + frozen theta)

Deterministic crystal-growth config (seeded psi disc, T = 0.5 Tm,
dh < 0), level 5, t_end 0.15.  estimator = embedded predictor-corrector
on tstep=bdf2 (genuine LTE, 1 solve/step — step-doubling also works via
the same adapter but the sustained growth front keeps dt small, so it is
costly here; embedded+bdf2 is the practical coupled path).  Error vs a
fine fixed-dt reference:

| ctrl | tol | acc | rej | solves | gerr | gerr/tol | dt range |
|---|---|---|---|---|---|---|---|
| pi  | 1e-3 |  61 | 16 |  77 | 6.4e-3 | 6.4 | 1.6e-5 → 1.6e-2 |
| pi  | 3e-4 |  97 | 15 | 112 | 4.9e-3 | 16.4 | 4.7e-6 → 8.8e-3 |
| pi  | 1e-4 | 160 | 22 | 182 | 2.7e-3 | 26.6 | 1.5e-6 → 5.5e-3 |
| pid | 3e-4 |  97 | 15 | 112 | 5.1e-3 | 17.0 | 4.7e-6 → 8.6e-3 |

- (i) achieved error is CONTROLLED and monotone in tol (6.4e-3 → 4.9e-3
  → 2.7e-3); the coupled phi/psi system tracks the tolerance.
- (ii) dt spans ~3-4 decades (1.5e-6 → 1.6e-2) on the same adapter.
- (iii) OFF-parity: adapt="ladder" == the native Appendix-A `.march()`,
  max|df| = 0.0.
- (iv) reject/rewind: the multiphase adapter (phi/psi/theta + two-level
  BDF history + film/T state) restores bit-exactly, max|df| = 0.0.
- AC scalar (non-conserved sibling) tracks tol on the same driver.

## User-facing API

    from diffsim.physics import lte
    # OFF (default): existing ladder, unchanged
    lte.march(stepper, t_end)                       # adapt="ladder"
    # ON: PI(D) LTE controller to a tolerance
    lte.march(stepper, t_end, adapt="lte", tol=1e-4)
    # or directly (controller="deadbeat"|"pi"|"pid";
    #               estimator="doubling"|"embedded"):
    lte.lte_march(stepper, t_end, tol=1e-4, controller="pi",
                  dt_min=1e-7, dt_max=0.02)

One tol knob + the `adapt` switch + the automatic noise notice.
Film frontend: `FilmParams(adapt="lte", lte_tol=...)` drives the
preflight notice.

## Gates encoded (tests/test_lte_controller.py)

`test_controller_deadbeat_rule_pi_memory_and_clamps` (no GPU),
`test_l1_ch_error_tracks_tol`, `test_l1_ch_off_parity`,
`test_l1_ch_reject_rewind_bitexact`, `test_l2_multiphase_error_and_rewind`,
`test_l2_multiphase_off_parity`, `test_l3_noise_fallback_and_notice`,
`test_l3_deterministic_lte_unaffected`, `test_l3_preflight_rule`.
Bounds carry >= 2x headroom; the two parity gates assert == 0.0.

## Follow-ups

- PID tuning beyond the locked default if the coupled config shows
  residual oscillation (gains are exposed: `kI/kP/kD`).
- Embedded predictor-corrector for multiphase BDF1 needs a second history
  level (currently increment-control); `tstep="bdf2"` or `doubling` gives
  the genuine LTE today.
- Film-frontend execution wiring of `adapt="lte"` on WodoFilmStepper (the
  preflight notice is live; the march wiring is the next step).
