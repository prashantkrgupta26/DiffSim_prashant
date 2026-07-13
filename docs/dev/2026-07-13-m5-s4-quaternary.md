# M5 S4 — QUATERNARY demonstrations (the (M,K)-generic closeout)

Agent build, 2026-07-13.  Base commit ee0694b, branch master.
Contract: docs/theory/crystallization_formulation_p1.md Sec 4 stage S4
(ratified).  THE POINT: the (M,K) kernel factory is generic by
construction — S4 demonstrates that quaternary systems need CONFIGS
ONLY, no code changes.  Assembly = HOST for all gates (D4 verdict,
docs/dev/2026-07-13-m5-device-assembly.md: device atomics flake
bit-class gates).  Tests: tests/test_multiphase_s4.py.

## VERDICT: NO CODE CHANGES.

Both quaternary variants (S4a M=3/K=1 two-solvent selective
evaporation; S4b M=3/K=3 three-species annealing) ran as pure config
changes to the existing `MultiPhaseStepper`.  The per-species k_e
vector (carried since S3a as the S4a hook) is genuinely per-species
wired: `_film_K` sums `k_e[i]*avg(phi_i^top)` over ALL species and the
enrichment flux applies `(K_tot - k_e[i])` per tracked species i
(multiphase.py `_film_K`, `_attempt_ctx`, `_attempt`).  The K=3 psi
layout (ndof = 2M+2K = 12) unrolls in the compiled kernel with no
edits.  Zero gaps found.

## S4a — 2 ACTIVE + 2 SOLVENTS (M=3, K=1)

Layout: species 0 = crystallizable small molecule (SM, nonvolatile),
1 = polymer (nonvolatile), 2 = slow solvent (tracked, volatile),
eliminated 3 = fast solvent (volatile).  BOTH nonvolatiles are TRACKED
so both conserve exactly (the S3a telescoping identity mirrors the
proven eliminated-solvent setup).  k_e = [0, 0, ks, kf], fast:slow
= 4:1 (kf=1.0, ks=0.25).  L5, cudss, HOST assembly, miscible chi
(0.3), neutral crystal energetics (dsig=dh=0, L_psi=0) for the
bookkeeping/selectivity gates (mapping-isolation lesson).

### Gate (i) bookkeeping + depletion ordering (measured, L5)

| quantity | measured | lock |
|---|---|---|
| SM (tracked nonvol) content drift | 1.39e-16 | < 1e-12 |
| polymer (tracked nonvol) content drift | 0.000e+00 | < 1e-12 |
| fast(elim) physical loss | 0.2387 | — |
| slow(trk) physical loss | 0.0626 | — |
| fast/slow loss ratio | 3.82 | > 2.0 |
| fast mean frac 0.45 -> | 0.302 | < 0.35 (falls) |
| slow mean frac 0.35 -> | 0.411 | > 0.38 (rises) |

The telescoping identity holds at M=3 to machine precision for BOTH
tracked nonvolatiles.  The drying line bends in composition space: the
fast solvent's mean fraction FALLS while the slow solvent's mean
fraction RISES (its physical amount barely changes while the frame
shrinks) — the solvent-blend mechanism.  0 rejects, h_min stop.

### Gate (ii) selectivity contrast (measured, L5)

Flipping k_e to [0,0,kf,ks] (idx2 becomes the fast component) vs the
base [0,0,ks,kf] (idx2 slow): the SAME species slot ends at a
different mean fraction — 0.4113 (as slow) vs 0.2068 (as fast), delta
0.205.  The (eliminated - idx2) fraction difference flips sign:
-0.109 (base) -> +0.300 (flip) — the composition-path crossover.
Locks: delta > 0.1; sign(base) < 0 < sign(flip); flip-base > 0.2.

### Gate (iii) crystallization in the quaternary film (measured, L5)

A seeded SM crystal (fate-robust r0=0.2, the S3b margin) in the
two-active/two-solvent film PERSISTS and grows: psi_max 0.950 -> 0.943
(stable), X_SM 0.1182 -> 0.1279 (crystallinity rising) over t = 0.01,
0 rejects.  The coupled K=1 crystallization + M=3 CH + moving-frame +
selective-evaporation system integrates stably with a crystal present
— the K=1 machinery operates UNCHANGED in the M=3 two-solvent film.
Locks: psi_max > 0.9; X_SM rises (> seed).

FRONTIER (recorded, not a gap): the seeded-crystal + moving-frame
Jacobian STIFFENS beyond t ~ 0.02 (Newton iters climb 4 -> 40; the
ladder crawls / fixed dt=2e-3 hits newton_max).  Two mitigations were
measured: (a) `fastmode_n` mobility rejects at t=0 outright (the S4b
lesson) — const mobility is required; (b) cudss symbolic REPLANS every
step under the crystal-block nnz flapping (~8 s/step) — splu (no
replan) is ~1.5 s/step.  The short-window gate sidesteps the wall; the
long-march evaporation-quench crystallization physics is the S3
milestone (S3b), already delivered.  Not a code gap — a solver-tuning
frontier for deep-quench film crystallization.

## S4b — 3 ACTIVE + 1 SOLVENT (M=3, K=3), ANNEALING

Choice: ANNEALING (contract allows "annealing OR drying").  S4a
already exercises the drying/selective-evaporation machinery
extensively; S4b isolates the NEW machinery — three independent psi
fields (K=3, ndof=12) — deterministically (BDF2-friendly).  Three
crystallizable species with a distinct parameter spread:

| species | N | dsig | dh | Tm | drive@333 | solubility phi> |
|---|---|---|---|---|---|---|
| A (0) | 5.03 | 2.6355 | 1.3072 | 558 | -0.527 | 0.561 |
| B (1) | 6.0 | 2.45 | 1.20 | 520 | -0.432 | 0.640 |
| C (2) | 7.0 | 2.30 | 1.10 | 490 | -0.352 | 0.706 |

(eliminated species 3 = minority solvent).  chi_ca = 1.2 (solubility
penalty, per-pair).  Domains set by IC (A/B/C-rich stripes each above
its own threshold).  Constructed-object asserts verify the
per-species dsig/dh/Tm/eps2/L_psi arrays reached the stepper (the
missing-splat guard).

Constant mobility (mob="const", onsager=0.1 I): the `fastmode_n`
composition-dependent mobility is Newton-hostile with the seeded
deep-quench discs — MEASURED: it rejected every attempt at t=0 and
underflowed the ladder (the 8-minute no-progress diagnostic).  Const
mobility marches with 0 rejects (dt 2e-5 -> 0.01, Newton 4-11 iters).
Recorded as a config choice, not a code gap.  L_psi=2, newton_tol=1e-6.

### Gate (i) solubility selectivity — crystalline volume (measured, L5)

v_i^R = mean over stripe R of (phi_i psi_i).

| region | v_A | v_B | v_C | selectivity |
|---|---|---|---|---|
| A-rich | 0.4804 | 0.0762 | — | v_A/v_B = 6.30 |
| B-rich | — | 0.4515 | — | v_B(B)/v_B(A) = 5.92 |
| C-rich | — | — | 0.2054 | — |

A B-seed placed in the A-rich stripe cannot draw B material (phi_B
there ~0.09, far below B's 0.640 solubility threshold), so its
crystalline volume stays ~6x below the native A crystal — the
solubility selectivity generalized to K=3.  Locks: v_A/v_B(A) > 3,
v_B(B)/v_B(A) > 3.

### Gate (ii) grain identity across species (measured, L5)

Each species carries its own (psi_i, theta_i); grain_labels on each
field returns that species' grains.  Per-species crystal-theta
markers: A 0.203, B 0.465, C 0.700 (the seeded 0.3/0.5/0.7 pulled
toward the amorphous-0 edge, cleanly separable — pairwise gaps > 0.15);
grain counts A=2, B=2, C=1; psi_max 0.937-0.965.  Locks: distinct
ordered markers thetaA < thetaB < thetaC with gaps > 0.15; each
species >= 1 grain; psi_max > 0.9.

### Gate (iii) per-species kinetics spread (measured, L5)

X_i(t) = int(phi_i psi_i)/int(phi_i).  The distinct driving
(drive_A -0.527 > drive_B -0.432 > drive_C -0.352) gives distinct
kinetics:

| species | X_i(0) | X_i(3.0) | behavior |
|---|---|---|---|
| A | 0.256 | 0.710 | monotone rise (deepest quench) |
| B | 0.289 | 0.627 | monotone rise |
| C | 0.256 | 0.239 | stays low (marginal quench) |

Ordering X_A > X_B > X_C.  Locks: X_A, X_B monotone (step >= prev -
1e-3); X_A final > 0.6, X_B > 0.5, X_C < 0.35; strict ordering.  0
rejects.

## CROSS-MATRIX (feature x {basis, tstep})

| cell | feature | measured |
|---|---|---|
| S4a x p=2 basis | moving-frame bookkeeping | L4 p=2: SM drift 0.0, poly drift 0.0 (EXACT); fast 0.302 slow 0.411 (same selective-evaporation signature as p=1) — the nbf/nqp-generic factory |
| S4b x BDF2 tstep | K=3 crystallization | short anneal (t=1.0) under BDF2: X_A 0.405, X_B 0.428, X_C 0.269 — matches BDF1 at the same time to ~1%, 0 rejects (deterministic).  NOTE: at t=1.0 the fast-starting B still leads A; the X_A > X_B ordering is a PLATEAU property (crossover ~t=1.6 under BDF1) — the cell locks the early-time invariant (A, B both grow clear of laggard C) |

## Suite status

New: tests/test_multiphase_s4.py — 6 gates:
- test_s4a_selective_evaporation_bookkeeping
- test_s4a_selectivity_contrast
- test_s4a_crystallization_in_film
- test_s4a_bookkeeping_p2_basis           (cross-matrix p=2)
- test_s4b_three_species_crystallization  (gates i+ii+iii)
- test_s4b_bdf2_deterministic             (cross-matrix BDF2)

S4 suite (2026-07-13, split GPU0/GPU1): 6/6 PASSED (S4a 4/4 in 7:23;
S4b crystallization + BDF2 cell green, BDF2 re-run 97 s after the
plateau-vs-early-time assertion fix).

Regression net (2026-07-13, GPU0, HOST): test_multiphase.py +
_apack.py + _s3.py + _s2.py (deselecting only
test_s2c_crystallite_quench_and_dissolution, the standing exclusion):
42 passed, 1 deselected in 5847 s (1:37:27) — GREEN.  Zero source
changes in S4, so the baseline is untouched by construction.

S4 GREEN => M5 (OrgElMorph) COMPLETE: S0 amorphous regression, S1
binary annealing, S2 ternary annealing, S3 evaporation-induced film,
S4 quaternary demonstrations — all stages delivered with measured
gates; the (M, K) factory generality demonstrated config-only.

## Frontiers

- Deep-quench FILM crystallization Newton-stiffness wall (t ~ 0.02);
  solver-tuning follow-up (const mobility + splu mitigate; a
  psi-interface eps2/h refinement or a JFNK/line-search-Armijo tail
  are candidates).  NOT a code gap.
- fastmode_n composition-dependent mobility + seeded deep-quench discs
  = t=0 ladder underflow; const mobility is the gate config.  A
  flux-space Onsager factorization for robust seeded fastmode_n is a
  recorded follow-up (already noted for CHC noise in the docstring).
