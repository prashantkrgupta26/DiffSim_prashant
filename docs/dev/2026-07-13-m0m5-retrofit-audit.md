# M0-M5 retrofit audit — standing rule compliance (basis-agnostic + BDF1/BDF2)

Date: 2026-07-13.  Base: 840d87e (master).  Auditor: Claude (retrofit agent).

THE RULE (Baskar, 2026-07-12): all developments (1) basis-order agnostic —
no hardcoded linear-element assumptions, basis values/gradients only via
tabulated basis arrays (nbf/nqp-generic), generic face quadrature on
boundary terms — and (2) BDF1+BDF2 capable where relevant (BDF2
DETERMINISTIC-ONLY; variable-coefficient BDF2 with reject-consistent
history; multiphase A4b = the reference pattern, see
docs/dev/2026-07-12-m5-apack.md Sec 4).

M5 (multiphase.py) already complies (A-pack gated it).  Scope here:
everything older.  Every "no"/"partial" verdict below was verified by
reading the kernel factory / assembly path, with file:line cites.

## R1 — audit table

Verdict key: **yes** = compliant; **partial** = generic machinery with a
specific documented truncation/limit; **no** = hardcoded assumption;
**n/a** = the column does not apply (static solve / no boundary term).

### Shared infrastructure

| unit | basis-agnostic? | BDF2? | face terms generic? | notes |
|---|---|---|---|---|
| mesh/basis.py `Tables` | yes — nbf/nqp from p via tensor-product tabulation; lapN table for p2 VMS | n/a | n/a | `gauss_1d` tabulates p in {1,2} only (program scope: p1+p2), generic beyond that would need `leggauss` fallback like faces.py:55 |
| mesh/faces.py `FaceTables` | yes — generic p, generic nq1 (leggauss fallback) | n/a | yes | the SBM face-integration reference implementation |
| mesh/nodes.py / constraints.py | yes — `_local_offsets(p, dim)`, hanging rows via `lagrange_1d(pv, xi)` per owner degree, chain resolution p-blind | n/a | n/a | equivalence oracle retained |
| assembly/operators.py (DeviceMesh, ConstrainedOperator, volume_triplets) | yes — per-(nbf,nqp,dim) kernel cache, mixed-p bins | n/a | n/a | lapN uploaded per bin (operators.py:165) — available to every kernel |
| assembly/matvec.py (ConstrainedVectorOperator) | yes — (nbf,nqp,dim,ndof) generic | n/a | n/a | |
| assembly/gp_field.py (DeviceGPField) | yes — all factories (nbf,nqp,dim[,ndof]) generic | n/a (per-GP axpbys are order-blind; BDF coefficients from the stepper) | n/a | `make_finescale_correct` hardcodes the BE-residual convention and drops nu·lap (shares gap G4's mechanism; conventions item 5 records BE as deliberate) |
| assembly/device_assembly.py (DeviceNSAssembler) | yes — ndof/nbf/nqp generic in both pattern modes; generic fill API physics-blind | n/a | n/a | consumed by NS, wodo film (ndof=4), multiphase |
| assembly/dirichlet.py | yes | n/a | n/a | |
| api/equation.py (CEquation brick API) | yes — framework owns the loops, nbf/nqp compile-time args | n/a | n/a | |

### Solvers / bricks (M0-M4)

| solver | basis-agnostic? | BDF2? | face terms generic? | notes |
|---|---|---|---|---|
| poisson.py (load / L2 / gauss_points) | yes | n/a static | n/a | |
| bratu.py (BratuProblem) | yes — nbf/nqp via dm.tables | n/a static (steady Newton) | n/a | uniform single-bin only (documented, asserts at first matvec) |
| SBM stack (sbm/poisson.py, vector.py, ns_shape.py, reference.py) | yes — all face terms ride `face_tables(pv, dim)` | n/a static (transient_adjoint composes the steppers below) | yes | |
| vms.py tau forms | yes — basis-blind algebra | n/a (sigma from caller) | n/a | `advection_matrix_dense` (vms.py:87-124) is a hardcoded p1/2-D **dense form GATE** (test oracle for the M_{a,s} adjoint identity, documented "form gate ... device kernels arrive fused inside the NS momentum brick") — test-only, not a production path: **accepted, no action** |
| scalar transport (physics/scalar_transport.py + steppers/coupled.py) | yes — VMS-COMPLETE residual incl. −kappa·lapN (the p2 order-3 requirement, measured; scalar_transport.py:92-96) | yes — BDF1+BDF2, constant-step coeffs with bootstrap (coupled.py:45-48); no adaptive/reject path so variable-coefficient is n/a | natural BCs + SBM face blocks reused (kappa-parameterized) | the formulation template for gap G4 |
| NS linearized monolithic (api/ns_bricks.py + steppers/linearized.py) | **partial** — kernels nbf/nqp-generic (run at p2 mechanically), but the SUPG/PSPG strong residual DROPS −nu·lap(u_h): ns_bricks.py:44 (docstring: "nu lap u_h dropped at p1"), ns_bricks.py:122-123 (`resu = sigma*Nb + conv`, no lapN term); same drop in the fine-scale correction (linearized.py:130-131, gp_field.py make_finescale_correct). At p1 exact (Q1 lapN ≡ 0); at p2 caps the achievable order — the exact mechanism measured on the scalar brick (2.11 vs 3.00). NO p2 NS gate exists in tests (all NS tests build p=1). | yes — BDF1+BDF2 via solvers/timestepping.py (production bootstrap t<1.5dt; constant dt; `bdf_coeffs` already carries the variable-step formula, unused) | n/a (do-nothing outflow; strong Dirichlet rows + pressure pin) | **GAP G4** |
| NS Leray projection (steppers/leray.py) | same as above (predictor uses assemble_linear_ns; PPE/mass einsum paths table-generic) | yes — BDF1+BDF2 | n/a | inherits G4 through the shared brick |
| Cahn-Hilliard binary (physics/cahn_hilliard.py, poly+FH) | yes — p1/p2 MMS gates exist (test_cahn_hilliard: p2 order ≥2.7) | **partial** — BDF2 with CONSTANT-dt coefficients hardcoded (cahn_hilliard.py:270 `(1.5, [2.0, -0.5])`) while `adaptive_march` (cahn_hilliard.py:373) varies dt per accept AND installs half-step-spaced history (accept keeps the two-half-step solution, so hist spacing = dt/2 ≠ next dt): formally inconsistent variable-step BDF2, locally order-degrading; no dt_prev tracking; adaptive_march rewind must also restore dt_prev once it exists | n/a (natural BCs) | **GAP G3** |
| Allen-Cahn (physics/allen_cahn.py) | yes — p1/p2 MMS gates exist | **partial** — same constant-coeff BDF2 (allen_cahn.py:178) | n/a | **GAP G3** |
| Ternary CH (physics/ternary_ch.py) | yes | **partial** — same constant-coeff BDF2 (ternary_ch.py:222); adaptive_march(stride=4) documented for it | n/a | **GAP G3** |
| **Wodo film (physics/wodo_film.py)** | **no** — `assert dm.mesh.p == 1, "Wodo film v1: linear elements only"` (wodo_film.py:421); `_build_top_faces` hardcodes the P1 edge mass `m1 = [[1/3,1/6],[1/6,1/3]]` (wodo_film.py:473-474) tensor-producted over lateral dims — indexes out/wrong at p2. Volume kernel itself is already nbf/nqp-generic. | **no** — BDF1 only: `order=1` to super (419), `sigma = 1.0/dt` hardcoded in `_attempt` (600) and `_attempt_device` (748), history rotation keeps hist[1] but it is never consumed | **no** — top-flux face mass hardcoded P1 (the ONLY non-generic boundary term left in the codebase) | **GAPS G1 (basis/face) + G2 (BDF2)** — top priority |
| film front-end (film/run.py, params.py) | pins p=1 (run.py:73 `build_mesh(tree, p=1)`, run.py:76 `basis_tables(1, ...)`) | pins BDF1 (no tstep knob; stepper had none) | (inherits stepper) | **GAP G5** (exposure; unblockable only after G1/G2). Note: the flight-recorder content integral is corner-average quadrature, "exact for P1" (run.py:19) — at p2 it becomes a diagnostic approximation (recorded) |
| multiphase.py (M5) | yes (A4a; `_face_mass` = the generic-face-mass reference builder, multiphase.py:1300) | yes (A4b; variable-coefficient, reject-consistent, multiphase.py:1973-1986) | yes | reference pattern, out of scope |

## Prioritized gap list (production relevance: wodo_film > CH/ternary > NS > static)

- **G1 — wodo_film basis + face quadrature** (fix): drop the p==1 assert;
  rebuild `_build_top_faces` on the A4a pattern (quadrature-built 1-D
  consistent edge mass per degree bin — exact per degree; p1 reproduces
  [[1/3,1/6],[1/6,1/3]] to 1 ulp (measured max dev 2.8e-17: NOT
  bit-identical — parity gate is measured-trajectory-class), p2 the
  Simpson-consistent le/30 [[4,2,-1],[2,16,2],[-1,2,4]]).
  IMPORT NOTE: multiphase.py imports FROM wodo_film (line 357), so wodo
  cannot import `_face_mass` from multiphase — the ~8-line quadrature
  build is duplicated locally; unification into mesh/faces.py is a
  recorded frontier (touching multiphase.py costs the full multiphase
  suite for zero functional gain today).
- **G2 — wodo_film BDF2** (fix): A4b sigma/hist rewiring (zero kernel
  change): tstep="bdf1"(default)|"bdf2"; per attempt r = dt/dt_prev,
  sigma = (1+2r)/(1+r)/dt, hist = ((1+r)x^n − r²/(1+r)·x^{n−1})/dt;
  BDF1 bootstrap (hist2/dt_prev None); rejects recompute coefficients
  from the actual dt and never mutate history; deterministic-only
  (assert noise == 0).  RECORDED LIMITS (same as multiphase film mode):
  the explicit h_curr update stays O(dt), and the advection/top-flux
  conservation pairing is telescoping-exact only under BDF1 — under
  BDF2 content drift is the BDF2 global error (order drift, not a leak).
  Temporal-order gate therefore runs at k_e=0 (h frozen: pure mapped
  ternary dynamics).
- **G3 — CH/AC/ternary variable-coefficient BDF2** (fix): track dt_prev
  in the steppers, compute (c0, ch) from r = dt/dt_prev (constant-dt
  arithmetic is bit-identical: r=1 gives exactly 1.5/2.0/−0.5);
  adaptive_march saves/restores dt_prev with the rest of the rewind
  state (reject consistency).  MEASURED PRE-FIX MECHANISM (CH poly,
  stable c0 = 0.8 relaxation, L4-p1, T = 0.096, ref BDF2 dt = 2e-4):
  alternating (dt, dt/2) sequence with the constant-step coefficients
  converges at order 0.90/0.95 with errors 4-20x the fixed-dt run;
  fixed-dt control is clean 2.18/2.07 — the inconsistency is real and
  first-order, exactly the variable-coefficient gap.
- **G4 — NS p2 VMS residual completeness** (fix, operator side).
  MEASURED PRE-FIX MECHANISM (steady Oseen vortex MMS, nu = 0.01,
  velocity L2): p1 orders 2.08/2.06 (correct), p2 orders 2.15/1.85 —
  capped at 2 instead of 3, the same incomplete-residual mechanism
  measured on the scalar brick (2.11 vs 3.00 complete).  Fix: add
  −nu·lapN(u_b)·(2/h)² to `resu` in make_linear_ns_Ae (lapN tables
  already on every bin); thread b["lapN"] through assemble_linear_ns and
  DeviceNSAssembler.assemble.  At p1 adds exact zeros (lapN ≡ 0) —
  bit-identical parity.  The fine-scale-correction residual
  (_corrected_gp / gp_field.make_finescale_correct) keeps the nu-lap
  drop: it is a modeling choice on the ADVECTING field, second-order
  GP evaluation of history fields is new machinery, and the measured
  order-cap mechanism lives in the operator residual — recorded
  deferral if the p2 order gate passes without it.
- **G5 — film front-end exposure** (fix if trivial): params p / tstep
  pass-through with defaults (1, "bdf1"); content diagnostic p2 caveat
  recorded.
- **G6 — vms.advection_matrix_dense**: accepted (documented test-only
  form gate), no action.
- **G7 — _face_mass unification** (deferred): move to mesh/faces.py and
  re-point multiphase + wodo; costs the full multiphase regression
  suite; do when multiphase is next open anyway.

## R2 — fix log

### G1 — wodo_film basis-generic + generic face quadrature (FIXED)

CHANGE (wodo_film.py only): `_build_top_faces` now builds the 1-D
consistent edge mass per degree bin by (p+1)-point Gauss quadrature over
the tabulated 1-D Lagrange basis (the A4a pattern; builder duplicated
locally because multiphase.py imports FROM wodo_film — G7 unification
frontier), tensor-producted over the lateral dims exactly as before; the
`p == 1` assert is lifted.  Volume kernel unchanged (was already
nbf/nqp-generic).

GATES (measured 2026-07-13, RTX 6000 Ada cuda:0):

| gate | measured | lock | verdict |
|---|---|---|---|
| (i) p1 regression parity (pre-change code vs post, fig67-class config, 10 fixed-dt steps, host path) | face-mass max dev 8.7e-19 (1-ulp class); trajectory rel dev max 5.9e-16; h_curr bit-identical | 1e-14 (17x headroom) | PASS |
| (i') face-mass analytic reproduction | p1 vs le[[1/3,1/6],[1/6,1/3]] and p2 vs le/30[[4,2,-1],[2,16,2],[-1,2,4]]: < 1e-15 | 1e-15 | PASS (test_wodo_face_mass_generic) |
| (ii) p2 capability — conservation mechanism | march to h=0.9 (L5 strip, k_e=1): content drift (1.4e-16, 0.0); enrichment + thinning clean, 0 rejects.  Mechanism: a p1-shaped face mass at p2 leaks solute at rate 2·K·Phi — exactness is the face-term correctness certificate | 1e-12 | PASS (test_wodo_film_p2) |
| (ii') p2 measured accuracy (A4a Richardson pattern: smooth deterministic run, fixed-dt common-mode, ref L6-p2) | L4-p1 err 2.02e-3 vs L4-p2 1.93e-4 — p2 beats p1 at the same h by 10.5x | e_p2 < 0.5 e_p1 (5x headroom) | PASS (test_wodo_p2_beats_p1) |
| (iii) cross-matrix: p2 x device assembly | 5-step host-vs-device trajectory parity 3.0e-16; h_curr identical | 1e-11 | PASS (folded into test_wodo_film_p2) |

SUITES: tests/test_wodo_film.py 6/6 passed (32.9 s) incl. the three new
gates; tests/test_ternary_ch.py 1/1 (31 s).  The film-frontend
end-to-end suite (4 full Negi marches) exceeded a 50-min timeout under
GPU contention on the first attempt and was re-run once after G2 with a
2 h budget (result recorded below) — the G1/G2 default path it
exercises is bit-identical to the pre-retrofit code by the parity
gates, so the deferral carried no risk window.

### G2 — wodo_film variable-coefficient BDF2 (FIXED)

CHANGE (wodo_film.py): tstep="bdf1" (default) | "bdf2" — the A4b
sigma/hist rewiring, zero kernel change.  `_bdf_time(dt)` computes
per-attempt sigma/hist from the ACTUAL (dt, dt_prev); `_commit`
(canonical accepted-step commit, used by march and the test drivers)
shifts hist2/dt_prev BEFORE the history rotation, so ladder rejects
never touch either.  Deterministic-only (noise == 0 asserted).
RECORDED LIMITS: explicit h_curr update stays O(dt) (order gate runs at
k_e = 0); the content pairing is telescoping-exact only under BDF1 —
BDF2 content drift is the scheme's global error (measured 3.5e-4 on a
k_e = 1 march, ~100x below the 2·K·Phi·dt·nsteps sign-flip leak scale).

GATES (measured 2026-07-13):

| gate | measured | lock | verdict |
|---|---|---|---|
| (i) tstep="bdf1" default regression (10-step fig67-class trajectory) | BIT-IDENTICAL to pre-G2 (max abs dev 0.0; h_curr 0.0) | 0-class | PASS |
| (ii) temporal self-convergence, k_e=0, fixed dt (ref BDF2 dt=2.5e-4) | BDF2 errs 5.4e-8/1.3e-8/3.1e-9 at dt 4/2/1e-3 — orders 2.01/2.13; BDF1 0.99/1.00 | orders > 1.7; BDF1 < 1.3 | PASS |
| (ii') BDF1-vs-BDF2 consistency | e_bdf2(2e-3) = 1.3e-8 vs e_bdf1(1e-3) = 2.1e-6 — 158x | e_bdf2(dt) < e_bdf1(dt/2) | PASS |
| (iii) ADAPTIVE-dt order study (directive 2026-07-13: alternating dt0, dt0/2 — r = 2 and 0.5 exercised EVERY step) | BDF2 orders 1.96/1.97; BDF1 0.99/0.99 | > 1.7 / < 1.3 | PASS |
| (iv) reject-consistency (directive): discarded attempts interleaved between accepted steps, k_e=1 production regime | accepted trajectory BIT-IDENTICAL to no-reject control (0.0) | 0-class | PASS |
| (v) cross cells | p2 x BDF2 temporal order 1.98; device-assembly x BDF2 parity 6.3e-16 | > 1.7; < 1e-11 | PASS |

SUITES: tests/test_wodo_film.py 8/8 (42.1 s).

### G3b — ternary_ch variable-coefficient BDF2 (FIXED)

Same (dt, dt_prev) rewiring as G3a (physics/ternary_ch.py, the 4-dof
stepper).  Gates (measured 2026-07-13): fixed-dt 12-step trajectory
bit-identical vs the pre-edit capture (0.0); ADAPTIVE-dt order study
(alternating dt0, dt0/2) orders 2.00/2.01 (lock 1.7).  Suite:
test_ternary_ch 2/2 (55 s).  NOTE (recorded, pre-existing): adaptive_
march cannot drive this stepper — its `.copy()` rewind assumes flat
hist arrays but ternary hist holds (phi1, phi2) tuples; the gate uses a
prescribed alternating-dt sequence instead.

### G3 — CH / AC / ternary variable-coefficient BDF2 (FIXED)

CHANGE (cahn_hilliard.py, allen_cahn.py; ternary_ch.py in the G3b
commit): (c0, ch) from r = dt/dt_prev per step (r = 1 reproduces
1.5/[2, -0.5] bit-exactly); dt_prev reset by set_initial and RESTORED
by adaptive_march's rewind on both the replay and the reject paths.

GATES (measured 2026-07-13):

| gate | measured | lock | verdict |
|---|---|---|---|
| (i) fixed-dt regression (12-step BDF2 trajectories) | CH 0.0, AC 0.0 (bit-identical) | 0-class | PASS |
| (ii) ADAPTIVE-dt order study (alternating dt0, dt0/2; pre-fix baseline 0.90/0.95) | CH 2.02/2.01 (fixed-dt control 2.18/2.07 unchanged); AC 2.01/2.01 | > 1.7 | PASS |
| (iii) cross cell: CH p2 x variable-dt BDF2 | order 2.02 | > 1.7 | PASS |
| (iv) reject-consistency (directive): adaptive_march WITH real LTE rejects vs replay of the accepted dt sequence | 27 accepted / 5 rejected; replay dev 0.0 (bit-identical) | 0-class | PASS |
| (v) adaptive_march health | 31 accepted, dt spans 28x, mass drift 7.8e-7 (Newton-tail class; conservation is scheme-exact: a = b − c holds exactly for the variable coefficients) | finite/physical | PASS |

SUITES: tests/test_cahn_hilliard.py + tests/test_allen_cahn.py 13/13
(160.7 s), incl. the three new gates.

### G4 — NS VMS residual completeness at p2 (FIXED)

CHANGE (api/ns_bricks.py + the assemble launch sites): the linearized
momentum strong residual factor gains its missing −nu·lapN(u_b)·(2/h)²
term (the lapN tables were already uploaded per bin).  The fine-scale
ADVECTING-field correction (_corrected_gp / gp_field
make_finescale_correct) keeps the nu-lap drop — a modeling choice on
the extrapolated field requiring second-derivative GP evaluation of
history fields; the measured order-cap mechanism lives in the operator
residual, and the gate passes without it (recorded deferral).

ADJOINT TWINS (fixed in the same commit — forced by a real gate): the
p2 tape-FD gate test_ad_gradients::test_p2_ns_tape_fd checks the taped
residual twin's dnu cotangent against FD of assemble_linear_ns and
FAILED once the forward operator was completed (the twin still dropped
nu·lap — measured rel err 4e-3-class vs the 1e-5 lock).  Both twins
(sbm/ns_adjoint.py make_lin_ns_residual, 2-D and 3-D) now carry the
same lapN completion, so forward operator and taped adjoint stay
exactly consistent at every p; the residual-matches-assembled and
dim-3 consistency gates re-verify the identity at 1e-12 class.

GATES (measured 2026-07-13):

| gate | measured | lock | verdict |
|---|---|---|---|
| (i) p1 regression (8-step transient vortex, linearized host + Leray) | BIT-IDENTICAL pre/post (0.0 for u and p) — lapN ≡ 0 at p1, the added term is an exact zero | 0-class | PASS |
| (ii) p2 spatial order (steady Oseen MMS, velocity quadrature-L2) | errs 1.45e-2/1.74e-3/2.10e-4 at L2/L3/L4 — orders 3.06/3.06 (pre-fix 2.15/1.85) | > 2.6 | PASS (test_oseen_p2_order) |
| (iii) cross cell: p1 device-assembly parity (DeviceNSAssembler launch-site) vs pre-G4 capture | 1.1e-15 (lapN adds exact zeros at p1) | 1e-13 | PASS |
| (iii') cross cell: p2 host-vs-device-assembly trajectory parity (8-step vortex) | 8.7e-14 relative | 1e-11 | PASS |
| (iv) p2 taped-adjoint dnu vs FD (test_p2_ns_tape_fd — the gate that forced the twin fix) | rel err back within 1e-5 (was 4e-3-class before the twins were completed) | 1e-5 | PASS |

SUITES: test_ns_bricks + test_ns_stepper + test_leray 14/14 (9.9 s);
test_device_assembly + test_ns_stepper + test_ns_bricks 30/30 (107 s);
test_ns_adjoint + test_ad_gradients 13/13 (23.5 s); test_leray_adjoint
+ test_transient_adjoint + test_ns_shape_gradient 7/7 (8.3 s).

### G3c — scalar-transport stepper variable-coefficient BDF2 (FIXED)

Same (dt, dt_prev) rewiring as G3a (steppers/coupled.py).  Gates:
fixed-dt 10-step trajectory bit-identical (0.0); adaptive-dt order
study 2.40/2.68 (lock 1.7).  Suite: test_scalar_transport 11/11.

### NS steppers under variable dt — DEFERRED (with mechanism)

The linearized/Leray steppers run BDF2 at CONSTANT dt by design (no
adaptive/reject path exists; `bdf_coeffs` already carries the
variable-step formula and `History` tracks dt_prev, unused).  Wiring
variable coefficients into ONLY the BDF term would be formally
inconsistent: the advecting-field extrapolation (2u^n − u^{n-1}), the
fine-scale BE residual history and tau's transient term all assume
uniform spacing.  A variable-dt NS march is a coherent feature (all
four pieces together), not a coefficient patch — deferred, recorded
here; the standing rule's BDF1+BDF2 requirement is met at constant dt
(gated in test_ns_stepper/test_leray).

### adaptive_march + ternary — pre-existing incompatibility (recorded)

`adaptive_march` rewinds hist entries via `.copy()`; the ternary
stepper's hist holds (phi1, phi2) TUPLES, so the stride=4 usage the
docstring anticipates has never been executable.  Not introduced by
this retrofit; recorded, deferred (the G3b variable-dt gate drives the
stepper with a prescribed dt sequence instead).

## R3 — verdict

(filled at close)
