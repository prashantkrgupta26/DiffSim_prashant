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
  state (reject consistency).
- **G4 — NS p2 VMS residual completeness** (fix, operator side): add
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

## R2/R3 — fix log and verdicts

(filled per green stage below)
