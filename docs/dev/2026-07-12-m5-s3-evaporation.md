# M5 S3 — evaporation-induced structure formation ledger

Agent build, 2026-07-12.  Contract: docs/theory/
crystallization_formulation_p1.md Sec 4 stage S3 (ratified): psi/theta
ride the Landau-mapped moving frame — mapped gradients + frame
advection apply to psi/theta EXACTLY as to phi; the top-flux term is
phi-only (solvent evaporates AMORPHOUS).  Sec 8's vapor-field
alternative is v2, out of scope.  Reference machinery:
src/diffsim/physics/wodo_film.py (mapped metric, advection, enrichment
flux, dt-cap, Biot parameterization — verbatim conventions).  Code:
src/diffsim/physics/multiphase.py (module docstring S3a carries the
mapped weak forms); gates: tests/test_multiphase_s3.py.  Style model:
the S2 ledger (2026-07-10-m5-s2-16390-replication.md).

## 1. S3a — film frame in the (M, K) factory

DESIGN.  Compile-time `film` flag on make_mpf_newton (factory key
extended; all-False = pre-S3 kernel bit-identically, wp.static-guarded
everywhere).  The wodo v1.1 generalized-metric conventions are kept
verbatim so film configs translate 1:1:

- MAPPED GRADIENTS: per-direction factors mlat = 1/lat_scale, mvert =
  Ycomp/h_curr on BOTH test and field gradients (1/h^2 on vertical
  grad.grad blocks) — CH transport, kappa, eps2 psi flux, KWC theta
  coefficient (|grad~ theta| in the KG smoothing), A3 anisotropy
  (angle + rotation flux from MAPPED components; rotw gains
  mlat*mvert).  The CHC stochastic flux contracts with the mapped
  TEST gradient only (physical-flux convention, wodo).
- FRAME ADVECTION K xi_y (1/h) d/dxi_y on the raw xi_y-derivative
  (Ycomp cancels) added to the phi rows, the psi rows (EXACTLY as
  phi, per the contract) and the theta rows (KWC: inside the (p+pf)
  bracket with the consistent psi-column block; frozen mode: the
  bookkeeping identity becomes MARKER ADVECTION — k_e = 0 restores
  the exact identity).
- TOP-SURFACE ENRICHMENT FLUX (phi rows ONLY): coef_i =
  (K - k_e_i)(1/h) Ycomp on the basis-generic consistent face mass
  (host natural-BC block, the wodo/A2 pattern); K = SUM_i k_e_i
  avg(phi_i^top) frozen at t_n; h_curr -= dt K on acceptance.  k_e is
  a FULL-species vector (scalar input = eliminated-solvent rate, the
  exact wodo Bi semantics); per-RETAINED-species k_e_i > 0 is the S4a
  hook — carried, not gated (its avg-vs-pointwise closure is an S4a
  question, recorded).
- A2 WALL ENERGY stays on the substrate face (assert: not the moving
  face); its natural term picks up the mapped boundary measure
  Ycomp/h (the computational mu rows are the physical rows divided by
  lat_scale h/Ycomp).
- march(): evaporation dt-cap dt <= dh_cap/K + h_min/phis_stop stops.
- A1 T-FIELD in the film frame: TemperatureField splits stiffness
  into lateral/vertical blocks recombined per attempt (mlat^2 Klat +
  mvert^2 Kvert), adds frame advection rho_cp (K/h) Int N_a xi_y
  dN_b/dxi_y, and the EVAPORATIVE-COOLING natural load
  -L_vap K (Ycomp/h) Int_top N_a dS (s_evap = -L_vap J_evap;
  L_vap = 0 default OFF).  LU cache keys on (dt, h, K).
- BASIS-GENERIC: the A2 wall face-mass builder was factored into the
  module-level _face_mass(dm, (axis, side)) (quadrature-built 1-D
  edge mass, exact per degree) and reused for the top faces and the
  T-field evap load.  BDF1 AND BDF2 carried (gates below); film
  h-update stays explicit first-order (the wodo contract).

CONSERVATION STRUCTURE (measured mechanism).  The BDF1 advection/
flux/h-update pairing is telescoping-EXACT: summing the phi row over
the partition of unity gives P^{n+1} = P^n / (1 - dt K/h^n) and
h^{n+1} = h^n - dt K, so h Int phi_i dtheta is conserved to machine
precision per step (the wodo SIGN NOTE, inherited).  The psi analogue
h Int psi is exact while psi_top = 0.  Under BDF2 the pairing is not
telescoping; the content drift is the scheme's O(dt^2) global error
on P' = (K/h) P — discretization-order, not a leak (measured below).

MEASURED FINDINGS (2026-07-12, L5 unless noted):

- TOP-TAIL LEAKAGE (recorded): with a psi blob at (0.5, 0.4), r 0.15,
  w 0.05, the blob's tanh tail advects into the receding surface and
  h Int psi drifts 1.76e-6 over h 1 -> 0.83 (CLIP-INDEPENDENT:
  clip_psi True/False identical — it is boundary leakage, not
  projection).  Physically this is crystallinity reaching the top
  face; the psi row carries no top flux by contract.  The gate
  isolates the mapping with a deeper blob (psi_top ~ 0); the shallow
  measurement is recorded here.
- Solute content: EXACT (0.0 relative drift) on every march run —
  the wodo pairing carries over to the (M, K) layout unchanged.

GATE-DESIGN FINDING (measured 2026-07-12): the first bookkeeping
config carried dsig = dh = 1 with L_psi = 0 — but W(psi) rides in
mu_0 REGARDLESS of L_psi (the crystal region is a free-energy well
for its species below Tm): phi_in_blob 0.25 -> 0.585 within t = 0.06
and the crystalline volume grew +131%.  REAL thermodynamic coupling,
not a mapping artifact; the mapping-isolation gate therefore runs
NEUTRAL energetics (dsig = dh = 0, chi_ca = 0), and the crystalline-
volume observable is locked against the ANALYTIC PHYSICS BRACKET
[0, 1/h_end - 1] (slow-diffusion limit 0: both fields frozen in
physical z; fast-diffusion limit 1/h: uniform enrichment through the
blob) — a mapping bug lands outside the bracket.

GATES (measured 2026-07-12, L5/L4-L5; locks >= 2x headroom unless
the lock IS the analytic bracket):

| gate | measured | lock | verdict |
|---|---|---|---|
| (iii) film MMS, manufactured h(t) = 0.6 - 0.05 t, r14 + KWC theta ON (L4->L5, dt 1e-3 x 4) | orders phi 2.03 / psi 1.99 / theta 1.99 (errs 4.17e-4 -> 1.02e-4 / 2.00e-3 -> 5.01e-4 / 5.98e-4 -> 1.51e-4); h parity vs analytic < 1e-14 | orders > 1.8 | PASS |
| (i) bookkeeping BDF1 (M=2 K=1, march h 1 -> 0.8487, dt-cap 4e-3, 0 rejects) | solute drift 6.7e-16 (BOTH species; the wodo pairing is telescoping-exact); psi content 9.2e-10 (residual tanh-tail top leakage, see below); crystalline volume +0.1649 in [0, 0.178]; max dh 2.37e-3 | 1e-12; 1e-8; the bracket; dh <= dh_cap | PASS |
| (i) bookkeeping BDF2 (same march) | solute 8.43e-5; psi content 8.43e-5 (identical pairing => identical error); cv +0.1646; 0 rejects | 5e-4 each (5.9x); bracket | PASS |
| (ii) amorphous regression vs wodo_film (M=2 K=0, wodo Fig-3-class blend, k_e = 1, fixed dt 1e-3 x 10) | per-step max |dx| 2.2e-16 (MACHINE-IDENTICAL trajectories), h parity 0.0 exactly, per-step K parity < 1e-14 | 1e-12; 1e-13 | PASS |
| A1 evap-cooling hook (L_vap = 2 vs 0, substrate Dirichlet 350) | deficit T_sub - T_top = 0.421 (end flux balance L_vap K h/k_th = 0.741; transient below balance as expected); L_vap=0 max dev 2.0e-11 | > 0.15; < 2 bal + 0.5; 1e-9 | PASS |

PSI TOP-TAIL LEAKAGE MODEL CHECK: the gate blob at (0.5, 0.35),
r = 0.12, w = 0.04 ends with its edge tanh((0.47)/0.04) ~ 1e-10 from
the receding surface — measured content drift 9.2e-10 matches; the
shallower (0.5, 0.4), r = 0.15, w = 0.05 blob measures 1.76e-6,
CLIP-INDEPENDENT (clip_psi True/False identical — boundary leakage,
not projection).  Physically: crystallinity reaching the top face
leaves through the frame; the psi row carries no top flux by the
ratified contract (Sec 4 frontier note).

## 2. S3b — evaporation-quench crystallization

### 2.0 Config pivot + solver finding (measured 2026-07-12)

FIRST ATTEMPT (recorded): 2310-absolute-seconds glued onto a 64 nm
box (l0 = h0 = 64 nm, time in s, D_self/l0^2, k_e = 0.08 l0/s).
MEASURED FAILURE: the K = 0 amorphous twin dried to phi_s = 0.05 with
the field UNIFORM to 4e-5 — although the code's own free energy is
spinodally unstable along the ENTIRE drying line (numeric exchange-
Hessian eigenvalues -0.46 .. -0.81 from phi_s = 0.60 to 0.04, np
mirror), and a single-mode dispersion probe measures the q = 2 pi
mode growing at sigma ~ 1.6e4/s (fastmode_n; BDF1 amplification
1/(1 - sigma dt) sign-flipping at dt = 1e-4).

MECHANISM (the D5 lesson's severe limit): BDF1 is L-stable — at
dt >> 1/sigma the amplification 1/(1 - sigma dt) -> 0^-, i.e. the
integrator ARTIFICIALLY STABILIZES a physically unstable state, and
the Appendix-A ladder gives no protection because Newton CONVERGES on
the overdamped trajectory (no rejects, no dt collapse).  The FDT
psi-noise partially rescues (per-step re-seeding — the wodo CHC-noise
lesson's mechanism), which is why the crystallization leg showed
structure while the noise-free amorphous twin froze uniform.  ROOT
CAUSE of the timescale gap: Bi = k_e l0 / D_s = 3.3e-7 at that
parameterization — 4-7 DECADES below the validated film range
(Wodo 0.1-10, Negi 3.3e-3): at 64 nm with real solvent D the box
equilibrates in microseconds while drying takes seconds; no
integrator can honestly bridge that without resolving 1/sigma.

RESOLUTION (per the spec's own words — "Bi from the validated set"):
S3b runs in FILM UNITS (h0 = 1, D_s = 1, t = h0^2/D_s, Bi = k_e =
0.1 — in the Wodo validated set), Vignes D_self keeping the
2310/16390 RATIOS (D_f in s = 0.5, in itself 1e-4, in p 4e-6; D_p in
s 1e-2, self 1e-7; solvent-in-dry-film 1e-5..1e-4), b_reg = 1e-3
(the wodo Fig-6 dilute-IC value: stabilizes the 85%-solvent start,
the quench arrives on enrichment), kappa = 2e-4 (wodo).  At the
crossing the local D_eff ~ 1e-2..1e-1 gives SD rates sigma ~ 10/unit
— COMMENSURATE with the dt ladder (that commensurability is what the
validated Bi range encodes).  DEVIATION (recorded): M_psi and
noise_psi become nondimensional calibration knobs — the implied
crystallization kinetics are faster than 2310's absolute M_psi =
0.1/s by the time-unit ratio; the 16390 SI9-vs-Fig6 finding
(morphology decoupled from crystallization kinetics at a 25x mobility
ratio, end states equal) is the recorded license.

### 2.1 Config (film units, house-parameter provenance)
- Ternary film (M = 2, K = 1): species 0 = fullerene-class small
  molecule (crystallizable; 2310.11844 Table-1 DIMENSIONLESS
  energetics: dsig = W-bar = 2.6355, dh = L-bar = 1.3072 in RT/v0,
  Tm = 558 K, T = 333 K, N = 5.0298), species 1 = polymer (N = 87,
  the Negi PDPP5T class), eliminated = solvent (N = 1).
- chi_aa: fp 1.0 (Negi film campaign), fs 0.7248 (2310 PCBM-oDCB),
  ps 0.3 (Wodo class); r14 delta-chi chi_ca[f, p] = chi_ca[f, s] =
  1.0836 (2310 crystal-contact penalty applied to both amorphous
  neighbors — house choice, recorded).
- Blend: phi_f0 = 0.10, phi_p0 = 0.05, phi_s0 = 0.85 (Negi-dilute
  2:1 f:p start; with b_reg = 1e-3 the IC is stable — the
  miscibility gap is entered on drying, the evaporation-quench
  pathway by construction).
- Film units (Sec 2.0): Bi = k_e = 0.1, kappa = 2e-4, b_reg = 1e-3,
  eps2_psi = 2e-3 at L6 (interface ~ 1.8 cells, the S1c dx-limited
  class); Vignes D_self ratios above; ls_drop (1e-6, 0.97, 35)
  [Fig-4]; noise_damp (1e-2, 0.85, 15); M_psi + noise_psi
  calibration knobs (Sec 2.0 deviation).

### 2.2 Calibration findings (measured 2026-07-12, L5/L6)

- SD-BURST NEWTON WALL: with the literal 2310/16390 Vignes dry-limit
  floors (D_p self 1e-7, contrast 3-4 decades) EVERY march stalls at
  the drying-induced spinodal burst (phi_s ~ 0.65): dt collapses to
  1e-5 with 30-50-iteration Newton grind (L5 AND L6; chi_fp 0.7
  does NOT help — mobility-contrast mechanism, not quench depth).
  The polymer-poor domains drive phi_p toward the 1e-3 projection
  floor where the b_reg curvature (2 b/phi^3 ~ 2e6) meets the
  composition-singular dLam blocks.  MITIGATION (production config,
  recorded deviation D-S3.1): Vignes dry-limit floors lifted to a
  2-decade max contrast ("dsoft": D_f self 1e-2, in p 1e-3; D_p in
  s 1e-2, self 1e-4; solvent in film 1e-3..1e-2) — the same march
  then traverses the whole burst at dt = 0.02 with 3-4 iterations
  and 0 rejects (L5: full dry in 56 s wall) and produces the
  expected AAPS morphology (f-rich domains to phi_f 0.85, iface
  MAD 0.31).  The full-contrast completion is recorded as a
  frontier (blockch-pairs + AC blocks preconditioner era, same as
  the S2 dissolution-march stall).
- NOISE WINDOW (evaporation-quench nucleation): FDT noise_psi is
  the P1-Sec-8 calibration knob; measured window at L5 (eps2 4e-3):
  5e-3/1e-2/1.4e-2/1.6e-2 -> NO nucleation through full drying +
  anneal (dense-domain fluctuations damp: psi_max 0.4 dilute ->
  0.15-0.2 at phi_f 0.8 — f'' ~ phi curvature); 2e-2 -> INSTANT
  spurious clusters at phi_s 0.84 (kT-dominated junk regime).  The
  SOLUBILITY SELECTION works beautifully in between: transient
  psi clusters at dilute (T = 300 probe: ncry 1 at t = 1) DISSOLVE
  (the r14 chi_ca crystal-solvent penalty makes crystallization
  thermodynamically forbidden below local phi_f ~ 0.51 at 333 K —
  the 2310 'no crystallization below 0.4' anchor's mechanism,
  operating in the film).  eps2 halves each level, so nucleation
  barriers drop at L6/L7 (hero-run lever).
- SEEDED MECHANISM CONTRAST (deterministic; the S2c implant
  pattern): identical psi = 0.95 discs implanted WET (t = 1,
  phi_s = 0.836) dissolve completely (psi_max 0.95 -> 7e-2 by
  Delta-t 0.5, -> 8e-4 by 2.0); implanted at the MARGINAL crossing
  (t = 11, domain phi_f 0.52-0.54 vs the 0.514 solubility) they
  ALSO dissolve (curvature: r0 = 0.06-0.08 discs are subcritical at
  marginal supersaturation); implanted in developed domains
  (phi_f ~ 0.75) they grow — the dissolution-vs-growth ordering IS
  the evaporation-quench mechanism, measured deterministically.
- IC-SEEDED psi = 1 discs at t = 0 (85% solvent): violent
  dissolution driving grinds the ladder (measured: no accepted
  march progress in minutes) — the mid-drying implant is the
  well-posed deterministic design (recorded).
- GIBBS-THOMSON at L5: r0 = 0.08 seeds dissolve even at phi_f ~
  0.75 domains (critical radius r* = gamma/|df| ~ 0.11 at eps2 =
  4e-3); r0 = 0.15 grows.  The same argument sizes the critical
  NOISE fluctuation at ~4-5 cells — why homogeneous FDT nucleation
  is rare-event-starved at gate scale (the physical film-units
  noise amplitude is 4.4e-4; even 30x that does not nucleate
  before the junk threshold ~2e-2 where kT rivals the barriers).
- CHI_CA (solubility strength) CALIBRATION: the 2310 literal 1.0836
  (phi* = 0.51) UNDER-confines — seeded crystals sweep the p-rich
  domains (measured area -> 1.0 at psi ~ 0.97 everywhere); the
  16390-class 4.48, and even 2.0 (phi* = 0.74), OVER-confine at L5
  (seeds dissolve at the phi_f ~ 0.75 domains).  Production value
  chi_ca = 1.6 (phi* = 0.67): seeds implanted at t = 14 (domains
  0.775) grow to the anchor-class end state X = 0.961, ncry 2 -> 1
  (impingement merge; KWC off), full dry (phis_stop 0.02) at
  t = 20.5 with 0 rejects.  NOTE at L5 the psi > 0.5 AREA still
  saturates ~1.0 late (the psi field rides over p-rich domains at
  32^2 resolution — confinement needs domains >> interface width;
  the L6/L7 hero carries the morphology claim; in-suite gates lock
  growth/dissolution/contrast, not confinement).

### 2.3 S3b gate design + campaign measurements + HANDOFF STATE

GATE FILE: tests/test_multiphase_s3.py::test_s3b_mechanism_dissolve_
vs_grow + test_s3b_coupling_contrast_and_variants (in the working
tree, NOT yet committed — commit-on-green pending, below).  Design:

- (i) MECHANISM (deterministic): identical psi = 0.5 embryos
  (r0 = 0.15, periodic-wrapped discs) implanted WET (t = 1, phi_s =
  0.836) dissolve; implanted DRY (t = 15 via a shared pre-marched
  state, phi_s ~ 0.15, domains phi_f ~ 0.8) grow — the ordering is
  the r14 solubility thermodynamics.  Design iterations (measured):
  psi = 0.95 discs make the wet dissolution impulsive enough to
  grind the ladder (dt reset to 2e-3 after the implant impulse also
  added); a NON-periodic box puts the phi_f-richest sites against
  the walls and wall-clipped half-discs are Gibbs-Thomson-
  subcritical (dry seeds dissolved — measured) => the gates run the
  laterally-PERIODIC film mesh with periodic-wrapped implant
  distances.
- (ii)+(iii) COUPLING + VARIANTS: one shared 15-unit pre-march
  feeds the deterministic leg, its repeat, and 3 noise-ON-at-implant
  growth variants (noise_psi 1e-2; growth-stage stochasticity —
  recorded honestly: the distribution lock is over GROWTH, not
  nucleation); K = 0 twin marched to the same horizon; contrast
  locks on crystalline area, phi_f purification (K1 phi_f_max vs
  K0) and max|phi_f(K1) - phi_f(K0)|.

CAMPAIGN MEASUREMENTS backing the locks (healthy-clock era, L5
periodic, chi_ca 1.6 unless noted):
- wet dissolution: r0 0.08/psi 0.95 seeds at phi_s 0.836: psi_max
  0.95 -> 7e-2 by +0.5, 8e-4 by +2 (chi 1.0836); r0 0.15 wall-
  clipped at chi 1.6: area 0.1625 -> 0.0, psi_max 2.2e-2, 1 reject.
- dry growth (t_implant 14, r0 0.15, psi 0.95): area 0.168 ->
  saturation, X -> 0.961 at full dry (t 20.5), 0 rejects (dry5b);
  end state matches the 2310/16390 near-total-crystallization
  plateau class (their 96%).
- K0 twin end state: phi_f_max 0.852, iface 0.309, X = 0 — the
  coupling contrast is structural (K1 purifies to phi_f ~ 0.94-0.98
  and crystallizes ~all f).
- L6 hero (seeded at quench + noise texture): 6 seeds -> 4 resolved
  crystals -> impingement merge, X 0.95 by t = 17.6.

BLOCKED-ON-HARDWARE (the ONLY missing piece): the pytest run of the
two S3b gates.  The WSL2 clock-governor pathology (Sec 3) pinned
both cards at idle clocks through the evening; every GPU-touching
path (warp assembly copies AND cuDSS) runs 10-50x slow, so the gate
suite could not complete within the session (it ran > 2 h in three
attempts; the identical marches took minutes in the healthy-clock
window).  HANDOFF: when the box recovers (host reboot likely),
  CUDA_VISIBLE_DEVICES=1 python -m pytest tests/test_multiphase_s3.py -k s3b -s
(~15-25 min healthy), read the printed measured values, tighten the
in-comment locks per house rules if the windows allow, and commit
(the S3a commit 645d10c carries the ledger; the gate file +
this section are the S3b delta).  FLAGGED LOCK: the deterministic-
repeat lock (rep_dev < 1e-8) may need the FP-fate treatment (house
assembly-atomics lesson) — if the repeat deviates at amplified-
roundoff scale, lock from the measured distribution instead.

SNAPSHOTS (Baskar's README request) — DELIVERED:
/tmp/claude-1000/-home-bglab-Baskar-DiffSim/de8f9b3d-9dd7-48ec-8acc-
9f5aeb605975/scratchpad/s3_renders_L6/: 38 npz field dumps
(coords, t, h, phi_f, phi_p, psi, theta; phi_s = 1 - phi_f - phi_p)
+ 38 preview pngs + config.json (exact config, seeds, script line).
Arc: t = 0 wet film (h = 1) -> drying + surface-directed layering
(polymer-rich skin) + lateral AAPS columns (t 2.5-13) -> implant/
nucleation in the f-rich columns (t = 14, 6 seeds, distinct theta
markers) -> growth (t 14.5, 4 crystals, area 0.375) -> impingement
+ near-full crystallization of the f phase (t 17.6, X 0.95) in the
dried film (h 0.156).  The t ~ 14.5 frame shows fullerene pillars
crystallizing under a polymer skin — production-quality.

## 3. Solver notes (S3 build)

- WSL2 GPU CLOCK-GOVERNOR PATHOLOGY (measured 2026-07-12 evening,
  FLAGGED for the supervisor): both RTX 6000 Ada cards
  intermittently pin at idle SM clocks (210-255 MHz vs 3105 max)
  UNDER 40-55% load (power 22 W, temp 43 C) — identical marches run
  10-15x slower in those windows (measured: the same L5 wet-leg
  march 7.2 s at boosted clocks vs > 7 min pinned; profiles show
  the time inside warp copies + cuDSS calls, i.e. GPU-side).
  nvidia-smi clock locking needs root (denied on the shared box);
  persistence mode already on.  S3b gate wall times in the suite
  are therefore CLOCK-STATE DEPENDENT; the gate LOCKS are physics
  quantities and unaffected.  First suspected as a pytest
  interaction (gates crawled while scripts flew) — disproved by a
  direct-call A/B; the governor state was the variable.

- cuDSS vs splu at L6 (25k dof, fastmode_n film march, line search):
  8.3 s/step vs 1.35 s/step at 4-7 iters — cuDSS is the production
  film-march solver from L6 up (the S2 Sec-3 finding extends to the
  film stack; plain DirectSolverOptions per the thread-leak lesson).
- The film dt ladder rides the EVAPORATION CAP dh_cap/K when K is
  large (measured: dt pinned at 0.086 = 0.004/0.046 with dt_max
  0.25), and the dt_max cap otherwise — both mechanisms exercised in
  the calibration marches.

## 4. Suite status

S3a stage (2026-07-12, this workstation):
- tests/test_multiphase.py + tests/test_multiphase_apack.py +
  tests/test_multiphase_s3.py (S3a set): 31 passed in 3088.6 s
  (under concurrent GPU-1 calibration load per the both-cards
  directive; the parity-class gates ran on the otherwise-quiet
  cuda:0 and their margins are 100-5000x — S3a regression measured
  2.2e-16 vs the 1e-12 lock).
- tests/test_multiphase_s2.py minus
  test_s2c_crystallite_quench_and_dissolution (the known multi-hour
  march; the ONLY deselect): 9 passed, 1 deselected in 2355.8 s.

S3b stage: gate file written and physics-calibrated (Sec 2.3); the
pytest confirmation run is BLOCKED on the GPU clock-governor state
(Sec 3) — handoff protocol in Sec 2.3.  S3b therefore UNCOMMITTED
per commit-on-green.

## 4b. S3c — front-end integration: HANDOFF (not started)

Per the spec's own contingency: S3a/S3b consumed the session.  The
S3c work items stand as specified: FilmParams species list (role
active|solvent, crystallizable flag, N_i, dh_i, Tm_i, dsig_i, eps_i,
D_i, k_e_i) + chi matrices per P1 Sec 5; RunLog preflight
crystallization rules (undercooling sign, psi interface resolution
eps_i/h, noise-vs-barrier magnitude); a config YAML for the S3b
case.  The S3b production config is fully specified in Sec 2.1 +
the snapshots' config.json — the YAML is a transliteration.

## 5. Frontiers recorded

- S4a per-retained-species evaporation: k_e vector carried; the
  avg-vs-pointwise flux closure for a VOLATILE retained species and
  the per-solvent drying line are S4a items (P1 Sec 4).
- psi top-tail leakage at the receding surface (above): a psi top
  flux is NOT in the ratified contract; if production runs show
  crystalline mass reaching the surface, the vapor-field v2 (P1 Sec
  8) is the recorded escalation path.
- wodo_film 3-D cudss mt-layer thread leak (S2 ledger Sec 8) remains
  open for the film stack; multiphase film mode uses the plain
  options already.
