# P1 — Coupled Multi-CH x Multi-AC: Phase Separation + Crystallization

Formulation memo for Baskar's ratification (2026-07-10). Anchors (see the
course bibliography `latex/refs.bib`): Siber-Ronsin-Harting 2512.16370 (the extended
Flory-Huggins theory for crystalline multicomponent mixtures — THE free
energy), 2310.11844 (binary crystallizing dynamics + morphology
pathways), 2512.16390 (crystallization x spinodal interplay during
thermal annealing, WITH experiments), Ameslon-Wodo d5cp00335k (ternary
amorphous phase-diagram taxonomy — parameter/IC selection map). Lineage:
Matkar-Kyu coupling; our own film stack (wodo_film) for the evaporation
stage.

## 1. Fields and layout

n species with volume fractions phi_i (sum = 1; ONE eliminated via the
simplex, giving M = n-1 conserved CH pairs), of which a subset
K = {crystallizable i} carries a non-conserved crystallinity psi_i in
[0, 1] (relative crystallinity of species i; phi_i psi_i = crystalline
volume fraction of i) AND an orientation theta_i (ruling R1: individual-
crystal identification + impingement; KWC-class, Kobayashi-Giga
regularized).

Node-major dof layout: [(phi_1, mu_1), ..., (phi_M, mu_M),
(psi_1, theta_1), ..., (psi_K, theta_K)] = 2M + 2K dofs per node. (M, K) are COMPILE-TIME
constants of the kernel factory (the nbf/nqp pattern), so the species
loops unroll; chi matrices ride in as small device arrays.

## 2. Free energy (their Eq. 26, volume density, kT/v0 units)

f = SUM_i (phi_i^2/N_i) [ psi_i(1-psi_i) dsig_i
                          + psi_i^2 dh_i (1 - T/Tm_i) ]         (crystal bulk)
  + SUM_i (phi_i/N_i) ln phi_i                                   (entropy)
  + SUM_{i<j} phi_i phi_j [ (1-psi_i)(1-psi_j) chi_aa_ij
        + (1-psi_i) psi_j chi_ac_ij + psi_i (1-psi_j) chi_ca_ij
        + psi_i psi_j chi_cc_ij ]                                (4-fold chi)
  + SUM_pairs (kap_ij/2) |grad phi_i|^2                          (comp. gradients)
  + SUM_{i in K} (eps_i^2/2) |grad psi_i|^2                      (cryst. gradients)
  + b SUM_i 1/phi_i                                              (simplex reg., existing)

Per-species inputs: N_i, dsig_i (amorphous-crystal surface parameter),
dh_i (latent heat), Tm_i, eps_i^2; pairwise: chi_aa/ac/ca/cc matrices,
kap_ij. T is a PARAMETER (annealing protocols = T(t) schedules).
Non-crystallizable species: psi_i = 0 fixed, all four chi collapse to
chi_aa — the amorphous FH limit is recovered EXACTLY (our current
ternary brick is the K = 0 special case; gate G0 asserts this).

## 3. Dynamics

CH (each independent phi_i): dphi_i/dt = div( SUM_j Lam_ij grad mu_j ),
mu_i = df/dphi_i - div(kap grad phi_i), with the FAST-MODE Onsager
matrix (their Eq. 15 generalized): Lam from per-species self-diffusion
D_i(phi) — constant-D "negi" and mixture-law "wodo" closures both kept
(the mobility-closure lesson from the Negi replication is now a named
config choice, never an assumption).

Stochastic AC (each psi_i, i in K):
dpsi_i/dt = -(v0 N_i/RT) M_i(phi) [ df/dpsi_i - div(eps_i^2 grad psi_i) ]
            + xi_i,
xi_i = FDT Gaussian (their Sec 2.2.1) — THERMAL NUCLEATION comes from
the noise, not from seeding (seeded ICs kept as an option for
deterministic gates). M_i(phi) composition-dependent (their Sec 5.4
closure as the default; constant as the gate variant).

Coupling physics to verify in gates: crystallization drives demixing
(the chi_ca coupling), demixing assists nucleation (fluctuation
pathways), dilution-enhanced + diffusion-limited growth modes (their
Sec 5 taxonomy).

## 4. Staging (Baskar's sequencing)

S0 AMORPHOUS REGRESSION: (M, K) machinery with K = 0 reproduces the
   existing ternary brick bit-for-bit-class (parity gate vs
   TernaryCHStepper on the spinodal config).
S1 BINARY ANNEALING (1 CH + 1 AC): MMS orders both blocks (coupled
   source terms); single-crystal Gibbs-Thomson/growth-rate sanity;
   replicate 2310.11844's reference regular-crystallization case +
   one pathway case (their Sec 4 kinetics curves: crystallinity vs t,
   nuclei count) — measured comparison, honest gaps recorded.
S2 TERNARY ANNEALING (2 CH + up to 2 AC): the 2512.16390 thermal-
   annealing interplay cases (crystallization vs AAPS competition);
   sim-vs-their-sim first, their experimental figures as the stretch
   comparison. Phase-diagram spot checks against the d5cp taxonomy
   (IC placement -> expected regime).
S3 EVAPORATION-INDUCED (the film): psi fields ride the Landau-mapped
   frame (mapped gradients + advection apply to psi exactly as to phi;
   the top-flux term is phi-only — solvent evaporates amorphous).
   Gate: solute + crystallinity bookkeeping conserved/consistent
   through the moving frame; evaporation-quench crystallization runs
   at Wodo/Negi-class configs.
S4 QUATERNARY, both requested variants:
   (a) 2 active + 2 SOLVENTS: M = 3; per-solvent evaporation rates
       k_e,s (selective evaporation — the flux and the dt-cap
       generalize per species; the solvent-blend drying line moves in
       composition space);
   (b) 3 active + 1 solvent: M = 3, K up to 3.
   Beyond: (M, K) generic by construction — S4 is the demonstration
   that no code changes are needed, only configs.

## 5. Solver and infrastructure (inherited, extensions named)

- Monolithic Newton over 2M+K dofs; the AC blocks add DIAGONAL-like
  (mass + eps^2 K) blocks — trivially preconditionable; blockch_pairs
  extends with per-psi scalar blocks alongside the M CH pairs
  (the deep-quench chi12 boundary caveat carries over; the reject
  ladder is the production fallback, as measured).
- Device assembly: ndof-generic slot maps already handle 2M+K; the
  node-graph pattern build is ndof-agnostic (kron(G, ones(ndof))).
- Front-end: FilmParams grows a SPECIES list (role: active|solvent,
  crystallizable flag, N_i, dh_i, Tm_i, dsig_i, eps_i, D_i, k_e_i) +
  chi matrices; the RunLog preflight gains crystallization rules
  (undercooling sign, psi-interface resolution eps_i/h, nucleation-
  noise magnitude vs barrier).
- Temporal: the annealing stage has no evaporation dt-cap; LTE control
  + the nucleation-burst dt signature to be measured (expected: dt
  collapses at nucleation events, regrows during growth — the
  adaptive-march story, third verse).

## 6. Estimates at house cadence

S0+S1 ~2 days (kernel factory + MMS + binary replication), S2 ~1-2
days, S3 ~1-2 days (film integration), S4 ~1 day (configs + gates).
Agent-driven with formulation control here; each stage commit-per-green
with measured tables.

## 7. Rulings (Baskar, 2026-07-10)

R1. ORIENTATION INCLUDED from S1: one theta_i per crystallizable
    species (KWC-class term p(psi_i)(alpha_i/2)|grad theta_i|, their
    Eq. 5) — required for individual-crystal identification +
    impingement. Layout becomes 2M + 2K dofs/node. Numerical care:
    the |grad theta| singularity at grad theta = 0 needs the standard
    Kobayashi-Giga regularization; theta is grain-periodic. Recorded
    fallback if the singular term fights the Newton/preconditioner
    stack: post-hoc psi-field watershed labeling identifies crystals
    without in-model impingement (identification-only degradation,
    physics loss recorded).
R2. QUANTITATIVE kinetics curves (crystallinity vs t, nuclei counts vs
    their parameter tables) AND morphology-class matching, both.
R3. THIS IS M5 — "OrgElMorph" (Organic Electronics Morphology). Spec:
    docs/dev/specs/2026-07-10-m5-orgelmorph.md. The film front-end +
    RunLog + preconditioner stack + Negi/Wodo validations delivered
    this week are M5's enabling tracks, recorded as delivered.

## 8. Fidelity checks vs Ronsin & Harting 2022 (2026-07-10; the ref-14
framework paper, arXiv:2204.11628)

THETA: the anchor framework prescribes NO theta kinetics — their theta_k
is a HEURISTIC marker field (nucleus detection assigns grain IDs;
markers propagate with growth fronts; the p(Phi) pi eps_g^2 |grad theta|
delta_D energy penalizes impingement at marker jumps; "separate
crystallites never merge in our framework" by construction). Our
dynamic Kobayashi-Giga gradient-flow theta is therefore a PHYSICS
SUPERSET (true OFPF-class dynamics) whose impingement energy matches
theirs in form (alpha <-> pi eps_g^2). Verdict: keep ours; their marker
heuristic recorded as a cheap alternative bookkeeping mode for
many-grain scale (follow-up, not required).

NOISE: their Eqs. 9/18 carry ADJUSTABLE intensity prefactors sigma_CH /
sigma_AC ("used to adjust the intensity of the noise") plus a
crystalline-domain damping interpolation f(phi_k). The anchor treats
absolute noise intensity as a calibration knob — our noise_psi /
noise_phi amplitudes (FDT-exact normalization, noise_psi^2 = kBT
nondim) ARE that knob; the 2-D depth ambiguity is absorbed into it
exactly as in the source framework. Absolute nucleation onsets are
calibration-bound IN THE ANCHOR TOO; quantitative onset comparisons
require matching their sigma values (unstated in 2310.11844 — recorded).

S3 DESIGN NOTE: their evaporation is a VAPOR-PHASE AC field
(phi_vap, Hertz-Knudsen outflux, Eq. 16) — an alternative to our
Wodo-lineage moving-frame film. S3 proceeds on the film frame per the
ratified contract; the vapor-field mode is recorded as a possible v2
(it naturally handles solvent BLENDS with distinct volatilities via
per-species saturation pressures — relevant to S4a).
