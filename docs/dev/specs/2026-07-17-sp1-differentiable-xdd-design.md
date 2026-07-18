# SP-1 — Differentiable Excitonic Drift-Diffusion: Instrument Twins + Learned Device Closures

**Design spec, ratified in brainstorm 2026-07-17.** The next milestone in the
OrgEl arc (chosen over M6 flow/film and crystallization-deepening; those tracks
are unaffected). Supersedes the port-only sketch in
`docs/theory/structure_property_p3.md` §SP-1 — this spec *extends* it with the
differentiable/instrument/learning program; the P3 rulings (standalone
simulator, no film-pipeline linkage) stand.

## 0. One paragraph

Port the group's excitonic drift-diffusion (XDD) simulator to the GPU-native
differentiable DiffSim stack — five fields (φ, n, p, X_D, X_A), time-dependent
by construction — and build on it: (i) an **instrument-twin layer** whose
forward+adjoint solves emulate J–V (light/dark/intensity-sweep), light-modulated
transient photocurrent J(t), and TRPL/PL, with IRF convolution; (ii) **design
gradients** d(Jsc)/d(morphology) at descriptor and voxel-field level; and (iii)
**learned device closures** — dissociation and recombination recovered from
instrument-space observables via the beyond-FH methodology (parametric
identifiability ladder → gauge-anchored neural heads). No prior work in this
lineage learns a closure *inside* the PDE; that is the headline.

## 1. Ratified decisions

| # | Decision
|---|---|
| D1 | Milestone = **SP-1 only** (port-first). SP-2 (OMIEC/OECT) is the successor milestone.
| D2 | Structure = **one integrated milestone, rung ladder R0→R3** (not decomposed into SP-1a/1b).
| D3 | Differentiability targets: **T1** d(Jsc)/d(descriptors), **T2** d(Jsc)/d(morphology field), **T3** learned k_diss/R from measured-style observables. All three on one forward+adjoint substrate.
| D4 | Instrument twins **gated this milestone: J–V (light + dark + intensity sweep), J(t) light-modulation, TRPL/PL**. pcAFM tip-BC twin = recorded successor (serves NSF AFM-twin proposal Task 1).
| D5 | T3 staging = **parametric → neural**: Stage A recovers known closure parameters and maps identifiability per observable set; Stage B adds gauge-anchored neural heads recovering a planted SRH-like perturbation.
| D6 | 2-D carries all gates; **one 3-D steady J–V parity rung on RVE morphologies** (~5 M dofs), not full tomographic volumes (~42 M dofs — infeasible and unnecessary; RVE validity is established by the lineage).
| D7 | Time-dependence is first-class: transient marching is the native mode (it already underlies steady J–V); BDF1 + BDF2 per the program's cross-matrix rule (BDF2 is new capability vs. the CPU code).

## 2. Source corpus (read 2026-07-17, five parallel deep-reads)

- **Framework paper** (Balaji/Nirmal/Dhruv): governing system Eqs. 2–7/16–20,
  closures (Onsager–Braun Eq. 8, Langevin Eq. 5, Beer–Lambert/spectra + RET),
  SUPG (not Scharfetter–Gummel), BE/BDF2, block Gauss–Seidel, full parameter
  tables and morphology J–V tables (§7 anchors below). PL/TRPL/pcAFM observables
  explicitly ABSENT — future work.
- **CPU code** `baskargroup-excitonic_drift_diffusion` (Dendrite-KT):
  `drift_diffusion_ts.cpp` is the live path (5 dofs, NDOF_EX=2, BDF1,
  block-iterated SNES/KSP, solver strategies incl. STEADY_PULSE = TRPL
  protocol skeleton); closures in `DDFields.h` (Onsager–Braun 5th-order series,
  three γ strategies, region mobilities with tanh blending); morphology contract
  `(ix,iy[,iz], morph, dist)`; canonical parity config `test/config.txt`
  (PM6:Y6 bilayer). Quirks adjudicated in §6.
- **NSF pcAFM proposal** (Wodo/Ganapathysubramanian/Mativetsky): the tip twin =
  movable circular Dirichlet patch + injection BCs (Φ_B, S, ρ_c) + arbitrary
  G(x,t); its Task 1 is the GPU twin this spec's successor will land.
- **Nirmal light-modulation paper**: J(t) under 10 GHz rectified-sinusoid
  illumination; ps-scale features (rise ~16 ps, fall ~30 ps); SHAP-selected
  features → descriptors (R² 0.94/0.92/0.79); 10–30 h per solve on 36 cores —
  the GPU case, quantified. Data on HuggingFace (BGLab/light-modulation-data).
- **Kodali 2012/2013 lineage**: original equations + f/f′ masks + Braun
  pair-distance integration; sensitivity ranking **Γ > L > ε > μ_n > μ_p**
  (adjoint validation anchor); PPV:PCBM and bilayer/BHJ/sawtooth J–V anchors.
- **RVE + dataset papers** (CompMatSci 2024, chemrxiv 2026, JMR 2026): 3-D
  anchors (RVE Jsc CB 1.880 / DCB 1.673; JMR triplets), canonical two-system
  parameter table, parameter sweep ranges (μ ∈ 10⁻⁷–10⁻³ m²/Vs,
  τ_x ∈ 10⁻¹⁰–10⁻⁷ s → L_D 5–160 nm) = Stage-A sampling priors, GraSPI
  descriptor ecosystem. All lineage ML is surrogate-over-outputs — none learns
  closures in the PDE.

## 3. Architecture — forward substrate

**Bricks** (`src/diffsim/physics/exciton_dd.py`), following the film front-end
pattern; basis-agnostic; BDF1 + BDF2:

- Poisson: −∇·(λ²ε(x)∇φ) = p − n (quasi-static algebraic constraint; **no
  SUPG on Poisson** — deliberate deviation from the CPU code, recorded).
- Carriers n, p: ∂t + drift-diffusion, Einstein relation, SUPG on the drift
  terms (Tezduyar-class τ per the M2 conventions).
- Excitons X_D, X_A: ∂t + diffusion-reaction (no drift, no SUPG), with
  dissociation sink, lifetime sink, generation, RET transfer, and the
  recombination back-feed source (region-assigned, kept for parity).

**Coupling:** monolithic 5-field Newton (default; clean adjoint), block
Gauss–Seidel (exciton ↔ DD) retained as an option for CPU-parity debugging.

**Variables & robustness (the 26-decade hazard):** contact BCs span n, p ∈
[e⁻⁶⁰, 1]. Primal (n, p) is the parity formulation; a **log-density
(Slotboom-class) formulation is a first-class option** — positivity by
construction, conditioning across the decades. The brick contract includes the
**continuation protocol**: dark equilibrium at V=0 → generation ramp (factor-5
ladder, as CPU) → voltage sweep, with log-spaced dt ramping. Newton runs a
positivity-guarded line search in primal mode.

**Closures** (`exciton_closures.py`) — every closure differentiable and
swappable behind one interface (the learned heads of R3 drop in here):

- `OnsagerBraunDissociation`: k_diss(|∇φ|, dist; a, ε) — 5th-order b-series,
  interface-masked; **exact field-coupling in the Jacobian** (CPU omits it;
  we take the better Newton and the correct adjoint; parity is
  tolerance-based, not iteration-identical).
- `LangevinRecombination`: R = γ(x) n p, γ strategies {sum, min, image-force}
  × reduction factor ζ × spatial strategy {uniform, interface}.
- `Generation`: {constant, Beer–Lambert, spectra-integrated (+RET)} ×
  **arbitrary G(x,t) waveform engine** (CW, step, pulse, rectified-sinusoid,
  user waveform) — the CPU stub (`get_generation_factor_at_time`→1.0) made
  real; this is the excitation-protocol layer for every twin.
- `RegionMobility`: majority/minority (μRatio) with tanh-blended interface
  masks on the dist field; Poole–Frenkel optional.
- Exciton sink split: **1/τ_x = 1/τ_r + 1/τ_nr** (radiative fraction is a
  named parameter) — required for a well-posed PL amplitude (fold-in #3).

**Morphology contract:** reader for the CPU `(ix,iy[,iz], morph, dist)` format;
reader for DiffSim film-output npz; **dist-field utility** (signed distance from
the binary morphology — absent from the CPU code, in scope here);
GraSPI-compatible descriptor export. Masks used by closures are **relaxed
(tanh) in the dist field** so morphology is differentiable (n6
relaxed-classification brief is the reference). **f′ (cul-de-sac connectivity
mask) is excluded, with reason**: discontinuous under field gradients and unused
by the current framework/CPU code (Kodali-legacy only).

**Nondimensionalization:** identical to the CPU code (x0 = height, φ0 = V_T,
C0 = N_C, μ0 = max μ, t0 = x0²φ0/μ0, U0, J0, λ²).

## 4. Instrument-twin layer

Protocols × observables, all differentiable functionals:

| Twin | Protocol | Observable
|---|---|---|
| J–V (light) | V-sweep, steady-marched per bias, continuation IC | J(V), Jsc, FF; **Voc via implicit-function theorem** (root of J(V)=0)
| J–V (dark) | G = 0, V-sweep | dark J(V) — transport/injection/recombination with no generation aliasing
| Intensity sweep | scale G by I | Jsc(I) power law α; Voc(ln I) ideality — the classic recombination-order separators
| Light-modulation J(t) | arbitrary G(t) (10 GHz rectified sinusoid = Nirmal config) | J(t) trace + feature vector (peak, amplitudes, rise/fall, phase, harmonics/THD)
| TRPL / PL | steady light → off (STEADY_PULSE); or pulsed | **PL_i(t) = ∫ X_i/τ_r,i dV** (new observable), optional spectral weighting; steady-state **PL-quenching ratio** (blend vs. neat)
| Instrument realism | — | **IRF convolution** on PL(t)/J(t) (differentiable); bandwidth parameter on J(t)

Current extraction: bulk-adjacent consistent-flux (DDJscBulk-equivalent, the
M2 machinery), reporting Jny and Jpy; the CPU's nonsmooth `J = min(|Jny|,|Jpy|)`
is kept for parity *reporting* but gradients use a designated contact (or
smooth-min). BCs: planar ohmic Dirichlet contacts (E_g-split built-in, as CPU),
lateral Neumann/periodic. Exciton contact BC: both quench (Dirichlet-0) and
natural modes supported; parity gates use the `_ts` behavior (§6).

## 5. Adjoint architecture — two modes

- **Steady-state (implicit) adjoint** for J–V-class observables: one adjoint
  linear solve at the converged state; **no taping through the pseudo-transient
  march**. Covers Jsc, J(V), FF, Voc (IFT), intensity sweeps, PL-quenching.
- **Taped transient adjoint** for J(t) and PL(t): BDF chain per findings-4c
  rules, dt schedule **frozen from the forward pass** (the adaptive controller
  is not differentiated through).

Both modes verified by three-way gradient checks (adjoint vs. FD, 1e-6 class)
and by **reproducing the Kodali-2013 sensitivity rankings (Γ > L > ε > μ_n >
μ_p) from single adjoint solves** — a published-anchor validation of the stack.

## 6. Honest-verdict ledger (CPU quirks, adjudicated)

| Item | CPU behavior | SP-1 ruling
|---|---|---|
| Exciton contact BC | legacy Dirichlet-0 vs. `_ts` natural-Neumann | support both; gate vs. `_ts`; physics note recorded
| SUPG on Poisson | present | **dropped** (elliptic, no advection); deviation recorded, parity tolerance absorbs it
| k_diss field-coupling in Jacobian | absent (frozen per iteration) | **exact** — better Newton, correct adjoint
| J convention | min(∣Jny∣,∣Jpy∣) | kept for reporting; smooth for gradients
| G(t) amplitude | stub (=1.0) | real waveform engine
| Rnp → exciton back-feed | present, region-assigned | kept (parity), flagged non-standard
| f′ mask | legacy only | excluded with reason
| NDOF_EX | legacy 1 vs. `_ts` 2 | 2 (donor + acceptor); legacy not parity-relevant

## 7. Rung ladder and gates

**R0 — forward port.**
- MMS on the coupled system: observed orders exact, cross-matrix (p1/p2 ×
  BDF1/BDF2).
- **Reduction gates (M5-G0 style):** collapsed-exciton limit reproduces the
  CMAME-2012 transport-only model; 1-D homogeneous matches the Kodali
  validation.
- **Standard-simulator cross-check:** 1-D homogeneous parity vs. **SimSalabim**
  (Koster group, open source) — the community-standard anchor.
- CPU parity: `drift_diffusion_ts` on `test/config.txt` (PM6:Y6 bilayer,
  Jsc at V=0, STEADY_PULSE) — tolerance from measured repeats, ≥2× headroom;
  ≥2 framework-paper Table-1 rows × 2 systems (e.g. P3HT:PCBM bilayer 0.746 /
  BHJ3 8.303 mA/cm²; PM6:Y6 2.422 / 19.810).
- Perf: one Nirmal J(t) trace (513×129, 1 ns, 10 GHz) — target **≥100×** vs.
  the 10–30 h / 36-core baseline; batched multi-waveform solves demonstrated.

**R1 — adjoints.**
- Both adjoint modes green (three-way checks); Kodali-2013 rankings reproduced;
  epoch-free transient chain at the 1e-6 class; backward ≤ 2.5× forward.

**R2 — design gradients (T1 + T2).**
- d(Jsc)/d(descriptors) via relaxed-mask morphology parameterization,
  GraSPI-interpreted; d(Jsc)/d(morph field) voxel map; **one inverse-design
  demo**: bilayer seed → optimized morphology under the relaxed masks, Jsc
  gain reported with the recovered morphology re-evaluated sharp.

**R3 — learned closures (T3), the headline.**
- **Stage A (parametric identifiability):** recover (a, ζ, τ_r, τ_nr, μ_n,
  μ_p, G) from synthetic data over the **observable ladder**
  {J–V(light)} ⊂ {+dark, +intensity sweep} ⊂ {+J(t)} ⊂ {+TRPL} — conditioning
  numbers per rung = the identifiability map. Expected aliasing to be
  demonstrated, not assumed: the (G·τ·k_diss) harvesting-efficiency degeneracy
  at steady state; the (ζ, μ) product degeneracy; their resolution by
  transients. Sampling priors from the chemrxiv sweep ranges. Recovery gates
  under **noisy observables + IRF** (no inverse crime); Laplace-approximation
  UQ on recovered parameters (from the GN Hessian — near-free).
- **Stage B (neural, gauge-anchored):** neural correction heads on
  k_diss(E, dist) and R(n, p, x), **anchored orthogonal to the parametric
  subspace** (the beyond-FH gauge lesson, instrument edition); truth = a
  **planted SRH/trap-assisted term**; gate = recovery with the diverse
  protocol and *demonstrated failure* without it (the 2.2e4×-gap analogue).
  Heads share a base class with `adjoint/neural_energy.py` (same
  infrastructure serves the solid-mech neural-constitutive track).

**R+ — 3-D rung:** steady J–V parity on RVE morphologies (~100³ voxels,
~5 M dofs): RVE anchors Jsc(CB) 1.880 / Jsc(DCB) 1.673 mA/cm², and/or one JMR
triplet member. Iterative/blockch-class solver path; cuDSS acceptable for the
2-D systems (nonsymmetric LU), with the program's known cuDSS caveats in view.

## 8. Out of scope (recorded, with owners-in-waiting)

- **pcAFM tip twin** (movable Dirichlet patch, injection BCs, raster/batched
  scans) → successor milestone; lands NSF Task 1.
- **SP-2** OMIEC/OECT → successor milestone (own six-rung ladder).
- Frequency-domain instruments (IMPS/IMVS/impedance) — forward covered by the
  waveform engine; efficient small-signal linearization recorded.
- Exciton–exciton annihilation (TRPL twin warns at fluences where it would
  bite); external-circuit RC for GHz J(t); transfer-matrix optics (spectra
  remain input files); trap *dynamics* as forward physics (SRH appears only as
  Stage-B truth); film-pipeline linkage (P3 ruling); real measured-data fits
  (the spec fixes the data interface — units, IRF, normalization — so the
  follow-on consumes experiments without rework).

## 9. Risks

| Risk | Mitigation
|---|---|
| Newton fragility across 26-decade densities | log-density option, continuation protocol in the brick contract, positivity-guarded line search
| Stiff interface-localized sources | implicit treatment, log-spaced dt (CPU-proven)
| Adjoint through adaptivity | frozen-dt taping; steady observables use the implicit adjoint (no taping at all)
| Morphology-gradient validity (relaxed vs. sharp) | R2 gate re-evaluates optimized morphologies sharp; relaxation width a reported parameter
| Identifiability failure modes | that's the R3 *result*, not a risk — the ladder measures it; Tikhonov-as-science lesson applies
| 3-D cost | RVE-sized gate only; full tomographic volumes recorded for the multi-GPU future

## 10. Execution notes

Build and 2-D gates on gpubox (RTX 6000 Ada; the remote toolkit is live);
the 3-D RVE rung on Nova A100-80 (`-A mech-ai`, `/work` recipe, proven
2026-07-17). Estimated shape: R0 is the largest rung (new brick family +
parity harness); R1 leans on existing taped-kernel machinery; R3 Stage A is
compute-heavy (batched transient solves — the perf gate pays for itself
here). Milestone report + roadmap entry on completion, per program practice.
