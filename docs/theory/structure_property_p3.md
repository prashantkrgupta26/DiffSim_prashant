# P3 — Structure-to-Property: Excitonic Drift-Diffusion Port

Planning memo (2026-07-11). Source: previous_codes/excitonic_drift_
diffusion (Dendrite-KT CEquation app; Balaji/Nirmal/Dhruv framework
paper + 7 more in MyPapers/OSC/StructureProperty/). Baskar's ruling: DISTINCT process-structure and structure-property
simulators; linkage (the differentiable process->structure->property
chain) is a separate later phase.

## Physics inventory (from the source headers, 299 lines total)

Fields: n, p (carriers), phi (potential), x_d, x_a (donor/acceptor
excitons). Closures: k_diss(dist_to_interface, |E|), generation
G(dist, depth, t; spectra — Beer-Lambert/transfer-matrix via
DDSpectra), Langevin R(n,p), region-dependent mobilities. BCs:
electrode Dirichlet + lateral Neumann/periodic. Output: J-V curves;
Jsc via BULK-integrated boundary-element gradients — the same
consistent-flux principle DiffSim validated at de Vahl Davis (M2);
port onto that machinery.

## Port design

1. diffsim/physics/exciton_dd.py: DD brick (n, p, phi monolithic
   Newton; VMS scalar stabilization at drift-dominated regimes) +
   exciton brick (x_d, x_a) + the closures module (their
   material_parameters API transcribed; dimensional -> nondim layer).
2. STANDALONE BY DESIGN (Baskar ruling 2026-07-11): the
   structure-property simulator is a DISTINCT tool, not coupled into
   the film pipeline. Input contract = morphology files: their
   (x,y,z,morph,dist) format PLUS a reader for DiffSim film-output
   npz (offline handoff) PLUS experimental voxel data (TEM/tomography)
   — the same simulator serves simulated AND measured morphologies.
   The live in-memory adapter and the end-to-end differentiable
   d(Jsc)/d(processing) chain are the FUTURE LINKAGE PHASE, planned
   only after both simulators stand on their own gates.
3. Gates: (i) their 2-D/3-D bilayer test cases — J-V parity vs the
   CPU code outputs; (ii) MMS on the coupled DD system; (iii) the
   Kodali-2012 class morphology studies (papers in StructureProperty);
   (iv) an M5 film-output npz -> Jsc through the OFFLINE handoff,
   with the Gu-2017 Jsc-vs-domain-size trend as the physics sanity
   anchor (adjoint d(Jsc)/d(morphology-params) stays WITHIN the
   property simulator; cross-simulator gradients = linkage phase).
4. Estimated 4-6 agent-days after M5 S3/S4; C++ -> brick transcription
   is the designed use of the Integrands API (zero-relearning promise).

## Sequencing

After M5 S4 (or interleaved when GPUs idle): port + file-mode parity
gates -> morphology adapter -> differentiable Jsc demo. Feeds the
flagship paper (process->structure->property gradients).

## SP-2 — Conjugate ionic-electronic transport (bioelectronics/OMIEC)

Added by Baskar's ruling 2026-07-11. Spec: MyPapers/OSC/StructureProperty/
bioelectronic_device_models.pdf ("Morphology-Aware Bioelectronic Device
Modeling" — a complete, verification-first framework document). The
Structure-Property milestone is therefore a TWO-SIMULATOR family:

  SP-1: excitonic drift-diffusion (OPV; the CPU-code port; J-V/Jsc).
  SP-2: conjugate ionic-electronic (OMIEC/OECT; NEW build to the spec):
    two-domain Omega_e (electrolyte PNP, Eqs. 16-17) + Omega_p (polymer:
    ion PNP + hole drift-diffusion + Poisson with fixed charge, Eqs.
    18-20), interface menu at Gamma_pe (potential/flux continuity;
    optional partitioning K_s, Donnan, Stern C_S closures), electrode BC
    menu (Dirichlet/blocking/reservoir/Butler-Volmer). FROZEN morphology
    contract (swelling/poro-mechanics = recorded future extension).

  CANONICAL GATE: the spec's OECT 2-D cross-section benchmark — its own
  six-rung verification ladder (1-D reduction -> electrolyte-only PNP ->
  homogeneous-polymer OECT -> morphology-sweep monotonicity -> mesh/time
  convergence of I_D(t) and front position -> interface-closure limits).
  QoIs: I_D transfer/output/transients, doping-front tracking, the
  speed-gain trade-off vs the four morphology knobs (ion-path
  percolation, correlation length, anisotropy, Gamma_pe roughness).

  SHARED INFRASTRUCTURE with SP-1 (build once): the morphology input
  contract (the spec's m(x) indicator == SP-1's morph field; same
  file/npz/voxel readers), VMS transport bricks, consistent-flux current
  extraction (I_D exactly as Jsc), electrode BC menu, octree adaptivity
  at doping fronts and Gamma_pe (the Debye-ratio lambda_D/L smallness is
  where adaptivity pays).

  DiffSim synergies: PNP is the recorded DendrIon-heritage roadmap item
  landing here; the stiff Poisson-PNP-DD coupling is a blockch-class
  preconditioning candidate (recorded, not assumed); the nondimensional
  groups (beta, lambda_D/L, transport contrasts, chi_fixed) drive the
  RunLog preflight rules for SP-2.

  Sequencing: SP-1 port first (validation targets exist as shipped test
  cases), SP-2 build second on the shared infrastructure; both distinct
  from the process-structure simulator per the standing ruling; linkage
  phase unchanged.
