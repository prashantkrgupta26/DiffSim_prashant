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
