# M4 — Differentiable Phase-Field & Learned Thermodynamics

Ratified by Baskar 2026-07-07 (night). Formulation: docs/
phasefield_formulation_p0.md (P0). Purpose: learn free-energy
functionals BEYOND Flory-Huggins from spatio-temporal data (synthetic ->
MD -> operando experimental) of ternary solvent-evaporation systems with
phase separation + crystallization. Replication anchor: Wodo &
Ganapathysubramanian (JCP 2011 adaptive-CH; CMS 2012 solvent-based OSC
fabrication — both in MyPapers/LearningFreeEnergy/).

## Tracks (Baskar's four + the learning arc)

(a) **CH solver, p1 + p2, ADAPTIVE octree, device-run**: mixed (phi,mu)
    brick (guide Sec 2.3 weak form), monolithic Newton (Suresh Jacobian
    pattern), BDF1/BDF2; spatial adaptivity = interface-band refinement
    (|grad phi| marker) with per-epoch re-mesh + M3 P-operator state
    transfer; gates: MMS p1/p2, mass 1e-12, energy decay, spinodal
    benchmark, JCP-2011 mesh-sensitivity comparison; A100-class device
    residency (M1d stack).
(b) **AC solver, p1 + p2, adaptive octree, device-run**: single-field
    brick (guide Sec 1.3); same adaptivity/gates; shrinking-circle
    benchmark (V(t) linear-in-time exact rate).
(c) **Wodo CMS-2012 REPLICATION** (solvent-based fabrication): ternary
    coupled CH + evaporation BC (P0 Sec 4, E1); reproduce the paper's
    morphology-evolution regimes (surface-directed waves, vertical
    stratification vs blend ratio / evaporation rate); the validation
    that "solvent-based evaporation is implemented". Device-run.
(d) **The teaching tutorial** (tutorials/F_phasefield/): students learn
    (i) phase-field theory (free energies, conserved/non-conserved
    dynamics, interface asymptotics), (ii) implementation on DiffSim
    bricks (weak forms -> kernels -> Newton), (iii) SPATIAL adaptivity
    (markers, epochs, transfer operators), (iv) TEMPORAL adaptivity
    (BDF1/BDF2 + LTE-controlled dt per Wodo JCP 2011). Format: the
    E-series style (runnable chapters + EXPLORE problems + honest
    EXPECTED RESULTS).
(e) **The learning arc** (the M4 punchline; P5-P7 of the brainstorm):
    basis-first f_mix surface on the simplex (linear-in-theta), then
    Lipschitz-MLP; gauge anchoring; Gramian identifiability discipline;
    composition-coverage diagnostics; instrument-space losses (S(q,t)
    FFT, confocal PSF, spectroscopy integrals, h(t)); mobility
    degeneracy handled by data-window design. FH -> basis -> MLP misfit
    ladder = the beyond-FH evidence.

## Order of work

P0 memo (DONE) -> (b) AC [simplest brick; adaptivity machinery debugs
here] -> (a) CH binary -> ternary coupled -> (c) evaporation +
replication -> (d) tutorial (written WITH the builds, E-series style)
-> (e) learning arc. Estimates at current cadence: (b) ~1-2 days, (a)
~2-3 days (Newton + adaptivity), (c) ~2 days, (d) alongside, (e) first
learned surface ~2 days after (c).

## Standing constraints

Non-dyadic feature dims; signal-scale checks before optimization;
Gramian before burning compute on inverse runs; findings-4c taped-kernel
rules; commit-per-green with measured tables; failures recorded with
mechanisms. Staffing: agent-driven build; student lands at (e)'s
instrument/experimental stage (Raghu has left the group).
