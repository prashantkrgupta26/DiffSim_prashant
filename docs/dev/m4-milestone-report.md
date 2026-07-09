# M4 Milestone Report — Differentiable Phase-Field & Learned Thermodynamics

Date: 2026-07-08. Status: **COMPLETE in substance** (ratified 2026-07-07
night; ~52 hours spec-to-done). Spec: specs/2026-07-08-m4-phasefield.md;
formulation: docs/phasefield_formulation_p0.md. One pending external:
the H200 3-D stretch case.

## 1. Contract vs delivered

| Track (Baskar's rulings) | Delivered | Evidence |
|---|---|---|
| (a) CH solver, p1+p2, adaptive octree, device-run | YES | Mixed (c,mu) brick: MMS orders 2.01/3.00, mass 1.9e-15 EXACT, energy decay, spinodal; spatial adaptivity ZERO mass drift (nested-refinement transfer); temporal adaptivity 178x dt growth (relative-L2 LTE; the max-norm lesson recorded); device-bound via ndof-generic slot maps |
| (b) AC solver, p1+p2, adaptive octree | YES | Orders 2.00/3.00 EXACT; energy decay; shrinking circle 0.4% vs theory |
| (c) Wodo CMS-2012 replication (solvent evaporation) | **YES — FULL RESOLUTION** | Landau-frame film stepper (agent-corrected sign pair; solute conserved 4e-16 through the moving frame); Fig 3 first-run (62x Bi drying ratio; surface-initiated separation corr -0.86); Figs 6+7 3/3 gates (A: 2.2 -> 1.5e6 percolated->multilayer; opposite-component surface wetting); **Nova A100: ALL 14 CASES at the paper's 250x100 mesh in ~11 min wall** incl. NEW Fig 4 (Bi ladder A 0.0017->27) and Fig 5 (off-critical 1:0.8 spontaneous stratification A=1.6e5); Fig-3 cross-machine agreement to 4 digits |
| (d) Teaching tutorial + LaTeX | YES | tutorials/F_phasefield/ F1-F3 (run-verified; measured EXPECTED RESULTS; EXPLORE problems) + docs/latex/m4_phasefield_tutorial.tex (~14 pp course doc); new-student onboarding framing |
| (e) The learning arc | **YES — FIRST LEARNED FREE ENERGY** | FH-4 (chi_pf, chi_ps, chi_fs, k_e) recovered to 1e-7 rel (J 52.7 -> 1.5e-12) in 147 s of device-bound marches, via h-CONTINUATION (nonconvexity beaten by the M3 pattern at parameter level); baseline m4_learned_fmix.json |

## 2. The science this milestone produced

1. **The T1 gauge mode, discovered numerically**: Chebyshev T1 of
   delta-f'(phi_p) is an EXACT invariance of the dynamics (verified
   5e-15) — the P0 memo's predicted affine gauge freedom found
   concretely by the optimizer sliding along it. Basis re-anchored
   T2..T4.
2. **Composition-coverage identifiability, measured**: T2..T4 alias
   the FH span over a single trajectory's composition range (Gramian
   eig 1e-5 of top; three genuine local minima) — beyond-FH functional
   recovery REQUIRES instrument-space observables (S(q,t), h(t)) or
   composition-diverse data. Exactly the brainstorm's Stage-2 design,
   now evidence-backed. = the next rung's requirements doc.
3. **Three measured-necessary model completions** for the Wodo regimes
   (each with mechanism): CHC conserved noise (IC fluctuations die
   before slow quenches without it); composition-dependent mobility
   with freeze-out (post-drying coarsening otherwise); the footnote
   b/phi regularizer is PHYSICS (stabilizes the dilute blend ->
   surface-first separation; A 1e5 vs 4 without).
4. **The evaporation dt-cap** (Baskar): the solvent flux is an external
   clock invisible to LTE control — dt <= tol*h_surf/(k_e*dphi);
   plateau-then-growth dt signature confirmed in the marches.
5. **Nested-refinement conservation**: refinement transfer is
   exact-conservative by construction (nested FE spaces); L2 projection
   needed only for coarsening.
6. **The C1-regularized log**: hard clamps freeze the FH restoring
   force (fields escaped the simplex WITH mass exactly conserved —
   the teaching version of gauge-vs-constraint); linear mu-extension
   restores confinement + Newton consistency.
7. **A generational benchmark**: Wodo-2012's 2-D cases at ~1.5 h /
   8 CPUs each -> 3-107 s/case device-bound on one A100 (~100x),
   with mass drift 1e-15.

## 3. Solver/infrastructure ledger

Film marches FULLY DEVICE-BOUND (slot-map scatter + zero-copy cuDSS;
parity 1.45e-13; 56x vs host; 36 ms/step; the nvmath plan-retention
trap found+fixed). DeviceNSAssembler ndof-generic. Nova campaign kit
(cluster/wodo_campaign/) exercised end-to-end by Baskar same-day.

## 4. Open items

1. **H200 3-D stretch** (128x128x48, ~3.3M dofs) — kit ready; result
   pending. Full-res 3-D (15.2M dofs) needs the CH BLOCK
   PRECONDITIONER — the recorded M4 solver item (task-6 experience
   applies; Boyanova-Neytcheva class).
2. Learning rung 2: instrument-space observables (S(q,t) via FFT,
   h(t), PSF confocal) + composition-diverse protocols -> the
   beyond-FH functional recovery. Student-scale; the new-student
   onboarding tutorial (track d) feeds directly into this.
3. Crystallization (the eta fields; Raghu's coupled model P4 rung) —
   machinery ready (AC brick + coupler pattern), not yet composed.
4. Deferred polish: consistent mu-init; convex splitting option;
   kap_12 cross-gradient; adaptive-mesh x film composition.

## 5. What M4 hands the program

The complete differentiable phase-field stack (bricks, adaptivity both
axes, ternary evaporation films, device-bound marches) + the learned-
thermodynamics demonstration with its gauge/identifiability structure
mapped + the paper spine: replication table + speedup + first learned
free energy + the identifiability story.
