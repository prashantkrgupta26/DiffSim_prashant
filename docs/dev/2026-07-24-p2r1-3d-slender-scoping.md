# P2-R1 — 3-D Single Slender Object (NSHT-SBM-Shell): Scoping

**Date:** 2026-07-24
**Status:** SCOPING (for kickoff). The P2 hero is RATIFIED (roadmap strawman
2026-07-20); this scopes the R1 rung. Full spec at kickoff needs the chenghauy
reference read in + the decisions below.

## Where we are

- **2-D two-sided shell surrogate: VALIDATED.** `tests/test_p2r1_thin_plate.py`
  (blocked-channel gate) passes on gpubox: the co-dim-1 shell blocks the flow,
  develops a pressure jump, matches a carved thin-rectangle reference, and the
  two-sided coupling is load-bearing (drop one side → block collapses).
- **Shell surrogate machinery is dim-generic in code** (`surrogate.py`:
  `classify_shell_intercepted`, `extract_two_sided_surrogate`,
  `GeometryData` `mode="shell"`; `vector.py`: `sbm_vector_dirichlet_twosided`) —
  written with `dim = tree.dim`, `n1^dim` Gauss lattice, `2^(dim-1)` sub-faces.
  Its **3-D validation is in progress** (extending `test_sbm_shell_surrogate.py`
  to 3-D: exclusion band, narrow-band fast-path, two-sided normal/face-integral
  invariants). This is R1's foundation.
- **Monolithic 3-D NS engine** (P2-R2a pivot): the monolithic VMS saddle solver
  (cuDSS L6 + block-precond) is the R2 3-D engine; projection-split is deferred
  to research. So the 3-D flow solver exists.

## R1 goal

Navier–Stokes (+ heat, NSHT) over a **single 3-D slender object** with the
two-sided SBM shell treatment — the single-object precursor to the maize canopy
(P4) and the first end-to-end exercise toward the 100M-DOF hero.

## Roadmap rungs (from the strawman)

R0 2-D verified (≈done) → **R1 3-D single slender object** → R2 device-ported +
preconditioner study → R3 100M-DOF hero.

Declared gates: NS + thermal **MMS** (±0.10/field), **CPU/C++ parity vs the
Dendro-shell reference**, projection divergence-free tolerance, **BDF2** temporal
order, and the 100M hero on the Horizon/GH200 tier with a preconditioner-scaling
table.

## Reference (parity anchor)

`local_code_old/chenghauy-nshtsbm_shell-*.zip` (Dendrite-kt / PETSc / MPI).
Salient source to read into the spec: `NSEquation.h` (VMS-NS weak form),
`HTEquation.h` (heat), `SBMCalc.h`/`SBMMarker.h` + `ShellSurrogate2True.h` (the
shell surrogate-boundary core), the IMGA loop, `CalcVorticity`/`CalcError`.
2-D/3-D via `-DENABLE_3D -DSHBM -DIBM`.

## Decisions needed at kickoff (Baskar)

1. **Benchmark & Re.** Options for a 3-D slender-object validation with
   literature Cd/St:
   - **(a) Finite flat plate normal to flow, Re 126/250** (the parked
     "finite-plate Re126/250 Cd/St" — a natural 2-D→3-D bridge; shedding Cd/St
     against a known benchmark). Recommend as the *first* R1 validation
     (deepens R0 into transient force/shedding before full 3-D geometry).
   - (b) 3-D slender cylinder / ellipsoid / thin airfoil at a chosen Re.
   Which benchmark + Re, and is a 2-D transient Cd/St bridge (finite plate)
   wanted first, or straight to 3-D?
2. **Parity depth.** Full CPU/C++ parity vs the Dendro-shell reference on the
   chosen benchmark (as XDD gated vs DDFields), or literature Cd/St only for R1
   with Dendro parity deferred to R2? (Parity needs the zip read in + a matched
   case built — a substantial sub-task.)
3. **Thermal (NSHT).** Include heat transfer (the T in NSHT) in R1, or NS-only
   first and add HT in a follow-on? The 2-D gate is NS-only.
4. **Solver.** Monolithic VMS saddle (the R2a engine, exists) vs the projection
   PPE path (now unblocked by the NCCL-CG proof for the eventual 100M scale).
   For R1 (single object, moderate size) the monolithic is ready; the projection
   + NCCL-CG path is the 100M-hero lever (R3).
5. **Time integration.** BDF2 is a declared gate — confirm BDF2 for the transient
   shedding runs (already have a BDF2 temporal-order gate from P2-R0).

## Recommended R1 path (pending the above)

1. **3-D shell surrogate validated** (in progress) — geometric/MMS, CPU.
2. **Finite-plate 2-D transient Cd/St** (bridge): finite plate in open flow,
   BDF2 march to shedding, extract Cd + Strouhal, validate vs literature. Reuses
   the 2-D shell machinery + the existing NS stepper + `test_cylinder_strouhal.py`
   patterns for St extraction.
3. **3-D single slender object**: build the 3-D shell geometry, MMS gate
   (±0.10/field), a moderate-Re run, forces + (if HT) Nusselt.
4. **Dendro-shell parity** on the matched case (depth per decision 2).
5. Hand to R2 (device port + preconditioner study) → R3 (100M hero, projection +
   NCCL-CG on the NVLink tier).

## Immediately actionable without kickoff decisions

- The **3-D shell surrogate validation** (in flight) — pure foundation.
- **St-extraction harness** reuse from `test_cylinder_strouhal.py` /
  `benchmarks/navier-stokes/cylinder_forces.py` for the finite-plate bridge.
