# P2-R1 — Thin-Shell NS-SBM CFD Validation (ThinShell.pdf reproduction)

**Date:** 2026-07-25
**Status:** APPROVED (kickoff). Reproduce the flow cases of the group's paper
"Flow Simulations Around Thin Shell Structures via Octree-based SBM" (Yang,
Shadkhah, Scovazzi, Krishnamurthy, Ganapathysubramanian), `local_code_old/ThinShell.pdf`.

## Decisions (locked with Baskar 2026-07-25)
- **Cases:** reproduce ALL the paper's flow cases; 2-D bridge first, then 3-D.
- **Solvers:** run every case with BOTH the **monolithic VMS saddle** and the
  **projection/PPE** engine — a solver comparison is a first-class deliverable.
- **Parity:** literature Cd/St now; Dendro-shell C++ parity deferred to R2.
- **Physics:** NS-only for R1 (heat/NSHT later).
- Method anchor (matches the code): two-sided surrogate, Nitsche weak form
  (α=β=20), VMS stabilization (C_M=36), **BDF2**, Re=1/ν.

## Validation targets (from ThinShell.pdf §4)

### 2-D (the bridge)
1. **Blockage by a thin plate (Re=126).** Domain [0,8]×[0,4], plate at x=4 full
   height; p=100 inlet / p=0 outlet; no-slip walls + plate (SBM). Δt=5e-3, base
   L6 / local L8. Gate: outlet flux ~0 (machine precision), pressure jump 100→0.
   *(≈ existing `p2r1_thin_plate_blocked_channel`; re-anchor to this domain/BCs.)*
2. **Flow past a thin plate, Re=126 and Re=250 (headline).** Domain [0,8]×[0,4]
   (Re=126) / [0,36]×[0,16] (Re=250), u∞=(1,0) all bdys except outlet p=0; plate
   centered at (3,2) / (5,8). Re=126: local L10, Δt=5e-5, vs carved-out ref
   (Cp, wake length, velocity/pressure jump at t=3,6). **Re=250 mesh-convergence
   (Table 1):** target **Cd≈3.29–3.45, St≈0.15** (Najjar & Balachandar Cd=3.36,
   St=0.14); base L7, wake L9, plate L9–11.
3. **Two thin plates, Re=126.** Domain [0,8]×[0,4], plates at (2,2) and (3.5,1),
   Δt=1e-3. **Table 2:** upstream Cd=**3.4255**, downstream Cd=**−0.2042**
   (downstream sits in the recirculating wake → negative mean drag).
4. **Cylinder + thin-plate attachments, Re=100.** Cylinder r=0.5 at (10,8),
   domain [0,32]×[0,16], u∞=(1,0), outlet p=0, Δt=0.025; cylinder L13, plates
   L12. Attachment angles θ∈{−π/2,−π/4,0,π/4,π/2} + double plates. **Table 3:**
   no-plate **Cd=1.311, St=0.17** (validates vs Kravchenko 1.32/0.16, Stålberg
   1.32/0.17, Posdziech 1.312/0.16); θ=0 Cd=2.308/St=0.14; full θ-sweep in table.

### 3-D (the rung goal)
5. **Flow past a thin plate (3-D).** The FiniteSheet primitive + 3-D two-sided
   shell (both validated at the geometry layer). Targets: read ThinShell.pdf
   pp.20+ for the exact 3-D plate case + Cd.
6. **Plant geometry / truck boat-tails (3-D).** Application cases — later in R1
   or R2. Read pp.20+ for specifics.

## Deliverables
- A CFD driver/harness for each case (geometry via `Plane`/`FiniteSheet` +
  two-sided shell surrogate; BDF2 march to statistically steady / shedding).
- **Cd** (drag from the surrogate traction, `surrogate_traction`) and **St**
  (shedding frequency from the Cd/Cl time history — reuse `test_cylinder_strouhal`
  St-extraction) for each case, compared to the tables above (±~5% for Cd/St).
- **Both solvers** per case: monolithic VMS saddle (`steppers/` monolithic) and
  projection/PPE (`steppers/leray.py`) — report Cd/St from each + agreement.
- In-CI smoke gates at small level (fast) + a documented full-resolution run on
  gpubox for the quantitative Cd/St numbers.

## Phasing
- **R1a** — 2-D flow past a thin plate Re=250: reproduce Table 1 Cd≈3.3/St≈0.15,
  BOTH solvers. The headline quantitative gate; anchors the harness + St
  extraction + both-solver path.
- **R1b** — the other 2-D cases (blockage re-anchor, Re=126 vs carved-out, two
  plates, cylinder+attachments).
- **R1c** — 3-D flow past a thin plate (FiniteSheet), Cd; then plant/truck.

## Notes
- Cd/St extraction: `surrogate_traction` (already in `sbm/vector.py`) for the
  force; St from the lift/drag history FFT (mirror `test_cylinder_strouhal.py`).
- The projection engine may need the thin-shell two-sided coupling wired into its
  predictor/PPE (the 2-D blocked-channel used monolithic-style assembly); part of
  R1a is confirming the projection path handles the two-sided shell.
- Reuse `benchmarks/navier-stokes/cylinder_forces.py` patterns for the force/St
  harness.

## Known geometry gap (Task-1 finding)
`Segment.psi` (2-D finite plate) is UNSIGNED → `grad psi = 0` on the segment →
singular Jacobian → NaN in `extract_two_sided_surrogate`'s Newton projection.
Task 1 works around it with a two-oracle trick (Segment classify + Plane extract),
SOUND only for VERTICAL (axis-aligned) plates. **Fix before angled plates**
(R1b cylinder-attachment theta-sweep, tilted cases): give `Segment` an analytic
`distance_vector` override + fixed-normal convention, mirroring the 3-D
`FiniteSheet` (`csg.py:224-291`). Correct the misleading `Segment` docstring too.

## R1c — 3-D prerequisite ladder (for GH200, ~L8-adaptive)
ALREADY PRESENT: `TriMeshOracle` (triangulated-surface oracle = the PLANT-geometry
consumer), 3-D shell geometry (validated), dim-generic shell assembly kernels,
AMR (`refine_elements`/`balance2to1`/hanging-node `build_constraints`),
`DeviceNSAssembler`, `gpu_cg`.
- **T1a (blocker):** the 3-D shell ASSEMBLY kernels (`sbm_vector_dirichlet_twosided`
  + `surrogate_traction`) have NEVER run with dim=3 — validate on a 3-D
  DomainManager on gpubox (warp-CUDA). #1 risk.
- **T1b (blocker):** shell surrogate on an ADAPTIVE/graded mesh + hanging-node
  constraints — never tested together; required for "adaptive".
- **T2a:** 3-D NS transient march on the adaptive shell mesh (`FiniteSheet` plate
  first).
- **T2b:** solver at L8-adaptive-3D DOF (~4-20M saddle): monolithic-cuDSS likely
  too big → **projection + `gpu_cg`** is the scalable path.
- **T3a:** assembler symbolic-setup at the adaptive node count (uniform-L7 host
  hang / L9 int32 assert — likely OK at adaptive's smaller N; confirm).
- **T3b:** plant mesh → `TriMeshOracle` → `classify_shell_intercepted`.

## Host/device split (GH200 GPU-resident PPE)
HOST: octree/mesh/constraints build, K_p **symbolic slot-map** (`_init_node_pattern`
— the L7/L8 hang + L9 int32 assert), `to_csr()` download + SPD-ification/RHS,
per-iter CG dot read-backs. DEVICE: DeviceMesh data, K_p **numeric fill** (warp),
the CG solve (torch.sparse SpMV + Jacobi + reductions).
LEVERS to push assembly on-device: (1) parallelize + int64-widen the symbolic
slot-map; (2) **drop the scipy round-trip** — CG consumes the assembler's
zero-copy `device_op()`/`CSROperator` instead of `to_csr()`+re-upload.
