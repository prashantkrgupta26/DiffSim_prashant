

# NewRun-no-shell-slope0p25 on the GH200 — Design

**Directive (Baskar):** evaluate the current framework on the GH200 on the
ThinShell.pdf truck case, no-shell baseline first
(`local_code_old/truck_4case_fresh_inputs/NewRun-no-shell-slope0p25/`).
Design decisions (Baskar): INCOMPLETE OCTREES (no domain scaling) — carve the
16×2×2 channel from the 16×16×16 cube (dyadic slab ⇒ carved walls == true
boundary ⇒ strong BCs by coordinate masks); then carve the truck (UnionList of
23 TriMeshOracle bodies, SBM surrogate at the carve). Pretty visualizations
required (V1/V2 below). Engine: the MONOLITHIC path (fgmres_bdiag, restart=120,
warm-start, device assembly — post-W2 for adaptive).

## Audit-verified foundations (2026-07-29)
TriMeshOracle (wp.Mesh BVH, winding-number sign, SDFOracle protocol) —
geometry/trimesh.py:96; Box/Complement/Intersection/Translate CSG;
classify_lambda(domain=...) carve entry; generic refine_elements+balance2to1;
BDF2+rbVMS; viz stack (vtu/meshio, vtp/pyvista, pvsm). DOF estimate ~10-12M
(within the proven 9.08M envelope; nnz ~1.3B < 2^31 ⇒ W1b NOT needed).

## Workpackages
- **T-U UnionList oracle** (~100 LOC): N-ary min-union SDFOracle over
  TriMeshOracle list; classify = np.minimum.reduce; distance_vector routes per
  point to argmin-|psi| member (index array); near_eikonal = all members;
  params = concat. Tests: 2-sphere union vs analytic; 3-body classify/dv parity
  vs brute-force min.
- **T-D truck driver** (`tests/truck_flow.py` + config loader): 
  1. Parse config.txt subset (config loader ~50 LOC: geometries list, region_refine,
     channel_mesh, Re/dt/BC blocks).
  2. Base octree level 7 on the cube; slab carve via Box+classify (dyadic walls);
     region-refine boxes (levels 10/12) via refine_elements; truck-band refine
     (refine_lvl 12 near each STL surface via UnionList psi bands); balance2to1.
  3. Truck carve (domain="outside") + SBM surrogate + volumetric Nitsche
     Dirichlet (the p2r1c pattern, UnionList geometry).
  4. Strong wall BCs by coordinate masks (dyadic planes): inflow x=0 (uniform
     first; ABL Phase 2), outlet x=16 (do-nothing + pin first; pressure-
     Dirichlet Phase 2), ground y=0 (no-slip; slope Phase 2), y=2/z walls
     (moving-wall [1,0,0]).
  5. Re ramp: nu(t) schedule knob (time-based first; solver-effort variant
     Phase 2).
  6. Engine knobs: mono bdiag + SADDLE_RESTART=120 + warm-start + device
     assembly (post-W2).
- **T-V1 viz export**: per-interval vtu (u, p, |u|, device-computed
  Q-criterion), truck vtp with sampled surface p/Cp, time-averaged fields;
  interval + ROI controls (~0.5 GB/frame at 10-12M). Verify hanging-node hex
  vtu renders in ParaView on a small carved mesh first.
- **T-V2 ParaView state**: pvsm/macro for centerline slice, ground-plane
  slice, Q-isosurface colored by |u| + surface-Cp composite, wake streamlines;
  animation from checkpoint series.
- **T-G GH200 evaluation** (the gate): mesh build + 50-step smoke (bounded,
  finite forces) on the hold-or-sbatch; then a 2000+-step production probe
  with viz output; record s/step vs the ~12-15 s projection, memory, and the
  first pretty pictures.
- **Phase 2** (post-baseline): ABL log-law inflow, sloped ground
  (slopeNearGround=0.25), pressure-Dirichlet outlet, solver-effort Re ramp,
  then the three shell-attachment cases (thin_structures via the P2 two-sided
  shell machinery).

## Order
T-U → T-D (smoke on CPU tiny carve) → T-V1 → T-G smoke → T-V2 + T-G production.
W2 must land first for device assembly on the adaptive mesh (else host-asm
fallback at ~4× step cost — acceptable for the smoke, not production).

## ADDED WORKPACKAGES (Baskar 2026-07-30: "bake A1+B4 into the truck plan")

- **T-A1 Mesh-sequenced transient**: run the from-rest transient (the Re-ramp
  phase) on a one-level-coarser carved mesh (~8-10x cheaper steps); interpolate
  the developed (u,p) field to the production mesh (octree-to-octree
  interpolation — the p1 nodal embedding is natural on nested trees; VERIFY
  what interpolation machinery exists in mesh/ or build the nodal-injection +
  prolongation for nested carved trees); hand off as the warm-start x0 +
  BDF history seed; march only the averaging window at full cost.
  Gate: drag statistics from a sequenced run match a full-resolution-from-rest
  reference run (short case) within statistical error; wall-time saving
  recorded. Projected: 1.5-2x off total truck wall.
- **T-B4 Incremental assembly**: split the per-step assembly — mass, viscous,
  and ALL SBM/Nitsche face blocks are mesh-static (assemble ONCE, keep the
  device CSR values); per-step re-assemble ONLY the convection block and add
  into the static baseline (values-add on identical sparsity — the AMGX
  replace_coefficients philosophy applied to our own operator). VERIFY the
  assembler's brick structure separates convection cleanly (rbVMS tau terms
  couple u through tauM — determine which stabilization terms are
  velocity-dependent and must re-assemble vs truly static; be honest about the
  split boundary). Gate: bit-identical assembled matrix vs full reassembly at
  every step of a smoke march; per-step assembly time before/after recorded.
  Projected: turns the ~110 s/step adaptive assembly share into a small
  fraction; synergizes with W2b (the constraint-expansion scatter runs once).
