# Truck No-Shell Bring-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the ThinShell.pdf truck case `NewRun-no-shell-slope0p25` (23-body STL SBM, incomplete-octree channel, ~10–12M DOF) on the GH200 with the merged engine, producing validated forces and the standard automotive visualizations — then accelerate it with incremental assembly and a mesh-sequenced transient.

**Architecture:** Per the approved spec (`docs/superpowers/specs/2026-07-30-truck-noshell-bringup-design.md`): incomplete octrees (dyadic slab carve from the 16³ cube; strong wall BCs by coordinate masks; NO coordinate transforms), truck carved via a new N-ary `UnionList` over `TriMeshOracle` bodies with the SBM surrogate at the carve, marching on the monolithic engine (`fgmres_bdiag`, device-resident CSR, warm-start; restart per adaptive guidance = 60). Interactive validation on GH200 hold 11783474 (48 h, queued).

**Tech Stack:** `TriMeshOracle` (geometry/trimesh.py), `Box`/CSG + `classify_lambda` (the carve), `refine_elements`+`balance2to1`, `run_flow_past_3d`'s machinery as the template (but a NEW driver file — the truck driver must not destabilize the validation driver), `DeviceNSAssembler` handoff path, viz stack (meshio/pyvista/pvsm).

## Global Constraints

- Branch `truck-bringup` off master (ab446af); one implementer at a time unless the controller partitions disjoint files; `.venv/bin/python`; GPU via hold 11783474 (`srun --jobid --overlap`, standard nova-arm env, one leg per process, timeout-capped, never cancel holds or others' jobs).
- Existing gates stay green at every commit: tests/test_device_assembly.py, test_saddle_ladder_cpu.py, test_saddle_precond.py, test_p2r1a_thin_plate_flow.py, test_p2r1c_thin_plate_flow_3d.py (timeout 600000).
- All new knobs opt-in; anything touching shared solver/assembly code needs the house parity discipline (bit-identical or 1e-14-trajectory tests).
- Physical fidelity to the config: geometry files + placements verbatim from `local_code_old/truck_4case_fresh_inputs/NewRun-no-shell-slope0p25/` (position [0.0,-0.002,-7.0] applies to the STL coordinates — VERIFY the STL native frame vs the [0,16]×[0,2]×[0,2] domain by bounding-box inspection FIRST; document the mapping).
- Commits: explicit paths; trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task T1: `UnionList` N-ary oracle

**Files:** Modify `src/diffsim/geometry/csg.py` (or a new `union_list.py` if csg.py conventions argue for it — implementer's call, justified); Test `tests/test_union_list.py` (new).
**Produces:** `UnionList(oracles: list[SDFOracle])` implementing the full SDFOracle protocol: `psi`/`classify` = min over members; `distance_vector(pts, y0=None)` routes each point to its argmin-|psi| member and returns that member's (d, n, ok) (index-array gather; y0 forwarded per-member where the member accepts it); `params` = concatenation; `near_eikonal` = all(members); `dim` consistency asserted; `velocity` = the argmin member's (static → zeros).
Steps (TDD): failing tests first — (1) two-sphere union vs analytic min-distance at points near/inside/far; (2) 3-body classify + distance_vector vs brute-force per-member min (including points equidistant-ish between bodies — assert the chosen member's result, tolerance-documented); (3) a 2-body `TriMeshOracle` union (two icospheres via the existing `icosphere` helper) exercising the BVH path. Implement; gates; commit `feat(geometry): UnionList N-ary SDF union` + trailer.

### Task T2: truck driver + config loader

**Files:** Create `tests/truck_flow.py` (driver) + `src/diffsim/cases/truck_config.py` (loader; new `cases/` package); Test `tests/test_truck_flow.py` (CPU-scale gates).
**Consumes:** `UnionList` (T1); the incomplete-octree recipe: `build_uniform(level, dim=3)` on the cube → slab retain via `classify_lambda(tree, Box(channel_in_unit_coords), lam, domain="inside")` (VERIFY classify_lambda's retain semantics vs the SBM usage in p2r1c — the slab boundary is dyadic so intercepted cells shouldn't exist; assert exactly that) → region-refine boxes (`refine_elements`+`balance2to1`) → truck-band refine via UnionList psi bands → truck carve (`domain="outside"`) → two-sided?? NO: VOLUMETRIC one-sided SBM (the p2r1c bluff-body pattern, `sbm_vector_dirichlet` one-sided — the truck is solid, not a shell).
**Produces:** `run_truck(cfg, nsteps, device, assembly, mono_solver, saddle_x0, nu_schedule=None, on_step=None, verbose)` returning history dict (Cd/Cl on the truck via the consistent-reaction arbiter pattern — port the reaction functional call from the 2-D driver as the canonical force observable, plate-indicator replaced by truck-enclosing indicator); `load_truck_config(path) -> TruckConfig` (dataclass: geometry list with per-body STL path/position/refine_lvl, channel_mesh, region_refine list, BC block, Re ramp params, dt/output params — parse ONLY what the driver consumes; unknown keys warn-and-ignore, listed in the test).
Strong BCs by coordinate masks at the dyadic planes: x=0 inflow (uniform [1,0,0] Phase 1), x=1 outlet (do-nothing + pin, the house pattern), y=0 ground no-slip, y=2/16 top moving-wall [1,0,0], z walls moving-wall [1,0,0] (per config's ABL sides — Phase 1 approximates ABL as uniform; recorded). `nu_schedule`: callable t→nu implementing time-based Re ramp from the config's Re_V/Re_ramping (solver-effort variant = Phase 2).
CPU gates: (1) tiny truck-in-a-box (level 5, ONE tire STL as the body) — mesh builds, slab carve exact (zero intercepted slab cells), march 3 steps finite with reaction Cd recorded; (2) config loader round-trip on the REAL config.txt (23 geometries parsed, boxes parsed, unknown-keys warning list matches expectation); (3) BC masks: assert node counts on each dyadic plane > 0 and disjoint. Commit(s) staged (loader; mesh+carve; march) + trailer.

### Task T3: viz V1 — field export

**Files:** Modify `src/diffsim/viz/export.py` (+ results.py plumbing as needed); driver hook in `tests/truck_flow.py` (`viz_interval`, `viz_dir` kwargs); Test `tests/test_truck_viz.py`.
**Produces:** per-interval export: `.vtu` of the carved octree (hex cells incl. hanging-node nonconformity) with point fields u(3), p, |u|, and Q-criterion — Q computed DEVICE-side (a small Warp kernel on the element-gradient machinery; VERIFY what gradient evaluation exists in postproc/ or assembly — reuse; else nodal least-squares from element gradients, documented) — plus truck-body `.vtp` per STL with sampled surface pressure (sample p at surrogate-face centroids mapped to body triangles via TriMeshOracle closest-point — approximate is fine, labeled); time-averaged (u,p) accumulators with `AverageOutput`-style interval; ROI clip option (box) to bound frame size.
CPU gate: tiny-case export → files exist, meshio re-reads them, field ranges finite, Q antisymmetry sanity (Q of a pure shear ≠ of a pure rotation — one crafted-field unit check on the kernel); visual verification deferred to T4 (ParaView on the box output). Commit + trailer.

### Task T4: GH200 smoke gate (the first truck run)

Campaign task on hold 11783474 (when RUNNING; if still queued when T1–T3 land, run a reduced smoke on gpubox 48 GiB first — level-6 base variant — and record). Ship branch by bundle. Legs (separate processes, timeout-capped, samplers):
1. **Mesh+carve probe**: build the full config mesh, report node/DOF count vs the ~10–12M estimate, slab-carve exactness (zero intercepted), per-body band cell counts, build wall time, host RSS.
2. **50-step smoke**: `SADDLE_DEVICE_CSR=1`, bdiag, warm-start, device assembly, nu_schedule ramp start; gates: bounded (early-exit |Cd|>1e3 hook from the house pattern), finite reaction forces, s/step recorded with the ASM_PROFILE breakdown for one step.
3. **Viz smoke**: viz_interval=10 on the 50-step leg; fetch one frame set; the implementer VISUALLY inspects the vtu/vtp in ParaView-headless (pvpython screenshot if available on any box; else structural checks + controller review of the files).
Record all in `docs/dev/2026-07-30-truck-campaign.md` (new; the truck record). DECISION POINT for the controller: smoke verdict → proceed to T5/T6 or fix cycle.

### Task T5: T-B4 incremental assembly

**Files:** Modify `src/diffsim/assembly/device_assembly.py` + the driver; Test extend `tests/test_device_assembly.py`.
VERIFY-FIRST (the spec's honesty clause): derive from `ns_bricks`/assembler which per-step terms are velocity-dependent (convection + any τ(u)-weighted VMS terms) vs static (mass σ-block, viscous, pressure blocks, ALL SBM face terms if τ-independent — determine exactly). Produce: `fill_static()` once per mesh/BDF-order + `fill_dynamic(u)` per step adding into a device values baseline (values-add on identical sparsity; the static baseline kept as a device copy, dynamic = memcpy(base)+add-kernels). Opt-in knob; parity gate: bit-identical assembled values vs full fill at every step of a 5-step CPU march AND one GPU step. Budget ~200 lines; STOP-and-report if the τ-coupling makes the static set too small to pay (that finding would redirect effort to T6). GPU leg: per-step assembly time before/after on the truck mesh. Commit + trailer.

### Task T6: T-A1 mesh-sequenced transient

**Files:** Driver-level (`tests/truck_flow.py`) + `src/diffsim/mesh/` interpolation helper; Tests in `tests/test_truck_flow.py` + a small interpolation unit test.
VERIFY-FIRST: existing nested-tree interpolation machinery in mesh/ (grep prolong/interp); else implement p1 nodal injection+prolongation for nested carved trees (coarse nodes are a subset — direct copy; fine-only nodes — parent-cell p1 interpolation; document hanging-node handling).
Produce: `run_truck_sequenced(cfg, coarse_delta=1, transient_steps, production_steps, ...)` — coarse-mesh ramp march → interpolate (u,p) + BDF history → production march seeded (x0 + history). Gate: on a SMALL sequenced-vs-direct pair (tiny case, short horizons), final-force statistics agree within documented tolerance; wall-time saving measured. GPU: the truck ramp on coarse (level-6 base variant) → handoff → 200 production steps; record the total-time arithmetic. Commit + trailer.

### Task T7: production probe + V2 + record

On the hold: the assembled best config (T5+T6 as landed) — 2,000+ step production probe with viz output (interval per config), time-averaged fields, checkpoint saves; `pvsm`/pvpython macro for the five standard shots (centerline slice, ground-plane slice, Q-isosurface colored by |u| + surface-Cp composite, wake streamlines) committed under `docs/viz/truck/` with generated PNGs fetched into the campaign doc. Verdict vs the spec gate: bounded march, physical forces (Cd trace plausible vs ThinShell.pdf truck discussion), s/step vs the ~15 s projection, viz delivered. Full regression; final whole-branch review; merge decision to Baskar. Phase-2 items (ABL, slope, pressure outlet, solver-effort ramp, shell cases) recorded as the next campaign.

## Self-Review

Spec coverage: T-U→T1, T-D→T2, T-V1→T3, T-G smoke→T4, T-B4→T5, T-A1→T6, T-G production+T-V2→T7; Phase 2 explicitly deferred with list. Verify-first items name oracles (STL frame mapping, classify retain semantics, gradient machinery, τ-coupling split, interpolation machinery). Interfaces: `UnionList` defined T1 consumed T2; `run_truck` signature defined T2 consumed T3–T7; force observable = consistent reaction (the adopted canonical). Types consistent. No placeholders; the two STOP-and-report escapes (τ-split too small; oversized changes) are explicit.
