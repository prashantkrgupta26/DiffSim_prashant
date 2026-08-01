# Master Cleanup Implementation Plan (post truck-bringup merge)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> Spec = `.superpowers/sdd/final-review-truck-bringup.md` section 3 ("Minor / post-merge
> cleanup") + the three nits in its Fix Verification section. This plan groups those
> items into tasks; the review doc carries the per-item detail and rationale.

**Goal:** Land every post-merge cleanup item from the truck-bringup final review on a
`master-cleanup` branch, ending with the 4-module split of `tests/truck_flow.py` into
`src/diffsim/cases/truck/`.

**Architecture:** Functional fixes first (Tasks 1–5) against the familiar file layout so
each diff is small and reviewable; the package move (Task 6) is last and pure motion.
Byte-identity discipline applies throughout: no default-path behavior change unless the
review item explicitly calls for one (asserts/logs are additive).

**Base:** branch `master-cleanup` off master `c56695e`.

## Global Constraints
- No behavior change on default paths except items that explicitly add guards/logs.
- Full suite `tests/test_truck_flow.py` (17 tests) must pass at Task 1, 5, 6 boundaries;
  targeted tests per task otherwise. Suite runtime ~4 min on Mac (CPU/splu paths).
- Do not touch `cluster/slurm/`, live-run configs, or anything under `results/`.
- Commit per task, message prefix `cleanup(<area>):`.

---

### Task 1: truck_flow.py code minors
**Files:** Modify `tests/truck_flow.py`; Test `tests/test_truck_flow.py`.
Items (locations per review §3 / Fix Verification):
1. Remove unused `import torch` in `refine_region_boxes`.
2. Delete dead `_tree`/`_anch` in backflow setup.
3. Outlet plane: replace hardcoded `x == 1.0` with the `x.max()` convention used by
   `truck_bc_masks` (identical on this mesh; unifies the convention).
4. Reword lumped-outlet-area comment: it is a kNN nearest-neighbor spacing squared
   (approximate, ~2x off at 2:1 interfaces), not a face-mass row-sum.
5. `_iter_solve`: import the real `ConvergenceError` class and use `isinstance`
   instead of `type(e).__name__` string match.
6. `interpolate_checkpoint`: read all needed npz members into memory before any
   unlink of source checkpoints (currently relies on POSIX open-handle semantics).
7. Nit I-1: `accept_miss_until` docstring — state the fgmres_bdiag-only restriction.
8. Nit I-2: on resume, also compare `n_cells` (saved but currently unchecked) when
   present in the checkpoint; fail loudly on mismatch.
9. Reaction/backflow parity guard: assert backflow outlet rows are disjoint from the
   `w_rxn` support (`w_rxn[rows] == 0`) so a future reaction-box change cannot
   silently split host/device cd_react definitions.
10. Checkpoint writer: remove the dead `dt_prev is None -> np.nan` sentinel branch
    (dt_prev is always assigned before save) and its resume-side isfinite check,
    keeping legacy-checkpoint compatibility on load.
11. Rename backflow-stab locals/attrs `_bf_rows/_bf_area/_bf_slots_d/_bf_comb_slots_d`
    (and friends) to `_obf_*` to end the collision with SBM `_bf_dofs_d/_bf_vals_d`.
Steps: implement → run full `pytest tests/test_truck_flow.py` (expect 17 passed) →
commit `cleanup(truck-flow): review §3 code minors + I-1/I-2 nits`.

### Task 2: solver/device minors
**Files:** Modify `src/diffsim/assembly/device_assembly.py`, `src/diffsim/solvers/linsolve.py`.
Items:
1. RHS-scatter duplication in `device_assembly.py` (~line 1060 pre-merge): factor the
   duplicated scatter block into one helper.
2. Remove the dead `gaq` buffer allocation (allocated, never consumed).
3. Silent-fallback log line: where the PCD fallback engages, log it explicitly
   (solver name, reason) instead of silently switching.
4. Restart auto-size suggestion: at fgmres_bdiag setup, compute bytes/Krylov vector
   (8·N, ×2 for flexible V+Z) and log a suggested max restart for the free-VRAM
   budget when the configured restart would exceed it. Log-only; never change the
   configured value.
Steps: implement → run `pytest tests/test_truck_flow.py -k "parity or warm"` →
commit `cleanup(solver): device-assembly dedup, gaq, fallback log, restart advisor`.

### Task 3: gate + shell runners
**Files:** Modify `cluster/t5_gate.py`, `cluster/t4b-run.sh`.
Items:
1. FIRST-CONTACT arrival detection: only log when the triggering step's solve
   converged (no accept-miss on that step), so the quotable line is trustworthy.
2. Tol-tighten schedule: log the *effective* step at which the t-threshold fires
   (under a dt ladder the step label otherwise misleads).
3. `t4b-run.sh:16`: first `tee` truncates its log — switch to append (`tee -a`) to
   match the t5 runner's append-only hygiene.
Steps: implement → syntax-check both (`python -m py_compile`, `bash -n`) →
commit `cleanup(gate): trustworthy first-contact, effective tol step, t4b tee -a`.

### Task 4: viz/tools
**Files:** Modify `tools/truck_viz.py`, `tools/render_frames.py`, `tools/encode_video.sh`,
`tools/render_mesh.py`, `tools/merged_trimesh.py` (paths per repo; locate by name).
Items:
1. `truck_viz` Cp: raise on `denom == 0` instead of silent zeros.
2. `render_frames` `_scan_range`: log filename+reason per swallowed exception; guard
   colormapping against NaN fields.
3. `encode_video.sh`: keep ffmpeg failure output (drop the `tail -5` pipe on error).
4. `render_mesh.py`: clear message on missing input file.
5. `merged_trimesh.py`: document the `np.round(flat, 8)` unit-scale dedup assumption;
   below ~1e-14 query distance fall back to the triangle geometric normal instead of
   normalizing a near-zero vector.
Steps: implement → py_compile all + run any existing tool tests →
commit `cleanup(tools): viz error-handling and doc fixes`.

### Task 5: test strengthening (I-8 softenings)
**Files:** Modify `tests/test_truck_flow.py`.
Items:
1. `test_interpolate_checkpoint_identity`: upgrade `allclose(atol=1e-12)` to bit-exact
   `assert_array_equal` for the A→A case (ledger records A→A bit-exact held).
2. `test_dt_schedule_identity_and_variable_table` (b): assert the actual
   `(1+2r)/(1+r), -(1+r), r²/(1+r)` BDF2 coefficients from `bdf_coeffs` for a
   representative r, not just finite/bounded.
3. `test_tau_knobs_identity_and_engagement`: extend to the device path (skip cleanly
   when no GPU) so tau_scale parity is asserted where it runs in production.
Steps: implement → run full `pytest tests/test_truck_flow.py` (17 passed) →
commit `cleanup(tests): harden I-8 assertions (bit-exact interp, BDF2 table, tau parity)`.

### Task 6: package move — 4-module split
**Files:** Create `src/diffsim/cases/truck/{__init__,truck_mesh,truck_bc,truck_ckpt,truck_march}.py`;
Modify `tests/truck_flow.py` (→ thin compat shim re-exporting the public API),
`tests/test_truck_flow.py`, `cluster/t5_gate.py` (imports; remove `sys.path` hacks).
Split per review §3 seams:
- `truck_mesh.py`: `_channel_box`, `slab_carve`, `refine_region_boxes`,
  `refine_truck_band`, `refine_walls`, `refine_ground`, `_face_components`,
  `flood_fill_retain`, delta-carve/seals, `build_truck_mesh`.
- `truck_bc.py`: `truck_bc_masks`, `truck_strong_bc`, `_pressure_pin`,
  `soft_start_amp`, backflow-stab setup, `truck_reaction_set`.
- `truck_ckpt.py`: checkpoint write/load helpers, `interpolate_checkpoint`.
- `truck_march.py`: `run_truck` (loop + `_iter_solve`), `make_nu_schedule`.
Pure motion: no logic edits; imports only. The shim keeps external references
(`results` scripts, ledger snippets) working.
Steps: move → run full `pytest tests/test_truck_flow.py` (17 passed) →
`python -c "import cluster-gate-import-check"` equivalent (py_compile t5_gate) →
commit `cleanup(structure): split truck driver into src/diffsim/cases/truck/`.

### Deferred (recorded, not done)
- `slab_carve` dyadic double-classification kept — it is the M1a tripwire.
- Typed meta object for `_pcd_cache` knobs — YAGNI until next solver campaign.
- Gate unknown-knob (typo) warning — optional per review; revisit with the
  solver-escalation campaign's knob additions.

### Finish
Full suite on the branch tip → final whole-branch review (review package
`c56695e..HEAD`) → report to Baskar for merge decision.
