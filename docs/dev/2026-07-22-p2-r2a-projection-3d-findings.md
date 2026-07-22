# P2-R2a findings — the 3-D projection failure, root-caused

**Status:** Investigation complete (2026-07-22). Decision: adopt the monolithic
solver as R2's 3-D forward engine; file the projection-split-in-3-D as a
dedicated research track. This memo is the record of the diagnostic chain, the
two controls that settled it, and the open research questions.

All runs on gpubox (host/`splu`, 40-core CPU). Fixture: level-4 octree, unit
box, shifted-boundary no-slip sphere `CTR=(0.35,0.5,0.5)`, `R=0.12`,
`D/h=3.84`, `n_free=4907`; strong inflow (x=0, `u=U_IN`) + lateral no-slip
walls; free outflow (x=1). Reference throughout: the **monolithic** SBM-NS
saddle solve on the *same mesh* (`tests/p2r0_task10_sphere_derisk.py::monolithic_cd`).

## 1. What we set out to fix

R0 delivered the VMS-incremental-projection + volumetric-SBM stepper at the
de-risk bar but flagged one open item: the projection split's drag **diverges
in 3-D** even at Stokes, while the monolithic is stable (`Cd=0.381` at the R0
Re=100 de-risk point). R2a's job was to make the 3-D projection stable and
faithful.

## 2. The diagnostic chain

Each step committed a measurement; the verdict of each drove the next.

1. **4-mechanism diagnostic** (`tests/p2r2a_diagnostic_3d.py`, `4194601`).
   Ruled out all four candidate mechanisms with machine-precision negatives:
   PPE conditioning (resid 1e-14), projection-space identity
   (`‖σBᵀû−K_pφ‖=1.2e-14`), null-space/outflow (φ-gauge drift 3.3e-13), and
   penalty magnitude (diverges through α=20000). Yet `‖p̂‖` grew ~4× over 12
   steps. Verdict: **PENALTY_NOT_THE_LEVER** — an incremental-pressure feedback,
   not any of the four.

2. **Reference study.** Two group codes were read term-by-term: **bkhara
   `ns_vms`** (VMS pressure-projection on *body-fitted* meshes — the split
   structure and the residual-based VMS fine-scale threaded through
   momentum/PPE/velocity-update, with 1st-order incremental pressure and a
   physical outflow pressure BC) and **chenghauy `nshtsbm_shell`** (monolithic
   RBVMS + *SBM*). Neither does projection **and** SBM together.

3. **Bake-off** (`tests/p2r2a_bakeoff_3d.py`, `556166d`). Swept
   `pressure_update ∈ {standard, rotational, chorin}` × `ppe_fine_scale`. **No
   stable+faithful config.** Only Chorin (`p*=0`, non-incremental, 1st-order)
   bounded `‖p̂‖`, and even it gave `Cd=−0.32`. Fine-scale consistency did *not*
   stabilise the incremental scheme (finescale-only diverged ≈ baseline);
   rotational made it worse. **Weak-divergence never decayed in any config.**

4. **Outflow BC** (`9a51441`, `1406d1e`). The stepper pinned pressure at a
   single "enclosed-flow" node — wrong for an external flow with a free
   outflow. Replacing it with Dirichlet `p=0` on the x=1 outflow face was the
   first change to make the weak-divergence **decay** and to cut the pressure
   runaway ~20×. But over 60 steps the projection Cd drifted to −0.27 while the
   converged monolithic gave **+40.04** (a valid, confined Stokes-regime value
   at Re=1) — bounded-ish but ~40× off.

5. **Field comparison** (`tests/p2r2a_field_compare.py`, `3df9128`). Over 3594
   fluid nodes: velocity rel-L2 **101%**, pressure rel-L2 **95%**;
   `‖u_proj‖=4.47` vs `‖u_mono‖=72.8` (~16× too weak); fore-aft `Δp` = −0.19
   (projection) vs +15.5 (monolithic); both drag components ≈0. Since both feed
   the *identical* `surrogate_traction`, an observable bug is impossible — **the
   whole projected flow field is wrong**, not the drag readout.

6. **Velocity-deficit probe** (`tests/p2r2a_velocity_deficit_probe.py`,
   `6d5a29d`). Strong BC is enforced exactly (inflow `ux=1.000`, walls
   `|u|=0`). The **momentum predictor `û` carries the deficit** (`‖û‖`(fluid)
   59.8× weaker than monolithic; `u_new≈û`), and the interior velocity
   **decays** step-over-step (mean `|u|`: 0.16→0.008). The projection is not
   collapsing a good predictor — the predictor itself never develops the flow.

## 3. The two controls that settled it

**Control A — the SBM block is correct.** The monolithic reference that gives
the *right* answer (`Cd=40.04`) calls the **identical `sbm_vector_dirichlet`
block** and simply adds it to the coupled NS system
(`monolithic_cd`, lines 151–178). So the velocity-only, component-diagonal SBM
Nitsche block — *without* explicit velocity↔pressure coupling — is sufficient:
the volume NS system supplies the coupling when pressure is solved
simultaneously. A stage-1 experiment that added the missing monolithic SBM
pressure-coupling terms (T3 pressure-consistency + T6 shifted no-penetration,
`4fe105a`, left flagged-off) had **negligible effect** — confirming the SBM
terms were never the cause.

**Control B — the Reynolds confound.** Every R2a diagnostic ran at **Re=1
(Stokes)**; R0's 2-D faithful cases were **Re=20–100**. Re-running the sphere
at **Re=100** changed the behaviour qualitatively: the flow **develops and
plateaus** (stable) instead of decaying, and the deficit shrank from 22× to
5.6× (`u_new`) / 60× to 13× (`û`). But it is **only partial** — even at Re=100
the projection settles at a *wrong, weak* fixed point (`Cd≈0` vs monolithic
+0.96; velocity 5–13× too weak; the plateau is a genuine steady state, not a
slow transient).

## 4. Conclusion

The 3-D failure is **the projection split, not the SBM**. The lagged-pressure
predictor never builds enough pressure to drive the flow around the immersed
body, and the split settles into a self-consistent weak fixed point
(weak `p` → weak predictor → weak divergence → weak `p`). The pathology is
worst at low Re (where pressure dominates and the pressure-lag splitting error
is largest) but is **not** eliminated at Re=100. The correct pressure gauge
(outflow Dirichlet) is necessary but not sufficient. This is the well-known
hard regime for incremental pressure-projection — a coarse-mesh immersed
boundary with a strong pressure balance — and needs a numerical-methods
remedy, not a knob.

## 5. Decision

- **R2's 3-D forward engine = the monolithic SBM-NS saddle solve.** It is
  proven correct with our *own* SBM block (`Cd=40.04` at Re=1, `+0.96` at
  Re=100 on the level-4 mesh) and unblocks the shell, the heroes, and the
  steady-adjoint — all of which need a faithful 3-D forward solve, not
  specifically the split. R2b (device port) and R2c (literature validation)
  re-scope around the monolithic path.
- **Projection-split-in-3-D → dedicated research track** (see the companion
  research item). It remains the intended 100M-scalability path (SPD
  pressure-Poisson → AMGX), so it is worth getting right — deliberately.

## 6. Open research questions (for the projection track)

1. Why does the incremental split converge to a *weak* steady state rather than
   the monolithic one? A consistent incremental scheme should share the
   monolithic fixed point — quantify the steady-state consistency error of our
   split on the immersed problem.
2. Does a **coupled / Uzawa predictor** (inner velocity↔pressure sub-iteration)
   or a **pressure-robust splitting** close the gap? The bkhara reference
   re-solves the *full* pressure each step (bounded) rather than accumulating
   an increment — worth reproducing exactly.
3. Is the pressure-Poisson's surrogate BC (homogeneous-Neumann, Suresh Rem 3.9)
   the right natural BC once the predictor genuinely enforces `ñ·ũ≈0`, and does
   the predictor ever do so on a 3-D staircase surrogate?
4. Re-dependence: characterise the deficit vs Re (1 → 100 → 300) to separate
   the splitting-error component from any residual structural defect.

## 7. Artifacts (branch `p2-r2a`)

Diagnostics (committed): `tests/p2r2a_diagnostic_3d.py`,
`tests/p2r2a_bakeoff_3d.py`, `tests/p2r2a_outflow_probe.py`,
`tests/p2r2a_field_compare.py`, `tests/p2r2a_velocity_deficit_probe.py`, and
the flagged-off `pressure_update`/`ppe_fine_scale`/`sbm_pressure_coupling`/
`pressure_outflow_nodes` knobs on `LerayProjectionStepper`/`LeraySBMStepper`.
Reports under `.superpowers/sdd/`. This memo is the synthesis.
