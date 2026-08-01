# Solver-Escalation Campaign — Design

Date: 2026-08-01. Status: approved design (Baskar), spec for planning.

## 1. Motivation and verdict being attacked

The T5 truck bring-up closed with a solver-class verdict: across Re
{10-hold, 100, 250, 500, 1000, ramps} × 5 mesh generations × 3 seal stages ×
every formulation lever, the march cascades at 55–85% inlet amplitude under
scalar-Jacobi-class saddle solves (miss-doubling ladder ~1e-3 → ~5e-3 →
~1e-2 → terminal). The C++ (nshtsbm) crossed identical physics — same p=1
basis, same SBM shift, same equal-order stabilization — with
BCGS + ASM(LU) + iterMaxBlock=5. The wall is the preconditioner, not the
mesh, the BCs, or the formulation.

**Campaign goal:** break that wall. **Closing criterion (the gate):** a
sustained full-amplitude truck march at Re=250 (see §6 for the precise
pass bar). The Re ladder to 10⁴ is a follow-on production run, not part of
this campaign.

## 2. Scope decisions (settled during brainstorm)

- **Quadratic-element band near the body: OUT.** The controlled experiment
  already exists — the C++ succeeded with identical p=1 enforcement; the
  full-amplitude chaos is the converged discrete dynamics (Picard-3 probe),
  not enforcement noise; and the failure is amplitude-locked across wall
  treatments. A p=2 band would densify the saddle exactly where it is
  stiffest while requiring p-nonconforming constraints and new device
  kernels. **Ledgered as a post-gate accuracy experiment** with
  prerequisites: a converging full-amplitude march to measure on; cheaper
  p=1 levers tried first (second-order Taylor shift term, deeper h-bands /
  band-13 after the M1a assert is addressed, improved shift gradients).
- **Ahmed body: IN**, as the middle rung (clean-geometry solver validation).
  Honest caveat recorded: classic Ahmed Cd literature is Re~10⁶; at Re=250
  the rung validates solver survival on clean geometry, not Cd. Literature
  comparison belongs to the production phase.
- **Lead approach: PCD-as-primary** (evidence: the PCD fallback converged on
  the actual failing truck systems; its problem was Jacobi-inner cost, and
  Track A2 already built the AMGX F-inner and AMG-on-Ap that collapsed the
  inner cost 16.7×). **ASM(LU) is the pre-scoped reserve** (§8), built only
  on failure evidence.

## 3. Campaign shape — gates in order

```
Rung 0  Geometry QA gate (human sign-off, per body)   — blocks rungs 2/3
Rung 1  Offline snapshot lab (preconditioner iteration engine)
Rung 2  Ahmed body, Re=250, full amplitude
Rung 3  Truck, Re=250, full amplitude  == campaign gate
Reserve ASM(LU) (paper-scoped now; built only if Rung 1 or 3 fails)
```

Each rung starts only when the previous passes. Every rung's verdict is
ledgered in `.superpowers/sdd/progress.md`.

## 4. Rung 0 — geometry QA gate (human-in-the-loop)

Purpose: certify the carved body has no spurious geometric artifacts before
any march spends GH200 hours on it. Applies to BOTH bodies (truck sealed
mesh; Ahmed).

Render package per body, delivered to `results/mesh_renders/` on the Mac:

1. **STL-in-domain render** (new, small extension to `tools/render_mesh.py`):
   the body STL (consolidated `Truck.stl` with its config placement; the
   generated Ahmed STL) placed inside the channel with the domain box and
   ground plane drawn for scale; ≥3 camera angles (iso, side, front).
2. **Mesh slices** (existing pipeline): level-colored z-plane slices at the
   body centerline plus ≥2 offsets through the body; the front-crop zoom
   view (the view that exposed the T5 fragmentation); y-slice at ground
   height; underbody 3-D crop; FULL `.vtu` export for ParaView.
3. **Automated checks printed in the render log** (all already implemented
   in the mesh pipeline; the render script re-asserts them): exactly one
   fluid component under face-adjacency flood-fill; zero pocket cells
   outside the body bbox; surrogate-face count sanity (nonzero,
   no partially-exposed-face M1a violation); carve delta in effect.

**Sign-off protocol:** Baskar reviews the package; approval is recorded as
an explicit ledger line naming the mesh dump file. Rungs 2/3 may not launch
a march on a mesh without its ledgered sign-off.

## 5. Rung 1 — offline snapshot lab

Purpose: iterate preconditioners in minutes, not GH200 legs, on the exact
systems that killed the campaign.

**Snapshot extraction.** Add a `dump_system` knob to the march driver
(`diffsim.cases.truck.truck_march.run_truck`): at a requested step list,
dump the assembled saddle CSR (A), RHS (b), the advection state, tolerance
in force, and mesh metadata to `.npz`. Snapshots are produced by replaying
short marches from existing Nova checkpoints (t5-sealed and any surviving
leg checkpoints) — or fresh short marches to the crest with checkpointing
on — capturing (a) at least 3 systems where fgmres_bdiag missed
(ACCEPT-MISS or BUDGET-EXHAUSTED events) and (b) 2 systems where it
converged (controls).

**The lab.** A standalone script (`cluster/solver_lab.py`) loads a snapshot
and runs candidate configs, reusing the Track A2 inner-solve telemetry:

- Control A: `fgmres_bdiag` at the leg's settings (must reproduce the miss —
  this validates the snapshot).
- Control B: `fgmres_pcd`, Jacobi inners (the slow fallback).
- Candidate: `fgmres_pcd` with the Track A2 AMGX F-inner and AMG-on-Ap
  paths, sweeping inner budgets/tolerances and outer restart.

Recorded per run: converged?, relres trace, outer iters, inner iters per
apply, wall time, peak VRAM.

**Pass bar:** the candidate converges every captured miss-system to the tol
that was in force at the failing step, at wall cost ≤2× the bdiag per-step
time of the same leg (misses included). If it passes only at >2×, the rung
reports the measured cost and the decision escalates to Baskar.

**In-rung escalation — convective Fp.** The current PCD Fp is
reaction–diffusion only (geometry-built Mp/Ap with σ, ν updates; no
convection). Trigger for adding the convective term: inners converge but
the outer stagnates on miss-systems. Wire-and-measure first; upgrade only
on that signature.

## 6. Rungs 2 and 3 — the marches, and the precise pass bar

**Definitions.** "Full amplitude" = inlet amplitude factor 1.0 (soft-start
permitted, ≤300 steps, to reach it). "Miss" = an ACCEPT-MISS or
BUDGET-EXHAUSTED event from the primary solver, or a PCD-fallback
engagement. "Post-transient" = all steps after soft-start completes.

**Pass bar (both rungs):** ≥500 consecutive steps at amplitude 1.0 with
(a) zero misses post-transient, (b) umax bounded — no monotone growth
trend and always < U_CAP/10 (i.e. far from the blow-up sentinel), and
(c) finite, sign-plausible cd_react (positive for the truck; recorded, not
gated, for Ahmed).

**Rung 2 — Ahmed.** Geometry: standard Ahmed proportions (slant 25°),
generated analytically as a watertight STL by a small in-repo script; scaled
into the same unit channel to approximately the truck's blockage ratio;
ground clearance sized ≥4 cells at band resolution (a resolvable gap);
**no support stilts** (sub-resolution features are exactly the T5 lesson).
Mesh: same body-agnostic pipeline (carve + SBM + ground refinement), band
depth chosen so the finest cells resolve the clearance gap; expected
shallower than the truck's band-12 since there are no fine features.
March at Re=250, PCD-primary with the Rung-1-winning config.

**Rung 3 — truck gate.** Fresh start (no hot resume across regime changes —
the n-leg lesson) on the sealed T5 campaign mesh generation, Re=250,
PCD-primary with the winning config. Pass = campaign closed.

## 7. Venues and operational discipline

- Rung 1: current GH200 hold (11825561, ~33 h left at spec time) when free;
  gpubox (2× RTX 6000 Ada, WSL2, `LD_LIBRARY_PATH=/usr/lib/wsl/lib`)
  fallback if the snapshot systems fit 48 GB. Snapshot files live on
  /work and are copied where needed.
- Rungs 2–3: GH200 holds (submit line ledgered; 48 h QoS).
- All T5 launch hygiene carries over verbatim: verify no live leg on the
  compute node before ANY launch (`srun --overlap` + ps filter), append-only
  timestamped logs, setsid nohup + `</dev/null`, never assume the hold is
  idle without the on-node check.
- Every leg gets a ledger entry with its verdict before the next launches.

## 8. Reserve — ASM(LU), paper scope only

Recorded now so failure evidence triggers a build, not a design session:
octree spatial-block partition (Morton-contiguous cell blocks, target
~50–100k DOF/subdomain), 1–2 cell overlap, coupled (u,p) subdomain blocks,
per-subdomain sparse LU via cuDSS (GH200 unified memory; factors held
resident; batched/sequential factorization at setup, re-factorized on ν or
dt regime changes), restricted-additive-Schwarz application, BCGS or FGMRES
outer. Build triggers: Rung 1 candidate (incl. convective-Fp escalation)
fails its pass bar, or Rung 3 fails with the Rung-1-winning config.

## 9. Deliverables and testing

- `dump_system` knob + `cluster/solver_lab.py` + snapshot set on /work.
- STL-in-domain render extension to `tools/render_mesh.py`.
- Ahmed geometry generator script + Ahmed case config; Ahmed becomes a
  permanent case (tiny-gate CPU test mirroring the truck's tiny-tire test,
  so the suite covers the second body).
- PCD-primary wiring in the march driver (primary/fallback roles, knob
  echo, telemetry rows extended with inner-iter counts).
- Two render packages + two Baskar sign-off ledger lines.
- Campaign verdict ledgered; if ASM(LU) triggers, a fresh spec for it.

Suite discipline: every new knob defaults OFF ⇒ byte-identical; new tests
must pass on the no-GPU Mac (CUDA-only legs use the skipif idiom).

## 10. Out of scope

- Re ladder past 250 (production phase), the Re~10⁶ / 100M-mesh program,
  Cd-fidelity work, the p=2 band (post-gate experiment), band-13 underbody
  (parked on the M1a assert), mixed precision.
