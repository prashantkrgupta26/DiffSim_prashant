# P2-R1c — 3-D thin-plate PROJECTION + two-sided shell + gpu_cg PPE (runbook)

**Date:** 2026-07-25
**Status:** scalable infrastructure LANDED; outflow-BC p′-scheme fix APPLIED
(Cd now positive + non-diverging); residual ~40% magnitude gap vs monolithic
(deeper defect) remains.

## What landed

The scalable projection path for the 3-D thin-plate case:

- `src/diffsim/steppers/leray_sbm.py::LeraySBMShellStepper` — the two-sided
  co-dim-1 thin-shell SBM projection stepper. Assembles the predictor no-slip
  from `sbm_vector_dirichlet_twosided` (both Γ̃+ and Γ̃−), drives the SPD
  pressure-Poisson (PPE) with `ppe_solver="gpu_cg"` (torch.sparse CG on device
  via `dist_cg.pcg + SerialComm`), and keeps the nonsymmetric Oseen predictor
  on `splu` (predictor is not SPD).
- `src/diffsim/steppers/leray.py` — `LerayProjectionStepper` gained an optional
  `ppe_solver=` knob (default `None` = use `solver` for everything, bit-for-bit
  backward compatible). When set, ONLY the symmetric PPE + mass-correction
  solves use it; the predictor stays on `solver`.
- `tests/p2r1c_thin_plate_flow_3d_projection.py` — the projection driver,
  mirroring the monolithic `tests/p2r1c_thin_plate_flow_3d.py` (same FiniteSheet
  plate, `build_adaptive_plate_mesh`, BDF2 march, `surrogate_traction` forces,
  `save_flow_run` export, same `__main__` env vars).
- `tests/test_p2r1c_thin_plate_flow_3d_projection.py` — the physics-honest
  smoke gate (see below).

## Outflow-BC p′-scheme fix (2026-07-25) — what changed

The wrong-signed / diverging 3-D Cd is **fixed** by wiring the validated
incremental van-Kan p′-scheme outflow BCs into the projection driver
(`docs/superpowers/specs/2026-07-25-projection-pprime-outflow-fix-spec.md`):

- **Whole outflow-FACE p′=0 Dirichlet** (Eq. 68d): `pressure_outflow_nodes` is
  now the FREE-node set of the entire outlet plane `x = x_max` (a 2-D sheet of
  nodes in 3-D) — NOT the single enclosed-flow corner pin the driver used
  before. The single-corner pin left the open-outflow pressure floating, so the
  incremental p\* drifted and the predictor never built the driving pressure
  (wrong-signed, diverging Cd — `leray.py:215-221`).
- **`consistent_projection=True`** (PSPG-consistent PPE + rotational
  incremental update + backflow) — makes the monolithic steady state a fixed
  point of the split's incompressibility.
- **`inner_iterate=True`** (stabilized within-step predictor↔PPE fixed point) —
  without it the consistent-projection Cd OSCILLATES step-to-step (weak fixed
  point); with it the Cd decays cleanly.
- **`rotational_pin_wall=True`** (FN4) — now forwarded through
  `LeraySBMShellStepper` to the base; pins the rotational `−ν q` correction on
  the plate shell nodes so the plate-surface stagnation pressure jump develops
  with the RIGHT SIGN (positive drag). Without it the plate pressure jump is
  reversed and the drag is a near-total cancellation → ~0.

**RESULT (L3 uniform, dt=0.01, nu=0.1, splu==gpu_cg):** the projection Cd is now
POSITIVE and DECAYING (e.g. 229 → 27.6 over 12 steps), tracking the monolithic
startup transient (237 → 45.8) in sign and shape.

### ⚠️ Residual gap (honest) — deeper defect, NOT the outflow-BC wiring

The split's **converged** fixed-point Cd is still **~40% below** the monolithic
on the same mesh (proj → +27.6 vs mono → +45.8; ratio ≈ 0.60). The plate-surface
pressure jump is now the right sign but too small in magnitude. Pushing the
inner iteration harder (inner_max 8→25, Anderson, tol 1e-10) does NOT close it —
it converges to the SAME 27.6. This is a **structural** difference between the
split's fixed point and the monolithic SBM-NS saddle for the two-sided immersed
shell (the deeper `p2-r2a-monolithic-pivot` defect), NOT the outflow-BC wiring
gap that this fix closed.

- **Quantitatively faithful 3-D engine:** the MONOLITHIC SBM-NS saddle
  (`tests/p2r1c_thin_plate_flow_3d.py`). Use it for a magnitude-accurate Cd.
- **This projection path:** now gives the correct-SIGN, non-diverging, SCALABLE
  (gpu_cg PPE) drag whose magnitude is ~40% low — the residual magnitude fix is
  the remaining `p2-r2a` research item.

## Smoke gate — what it asserts (and deliberately does NOT)

`tests/test_p2r1c_thin_plate_flow_3d_projection.py` (L3 uniform, fast on Mac CPU)
now asserts the real physics the fix delivers:

1. **positive + non-diverging Cd** — every step `> 0` and `Cd[-1] <= Cd[0]`
   (decaying, not diverging), PLUS the projection final Cd is the SAME SIGN as
   the monolithic reference on the same mesh and within a generous band
   (0.3× .. 1.2×) — sign + shape agreement.
2. **PPE provably ran on gpu_cg AND converged** — spies on
   `diffsim.solvers.dist_cg.pcg`, asserts it was called and every call reported
   `converged=True`.
3. **two-sided coupling is load-bearing** — the two-sided force differs
   materially from the one-sided (drop-Γ̃+) force (measured rel-diff ≫ 5%).

It does NOT assert tight magnitude agreement (projection ≈ monolithic) — the
~40% gap above is expected and is the deeper defect, not a wiring failure.

## Run commands

Mac CPU smoke (green):
```
.venv/bin/python -m pytest tests/test_p2r1c_thin_plate_flow_3d_projection.py \
    tests/test_p2r0_projection_sbm.py -q
```

Local quick driver run (L3, 3 steps):
```
.venv/bin/python tests/p2r1c_thin_plate_flow_3d_projection.py
```

GH200 hero run (L6 = 64³ cells, 200 steps, gpu_cg on the PPE) — **Baskar drives
this, not CI**:
```
LEVEL=6 NSTEPS=200 DT=0.005 NU=0.004 U_INF=1.0 PPE_SOLVER=gpu_cg \
    .venv/bin/python tests/p2r1c_thin_plate_flow_3d_projection.py
```
Adaptive-mesh variant (uniform base + plate-band refinement to L9):
```
BASE_LEVEL=6 REFINE_LEVEL=9 NSTEPS=200 DT=0.005 NU=0.004 PPE_SOLVER=gpu_cg \
    .venv/bin/python tests/p2r1c_thin_plate_flow_3d_projection.py
```

## Open follow-ons

- **Predictor scalability (still splu-bound):** the Oseen predictor still uses
  `splu`. ONLY the PPE scales via gpu_cg; at L9 the predictor becomes the
  bottleneck — port it to `fused` (device BiCGStab) or AMGX (the L9 predictor
  follow-on). The PPE gpu_cg lever removes only the PPE factorization wall.
- **Residual ~40% magnitude gap (deeper `p2-r2a` defect):** the outflow-BC
  p′-scheme fixed the SIGN + divergence; the split's converged fixed-point Cd
  magnitude is still ~40% below the monolithic for the two-sided immersed shell.
  This is the remaining research item (the split's fixed point ≠ the monolithic
  saddle) — NOT closable by harder inner iteration.
