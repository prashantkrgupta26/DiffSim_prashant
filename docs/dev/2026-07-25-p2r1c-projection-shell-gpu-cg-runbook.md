# P2-R1c — 3-D thin-plate PROJECTION + two-sided shell + gpu_cg PPE (runbook)

**Date:** 2026-07-25
**Status:** scalable infrastructure LANDED; 3-D physics fix pending (separate track).

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

## ⚠️ Physics caveat — READ THIS

**The 3-D projection Cd produced here is FINITE but NOT physically faithful.**

The lagged-pressure projection split has a **documented 3-D defect**: with an
open outflow, the momentum predictor cannot build the driving stagnation
pressure from rest, so the split settles into a weak / wrong steady state (the
monolithic steady state is NOT a fixed point of the lagged-p\* split;
‖L − K_p‖/‖K_p‖ ≈ 0.67 — the projection K_p and the monolithic PSPG enforce
different discrete incompressibility). On the tiny L3 smoke the projection Cd is
wrong-signed and grows under the startup transient, while the monolithic driver
on the identical geometry gives correct positive drag.

- **Correct 3-D engine:** the MONOLITHIC SBM-NS saddle solve
  (`tests/p2r1c_thin_plate_flow_3d.py`). Use it for any faithful 3-D Cd.
- **This projection path:** delivers the SCALABLE gpu_cg PPE lever (the escape
  from the host-splu wall at L9-near-plate, ~186k nodes) ahead of the physics
  fix — the fix is exactly what will scale on this lever.
- **The physics fix** (consistent PPE operator + outflow-BC / pressure-
  correction p′ co-design — Baskar's scheme) is a **separate research track**.
  See the `p2-r2a-monolithic-pivot` memory verdict and
  `docs/dev/2026-07-23-projection-ladder-verdict.md`.

## Smoke gate — what it asserts (and deliberately does NOT)

`tests/test_p2r1c_thin_plate_flow_3d_projection.py` (L3 uniform, 3 steps, fast
on Mac CPU) asserts ONLY the wiring guarantees:

1. **runs end-to-end + FINITE Cd** — `np.isfinite`, not `Cd > 0`.
2. **PPE provably ran on gpu_cg AND converged** — spies on
   `diffsim.solvers.dist_cg.pcg`, asserts it was called and every call reported
   `converged=True` (measured: 3 calls, ~47–48 CG iterations each).
3. **two-sided coupling is load-bearing** — the two-sided force differs
   materially from the one-sided (drop-Γ̃+) force (measured rel-diff ≫ 5%).

It does NOT assert `Cd > 0` or projection ≈ monolithic — both are blocked by
the projection-split defect above.

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

- **Predictor scalability:** the Oseen predictor still uses `splu`. At L9 the
  predictor becomes the bottleneck; port it to `fused` (device BiCGStab) or
  AMGX. The PPE gpu_cg lever removes only the PPE factorization wall.
- **3-D physics fix:** the pressure-correction (p′) / outflow-BC scheme — the
  separate research track that makes the projection Cd faithful.
