# Leak-Drag Discriminator — Design

**Date:** 2026-07-26. **Approved:** Baskar (brainstorm 2026-07-26).
**Question:** the corrected Re=250 thin-plate runs give Cd 5.1–5.7 vs literature 3.36
(mesh-converged at r9→r10; averaging-window-insensitive; confinement moves it only
partway). Is the excess (a) real flow drag, (b) an overestimate by the
`surrogate_traction` observable, and/or (c) α-penalty leakage through the shell?

**Method:** measure drag a second, SBM-independent way — a time-averaged
control-volume momentum balance on faces far from the plate — and cross it with an
α-sensitivity sweep. Two independent axes give a 2×2 attribution.

## Components

### 1. `src/diffsim/postproc/cv_drag.py` (new, reusable observable; 2-D)

```python
def cv_drag_box(coords, u, p, box, nu, U_inf=1.0, L_ref=1.0/16.0):
    """Instantaneous control-volume drag coefficient for one rectangular box.

    Cd_CV = ∮_box [ rho*u*(u·n) + p*n_x − mu*(du/dn) ] dS / (0.5*rho*U_inf^2*L_ref)

    coords : [n,2] nodal coordinates (full mesh)
    u      : [n,2] nodal velocity;  p : [n] nodal pressure
    box    : (x0, x1, y0, y1) — all four faces must lie away from the shell band
    returns: float (instantaneous; caller accumulates the time average)
    """
```

- Face sampling: nodal values along each box line located by coordinate matching
  (tolerance h_local/4); per-face integration by trapezoid at the LOCAL node
  pitch (front/top/bottom lie in uniform-L7 territory; the rear face crosses the
  L9 wake band and samples at the band's finer pitch).
- Viscous term: central difference normal to the face using the neighbor node
  line at ±h_local.
- ρ = 1 (code units). Sign convention: drag positive downstream (+x).
- No time handling inside — pure instantaneous functional of one field.

### 2. Driver hook: `run_flow_past(..., on_step=None)`

`tests/p2r1a_thin_plate_flow.py`: after each step's solve (once `u_new`/pressure
are available), if `on_step is not None`, call
`on_step(step, t_new, u_full, p_full)` where `u_full` is the [n,2] nodal velocity
and `p_full` the [n] nodal pressure on the FULL mesh (constraint-expanded, same
arrays the VTU export path uses — reuse that assembly of full fields). Default
`None` ⇒ bit-for-bit today's behavior (same non-default-knob pattern as the rest
of the branch; regression-tested).

### 3. Probe: `tests/gpu_leakdrag_discriminator.py` (new; not pytest-collected)

- Config: the corrected octree-units Re=250 recipe (ν=U·L/250, L=1/16,
  x_c=5/16, level=7/refine 9/wake 9, dt=5e-4, 8000 steps, kick 0.03→t=0.5),
  `mono_solver=cudss, assembly=device, device=cuda:0`.
- Per step (via `on_step`), for t ≥ t_start=1.5, accumulate:
  - `Cd_CV` for THREE nested boxes with margins ≈ 4L, 6L, 8L around the plate
    (exact box edges snapped to uniform-L7 node lines; asserted to clear the
    L9 plate band),
  - `Cd_surrogate` (the driver's existing traction value for that step),
  - leak flux: net ∮ u·n over BOTH surrogate sides (reusing the driver's face
    tables — the same loop structure as `LeraySBMShellStepper.surrogate_normal_flux`).
- α-sweep: run the whole march for α ∈ {20, 50, 100} sequentially.
- Output: per-α row {Cd_surr, Cd_CV(4L), Cd_CV(6L), Cd_CV(8L), box spread,
  mean|leak|, St} + an `LEAKDRAG-OK` sentinel; histories saved to
  `results/leakdrag_a{alpha}_hist.npz`.
- Self-check: if the nested-box spread exceeds 5% of Cd_CV, the run flags
  CV-UNRELIABLE and the verdict is withheld (integration/unsteady residual must
  be understood first) — no silent verdict on bad numerics.

## Verdict table (read directly off the output)

| | α-insensitive (≤5% across sweep) | α-sensitive |
|---|---|---|
| Cd_CV ≈ Cd_surr (≤10%) | real flow drag → physics/BC investigation | leakage alters the flow → penalty scaling |
| Cd_CV < Cd_surr (>10%) | traction observable overestimates → fix integration | both |

Leak-flux magnitude per α quantifies penetration independently.

## Testing

- `tests/test_cv_drag.py` (CPU, always-on): (1) uniform freestream on a small
  built mesh ⇒ Cd_CV = 0 within 1e-10; (2) synthetic linear/quadratic field with
  hand-computed momentum flux on a box ⇒ exact match to trapezoid tolerance;
  (3) `on_step=None` bit-for-bit regression of `run_flow_past` (existing parity
  pattern); (4) `on_step` callback receives arrays of the right shapes and is
  called nsteps times.
- GPU: the probe itself, ~3 × 30 min on gpubox via `gpubox-run.sh`.

## Recording

Append the verdict table + numbers to `docs/dev/thinshell-gpu-runbook.md`
(the open-item section) and the p2r1a runbook campaign section. Ledger entry.

## Out of scope

3-D CV drag; the projection leg; fixes that the verdict may motivate (observable
correction, α rescaling — each is its own follow-up once attributed).
