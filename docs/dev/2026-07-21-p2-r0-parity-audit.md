# P2-R0 §4 parity audit — `LerayProjectionStepper` vs the VMS-projection paper

**Task:** P2-R0 Task 1. **Oracle:** `local_code_old/ns_projection_vms_paper.pdf`
(VMS-stabilized incremental projection, van Kan; Algorithm 1; Eqs. (4), (29)–(33),
(43)–(45); Remarks 3.7, 3.9). **Code under audit:**
`src/diffsim/steppers/leray.py::LerayProjectionStepper` (host, numpy/`splu`).

**Verdict (headline):** the stepper **matches** the paper. All six checklist
items PASS. **No code fix was required.** Two decisions are recorded for the
supervisor (device path, `ppe_finescale`); one latent non-gate note is filed.
The confirmed invariants are pinned by `tests/test_p2r0_parity.py` (4 tests,
all green on host/`splu`).

---

## Checklist verdicts

### 1. Incremental scheme & first-order pressure extrapolation — **PASS**

* **Paper:** van Kan incremental pressure-correction (§2, Eq. (29)–(33)):
  `p* = p^{n-1}` is the *first-order extrapolation rule* (Eq. (33), text
  "we use the first order extrapolation rule … incremental pressure
  correction"). Remark 3.9: `p* = p̂^{n-1}` renders `φ^n = O(Δt)`, combining
  with the projection's `σ^{-1}=O(Δt)` to give the `O(Δt²)` pressure.
* **Code:** `step()` ends with `self.p_star = p_hat` (leray.py:309, Algorithm 1
  Step 4) — i.e. `p* ← p̂^n`, the first-order extrapolation. The predictor is
  driven by `p_node_full = self.p_star` (leray.py:174, pinned into the block at
  leray.py:207–211). The PPE unknown is the increment `φ`, and
  `p_hat = self.p_star + phi` (leray.py:285). The double-count failure mode is
  documented in the code comment at leray.py:281–284.
* **Test:** `test_incremental_increment_is_phi` — a constant shift of `p*`
  shifts the returned pressure by exactly that constant and leaves `φ` and the
  corrected velocity invariant (the defining property of the incremental /
  van-Kan scheme: `p*` enters only through `∇p*`). Also asserts
  `p_hat == st.p_star` after the step (Step 4).

### 2. BDF-r discretization and σ = β₀/Δt — **PASS**

* **Paper:** Eq. (4) — `∂u/∂t|_n = σ u^n + (1/Δt)Σ β_{-i} u^{n-i}` with
  `σ = β₀/Δt`; `u^{n-1..n-r}` known on the RHS (paper line "σ = β0/∆t … u^{n-1},
  …, u^{n-r} are known"). BDF2 used in the experiments (Remark 3.2).
* **Code:** `o = bdf_order_now(t_new, dt, order, have_history=hist.have(2))`
  (leray.py:162), `b0,b1,b2 = bdf_coeffs(o, dt)` (leray.py:164),
  `sigma = b0/self.dt` (leray.py:165), history on the RHS as
  `h_node = (b1·u1 + b2·u2)/dt` (leray.py:168–169). `bdf_coeffs(2,dt) =
  (1.5,-2,0.5)`, `bdf_coeffs(1,dt) = (1,-1,0)`; bootstrap gate returns BDF1 for
  `t < 1.5·dt` or missing 2nd-order history (timestepping.py:33–39).
* **Test:** `test_bdf_coeffs_and_bootstrap` pins the signed tables, the
  `t<1.5·dt` and `no-history` bootstrap branches, and `β₀/Δt = 30` at the gate
  `dt=0.05`.

### 3. Three subproblems match Algorithm 1 — **PASS**

* **Paper Algorithm 1:** Step 1 momentum predictor (nonlinear, driven by
  `∇p*`, VMS `τ_m` fine-scale terms); Step 2 PPE `(∇p̂,∇q) = (∇p*,∇q) −
  σ(∇·ũ,q) − σ Σ (τ_m r_m^e, ∇q)` — a coercive Poisson on pressure whose
  unknown is the increment; Step 3 velocity projection `(û,w) = (ũ,w) −
  Σ(τ_m r_m, w) − (1/σ)(∇(p̂−p*), w)`; Step 4 `p* ← p̂`.
* **Code:**
  * *Step 1* — Picard/Newton loop over the monolithic `assemble_linear_ns`
    block with strong Dirichlet rows from `g_fn` at `dir_nodes` and **all
    pressure DOFs pinned to `p*`** (leray.py:182–221). Predictor uses `p*`,
    not `p̂` (leray.py:174, 207–211). ✔
  * *Step 2* — SPD scalar Poisson `self.K_p` (leray.py:273–280), pinned at
    free-node 0 for enclosed flow (leray.py:274–276); classic-incremental RHS
    flux `σ·ũ` → `(∇q, σũ)` (leray.py:251, 259–261) so the solution is exactly
    the increment `φ` (leray.py:281–285). ✔ (paper's `−σ(∇·ũ,q)` term via
    integration by parts to the `(∇q, σũ)` flux form on enclosed Ω.)
  * *Step 3* — consistent-mass L2 projection `u = ũ − (1/σ)∇φ` (leray.py:286–305)
    via `self.M`, then strong Dirichlet on the updated trace
    (leray.py:307, "trace preserved"). ✔
  * *Step 4* — `self.p_star = p_hat` (leray.py:309). ✔
* **Test:** `test_incremental_increment_is_phi`,
  `test_predictor_pins_pressure_to_pstar`.

### 4. Fine scale supplies advective + pressure stability → equal-order — **PASS (with recorded decision)**

* **Paper:** Threefold contribution (§1) — "the fine scales of the predicted
  velocity … supply both advective and pressure stability, thereby enabling
  equal-order velocity–pressure interpolation without the inf-sup condition."
  Remark 3.7 (Pressure stability): the PPE form `a(p,q)=(∇p,∇q)` is coercive on
  `Q∩L²₀` by Poincaré, *no discrete inf-sup (LBB) needed* — the projection
  replaced the saddle point by a coercive elliptic problem; the fine scale adds
  the "projection counterpart of PSPG" residual term.
* **Code:** the predictor's residual-based fine scale (`τ_m`, SUPG/PSPG/grad-div
  contributions) lives in `assemble_linear_ns` (`api/ns_bricks.py`) with
  `sig2tau=(2σ)²` and `tau_m_metric` (`physics/vms.py`); the PPE is the SPD
  `self.K_p` — coercive equal-order Poisson, no LBB, exactly per Remark 3.7.
* **DECISION — `ppe_finescale=False` (classic incremental), recorded per
  supervisor resolution 3:** R0 uses the group's as-implemented stable default:
  the classic incremental PPE flux `σ·ũ` (leray.py:251), **NOT** the
  paper's explicit fine-scale PPE RHS `−σΣ(τ_m r_m, ∇q)` (Algorithm 1, Step 2).
  *Justification (physically grounded):* pressure stability for equal-order u/p
  comes from the **predictor's** VMS fine scale (the PSPG-like mechanism of
  Remark 3.7), so dropping the PPE fine-scale term does **not** forfeit
  equal-order stability. The draft-faithful *explicit* fine-scale PPE RHS is
  **unstable at practical Δt**: the documented finding `σ·τ_m ≈ 0.8` at gate Δt
  drives a measured blow-up (`~7e5` on the vortex MMS) because `r_m` carries
  `σ·ũ` and `∇p*` into an explicit RHS (leray.py:48–56). The draft-faithful
  variant would require the *implicit* `(1/σ + τ_m)`-weighted PPE operator
  (leray.py:100–124, `_weighted_stiffness`) which is gated behind
  `ppe_finescale=True` and not part of R0. This is a **deliberate,
  physically-justified deviation from the paper's ideal**, validated by M1b on
  exactly this `ppe_finescale=False` path (cavity/cylinder). Revisit only if a
  later hero regime demands the implicit fine-scale PPE.
* **Test:** `test_ppe_finescale_default_is_classic_incremental`.

### 5. τ form (h-based, Eq. 43/45) — **PASS (with one latent non-gate note)**

* **Paper:** Eq. (43)/(45): `τ_m = [σ² + c1|ũ|²/h² + c2 C_I ν²/h⁴]^{-1/2}`,
  the h-based (metric) form; `τ_m ∼ 1/σ` (dimensional consistency, Eq. (43)).
* **Code:** `tau_hbased_host(u_mag, h, nu, dt, dim, c1=4, c2CI=C_I·16·dim)`
  (`physics/vms.py:48–56`); `tau_m_metric` for the predictor path
  (`physics/vms.py:60–67`). Group convention: the **transient term is
  `(2·β₀/Δt)²`** (a consistent factor-of-2 in the group's `sqrt((2/Δt)²+…)`
  form — used identically in `tau_metric_host` and `tau_hbased_host`, the
  Bazilevs/Dendrite lineage), rather than the bare `σ²` printed in Eq. (43).
  This is a documented group convention, not a deviation; it is applied
  uniformly.
* **LIVE (gate-relevant) path — CORRECT:** the predictor passes
  `sig2tau=(2·sigma)² = (2·β₀/Δt)²` with `sigma=b0/dt` (leray.py:198) →
  transient term `(2σ)²`, matching σ=β₀/Δt under the group convention. ✔
* **LATENT non-gate NOTE (no fix in R0):** in the *dead* `ppe_finescale=True`
  branch, `τ_m` is computed with `dt=self.dt/b0` (leray.py:234–236), so inside
  `tau_hbased_host` `sigma=(b0/(dt/b0))=b0²/dt`, giving transient
  `(2·b0²/Δt)²` — an **extra factor of `b0`** relative to the predictor's
  `(2·b0/Δt)²`. For BDF1 (`b0=1`) they coincide; for BDF2 (`b0=1.5`) the PPE
  `τ_m` transient is over-scaled by `b0²=2.25`. This `τ_m` is **only consumed
  under `ppe_finescale=True`** (leray.py:246–249); the R0 default
  `ppe_finescale=False` uses `flux = σ·ũ` and never reads it (leray.py:251), so
  **it does not affect any R0 gate**. Filed as a latent bug to fix *if/when* the
  implicit fine-scale PPE lands; deliberately NOT fixed here (would touch
  untested dead code without a gate to guard it — out of R0's minimal-fix
  scope). Recorded for the supervisor.

### 6. Device-solve-path decision — **PASS → host/`splu` reference; device port stays R2 prerequisite**

* **Spec §8 / supervisor resolution 1:** R0 stays host/cuDSS-reference; the
  SPD-PPE→AMG/AMGX device port is the R2 prerequisite *unless the audit finds
  it free to include*.
* **Audit verdict:** the audit surfaces **no formulation reason** to build the
  device port now. All three sub-solves route through
  `solvers.linsolve.solve_linear(..., solver="splu")` (leray.py:212–215,
  268–280, 301–305), which suffices for all R0 gates at 2-D / small-3-D
  validation scale. The projection choice already yields the SPD pressure
  Poisson (`self.K_p`, Remark 3.7) that is the *scalable* win for R2, but
  nothing in the formulation makes wiring AMG/AMGX "free" — it is a separate
  solver-port effort with its own validation. **DECISION: host/`splu`
  reference for R0; SPD-PPE→AMG/AMGX device port remains the R2 prerequisite.**

---

## Summary of decisions recorded for the supervisor

1. **`ppe_finescale`:** R0 runs **`ppe_finescale=False`** (classic incremental,
   flux `σ·ũ`) — a deliberate, physically-justified deviation from Algorithm 1
   Step 2's explicit fine-scale PPE RHS (unstable at practical Δt per the
   `σ·τ_m≈0.8` finding). Pressure stability comes from the predictor's VMS fine
   scale (Remark 3.7). *(Per resolution 3.)*
2. **Device path:** **host/`splu` reference** for R0; **SPD-PPE→AMG/AMGX device
   port remains the R2 prerequisite** — the audit found no formulation reason to
   include it now. *(Per resolution 1; spec §8.)*
3. **Latent note (no R0 fix):** the `ppe_finescale=True` PPE `τ_m` uses
   `dt=Δt/b0`, over-scaling its transient term by `b0²` for BDF2. Dead in R0
   (`ppe_finescale=False`); fix when the implicit fine-scale PPE lands.

## Bottom line

`LerayProjectionStepper` **matches the VMS-projection paper** for the R0
formulation. No fix required. Deviations found: (a) the intentional
`ppe_finescale=False` classic-incremental choice (recorded, justified); (b) a
latent, non-gate `τ_m` transient over-scaling confined to the dead
`ppe_finescale=True` branch (filed, not fixed). Confirmed invariants pinned by
`tests/test_p2r0_parity.py` (4 tests green on host/`splu`).
