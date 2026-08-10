# SP-0 CHNS Coupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build DiffSim's coupled two-phase Cahn–Hilliard–Navier–Stokes brick (variable ρ(φ)/η(φ), μ∇φ surface tension, u-advected CH), decided between monolithic and staggered coupling by a measured spike, validated on the Khanwale benchmark set, with a three-way-verified adjoint.

**Architecture:** Approach 1 from the spec (`docs/superpowers/specs/2026-08-10-sp0-chns-coupling-design.md`): shared scaffolding first (closures, CH advection, variable-coefficient NS assembly, benchmark harness, numpy mirror), then two thin coupling prototypes compared on 2-D bubble rise (Task 8 = human decision gate), then winner build-out (benchmarks → CAC A/B → adjoint → GPU/3-D). Nothing built for the spike is throwaway.

**Tech Stack:** Python, Warp kernels (pattern: `make_mpf_newton` in `src/diffsim/physics/multiphase.py:479`), scipy.sparse/splu, numpy mirror + discrete-IFT adjoint (pattern: `src/diffsim/adjoint/crystallization.py`), torch twin (`src/diffsim/adjoint/torch_twin.py`), meshio VTU/PVD (`src/diffsim/viz/export.py`), matplotlib movies.

## Global Constraints

- Branch `feat/sp0-chns` off `master`; work in `/Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim`.
- Python: `/Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim/.venv/bin/python` (Mac, CPU). GPU tasks (Task 12) run on gpubox per `~/Dropbox/work/Projects/ClaudeCode/GPU-COMPUTE-SETUP.md`.
- Do NOT break existing tests. In particular `tests/test_multiphase.py`, `tests/test_multiphase_adjoint.py`, `tests/test_ns_stepper.py` must stay green after every task touching `multiphase.py` or `ns_bricks.py`.
- Nondimensional form and symbols exactly as spec §1: Re, We, Cn, Pe, Fr; f(φ)=¼(φ²−1)²; surface tension (Cn·We)⁻¹ μ∇φ; density/viscosity ratios capped at 10³.
- τ_m computed per quadrature point from local η(φ) (spec §1 amendment).
- Parity tolerances: Warp kernel vs numpy mirror rel ≤ 1e-10 (same arithmetic, few steps); adjoint vs twin rel ≤ 1e-6; FD to its plateau.
- Scope guards (spec §6): no thermal, no non-Newtonian rheology, no toolpath machinery (spike source = fixed Gaussian blob), no contact angle.
- Commits end with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Structure

| File | Role | Task |
|---|---|---|
| `src/diffsim/physics/chns.py` | Closures (ρ/η pullback), capillary force, per-GP τ_m, later `make_chns_newton` | 1, 7 |
| `src/diffsim/physics/multiphase.py` | + optional prescribed-velocity advection in CH kernel | 2 |
| `src/diffsim/api/ns_bricks.py` | + per-GP variable ρ/η coefficient support in linear-NS assembly | 3 |
| `benchmarks/chns/cases.py`, `metrics.py`, `movie.py` | Case dataclasses, reference data, metric extractors, interface movie | 4 |
| `src/diffsim/adjoint/chns.py` | `CHNSDiscrete` numpy mirror (Task 5), `CHNSAdjoint` (Task 11) | 5, 11 |
| `src/diffsim/steppers/chns.py` | `CHNSStaggeredStepper` (6), `CHNSMonolithicStepper` (7), `CHNSStepper` facade (9) | 6, 7, 9 |
| `docs/dev/2026-08-XX-sp0-coupling-decision.md` | Spike decision memo | 8 |
| `docs/dev/2026-08-XX-sp0-interface-decision.md` | CAC-vs-CH memo | 10 |
| `tests/test_chns_scaffold.py`, `test_chns_forward.py`, `test_chns_benchmarks.py`, `test_chns_adjoint.py`, `test_chns_device.py` | Test tiers per spec §5 | 1–12 |

---

### Task 1: Closure + capillary + τ_m helpers in `physics/chns.py`

**Files:**
- Create: `src/diffsim/physics/chns.py`
- Test: `tests/test_chns_scaffold.py`

**Interfaces:**
- Produces (consumed by Tasks 3, 5, 6, 7):
  - `mix_props(phi, rho_h, rho_l, eta_h, eta_l, floor_frac=1e-3) -> (rho, eta, n_clamped)` — vectorized over any-shape `phi`; linear interpolation `a*phi + b` with `a=(hi-lo)/2, b=(hi+lo)/2`, clamped from below at `floor_frac*lo`; `n_clamped` = int count of clamped entries.
  - `capillary_gp(mu_gp, grad_phi_gp, Cn, We) -> f_gp` — `(Cn*We)**-1 * mu * grad_phi`, shapes `[ngp]`, `[ngp, dim] -> [ngp, dim]`.
  - `tau_m_gp(u_gp, rho_gp, eta_gp, h, dt, Re, Ci=(4.0, 36.0)) -> [ngp]` — per-GP VMS parameter `((Ci0/dt**2) + (Ci0*|u|/h)**2 + (Ci1*eta/(rho*Re*h**2))**2)**-0.5 / rho`, the h-form of `ns_bricks` generalized to local `rho, eta` (read `src/diffsim/api/ns_bricks.py:341-514` for the existing constant-ν form and match its constants).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_chns_scaffold.py
import numpy as np
from diffsim.physics.chns import mix_props, capillary_gp, tau_m_gp

def test_mix_props_pure_phases():
    rho, eta, nc = mix_props(np.array([1.0, -1.0]), 1000.0, 1.0, 100.0, 0.1)
    assert np.allclose(rho, [1000.0, 1.0]) and np.allclose(eta, [100.0, 0.1])
    assert nc == 0

def test_mix_props_clamps_overshoot():
    # phi = -1.2 overshoots: raw rho = -0.1*999.5+500.5 < 1.0 -> clamped to floor
    rho, eta, nc = mix_props(np.array([-1.2]), 1000.0, 1.0, 100.0, 0.1)
    assert rho[0] >= 1e-3 * 1.0 and nc == 1

def test_capillary_zero_for_uniform_phi():
    f = capillary_gp(np.array([0.7]), np.zeros((1, 2)), Cn=0.01, We=10.0)
    assert np.allclose(f, 0.0)

def test_tau_m_local_viscosity_matters():
    t_lo = tau_m_gp(np.zeros((1, 2)), np.ones(1), np.array([0.1]), 0.05, 1e-2, 35.0)
    t_hi = tau_m_gp(np.zeros((1, 2)), np.ones(1), np.array([100.0]), 0.05, 1e-2, 35.0)
    assert t_hi[0] < t_lo[0]  # stiffer viscosity -> smaller tau
```

- [ ] **Step 2:** Run `.venv/bin/python -m pytest tests/test_chns_scaffold.py -v` — expect ImportError/FAIL.
- [ ] **Step 3:** Implement the three functions in `src/diffsim/physics/chns.py` (pure numpy, docstring citing spec §1 and legacy `CHNSIntegrandsGenForm.hpp:125-217` pullback pattern). Match `ns_bricks` τ_m constants exactly — read them, don't guess.
- [ ] **Step 4:** Run the test file — expect PASS.
- [ ] **Step 5:** Commit `feat(chns): phase-mixture closures, capillary force, per-GP tau_m helpers`.

---

### Task 2: Prescribed-velocity advection in the CH kernel

**Files:**
- Modify: `src/diffsim/physics/multiphase.py` (`make_mpf_newton` at `:479`, host assembly path, `MultiPhaseStepper.__init__` at `:1531`)
- Test: `tests/test_chns_scaffold.py` (append)

**Interfaces:**
- Consumes: existing `src` Gauss-point plumbing (`src_fns` → `src_gp`, see `multiphase.py:2631` and kernel arg `src` at `:563`) as the pattern to copy.
- Produces: `MultiPhaseStepper(..., adv_gp=None)` — optional `[n_elem, nqp, dim]` array of advecting velocity at Gauss points, fixed per step (re-settable via `stepper.adv_gp = ...` between steps). When given, each φᵢ-row residual gains `+ Na * (u·∇φᵢ) * dJxW` (convective form; solenoidal u assumed — document this). `adv_gp=None` must be bit-identical to today.

- [ ] **Step 1: Write the failing test** — translation of a tanh blob under uniform u at high Pe:

```python
def test_ch_advection_translates_blob():
    # 2-D level-5 uniform mesh, M=1 K=0, uniform u=(0.5,0), Pe high (mobility tiny)
    # after t=0.2 the phi centroid must move ~0.1 in x (tol 20% — diffuse smearing)
    from diffsim.physics.multiphase import MultiPhaseStepper
    ... # build dm as in tests/test_multiphase.py::<smallest 2-D gate> (copy its mesh setup)
    adv = np.zeros((n_elem, nqp, 2)); adv[..., 0] = 0.5
    st = MultiPhaseStepper(dm, M=1, K=0, chi_aa=..., dt=2e-3, adv_gp=adv, ...)
    c0 = centroid_x(st.phi); st.march(t_end=0.2)
    assert abs((centroid_x(st.phi) - c0) - 0.1) < 0.02
```

(Copy the exact small-mesh construction from the smallest existing 2-D gate in `tests/test_multiphase.py`; `centroid_x` = φ-weighted mean of node x on the `phi>0` mask, inline in the test.)

- [ ] **Step 2:** Run — FAIL (unexpected kwarg `adv_gp`).
- [ ] **Step 3:** Implement: thread `adv` through the kernel factory exactly like `src` (new `[ngp, dim]` kernel arg; static-gate it so `adv is None` compiles the term out — same `wp.static` pattern the kernel uses for other optional blocks); add the convective term to the φ-row residual AND its Jacobian contribution w.r.t. φ (u is prescribed → no u-derivative block). Host-side: accept `adv_gp` in `__init__`, slice per-element like `src_gp`.
- [ ] **Step 4:** Run new test AND `.venv/bin/python -m pytest tests/test_multiphase.py -x -q` — both green (no-adv path unchanged).
- [ ] **Step 5:** Commit `feat(multiphase): optional prescribed-velocity advection term in CH kernel (adv_gp)`.

---

### Task 3: Variable-coefficient NS assembly

**Files:**
- Modify: `src/diffsim/api/ns_bricks.py` (`make_linear_ns_Ae:341`, `make_linear_ns_be:455`, `assemble_linear_ns:514`)
- Test: `tests/test_chns_scaffold.py` (append)

**Interfaces:**
- Consumes: Task 1 `mix_props`, `tau_m_gp`.
- Produces: `assemble_linear_ns(..., nu, ..., nu_q_by_bin=None, rho_q_by_bin=None, f_q_by_bin=None)` — optional per-bin `[n_elem_bin, nqp]` viscosity and density arrays and `[n_elem_bin, nqp, dim]` body-force-at-GP arrays. Semantics: momentum time+convection terms scaled by `rho_q`; viscous term uses `nu_q`; τ_m per GP via `tau_m_gp`; `f_q` added to the RHS load. All-`None` → bit-identical to today (constant-ν path untouched).

- [ ] **Step 1: Write the failing test** — two-viscosity Poiseuille sanity:

```python
def test_variable_viscosity_ns_stokes_layers():
    # channel with eta=1 lower half, eta=10 upper half, driven lid:
    # assemble with nu_q_by_bin; solve one linear step; velocity in the
    # stiff layer must be smaller than in the soft layer at matched depth.
```

(Mesh + solve harness copied from `tests/test_ns_stepper.py`'s smallest cavity test; assertion on the u-profile asymmetry, not exact values.)

- [ ] **Step 2:** Run — FAIL (unexpected kwarg).
- [ ] **Step 3:** Implement: extend the two kernel factories with optional per-GP coefficient arrays (same static-gating discipline as Task 2); wire through `assemble_linear_ns`. Do not touch `LerayProjectionStepper` yet.
- [ ] **Step 4:** New test green AND `pytest tests/test_ns_stepper.py -x -q` green.
- [ ] **Step 5:** Commit `feat(ns): per-GP variable rho/eta/body-force in linear NS assembly`.

---

### Task 4: Benchmark harness — cases, metrics, movie

**Files:**
- Create: `benchmarks/chns/__init__.py`, `benchmarks/chns/cases.py`, `benchmarks/chns/metrics.py`, `benchmarks/chns/movie.py`
- Test: `tests/test_chns_scaffold.py` (append)

**Interfaces:**
- Produces:
  - `CHNSCase` dataclass: `name, dim, level, Re, We, Cn, Pe, Fr, rho_ratio, eta_ratio, t_end, dt0, ic_fn(x)->phi0, domain_aspect=(1,1)`.
  - `BUBBLE_RISE_RE35_WE10`, `BUBBLE_RISE_RE35_WE125`, `DAM_BREAK_2D`, `RT_2D` case instances with parameters transcribed from the legacy configs.
  - `metrics.centroid_y(phi, coords) -> float`, `metrics.rise_velocity(centroid_series, dt) -> np.ndarray`, `metrics.circularity(phi, coords, h) -> float` (perimeter via |∇φ| integral, area via φ>0 mask; circularity = perimeter of equal-area circle / measured perimeter).
  - `movie.write_interface_movie(snapshots, coords, path, fps=10)` — φ=0 contour per frame via `matplotlib.pyplot.contour`, assembled with `matplotlib.animation.PillowWriter` to gif.

- [ ] **Step 1: Transcribe legacy config parameters.** Extract without leaving the repo:

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim/local_code_old/AM
Z=baskargroup-chns_nonnewtonian-b4cc4ff1aaf0
for c in bubble_rise_2d/Re35We10 bubble_rise_2d/Re35We125 Dam_break_2d RT_instability/2D; do
  echo "=== $c ==="; unzip -p $Z.zip "$Z/config/$c/config.txt"; done
```

Copy Re, We, Cn, Pe, Fr, density/viscosity ratios, domain, t_end, IC geometry into `cases.py` **verbatim, with a comment naming the source config**. If a legacy ratio exceeds 10³, cap it at 10³ and record the cap in the case docstring (Global Constraints).

- [ ] **Step 2: Write failing metric tests** — analytic disk: φ = tanh circle of radius 0.15 at (0.5, 0.3) on a 128² grid → `centroid_y ≈ 0.3` (tol 1e-2), `circularity ≈ 1.0` (tol 5e-2); ellipse 2:1 → circularity < 0.95.
- [ ] **Step 3:** Run — FAIL. Implement `cases.py`, `metrics.py`, `movie.py`. Movie smoke test: 3 synthetic frames → gif file exists, nonzero size.
- [ ] **Step 4:** Tests green.
- [ ] **Step 5:** Commit `feat(chns-bench): benchmark cases from legacy configs, Hysing metrics, interface movie writer`.

---

### Task 5: `CHNSDiscrete` numpy mirror (monolithic ground truth)

**Files:**
- Create: `src/diffsim/adjoint/chns.py`
- Test: `tests/test_chns_forward.py`

**Interfaces:**
- Consumes: Task 1 helpers; mesh/basis utilities as used by `MultiCHDiscrete` (`src/diffsim/adjoint/multiphase.py:150` — copy its uniform-mesh Q1 scaffolding).
- Produces: `CHNSDiscrete(level, dim, case: CHNSCase, dt, newton_tol=1e-10, newton_max=30)` with:
  - state `self.u [n,dim], self.p [n], self.phi [n], self.mu [n]`; DOF vector node-major `(u.., p, phi, mu)`;
  - `residual(x_new) -> r`, `jacobian(x_new) -> scipy.sparse` (assembled with FD-checkable structure), `step() -> dict(newton_iters=..., clamped=...)`, `march(t_end)`;
  - discretization: BDF1; convective form ρ(φⁿ⁺¹)(u·∇)u; implicit ρ(φⁿ⁺¹), η(φⁿ⁺¹) via `mix_props`; PSPG pressure stabilization + SUPG momentum with `tau_m_gp`; capillary via `capillary_gp`; gravity ρĝ/Fr², ĝ=(0,−1); CH rows = existing polynomial free energy + Cn²∇²φ + advection u·∇φ; AGG flux term J·∇u with J = −((ρ_h−ρ_l)/2)·(1/Pe)·M∇μ;
  - no VMS fine-scale beyond PSPG/SUPG (mirror is reference-grade, not performance-grade).

- [ ] **Step 1: Failing test — Jacobian vs FD** on a level-3 2-D mesh, random admissible state: `‖(J v − (R(x+εv)−R(x−εv))/2ε)‖ / ‖Jv‖ < 1e-6` for 3 random v.
- [ ] **Step 2: Failing test — stationary drop:** tanh disk, u=0, g=0: after 5 steps `max|u| < 1e-3` (parasitic-current bound with potential-form surface tension at Cn=2h) and `|∫φ − ∫φ₀| < 1e-12` (mass flat).
- [ ] **Step 3:** Implement `CHNSDiscrete`. Newton on the full coupled system, `scipy.sparse.linalg.splu`. Pin pressure at node 0 (nullspace), as `LerayProjectionStepper` does for enclosed flow.
- [ ] **Step 4:** Both tests green. Also add mass-conservation and clamp-counter asserts inside `step()` (`guards=True` semantics, spec §6: raise if `clamped > 0.01*n` in any step).
- [ ] **Step 5:** Commit `feat(chns): CHNSDiscrete numpy mirror — coupled BDF1 residual/Jacobian, FD-verified`.

---

### Task 6: Staggered prototype

**Files:**
- Create: `src/diffsim/steppers/chns.py`
- Test: `tests/test_chns_forward.py` (append)

**Interfaces:**
- Consumes: Task 2 `adv_gp`, Task 3 variable-coefficient assembly, Task 1 helpers, Task 4 cases/metrics.
- Produces: `CHNSStaggeredStepper(dm, case: CHNSCase, dt, linsolver="splu")` with `.step()`, `.march(t_end, snap_every=None) -> list[snapshot dict]`, `.phi/.u/.p` properties. Per step: (1) CH solve via `MultiPhaseStepper` with `adv_gp` = uⁿ interpolated to GPs; (2) compute ρ/η/capillary at GPs from the fresh φ; (3) NS via `LerayProjectionStepper`-style predictor + PPE + update using Task 3 assembly (`nu_q/rho_q/f_q`). Shared protocol (duck-typed, also satisfied by Task 7): `.step()`, `.march()`, `.phi`, `.u`, `.p`, `.last_newton_iters`, `.last_clamped`.

- [ ] **Step 1: Failing smoke test:** `BUBBLE_RISE_RE35_WE10` at level 5, ratio 10, 20 steps: no NaN, mass drift < 1e-8, bubble centroid strictly rising after step 5.
- [ ] **Step 2:** Implement. Reuse `LerayProjectionStepper` internals by composition, not inheritance: instantiate it with `f_fn=None` and monkey-free explicit calls — if its internals resist per-step coefficient updates, assemble the predictor/PPE directly from `ns_bricks` functions inside this class (keep it thin; this is a prototype).
- [ ] **Step 3:** Smoke green. Record wall-time/step and Newton/Krylov iterations in the snapshot dicts (spike evidence).
- [ ] **Step 4:** Commit `feat(chns): staggered CH->NS projection prototype stepper`.

---

### Task 7: Monolithic prototype

**Files:**
- Modify: `src/diffsim/physics/chns.py` (add `make_chns_newton`), `src/diffsim/steppers/chns.py` (add `CHNSMonolithicStepper`)
- Test: `tests/test_chns_forward.py` (append)

**Interfaces:**
- Consumes: Task 1 helpers (re-expressed inside the kernel), Task 5 mirror (parity target), Task 4 cases.
- Produces: `make_chns_newton(nbf, nqp, dim)` Warp kernel factory — node-major `(u₁..u_dim, p, φ, μ)` = dim+3 DOF/node, residual+Jacobian for the same discretization as `CHNSDiscrete` (Task 5 lists every term — implement THAT, term for term); `CHNSMonolithicStepper(dm, case, dt, linsolver="splu")` satisfying the Task 6 shared protocol.

- [ ] **Step 1: Failing parity test:** level-4 2-D bubble rise, ratio 10, 3 steps, same dt/tolerances: `max_rel(phi, u, p vs CHNSDiscrete) ≤ 1e-10`.
- [ ] **Step 2:** Implement the kernel factory following `make_mpf_newton`'s structure (`multiphase.py:479`): static blocks per optional term, `dJxW` quadrature loop, node-major indexing. The term list and their Jacobian blocks are exactly Task 5's — the mirror is the executable spec. Jacobian: exact for all blocks except (allowed, matching MultiPhaseStepper precedent, `multiphase.py` docstring lines 176–183): τ_m frozen per Newton iterate.
- [ ] **Step 3:** Parity green. Then the Task 6 smoke test parameterized over both steppers (`@pytest.mark.parametrize("cls", [CHNSStaggeredStepper, CHNSMonolithicStepper])`).
- [ ] **Step 4:** Commit `feat(chns): monolithic (u,p,phi,mu) Warp kernel + stepper, mirror-parity gated`.

---

### Task 8: Spike evaluation + decision memo — **HUMAN GATE**

**Files:**
- Create: `benchmarks/chns/spike_run.py`, `docs/dev/2026-08-XX-sp0-coupling-decision.md` (XX = actual date)

**Interfaces:**
- Consumes: both prototypes, Task 4 metrics.

- [ ] **Step 1:** `spike_run.py`: for each stepper × ratio ∈ {10, 100} run `BUBBLE_RISE_RE35_WE10` (level 6, full t_end), and ratio 10³ for 50 steps (stress probe). Record: wall/step, Newton (outer) iterations, dt survived (halving events), mass drift, energy monotonicity, clamp counts, centroid/circularity vs the Khanwale/Hysing reference curves (transcribe reference point values into `cases.py` from the Khanwale 2020 paper figures — cite figure numbers).
- [ ] **Step 2:** Write the adjoint transpose plan for each (one page each in the memo): monolithic — JᵀΛ per step on the Task 7 Jacobian, parameter cotangents like `crystallization.py`; staggered — reverse chain CH-solve → predictor → PPE → update, enumerating which pieces are currently untaped (`leray_adjoint.py` covers PPE; predictor Newton and the CH↔NS hand-offs are new).
- [ ] **Step 3:** Write the GPU-path assessment: monolithic — does the `blockch` G4 2×2-block Schur pattern (`multiphase.py:329-339`) extend to the (dim+3) saddle (write the proposed block partition); staggered — map each sub-solve to an existing device solver.
- [ ] **Step 4:** Memo with the evidence table and a recommendation per the spec tiebreak (mixed evidence → monolithic; both-modes outcome admissible). Commit memo + spike script.
- [ ] **Step 5: STOP. Present the memo to Baskar for the coupling decision.** Do not start Task 9 without an explicit decision. Record the decision in the memo header.

---

### Task 9: Winner build-out — `CHNSStepper` facade, BDF2, full 2-D benchmark gates

**Files:**
- Modify: `src/diffsim/steppers/chns.py` (add `CHNSStepper` facade)
- Create: `benchmarks/chns/run_case.py`
- Test: `tests/test_chns_benchmarks.py`

**Interfaces:**
- Produces: `CHNSStepper(dm, case, dt, mode="auto", tstep="bdf1", linsolver="splu", src_fns=None)` — thin facade instantiating the decided implementation (`mode="auto"` = Task 8 winner; the loser stays importable, documented forward-only). `tstep="bdf2"` variable-step coefficients copied from `MultiPhaseStepper`'s A4b pattern. `src_fns` passthrough to the CH rows (SP-1 hook, spec §2).
- `run_case.py`: config → march → npz snapshots → VTU/PVD (`export_vtu(..., time_series=True)`) → movie → metrics JSON → overlay plot vs reference (matplotlib PNG). This is the §4 data-flow, end to end.

- [ ] **Step 1:** Failing benchmark tests (CI-sized: one level below production, truncated t_end, tolerance doubled):
  - bubble rise ×2 configs: terminal centroid + max rise velocity + min circularity within tolerance of transcribed references;
  - dam break: surge-front x(t) at 3 sampled times vs Martin–Moyce (tol 10%);
  - RT: spike-tip position at 2 times vs Khanwale (tol 10%);
  - MMS: manufactured `u=(sin πx cos πy, −cos πx sin πy)·g(t)`, `p=cos πx cos πy·g(t)`, `φ=tanh((y−0.5−0.1 sin 2πx·g(t))/√2 Cn)`, `g(t)=cos t`; forcing computed symbolically with sympy (`pytest.importorskip("sympy")`; add sympy to the venv if absent) and injected via `src_fns` + `f_q`; assert L2 convergence order ≥ 1.8 for u, φ over levels 4→5→6 at fixed small dt.
  - structural diagnostics asserted on every benchmark run: mass drift < 1e-8 (no source), energy non-increasing between consecutive unforced steps (tol 1e-10 slack), parasitic currents on the static drop < 1e-3.
- [ ] **Step 2:** Implement facade + BDF2 + `run_case.py`; iterate until CI-sized gates pass. Full-sized runs: `benchmarks/chns/run_case.py --case bubble_rise_re35_we10 --full` executed once, artifacts (PVD + movie + overlay PNG) committed under `benchmarks/chns/results/` (small: metrics JSON + PNG + gif only, no VTU).
- [ ] **Step 3:** Commit `feat(chns): CHNSStepper facade, BDF2, benchmark gates green (bubble rise, dam break, RT, MMS)`.

---

### Task 10: CAC variant + A/B with mass source

**Files:**
- Modify: `src/diffsim/physics/chns.py`, `src/diffsim/steppers/chns.py` (`interface="ch"|"cac"` knob)
- Create: `docs/dev/2026-08-XX-sp0-interface-decision.md`
- Test: `tests/test_chns_forward.py` (append)

**Interfaces:**
- Produces: `CHNSStepper(..., interface="ch")` default; `"cac"` swaps the φ-row to ∂ₜφ + ∇·(uφ) = γ[Cn∇²φ − F′(φ)/Cn − β(t)√F(φ)] with β(t) = (∫F′(φ) − S_src)/∫√F(φ) where S_src = ∫s_φ dΩ (source-respecting Lagrange multiplier, spec §3); μ-row eliminated (2 DOF/node fewer); AGG flux dropped.

- [ ] **Step 1:** Failing test: with a fixed Gaussian `src_fns` blob (amplitude A, radius 3h) both interfaces satisfy `∫φ(t) − ∫φ(0) = ∫₀ᵗ S_src dt` to rel 1e-6, and interface width (measured as the 10–90% tanh distance across a cut) stays within 20% of the analytic √2·Cn for CAC and within 40% for CH over the run.
- [ ] **Step 2:** Implement CAC in mirror first (extend `CHNSDiscrete` with `interface="cac"`), then kernel, parity-gated as in Task 7.
- [ ] **Step 3:** A/B memo: profile fidelity, conservation, cost, robustness at ratio 10³, on bubble rise + the sourced blob. Recommendation per spec (CAC favored for AM robustness unless it loses on the benchmarks). Commit memo. This decision is presentable to Baskar with the Task 11 kickoff (not a hard stop — the adjoint applies to either).
- [ ] **Step 4:** Commit `feat(chns): conservative Allen-Cahn interface variant with source-respecting multiplier + A/B memo`.

---

### Task 11: Adjoint — mirror adjoint, hand adjoint, twin, three-way gates

*Written for the monolithic path (spec tiebreak default). If Task 8 chose staggered-primary, STOP and re-plan this task against the memo's staggered transpose plan.*

**Files:**
- Modify: `src/diffsim/adjoint/chns.py` (add `CHNSAdjoint`), `src/diffsim/adjoint/torch_twin.py` (add coupled step)
- Test: `tests/test_chns_adjoint.py`

**Interfaces:**
- Consumes: Task 5/7 residual structure; `crystallization.py` `CACHAdjoint` as the structural template; `torch_twin.py` existing conventions.
- Produces: `CHNSAdjoint(mirror: CHNSDiscrete, objective="terminal_phi_mismatch"|"centroid_y")` with `.gradients(params=("rho_ratio","eta_ratio","We","mobility","Fr")) -> dict`; per-step JᵀΛ solves on the stored forward Jacobians; parameter cotangents ∂R/∂p assembled analytically (each is a re-assembly of the residual with the coefficient's partial — e.g. ∂R/∂We touches only the capillary term with factor −(Cn·We²)⁻¹μ∇φ).

- [ ] **Step 1:** Failing three-way gate, coarse bubble rise (level 4, 5 steps, ratio 10): for each of the 5 params and both objectives — hand adjoint vs central FD (3-point, ε swept {1e-4, 1e-5, 1e-6}, best plateau) rel ≤ 1e-4; hand adjoint vs torch twin rel ≤ 1e-6.
- [ ] **Step 2:** Implement the torch twin coupled step (mirror the Task 5 residual in torch, autograd through `torch.linalg.solve` on the dense small system — level 4 is small enough).
- [ ] **Step 3:** Implement `CHNSAdjoint`. Debug order (crystallization discipline): twin-vs-FD first (validates the twin), then hand-vs-twin (isolates transposition bugs).
- [ ] **Step 4:** Gates green, marked `@pytest.mark.ad`. Commit `feat(chns): discrete-IFT adjoint, three-way verified (adjoint=twin=FD) for 5 scalar params x 2 objectives`.

---

### Task 12: GPU + 3-D gate

**Files:**
- Modify: `src/diffsim/steppers/chns.py` (device wiring), `benchmarks/chns/cases.py` (add `DAM_BREAK_3D`)
- Test: `tests/test_chns_device.py`

**Interfaces:**
- Consumes: Task 8 GPU-path decision (blockch-extension or per-sub-solve); `GPU-COMPUTE-SETUP.md` for gpubox execution.

- [ ] **Step 1:** `@pytest.mark.gpu` device-parity test: level-5 2-D bubble rise, 5 steps, CUDA vs CPU rel ≤ 1e-8.
- [ ] **Step 2:** Run on gpubox (LD_LIBRARY_PATH recipe; check `nvidia-smi` first). Fix device issues (dtype, launch dims) until green.
- [ ] **Step 3:** 3-D smoke: `DAM_BREAK_3D` transcribed from legacy config, smallest level that fits cuDSS (<~1M DOF at dim+3=6 DOF/node → level ≤ 5) or the decided iterative path; 20 steps, no NaN, mass flat, VTU written. Detached-run recipe from the setup doc; artifacts (metrics JSON + one VTU snapshot + movie) copied back to Mac.
- [ ] **Step 4:** Commit `feat(chns): device parity + 3-D dam-break smoke on GPU`.

---

### Task 13: Docs + exit-gate audit

**Files:**
- Modify: `README.md` (one line under M6 block: CHNS coupling), `docs/superpowers/specs/2026-08-10-sp0-chns-coupling-design.md` (append "Delivered" status), memory files are the session controller's job, not this task's.

- [ ] **Step 1:** Audit spec §7 exit gate line by line against the delivered tests/artifacts; list any gap in the PR description instead of silently passing.
- [ ] **Step 2:** Run the full suite `.venv/bin/python -m pytest tests/ -x -q -m "not gpu"` — green.
- [ ] **Step 3:** Commit `docs(chns): SP-0 delivered — exit-gate audit + README status`.

---

## Self-Review Notes

- Spec coverage: §1 formulation → Tasks 5/7 (term list lives in Task 5); §2 components → file map; §3 spike → Tasks 6/7/8 (+ interface spike Task 10); §4 gates/data-flow → Tasks 4/9/12; §5 adjoint/testing → Tasks 5/11 + tier markers; §6 risks → guards in Tasks 5/6/9 (clamp counters, dt halving lives in the steppers' `march` copied from `MultiPhaseStepper.march:3074`); §7 exit gate → Task 13 audit.
- The Task 8 human gate and the Task 11 re-plan tripwire are deliberate: coupling choice and adjoint architecture are Baskar's decisions, not the implementer's.
- Type consistency: the shared stepper protocol (`.step/.march/.phi/.u/.p/.last_newton_iters/.last_clamped`) is defined once in Task 6 and referenced by 7/9; `CHNSCase` fields defined in Task 4 and consumed unchanged after.
