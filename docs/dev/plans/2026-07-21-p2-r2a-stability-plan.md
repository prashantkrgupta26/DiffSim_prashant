# P2-R2a — 3-D Projection Pressure-Coupling Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cure the 3-D projection stepper's Stokes divergence with the **minimal lever set that is BOTH stable AND preserves BDF2 2nd-order-in-time accuracy**, matching the working Taly `ns_vms` C++ reference (our exact scheme). The order is Baskar's: **"Do 2 then 1"** — (Task 2) implement the projection knobs, (Task 3) run a one-lever-at-a-time bake-off diagnostic to isolate the minimal stable lever set, (Task 4) lock the winner behind a DUAL gate — a stability gate (Cd vs. monolithic + weak-div machine-zero + BDF2 engaged) AND a hard 2nd-order temporal-convergence gate ("recover 2nd order in time"). Chorin (`p*=0`) will confirm the mechanism but is only 1st-order in time, so it CANNOT be the final answer; the target is incremental pressure (`p*=p^n`) + VMS-consistent fine-scale, possibly + rotational. Task 5 is the monolithic-fallback decision if no stable config also achieves 2nd order.

**Architecture:** Diagnostic-first, then Taly-consistent fix. Task 1 (DONE, committed `4194601`) ruled out all four original mechanisms with machine-precision negatives (M1 PPE-conditioning residual 1e-14; M2 projection-space identity `‖σBᵀû−K_pφ‖`=1.16e-14 holds in 3-D; M3 φ-gauge drift 3.3e-13; M4 penalty diverges through α=20000) and pinned the divergence to an **incremental-pressure feedback**: `‖p̂‖` grows ~4× over 12 steps (162→1307) with Cd→−8 at Stokes, while the PPE residual stays 1e-14 and — the KEY SIGNATURE — the weak-divergence stays FLAT ~0.045 (never decays). The working Taly reference cures this WITHOUT a rotational term, via three coupled levers we mirror as knobs on `LerayProjectionStepper` (threaded through `LeraySBMStepper`), each defaulting to CURRENT behavior so the default path is bit-for-bit unchanged:
- **`pressure_update ∈ {standard, rotational, chorin}`** (default `standard`) — the pressure-treatment enum. `standard` = classic incremental `p̂=p*+φ` (current); `rotational` = Timmermans consistent-incremental `p̂=p*+φ−ν·q`, `M_p q = Bᵀû`; `chorin` = non-incremental confirmation mode (zeros `p*` each step, `p̂=φ`). This enum ALSO carries Taly's `pressure_extrap_c` semantics: `standard`/`rotational` are 1st-order pressure extrapolation (`p*=p^n`, BDF2-compatible); `chorin` is 0th-order (`p*=0`). We NEVER use 2nd-order pressure extrapolation (Taly comment: "keep pressure extrapolation zero or first order for now"). See the interaction table in Task 2.
- **`ppe_fine_scale ∈ {False, True}`** (default `False`) — the VMS fine-scale consistency lever. When `True`, the fine-scale velocity `−τ_M R_m` enters BOTH the PPE source (`σ(∇q, −τ_M R_m)`) AND the velocity update (`u = û − τ_M R_m − (1/σ)(∇p̂−∇p*)`), matching Taly `Proj_Linear_PPE_Integrands` and `Proj_Linear_VUE_Integrands`. This is the leading suspected root cause of the flat, non-decaying weak-divergence (our PPE source is pure `σ·Bᵀû`, missing the fine-scale). It computes τ_M with the CORRECT `dt` (NOT the latent `dt=Δt/b0` bug of the old `ppe_finescale=True` path — that path stays untouched).

The bake-off (Task 3) sweeps these levers one-at-a-time from the current baseline, plus the 2nd-order-preserving combos, records the `‖p̂‖` trajectory (growing vs bounded), the weak-divergence trajectory (flat vs DECAYING), and Cd, and emits a verdict naming the MINIMAL lever set that makes `‖p̂‖` bounded AND weak-div decay. Task 4 wires the winner and gates it on BOTH stability (Cd within 20% of monolithic, weak-div machine-zero, BDF2 engaged, mutation leg = baseline diverges) AND a 2nd-order temporal-convergence MMS study (velocity temporal order ≥ 1.9). Task 5 documents the monolithic fallback if the only stable config is 1st-order (Chorin).

**Tech Stack:** Python / NumPy / SciPy (host-side; all 3-D marches on gpubox CPU via `splu`); pytest; JSON baselines. All knobs live in the Python `step()` path of `leray.py` (no Warp kernel change): the rotational `−ν(∇·û)` reuses the assembled consistent mass `self.M` as `M_p` and `rhs_free/σ` as `Bᵀû`; the `ppe_fine_scale` term reuses the coarse momentum residual `r_m` and `τ_m` already computed in the predictor/PPE loop (leray.py lines ~404–418).

## Global Constraints

- R2a DUAL GATE (Task 4): (a) **STABILITY** — 3-D sphere projection Cd matches monolithic same-mesh reference within 20% (faithfulness in 3-D) + weak-div machine-zero + BDF2 engaged, with a mutation leg (baseline `standard`, no fine-scale) that diverges; AND (b) **2nd-ORDER TEMPORAL** — an MMS temporal-convergence study run WITH the winning config asserts velocity temporal order ≥ 1.9 (pressure order reported). BOTH are HARD gates; the winner must pass both. "Recover 2nd order in time" (Baskar) is non-negotiable.
- Winner definition (Task 3 → Task 4): the winner is the MINIMAL lever set that makes `‖p̂‖` bounded AND weak-div DECAY. If the ONLY stable config is Chorin (`pressure_update="chorin"`, 1st-order in time), that is a **NEEDS_CONTEXT escalation** to the supervisor (Task 5) — the VMS fine-scale consistency is expected to make the incremental/2nd-order path stable.
- Gate-hygiene: independent reference (3-D monolithic on matched mesh) + mutation/planted-break leg; no vacuous gates.
- REPO-IDENTITY GUARD: before every commit, verify `git rev-parse --abbrev-ref HEAD` == `p2-r2a`; abort if not.
- Agents never push; supervisor pushes.
- All 3-D sphere marches (bake-off, stability gate) run on **gpubox** (host/splu, 40-core CPU). The knob unit tests (2-D `dm` fixtures) and the MMS temporal gate run locally.
- **τ_m `dt` correctness:** the NEW `ppe_fine_scale=True` knob (Task 2) MUST compute τ_m with the CORRECT `dt` (not `dt/b0`). The pre-existing `ppe_finescale=True` path (leray.py line ~405) carries the latent `dt=Δt/b0` over-scaling bug (documented `docs/dev/2026-07-21-p2-r0-parity-audit.md` §5); R2a leaves that OLD path untouched (`ppe_finescale=False` throughout for the old flag) and the new lever must not inherit the bug — a guard/note is required in Task 2.
- R2b (device port) and R2c (literature validation) are separate follow-on plans, out of scope.
- MMS parity: ±0.10 order is a standing contract; do not regress `tests/test_p2r0_parity.py`. The DEFAULT knob path (`pressure_update="standard"`, `ppe_fine_scale=False`) must reproduce current `p_hat`/`u_new` bit-for-bit.
- Absolute paths only. Working directory: `/Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim`.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `tests/p2r2a_diagnostic_3d.py` | [DONE, committed `4194601`] 3-D pressure-coupling mechanism measurement (Task 1) |
| Create | `tests/baselines/p2r2a_diagnostic_3d.json` | [DONE] Diagnostic verdict record written by the diagnostic script |
| Modify | `src/diffsim/steppers/leray.py` | Add `pressure_update="standard\|rotational\|chorin"` enum + `ppe_fine_scale=False\|True` knob. Rotational `p̂=p*+φ−ν·q` (reuses `self.M` as `M_p`, `rhs_free/σ` as `Bᵀû`); Chorin `p*=0` reset; fine-scale `σ(∇q,−τ_M R_m)` in PPE source + `−τ_M R_m` in velocity update, τ_m with correct `dt` (Task 2) |
| Modify | `src/diffsim/steppers/leray_sbm.py` | Thread `pressure_update` and `ppe_fine_scale` through `__init__` to the base stepper (Task 2) |
| Create | `tests/test_p2r2a_pressure_knobs.py` | Unit tests: default bit-for-bit parity; rotational-zero-for-solenoidal-û; fine-scale adds exactly `σ(∇q,−τ_M R_m)`; knob validation. 2-D `dm` fixtures, run locally (Task 2) |
| Create | `tests/p2r2a_bakeoff_3d.py` | Bake-off diagnostic: sweep levers one-at-a-time + combos on the level-4 Stokes sphere; record `‖p̂‖`/weak-div/Cd trajectories; emit minimal-stable-lever verdict (gpubox) (Task 3) |
| Create | `tests/baselines/p2r2a_bakeoff_3d.json` | Bake-off config matrix + per-config trajectories + verdict (Task 3) |
| Modify | `tests/p2r0_task10_sphere_derisk.py` | Add `march_projection_r2a` helper parameterized by `pressure_update` + `ppe_fine_scale` (Task 4) |
| Modify | `tests/test_p2r0_projection_sbm.py` | Add `test_g6_sphere_3d_r2a_stability_gate` (Task 4) |
| Create | `tests/test_p2r2a_temporal_order.py` | 2nd-order temporal-convergence gate (MMS vortex, mirrors `tests/test_leray.py::test_leray_temporal_order`), run WITH the winning config; asserts velocity order ≥ 1.9 (Task 4) |
| Modify | `tests/baselines/p2r0_task10_sphere.json` | Update with the R2a matched-mesh result (Task 4) |
| Create | `docs/dev/2026-07-21-p2-r2a-fallback-decision.md` | Fallback-decision record (Task 5, if escalation triggered) |

---

## Task 1: 3-D Pressure-Coupling Diagnostic  ✅ DONE (committed `4194601`)

> **VERDICT (2026-07-21):** All four candidate mechanisms ruled out with machine-precision negatives (M1 PPE-conditioning residual 1e-14; M2 projection-space identity `‖σBᵀû−K_pφ‖`=1.16e-14; M3 φ-gauge drift 3.3e-13; M4 penalty diverges through α=20000). The surviving signature — `‖p̂‖` growing ~4× over 12 steps with Cd→−8 at Stokes while PPE residual stays 1e-14 — is an **incremental-pressure feedback instability** (`p* += φ` accumulation). This OVERTURNS the original penalty-law hypothesis. Tasks 2–3 below are re-scoped to the rotational / consistent-incremental fix. **Do not re-run or modify Task 1.**

**Purpose:** Produce a committed, decisive measurement naming WHICH pressure-coupling mechanism drives the 3-D projection divergence. The output must be as decisive as the 2-D diagnostic (`tests/p2r0_divergence_diagnostic.py`), which ruled out all three 2-D candidates and concluded the correction's surrogate treatment was a no-op. The 3-D diagnostic must similarly test each candidate and emit a clear mechanism verdict to `tests/baselines/p2r2a_diagnostic_3d.json`.

The four candidate mechanisms (from spec §3.R2a) are:

1. **M1 — PPE conditioning at the immersed boundary in 3-D.** The pressure-Poisson operator `K_p` is the scalar stiffness on the octree mesh. The immersed sphere introduces cut elements; in 3-D the number of surrogate-face DOFs and the conditioning of `K_p` both increase. Measure: the condition number of `K_p` (computed via `scipy.sparse.linalg.eigsh` on the smallest/largest eigenvalues of the splu-factored system, or via `norm(K_p^{-1} r) / norm(r)` ratio) and the relative residual of the PPE solve. If the PPE is converging well but the field is still diverging, M1 is not the mechanism.

2. **M2 — Surrogate-consistent PPE BC 3-D consistency (homogeneous-Neumann natural BC).** In 2-D the diagonal (2-D diagnostic Q1b) showed the PPE projection-space identity `‖sigma B^T u_hat − K_p phi‖ < 1e-13` — the PPE is solving correctly. In 3-D: does the same identity hold? If `‖sigma B^T u_hat − K_p phi‖` is small, the PPE is consistent; if it is large (O(1)), the 3-D natural-BC assembly is broken. This is the direct 3-D extension of the Q1b test. The 3-D `B^T` operator is assembled independently (same GP loop as in 2-D, extended to 3-D).

3. **M3 — Pressure null-space / outflow-boundary handling in 3-D.** The PPE is pinned at free-node 0 (enclosed flow). In 3-D the domain has a free-outflow face (`x=1`). If the pressure null-space handling (the single pin) is insufficient in 3-D due to the outflow face's natural pressure condition interacting with the PPE, the PPE RHS can drift. Measure: the PPE residual norm after solve (`‖K_p phi − rhs_free‖`) and the outflow-face pressure trace. Also check: does pinning a different free node (e.g., the one nearest the outflow centroid) change the stability?

4. **M4 — Nitsche-penalty α too small in 3-D (the leading fix candidate).** The 2-D stable window found α=100–1000; the 3-D divergence occurs even at α=100. The principled scaling is `α ~ Pe·p²` (Péclet number × polynomial-order squared, from the Nitsche coercivity condition). In 3-D with `h=1/16` (level-4), `U=1`, `ν=2UR/Re`: for Re=1, Pe=|u|h/(2ν)=Re·h/(4R)=1·(1/16)/(4·0.12)=0.13, so `α_Pe = Pe·p² = 0.13` — far smaller than 100. For Re=100, Pe=13. The FIXED α=100 vastly exceeds the principled value at Stokes, but the 3-D geometry introduces larger constraint errors than 2-D (more surrogate faces, larger h-surface ratio). Measure: at Stokes (Re=1), does the pressure increment `phi` grow monotonically step-over-step, or does it stay bounded? Plot `‖phi‖` per step. If `‖phi‖` grows while the PPE residual is small, the penalty is not enforcing no-penetration sufficiently in 3-D and the no-penetration leak feeds back into the pressure.

**The decisive verdict structure** (modelled on the 2-D diagnostic):
- M1 verdict: PPE conditioning. Report `cond_estimate`, `ppe_resid_rel`. If `ppe_resid_rel < 1e-8`, M1 is NOT the mechanism.
- M2 verdict: PPE projection-space identity. Report `‖sigma B^T u_hat − K_p phi‖`. If < 1e-9, M2 is NOT the mechanism.
- M3 verdict: null-space / outflow. Report `‖K_p phi − rhs_free‖` and pressure-pin sensitivity. If residual is small and changing the pin does not change stability, M3 is NOT the mechanism.
- M4 verdict: penalty insufficiency. Report `‖phi‖` trajectory over 10 steps, `mean_normal_flux` at surrogate (the no-penetration leak metric from `surrogate_normal_flux()`). If `‖phi‖` grows monotonically while PPE residual is small, M4 IS the mechanism.

The test file is a gpubox helper script (not in CI), following `tests/p2r0_task10_sphere_derisk.py`'s pattern of importing `diffsim` modules directly. The script writes its verdict to `tests/baselines/p2r2a_diagnostic_3d.json`.

**Files:**
- Create: `tests/p2r2a_diagnostic_3d.py`
- Create: `tests/baselines/p2r2a_diagnostic_3d.json` (written by the script; commit the result)

**Interfaces:**
- Consumes: `build_sphere_3d` from `tests/p2r0_task10_sphere_derisk.py`; `weak_divergence` pattern from `tests/p2r0_divergence_diagnostic.py`; `LeraySBMStepper` from `src/diffsim/steppers/leray_sbm.py`
- Produces: `tests/baselines/p2r2a_diagnostic_3d.json` with keys `M1`, `M2`, `M3`, `M4`, `verdict` (string naming the primary mechanism); `march_and_measure_3d(fx, alpha, dt, nsteps)` function callable by Task 3

- [ ] **Step 1: Write the diagnostic script `tests/p2r2a_diagnostic_3d.py`**

```python
"""P2-R2a 3-D pressure-coupling diagnostic.

Committed measurement isolating WHY the 3-D projection+SBM split
diverges at Stokes (Re=1). Tests four candidate mechanisms (M1-M4)
on the Task-10 level-4 sphere fixture. Writes verdict to
tests/baselines/p2r2a_diagnostic_3d.json. Runs on gpubox (splu).

Usage:
    cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim
    python tests/p2r2a_diagnostic_3d.py
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

sys.path.insert(0, os.path.dirname(__file__))

from p2r0_task10_sphere_derisk import build_sphere_3d, R, CTR, U_IN

from diffsim.steppers.leray_sbm import LeraySBMStepper
from diffsim.solvers.timestepping import bdf_coeffs, bdf_order_now

DT = 0.05
LEVEL = 4
RE_STOKES = 1.0        # Re=1: advection negligible, isolates pressure coupling
ALPHA = 100.0
NSTEPS_DIAG = 10      # short: enough to see growth, not long enough to blow up


# ---------------------------------------------------------------------------
# Shared: build a fresh stepper on the level-4 sphere fixture
# ---------------------------------------------------------------------------

def _make_stepper(fx, alpha=ALPHA, dt=DT, order=1, picard=2):
    dim = fx["dim"]

    def f_fn(x, t):
        return np.zeros((len(x), dim))

    st = LeraySBMStepper(
        fx["oracle"], fx["dm"], fx["nu"], dt, f_fn,
        u_inf=fx["u_inf"], strong_mask=fx["strong_mask"],
        lam=0.5, domain="outside", order=order, picard_iters=picard,
        solver="splu", ppe_finescale=False, alpha=alpha,
        beta_backflow=1.0, velocity_update="consistent")
    st.set_initial(lambda c: np.zeros((len(c), dim)))
    return st


# ---------------------------------------------------------------------------
# Utility: independent B^T u (weak divergence) — same loop as 2-D diagnostic
# ---------------------------------------------------------------------------

def _weak_divergence_3d(st):
    """Independent B^T u assembly in 3-D (matches the 2-D diagnostic pattern).
    Returns (||B^T u||_2, ||B^T u||_inf)."""
    dm = st.dm
    dim = dm.dim
    u_free = st.base._uvec(st.base.hist.pre1)
    u_full = np.asarray(dm.constraints.T @ u_free)
    rhs = np.zeros(dm.n_nodes)
    for pv, _b in dm.bins.items():
        tb = dm.tables_by_p[pv]
        h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
        jac = (h / 2.0) ** dim
        dsc = (2.0 / h)
        conn = dm.mesh.conn_of[pv]
        uq = np.einsum("qa,ead->eqd", tb.N, u_full[conn])
        be = np.einsum("qad,eqd,q,e->ea", tb.dN, uq, tb.w, jac * dsc)
        np.add.at(rhs, conn.ravel(), be.ravel())
    bt_free = np.asarray(dm.constraints.T.T @ rhs)
    bt_free[0] = 0.0
    return float(np.linalg.norm(bt_free)), float(np.abs(bt_free).max())


# ---------------------------------------------------------------------------
# M1: PPE conditioning check
# ---------------------------------------------------------------------------

def measure_m1_ppe_conditioning(st):
    """Measure PPE conditioning: relative residual of the PPE solve.
    Run one step and intercept phi; recompute K_p @ phi to get residual."""
    # Run one step — then re-examine the PPE residual by direct check
    u, p = st.step()
    phi = st.base.p_star.copy()    # after step: p_star = phi (first step: p*=0, so p_hat=phi)
    # Also compute the actual PPE RHS from the predictor uhat
    uhat = st.base._uvec(st.base.hist.pre1)
    dm = st.dm
    dim = dm.dim
    b0, _, _ = bdf_coeffs(
        bdf_order_now(st.base.t, st.base.dt, st.base.order,
                      have_history=False), st.base.dt)
    sigma = b0 / st.base.dt
    rhs = np.zeros(dm.n_nodes)
    for pv, _b in dm.bins.items():
        tb = dm.tables_by_p[pv]
        h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
        jac = (h / 2.0) ** dim
        dsc = (2.0 / h)
        conn = dm.mesh.conn_of[pv]
        ne, nqp = len(h), tb.nqp
        u_full = np.asarray(dm.constraints.T @ uhat)
        uq = np.einsum("qa,ead->eqd", tb.N, u_full[conn])
        fl = sigma * uq
        be = np.einsum("qad,eqd,q,e->ea", tb.dN, fl, tb.w, jac * dsc)
        np.add.at(rhs, conn.ravel(), be.ravel())
    rhs_free = np.asarray(dm.constraints.T.T @ rhs)
    rhs_free[0] = 0.0
    Kp = st.base.K_p.tolil()
    Kp.rows[0] = [0]
    Kp.data[0] = [1.0]
    Kp_csr = Kp.tocsr()
    actual_rhs = Kp_csr @ phi
    resid = rhs_free - actual_rhs
    resid[0] = 0.0
    ppe_resid_rel = float(np.linalg.norm(resid) / (np.linalg.norm(rhs_free) + 1e-300))
    # Rough condition estimate via power iteration on K_p^{-1}
    Kp_lu = splu(Kp.tocsr().tocsc())
    v = np.random.default_rng(42).standard_normal(len(rhs_free))
    v /= np.linalg.norm(v)
    for _ in range(10):
        v = Kp_lu.solve(v)
        v /= np.linalg.norm(v)
    lam_max_inv = float(np.dot(v, Kp_lu.solve(v)) / np.dot(v, v))
    v2 = np.random.default_rng(43).standard_normal(len(rhs_free))
    v2 /= np.linalg.norm(v2)
    for _ in range(10):
        v2 = Kp_csr @ v2
        v2 /= np.linalg.norm(v2)
    lam_max = float(np.dot(v2, Kp_csr @ v2) / np.dot(v2, v2))
    cond_estimate = lam_max * lam_max_inv
    return dict(ppe_resid_rel=ppe_resid_rel, cond_estimate=cond_estimate,
                verdict="NOT_M1" if ppe_resid_rel < 1e-6 else "POSSIBLE_M1")


# ---------------------------------------------------------------------------
# M2: PPE projection-space identity in 3-D  (B^T u_hat → K_p phi identity)
# ---------------------------------------------------------------------------

def measure_m2_projection_space_identity(st):
    """3-D extension of Q1b from the 2-D diagnostic.
    Verifies sigma B^T u_hat == K_p phi to machine precision after one step.
    If identity > 1e-9, the 3-D PPE assembly is broken (M2 is mechanism)."""
    u, p = st.step()
    # After step: p_star = p_hat = phi (p* was 0 before step 1)
    phi = st.base.p_star.copy()
    dm = st.dm
    dim = dm.dim
    b0, _, _ = bdf_coeffs(
        bdf_order_now(st.base.t, st.base.dt, st.base.order,
                      have_history=False), st.base.dt)
    sigma = b0 / st.base.dt
    uhat = st.base._uvec(st.base.hist.pre1)

    def _bt(u_free):
        u_full = np.asarray(dm.constraints.T @ u_free)
        rhs_bt = np.zeros(dm.n_nodes)
        for pv, _b in dm.bins.items():
            tb = dm.tables_by_p[pv]
            h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
            jac = (h / 2.0) ** dim
            dsc = (2.0 / h)
            conn = dm.mesh.conn_of[pv]
            uq = np.einsum("qa,ead->eqd", tb.N, u_full[conn])
            be = np.einsum("qad,eqd,q,e->ea", tb.dN, uq, tb.w, jac * dsc)
            np.add.at(rhs_bt, conn.ravel(), be.ravel())
        return np.asarray(dm.constraints.T.T @ rhs_bt)

    bt_uhat = _bt(uhat)
    lhs = sigma * bt_uhat.copy()
    lhs[0] = 0.0
    Kp = st.base.K_p.tolil()
    Kp.rows[0] = [0]
    Kp.data[0] = [1.0]
    Kp_csr = Kp.tocsr()
    rhs_check = Kp_csr @ phi
    resid = lhs - rhs_check
    resid[0] = 0.0
    identity_resid = float(np.linalg.norm(resid))
    return dict(identity_resid=identity_resid,
                bt_uhat_norm=float(np.linalg.norm(bt_uhat)),
                verdict="NOT_M2" if identity_resid < 1e-9 else "POSSIBLE_M2")


# ---------------------------------------------------------------------------
# M3: pressure null-space / outflow
# ---------------------------------------------------------------------------

def measure_m3_nullspace_outflow(st):
    """Test pressure null-space / outflow sensitivity.
    (a) PPE solve residual after one step.
    (b) Re-solve with the pin at a different free node (nearest outflow face)
        and compare phi norms."""
    u, p = st.step()
    dm = st.dm
    dim = dm.dim
    phi = st.base.p_star.copy()
    b0, _, _ = bdf_coeffs(
        bdf_order_now(st.base.t, st.base.dt, st.base.order,
                      have_history=False), st.base.dt)
    sigma = b0 / st.base.dt
    uhat = st.base._uvec(st.base.hist.pre1)
    u_full = np.asarray(dm.constraints.T @ uhat)
    rhs = np.zeros(dm.n_nodes)
    for pv, _b in dm.bins.items():
        tb = dm.tables_by_p[pv]
        h = dm.mesh.tree.h()[dm.mesh.bins[pv]]
        jac = (h / 2.0) ** dim
        dsc = (2.0 / h)
        conn = dm.mesh.conn_of[pv]
        uq = np.einsum("qa,ead->eqd", tb.N, u_full[conn])
        fl = sigma * uq
        be = np.einsum("qad,eqd,q,e->ea", tb.dN, fl, tb.w, jac * dsc)
        np.add.at(rhs, conn.ravel(), be.ravel())
    rhs_free = np.asarray(dm.constraints.T.T @ rhs)
    # Standard pin: free-node 0
    Kp = st.base.K_p.tolil()
    Kp.rows[0] = [0]
    Kp.data[0] = [1.0]
    rhs_free0 = rhs_free.copy()
    rhs_free0[0] = 0.0
    Kp_csr = Kp.tocsr()
    phi_std = splu(Kp_csr.tocsc()).solve(rhs_free0)
    resid_std = float(np.linalg.norm(rhs_free0 - Kp_csr @ phi_std))
    # Alternate pin: free node nearest outflow face x=1
    coords = dm.mesh.node_coords[dm.constraints.free_nodes]
    alt_pin = int(np.argmax(coords[:, 0]))
    Kp2 = st.base.K_p.tolil()
    Kp2.rows[alt_pin] = [alt_pin]
    Kp2.data[alt_pin] = [1.0]
    rhs_free2 = rhs_free.copy()
    rhs_free2[alt_pin] = 0.0
    phi_alt = splu(Kp2.tocsr().tocsc()).solve(rhs_free2)
    phi_diff = float(np.linalg.norm(phi_std - phi_alt))
    return dict(ppe_resid_std=resid_std, phi_norm_std=float(np.linalg.norm(phi_std)),
                phi_norm_alt=float(np.linalg.norm(phi_alt)),
                phi_pin_diff=phi_diff,
                verdict="NOT_M3" if (resid_std < 1e-8 and phi_diff < 0.1) else "POSSIBLE_M3")


# ---------------------------------------------------------------------------
# M4: penalty insufficiency / no-penetration leak drives pressure growth
# ---------------------------------------------------------------------------

def measure_m4_penalty_phi_growth(fx, alpha=ALPHA, dt=DT, nsteps=NSTEPS_DIAG):
    """Track ||phi|| and the surrogate normal flux per step.
    If ||phi|| grows monotonically while PPE residual stays small, M4 is the
    mechanism: the Nitsche penalty is not enforcing no-penetration in 3-D
    and the resulting mass leak through the body amplifies the pressure
    increment each step."""
    st = _make_stepper(fx, alpha=alpha, dt=dt, order=1, picard=2)
    phi_norms = []
    normal_fluxes = []
    q = 0.5 * U_IN ** 2 * np.pi * R ** 2
    cds = []
    for step in range(nsteps):
        u, p = st.step()
        phi_norms.append(float(np.linalg.norm(st.base.p_star)))
        mean_un, net, area = st.surrogate_normal_flux()
        normal_fluxes.append(float(abs(mean_un)))
        F = st.surrogate_traction()
        cds.append(float(F[0] / q))
        print(f"[M4] step{step+1}: ||phi||={phi_norms[-1]:.4f} "
              f"|mean_un|={normal_fluxes[-1]:.4e} Cd={cds[-1]:+.4f}",
              flush=True)
    # Monotone growth test: 7 of the last 8 steps should be increasing
    phi_growth = sum(phi_norms[i] > phi_norms[i-1]
                     for i in range(2, len(phi_norms)))
    monotone_growth = phi_growth >= (nsteps - 3)
    return dict(phi_norms=phi_norms, normal_fluxes=normal_fluxes, cds=cds,
                monotone_phi_growth=monotone_growth,
                phi_growth_count=phi_growth,
                verdict="M4_LIKELY" if monotone_growth else "NOT_M4")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[diag3d] Building level-4 sphere fixture at Re=1 (Stokes)...",
          flush=True)
    fx = build_sphere_3d("cpu", level=LEVEL, Re=RE_STOKES)
    dh = 2 * R / (1.0 / 2 ** LEVEL)
    print(f"[diag3d] n_free={len(fx['coords'])} sf={fx['sf'].elem.size} "
          f"D/h={dh:.2f}", flush=True)

    print("\n--- M1: PPE conditioning ---", flush=True)
    st1 = _make_stepper(fx)
    m1 = measure_m1_ppe_conditioning(st1)
    print(f"  cond_estimate={m1['cond_estimate']:.3e}  "
          f"ppe_resid_rel={m1['ppe_resid_rel']:.3e}  verdict={m1['verdict']}",
          flush=True)

    print("\n--- M2: 3-D PPE projection-space identity ---", flush=True)
    st2 = _make_stepper(fx)
    m2 = measure_m2_projection_space_identity(st2)
    print(f"  ||sigma B^T u_hat - K_p phi||={m2['identity_resid']:.3e}  "
          f"verdict={m2['verdict']}", flush=True)

    print("\n--- M3: null-space / outflow sensitivity ---", flush=True)
    st3 = _make_stepper(fx)
    m3 = measure_m3_nullspace_outflow(st3)
    print(f"  ppe_resid={m3['ppe_resid_std']:.3e}  "
          f"phi_pin_diff={m3['phi_pin_diff']:.4f}  verdict={m3['verdict']}",
          flush=True)

    print("\n--- M4: penalty/phi growth ---", flush=True)
    m4 = measure_m4_penalty_phi_growth(fx)
    print(f"  monotone_growth={m4['monotone_phi_growth']}  "
          f"growth_count={m4['phi_growth_count']}/{NSTEPS_DIAG}  "
          f"verdict={m4['verdict']}", flush=True)

    # Overall verdict: the mechanism(s) where verdict is NOT "NOT_*"
    mechanisms = []
    if m1["verdict"] != "NOT_M1":
        mechanisms.append("M1")
    if m2["verdict"] != "NOT_M2":
        mechanisms.append("M2")
    if m3["verdict"] != "NOT_M3":
        mechanisms.append("M3")
    if m4["verdict"] == "M4_LIKELY":
        mechanisms.append("M4")

    if not mechanisms:
        overall = "UNKNOWN — all M1-M4 ruled out; need new candidate"
    elif mechanisms == ["M4"]:
        overall = ("M4_PENALTY_INSUFFICIENCY — phi grows while PPE is accurate; "
                   "no-penetration leak through penalty; fix: alpha~Pe*p^2")
    else:
        overall = f"NEEDS_CONTEXT — multiple mechanisms: {mechanisms}"

    print(f"\n[diag3d] OVERALL VERDICT: {overall}", flush=True)

    results = {
        "config": dict(level=LEVEL, Re=RE_STOKES, alpha=ALPHA, dt=DT,
                       nsteps=NSTEPS_DIAG, R=R, U_IN=U_IN,
                       n_free=int(len(fx["coords"])),
                       sf_faces=int(fx["sf"].elem.size), D_over_h=dh),
        "M1": m1, "M2": m2, "M3": m3, "M4": m4,
        "verdict": overall,
    }
    out = os.path.join(os.path.dirname(__file__), "baselines",
                       "p2r2a_diagnostic_3d.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[diag3d] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the diagnostic on gpubox**

On gpubox, from `/Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim`:
```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim
python tests/p2r2a_diagnostic_3d.py 2>&1 | tee /tmp/diag3d.log
cat tests/baselines/p2r2a_diagnostic_3d.json
```

Expected: the script runs without crash, prints M1/M2/M3/M4 verdicts, and writes the JSON. Based on the Task-10 report (PPE projection-space identity passed in 3-D at `2.6e-15` in the pipeline smoke), M2 should print NOT_M2. If the script prints `M4_PENALTY_INSUFFICIENCY`, proceed to Task 2 (penalty-law fix). If a different mechanism fires, stop and escalate to the supervisor before Task 2.

- [ ] **Step 3: Review the verdict and decide the path forward**

Read `tests/baselines/p2r2a_diagnostic_3d.json`. Confirm one of:
  - `verdict == "M4_PENALTY_INSUFFICIENCY"` → proceed to Task 2.
  - Any other verdict → NEEDS_CONTEXT escalation: document in `docs/dev/2026-07-21-p2-r2a-fallback-decision.md` and skip to Task 5 (fallback). [Historical Task-1 note; superseded by the DONE verdict banner above — do not re-run Task 1.]

- [ ] **Step 4: Commit the diagnostic script and baseline**

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim
git rev-parse --abbrev-ref HEAD   # must print "master"
git add tests/p2r2a_diagnostic_3d.py tests/baselines/p2r2a_diagnostic_3d.json
git commit -m "$(cat <<'EOF'
feat(p2-r2a): committed 3-D pressure-coupling diagnostic (Task 1)

Four-mechanism measurement (M1 PPE conditioning, M2 projection-space
identity, M3 null-space/outflow, M4 penalty/phi-growth) on the Re=1
Stokes sphere at level-4. Verdict recorded in
tests/baselines/p2r2a_diagnostic_3d.json.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Implement the Projection Knobs

**Prerequisite:** Task 1 diagnostic (DONE, `4194601`) pinned the 3-D Stokes divergence to an incremental-pressure feedback: `‖p̂‖` grows while the PPE residual stays 1e-14 and the weak-divergence stays FLAT (never decays). This task implements the levers — matching the working Taly `ns_vms` reference — that Task 3 will bake off. Every knob DEFAULTS to current behavior, so the default path is bit-for-bit unchanged.

**What this task builds — the knob API (this is the ONE coherent API for Tasks 2–4):**

Two knobs on `LerayProjectionStepper.__init__` (mirroring the `velocity_update` knob pattern at leray.py lines ~39, 55–59), both threaded through `LeraySBMStepper.__init__`:

1. **`pressure_update="standard"` (str; one of `standard|rotational|chorin`)** — the pressure-treatment enum. This SINGLE enum subsumes Taly's `pressure_extrap_c` knob (the choice was to fold pressure-extrapolation order INTO this enum rather than add a separate `pressure_extrap_order` int — see the "Why one enum" note below):
   - `"standard"` (default): classic incremental `p̂ = p* + φ`, with `p*` accumulating (`self.p_star = p̂` at end of step). Pressure-extrapolation order **1** (`p*` carries the previous full pressure `p^n`). BDF2-compatible. **Bit-for-bit the current behavior.**
   - `"rotational"`: Timmermans consistent-incremental `p̂ = p* + φ − ν·q`, where `M_p q = Bᵀû` (nodal weak divergence of the predictor `û`). Also pressure-extrapolation order 1 (`p*` still accumulates the previous pressure). BDF2-compatible. The `−ν(∇·û)` term removes the spurious pressure boundary layer the classic form's implicit homogeneous-Neumann pressure BC creates. Reuses the `3942251` rotational spec below.
   - `"chorin"`: non-incremental confirmation mode — pressure-extrapolation order **0** (`p* ≡ 0`). `self.p_star` is zeroed at the TOP of `step()` so the predictor never sees an accumulating `∇p*`, and `p̂ = φ` with no step-over-step accumulation. **1st-order in time** (Chorin/non-incremental), so it can CONFIRM the mechanism but CANNOT be the final winner.

2. **`ppe_fine_scale=False` (bool)** — the VMS fine-scale-consistency lever (the most physics-laden knob; the leading suspected root cause of the flat non-decaying weak-divergence). When `True`, ONE fine-scale velocity object `u' = −τ_M R` is threaded CONSISTENTLY into BOTH sub-solves (this consistency is the whole point — the flat divergence comes from the PPE source and the velocity update disagreeing). Here `R` is the coarse momentum residual evaluated with the LAGGED pressure gradient (Taly NL integrands `Proj_NL_PPE`@585 / `Proj_NL_VUE`@526 / `Proj_NL_Momentum_BDF12`@637 all share this one `u'`):
   ```
   R(i) = (b0·u_curr + b1·u_pre1 + b2·u_pre2)(i)/dt   # BDF time derivative
        + (u_curr·∇)u_curr(i)                          # convection
        − (1/Re)·laplacian(u_curr)(i)                  # viscous
        + ∇(p^n)(i)                                    # LAGGED pressure grad (= ∇p_star in our incremental setting; NOT p^{n+1}, NOT extrapolated)
        − forcing(i)
   u'  = − τ_M · R
   ```
   - PPE source gains `σ(∇q, u') = σ(∇q, −τ_M R)`, in addition to the resolved `σ·Bᵀû`.
   - Velocity update gains `u'`: `u^{n+1} = û + u' − (1/σ)(∇p̂ − ∇p*)` (note `∇p̂ − ∇p* = ∇φ`, already present).
   - The predictor keeps its own RBVMS τ_m terms — **leave the predictor as-is for R2a** (do not re-thread `u'` into the predictor).
   When `False` (default), the PPE source is the current pure `σ·Bᵀû` and the velocity update is the current `û − (1/σ)∇φ`. **This is a NEW, separate flag from the pre-existing `ppe_finescale` flag** (Naming note below). Implementer: our leray.py PPE loop ALREADY assembles this exact `R` (lines ~404–418: `r_m = sigma*aqv + agu + pq_g − fq_base`, with `pq_g = ∇p_star` the lagged pressure) — REUSE that residual/τ_m object; do NOT recompute a second, subtly-different residual, and confirm the pressure term stays `∇p_star` (lagged), matching the incremental pressure the PPE re-solves against.

**Why ONE enum (not a separate `pressure_extrap_order` int):** Taly's `pressure_extrap_c` and the incremental-accumulation choice are NOT independent in our stepper: `chorin` IS `pressure_extrap_order=0` (`p*=0`, no accumulation), and `standard`/`rotational` ARE `pressure_extrap_order=1` (`p*=p^n`). A separate int knob would admit contradictory states (e.g. `pressure_update="chorin"` with `pressure_extrap_order=1`) and duplicate the accumulation-vs-reset decision. Folding it into the single `pressure_update` enum makes the pressure-treatment ONE coherent concept with no illegal combinations. The mapping (interaction table):

| `pressure_update` | pressure-extrap order (Taly `pressure_extrap_c`) | `p*` handling | `p̂` | time order |
|---|---|---|---|---|
| `standard`  | 1 (`p*=p^n`) | accumulates (`self.p_star=p̂`) | `p*+φ` | 2 (BDF2) |
| `rotational`| 1 (`p*=p^n`) | accumulates (`self.p_star=p̂`) | `p*+φ−ν·q` | 2 (BDF2) |
| `chorin`    | 0 (`p*=0`)   | zeroed each step (no accumulation) | `φ` | 1 |

We NEVER expose 2nd-order pressure extrapolation (Taly: "keep pressure extrapolation zero or first order for now").

**Naming note (`ppe_fine_scale` vs the old `ppe_finescale`):** `leray.py` already has a `ppe_finescale` flag (no underscore between fine/scale) whose `True` branch (lines ~411–419, 439–447) carries the latent `dt=Δt/b0` τ_m over-scaling bug. The NEW knob is spelled **`ppe_fine_scale`** (WITH the underscore) to (a) not disturb the old flag/path and (b) compute τ_m with the CORRECT `dt`. R2a keeps the OLD `ppe_finescale=False` throughout; the new lever is what Task 3 bakes off. During implementation, add a one-line comment at the new knob's site pointing to this distinction and the audit doc §5.

**The rotational formula (exact; from the `3942251` spec):**
```
p̂ = p_star + φ − ν · q,   where   M_p q = Bᵀû
```
INSIDE `LerayProjectionStepper.step()`:
- `φ` is the existing PPE increment (line ~454), unchanged.
- `Bᵀû` (nodal weak divergence of the predictor `û`) is obtained WITHOUT re-assembly ONLY when `ppe_fine_scale=False`: there the PPE RHS is built with `flux = sigma * aqv` (line ~421), so `rhs_free = σ·Bᵀû`, hence **`rhs_free / sigma` IS the free-node weak divergence `Bᵀû`** (pin already applied at free-node 0). When `ppe_fine_scale=True`, `rhs_free` carries the fine-scale term, so `rotational` must recompute `Bᵀû` from `û` directly (or is disallowed with fine-scale — see the guard in Step 4).
- `M_p` is the scalar consistent mass **already assembled as `self.M`** (leray.py `_mass_matrix`, lines ~115–134, `T.T @ M @ T`, the same free-node scalar space as `φ`/`p_star`). SPD, invertible without a pin — do NOT pin `M_p`. Solve via `solve_linear(self.M, ..., cache_key="mass")` — the SAME key the velocity update uses, so the factorization is shared.

**The fine-scale discrete form (exact; matches Taly + leray.py's existing residual):** see Task 2 Step 4.

**Files:**
- Modify: `src/diffsim/steppers/leray.py` (add both knobs to `__init__`; wire the pressure-update branch + Chorin reset + fine-scale PPE source + fine-scale velocity-update term in `step()`)
- Modify: `src/diffsim/steppers/leray_sbm.py` (thread `pressure_update` and `ppe_fine_scale` through `__init__` to the base stepper)
- Create: `tests/test_p2r2a_pressure_knobs.py` (unit tests: default bit-for-bit parity; rotational-zero-for-solenoidal; fine-scale exact-term; validation — all on 2-D `dm` fixtures, run locally)

**Interfaces:**
- Consumes: `LerayProjectionStepper(dm, nu, dt, f_fn, g_fn, ..., velocity_update="consistent", graddiv_scale=1.0, pressure_update="standard", ppe_fine_scale=False)`; the 2-D `dm` builder in `tests/test_p2r0_projection_sbm.py` (grep lines ~44–51 for `Sphere(CTR,R)` / `build_uniform` / `build_mesh` / `build_constraints` / `DeviceMesh.from_mesh`).
- Produces: `LeraySBMStepper(..., pressure_update=..., ppe_fine_scale=...)` for Tasks 3–4.

- [ ] **Step 1: Add both knobs to `LerayProjectionStepper.__init__`**

In `src/diffsim/steppers/leray.py`, add to the `__init__` signature (append after `graddiv_scale=1.0`):

```python
    def __init__(self, dm, nu, dt, f_fn, g_fn, order=2, picard_iters=2,
                 solver="splu",
                 timestab=True, ppe_finescale=False, predictor="picard",
                 velocity_update="consistent", graddiv_scale=1.0,
                 pressure_update="standard", ppe_fine_scale=False):
```

After the `velocity_update` validation + `self.graddiv_scale = ...` lines (~line 60), insert:

```python
        # P2-R2a pressure-treatment enum (default "standard" = unchanged classic
        # incremental p_hat = p* + phi). Folds Taly's pressure_extrap_c into one
        # enum (see the plan's interaction table): standard/rotational are
        # pressure-extrap order 1 (p*=p^n, BDF2-compatible); chorin is order 0
        # (p*=0, 1st-order). We never use 2nd-order pressure extrapolation.
        #   "standard"   — classic incremental: p_hat = p* + phi (Algorithm 1).
        #   "rotational" — Timmermans consistent-incremental:
        #                    p_hat = p* + phi - nu * q,  M_p q = B^T u_hat.
        #   "chorin"     — non-incremental confirmation: p* reset to 0 each step
        #                  (no accumulation); p_hat = phi. 1st-order in time.
        if pressure_update not in ("standard", "rotational", "chorin"):
            raise ValueError(
                f"pressure_update must be standard|rotational|chorin, "
                f"got {pressure_update!r}")
        self.pressure_update = pressure_update
        # P2-R2a VMS fine-scale-consistency lever (default False = unchanged).
        # NOTE: spelled with an underscore to distinguish from the PRE-EXISTING
        # self.ppe_finescale flag (leray.py, dt=Δt/b0 τ_m bug, audit §5) which
        # R2a leaves untouched. When True: the fine-scale velocity -tau_M r_m
        # enters BOTH the PPE source (sigma (grad q, -tau_M r_m)) AND the
        # velocity update (u = u_hat - tau_M r_m - (1/sigma)(grad p_hat -
        # grad p*)) — matching Taly Proj_Linear_PPE/VUE_Integrands. tau_M here
        # is computed with the CORRECT dt (NOT dt/b0).
        self.ppe_fine_scale = bool(ppe_fine_scale)
```

- [ ] **Step 2: Write the failing knob-validation + default-parity unit tests (2-D, local)**

Create `tests/test_p2r2a_pressure_knobs.py`. Reuse the 2-D `dm` builder from `tests/test_p2r0_projection_sbm.py` (grep it for the `Sphere(CTR,R)` → `build_uniform(level,dim=2)` → `classify_lambda` → `build_mesh` → `build_constraints` → `DeviceMesh.from_mesh` sequence; wrap it into a local `_small_2d_stepper(pressure_update="standard", ppe_fine_scale=False)` helper that returns a plain `LerayProjectionStepper` with a zero forcing and a zero box-Dirichlet `g_fn`). Then:

```python
"""P2-R2a projection-knob unit tests (2-D dm fixtures, run locally).

Pins the knob CONTRACTS without a 3-D march:
  (a) DEFAULT path is bit-for-bit unchanged.
  (b) rotational term vanishes when u_hat is exactly divergence-free.
  (c) ppe_fine_scale adds EXACTLY sigma (grad q, -tau_M r_m) to the PPE source.
  (d) both knobs validate.
"""
import numpy as np
import pytest

# --- _small_2d_stepper(...) built from the tests/test_p2r0_projection_sbm.py
#     2-D dm sequence; see grep note above. Returns a fresh
#     LerayProjectionStepper on a small level-3 2-D box + sphere fixture. ---


def test_pressure_update_validates():
    """Unknown pressure_update raises (mirrors velocity_update)."""
    with pytest.raises(ValueError, match="pressure_update must be"):
        _small_2d_stepper(pressure_update="bogus")


def test_default_pressure_treatment_is_standard():
    """The no-arg default equals pressure_update='standard', ppe_fine_scale=False."""
    st = _small_2d_stepper()
    assert st.pressure_update == "standard"
    assert st.ppe_fine_scale is False


def test_default_step_bitforbit(device):
    """DEFAULT path bit-for-bit: a default stepper and an explicit
    (standard, ppe_fine_scale=False) stepper from the same IC give identical
    p_hat AND u_new after one step (np.array_equal, not allclose)."""
    st_a = _small_2d_stepper()                                  # default
    st_b = _small_2d_stepper(pressure_update="standard",
                             ppe_fine_scale=False)              # explicit
    # identical IC
    ic = lambda c: 0.1 * np.stack([c[:, 1], -c[:, 0]], axis=1)
    st_a.set_initial(ic); st_b.set_initial(ic)
    ua, pa = st_a.step()
    ub, pb = st_b.step()
    assert np.array_equal(pa, pb), "p_hat drifted from the default path"
    assert np.array_equal(ua, ub), "u_new drifted from the default path"
```

Run locally:
```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim
python -m pytest tests/test_p2r2a_pressure_knobs.py::test_pressure_update_validates \
                 tests/test_p2r2a_pressure_knobs.py::test_default_pressure_treatment_is_standard \
                 tests/test_p2r2a_pressure_knobs.py::test_default_step_bitforbit -v
```
Expected after Steps 1+3: validation + default-parity PASS. `test_default_step_bitforbit` is the load-bearing "default unchanged" anchor (it must stay green through Steps 3–5).

- [ ] **Step 3: Wire the Chorin reset + the three-way pressure-update branch + fine-scale PPE source in `step()`**

(a) **Chorin reset** — at the very top of `step()`, BEFORE the predictor call (`uhat = self._predict(...)`, line ~390):
```python
        # P2-R2a: Chorin (non-incremental) confirmation mode — zero the
        # accumulated pressure each step so the predictor never sees a
        # compounding grad p* and p_hat = phi (pressure-extrap order 0).
        if self.pressure_update == "chorin":
            self.p_star = np.zeros(self.n_free)
```

(b) **Fine-scale PPE source** — the PPE flux is built in the `for pv, b_ in dm.bins.items()` loop (lines ~397–431). The coarse momentum residual `r_m` and `taum` are ALREADY computed there for the OLD `ppe_finescale` branch (lines ~404–418: `taum = tau_hbased_host(...)`, `agu = einsum(aqv, guq...)`, `r_m = sigma*aqv + agu + pq_g - fq_base`). For the NEW lever, in the `else` (classic) branch replace the pure `flux = sigma * aqv` with a fine-scale-augmented flux computed with the CORRECT `dt`:
```python
            if self.ppe_fine_scale:
                # NEW VMS-consistent PPE source (Taly Proj_Linear_PPE_Integrands
                # line ~294): flux = sigma*(u_hat - tau_M R). tau_M uses the
                # CORRECT dt (self.dt), NOT self.dt/b0 (the old ppe_finescale
                # bug). R is the coarse momentum residual already assembled here
                #   R = sigma*u_hat + a.grad u_hat + grad(p*) - f
                # where the pressure term is grad(p_star) — the LAGGED pressure
                # p^n (Taly NL integrands use vpre1.gradp, NOT p^{n+1}, NOT the
                # extrapolated p*). In our INCREMENTAL setting p_star IS the
                # lagged pressure the PPE re-solves against, so pq_g (=grad p*)
                # is exactly Taly's lagged-pressure gradient — REUSE it, do NOT
                # recompute a second, subtly-different residual.
                taum_fs = tau_hbased_host(umag, he, self.nu,
                                          dt=(self.dt if self.timestab else None),
                                          dim=dim)
                r_m = (sigma * aqv + agu + pq_g[pv].reshape(-1, dim)
                       - fq_base[pv])
                flux = sigma * (aqv - taum_fs[:, None] * r_m)
            elif self.ppe_finescale:      # PRE-EXISTING flag, untouched
                r_m = (sigma * aqv + agu + pq_g[pv].reshape(-1, dim)
                       - fq_base[pv])
                flux = aqv - taum[:, None] * r_m
                w_gp[pv] = 1.0 / sigma + taum
            else:
                flux = sigma * aqv
```
(So the PPE source becomes `int grad(N) . sigma(u_hat - tau_M r_m) = sigma*B^T u_hat + sigma(grad q, -tau_M r_m)` — exactly the Taly term. `agu` and `pq_g` must be in scope inside the loop; they already are for the old branch. Guard: the new `ppe_fine_scale` uses the UNWEIGHTED `self.K_p` stiffness — the fine scale is on the RHS only — so it does NOT set `w_gp[pv]` and does NOT trigger the `_weighted_stiffness` path. Confirm the `if self.ppe_finescale:` stiffness-assembly branch at line ~439 keys off the OLD flag only.)

(c) **The pressure-update branch** — replace `p_hat = self.p_star + phi` (line ~461) with:
```python
        # ---- pressure update (P2-R2a pressure_update enum) ----
        if self.pressure_update == "chorin":
            p_hat = phi.copy()             # p* zeroed at top of step(); no accum.
        elif self.pressure_update == "rotational":
            # Timmermans: p_hat = p* + phi - nu * q, M_p q = B^T u_hat.
            if self.ppe_fine_scale:
                # rhs_free carries the fine-scale term, so recompute B^T u_hat
                # directly from u_hat (weak divergence, pinned at node 0).
                bt_uhat = self._weak_div_free(uhat)
            else:
                bt_uhat = rhs_free / sigma     # rhs_free = sigma * B^T u_hat
            from ..solvers.linsolve import solve_linear
            q = solve_linear(self.M, bt_uhat, solver=self.solver, sym=True,
                             device=self.dm.device, cache=self._solver_cache,
                             cache_key="mass")   # SAME key as velocity update
            p_hat = self.p_star + phi - self.nu * q
        else:                                    # "standard" (default)
            p_hat = self.p_star + phi
```
where `_weak_div_free(uhat)` is a small helper assembling `B^T u_hat` in the free-node scalar space (the same GP loop as the diagnostic's `_weak_divergence_3d`, WITH `bt_free[0]=0` pin). It is only reached on the `rotational + ppe_fine_scale` combo (Task 3's richest config).

- [ ] **Step 4: Wire the fine-scale velocity-update term in `step()`**

The velocity update (lines ~462–487) currently integrates `uq - dphi_g/sigma` per component against the consistent mass. Taly's VUE (`Proj_Linear_VUE_Integrands` line ~230) is `u = u_hat - tau_M r_m - (1/sigma)(grad p_hat - grad p*)`. Our current update already has the `-(1/sigma)(grad phi)` piece (note `phi = p_hat - p*`, so `grad phi = grad p_hat - grad p*` — this term MATCHES Taly). The MISSING piece under `ppe_fine_scale=True` is the `-tau_M r_m` fine-scale velocity. Add it to the per-component integrand:
```python
                integ = (uq[pv].reshape(ne, nqp, dim)[:, :, c]
                         - dphi_g[pv].reshape(ne, nqp, dim)[:, :, c] / sigma)
                if self.ppe_fine_scale:
                    # -tau_M r_m fine-scale velocity (Taly VUE line ~230);
                    # r_m and tau_M recomputed here at the GP (same coarse
                    # residual as the PPE source, correct dt).
                    integ = integ - fs_vel[pv].reshape(ne, nqp, dim)[:, :, c]
```
where `fs_vel[pv] = taum_fs[:, None] * r_m` is cached per bin from the SAME `taum_fs`/`r_m` computed in the PPE loop (Step 3b) — compute it once in the PPE loop and stash into a `fs_vel` dict so the velocity-update loop reuses it (avoids recomputing `tau_hbased_host` and the residual). This keeps the fine-scale velocity IDENTICAL between the PPE source and the velocity update — the consistency the flat-divergence signature is missing.

- [ ] **Step 5: Write the failing rotational-zero + fine-scale-exact unit tests (2-D, local)**

Append to `tests/test_p2r2a_pressure_knobs.py`:

```python
def test_rotational_zero_for_solenoidal_predictor(device):
    """When u_hat is exactly divergence-free, B^T u_hat = 0 => q = 0, so
    rotational p_hat == standard p_hat (the -nu*q term vanishes). Build a
    solenoidal field (e.g. curl of a stream function on the box), set it as the
    predictor state, and assert ||q|| < 1e-10 and rotational-vs-standard p_hat
    agree to 1e-12. (ppe_fine_scale=False so bt_uhat = rhs_free/sigma.)"""
    # ... construct solenoidal u_hat; step both; assert.
    ...


def test_ppe_fine_scale_adds_exact_term(device):
    """ppe_fine_scale=True adds EXACTLY sigma (grad q, -tau_M r_m) to the PPE
    source vs ppe_fine_scale=False: intercept rhs_free in both, assert the
    difference equals the independently-assembled sigma*int grad(N) .
    (-tau_M r_m) to 1e-12, with tau_M computed at the CORRECT dt (self.dt)."""
    # ... assemble the reference fine-scale source independently; compare.
    ...
```
(Both are load-bearing math checks of the isolated levers — the plan's "the knob's math is correct in isolation" requirement. They run locally on the 2-D fixture.)

Run locally:
```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim
python -m pytest tests/test_p2r2a_pressure_knobs.py -v
```
Expected: all knob unit tests PASS (validation, default bit-for-bit parity, rotational-zero, fine-scale-exact).

- [ ] **Step 6: Thread both knobs through `LeraySBMStepper`**

In `src/diffsim/steppers/leray_sbm.py`, add to `__init__` (after `graddiv_scale=1.0`, line ~86):
```python
                 graddiv_scale=1.0, pressure_update="standard",
                 ppe_fine_scale=False):
```
and pass them to the base stepper (line ~118–122):
```python
        base = LerayProjectionStepper(
            dm, nu, dt, f_fn, self._g_box, order=order,
            picard_iters=picard_iters, solver=solver,
            ppe_finescale=ppe_finescale,
            velocity_update=velocity_update, graddiv_scale=graddiv_scale,
            pressure_update=pressure_update, ppe_fine_scale=ppe_fine_scale)
```

- [ ] **Step 7: Run MMS parity to confirm no regression**
```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim
python -m pytest tests/test_p2r0_parity.py tests/test_leray.py::test_leray_temporal_order -v
```
Expected: all parity tests PASS and the existing Leray temporal-order test PASS (default `standard`/`ppe_fine_scale=False` leaves both paths untouched).

- [ ] **Step 8: Commit the knobs**
```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim
git rev-parse --abbrev-ref HEAD   # must print "p2-r2a"
git add src/diffsim/steppers/leray.py src/diffsim/steppers/leray_sbm.py \
        tests/test_p2r2a_pressure_knobs.py
git commit -m "$(cat <<'EOF'
feat(p2-r2a): projection knobs — pressure_update enum + ppe_fine_scale (Task 2)

Adds pressure_update="standard|rotational|chorin" (default standard =
unchanged; folds Taly pressure_extrap_c into one enum) and ppe_fine_scale
(default False = unchanged; new flag distinct from the old ppe_finescale
dt/b0-bug path). Rotational: p_hat = p* + phi - nu*q, M_p q = B^T u_hat.
Fine-scale: sigma(grad q, -tau_M r_m) in the PPE source AND -tau_M r_m in
the velocity update, tau_M at the correct dt. Threaded through
LeraySBMStepper. Unit tests: default bit-for-bit parity, rotational-zero
for solenoidal predictor, fine-scale exact-term, validation. MMS parity +
Leray temporal-order clean.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Bake-Off Diagnostic (isolate the minimal stable lever set)

**Purpose:** A committed, decisive bake-off (styled on the Task-1 diagnostic) that sweeps the levers ONE AT A TIME from the current baseline plus the key combos on the level-4 3-D Stokes sphere, and emits a verdict naming the MINIMAL lever set that makes `‖p̂‖` bounded AND weak-div DECAY. Runs on gpubox.

**Fixture:** `build_sphere_3d` (from `tests/p2r0_task10_sphere_derisk.py`), Re=1 (Stokes, isolates pressure coupling), dt=0.05, level 4, ~12–15 steps.

**The config matrix** (each is a `(pressure_update, ppe_fine_scale)` pair; `alpha=100`, `velocity_update="consistent"`, `order=2`):

| # | label | `pressure_update` | `ppe_fine_scale` | purpose | expected `‖p̂‖` | expected weak-div | 2nd-order? |
|---|---|---|---|---|---|---|---|
| 0 | baseline | `standard` | `False` | reproduce Task-1 divergence | GROWING (~4×) | FLAT ~0.045 | (n/a; diverges) |
| 1 | chorin | `chorin` | `False` | mechanism confirmation | BOUNDED | (bounded) | NO (1st-order) |
| 2 | finescale-only | `standard` | `True` | fine-scale w/ incremental pressure | ? (target: bounded) | target: DECAYING | YES (BDF2) |
| 3 | rotational-only | `rotational` | `False` | rotational w/ incremental pressure | ? | ? | YES (BDF2) |
| 4 | incremental+finescale | `standard` | `True` | **2nd-order-preserving candidate** | target: bounded | target: DECAYING | YES (BDF2) |
| 5 | incremental+finescale+rotational | `rotational` | `True` | belt-and-suspenders | target: bounded | target: DECAYING | YES (BDF2) |

(Configs 2 and 4 are the SAME `(standard, True)` pair — keep BOTH rows in the emitted matrix for narrative clarity but the march is run ONCE and reused; the implementer may collapse to five distinct marches: baseline, chorin, `(standard,True)`, `(rotational,False)`, `(rotational,True)`.)

**Per-config records:** `phat_norms` (list, `‖p̂‖` per step — growing vs bounded), `weakdiv` (list, `‖Bᵀu‖₂` per step — FLAT vs DECAYING, the key signature), `cds` (list, Cd per step — physical vs divergent), `bounded` (bool: final `‖p̂‖` < 2× initial and finite), `weakdiv_decays` (bool: `weakdiv[-1] < 0.5 * weakdiv[2]`), `stable` (bool: `bounded and weakdiv_decays and Cd finite`).

**Verdict logic (emitted to JSON `verdict` + `winner`):**
- The `winner` is the MINIMAL stable config: among configs with `stable == True`, prefer (in order) fewest levers off-default, then `(standard,True)` over `(rotational,*)` (incremental+finescale is the expected 2nd-order-preserving winner).
- If the ONLY stable config is `chorin` (1st-order), emit `verdict = "NEEDS_CONTEXT — only Chorin (1st-order) stable; fine-scale consistency did not stabilize the incremental/2nd-order path"` — this is a Task-5 escalation.
- If NO config is stable, emit `verdict = "NEEDS_CONTEXT — no stable config"` (Task-5 escalation).
- Otherwise `verdict = "WINNER: <label> (pressure_update=<..>, ppe_fine_scale=<..>) — bounded ||p_hat|| and decaying weak-div"`.
- The plan STATES: the winner is the minimal STABLE config, but Task 4 ADDITIONALLY requires it to pass a 2nd-order temporal gate. If the winner is Chorin, that fails the 2nd-order requirement → NEEDS_CONTEXT to the supervisor (the fine-scale consistency is EXPECTED to make the incremental/2nd-order path stable, so a Chorin-only outcome is a genuine surprise worth escalating).

**Files:**
- Create: `tests/p2r2a_bakeoff_3d.py` (gpubox helper, imports `diffsim` directly, model on `tests/p2r2a_diagnostic_3d.py`)
- Create: `tests/baselines/p2r2a_bakeoff_3d.json` (config matrix + trajectories + verdict)

**Interfaces:**
- Consumes: `build_sphere_3d, R, CTR, U_IN` from `tests/p2r0_task10_sphere_derisk.py`; `_weak_divergence_3d` from `tests/p2r2a_diagnostic_3d.py`; `LeraySBMStepper` with the Task-2 knobs.
- Produces: `tests/baselines/p2r2a_bakeoff_3d.json` with keys `config_matrix` (list of per-config dicts), `winner` (dict: label + `pressure_update` + `ppe_fine_scale`), `verdict` (string).

- [ ] **Step 1: Write `tests/p2r2a_bakeoff_3d.py`**

Model on the Task-1 diagnostic. Sketch:
```python
"""P2-R2a bake-off diagnostic — isolate the minimal stable lever set.

Sweeps (pressure_update, ppe_fine_scale) one lever at a time from the current
baseline plus the 2nd-order-preserving combos on the level-4 Re=1 Stokes
sphere. For each config records the ||p_hat||, weak-div, and Cd trajectories,
and emits a verdict naming the minimal config with bounded ||p_hat|| AND
decaying weak-div. Runs on gpubox. Writes tests/baselines/p2r2a_bakeoff_3d.json.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from p2r0_task10_sphere_derisk import build_sphere_3d, R, U_IN
from p2r2a_diagnostic_3d import _weak_divergence_3d
from diffsim.steppers.leray_sbm import LeraySBMStepper

DT, LEVEL, RE, ALPHA, NSTEPS = 0.05, 4, 1.0, 100.0, 15

CONFIGS = [
    ("baseline",                        "standard",   False),
    ("chorin",                          "chorin",     False),
    ("incremental+finescale",           "standard",   True),
    ("rotational-only",                 "rotational", False),
    ("incremental+finescale+rotational","rotational", True),
]

def _march(fx, pressure_update, ppe_fine_scale):
    dim = fx["dim"]
    f_fn = lambda x, t: np.zeros((len(x), dim))
    st = LeraySBMStepper(
        fx["oracle"], fx["dm"], fx["nu"], DT, f_fn,
        u_inf=fx["u_inf"], strong_mask=fx["strong_mask"],
        lam=0.5, domain="outside", order=2, picard_iters=2,
        solver="splu", ppe_finescale=False, alpha=ALPHA,
        beta_backflow=1.0, velocity_update="consistent",
        pressure_update=pressure_update, ppe_fine_scale=ppe_fine_scale)
    st.set_initial(lambda c: np.zeros((len(c), dim)))
    q = 0.5 * U_IN**2 * np.pi * R**2
    pnorms, wdiv, cds = [], [], []
    for k in range(NSTEPS):
        u, p = st.step()
        pnorms.append(float(np.linalg.norm(st.base.p_star)))
        w2, _ = _weak_divergence_3d(st)
        wdiv.append(float(w2))
        F = st.surrogate_traction()
        cds.append(float(F[0] / q))
        print(f"[{pressure_update},fs={ppe_fine_scale}] step{k+1}: "
              f"||p||={pnorms[-1]:.3f} wdiv={wdiv[-1]:.4e} Cd={cds[-1]:+.3f}",
              flush=True)
    bounded = bool(np.isfinite(pnorms[-1]) and pnorms[-1] < 2.0 * (pnorms[0] + 1e-30))
    decays = bool(wdiv[-1] < 0.5 * wdiv[2])
    stable = bool(bounded and decays and np.isfinite(cds[-1]))
    return dict(phat_norms=pnorms, weakdiv=wdiv, cds=cds,
                bounded=bounded, weakdiv_decays=decays, stable=stable)

def main():
    fx = build_sphere_3d("cpu", level=LEVEL, Re=RE)
    matrix = []
    for label, pu, fs in CONFIGS:
        print(f"\n=== {label}  (pressure_update={pu}, ppe_fine_scale={fs}) ===", flush=True)
        res = _march(fx, pu, fs)
        res.update(label=label, pressure_update=pu, ppe_fine_scale=fs)
        matrix.append(res)
    # verdict: minimal stable config; Chorin-only => NEEDS_CONTEXT
    stable = [m for m in matrix if m["stable"]]
    non_chorin_stable = [m for m in stable if m["pressure_update"] != "chorin"]
    if non_chorin_stable:
        # prefer fewest levers off default, then standard over rotational
        def rank(m):
            n_off = (m["pressure_update"] != "standard") + (m["ppe_fine_scale"] is True)
            return (n_off, m["pressure_update"] == "rotational")
        win = sorted(non_chorin_stable, key=rank)[0]
        verdict = (f"WINNER: {win['label']} (pressure_update="
                   f"{win['pressure_update']}, ppe_fine_scale="
                   f"{win['ppe_fine_scale']}) — bounded ||p_hat|| and decaying weak-div")
        winner = dict(label=win["label"], pressure_update=win["pressure_update"],
                      ppe_fine_scale=win["ppe_fine_scale"])
    elif stable:   # only Chorin stable
        verdict = ("NEEDS_CONTEXT — only Chorin (1st-order) stable; fine-scale "
                   "consistency did not stabilize the incremental/2nd-order path")
        winner = dict(label="chorin", pressure_update="chorin", ppe_fine_scale=False)
    else:
        verdict = "NEEDS_CONTEXT — no stable config"
        winner = None
    print(f"\n[bakeoff] VERDICT: {verdict}", flush=True)
    out = dict(config=dict(level=LEVEL, Re=RE, dt=DT, alpha=ALPHA, nsteps=NSTEPS),
               config_matrix=matrix, winner=winner, verdict=verdict)
    path = os.path.join(os.path.dirname(__file__), "baselines", "p2r2a_bakeoff_3d.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[bakeoff] wrote {path}", flush=True)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the bake-off on gpubox**
```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim
python tests/p2r2a_bakeoff_3d.py 2>&1 | tee /tmp/bakeoff.log
cat tests/baselines/p2r2a_bakeoff_3d.json
```
Expected: baseline reproduces Task-1 divergence (`‖p̂‖` growing, weak-div FLAT ~0.045, Cd→negative); chorin BOUNDED. The expected winner is `incremental+finescale` = `(standard, True)` = **1st-order incremental pressure (`pressure_extrap_order=1`, `p*=p^n`, bounded/re-solved as full `p`) + `ppe_fine_scale=True`** — which the Taly reference achieves WITHOUT rotational and which is stable AND 2nd-order in VELOCITY (bounded `‖p̂‖`, decaying weak-div). Rotational (`(rotational, *)`) is NOT expected to be needed for stability or velocity 2nd-order; it is only the pressure-order booster if 2nd-order PRESSURE is later required (see the Task-4 temporal gate's reported pressure order). Verdict names the minimal stable config.

- [ ] **Step 3: Review the verdict and decide the Task-4 winner**

Read `tests/baselines/p2r2a_bakeoff_3d.json`.
  - `verdict` starts with `WINNER:` and the winner is NOT chorin → carry `winner.pressure_update` / `winner.ppe_fine_scale` into Task 4.
  - `verdict` is `NEEDS_CONTEXT — only Chorin ...` or `no stable config` → skip to Task 5 (fallback), do NOT attempt Task 4's gates with a 1st-order-only config.

- [ ] **Step 4: Commit the bake-off**
```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim
git rev-parse --abbrev-ref HEAD   # must print "p2-r2a"
git add tests/p2r2a_bakeoff_3d.py tests/baselines/p2r2a_bakeoff_3d.json
git commit -m "$(cat <<'EOF'
feat(p2-r2a): bake-off diagnostic isolating the minimal stable lever set (Task 3)

Sweeps (pressure_update, ppe_fine_scale) one lever at a time from the
baseline plus the 2nd-order-preserving combos on the level-4 Re=1 Stokes
sphere. Records ||p_hat|| (growing vs bounded), weak-div (flat vs
decaying — the key signature), and Cd trajectories. Emits the minimal
stable-lever verdict; Chorin-only is a NEEDS_CONTEXT escalation. Baseline
JSON committed.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Lock the Winner + Dual Gate (stability AND 2nd-order temporal)

**Purpose:** Wire the bake-off winner as the R2a setting and gate it on BOTH (b) a stability gate and (c) a hard 2nd-order temporal-convergence gate. This is where "recover 2nd order in time" is enforced.

**Precondition:** Task 3's `winner` is a NON-Chorin stable config (else Task 5). Below, `WIN_PU` / `WIN_FS` denote `winner.pressure_update` / `winner.ppe_fine_scale` from `tests/baselines/p2r2a_bakeoff_3d.json` (expected `standard` / `True`).

**Files:**
- Modify: `tests/p2r0_task10_sphere_derisk.py` (add `march_projection_r2a` parameterized by `pressure_update` + `ppe_fine_scale`)
- Modify: `tests/test_p2r0_projection_sbm.py` (add `test_g6_sphere_3d_r2a_stability_gate`)
- Create: `tests/test_p2r2a_temporal_order.py` (the 2nd-order temporal gate)
- Modify: `tests/baselines/p2r0_task10_sphere.json` (updated by the stability-gate run)

**Interfaces:**
- Consumes: `build_sphere_3d`, `monolithic_cd` from `tests/p2r0_task10_sphere_derisk.py`; `_weak_divergence_3d` from `tests/p2r2a_diagnostic_3d.py`; the vortex MMS harness pattern from `tests/test_leray.py::test_leray_temporal_order` (`u_ex`, `f_ex`, `NU`, `_make`, `_run` — the SAME MMS the monolithic and Leray steppers share, m1b findings).
- Produces: `march_projection_r2a(...)`; `test_g6_sphere_3d_r2a_stability_gate` (gpubox); `test_p2r2a_temporal_order` (local); updated `tests/baselines/p2r0_task10_sphere.json` with `r2a_stability` key.

- [ ] **Step 1: Add `march_projection_r2a` to `tests/p2r0_task10_sphere_derisk.py`**

Append after `monolithic_cd`:
```python
def march_projection_r2a(fx, dt, max_steps, rate_tol, order=2,
                         beta_backflow=1.0, picard_iters=2,
                         pressure_update="standard", ppe_fine_scale=True,
                         alpha=100.0):
    """March LeraySBMStepper with the R2a winning config (default
    standard + ppe_fine_scale=True, the expected bake-off winner). Returns
    dict(cd, clat, steps, finite, bdf2_engaged, st, u, p)."""
    from diffsim.steppers.leray_sbm import LeraySBMStepper
    dim = fx["dim"]
    def f_fn(x, t):
        return np.zeros((len(x), dim))
    st = LeraySBMStepper(
        fx["oracle"], fx["dm"], fx["nu"], dt, f_fn,
        u_inf=fx["u_inf"], strong_mask=fx["strong_mask"],
        lam=0.5, domain="outside", order=order, picard_iters=picard_iters,
        solver="splu", ppe_finescale=False, alpha=alpha,
        beta_backflow=beta_backflow, velocity_update="consistent",
        pressure_update=pressure_update, ppe_fine_scale=ppe_fine_scale)
    st.set_initial(lambda c: np.zeros((len(c), dim)))
    q = qref()
    cd_prev, steps, orders = None, 0, []
    for step in range(max_steps):
        u, p = st.step()
        F = st.surrogate_traction()
        cd = float(F[0] / q)
        orders.append(int(st.base.order))
        steps = step + 1
        if cd_prev is not None and step > 10 and abs(cd - cd_prev) / dt < rate_tol:
            break
        cd_prev = cd
    clat = float(np.hypot(F[1], F[2]) / q)
    finite = bool(np.isfinite(u).all() and np.isfinite(p).all())
    return dict(cd=cd, clat=clat, steps=steps, finite=finite,
                bdf2_engaged=(2 in orders), st=st, u=u, p=p)
```

- [ ] **Step 2: Write the stability gate `test_g6_sphere_3d_r2a_stability_gate` in `tests/test_p2r0_projection_sbm.py`**

Append at the end (mirror the Task-10/G-battery style). The gate: projection Cd within 20% of monolithic same-mesh, weak-div < 1e-8, BDF2 engaged, Cd > 0; MUTATION leg = baseline `standard`/`ppe_fine_scale=False` diverges (Cd < −0.5). Wire the winning config from the bake-off JSON (fall back to `standard`/`True` if unread).
```python
@pytest.mark.tier4
def test_g6_sphere_3d_r2a_stability_gate(device):
    """R2a STABILITY gate: 3-D sphere projection Cd matches monolithic
    same-mesh within 20%, weak-div machine-zero, BDF2 engaged, using the
    bake-off winner. Mutation leg: baseline (standard, ppe_fine_scale=False)
    diverges — proving the gate is load-bearing. gpubox (DIFFSIM_NIGHTLY)."""
    import json, os, sys
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from p2r0_task10_sphere_derisk import (build_sphere_3d, monolithic_cd,
                                           march_projection_r2a, R, U_IN)
    from p2r2a_diagnostic_3d import _weak_divergence_3d
    if not os.environ.get("DIFFSIM_NIGHTLY"):
        pytest.skip("nightly / gpubox only — R2a 3-D stability gate")
    # winning config from the bake-off (expected standard + ppe_fine_scale=True)
    bo = os.path.join(os.path.dirname(__file__), "baselines", "p2r2a_bakeoff_3d.json")
    WIN_PU, WIN_FS = "standard", True
    if os.path.exists(bo):
        w = json.load(open(bo)).get("winner") or {}
        WIN_PU = w.get("pressure_update", WIN_PU)
        WIN_FS = bool(w.get("ppe_fine_scale", WIN_FS))
    assert WIN_PU != "chorin", "winner is Chorin (1st-order) — escalate to Task 5"

    level, Re, dt, max_steps, rate_tol = 4, 100.0, 0.05, 120, 5e-3
    fx = build_sphere_3d(device, level=level, Re=Re)
    pr = march_projection_r2a(fx, dt=dt, max_steps=max_steps, rate_tol=rate_tol,
                              order=2, pressure_update=WIN_PU,
                              ppe_fine_scale=WIN_FS, alpha=100.0)
    q = 0.5 * U_IN**2 * np.pi * R**2
    w2, winf = _weak_divergence_3d(pr["st"])
    mono = monolithic_cd(fx, alpha=100.0, dt=dt, max_steps=max_steps, rate_tol=rate_tol)
    rel = abs(pr["cd"] - mono["cd"]) / (abs(mono["cd"]) + 1e-300)
    assert pr["finite"], "projection field is not finite"
    assert pr["bdf2_engaged"], "BDF2 never engaged"
    assert w2 < 1e-8, f"weak-div not machine-zero: {w2:.3e}"
    assert pr["cd"] > 0, f"projection Cd negative: {pr['cd']:.4f}"
    assert rel < 0.20, f"Cd {pr['cd']:.4f} vs mono {mono['cd']:.4f}: rel {rel:.1%} > 20%"
    # --- MUTATION: baseline standard + no fine-scale must diverge ---
    fx2 = build_sphere_3d(device, level=level, Re=Re)
    mut = march_projection_r2a(fx2, dt=dt, max_steps=15, rate_tol=0.0,
                               order=2, pressure_update="standard",
                               ppe_fine_scale=False, alpha=100.0)
    assert (mut["cd"] < -0.5) or (not np.isfinite(mut["cd"])), (
        f"mutation (baseline) expected to diverge, got Cd={mut['cd']:.3f} — gate vacuous")
    # --- update baseline JSON r2a_stability key (as before) ---
    ...
```
(The mutation march uses `max_steps=15, rate_tol=0.0` so it does NOT early-exit and shows the divergence.)

- [ ] **Step 3: Write the 2nd-order temporal gate `tests/test_p2r2a_temporal_order.py` (local)**

Mirror `tests/test_leray.py::test_leray_temporal_order` (the shared vortex MMS: `u_ex`, `f_ex`, `NU` from `test_ns_stepper`; `_make`/`_run` build a `LerayProjectionStepper` on the box). Run WITH the winning config, assert **velocity** temporal order ≥ 1.9 (the HARD "recover 2nd order in time" gate) AND **report pressure** temporal order. Rationale (coordinator, Taly NL): the reference is 1st-order incremental pressure + fine-scale, which gives stable AND 2nd-order VELOCITY, but pressure may be lower order; rotational is the documented pressure-order booster if the pressure order is poor. So velocity ≥ 1.9 is the gate; pressure order is reported, and if pressure order < ~1.5 the test PRINTS a note flagging `pressure_update="rotational"` as the pressure-order booster to try (not a hard failure).
```python
"""P2-R2a 2nd-order temporal-convergence gate.

Mirrors tests/test_leray.py::test_leray_temporal_order (shared vortex MMS),
but runs WITH the bake-off winning config (expected standard + ppe_fine_scale
=True). HARD gate: velocity temporal order >= 1.9 ("recover 2nd order in
time"). Pressure temporal order is REPORTED (not gated); pressure order
< ~1.5 prints a note flagging pressure_update=rotational as the booster."""
import os, sys, json
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(__file__))
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.steppers.leray import LerayProjectionStepper
from test_ns_stepper import u_ex, f_ex, NU

def _win():
    bo = os.path.join(os.path.dirname(__file__), "baselines", "p2r2a_bakeoff_3d.json")
    pu, fs = "standard", True
    if os.path.exists(bo):
        w = json.load(open(bo)).get("winner") or {}
        pu, fs = w.get("pressure_update", pu), bool(w.get("ppe_fine_scale", fs))
    return pu, fs

def _make(level, dt, device, pu, fs, order=2, picard=2):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    st = LerayProjectionStepper(
        dm, NU, dt, f_fn=f_ex, g_fn=lambda x, t: np.zeros((len(x), 2)),
        order=order, picard_iters=picard, timestab=(fs),
        pressure_update=pu, ppe_fine_scale=fs)
    st.set_initial(lambda x: u_ex(x, 0.0))
    return st

def _run(level, nsteps, T, device, pu, fs):
    st = _make(level, T / nsteps, device, pu, fs)
    for _ in range(nsteps):
        u, p = st.step()
    return u, p          # u: (n_free, dim) velocity; p: (n_free,) pressure

def _order( refs, seqs):
    """log2 rates of the coarse->fine self-convergence to the reference."""
    errs = [np.sqrt(((s - refs) ** 2).sum(-1).mean()) if s.ndim > 1
            else np.sqrt(((s - refs) ** 2).mean()) for s in seqs]
    rates = [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]
    return errs, rates

def test_r2a_temporal_order_second_order(device):
    """HARD: velocity temporal order >= 1.9 (BDF2 preserved). REPORT pressure
    temporal order; if pressure order < 1.5, print the rotational-booster note."""
    pu, fs = _win()
    if pu == "chorin":
        pytest.skip("winner is Chorin (1st-order) — Task 5 escalation, not this gate")
    T = 0.25
    uref, pref = _run(4, 64, T, device, pu, fs)
    us, ps = [], []
    for n in (8, 16, 32):
        u, p = _run(4, n, T, device, pu, fs)
        us.append(u); ps.append(p)
    u_errs, u_rates = _order(uref, us)
    p_errs, p_rates = _order(pref, ps)
    print(f"\n[r2a-temporal] config=({pu},fs={fs})", flush=True)
    print(f"  VELOCITY errs={u_errs} rates={u_rates}", flush=True)
    print(f"  PRESSURE errs={p_errs} rates={p_rates}  (reported, not gated)", flush=True)
    if p_rates[0] < 1.5:
        print(f"  NOTE: pressure temporal order {p_rates[0]:.2f} < 1.5 — try "
              f"pressure_update='rotational' as the documented pressure-order "
              f"booster (velocity order is the hard gate).", flush=True)
    # HARD gate: velocity 2nd order in the BDF2 regime (coarse rate).
    assert u_rates[0] >= 1.9, (
        f"velocity temporal order {u_rates[0]:.2f} < 1.9 — 2nd order NOT "
        f"recovered (u_errs={u_errs}, u_rates={u_rates})")
```
Note the velocity threshold is 1.9 on the FIRST (BDF2-regime) rate, matching the R0/Leray convention that the finest ratios drift toward the splitting floor. Pressure order is REPORTED only (per coordinator: the 1st-order-incremental + fine-scale reference gives stable + 2nd-order VELOCITY but pressure may be lower; rotational is the pressure-order booster if 2nd-order PRESSURE is later required). (If the winning config also makes the finest velocity rate ≥ 1.9, tighten to assert all velocity rates ≥ 1.9.)

Run locally:
```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim
python -m pytest tests/test_p2r2a_temporal_order.py -v -s
```
Expected: velocity order ≥ 1.9 (2nd order recovered). If it is ~1.0, the winning config is NOT 2nd-order → NEEDS_CONTEXT (Task 5): the stability came at the cost of time accuracy.

- [ ] **Step 4: Run the stability gate on gpubox**
```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim
DIFFSIM_NIGHTLY=1 python -m pytest \
    tests/test_p2r0_projection_sbm.py::test_g6_sphere_3d_r2a_stability_gate \
    -v -s 2>&1 | tee /tmp/r2a_gate.log
cat tests/baselines/p2r0_task10_sphere.json
```
Expected: PASS — projection Cd positive, within 20% of monolithic; weak-div < 1e-8; BDF2 engaged; mutation (baseline) diverges.

- [ ] **Step 5: Confirm both gates green, then commit**

Both the stability gate (Step 4, gpubox) AND the 2nd-order temporal gate (Step 3, local) must be green. If EITHER fails, do NOT commit as a pass — go to Task 5.
```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim
git rev-parse --abbrev-ref HEAD   # must print "p2-r2a"
git add tests/test_p2r0_projection_sbm.py tests/p2r0_task10_sphere_derisk.py \
        tests/test_p2r2a_temporal_order.py tests/baselines/p2r0_task10_sphere.json
git commit -m "$(cat <<'EOF'
feat(p2-r2a): lock the winner + dual gate (stability + 2nd-order temporal) (Task 4)

Wires the bake-off winner (expected standard + ppe_fine_scale=True) as the
R2a 3-D setting. Stability gate: projection Cd within 20% of monolithic
same-mesh, weak-div machine-zero, BDF2 engaged, mutation (baseline)
diverges. 2nd-order temporal gate (MMS vortex, mirrors test_leray
temporal-order): velocity temporal order >= 1.9 — recovers 2nd order in
time. Baseline updated.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Fallback-Decision (NEEDS_CONTEXT escalation)

**When to execute:** ONLY if **no stable config also achieves 2nd-order temporal accuracy** — i.e. Task 3's bake-off verdict is `NEEDS_CONTEXT` (Chorin-only or no-stable), OR Task 4's 2nd-order temporal gate fails for the stable winner (stability bought at the cost of time accuracy). If Tasks 2–4 all pass (a non-Chorin config is stable AND ≥1.9 temporal order), SKIP this task.

**What it does:** Documents the monolithic-block-preconditioner 3-D path as the R2 fallback and hands the supervisor a concrete re-scoping decision.

**Files:**
- Create: `docs/dev/2026-07-21-p2-r2a-fallback-decision.md`

**Interfaces:**
- Consumes: Task-1 verdict (`tests/baselines/p2r2a_diagnostic_3d.json`); Task-3 bake-off (`tests/baselines/p2r2a_bakeoff_3d.json`); Task-4 gate results; `src/diffsim/solvers/block_precond.py` (AMGX-backed monolithic path).
- Produces: a supervisor decision document.

- [ ] **Step 1: Write the fallback-decision document**
```markdown
# P2-R2a Fallback Decision — Monolithic Block-Preconditioner Path

**Date:** 2026-07-21
**Status:** NEEDS_CONTEXT escalation — supervisor decision required.

## Trigger
[Fill in: which trigger fired — Task-3 bake-off Chorin-only/no-stable, OR
Task-4 2nd-order temporal gate failed for the stable winner. Include the
bake-off config matrix summary and the measured temporal order.]

## What was tried
[Fill in: Task-1 incremental-feedback verdict (4194601); the bake-off matrix
(||p_hat|| bounded/growing, weak-div flat/decaying, Cd per config); the
winner (if any) and its temporal order.]

## The stable 3-D solver: monolithic block preconditioner
`src/diffsim/solvers/block_precond.py::BlockAMGPreconditioner` +
`solve_block_preconditioned` (AMGX-backed). The 3-D monolithic (no-split)
SBM-NS solver is already demonstrated stable (Cd=0.381, Task-10 report) on the
same mesh, and is 2nd-order in time by construction (no pressure splitting).

## Re-scoping proposal
If no projection config is BOTH stable and 2nd-order:
- R2b device port targets the monolithic block preconditioner as the scalable
  3-D solver.
- R2c literature validation uses the monolithic stepper.
- The projection limitation is a documented R2c finding.

## Decision required
- [ ] Accept re-scoping: monolithic as the R2 3-D solver.
- [ ] Continue projection: the surprise is [X]; next diagnostic step is [Y].
```

- [ ] **Step 2: Commit the fallback document**
```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim
git rev-parse --abbrev-ref HEAD   # must print "p2-r2a"
git add docs/dev/2026-07-21-p2-r2a-fallback-decision.md
git commit -m "$(cat <<'EOF'
docs(p2-r2a): fallback-decision for monolithic path (Task 5)

NEEDS_CONTEXT escalation: if no projection config is both stable and
2nd-order in time, documents the monolithic block-preconditioner
(block_precond.py, AMGX-backed) as the R2 3-D solver re-scoping option.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

### 1. Spec Coverage (Baskar: "Do 2 then 1 … recover second order in time")

| Requirement | Task |
|---|---|
| Diagnostic isolates the 3-D pressure-coupling divergence | Task 1 (DONE, `4194601`) — incremental-feedback, flat weak-div |
| Projection knobs, all defaulting to CURRENT behavior (default bit-for-bit) | Task 2 — `pressure_update` enum + `ppe_fine_scale`; `test_default_step_bitforbit` |
| `pressure_update="standard\|rotational\|chorin"` (rotational reuses `3942251` spec; chorin zeros `p*`) | Task 2 Steps 1, 3 |
| Bounded-pressure / pressure-extrapolation knob (Taly `pressure_extrap_c`) | Task 2 — folded into the `pressure_update` enum (interaction table); never 2nd-order extrap |
| `ppe_fine_scale`: ONE `u'=−τ_M R` (R with LAGGED `∇p_star`) threaded into BOTH PPE source `σ(∇q,u')` AND velocity update, τ_m at correct `dt`; predictor left as-is | Task 2 knob-2 bullet + Steps 3(b), 4 |
| New `ppe_fine_scale` must NOT inherit the `dt=Δt/b0` bug; reuse leray.py's existing `r_m` (lagged `∇p_star`), not a second residual | Task 2 knob-2 bullet + Step 1 note + Step 3(b) `taum_fs` with `self.dt`; Global Constraints |
| Bake-off + temporal gate REPORT BOTH velocity AND pressure temporal order (velocity ≥1.9 hard; pressure reported, rotational = booster if pressure order <1.5) | Task 3 Step 2 expectation + Task 4 Step 3 (`_order`, pressure note) |
| Per-knob unit tests: (a) default bit-for-bit, (b) isolated math (rotational-zero, fine-scale-exact) | Task 2 Steps 2, 5 (2-D `dm` fixtures, local) |
| Bake-off sweeping levers one-at-a-time + combos; record `‖p̂‖`/weak-div/Cd; verdict = minimal stable set | Task 3 |
| Bake-off winner = minimal STABLE config; Chorin-only ⇒ NEEDS_CONTEXT | Task 3 verdict logic; Global Constraints |
| Task 4 stability gate: Cd within 20% of monolithic, weak-div machine-zero, BDF2, mutation=baseline diverges | Task 4 Step 2 |
| Task 4 HARD 2nd-order temporal gate (mirrors `test_leray` temporal-order), velocity order ≥ 1.9 | Task 4 Step 3 |
| Fallback = monolithic path; trigger = no stable config also achieves 2nd order | Task 5 |
| gpubox for 3-D marches; unit + temporal gate local | Task 3/Task 4 Step 4 (gpubox); Task 2, Task 4 Step 3 (local) |
| REPO-IDENTITY GUARD = `p2-r2a` before every commit; agents never push | every commit step; Global Constraints |
| `Co-Authored-By: Claude Opus 4.8` trailer | Tasks 2–5 commit messages |

### 2. Placeholder Scan
Intentional, flagged wiring placeholders remain in Task 2 (the `_small_2d_stepper` 2-D `dm` builder — grep `tests/test_p2r0_projection_sbm.py`; the two `...`-bodied isolated-math tests) and Task 4/Task 5 (`...` baseline-JSON write; `[Fill in]` in the fallback doc). Each has an adjacent NOTE naming the exact source to wire (the `test_p2r0_projection_sbm.py` `dm` sequence, the `test_leray.py` MMS harness, the bake-off JSON `winner`). No unresolved code `TODO`. The load-bearing assertions are fully specified: default `np.array_equal` parity, rotational `‖q‖<1e-10`, fine-scale exact-term to 1e-12, stability gate (Cd/weak-div/BDF2/mutation), temporal order ≥ 1.9.

### 3. Type / Name Consistency of the knob API across Tasks 2–4
- `pressure_update` (str ∈ `{"standard","rotational","chorin"}`, default `"standard"`) — defined Task 2 Step 1; threaded Task 2 Step 6; consumed Task 3 (`CONFIGS`), Task 4 (`march_projection_r2a`, `WIN_PU`, temporal `_make`). Same literals throughout.
- `ppe_fine_scale` (bool, default `False`) — defined Task 2 Step 1; threaded Task 2 Step 6; consumed Task 3 (`CONFIGS`), Task 4 (`march_projection_r2a`, `WIN_FS`, temporal `_make`). Distinct from the pre-existing `ppe_finescale` (no underscore) — Naming note in Task 2; both flags coexist on `__init__`.
- `march_projection_r2a(fx, dt, max_steps, rate_tol, order=2, beta_backflow=1.0, picard_iters=2, pressure_update="standard", ppe_fine_scale=True, alpha=100.0) -> dict(cd,clat,steps,finite,bdf2_engaged,st,u,p)` — defined Task 4 Step 1; called Task 4 Steps 2, 4.
- `_weak_divergence_3d(st) -> (float,float)` — Task 1 (`4194601`); used Task 3 (`_march`) and Task 4 Step 2 (`pr["st"]`).
- Bake-off JSON contract `winner = {label, pressure_update, ppe_fine_scale}` / `verdict` (str) — produced Task 3; consumed Task 4 Steps 2, 3 and the temporal gate `_win()`.
- Rotational internals (Task 2 Step 3c): `bt_uhat = rhs_free/sigma` (fine-scale-off) or `self._weak_div_free(uhat)` (fine-scale-on); `q = solve_linear(self.M, ..., cache_key="mass")`; `p_hat = self.p_star + phi - self.nu*q`. Fine-scale internals (Steps 3b, 4): `taum_fs` at `self.dt`, `r_m = sigma*aqv + agu + pq_g - fq_base`, `flux = sigma*(aqv - taum_fs*r_m)`, velocity `integ -= fs_vel`.

All names/types are consistent across Tasks 2–4.

## Supervisor resolutions (2026-07-21)

1. **Scope — DECIDED (Baskar).** "Do 2 then 1" + "recover second order in time": bake-off FIRST (Task 3) to isolate the minimal lever set, then lock+gate (Task 4). The final config must preserve BDF2 2nd order; Chorin confirms the mechanism but is 1st-order and CANNOT be the answer. Target = incremental (`p*=p^n`) + VMS-consistent fine-scale, possibly + rotational.
2. **Knob API — DECIDED: one `pressure_update` enum + one `ppe_fine_scale` bool.** Taly's `pressure_extrap_c` is FOLDED into `pressure_update` (chorin=order 0, standard/rotational=order 1) rather than a separate int, to forbid illegal combinations (interaction table, Task 2). We never expose 2nd-order pressure extrapolation.
3. **New `ppe_fine_scale` flag is distinct from the old `ppe_finescale`** (underscore vs none) and computes τ_M at the CORRECT `dt` — the old `dt=Δt/b0` path (audit §5) is left untouched.
4. **Fine-scale discrete form — VERIFIED against Taly + leray.py.** PPE source `flux = sigma*(u_hat - tau_M r_m)` (Taly `Proj_Linear_PPE_Integrands` line ~294); velocity update adds `-tau_M r_m` (Taly `Proj_Linear_VUE_Integrands` line ~230). `r_m = sigma*u_hat + a.grad u_hat + grad p* - f` is ALREADY assembled in leray.py's PPE loop (lines ~404–418) for the old branch; the new lever reuses it with the correct `tau_M`.
5. **Winner escalation — DECIDED.** If the only stable config is Chorin (1st-order), or the stable winner fails the ≥1.9 temporal gate, escalate to Task 5 (monolithic fallback). The VMS fine-scale consistency is EXPECTED to make the incremental/2nd-order path stable — a Chorin-only outcome is a genuine surprise.
6. **Fine-scale = ONE shared object with LAGGED pressure (coordinator, from Taly NL integrands `Proj_NL_PPE`@585 / `Proj_NL_VUE`@526 / `Proj_NL_Momentum_BDF12`@637).** All three share `u' = −τ_M R`, where `R`'s pressure term is the LAGGED `∇p^n` (Taly `vpre1.gradp`), = `∇p_star` in our incremental setting. Same `u'` in the PPE source (`σ(∇q,u')`) and the velocity update (`u^{n+1}=û+u'−(1/σ)∇φ`); predictor left as-is for R2a. Implementer REUSES leray.py's already-assembled `r_m` (lagged `∇p_star`) — no second residual — with τ_M at the correct `dt`.
7. **Report BOTH temporal orders (coordinator).** The 1st-order-incremental + fine-scale reference is stable AND 2nd-order in VELOCITY without rotational; pressure may be lower order. So the Task-4 temporal gate + bake-off REPORT both velocity and pressure order: velocity ≥ 1.9 is the HARD gate; pressure order is reported, and if pressure order < ~1.5 the plan documents `pressure_update="rotational"` as the pressure-order booster to try (not a hard failure). The expected bake-off winner is therefore `(standard, ppe_fine_scale=True)`; rotational is reserved for a later 2nd-order-PRESSURE requirement.
