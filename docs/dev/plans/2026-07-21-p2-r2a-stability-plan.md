# P2-R2a — 3-D Projection Pressure-Coupling Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gate the 3-D projection stepper's stability by (1) [DONE] diagnosing which pressure-coupling mechanism drives the 3-D divergence, (2) implementing the rotational / consistent-incremental pressure form (Timmermans) — `p_hat = p* + φ − ν(∇·û)` — which removes the spurious pressure boundary layer that the classic-incremental update's implicit homogeneous-Neumann pressure BC creates, and (3) confirming 3-D sphere Cd faithfulness vs. the monolithic reference — or escalating to the monolithic fallback if the fix is insufficient.

**Architecture:** Diagnostic-first: Task 1 (DONE, committed `4194601`) ruled out all four candidate mechanisms with machine-precision negatives (M1 PPE-conditioning residual 1e-14; M2 projection-space identity `‖σBᵀû−K_pφ‖`=1.16e-14 holds in 3-D; M3 φ-gauge drift 3.3e-13; M4 penalty diverges through α=20000). The surviving signature — pressure `‖p̂‖` growing ~4× over 12 steps (162→1307) with Cd→−8 at Stokes, while PPE residual stays 1e-14, weak-divergence stays flat, and the no-penetration leak stays ~1e-3 — is an **incremental-pressure feedback instability**: the classic-incremental update `p* += φ` accumulates (stable in 2-D, unstable in 3-D). The targeted fix adds a `pressure_update="standard|rotational|chorin"` knob to `LerayProjectionStepper` (mirroring the existing `velocity_update` knob), threaded through `LeraySBMStepper`. `"standard"` (default) is the unchanged `p_hat = p* + φ`. `"rotational"` computes `p_hat = p* + φ − ν·q` where `q` solves the pressure mass system `M_p q = Bᵀû` (nodal weak divergence of the predictor). `"chorin"` is a non-incremental mechanism-confirmation mode that resets `p* ≡ 0` each step (no accumulation). The R2a stability gate re-uses the Task-10 sphere fixture and asserts projection Cd matches monolithic to within 20%, weak-div machine-zero, BDF2 engaged. A fallback-decision task documents the monolithic path if Tasks 2–3 cannot stabilize.

**Tech Stack:** Python / NumPy / SciPy (host-side; all 3-D runs on gpubox CPU via `splu`); pytest; JSON baselines. The rotational fix is entirely in the Python `step()` path of `leray.py` (no Warp kernel change: the `−ν(∇·û)` term reuses the already-assembled scalar consistent mass matrix `self.M` as `M_p` and the already-computed PPE `rhs_free` as `σ·Bᵀû`).

## Global Constraints

- R2a GATE: 3-D sphere projection Cd matches monolithic same-mesh reference (faithfulness in 3-D) + weak-div machine-zero + BDF2 engaged.
- Gate-hygiene: independent reference (3-D monolithic on matched mesh) + mutation/planted-break leg; no vacuous gates.
- Rotational-fix non-vacuity: the gate must include a mutation leg that shows the classic-incremental form (`pressure_update="standard"`) diverges on the 3-D sphere while `"rotational"` stays finite/positive — proving the gate is load-bearing.
- REPO-IDENTITY GUARD: before every commit, verify `git rev-parse --abbrev-ref HEAD` == `p2-r2a`; abort if not.
- Agents never push; supervisor pushes.
- All 3-D sphere marches run on **gpubox** (host/splu, 40-core CPU). The 2-D diagnostic and unit-level tests run locally.
- `ppe_finescale=True` τ_m bug (latent: `dt=Δt/b0` over-scales transient by b0² for BDF2, documented in `docs/dev/2026-07-21-p2-r0-parity-audit.md` §5) MUST be fixed before any task that enables `ppe_finescale=True`. Do not enable `ppe_finescale=True` in any task here unless the bug fix is included in that task.
- R2b (device port) and R2c (literature validation) are separate follow-on plans, out of scope.
- MMS parity: ±0.10 order is a standing contract; do not regress `tests/test_p2r0_parity.py`.
- Absolute paths only. Working directory: `/Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim`.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `tests/p2r2a_diagnostic_3d.py` | [DONE, committed `4194601`] 3-D pressure-coupling mechanism measurement (Task 1) |
| Create | `tests/baselines/p2r2a_diagnostic_3d.json` | [DONE] Diagnostic verdict record written by the diagnostic script |
| Modify | `src/diffsim/steppers/leray.py` | Add `pressure_update="standard\|rotational\|chorin"` knob + rotational `p_hat = p* + φ − ν·q` update (reuses `self.M` as `M_p`, `rhs_free/σ` as `Bᵀû`) (Task 2) |
| Modify | `src/diffsim/steppers/leray_sbm.py` | Thread `pressure_update` through `__init__` to the base stepper (Task 2) |
| Create | `tests/test_p2r2a_rotational_pressure.py` | Unit + gate tests for the rotational-incremental fix, Chorin confirmation, standard-parity, and non-vacuity mutation (Task 2) |
| Modify | `tests/p2r0_task10_sphere_derisk.py` | Add `march_projection_r2a` helper with `pressure_update` parameter (Task 3) |
| Modify | `tests/test_p2r0_projection_sbm.py` | Add `test_g6_sphere_3d_r2a_stability_gate` (Tasks 3) |
| Modify | `tests/baselines/p2r0_task10_sphere.json` | Update with the R2a matched-mesh result (Task 3) |
| Create | `docs/dev/2026-07-21-p2-r2a-fallback-decision.md` | Fallback-decision record (Task 4, if escalation triggered) |

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
  - Any other verdict → NEEDS_CONTEXT escalation: document in `docs/dev/2026-07-21-p2-r2a-fallback-decision.md` and skip to Task 4.

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

## Task 2: Rotational / Consistent-Incremental Pressure Form

**Prerequisite:** Task 1 diagnostic (DONE, `4194601`) confirmed the incremental-pressure feedback instability (`p* += φ` accumulation) after ruling out M1–M4. This task implements the fix. If any re-run of Task 1's diagnostic were to name a different mechanism, escalate to Task 4 — but the verdict is already committed and decisive.

**What this task builds:**
- A `pressure_update="standard|rotational|chorin"` keyword on `LerayProjectionStepper.__init__`, mirroring the existing `velocity_update` knob (lines ~39, 55–59, 90–111 of `leray.py`). `"standard"` (default) is the unchanged classic-incremental `p_hat = p* + φ`.
- `"rotational"`: the Timmermans consistent-incremental form `p_hat = p* + φ − ν·q`, where `q` solves the pressure mass system `M_p q = Bᵀû` (the nodal weak divergence of the predictor velocity `û`). The `−ν(∇·û)` term removes the spurious pressure boundary layer the classic-incremental form's implicit homogeneous-Neumann pressure BC creates.
- `"chorin"`: a non-incremental mechanism-confirmation mode. Each step resets `p* ≡ 0` before the predictor (so the momentum predictor never sees an accumulating `grad p*`, and `p_hat = φ` with no step-over-step accumulation). Used once, at the start of this task, to CONFIRM the accumulation is the cause — Chorin stays bounded (finite Cd) while `"standard"` diverges.
- Threading the knob through `LeraySBMStepper.__init__` to the base stepper.
- Unit-level parity/zero tests (no gpubox), plus a gpubox non-vacuity mutation leg.

**The rotational formula (exact):**
```
p_hat = p_star + phi - nu * q,    where   M_p q = B^T u_hat
```
Where, INSIDE `LerayProjectionStepper.step()`:
- `phi` is the existing PPE increment (line ~454), unchanged.
- `Bᵀû` (the nodal weak divergence of the predictor `uhat`) is obtained WITHOUT re-assembly: in the `ppe_finescale=False` branch the PPE RHS is built with `flux = sigma * aqv` (line ~421), so `rhs = sigma · Bᵀû` and therefore **`rhs_free / sigma` IS the free-node weak divergence `Bᵀû`** (with the pin already applied at free-node 0). Verified against `leray.py` lines 421, 429–438, 452. In the `ppe_finescale=True` branch the flux is NOT `sigma · Bᵀû` (it carries the tau_m fine-scale residual), so the rotational term must NOT reuse `rhs_free` there — see the guard below.
- `M_p` is the scalar consistent mass matrix **already assembled as `self.M`** (built in `_mass_matrix()`, lines ~115–134, as `T.T @ M @ T` — the same free-node scalar space that `phi` and `p_star` live in; `phi` and `K_p` are also `T.T`-constrained). No new assembly is required. `M_p` (a mass matrix) is SPD and invertible without any pin, so DO NOT apply the free-node-0 pin to it. Solve via the existing `solve_linear(self.M, ...)` path with `cache_key="mass"` — the SAME key the velocity update uses, since both invert the identical `self.M`, so the factorization is shared (no duplicate factorize).

**Pressure-mass decision (recorded):** reuse the existing consistent scalar mass `self.M` as `M_p` — do NOT assemble a new (lumped) pressure mass matrix. Rationale: (1) `self.M` already exists, is SPD, is in the exact free-node scalar pressure space, and is cached through `solve_linear`; (2) the consistent (not lumped) mass gives the theoretically-correct L2 projection of `∇·û` onto nodal pressure space that the Timmermans form specifies; (3) it costs one extra back-substitution per step against an already-factorized operator. A lumped `M_p` would be a micro-optimization with no scalability payoff at these mesh sizes and would introduce a projection error into the rotational term. **Flagged for supervisor:** if a future device port makes the consistent-mass solve a bottleneck, revisit lumped `M_p`; for R2a (host/splu) consistent is correct and cheap.

**Files:**
- Modify: `src/diffsim/steppers/leray.py` (add `pressure_update` knob + rotational/chorin branches in `__init__` and `step()`)
- Modify: `src/diffsim/steppers/leray_sbm.py` (thread `pressure_update` through `__init__` to the base stepper)
- Create: `tests/test_p2r2a_rotational_pressure.py` (Chorin confirmation + standard-parity + divergence-free-zero unit tests + non-vacuity gate)

**Interfaces:**
- Consumes: `LerayProjectionStepper(dm, nu, dt, f_fn, g_fn, ..., velocity_update="consistent", graddiv_scale=1.0, pressure_update="standard")` from `src/diffsim/steppers/leray.py`; `build_sphere_3d, R, U_IN` from `tests/p2r0_task10_sphere_derisk.py`.
- Produces: `LeraySBMStepper(..., pressure_update="standard")` — the `pressure_update="rotational"` path for Task 3.

- [ ] **Step 1: Chorin (non-incremental) confirmation of the accumulation mechanism**

Before implementing the rotational fix, add the `pressure_update` knob with the `"chorin"` mode and confirm the mechanism: a non-incremental march (no `p*` accumulation) stays bounded while classic-incremental diverges.

First, add the knob to `LerayProjectionStepper.__init__` in `src/diffsim/steppers/leray.py`. Mirror the `velocity_update` validation block (lines ~55–59). After the `velocity_update` validation and `self.velocity_update = velocity_update` line, insert:

```python
        # P2-R2a pressure-update knob (default "standard" = unchanged classic-
        # incremental p_hat = p* + phi). Diagnostic 4194601 traced the 3-D
        # divergence to p*-accumulation feedback; see the plan.
        #   "standard"   — classic incremental: p_hat = p* + phi (Algorithm 1).
        #   "rotational" — Timmermans consistent-incremental:
        #                    p_hat = p* + phi - nu * q,  M_p q = B^T u_hat.
        #                  Removes the spurious pressure boundary layer from the
        #                  classic form's implicit homogeneous-Neumann p-BC.
        #   "chorin"     — non-incremental confirmation mode: p* is reset to 0
        #                  each step (no accumulation); p_hat = phi. Used to
        #                  confirm the accumulation is the divergence cause.
        if pressure_update not in ("standard", "rotational", "chorin"):
            raise ValueError(
                f"pressure_update must be standard|rotational|chorin, "
                f"got {pressure_update!r}")
        self.pressure_update = pressure_update
```

And add `pressure_update="standard"` to the `__init__` signature (append to the keyword list after `graddiv_scale=1.0`):

```python
    def __init__(self, dm, nu, dt, f_fn, g_fn, order=2, picard_iters=2,
                 solver="splu",
                 timestab=True, ppe_finescale=False, predictor="picard",
                 velocity_update="consistent", graddiv_scale=1.0,
                 pressure_update="standard"):
```

Now wire the two behavioral hooks in `step()`. (a) The Chorin reset: at the very top of `step()`, before the predictor call (`uhat = self._predict(...)`, line ~390), insert:

```python
        # P2-R2a: Chorin (non-incremental) confirmation mode — zero the
        # accumulated pressure each step so the predictor never sees a
        # compounding grad p* and p_hat = phi (no step-over-step feedback).
        if self.pressure_update == "chorin":
            self.p_star = np.zeros(self.n_free)
```

(b) Replace the pressure update at line ~461 (`p_hat = self.p_star + phi`) with the branch (the rotational branch is implemented in Step 4; for THIS step, implement `standard` and `chorin` only, with rotational falling through to standard so nothing breaks yet):

```python
        # ---- pressure update (P2-R2a pressure_update knob) ----
        if self.pressure_update == "chorin":
            p_hat = phi.copy()            # p* was zeroed above; no accumulation
        else:
            # "standard" (and, until Step 4, "rotational") classic incremental
            p_hat = self.p_star + phi
```

Create `tests/test_p2r2a_rotational_pressure.py` with the Chorin confirmation (gpubox-guarded) and the knob-validation unit test:

```python
"""Tests for the P2-R2a rotational / consistent-incremental pressure fix.

Diagnostic 4194601 traced the 3-D projection divergence to classic-
incremental p*-accumulation feedback. This module confirms the mechanism
(Chorin stays bounded) and gates the rotational fix.
"""
import os
import sys

import numpy as np
import pytest


def test_pressure_update_knob_validates():
    """The pressure_update knob rejects unknown modes (mirrors velocity_update)."""
    import scipy  # noqa: F401  (import kept parallel to other unit tests)
    from diffsim.steppers.leray import LerayProjectionStepper
    with pytest.raises(ValueError, match="pressure_update must be"):
        # dm=None is fine: the validation happens before dm is touched.
        LerayProjectionStepper.__init__.__wrapped__ if False else None
        # Construct enough to hit the validation. Use a minimal real dm via the
        # shared 2-D fixture builder if available; otherwise assert the guard
        # string is present (the ValueError above is the load-bearing check).
        raise ValueError("pressure_update must be standard|rotational|chorin")


def test_chorin_bounded_while_standard_diverges(device):
    """MECHANISM CONFIRMATION (gpubox): on the level-4 Stokes sphere the
    non-incremental (Chorin) march stays bounded (finite Cd) while classic-
    incremental (standard) accumulates and diverges (Cd runs strongly
    negative). This confirms p*-accumulation is the divergence cause.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from p2r0_task10_sphere_derisk import build_sphere_3d, R, U_IN
    if not os.environ.get("DIFFSIM_NIGHTLY"):
        pytest.skip("nightly / gpubox only")
    from diffsim.steppers.leray_sbm import LeraySBMStepper

    fx = build_sphere_3d(device, level=4, Re=1.0)
    q = 0.5 * U_IN ** 2 * np.pi * R ** 2
    dt = 0.05
    dim = fx["dim"]

    def f_fn(x, t):
        return np.zeros((len(x), dim))

    def _march(pressure_update, nsteps=15):
        st = LeraySBMStepper(
            fx["oracle"], fx["dm"], fx["nu"], dt, f_fn,
            u_inf=fx["u_inf"], strong_mask=fx["strong_mask"],
            lam=0.5, domain="outside", order=1, picard_iters=2,
            solver="splu", ppe_finescale=False, alpha=100.0,
            beta_backflow=1.0, velocity_update="consistent",
            pressure_update=pressure_update)
        st.set_initial(lambda c: np.zeros((len(c), dim)))
        cds = []
        for _ in range(nsteps):
            u, p = st.step()
            F = st.surrogate_traction()
            cds.append(float(F[0] / q))
        return cds

    cds_std = _march("standard")
    cds_chorin = _march("chorin")
    print(f"\n[chorin] standard final Cd={cds_std[-1]:+.3f}  "
          f"chorin final Cd={cds_chorin[-1]:+.3f}", flush=True)
    # Classic-incremental diverges (Cd strongly negative); Chorin stays bounded.
    assert cds_std[-1] < -1.0, (
        f"expected standard to diverge (Cd < -1), got {cds_std[-1]:.3f}")
    assert abs(cds_chorin[-1]) < 5.0 and np.isfinite(cds_chorin[-1]), (
        f"expected Chorin to stay bounded (|Cd| < 5), got {cds_chorin[-1]:.3f}")
```

> **NOTE on `test_pressure_update_knob_validates`:** the stub above is a placeholder shape only — replace it during implementation with a construction that reaches the guard using the repo's smallest real 2-D `dm` fixture (see how `tests/test_p2r0_projection_sbm.py` builds a `dm`; reuse that builder and pass `pressure_update="bogus"` to `LerayProjectionStepper`, asserting the `ValueError`). The load-bearing assertion is that an unknown mode raises `ValueError` with the "pressure_update must be" message.

- [ ] **Step 2: Run the knob-validation unit test locally; run the Chorin confirmation on gpubox**

Locally (no gpubox):
```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim
python -m pytest tests/test_p2r2a_rotational_pressure.py::test_pressure_update_knob_validates -v
```
Expected: PASS (the `ValueError` guard fires).

On gpubox:
```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim
DIFFSIM_NIGHTLY=1 python -m pytest \
    tests/test_p2r2a_rotational_pressure.py::test_chorin_bounded_while_standard_diverges \
    -v -s 2>&1 | tee /tmp/chorin.log
```
Expected: `standard` final Cd < −1 (diverging); `chorin` final Cd bounded (|Cd| < 5, finite). PASS. This CONFIRMS the accumulation is the cause. If Chorin ALSO diverges, the mechanism is not accumulation — STOP and escalate to Task 4 (the diagnostic's incremental-feedback reading would be contradicted).

- [ ] **Step 3: Write the failing unit tests for the rotational term (no gpubox)**

Append to `tests/test_p2r2a_rotational_pressure.py`. These pin the rotational contract without needing a 3-D march: (a) `pressure_update="standard"` reproduces the current `p_hat` bit-for-bit (parity), and (b) the rotational correction `−ν·q` is (near) zero when the predictor `û` is exactly divergence-free (so `Bᵀû ≈ 0 ⟹ q ≈ 0 ⟹ p_hat ≈ p* + φ`). Both use the small 2-D MMS/box `dm` fixture (the same one `tests/test_p2r0_projection_sbm.py` uses).

```python
def _small_2d_stepper(pressure_update):
    """Build a minimal 2-D LerayProjectionStepper on the shared box fixture.
    Mirrors the dm/g_fn construction used in tests/test_p2r0_projection_sbm.py;
    replace the import below with that module's fixture builder."""
    from diffsim.steppers.leray import LerayProjectionStepper
    # --- fixture: reuse the project's smallest 2-D dm builder ---
    # from <project 2-D fixture> import build_box_2d   # <-- wire to the real one
    # dm, nu, dt, f_fn, g_fn = build_box_2d(...)
    # return LerayProjectionStepper(dm, nu, dt, f_fn, g_fn, order=1,
    #                               pressure_update=pressure_update)
    raise NotImplementedError("wire to the shared 2-D dm fixture")


def test_standard_parity_bitforbit(device):
    """pressure_update='standard' reproduces the current p_hat bit-for-bit:
    stepping a 'standard' stepper and a default (no-arg) stepper from the same
    initial state gives identical p_hat after one step."""
    from diffsim.steppers.leray import LerayProjectionStepper
    # Build two identical steppers: one with the explicit default, one with
    # pressure_update='standard'. Marching one step from the same IC must give
    # p_hat identical to machine zero (the default IS 'standard').
    st_default = _small_2d_stepper(None) if False else None
    # Implementation: construct st_a (default) and st_b (pressure_update=
    # "standard") on the SAME dm/IC; assert np.array_equal after one step().
    pytest.skip("wire _small_2d_stepper to the shared 2-D dm fixture, then "
                "assert np.array_equal(p_hat_default, p_hat_standard)")


def test_rotational_term_zero_for_divergence_free_predictor(device):
    """When the predictor u_hat is exactly divergence-free, B^T u_hat = 0,
    so q = 0 and p_hat_rotational == p_hat_standard (the -nu*q term vanishes).
    Constructed by projecting a divergence-free field and checking ||q|| ~ 0."""
    pytest.skip("wire _small_2d_stepper to the shared 2-D dm fixture; set a "
                "solenoidal u_hat, assert ||M_p^{-1} B^T u_hat|| < 1e-10 and "
                "p_hat_rot == p_hat_std to 1e-12")
```

> **NOTE:** both tests above are skip-guarded pending the shared 2-D fixture wiring. During implementation, replace `_small_2d_stepper` with the repo's real 2-D `dm` builder (grep `tests/test_p2r0_projection_sbm.py` for how it constructs `dm`, `nu`, `dt`, `f_fn`, `g_fn`), then remove the `pytest.skip` lines and assert the stated contracts. These are the cheap, gpubox-free correctness anchors for the rotational term.

- [ ] **Step 4: Implement the rotational branch in `src/diffsim/steppers/leray.py`**

Replace the Step-1 pressure-update branch (the `if self.pressure_update == "chorin": ... else: p_hat = self.p_star + phi` block) with the full three-way branch. `phi`, `rhs_free`, and `sigma` are all in scope at line ~461 (the PPE just solved). Insert:

```python
        # ---- pressure update (P2-R2a pressure_update knob) ----
        if self.pressure_update == "chorin":
            # non-incremental: p* was zeroed at the top of step(); p_hat = phi.
            p_hat = phi.copy()
        elif self.pressure_update == "rotational":
            # Timmermans consistent-incremental: p_hat = p* + phi - nu * q,
            # M_p q = B^T u_hat. In the classic (ppe_finescale=False) branch the
            # PPE flux is sigma*u_hat, so rhs_free = sigma * B^T u_hat and the
            # weak divergence of the predictor is exactly rhs_free / sigma
            # (pin already applied at free-node 0). Solve the SPD consistent
            # pressure mass M_p = self.M (NO pin; a mass matrix is invertible).
            if self.ppe_finescale:
                raise NotImplementedError(
                    "pressure_update='rotational' requires ppe_finescale=False "
                    "(the fine-scale PPE flux is not sigma*B^T u_hat, so "
                    "rhs_free/sigma is not the weak divergence). R2a uses "
                    "ppe_finescale=False throughout.")
            bt_uhat = rhs_free / sigma           # nodal weak divergence B^T u_hat
            from ..solvers.linsolve import solve_linear
            # cache_key="mass": the velocity update inverts the SAME self.M, so
            # sharing the key reuses its factorization (no second factorize).
            q = solve_linear(self.M, bt_uhat, solver=self.solver, sym=True,
                             device=self.dm.device, cache=self._solver_cache,
                             cache_key="mass")
            p_hat = self.p_star + phi - self.nu * q
        else:                                    # "standard" (default)
            p_hat = self.p_star + phi
```

Notes:
- `rhs_free` here is the post-pin PPE RHS (its entry 0 was set to 0 at line ~452), consistent with `phi[0]` being the pinned pressure gauge; dividing by `sigma` preserves that gauge in `bt_uhat`.
- `self.M` is `T.T @ M @ T` (free-node scalar space) — the same space as `phi`, `q`, `p_star`. No `dm.constraints.T` transform is needed on `bt_uhat` (it is already free-node-sized).
- `cache_key="mass"` (shared with the velocity-update solve, which inverts the identical `self.M`) avoids a duplicate factorization.

- [ ] **Step 5: Run the rotational unit tests locally (parity + divergence-free zero)**

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim
python -m pytest tests/test_p2r2a_rotational_pressure.py::test_standard_parity_bitforbit \
                 tests/test_p2r2a_rotational_pressure.py::test_rotational_term_zero_for_divergence_free_predictor \
                 -v
```
Expected: both PASS (after the fixture wiring in Step 3 is completed). Parity: `p_hat` identical between default and `"standard"`. Divergence-free: `‖q‖ < 1e-10` and rotational `p_hat` equals standard `p_hat` to 1e-12.

- [ ] **Step 6: Thread `pressure_update` through `LeraySBMStepper`**

In `src/diffsim/steppers/leray_sbm.py`, add `pressure_update="standard"` to the `__init__` signature (after `graddiv_scale=1.0`, line ~86):

```python
    def __init__(self, oracle, dm, nu, dt, f_fn, *, u_inf, strong_mask,
                 lam=0.5, domain="outside", order=2, picard_iters=2,
                 solver="splu", ppe_finescale=False, alpha=10.0,
                 beta_backflow=1.0, velocity_update="consistent",
                 graddiv_scale=1.0, pressure_update="standard"):
```

Pass it through to the base stepper construction (line ~118–122):

```python
        base = LerayProjectionStepper(
            dm, nu, dt, f_fn, self._g_box, order=order,
            picard_iters=picard_iters, solver=solver,
            ppe_finescale=ppe_finescale,
            velocity_update=velocity_update, graddiv_scale=graddiv_scale,
            pressure_update=pressure_update)
```

- [ ] **Step 7: Write the non-vacuity gate (mutation leg)**

Append to `tests/test_p2r2a_rotational_pressure.py`. The mutation leg proves the fix is load-bearing: classic-incremental (`pressure_update="standard"`) diverges on the 3-D sphere while `"rotational"` stays finite/positive.

```python
def test_rotational_non_vacuity_standard_diverges(device):
    """Non-vacuity gate (gpubox): on the level-4 Stokes sphere,
    pressure_update='standard' (classic-incremental) accumulates and diverges
    (Cd strongly negative) while 'rotational' stays finite and positive.
    This proves the rotational fix is load-bearing, not decorative.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from p2r0_task10_sphere_derisk import build_sphere_3d, R, U_IN
    if not os.environ.get("DIFFSIM_NIGHTLY"):
        pytest.skip("nightly / gpubox only")
    from diffsim.steppers.leray_sbm import LeraySBMStepper

    fx = build_sphere_3d(device, level=4, Re=1.0)
    q = 0.5 * U_IN ** 2 * np.pi * R ** 2
    dt = 0.05
    dim = fx["dim"]

    def f_fn(x, t):
        return np.zeros((len(x), dim))

    def _march(pressure_update, nsteps=15):
        st = LeraySBMStepper(
            fx["oracle"], fx["dm"], fx["nu"], dt, f_fn,
            u_inf=fx["u_inf"], strong_mask=fx["strong_mask"],
            lam=0.5, domain="outside", order=1, picard_iters=2,
            solver="splu", ppe_finescale=False, alpha=100.0,
            beta_backflow=1.0, velocity_update="consistent",
            pressure_update=pressure_update)
        st.set_initial(lambda c: np.zeros((len(c), dim)))
        cds = []
        for _ in range(nsteps):
            u, p = st.step()
            F = st.surrogate_traction()
            cds.append(float(F[0] / q))
        return cds

    cds_std = _march("standard")
    cds_rot = _march("rotational")
    print(f"\n[nonvac] standard final Cd={cds_std[-1]:+.3f}  "
          f"rotational final Cd={cds_rot[-1]:+.3f}  "
          f"std_traj={[f'{c:.2f}' for c in cds_std[:5]]}", flush=True)
    assert cds_std[-1] < -1.0, (
        f"expected standard (classic-incremental) to diverge (Cd < -1), "
        f"got {cds_std[-1]:.3f}")
    assert np.isfinite(cds_rot[-1]) and cds_rot[-1] > 0.0, (
        f"expected rotational to stabilize (Cd > 0), got {cds_rot[-1]:.3f}")
```

- [ ] **Step 8: Run the non-vacuity mutation test on gpubox**

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim
DIFFSIM_NIGHTLY=1 python -m pytest \
    tests/test_p2r2a_rotational_pressure.py::test_rotational_non_vacuity_standard_diverges \
    -v -s 2>&1 | tee /tmp/rot_nonvac.log
```
Expected: `standard` final Cd < −1 (diverging); `rotational` final Cd > 0 (stable). Both assertions PASS.

If `rotational` also diverges, this is a NEEDS_CONTEXT: the `−ν(∇·û)` term did not remove the feedback. Verify (i) `bt_uhat = rhs_free/sigma` matches an independent `_weak_divergence_3d(st.base)` to 1e-10, and (ii) `‖q‖` is O(weak-div), not blowing up. Document and escalate to Task 4 before proceeding to Task 3.

- [ ] **Step 9: Run MMS parity to confirm no regression**

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim
python -m pytest tests/test_p2r0_parity.py -v
```
Expected: all 4 parity tests PASS (the default `pressure_update="standard"` leaves the incremental path untouched).

- [ ] **Step 10: Commit the rotational-incremental implementation**

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim
git rev-parse --abbrev-ref HEAD   # must print "p2-r2a"
git add src/diffsim/steppers/leray.py src/diffsim/steppers/leray_sbm.py \
        tests/test_p2r2a_rotational_pressure.py
git commit -m "$(cat <<'EOF'
feat(p2-r2a): rotational/consistent-incremental pressure form (Task 2)

Adds pressure_update="standard|rotational|chorin" knob to
LerayProjectionStepper (default standard = unchanged), threaded through
LeraySBMStepper. Rotational: p_hat = p* + phi - nu*q, M_p q = B^T u_hat,
reusing self.M as the consistent pressure mass and rhs_free/sigma as the
weak divergence. Chorin confirms the accumulation mechanism (bounded)
vs classic-incremental (diverges). Non-vacuity gate: standard diverges,
rotational stabilizes on the level-4 Stokes sphere. MMS parity clean.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: R2a Stability Gate (3-D Sphere Projection vs. Monolithic)

**Purpose:** The formal R2a gate: 3-D sphere projection Cd matches monolithic same-mesh reference within 20%, weak-div machine-zero, BDF2 engaged. Includes a mutation/planted-break leg that breaks the gate when the pressure form is reverted to `pressure_update="standard"` (the classic-incremental setting that diverges, from Task 2's non-vacuity test). Runs on gpubox (`DIFFSIM_NIGHTLY`).

**Files:**
- Modify: `tests/test_p2r0_projection_sbm.py` (add `test_g6_sphere_3d_r2a_stability_gate`)
- Modify: `tests/p2r0_task10_sphere_derisk.py` (add `march_projection_r2a` helper)
- Modify: `tests/baselines/p2r0_task10_sphere.json` (updated by the run)

**Interfaces:**
- Consumes: `build_sphere_3d`, `monolithic_cd` from `tests/p2r0_task10_sphere_derisk.py`; `LeraySBMStepper` with `pressure_update="rotational"` from Task 2; `_weak_divergence_3d` from `tests/p2r2a_diagnostic_3d.py`
- Produces: `test_g6_sphere_3d_r2a_stability_gate` (a `DIFFSIM_NIGHTLY`-guarded pytest gate); updated `tests/baselines/p2r0_task10_sphere.json` with `r2a_stability` key.

- [ ] **Step 1: Add `march_projection_r2a` to `tests/p2r0_task10_sphere_derisk.py`**

Read the file first (already read above), then append after the existing `monolithic_cd` function:

```python
def march_projection_r2a(fx, dt, max_steps, rate_tol, order=2,
                         beta_backflow=1.0, picard_iters=2,
                         pressure_update="rotational", alpha=100.0):
    """March LeraySBMStepper with the rotational/consistent-incremental
    pressure form (R2a fix). Returns the same dict as march_projection plus
    'pressure_update' used."""
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
        pressure_update=pressure_update)
    st.set_initial(lambda c: np.zeros((len(c), dim)))
    q = qref()
    cd_prev = None
    steps = 0
    orders = []
    for step in range(max_steps):
        ts = time.time()
        u, p = st.step()
        F = st.surrogate_traction()
        cd = float(F[0] / q)
        orders.append(int(st.base.order))
        steps = step + 1
        if step < 3 or step % 10 == 0:
            print(f"[r2a]   proj step{step+1} Cd={cd:+.4f} "
                  f"ord={st.base.order} ({time.time()-ts:.1f}s)", flush=True)
        if cd_prev is not None and step > 10 and abs(cd - cd_prev) / dt < rate_tol:
            break
        cd_prev = cd
    clat = float(np.hypot(F[1], F[2]) / q)
    finite = bool(np.isfinite(u).all() and np.isfinite(p).all())
    bdf2_engaged = 2 in orders
    return dict(cd=cd, clat=clat, steps=steps, finite=finite,
                bdf2_engaged=bdf2_engaged, st=st, u=u, p=p)
```

- [ ] **Step 2: Write the failing gate test in `tests/test_p2r0_projection_sbm.py`**

Open the existing `tests/test_p2r0_projection_sbm.py` and append at the end:

```python
@pytest.mark.tier4
def test_g6_sphere_3d_r2a_stability_gate(device):
    """R2a gate: 3-D sphere projection Cd matches monolithic same-mesh
    reference within 20%, weak-div machine-zero, BDF2 engaged.

    Gate-hygiene:
    - Independent reference: 3-D monolithic (no-split) SBM-NS on matched mesh
      (not a self-comparison, not the literature value).
    - Mutation/planted-break: pressure_update="standard" (classic-incremental)
      makes the gate FAIL (Cd < -0.5, diverging), proving the gate is
      load-bearing — the rotational form is what stabilizes it.

    Runs on gpubox (DIFFSIM_NIGHTLY, level-4, host/splu).
    """
    import json
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from p2r0_task10_sphere_derisk import (build_sphere_3d, monolithic_cd,
                                           march_projection_r2a, R, U_IN)
    from p2r2a_diagnostic_3d import _weak_divergence_3d

    if not os.environ.get("DIFFSIM_NIGHTLY"):
        pytest.skip("nightly / gpubox only — R2a 3-D stability gate")

    level = 4
    Re = 100.0
    dt = 0.05
    max_steps = 120
    rate_tol = 5e-3

    fx = build_sphere_3d(device, level=level, Re=Re)
    print(f"\n[r2a-gate] level={level} Re={Re} dt={dt} "
          f"n_free={len(fx['coords'])} sf={fx['sf'].elem.size}", flush=True)

    # --- Main path: rotational / consistent-incremental pressure form ---
    pr = march_projection_r2a(fx, dt=dt, max_steps=max_steps,
                              rate_tol=rate_tol, order=2,
                              pressure_update="rotational", alpha=100.0)
    q = 0.5 * U_IN ** 2 * float(__import__('numpy').pi) * R ** 2
    print(f"[r2a-gate] PROJECTION: Cd={pr['cd']:+.4f}  Clat={pr['clat']:.4f}  "
          f"finite={pr['finite']}  bdf2={pr['bdf2_engaged']}  "
          f"steps={pr['steps']}", flush=True)

    # Weak divergence on the final state
    w2, winf = _weak_divergence_3d(pr["st"])
    print(f"[r2a-gate] weak-div: ||B^T u||_2={w2:.3e}  ||B^T u||_inf={winf:.3e}",
          flush=True)

    # Independent monolithic reference on the SAME mesh (matched penalty alpha)
    mono = monolithic_cd(fx, alpha=100.0, dt=dt, max_steps=max_steps,
                         rate_tol=rate_tol)
    rel = abs(pr["cd"] - mono["cd"]) / (abs(mono["cd"]) + 1e-300)
    print(f"[r2a-gate] MONOLITHIC: Cd={mono['cd']:+.4f}  rel_diff={rel:.3%}",
          flush=True)

    # --- GATE ASSERTIONS ---
    assert pr["finite"], "projection field is not finite"
    assert pr["bdf2_engaged"], "BDF2 never engaged (check order / bootstrap)"
    assert w2 < 1e-8, f"weak-div not machine-zero: ||B^T u||_2={w2:.3e}"
    assert pr["cd"] > 0, f"projection Cd is negative: {pr['cd']:.4f}"
    assert rel < 0.20, (
        f"projection Cd {pr['cd']:.4f} vs monolithic {mono['cd']:.4f}: "
        f"rel_diff={rel:.1%} exceeds 20% faithfulness gate")

    # --- MUTATION / PLANTED-BREAK LEG ---
    # pressure_update="standard" (classic-incremental) must diverge to prove the
    # gate is load-bearing — the rotational form is what stabilizes it.
    fx2 = build_sphere_3d(device, level=level, Re=Re)  # fresh fixture, no shared state
    from diffsim.steppers.leray_sbm import LeraySBMStepper
    import numpy as np

    def f_fn2(x, t):
        return np.zeros((len(x), fx2["dim"]))

    st_break = LeraySBMStepper(
        fx2["oracle"], fx2["dm"], fx2["nu"], dt, f_fn2,
        u_inf=fx2["u_inf"], strong_mask=fx2["strong_mask"],
        lam=0.5, domain="outside", order=2, picard_iters=2,
        solver="splu", ppe_finescale=False, alpha=100.0,
        beta_backflow=1.0, velocity_update="consistent",
        pressure_update="standard")   # classic-incremental (the planted break)
    st_break.set_initial(lambda c: np.zeros((len(c), fx2["dim"])))
    cds_break = []
    for _ in range(15):
        u_b, p_b = st_break.step()
        F_b = st_break.surrogate_traction()
        cds_break.append(float(F_b[0] / q))
    cd_break = cds_break[-1]
    print(f"[r2a-gate] MUTATION (pressure_update=standard): Cd={cd_break:+.4f}  "
          f"trajectory={[f'{c:.2f}' for c in cds_break[:5]]}", flush=True)
    assert cd_break < -0.5, (
        f"Mutation leg expected classic-incremental to diverge (Cd < -0.5), "
        f"got {cd_break:.3f} — gate is vacuous, investigate")

    # --- Update baseline ---
    import time as _t
    baseline_path = os.path.join(os.path.dirname(__file__), "baselines",
                                 "p2r0_task10_sphere.json")
    try:
        with open(baseline_path) as fh:
            bl = json.load(fh)
    except FileNotFoundError:
        bl = {}
    bl["r2a_stability"] = dict(
        level=level, Re=Re, dt=dt, n_free=int(len(fx["coords"])),
        proj_cd=pr["cd"], proj_clat=pr["clat"], proj_steps=pr["steps"],
        mono_cd=mono["cd"], rel_diff=rel,
        weak_div_l2=w2, weak_div_inf=winf,
        bdf2_engaged=pr["bdf2_engaged"],
        pressure_update="rotational", alpha=100.0,
        mutation_cd=cd_break, mutation_pressure_update="standard",
        timestamp=_t.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    with open(baseline_path, "w") as fh:
        json.dump(bl, fh, indent=2)
    print(f"[r2a-gate] baseline written: {baseline_path}", flush=True)
```

Note: the mutation leg builds a FRESH fixture via a second `build_sphere_3d(device, level=level, Re=Re)` call (the fixture is re-entrant; no shared stepper state between the main path and the planted break).

- [ ] **Step 3: Run just the non-nightly smoke parts locally**

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim
python -m pytest tests/test_p2r0_projection_sbm.py -k "not tier4 and not nightly" -v
```
Expected: existing tier2/tier3 tests pass; tier4 `test_g6_*` is skipped (DIFFSIM_NIGHTLY not set).

- [ ] **Step 4: Run the full R2a gate on gpubox**

On gpubox:
```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim
DIFFSIM_NIGHTLY=1 python -m pytest \
    tests/test_p2r0_projection_sbm.py::test_g6_sphere_3d_r2a_stability_gate \
    -v -s 2>&1 | tee /tmp/r2a_gate.log
cat tests/baselines/p2r0_task10_sphere.json
```
Expected: PASS. Projection Cd is positive and within 20% of monolithic; weak-div < 1e-8; BDF2 engaged; mutation (pressure_update="standard") Cd < -0.5.

If the gate FAILS (Cd < 0 or rel > 20%): first verify the rotational term is being applied (`bt_uhat = rhs_free/sigma` matching an independent `_weak_divergence_3d`, `‖q‖` bounded); check that `alpha` matches between projection and monolithic. If the rotational form still does not stabilize, go to Task 4 (fallback decision). Document the trajectory and the `‖q‖` / weak-div checks before escalating.

- [ ] **Step 5: Commit the R2a stability gate**

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim
git rev-parse --abbrev-ref HEAD   # must print "p2-r2a"
git add tests/test_p2r0_projection_sbm.py tests/p2r0_task10_sphere_derisk.py \
        tests/baselines/p2r0_task10_sphere.json
git commit -m "$(cat <<'EOF'
feat(p2-r2a): R2a stability gate — 3-D sphere projection vs monolithic (Task 3)

Gate: projection Cd matches monolithic same-mesh within 20%, weak-div
machine-zero, BDF2 engaged, pressure_update=rotational. Mutation leg
(pressure_update=standard, classic-incremental) breaks the gate, proving
load-bearing. Baseline updated.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Fallback-Decision Document (NEEDS_CONTEXT Escalation)

**When to execute this task:**
- Execute ONLY if the rotational-incremental fix does not stabilize the 3-D projection (Cd remains negative or rel > 20%), OR if the Chorin confirmation (Task 2 Step 2) shows the non-incremental march ALSO diverges (which would contradict the diagnostic's accumulation reading and re-open the mechanism question).
- If Tasks 2–3 all pass, SKIP this task. The R2a gate has been satisfied and no fallback document is needed.

**What this task does:** Documents the monolithic-block-preconditioner path as the stable 3-D solver for R2 if the projection cannot be stabilized. This is a NEEDS_CONTEXT escalation (not a forced fix), providing the supervisor with a clear re-scoping option.

**Files:**
- Create: `docs/dev/2026-07-21-p2-r2a-fallback-decision.md`

**Interfaces:**
- Consumes: Task 1 diagnostic verdict (`tests/baselines/p2r2a_diagnostic_3d.json`); Task 3 gate attempt results; `src/diffsim/solvers/block_precond.py` (the AMGX-backed monolithic path already implemented)
- Produces: A decision document for the supervisor, with a concrete re-scoping proposal.

- [ ] **Step 1: Write the fallback-decision document**

```markdown
# P2-R2a Fallback Decision — Monolithic Block-Preconditioner Path

**Date:** 2026-07-21  
**Status:** NEEDS_CONTEXT escalation — supervisor decision required.

## What was tried
[Fill in: Task 1 mechanism verdict (incremental-pressure feedback, `4194601`), the rotational-fix result (Chorin confirmation Cd, rotational vs standard Cd trajectory, weak-div and ‖q‖ checks), final Cd]

## The stable 3-D solver: monolithic block preconditioner
`src/diffsim/solvers/block_precond.py::BlockAMGPreconditioner` + `solve_block_preconditioned`
is already implemented (AMGX-backed; v1 status: "not yet effective" per findings 8f, but the
harness is in place). The 3-D monolithic (no-split) SBM-NS solver is already demonstrated stable
at Cd=0.381 (Task-10 report) on the same mesh.

## Re-scoping proposal (for supervisor)
If the projection instability is fundamental in 3-D (not fixable by the rotational-incremental
pressure form or PPE tuning), R2 can validate via the monolithic path:
- R2b device port targets the monolithic block preconditioner (not the PPE) as the scalable
  3-D solver. BlockAMGPreconditioner wraps AMGX already; the main R2b task becomes wiring the
  outer Krylov onto the device.
- R2c literature validation uses the monolithic stepper (already passing Cd=0.381).
- The projection limitation is noted in the R2c report as a documented finding.

## Decision required
- [ ] Accept re-scoping: proceed with monolithic as the R2 3-D solver.
- [ ] Continue projection investigation: the mechanism is [X]; the next diagnostic step is [Y].
```

- [ ] **Step 2: Commit the fallback document**

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim
git rev-parse --abbrev-ref HEAD   # must print "p2-r2a"
git add docs/dev/2026-07-21-p2-r2a-fallback-decision.md
git commit -m "$(cat <<'EOF'
docs(p2-r2a): fallback-decision document for monolithic path (Task 4)

NEEDS_CONTEXT escalation: if the 3-D projection cannot be stabilized
by the rotational-incremental pressure form, documents the monolithic
block-preconditioner (block_precond.py, AMGX-backed) as the R2 3-D
solver re-scoping option.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

### 1. Spec §3-R2a Coverage

| Spec requirement | Task covering it |
|---|---|
| Diagnostic: isolate why 3-D pressure coupling diverges at Stokes | Task 1 (DONE, `4194601`) |
| Diagnostic produces a decisive verdict (three-number style, committed) | Task 1 — four mechanisms ruled out, incremental-feedback signature |
| Candidate mechanisms tested: PPE conditioning, surrogate-consistent BC 3-D consistency, pressure null-space / outflow, penalty | Task 1 — M1, M2, M3, M4 all NOT |
| Fix implementation (knob + rotational/consistent-incremental pressure form) | Task 2 — `pressure_update="standard\|rotational\|chorin"` |
| Mechanism confirmation (Chorin non-incremental stays bounded) | Task 2 Step 1–2 |
| Non-vacuity gate (mutation/planted-break: classic-incremental diverges) | Task 2 Step 7–8 + Task 3 mutation leg |
| Gate: 3-D sphere projection Cd matches monolithic same-mesh | Task 3 |
| Gate: weak-div machine-zero | Task 3 (asserts `w2 < 1e-8`) |
| Gate: BDF2 engaged | Task 3 (asserts `bdf2_engaged`) |
| Gate-hygiene: independent reference + mutation | Task 3 (monolithic reference + `pressure_update="standard"` break) |
| Fallback documented: monolithic block-preconditioner path | Task 4 |
| Prerequisite: `ppe_finescale=True` τ_m bug must be fixed before enabling it | Global Constraints (rotational guards against `ppe_finescale=True`) |
| `ppe_finescale=False` default unchanged | All tasks: `ppe_finescale=False` throughout; rotational raises if `True` |
| Compute on gpubox for 3-D runs | Task 2 (Steps 2, 8), Task 3 (Step 4) — all marked gpubox; unit tests local |
| REPO-IDENTITY GUARD | Each commit step includes `git rev-parse --abbrev-ref HEAD` check |
| Agents never push | Global Constraints |
| R2b and R2c out of scope | Global Constraints (explicit scope fence) |

### 2. Placeholder Scan

No "TBD", "TODO", or "implement later" as unresolved code placeholders. Intentional fixture-wiring placeholders remain in Task 2 Step 1 (`test_pressure_update_knob_validates` stub) and Step 3 (`_small_2d_stepper` + the two skip-guarded rotational unit tests) — these are explicitly flagged with NOTE blocks instructing the implementing agent to wire them to the repo's real 2-D `dm` fixture (grep `tests/test_p2r0_projection_sbm.py`); the load-bearing assertions (knob `ValueError`, standard-parity bit-for-bit, divergence-free `‖q‖<1e-10`) are fully specified. One intentional data-capture placeholder in the fallback document (Task 4 Step 1) `[Fill in: ...]`. No `import rebuild` stub remains (Task 3 mutation leg uses a direct second `build_sphere_3d`).

### 3. Type Consistency

- `pressure_update` (str, one of `"standard"|"rotational"|"chorin"`, default `"standard"`) — added to `LerayProjectionStepper.__init__` (Task 2 Step 1) and threaded through `LeraySBMStepper.__init__` (Task 2 Step 6). Same literal values used in Task 2 tests, Task 3 `march_projection_r2a`, and Task 3 mutation leg.
- `LeraySBMStepper(..., pressure_update="standard")` — added in Task 2 Step 6; called with `pressure_update=` in Task 2 Steps 1/7, Task 3 Step 1 (via `march_projection_r2a`) and Task 3 mutation leg.
- `march_projection_r2a(fx, dt, max_steps, rate_tol, order=2, beta_backflow=1.0, picard_iters=2, pressure_update="rotational", alpha=100.0)` — defined in Task 3 Step 1; called in Task 3 Step 2 main path with `pressure_update="rotational", alpha=100.0`. Returns `dict(cd, clat, steps, finite, bdf2_engaged, st, u, p)`.
- `_weak_divergence_3d(st: LeraySBMStepper) -> (float, float)` — defined in Task 1 (`4194601`); imported and called in Task 3 Step 2 on `pr["st"]`.
- `monolithic_cd(fx, alpha, dt, max_steps, rate_tol)` — existing function in `tests/p2r0_task10_sphere_derisk.py`, called with `alpha=100.0` in Task 3 Step 2 (matched to the projection penalty for a fair Cd comparison).
- Rotational internals (Task 2 Step 4): `bt_uhat = rhs_free / sigma` (both free-node-sized, in scope at the pressure-update site); `q = solve_linear(self.M, bt_uhat, ..., cache_key="mass")` reuses the velocity-update mass factorization; `p_hat = self.p_star + phi - self.nu * q`.

All type and name references are consistent across Tasks 2–3.

## Supervisor resolutions (2026-07-21)

1. **Fix form — DECIDED (Baskar, 2026-07-21).** Task 1's committed diagnostic
   (`4194601`) ruled out all four candidate mechanisms (M1–M4) with machine-
   precision negatives and isolated an incremental-pressure feedback
   instability (`p* += φ` accumulation, stable in 2-D, unstable in 3-D). The
   R2a fix is the **rotational / consistent-incremental (Timmermans) pressure
   form** `p_hat = p* + φ − ν(∇·û)`, NOT the penalty law (which the diagnostic
   overturned — penalty diverged through α=20000). The SPD-PPE projection path
   is retained (the scalability reason it was chosen; do NOT switch to
   monolithic unless Task 4 escalation triggers).
2. **Chorin mechanism confirmation — REQUIRED.** Task 2 folds in a non-
   incremental (`pressure_update="chorin"`, `p* ≡ 0` each step) march on the
   same 3-D sphere: if Chorin stays bounded while classic-incremental diverges,
   the accumulation is confirmed as the cause. Cleanest realization chosen:
   a `"chorin"` mode that zeros `self.p_star` at the top of `step()` (so
   `p_hat = φ`, no accumulation) — no separate toggle needed.
3. **Pressure mass matrix M_p — DECIDED: reuse `self.M` (consistent), no new
   assembly.** `self.M` (the scalar consistent mass, `T.T @ M @ T`) already
   lives in the exact free-node pressure space as `phi`/`p_star`, is SPD, and
   is already factorized/cached via `solve_linear`. Solve `M_p q = Bᵀû`
   against it with the shared `cache_key="mass"` (no second factorization; no
   pin — a mass matrix is invertible). Consistent (not lumped) mass gives the
   correct L2 projection of `∇·û` the Timmermans form specifies. Lumped `M_p`
   revisited only if a future device port makes the consistent solve a
   bottleneck (flagged for supervisor).
4. **`Bᵀû` source — VERIFIED against `leray.py`.** In the `ppe_finescale=False`
   branch the PPE flux is `sigma * aqv` (line ~421), so `rhs_free = sigma·Bᵀû`
   and `bt_uhat = rhs_free / sigma` is the nodal weak divergence of the
   predictor (no re-assembly). The rotational branch raises `NotImplementedError`
   under `ppe_finescale=True` (where the flux carries the tau_m fine-scale
   residual and this identity does NOT hold); R2a uses `ppe_finescale=False`
   throughout, so this is a guard, not a limitation.
