# Leak-Drag Discriminator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attribute the +50% Cd excess in the corrected Re=250 thin-plate runs to (a) real flow drag, (b) `surrogate_traction` overestimation, and/or (c) α-penalty leakage — via a time-averaged control-volume momentum balance crossed with an α-sweep.

**Architecture:** Three units per the approved spec (`docs/superpowers/specs/2026-07-26-leakdrag-discriminator-design.md`): a pure-function CV-drag observable (`postproc/cv_drag.py`, nodal-line sampling + trapezoid), a default-None `on_step` callback hook in `run_flow_past` (bit-for-bit when unused), and a GPU probe script that marches the corrected Re=250 config at α∈{20,50,100}, accumulating Cd_CV (3 nested boxes) + Cd_surrogate + leak flux each step, printing the 2×2 verdict table.

**Tech Stack:** numpy (CV integrals), the existing `run_flow_past` monolithic march (cudss + device assembly on GPU), `face_tables`/`GeometryData` for the leak flux, gpubox remote toolkit.

## Global Constraints

- Branch `leakdrag-discriminator` off `master` at `0d839f3`; repo `/Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim`; every bash command starts `cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && `.
- Python `.venv/bin/python` locally; box runs via `bash scripts/remote/gpubox-run.sh "<cmd>" <tag>` (injects the WSL CUDA lib path), poll log FILES only, never edit .py mid-run.
- `on_step=None` default MUST leave `run_flow_past` bit-for-bit (existing parity tests prove it; add the explicit regression).
- ALL quantities in octree units (the unit-system rule, `docs/dev/thinshell-gpu-runbook.md`): L=1/16, ν=U·L/Re=2.5e-4, x_c=5/16, dt=5e-4, t_start=1.5, St=f·L/U.
- Spec thresholds verbatim: nested-box spread >5% of Cd_CV ⇒ flag CV-UNRELIABLE and withhold the verdict; α-insensitive ≤5% across sweep; CV≈surr ≤10%.
- Existing gates stay green: tests/test_p2r1a_thin_plate_flow.py, tests/test_p2r1a_thin_plate_flow_projection.py.
- `git add` explicit paths only, NEVER `-A`/`.`. Commits end with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: `cv_drag` observable + analytic unit tests

**Files:**
- Create: `src/diffsim/postproc/cv_drag.py`
- Test: `tests/test_cv_drag.py`

**Interfaces:**
- Consumes: nothing from this plan (pure numpy).
- Produces: `cv_drag_box(coords, u, p, box, nu, U_inf=1.0, L_ref=1.0/16.0) -> float` — instantaneous Cd of the box balance; `box=(x0,x1,y0,y1)`. Sign: positive = downstream (+x) force ON the enclosed body, i.e. `Cd = -∮[u_x(u·n) + p n_x − ν ∂u_x/∂n] dS / (0.5 U_inf² L_ref)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cv_drag.py
"""Analytic gates for the control-volume drag observable (spec 2026-07-26)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.postproc.cv_drag import cv_drag_box

BOX = (0.25, 0.75, 0.25, 0.75)

def _mesh_coords(level=5):
    mesh = build_mesh(build_uniform(level, dim=2), p=1)
    return np.asarray(mesh.node_coords)

def test_uniform_freestream_zero_drag():
    """u=(U,0), p=0: every flux contribution cancels; Cd must be ~0."""
    coords = _mesh_coords()
    u = np.zeros((len(coords), 2)); u[:, 0] = 1.0
    p = np.zeros(len(coords))
    cd = cv_drag_box(coords, u, p, BOX, nu=2.5e-4)
    assert abs(cd) < 1e-10, f"uniform flow gave Cd={cd}"

def test_analytic_momentum_flux():
    """u=(x,-y) (div-free), p=0, nu=0 on box (x0,x1,y0,y1):
      ∮u_x(u·n)dS = x1²Δy − x0²Δy − Δ(x²)/2·y1 + Δ(x²)/2·y0
    with Δy=y1−y0, Δ(x²)=x1²−x0². Cd = −that / (0.5·U²·L_ref)."""
    coords = _mesh_coords()
    u = np.stack([coords[:, 0], -coords[:, 1]], axis=1)
    p = np.zeros(len(coords))
    x0, x1, y0, y1 = BOX
    dy, dx2 = (y1 - y0), (x1**2 - x0**2)
    flux = x1**2 * dy - x0**2 * dy - 0.5 * dx2 * y1 + 0.5 * dx2 * y0
    expect = -flux / (0.5 * 1.0**2 * (1.0 / 16.0))
    cd = cv_drag_box(coords, u, p, BOX, nu=0.0)
    assert np.isclose(cd, expect, rtol=1e-6), f"{cd} vs {expect}"

def test_pressure_term():
    """u=0, p=x, nu=0: ∮p·n_x dS = (x1−x0)·... → p(x1)Δy − p(x0)Δy = Δx·Δy? No:
    right face contributes +p(x1)Δy = x1Δy, left −x0Δy → net (x1−x0)Δy."""
    coords = _mesh_coords()
    u = np.zeros((len(coords), 2))
    p = coords[:, 0].copy()
    x0, x1, y0, y1 = BOX
    expect = -((x1 - x0) * (y1 - y0)) / (0.5 * (1.0 / 16.0))
    cd = cv_drag_box(coords, u, p, BOX, nu=0.0)
    assert np.isclose(cd, expect, rtol=1e-6), f"{cd} vs {expect}"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && .venv/bin/python -m pytest tests/test_cv_drag.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'diffsim.postproc.cv_drag'`

- [ ] **Step 3: Implement `cv_drag_box`**

```python
# src/diffsim/postproc/cv_drag.py
"""Control-volume drag observable (2-D) — spec 2026-07-26-leakdrag-discriminator.

Instantaneous box momentum balance from NODAL fields, SBM-independent:
    Cd = -∮ [ u_x (u·n) + p n_x - nu du_x/dn ] dS / (0.5 U² L_ref)
Faces are axis-aligned lines sampled at nodal locations (trapezoid at the
local node pitch); viscous normal derivative by one-sided difference to the
parallel node line one local pitch inside the box. rho = 1 (code units).
"""
import numpy as np

_TOLF = 0.25  # face-matching tolerance as a fraction of the local pitch


def _line_nodes(coords, axis, value, lo, hi, tol):
    """Sorted node indices on the line coords[axis]==value within [lo,hi]."""
    on = np.abs(coords[:, axis] - value) < tol
    span = (coords[:, 1 - axis] >= lo - tol) & (coords[:, 1 - axis] <= hi + tol)
    idx = np.where(on & span)[0]
    order = np.argsort(coords[idx, 1 - axis])
    return idx[order]


def _face_integral(coords, u, p, nu, axis, value, lo, hi, n_sign):
    """∫ [u_x(u·n) + p n_x - nu du_x/dn] ds over one axis-aligned face.

    axis=0: vertical face x=value, n=(n_sign,0), integrate over y in [lo,hi].
    axis=1: horizontal face y=value, n=(0,n_sign), integrate over x.
    """
    # local pitch from the nearest node spacing on the face
    probe = _line_nodes(coords, axis, value, lo, hi, tol=1e-6)
    if len(probe) < 2:
        raise ValueError(f"no node line at axis{axis}={value} — box edges must "
                         "lie on mesh node lines")
    s = coords[probe, 1 - axis]
    h = np.min(np.diff(s))
    tol = _TOLF * h
    idx = _line_nodes(coords, axis, value, lo, hi, tol)
    s = coords[idx, 1 - axis]
    ux = u[idx, 0]
    un = u[idx, axis] * n_sign          # u·n on this face
    pn = p[idx] * (n_sign if axis == 0 else 0.0)   # p n_x
    # du_x/dn: one-sided toward the inside line at value - n_sign*h
    inner = _line_nodes(coords, axis, value - n_sign * h, lo, hi, tol)
    ux_in = np.interp(s, coords[inner, 1 - axis], u[inner, 0])
    duxdn = (ux - ux_in) / h
    integrand = ux * un + pn - nu * duxdn
    return np.trapz(integrand, s)


def cv_drag_box(coords, u, p, box, nu, U_inf=1.0, L_ref=1.0 / 16.0):
    """Instantaneous CV drag coefficient for rectangular `box`=(x0,x1,y0,y1)."""
    x0, x1, y0, y1 = box
    total = 0.0
    total += _face_integral(coords, u, p, nu, 0, x1, y0, y1, +1)  # right
    total += _face_integral(coords, u, p, nu, 0, x0, y0, y1, -1)  # left
    total += _face_integral(coords, u, p, nu, 1, y1, x0, x1, +1)  # top
    total += _face_integral(coords, u, p, nu, 1, y0, x0, x1, -1)  # bottom
    return -total / (0.5 * U_inf ** 2 * L_ref)
```

NOTE for the implementer: `np.trapz` is deprecated in new numpy — if the venv's numpy warns, use `np.trapezoid` (same signature); test output must be pristine.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && .venv/bin/python -m pytest tests/test_cv_drag.py -q`
Expected: 3 passed, no warnings.

- [ ] **Step 5: Commit**

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && git add src/diffsim/postproc/cv_drag.py tests/test_cv_drag.py && git commit -m "$(cat <<'EOF'
feat(postproc): control-volume drag observable + analytic gates

cv_drag_box: SBM-independent 2-D box momentum balance from nodal fields
(nodal-line trapezoid, one-sided viscous normal derivative). Gated by
uniform-freestream zero, analytic momentum-flux, and pressure-term tests.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `on_step` hook in `run_flow_past`

**Files:**
- Read: `tests/p2r1a_thin_plate_flow.py` — the march loop (find where `x_cur` yields `u_new` and where the `_return_fields=True` path assembles FULL nodal fields for VTU export; reuse that exact expansion), and the per-step Cd/Cl computation site
- Modify: `tests/p2r1a_thin_plate_flow.py`
- Test: `tests/test_p2r1a_thin_plate_flow.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `run_flow_past(..., on_step=None)`; when set, called AFTER each step's solve as `on_step(step, t_new, u_full, p_full, cd_step)` with `u_full` [n_nodes,2], `p_full` [n_nodes] (constraint-expanded full-mesh nodal fields, same construction as the `_return_fields` export path) and `cd_step` = that step's surrogate-traction Cd (float). Default None ⇒ bit-for-bit.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_p2r1a_thin_plate_flow.py`:

```python
def test_on_step_default_parity():
    """on_step=None must be bit-for-bit identical to the legacy march."""
    from p2r1a_thin_plate_flow import run_flow_past
    kw = dict(level=4, nsteps=3, dt=0.01, nu=0.1, U_inf=1.0, verbose=False)
    res_a = run_flow_past(**kw)
    res_b = run_flow_past(on_step=None, **kw)
    assert np.allclose(res_a["cd"], res_b["cd"], rtol=0, atol=1e-14)

def test_on_step_callback_contract():
    """Callback fires once per step with full-mesh fields + that step's Cd."""
    from p2r1a_thin_plate_flow import run_flow_past
    calls = []
    def spy(step, t, u_full, p_full, cd_step):
        calls.append((step, t, u_full.shape, p_full.shape, cd_step))
    res = run_flow_past(level=4, nsteps=3, dt=0.01, nu=0.1, U_inf=1.0,
                        verbose=False, on_step=spy)
    assert len(calls) == 3
    n = calls[0][2][0]
    assert calls[0][2] == (n, 2) and calls[0][3] == (n,)
    # cd passed to the callback matches the returned history per step
    for k, (step, t, _, _, cd_step) in enumerate(calls):
        assert step == k and np.isclose(cd_step, res["cd"][k], rtol=0, atol=1e-14)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && .venv/bin/python -m pytest tests/test_p2r1a_thin_plate_flow.py::test_on_step_default_parity tests/test_p2r1a_thin_plate_flow.py::test_on_step_callback_contract -q`
Expected: FAIL — unexpected keyword argument `on_step`.

- [ ] **Step 3: Implement**

Add `on_step=None` to `run_flow_past`'s signature (docstring: one line, the Produces contract above). In the march loop, AFTER the step's Cd/Cl are computed (read the loop to place it after the traction evaluation), add:

```python
        if on_step is not None:
            # Full-mesh nodal fields, same construction as _return_fields:
            # x_cur is [nfree*ndof] free-dof interleaved (u_x,u_y,p);
            # expand through the constraint prolongation exactly as the
            # VTU-export path does — REUSE that code path's arrays/vars,
            # do not re-derive (read the _return_fields block and mirror it).
            on_step(step, t_new, u_full_cb, p_full_cb, float(cd_hist[step]))
```

VERIFY-FIRST: the exact names (`x_cur`, `cd_hist`, the expansion) must be read from the driver — if `_return_fields` builds fields only ONCE at the end, hoist that expansion into a small local helper `_full_fields(x_cur)` used by both sites (end-of-run and per-step callback) so there is ONE construction, not two divergent copies.

- [ ] **Step 4: Run the new tests + the file's full gate**

Run: `cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && .venv/bin/python -m pytest tests/test_p2r1a_thin_plate_flow.py -q`
Expected: all pass (previous count + 2).

- [ ] **Step 5: Commit**

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && git add tests/p2r1a_thin_plate_flow.py tests/test_p2r1a_thin_plate_flow.py && git commit -m "$(cat <<'EOF'
feat(p2r1a): on_step callback hook in run_flow_past (default None, bit-for-bit)

Per-step (step, t, u_full, p_full, cd_step) callback with full-mesh nodal
fields via the same expansion as the _return_fields path. Enables inline
observables (CV drag, leak flux) without touching the march.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Probe script (CV boxes + leak flux + α-sweep) with CPU mini-gate

**Files:**
- Read: `tests/p2r1a_thin_plate_flow.py` `_build_shell` (returns the two-sided surrogate `(sfp, gp), (sfm, gm)` — confirm the dict keys) and `src/diffsim/steppers/leray_sbm.py` `LeraySBMShellStepper.surrogate_normal_flux` (the loop to mirror)
- Create: `tests/gpu_leakdrag_discriminator.py`
- Test: `tests/test_leakdrag_probe_cpu.py`

**Interfaces:**
- Consumes: `cv_drag_box` (Task 1), `on_step` (Task 2), `run_flow_past`, `face_tables`, the driver's shell-build machinery.
- Produces: `run_discriminator(alpha, nsteps, level=7, refine_to=9, wake_refine=9, dt=5e-4, device="cuda:0", mono_solver="cudss", assembly="device", t_start_lu=24.0) -> dict` with keys `cd_surr_mean, cd_cv_mean (dict box->float), box_spread, leak_mean_abs, St, n_steps_avg`; `__main__` runs α∈{20,50,100} and prints the verdict table + `LEAKDRAG-OK`.

- [ ] **Step 1: Write the probe**

```python
# tests/gpu_leakdrag_discriminator.py
"""Leak-drag discriminator (spec 2026-07-26): CV drag vs surrogate traction
x alpha-sweep on the corrected Re=250 config.  GPU box:

    bash scripts/remote/gpubox-run.sh \
        ".venv/bin/python tests/gpu_leakdrag_discriminator.py" leakdrag

Env: ALPHAS ("20,50,100"), NSTEPS (8000), DEVICE (cuda:0), MONO_SOLVER
(cudss), ASSEMBLY (device).  CPU mini-gate uses run_discriminator directly.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from diffsim.postproc.cv_drag import cv_drag_box
from diffsim.postproc.shedding import strouhal
from diffsim.mesh.faces import face_tables

L = 1.0 / 16.0
U = 1.0
RE = 250.0
NU = U * L / RE
X_C = 5.0 / 16.0
Y_C = 0.5

# Nested CV boxes: margins ~4L, 6L, 8L, edges snapped to L7 node lines (h=1/128)
def _snap(v):
    return round(v * 128.0) / 128.0

def _boxes():
    out = {}
    for tag, m in (("4L", 4 * L), ("6L", 6 * L), ("8L", 8 * L)):
        out[tag] = (_snap(X_C - m), _snap(X_C + m),
                    _snap(Y_C - m), _snap(Y_C + m))
    return out


def _leak_flux(mesh, sfp, gp, sfm, gm, u_full, dim=2):
    """Net ∮ u·n over BOTH shell sides (mirrors surrogate_normal_flux)."""
    ftab = face_tables(1, dim)
    nqf = ftab.nqf
    net = 0.0
    for sf, geo in ((sfp, gp), (sfm, gm)):
        conn = mesh.conn_of[1][np.searchsorted(mesh.bins[1], sf.elem)]
        h = mesh.tree.h()[sf.elem]
        jacS = (h / 2.0) ** (dim - 1)
        for fi in range(len(sf.elem)):
            f = int(sf.face[fi])
            un = u_full[conn[fi]]
            for q in range(nqf):
                w = ftab.w[q] * jacS[fi] * geo.corr[fi * nqf + q]
                n = geo.n[fi * nqf + q]
                net += w * ((ftab.N[f][q] @ un) @ n)
    return net


def run_discriminator(alpha, nsteps, level=7, refine_to=9, wake_refine=9,
                      dt=5e-4, device="cuda:0", mono_solver="cudss",
                      assembly="device", t_start_lu=24.0):
    from p2r1a_thin_plate_flow import run_flow_past, _build_shell
    t_start = t_start_lu * L / U
    boxes = _boxes()
    acc = {k: 0.0 for k in boxes}
    acc_surr = 0.0
    acc_leak = 0.0
    n_avg = 0
    coords_ref = {}
    shell_ref = {}

    def cb(step, t, u_full, p_full, cd_step):
        nonlocal acc_surr, acc_leak, n_avg
        if t < t_start:
            return
        coords = coords_ref["coords"]
        for k, b in boxes.items():
            acc[k] += cv_drag_box(coords, u_full, p_full, b, NU,
                                  U_inf=U, L_ref=L)
        acc_surr += cd_step
        acc_leak += abs(_leak_flux(shell_ref["mesh"], *shell_ref["sg"], u_full))
        n_avg += 1

    # Build once OUTSIDE the run to grab mesh/shell refs for the callback:
    # VERIFY-FIRST — read _build_shell's return dict keys and run_flow_past's
    # internals: if run_flow_past rebuilds the shell itself, add a
    # `_shell_out=None` optional dict arg it fills, or read mesh/sfp/... from
    # the _return_fields result. Choose the least-invasive option the driver
    # actually supports and document which you used.
    fx = _build_shell(level, X_C, Y_C, L, refine_to=refine_to,
                      wake_refine=wake_refine)
    coords_ref["coords"] = np.asarray(fx["mesh"].node_coords)
    shell_ref["mesh"] = fx["mesh"]
    shell_ref["sg"] = (fx["sfp"], fx["gp"], fx["sfm"], fx["gm"])

    t0 = time.time()
    r = run_flow_past(level=level, refine_to=refine_to, wake_refine=wake_refine,
                      nsteps=nsteps, dt=dt, nu=NU, U_inf=U,
                      plate_xc=X_C, plate_yc=Y_C, plate_L=L,
                      pert_eps=0.03, pert_t_end=0.5, alpha=alpha,
                      mono_solver=mono_solver, assembly=assembly,
                      device=device, verbose=False, on_step=cb)
    el = time.time() - t0
    t = np.arange(1, nsteps + 1) * dt
    St, _f = strouhal(t, np.asarray(r["cl"]), U, L)
    cv_means = {k: v / max(n_avg, 1) for k, v in acc.items()}
    vals = np.array(list(cv_means.values()))
    out = dict(alpha=alpha,
               cd_surr_mean=acc_surr / max(n_avg, 1),
               cd_cv_mean=cv_means,
               box_spread=float(vals.max() - vals.min()) / max(abs(vals.mean()), 1e-9),
               leak_mean_abs=acc_leak / max(n_avg, 1),
               St=float(St), n_steps_avg=n_avg, elapsed=el)
    np.savez(f"results/leakdrag_a{int(alpha)}_hist.npz",
             t=t, cd=r["cd"], cl=r["cl"], **{f"cv_{k}": v for k, v in cv_means.items()})
    return out


if __name__ == "__main__":
    alphas = [float(a) for a in os.environ.get("ALPHAS", "20,50,100").split(",")]
    nsteps = int(os.environ.get("NSTEPS", "8000"))
    os.makedirs("results", exist_ok=True)
    rows = [run_discriminator(a, nsteps,
                              device=os.environ.get("DEVICE", "cuda:0"),
                              mono_solver=os.environ.get("MONO_SOLVER", "cudss"),
                              assembly=os.environ.get("ASSEMBLY", "device"))
            for a in alphas]
    print(f"\n{'alpha':>6} {'Cd_surr':>8} {'CV(4L)':>8} {'CV(6L)':>8} "
          f"{'CV(8L)':>8} {'spread':>7} {'|leak|':>9} {'St':>7}")
    for r in rows:
        c = r["cd_cv_mean"]
        print(f"{r['alpha']:6.0f} {r['cd_surr_mean']:8.3f} {c['4L']:8.3f} "
              f"{c['6L']:8.3f} {c['8L']:8.3f} {r['box_spread']:7.3f} "
              f"{r['leak_mean_abs']:9.2e} {r['St']:7.4f}")
    # Verdict per spec thresholds
    mid = rows[len(rows) // 2]
    cv_mid = np.mean(list(mid["cd_cv_mean"].values()))
    if mid["box_spread"] > 0.05:
        print("VERDICT: CV-UNRELIABLE (box spread >5%) — verdict withheld")
    else:
        surr = np.array([r["cd_surr_mean"] for r in rows])
        alpha_sens = (surr.max() - surr.min()) / max(abs(surr.mean()), 1e-9) > 0.05
        cv_low = (mid["cd_surr_mean"] - cv_mid) / max(abs(mid["cd_surr_mean"]), 1e-9) > 0.10
        quad = ("observable-overestimates" if cv_low else "real-flow-drag")
        quad += "+alpha-sensitive(leak)" if alpha_sens else "+alpha-insensitive"
        print(f"VERDICT: {quad}")
    print("LEAKDRAG-OK")
```

- [ ] **Step 2: CPU mini-gate (failing first)**

```python
# tests/test_leakdrag_probe_cpu.py
"""CPU smoke for the leak-drag probe machinery (tiny config, splu/host)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

def test_probe_runs_tiny_cpu():
    from gpu_leakdrag_discriminator import run_discriminator
    out = run_discriminator(alpha=50.0, nsteps=4, level=4, refine_to=None,
                            wake_refine=None, dt=0.01, device="cpu",
                            mono_solver="splu", assembly="host",
                            t_start_lu=0.0)
    assert np.isfinite(out["cd_surr_mean"])
    for v in out["cd_cv_mean"].values():
        assert np.isfinite(v)
    assert np.isfinite(out["leak_mean_abs"])
    assert out["n_steps_avg"] == 4
```

NOTE: at level=4 uniform the CV boxes must still land on node lines (h=1/16 ⊂ 1/128 snapping — 4L=0.25 margins are multiples of 1/16: verify; if a box edge misses the coarse mesh's node lines, `cv_drag_box` raises — adjust `_boxes()` to snap to the ACTUAL coarsest pitch by parameter if needed; keep the GPU config's snapping at 1/128). Also `strouhal` may raise on 4 steps — guard the probe's St computation with try/except returning NaN for short runs (mirror `gpu_re250_corrected.py`).

- [ ] **Step 3: Run the CPU gate**

Run: `cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && .venv/bin/python -m pytest tests/test_leakdrag_probe_cpu.py -q`
Expected: 1 passed (fix probe until green; the analytic correctness is Task 1's job — this gate is wiring/finiteness).

- [ ] **Step 4: Full local regression**

Run: `cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && .venv/bin/python -m pytest tests/test_p2r1a_thin_plate_flow.py tests/test_cv_drag.py tests/test_leakdrag_probe_cpu.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && git add tests/gpu_leakdrag_discriminator.py tests/test_leakdrag_probe_cpu.py && git commit -m "$(cat <<'EOF'
feat(p2r1a): leak-drag discriminator probe (CV boxes + leak flux + alpha sweep)

run_discriminator marches the corrected Re=250 config accumulating
time-averaged CV drag (3 nested boxes), surrogate Cd, and two-sided leak
flux via on_step; __main__ sweeps alpha {20,50,100} and prints the 2x2
verdict per spec thresholds. CPU mini-gate included.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: GPU execution + verdict recording

**Files:**
- Modify: `docs/dev/thinshell-gpu-runbook.md` (open-item section), `docs/dev/p2r1a-thin-plate-runbook.md` (campaign section)

**Interfaces:**
- Consumes: everything above; the gpubox toolkit.
- Produces: the recorded verdict table + attribution.

- [ ] **Step 1: Ship + run on gpubox**

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && git push gpubox +leakdrag-discriminator:refs/heads/ts-sync
ssh gpubox "cd /home/bglab/Baskar/DiffSim && git reset --hard ts-sync && git log --oneline -1"
bash scripts/remote/gpubox-run.sh ".venv/bin/python tests/gpu_leakdrag_discriminator.py" leakdrag
# poll every ~10 min until LEAKDRAG-OK or Traceback (expect ~1.5-2.5h: 3 x 8000 steps
# + per-step CV/leak host work — if s/step balloons >1s from the python leak-flux loop,
# note it; the run still completes inside ~4h)
bash scripts/remote/gpubox-poll.sh <log> 40
```

- [ ] **Step 2: Record** — append the printed table + VERDICT line to both runbooks (dated subsection "Leak-drag discriminator"), with one paragraph interpreting the quadrant per the spec's table and naming the implied follow-up (observable fix / α scaling / physics-BC investigation). Update the ledger.

- [ ] **Step 3: Commit**

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && git add docs/dev/thinshell-gpu-runbook.md docs/dev/p2r1a-thin-plate-runbook.md && git commit -m "$(cat <<'EOF'
docs(p2r1a): leak-drag discriminator verdict — Cd-excess attribution

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Against Spec

1. ✅ `cv_drag_box` signature, sampling, viscous FD, sign convention, node-line requirement (raises) — Task 1.
2. ✅ `on_step` default-None bit-for-bit + full-field contract + single-construction requirement — Task 2.
3. ✅ Probe: corrected config, 3 snapped nested boxes, leak flux both sides, α∈{20,50,100}, npz histories, spec thresholds (5% spread withhold / 5% α / 10% CV-vs-surr), verdict quadrants, sentinel — Task 3.
4. ✅ CPU analytic + wiring gates; GPU protocol + runbook recording — Tasks 1/3/4.
5. Placeholder scan: verify-first notes name their oracles (driver's `_return_fields` block, `_build_shell` keys, `surrogate_normal_flux` loop); no TBDs. Type check: `run_discriminator` return keys match the `__main__` table and Task 4's recording; `cv_drag_box` call sites match Task 1's signature.
6. Known risk (stated): per-step python leak-flux + CV loops may slow the march (host work per step); acceptable for a diagnostic — flagged for the report if s/step balloons.
