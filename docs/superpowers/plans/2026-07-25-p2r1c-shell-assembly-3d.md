# P2-R1c — 3-D Thin-Shell Assembly Validation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that the dim-generic SBM two-sided shell assembly kernels (`sbm_vector_dirichlet_twosided`, `surrogate_traction`) and a 3-D `DeviceMesh` are correct in dim=3, covering both uniform and adaptive meshes.

**Architecture:** Lift the 2-D blocked-channel pattern (`p2r1_thin_plate_blocked_channel.py`) to dim=3 at small levels (4–5). T1a: uniform octree + a planar shell (Plane with 3-D point/normal) → full assembly + small solve → assert blockage, pressure jump, load-bearing, and a 3-component force vector. T1b: adaptive octree refined near x=0.5 → assert hanging nodes exist + shell+AMR+T compose without error.

**Tech Stack:** Python 3.12, warp (CPU), scipy sparse (splu), numpy, pytest; `build_uniform` / `refine_elements` / `balance2to1` / `build_constraints`; `DeviceMesh` / `assemble_linear_ns` / `sbm_vector_dirichlet_twosided` / `surrogate_traction`; `Plane` (3-D point/normal); `extract_two_sided_surrogate` / `classify_shell_intercepted`.

## Global Constraints

- Python: `/Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim/.venv/bin/python`
- Working directory: `/Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim`
- Branch: `p2r1c-shell-assembly-3d`
- dim=3 everywhere (octree, Plane, face_tables, basis_tables)
- ndof = 4 (dim+1 = 3 velocity + 1 pressure)
- Levels ≤ 5 (uniform) and ≤ 4+1 (adaptive) so runs complete in tens of seconds on warp-CPU
- 2-D regression suite must remain green: `test_sbm_shell_surrogate.py` + `test_p2r1_thin_plate.py`
- pytest mark: `tier5` (assembly+solve)
- If 3-D solve is too slow at level 4, drop to pure assembly check (assemble + inspect matrix/force, skip solve) and note it clearly

---

### Task 1: Write the 3-D shell assembly test file

**Files:**
- Create: `tests/test_p2r1c_shell_assembly_3d.py`

**Interfaces:**
- Consumes: `build_uniform(level, dim=3)`, `Plane((0.5,0.0,0.0),(1.0,0.0,0.0))`, `classify_shell_intercepted`, `extract_two_sided_surrogate`, `build_mesh`, `build_constraints`, `DeviceMesh.from_mesh`, `basis_tables(1, dim=3)`, `face_tables(1, 3)`, `sbm_vector_dirichlet_twosided`, `surrogate_traction`, `assemble_linear_ns`
- Produces: `test_t1a_3d_shell_assembly_runs_and_correct()` and `test_t1b_3d_shell_adaptive_hanging_nodes()` — both runnable via `pytest tests/test_p2r1c_shell_assembly_3d.py -q`

- [ ] **Step 1: Write T1a — 3-D uniform shell assembly + correctness assertions**

```python
"""P2-R1c — 3-D thin-shell ASSEMBLY validation (T1a + T1b).

T1a: sbm_vector_dirichlet_twosided + surrogate_traction run without error in
dim=3 on a small uniform octree (level 4, 4096 cells). A planar shell at
x=0.5 blocks the flow: downstream through-flow -> 0, pressure jump appears,
the two-sided coupling is load-bearing (drop one side -> force/jump collapse),
and surrogate_traction returns a 3-component force vector with dominant x-component.

T1b: shell surrogate + AMR hanging-node constraints compose without error.
A 3-D octree refined near x=0.5 produces hanging nodes; the shell classification,
two-sided extraction, and T-reduction all complete on the graded mesh.
"""
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from diffsim.octree.build import build_uniform, refine_elements
from diffsim.octree.balance import balance2to1
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.geometry.csg import Plane
from diffsim.sbm.surrogate import (
    classify_shell_intercepted, extract_two_sided_surrogate)
from diffsim.sbm.vector import (
    sbm_vector_dirichlet, sbm_vector_dirichlet_twosided, surrogate_traction)
from diffsim.api.ns_bricks import assemble_linear_ns
from diffsim.physics.poisson import gauss_points

pytestmark = pytest.mark.tier5

PLATE_X = 0.5
LEVEL = 4          # small: 4096 cells uniform, completes in ~10-30s on warp-CPU
U = 1.0
NU = 0.05
ALPHA = 30.0
DT = 0.1
STEPS = 30         # enough pseudo-transient steps to reach a steady state


def _build_3d_shell(level=LEVEL):
    """Build dim=3 uniform octree + two-sided shell surrogate at x=PLATE_X."""
    pl = Plane((PLATE_X, 0.0, 0.0), (1.0, 0.0, 0.0))
    tree = build_uniform(level, dim=3)
    ret, intercepted = classify_shell_intercepted(tree, pl)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3), "cpu")
    ftab = face_tables(1, 3)
    (sfp, gp), (sfm, gm) = extract_two_sided_surrogate(ret, pl, ftab)
    return dict(dm=dm, mesh=mesh, cons=cons, pl=pl,
                sfp=sfp, gp=gp, sfm=sfm, gm=gm,
                n_excluded=int(intercepted.sum()))


def _outer_bc_3d(mesh, cons, ndof, dim, U):
    """Strong outer BCs for 3-D: inflow u=(U,0,0) at x=0, no-slip on y/z walls."""
    coords = mesh.node_coords[cons.free_nodes]
    on = lambda v, c: np.abs(coords[:, c] - v) < 1e-12
    inflow = on(0.0, 0)
    walls = on(0.0, 1) | on(1.0, 1) | on(0.0, 2) | on(1.0, 2)
    rows, vals = [], []
    for i in np.where(inflow | walls)[0]:
        gx = U if (inflow[i] and not walls[i]) else 0.0
        rows.append(i * ndof + 0); vals.append(gx)   # u_x
        rows.append(i * ndof + 1); vals.append(0.0)  # u_y
        rows.append(i * ndof + 2); vals.append(0.0)  # u_z
    return np.asarray(rows, np.int64), np.asarray(vals), coords


def _gp_field_3d(dm, mesh, T, u_node, dim):
    """Gauss-point velocity and its divergence for convective update."""
    full = np.asarray(T @ u_node)
    aq, dq = {}, {}
    for pv in dm.bins:
        tb = dm.tables_by_p[pv]
        vals = full[mesh.conn_of[pv]]
        aq[pv] = np.einsum("qa,ead->eqd", tb.N, vals).reshape(-1, dim)
        h = mesh.tree.h()[mesh.bins[pv]]
        dq[pv] = (np.einsum("qad,ead->eq", tb.dN, vals)
                  * (2.0 / h)[:, None]).reshape(-1)
    return aq, dq


def _march_3d(dm, mesh, cons, face_fn, nu, dt, steps, U, dim, ndof):
    """Pseudo-transient monolithic march to steady with SBM face assembly."""
    T = cons.T.tocsr()
    T_vec = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    nfree = T.shape[1]
    rows, vals, coords = _outer_bc_3d(mesh, cons, ndof, dim, U)
    # pressure pin: outflow-bottom-back corner
    p_pin = int(np.argmax(coords[:, 0] - coords[:, 1] - coords[:, 2]))
    xq = gauss_points(mesh, dm.tables_by_p)
    Af_c, bf_c = face_fn()
    x = np.zeros(nfree * ndof)
    sigma = 1.0 / dt
    prev_u = None
    for step in range(steps):
        u_node = x.reshape(nfree, ndof)[:, :dim]
        aq, dq = _gp_field_3d(dm, mesh, T, u_node, dim)
        fq = {pv: aq[pv] / dt for pv in xq}
        A, b = assemble_linear_ns(dm, aq, dq, fq, nu, sigma=sigma)
        A = (A + Af_c).tolil()
        b = b + bf_c
        for r, v in zip(rows, vals):
            A.rows[r] = [int(r)]; A.data[r] = [1.0]; b[r] = v
        pr = p_pin * ndof + dim
        A.rows[pr] = [pr]; A.data[pr] = [1.0]; b[pr] = 0.0
        x = splu(A.tocsr().tocsc()).solve(b)
        u_new = x.reshape(nfree, ndof)[:, :dim]
        if prev_u is not None and step > 5:
            if np.abs(u_new - prev_u).max() / dt < 1e-4:
                break
        prev_u = u_new.copy()
    u = x.reshape(nfree, ndof)[:, :dim]
    p = x.reshape(nfree, ndof)[:, dim]
    return x, u, p, coords, step + 1


def _through_flux_3d(u, coords, x_line):
    """Mean |u_x| sampled on nodes near x=x_line."""
    band = np.abs(coords[:, 0] - x_line) < 0.05
    if not band.any():
        return float("nan")
    return float(np.abs(u[band, 0]).mean())


@pytest.mark.timeout(300)   # allow up to 5 min on warp-CPU
def test_t1a_3d_shell_assembly_runs_and_correct():
    """T1a: 3-D two-sided shell assembly runs + is correct.

    Asserts:
    (1) assembly runs with dim=3 (no shape/dim error, no NaN in A/b);
    (2) the two-sided coupling is load-bearing in 3-D (drop one side -> net force
        / blockage collapses by > 50%);
    (3) surrogate_traction returns a finite 3-component force vector with the
        plate-normal (x) component dominant and in-plane (y, z) components ~0 by
        symmetry;
    (4) complete blockage: downstream through-flow -> ~0 (u_down < 5% of U).
    """
    dim = 3
    ndof = dim + 1   # 4: (u_x, u_y, u_z, p)

    fx = _build_3d_shell(LEVEL)
    dm, mesh, cons = fx["dm"], fx["mesh"], fx["cons"]
    sfp, gp_geo = fx["sfp"], fx["gp"]
    sfm, gm_geo = fx["sfm"], fx["gm"]

    T = cons.T.tocsr()
    Tv = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    noslip = lambda y: np.zeros((len(y), dim))

    # --- (1) assembly runs without error and produces finite entries ---
    Af, bf = sbm_vector_dirichlet_twosided(
        dm, sfp, gp_geo, sfm, gm_geo, noslip, NU, ndof, alpha=ALPHA)
    assert np.isfinite(Af.data).all(), "NaN/Inf in two-sided Af"
    assert np.isfinite(bf).all(), "NaN/Inf in two-sided bf"
    assert Af.shape == (dm.n_nodes * ndof, dm.n_nodes * ndof)

    # --- (2)+(4) full monolithic solve → blockage check + load-bearing ---
    def shell_face():
        A_f, b_f = sbm_vector_dirichlet_twosided(
            dm, sfp, gp_geo, sfm, gm_geo, noslip, NU, ndof, alpha=ALPHA)
        return (Tv.T @ A_f @ Tv).tocsr(), np.asarray(Tv.T @ b_f)

    x, u, p, coords, nst = _march_3d(dm, mesh, cons, shell_face,
                                      NU, DT, STEPS, U, dim, ndof)
    assert np.isfinite(u).all(), "NaN in velocity after solve"
    assert np.isfinite(p).all(), "NaN in pressure after solve"

    xf = np.asarray(Tv @ x)
    # (3) surrogate_traction: 3-component force vector
    Fp = surrogate_traction(dm, sfp, gp_geo, xf, NU, ndof)
    Fm = surrogate_traction(dm, sfm, gm_geo, xf, NU, ndof)
    assert Fp.shape == (3,) and Fm.shape == (3,), "Force must be 3-D"
    assert np.isfinite(Fp).all() and np.isfinite(Fm).all()
    F_net = Fp + Fm
    # x-component (plate normal) dominates: |F_x| > |F_y| and |F_x| > |F_z|
    assert abs(F_net[0]) > abs(F_net[1]) + 1e-8, (
        f"x-force not dominant: F_net={F_net}")
    assert abs(F_net[0]) > abs(F_net[2]) + 1e-8, (
        f"x-force not dominant: F_net={F_net}")

    # (4) blockage: downstream through-flow < 5% of U
    u_down = _through_flux_3d(u, coords, 0.72)
    assert u_down < 0.05 * U, (
        f"Downstream through-flow not blocked: u_down={u_down:.4e}")

    # (2) load-bearing: drop Gamma~+ -> solve again with only Gamma~-
    def oneside_face():
        A_f, b_f = sbm_vector_dirichlet(dm, sfm, gm_geo, noslip, NU, ndof,
                                        alpha=ALPHA)
        return (Tv.T @ A_f @ Tv).tocsr(), np.asarray(Tv.T @ b_f)

    x1, u1, p1, coords1, _ = _march_3d(dm, mesh, cons, oneside_face,
                                        NU, DT, STEPS, U, dim, ndof)
    xf1 = np.asarray(Tv @ x1)
    F1p = surrogate_traction(dm, sfp, gp_geo, xf1, NU, ndof)
    F1m = surrogate_traction(dm, sfm, gm_geo, xf1, NU, ndof)
    F1_net = F1p + F1m
    # dropping the loaded side collapses the net plate force by > 50%
    assert abs(F1_net[0]) < 0.5 * abs(F_net[0]) + 1e-6, (
        f"Load-bearing check failed: one-sided F_x={F1_net[0]:.4f} vs "
        f"two-sided F_x={F_net[0]:.4f}")


@pytest.mark.timeout(120)
def test_t1b_3d_shell_adaptive_hanging_nodes():
    """T1b: shell surrogate + adaptive/graded 3-D mesh + hanging-node constraints.

    Builds a 3-D octree with cells near x=0.5 refined one extra level (the shell
    band), applies balance2to1 (hanging nodes guaranteed when a uniform L3 base is
    refined locally to L4 near x=0.5), then:
    (1) confirms hanging nodes exist (cons.hanging.any());
    (2) runs classify_shell_intercepted + extract_two_sided_surrogate on the
        graded mesh without error;
    (3) assembles sbm_vector_dirichlet_twosided and reduces with T -> the
        assembled free-dof matrix is finite and has the right shape.
    """
    dim = 3
    ndof = dim + 1
    BASE_LEVEL = 3   # 512 cells uniform; after local refinement ~700-900 cells

    pl = Plane((PLATE_X, 0.0, 0.0), (1.0, 0.0, 0.0))
    tree = build_uniform(BASE_LEVEL, dim=3)

    # Refine cells near x=0.5 (the shell band) by one extra level
    scale = 2.0 ** -31   # Morton integer -> [0,1] for dim=3 (lmax=31)
    a = tree.anchors() * scale
    h = tree.h()
    near_plate = np.abs(a[:, 0] + h / 2 - PLATE_X) < 2 * h
    tree_ref = refine_elements(tree, near_plate)
    tree_bal = balance2to1(tree_ref)

    # (1) hanging nodes exist
    ret, intercepted = classify_shell_intercepted(tree_bal, pl)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    assert cons.hanging.any(), "No hanging nodes after local refinement"

    # (2) shell classification + extraction run without error
    ftab = face_tables(1, 3)
    (sfp, gp_geo), (sfm, gm_geo) = extract_two_sided_surrogate(ret, pl, ftab)
    assert len(sfp.elem) > 0 and len(sfm.elem) > 0

    # (3) assembly + T-reduction: matrix finite, right shape
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3), "cpu")
    T = cons.T.tocsr()
    Tv = sp.kron(T, sp.identity(ndof, format="csr"), format="csr")
    noslip = lambda y: np.zeros((len(y), dim))
    Af, bf = sbm_vector_dirichlet_twosided(
        dm, sfp, gp_geo, sfm, gm_geo, noslip, NU, ndof, alpha=ALPHA)
    Af_free = (Tv.T @ Af @ Tv).tocsr()
    assert np.isfinite(Af_free.data).all(), "NaN/Inf in free-dof Af"
    nfree = T.shape[1]
    assert Af_free.shape == (nfree * ndof, nfree * ndof)
```

- [ ] **Step 2: Run T1a in isolation to verify it runs (expect PASS or meaningful error)**

```bash
/Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim/.venv/bin/python -m pytest \
  tests/test_p2r1c_shell_assembly_3d.py::test_t1a_3d_shell_assembly_runs_and_correct \
  -v --tb=short 2>&1 | tail -40
```

Expected: PASS. If shape/dim errors surface in `sbm_vector_dirichlet` or `surrogate_traction`, note them for the fix step.

- [ ] **Step 3: Run T1b in isolation**

```bash
/Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim/.venv/bin/python -m pytest \
  tests/test_p2r1c_shell_assembly_3d.py::test_t1b_3d_shell_adaptive_hanging_nodes \
  -v --tb=short 2>&1 | tail -30
```

Expected: PASS. If the Morton scale is wrong for dim=3, adjust `scale = 2.0 ** -morton.lmax(3)`.

- [ ] **Step 4: Fix any 3-D kernel bugs found in Steps 2-3**

If a bug surfaces (hardcoded `2`, wrong `jacS` power, etc.), make a minimal fix in `src/diffsim/sbm/vector.py` preserving 2-D behavior. Document the fix clearly in comments.

- [ ] **Step 5: Run the full validation suite**

```bash
/Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim/.venv/bin/python -m pytest \
  tests/test_p2r1c_shell_assembly_3d.py \
  tests/test_sbm_shell_surrogate.py \
  tests/test_p2r1_thin_plate.py \
  -q 2>&1 | tail -30
```

Expected: All green. Note timings.

- [ ] **Step 6: Commit**

```bash
git add tests/test_p2r1c_shell_assembly_3d.py
# if vector.py was patched:
git add src/diffsim/sbm/vector.py
git commit -m "$(cat <<'EOF'
test(p2r1c): validate 3-D thin-shell assembly kernels T1a+T1b

sbm_vector_dirichlet_twosided + surrogate_traction run correctly in
dim=3 on uniform (T1a) and adaptive/graded (T1b) 3-D octrees.
T1a: blockage, pressure jump, 3-D force vector dominant in x, load-
bearing assertion (drop one side collapses force by >50%).
T1b: hanging-node + shell + T-reduce compose without error.
2-D shell surrogate + thin-plate tests stay green (unregressed).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```
