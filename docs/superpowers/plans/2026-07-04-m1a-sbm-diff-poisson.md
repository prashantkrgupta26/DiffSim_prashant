# DiffSim M1a: SBM Core + Geometry Oracles + Differentiable Poisson — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**M1 phasing:** M1 (spec §1.3, roadmap §10) ships in three plans. **M1a (this plan):** SBM core + three geometry backends (AnalyticCSG, STL/TriMesh, GridSDF) + differentiable Poisson with shape and conductivity gradients — Tier 4/5 + the AD foundations (Tier-2 VJPs #1 and #3). **M1b (next):** NS-VMS forward — both steppers, BDF1/BDF2, cylinder/sphere/cavity benchmarks; the domain flag flips to `"outside"` (flow around the object). **M1c:** NS adjoints (shape + viscosity gradients) + NeuralSDF backend + neural-SDF hero demo (completing M1's four-backend gate). M1b's plan is authored after M1a merges, absorbing its findings (the M0→M0.5 pattern).

**Goal:** Octree-SBM Poisson (Dirichlet + Neumann) on carved k-D meshes, with the GeometryOracle contract implemented by AnalyticCSG, STL/TriMesh, and GridSDF backends (NeuralSDF completes the set in M1c), and end-to-end adjoint gradients w.r.t. geometry parameters and conductivity κ validated against finite differences. Gates: **P4 keystone** (SBM linear patch on rotated geometries at all λ, machine precision), SBM MMS orders (p1 → 2, p2 → 3, error measured on Ω), the **π/4 area-correction locked test**, the **§13.3.3 p2-band Neumann acceptance test** (restored flux order + layer-sweep insensitivity), and **three-way gradient checks** (adjoint vs torch-twin vs central FD, rel < 1e-6).

**Architecture:** Geometry lives in torch (FP64) behind the SDF-oracle contract; distance vectors come from the framework Newton closest-point projection whose backward is the implicit-function theorem (Tier-2 VJP #3, spec §5.2). Per topology epoch (spec §4.2): carve → λ-classify at Gauss points → extract surrogate faces → one batched oracle evaluation of (d, n, n̄·n) cached FP64. SBM face terms are new Warp kernel factories on face-restricted basis tables; M1a solves use the **assembled-CSR path** (matrix-free SBM matvec arrives with NS in M1b), so the adjoint operator is literally `A.T` (Tier-2 VJP #1). ∂R/∂θ is a Warp-tape sweep over pure face-residual kernels chained into the torch geometry graph. Spec: §4 (pipeline), §1.5/§5.2 (differentiability), §13.1–13.3 (Neumann shift, p2 band), §9 (verification).

**Tech stack additions:** PyTorch (FP64, CUDA build) in `.venv`. Everything else unchanged (Python 3.12, warp-lang 1.14, NumPy, SciPy, pytest).

## Global Constraints

- **All 109 existing tests pass unmodified after every task** (additive-only edits to existing test files where a task says so). Run `.venv/bin/pytest -q` at the end of every task.
- FP64 spine (spec §2.2.4): all geometry data (ψ, d, n, corr), residuals, and reductions are FP64. Torch code uses `torch.float64` explicitly — never rely on default dtype.
- Verification pass criteria are fixed (spec §9.2): patch tests atol 1e-11; observed orders within ±0.10 over the last 3 of ≥4 levels; gradient checks rel < 1e-6 (FP64); baseline comparisons rtol 1e-6.
- Kernel cache keys: every new factory keys on a distinct name string plus `(nbf, nqf, dim)` — no two integrands share a key (M0 deferred finding #3).
- **Domain-side flag (locked here, used everywhere):** primitives keep ψ < 0 inside the shape; the **computational domain is selected by `domain ∈ {"inside", "outside"}`**, threaded through `classify_lambda`, `GeometryData.evaluate`, and the error masks — an eventual user config knob (spec §3.1 InputData). M1a solves the physics **inside** the object: `domain="inside"` (Ω = {ψ<0}) is the default. M1b's flow problems are **outside** the object: `domain="outside"` (Ω = {ψ>0}) — exercised now by this plan's exterior-configuration tests so M1b inherits a tested path. Retention counts Gauss points on the domain side. True-boundary outward normal n = s·∇ψ(y)/‖∇ψ(y)‖ with s = +1 ("inside") / −1 ("outside") — always pointing out of Ω; corr = ñ·n uses this oriented n. Surrogate normal ñ = axis-aligned outward from Ω̃. d = y − x̃ points from the surrogate Gauss point to its closest point y on Γ.
- **No partially-exposed surrogate faces (M1a invariant):** every surrogate face is a whole element face (validated by sub-face probes at extraction; `ValueError` on violation). Narrowband refinement to a uniform level near Γ guarantees this. General nonconforming surrogate facets are deferred to M4 (epochs/AMR). Rationale: SC'21 cancellation-node rule, spec §2.3.
- Weak-form sign/placement conventions follow Main & Scovazzi (JCP 2018, Dirichlet) and Atallah–Scovazzi (CMAME 2020, Neumann) as transcribed in Tasks 6/8; the transcriptions below are the contract (spec §13.1 requires this verification — it is done in this plan; re-derive the Γ̃→Γ limit sanity checks in the tests).
- Purity (spec §3.2): all new face kernels are pure — every Gauss-point datum arrives via arrays; no hidden state. This is what makes the ∂R/∂θ tape sweep valid.
- The `.numpy()` host boundaries in Krylov/BLAS (M0 deferred finding #1) are **routed around, not through**: gradients never tape through a solve; solves are Tier-2 VJPs (adjoint solve on `A.T`).
- SBM Nitsche operators are **nonsymmetric** (the adjoint-consistency term −κ(∇w·ñ)(∇u·d) has no transpose partner): SBM solves use `bicgstab` (guards added in Task 1), never `cg`.
- New pytest markers: `tier4` (geometry oracles, classification, surrogate extraction), `tier5` (SBM solves, MMS, backends), `ad` (gradient checks).
- Code standards (spec §16): header docblocks (equation + spec/paper reference, symbol glossary, layout notes, invariants) on every new module/kernel; block-level comments only. Commits: conventional style.

## File Structure

```
Modify: pyproject.toml                       # torch dep; tier4/tier5/ad markers
Modify: src/diffsim/assembly/femelm.py       # drop dead fe_dN/fe_detJxW; add fe_d2N_s
Modify: src/diffsim/assembly/operators.py    # drop dead imports; add CSROperator
Modify: src/diffsim/mesh/nodes.py            # p_elem assert -> ValueError (M0.5 finding)
Modify: src/diffsim/mesh/basis.py            # lagrange_1d_d2(p, x)
Modify: src/diffsim/solvers/krylov.py        # bicgstab breakdown guards
Modify: src/diffsim/physics/bratu.py         # "uniform mesh only" docstring (M0.5 finding)
Create: src/diffsim/geometry/__init__.py
Create: src/diffsim/geometry/oracle.py       # SDFOracle base + numpy bridge + admissibility diagnostics
Create: src/diffsim/geometry/csg.py          # AnalyticCSG: primitives, rigid transforms, blends (torch)
Create: src/diffsim/geometry/project.py      # Newton closest-point projection + IFT VJP (Tier-2 #3)
Create: src/diffsim/geometry/gridsdf.py      # GridSDF: voxel grid + multilinear interp (torch)
Create: src/diffsim/geometry/trimesh.py      # TriMeshOracle: wp.Mesh queries + FP64 torch refinement
Create: src/diffsim/mesh/faces.py            # FaceTables: N/dN/d2N at face Gauss points
Create: src/diffsim/mesh/pointeval.py        # point-evaluation weight matrix (probe QoIs)
Create: src/diffsim/sbm/__init__.py
Create: src/diffsim/sbm/surrogate.py         # lambda classification, face extraction, GeometryData cache
Create: src/diffsim/sbm/poisson.py           # SBMPoisson: assembly, solve, surrogate flux
Create: src/diffsim/sbm/adjoint.py           # adjoint solve + shape/kappa gradient driver (Tier-2 #1)
Create: src/diffsim/sbm/reference.py         # torch dense clear-path twin (spec S16.2)
Create: tests/test_geometry_oracles.py       # Tasks 2, 3, 9
Create: tests/test_face_tables.py            # Task 4
Create: tests/test_sbm_surrogate.py          # Task 5
Create: tests/test_sbm_poisson.py            # Tasks 6, 7
Create: tests/test_sbm_neumann.py            # Task 8
Create: tests/test_ad_gradients.py           # Tasks 10, 11
Create: tests/baselines/m1a_baselines.json   # locked in Task 11
```

**Interface summary (locked here, used everywhere):**

- `geometry.oracle.SDFOracle`: abstract base — `psi(x: torch.Tensor[N,dim]) -> torch.Tensor[N]` (dtype float64, graph-connected to `self.params`), `params: list[torch.Tensor]`, `near_eikonal: bool = False`. Provides numpy-facing `classify(pts) -> psi[N]`, `distance_vector(pts) -> (d[N,dim], n[N,dim], ok[N] bool)`, `velocity(pts, t) -> zeros` (static, M1). `admissibility(oracle, band_pts) -> dict(eps_inf, c0, d_hausdorff_bound, newton_ok_frac)`.
- `geometry.project.closest_point(oracle, x_t: torch.Tensor) -> (y_t, ok)` — custom `torch.autograd.Function`, IFT backward; `distance_torch(oracle, x_np) -> (d_t, n_t, ok)` — full torch graph from `oracle.params` (adjoint driver re-runs this in backward); `distance_numpy(oracle, x_np)` — detached FP64 arrays for the forward pipeline.
- `mesh.faces.face_tables(p, dim) -> FaceTables(p, dim, nqf, N[2*dim,nqf,nbf], dN[2*dim,nqf,nbf,dim], d2N[2*dim,nqf,nbf,dim,dim], w[nqf])` — face id `f = 2*ax + side`, side 0 = minus (ñ = −e_ax), matching `face_offsets` order; reference-element quantities (physical: dN·(2/h), d2N·(2/h)², dS = w·(h/2)^(dim−1)).
- `sbm.surrogate.classify_lambda(tree, oracle, lam, domain="inside") -> (tree_retained, frac_in[Nret])` and `extract_surrogate(tree_retained) -> SurrogateFaces(elem[Nf] int64, face[Nf] int8)`; `face_gauss_points(tree, sf, ftab) -> xq[Nf*nqf, dim]`; `GeometryData.evaluate(oracle, tree, sf, ftab, domain="inside") -> GeometryData(xq, d, n, corr, ok)` (all FP64 numpy, frozen per epoch; `n` oriented out of Ω per the domain flag).
- `assembly.operators.CSROperator(A_scipy_csr, device)` — `.matvec(x_wp, y_wp)`, `.matvec_numpy(x)`, `.n_free`, `.device` (the operator protocol, formalizing the M0 deferred item).
- `sbm.poisson.SBMPoisson(dm, geo, sf, oracle_g=None, neumann=None, kappa=1.0, alpha=10.0)` — `.assemble() -> (A_csr, b, meta)` (constrained, outer strong Dirichlet by row replacement), `.solve(...) -> u_all[Nn]`, `.surrogate_flux(u_all) -> float` (area-corrected shifted flux ∫_Γ̃ κ(S∇u·n)(n·ñ) dS̃).
- `sbm.adjoint.solve_adjoint(A_csr, dJdu_free, device) -> lam_free`; `shape_gradient(problem, u_all, lam_free, oracle) -> dict(param_tensor -> grad)`; `kappa_gradient(problem, u_all, lam_free) -> float`.
- QoIs: `volume_qoi(dm) -> (J(u_all), dJdu_free)` for J = ∫_Ω̃ u dV (dJ/du = mass row-sums via the existing load kernel with f ≡ 1); `probe_qoi(mesh, T, pts, targets)` via `mesh.pointeval.point_eval_weights(mesh, pts) -> scipy.sparse [Npts, Nn]`.

---

### Task 1: Dependencies + M0.5 deferred cleanups + bicgstab guards

**Files:**
- Modify: `pyproject.toml`, `src/diffsim/assembly/femelm.py`, `src/diffsim/assembly/operators.py`, `src/diffsim/mesh/nodes.py`, `src/diffsim/physics/bratu.py`, `src/diffsim/solvers/krylov.py`
- Test: append to `tests/test_s_solvers.py` and `tests/test_mixedp.py` (additive only)

**Steps:**

- [x] **Step 1: Install torch and record versions**

```bash
.venv/bin/pip install torch
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected: CUDA available `True`. Add `"torch>=2.4"` to `pyproject.toml` dependencies and the new markers:

```toml
markers = [
  "tier1: sanity & determinism (octree infrastructure)",
  "tier2: connectivity & patch tests",
  "tier3: operator & assembly correctness",
  "tier4: geometry oracles, classification, surrogate extraction",
  "tier5: SBM solves, MMS orders, backend cross-verification",
  "ad: adjoint/gradient checks (Tier AD)",
]
```

- [x] **Step 2: M0.5 deferred cleanups (docs/superpowers/m0-deferred-findings.md items)**

1. Delete `fe_dN` and `fe_detJxW` from `femelm.py` (zero callers); drop them from `assembly/__init__.py` re-exports and the unused import in `operators.py` (keep `fe_N`).
2. `nodes.py`: replace `assert set(np.unique(p_elem)) <= {1, 2}` with `raise ValueError(f"p_elem must contain only 1 and 2, got {sorted(set(np.unique(p_elem)))}")` guarded by an `if`.
3. Docstrings: add "Uniform mesh only (single-bin DeviceMesh); mixed-p meshes raise at first back-compat accessor" to `integrate_volume` and `BratuProblem`; add the deliberate `dm.tables` no-assert vs `conn/h/N/dN/w` assert asymmetry note to the `DeviceMesh` class docstring.
4. `krylov.py` bicgstab breakdown guards: after `rho_new = blas.dot(rhat, r, d)`, and after computing `rhat_v = blas.dot(rhat, v, d)`, bail out cleanly instead of dividing by ~0:

```python
        if abs(rho_new) < 1e-300 * bnorm * bnorm:
            return x.numpy(), {"iters": it, "relres": rnorm / bnorm,
                               "converged": False, "breakdown": "rho"}
        ...
        rhat_v = blas.dot(rhat, v, d)
        if abs(rhat_v) < 1e-300 * bnorm * bnorm:
            return x.numpy(), {"iters": it, "relres": rnorm / bnorm,
                               "converged": False, "breakdown": "rhat_v"}
        alpha = rho_new / rhat_v
```

(`rnorm` must be initialized before the loop: `rnorm = np.sqrt(blas.dot(r, r, d))` after `r = wp.clone(bd)`.)

- [x] **Step 3: Tests (additive)**

Append to `tests/test_mixedp.py`: `build_mesh(tree, p=np.full(len(tree), 3, np.int8))` raises `ValueError`. Append to `tests/test_s_solvers.py`: bicgstab on a 2×2 system with `b` orthogonal to the Krylov direction that forces `rho` breakdown returns `converged=False` with a `"breakdown"` key (construct: `op` = identity CSR-like operator via a tiny stub with `.device`/`.n_free`/`.matvec`, `rhat ⊥ r` is impossible at it=1 since `rhat = r = b`; instead assert the guard path via a singular operator `A = [[0,1],[0,0]]`-like stub driving `rhat·v → 0`). Keep this test minimal — the guard is defensive; the assertion is "no ZeroDivisionError, clean info dict".

- [x] **Step 4: Full suite + commit**

`.venv/bin/pytest -q` → 109 + new pass.

```bash
git add -A && git commit -m "chore: torch dep, M0.5 deferred cleanups, bicgstab breakdown guards"
```

---

### Task 2: GeometryOracle protocol + AnalyticCSG backend

**Files:**
- Create: `src/diffsim/geometry/__init__.py`, `src/diffsim/geometry/oracle.py`, `src/diffsim/geometry/csg.py`
- Test: `tests/test_geometry_oracles.py`

**Interfaces:** as in the summary. CSG classes (all torch FP64, params are `torch.nn.Parameter`-free plain tensors with `requires_grad_(True)` set by the caller when differentiating):

- `Sphere(center, radius)` — exact SDF, `near_eikonal=True`. Works in any dim (circle at dim=2).
- `Box(center, half, rotation=None)` — exact SDF `|max(q,0)| + min(max_i q_i, 0)`, `q = |R^T(x−c)| − half`; rotation: dim=2 scalar angle, dim=3 axis-angle vector (Rodrigues). `near_eikonal=True`.
- `Complement(child)` — ψ → −ψ (turns "inside the shape" into "outside is the domain": the exterior-flow configuration).
- `Union(a, b, k=0.0)` / `Intersection(a, b, k=0.0)` — exact min/max at k=0; polynomial smooth blend for k>0 (`h = clamp(0.5 + 0.5*(ψb−ψa)/k, 0, 1); smin = lerp(ψb, ψa, h) − k*h*(1−h)`). Smooth blends are NOT eikonal → `near_eikonal=False` (exercises Newton projection).
- `Translate(child, offset)`.

Sign convention: primitives are ψ < 0 inside the shape; **which side is the computational domain is chosen by the `domain` flag downstream** (see Global Constraints), not by the oracle. `Complement` remains a set operation for composing shapes (e.g., a plate with a hole), not the mechanism for exterior domains.

`oracle.py` provides:

```python
class SDFOracle:
    near_eikonal: bool = False
    @property
    def params(self) -> list: ...            # leaf torch tensors
    def psi(self, x): ...                    # torch [N] from torch [N,dim]
    # numpy bridge (forward pipeline; FP64, detached)
    def classify(self, pts):
        with torch.no_grad():
            return self.psi(torch.as_tensor(pts, dtype=torch.float64)).numpy()
    def distance_vector(self, pts):          # delegates to project.distance_numpy
    def velocity(self, pts, t=0.0):          # zeros [N,dim] (static geometry, M1)
```

plus `admissibility(oracle, band_pts) -> dict` (Task 3 fills the projection-dependent entries).

**Steps:**

- [x] **Step 1: Failing tests** (`tests/test_geometry_oracles.py`, `pytestmark = pytest.mark.tier4`)

```python
import numpy as np
import pytest
import torch
from diffsim.geometry.csg import Sphere, Box, Complement, Union, Intersection, Translate

def test_sphere_sdf_exact():
    for dim in (2, 3):
        s = Sphere((0.5,) * dim, 0.3)
        rng = np.random.default_rng(7)
        pts = rng.uniform(0, 1, (200, dim))
        psi = s.classify(pts)
        assert np.allclose(psi, np.linalg.norm(pts - 0.5, axis=1) - 0.3, atol=1e-14)

def test_rotated_box_sdf():
    th = 0.4
    b = Box((0.5, 0.5), (0.2, 0.1), rotation=torch.tensor(th, dtype=torch.float64))
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    corner = 0.5 + R @ np.array([0.2, 0.1])          # rotated corner lies on the boundary
    assert abs(b.classify(corner[None, :])[0]) < 1e-14
    assert b.classify(np.array([[0.5, 0.5]]))[0] < 0  # center inside

def test_complement_and_blend():
    dom = Complement(Sphere((0.5, 0.5), 0.25))        # exterior domain
    assert dom.classify(np.array([[0.5, 0.5]]))[0] > 0     # inside sphere = OUTSIDE domain
    assert dom.classify(np.array([[0.05, 0.05]]))[0] < 0
    a, b = Sphere((0.4, 0.5), 0.2), Sphere((0.6, 0.5), 0.2)
    uni = Union(a, b, k=0.05)
    x = np.array([[0.5, 0.5]])
    assert uni.classify(x)[0] <= min(a.classify(x)[0], b.classify(x)[0]) + 1e-12
    assert not uni.near_eikonal and Sphere((0, 0), 1.0).near_eikonal

def test_params_graph():
    s = Sphere((0.5, 0.5), 0.3)
    for p in s.params:
        p.requires_grad_(True)
    x = torch.tensor([[0.9, 0.5]], dtype=torch.float64)
    s.psi(x).sum().backward()
    grads = [p.grad for p in s.params]
    assert all(g is not None for g in grads)
    # d psi/d r = -1 exactly
    assert abs(float(s.radius.grad) + 1.0) < 1e-14
```

- [x] **Step 2: Run to verify failure** — `.venv/bin/pytest tests/test_geometry_oracles.py -v` → FAIL (module missing).

- [x] **Step 3: Implement `oracle.py` + `csg.py`** per interfaces above. Every tensor `torch.float64`. `params` collects leaf tensors recursively (children first). Rodrigues rotation for dim=3; scalar angle for dim=2. Header docblocks cite spec §4.1 and note the ψ<0-inside-Ω convention.

- [x] **Step 4: Run + full suite + commit**

```bash
git add -A && git commit -m "feat: GeometryOracle protocol + AnalyticCSG torch backend"
```

---

### Task 3: Newton closest-point projection with IFT adjoint (Tier-2 VJP #3)

**Files:**
- Create: `src/diffsim/geometry/project.py`; extend `oracle.py` (`distance_vector`, `admissibility`)
- Test: append to `tests/test_geometry_oracles.py`

**Math (spec §4.1):** forward iteration `y ← y − ψ(y)∇ψ(y)/‖∇ψ(y)‖²` from `y₀ = x` (cap 50, tol |ψ| < 1e-13, per-point convergence mask). At convergence define the augmented system in unknowns `z = (y, s)`:

```
F(y, s; θ, x) = [ y − x + s ∇ψ(y; θ) ;  ψ(y; θ) ] = 0,      s = (x − y)·∇ψ / ‖∇ψ‖²
J_z = [[ I + s Hψ(y), ∇ψ(y) ], [ ∇ψ(y)^T, 0 ]]              ((dim+1)×(dim+1), batched)
```

Backward for output `y` with cotangent ȳ: solve `J_z^T μ = [ȳ; 0]`, then `θ̄ = ∂⟨F, −μ⟩/∂θ|_{z fixed}` (one `torch.autograd.grad` on the params) and `x̄ = μ_y`. **Never unrolled** (spec §4.1); the unrolled twin exists only as a test reference. Downstream quantities are ordinary torch ops on `y`: `d = y − x`, `n = ∇ψ(y)/‖∇ψ(y)‖` — autograd composes them with the custom `Function`, so n's direct θ-dependence and curvature effects are handled exactly.

Eikonal shortcut (`near_eikonal=True` backends): `d = −ψ(x)∇ψ(x)/‖∇ψ(x)‖²` fully in autograd, with a runtime check `max |‖∇ψ‖ − 1| < 1e-9` in the band (fall back to Newton + warn otherwise).

`distance_torch(oracle, x_np) -> (d_t, n_t, ok)` picks the path; `distance_numpy` detaches. Hessians via batched double-grad (`torch.autograd.grad(g[:, i].sum(), y, create_graph=True)` per axis — dim ≤ 3 for geometry, fine for the prototype).

`admissibility(oracle, band_pts)`: run projection; report `eps_inf = max |ψ(y)|` over converged points (projection residual), `c0 = min ‖∇ψ‖` over band, `d_hausdorff_bound = eps_inf / min(1, c0)`, `newton_ok_frac`. Add `warn_refinement_plateau(diag, h, p)` → `warnings.warn` when `d_hausdorff_bound > h**(p+1)` (the ε∞ ~ h^(k+1) rule, spec §4.1).

**Steps:**

- [x] **Step 1: Failing tests** (append; still `tier4`)

```python
from diffsim.geometry.project import distance_numpy, distance_torch, closest_point
from diffsim.geometry.oracle import admissibility

def test_projection_lands_on_boundary():
    blend = Union(Sphere((0.4, 0.5), 0.2), Sphere((0.6, 0.5), 0.2), k=0.05)
    for oracle in (Sphere((0.5, 0.5, 0.5), 0.3), blend):
        dim = 3 if oracle is not blend else 2
        rng = np.random.default_rng(3)
        x = 0.5 + 0.25 * rng.uniform(-1, 1, (100, dim))
        d, n, ok = distance_numpy(oracle, x)
        assert ok.all()
        assert np.abs(oracle.classify(x + d)).max() < 1e-11        # y on the zero set
        assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-12)

def test_projection_matches_analytic_sphere():
    s = Sphere((0.5, 0.5), 0.3)
    x = np.array([[0.9, 0.5], [0.5, 0.65], [0.2, 0.2]])
    d, n, ok = distance_numpy(s, x)
    r = np.linalg.norm(x - 0.5, axis=1)
    d_exact = (0.3 - r)[:, None] * (x - 0.5) / r[:, None]
    assert np.allclose(d, d_exact, atol=1e-11)
    assert np.allclose(n, (x - 0.5) / r[:, None], atol=1e-11)

def test_ift_vs_unrolled_gradient():
    """Tier-2 VJP #3 three-way leg: IFT backward == unrolled autograd == FD."""
    def make(requires):
        o = Union(Sphere((0.4, 0.5), 0.2), Sphere((0.6, 0.5), 0.2), k=0.05)
        for p in o.params:
            p.requires_grad_(requires)
        return o
    x = np.array([[0.52, 0.71], [0.30, 0.44]])
    o = make(True)
    d_t, n_t, ok = distance_torch(o, x)                     # IFT path
    L = (d_t * torch.tensor([[1.0, 2.0], [3.0, -1.0]], dtype=torch.float64)).sum() \
        + (n_t * 0.3).sum()
    g_ift = torch.autograd.grad(L, o.params)
    # unrolled twin: fixed 40 iterations with create_graph
    o2 = make(True)
    xt = torch.as_tensor(x, dtype=torch.float64)
    y = xt.clone()
    for _ in range(40):
        yv = y.detach().requires_grad_(True)
        psi = o2.psi(yv)
        (g,) = torch.autograd.grad(psi.sum(), yv, create_graph=True)
        y = y - (o2.psi(y)[:, None] if False else psi[:, None]) * g / (g * g).sum(1, keepdim=True)
    # NOTE to implementer: write the unrolled loop cleanly (recompute psi/grad of the
    # CURRENT y each iteration with create_graph=True); the goal is graph-through-iteration.
    d2 = y - xt
    g2y = torch.autograd.grad(o2.psi(y).sum(), y, create_graph=True)[0]
    n2 = g2y / g2y.norm(dim=1, keepdim=True)
    L2 = (d2 * torch.tensor([[1.0, 2.0], [3.0, -1.0]], dtype=torch.float64)).sum() \
         + (n2 * 0.3).sum()
    g_unrolled = torch.autograd.grad(L2, o2.params)
    for a, b in zip(g_ift, g_unrolled):
        assert torch.allclose(a, b, atol=1e-9), (a, b)

def test_admissibility_sphere():
    s = Sphere((0.5, 0.5, 0.5), 0.3)
    band = 0.5 + 0.31 * np.random.default_rng(1).uniform(-1, 1, (200, 3))
    diag = admissibility(s, band)
    assert diag["newton_ok_frac"] == 1.0
    assert diag["eps_inf"] < 1e-11 and abs(diag["c0"] - 1.0) < 1e-12
```

- [x] **Step 2: Verify failure, then implement `project.py`.** Core shape:

```python
class _ClosestPoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, oracle_ref, *params):
        with torch.no_grad():
            y, s, ok = _newton_iterate(oracle_ref, x)     # cap 50, tol 1e-13
        ctx.oracle_ref, ctx.save = oracle_ref, (x, y, s, ok)
        return y, ok

    @staticmethod
    def backward(ctx, ybar, _okbar):
        oracle, (x, y, s, ok) = ctx.oracle_ref, ctx.save
        # Rebuild grad/Hessian graph at the converged y (fresh leaf)
        yv = y.detach().requires_grad_(True)
        psi = oracle.psi(yv)
        (g,) = torch.autograd.grad(psi.sum(), yv, create_graph=True)
        H = torch.stack([torch.autograd.grad(g[:, i].sum(), yv, create_graph=True)[0]
                         for i in range(y.shape[1])], dim=1)      # [N, dim, dim]
        Jz = _assemble_jz(H, g, s)                                # [N, dim+1, dim+1]
        rhs = torch.cat([ybar, torch.zeros_like(psi)[:, None]], 1)
        mu = torch.linalg.solve(Jz.transpose(1, 2), rhs[..., None])[..., 0]
        mu_y, mu_s = mu[:, :-1], mu[:, -1]
        # theta_bar = d<F, -mu>/d theta at fixed (y, s):
        F = torch.cat([yv - x + s[:, None] * g, psi[:, None]], 1)
        grads = torch.autograd.grad((F * (-mu)).sum(), list(oracle.params),
                                    retain_graph=False, allow_unused=True)
        xbar = mu_y                                              # dF/dx = [-I; 0]
        return (xbar, None, *grads)
```

Zero out rows where `ok == False` (mask ȳ) and surface the mask. `distance_torch` composes: `y, ok = closest_point(...)`; `d = y − x`; `n` from a fresh autograd `∇ψ(y)` normalized (this consumes the custom Function's backward for the y-dependence and the params' direct dependence together).

- [x] **Step 3: Run new tests + full suite + commit**

```bash
git add -A && git commit -m "feat: Newton closest-point projection with IFT adjoint (Tier-2 VJP 3)"
```

---

### Task 4: Face basis tables + second derivatives

**Files:**
- Modify: `src/diffsim/mesh/basis.py` (add `lagrange_1d_d2(p, x) -> np.ndarray[p+1]`; p1 → zeros, p2 → the constant second derivatives)
- Create: `src/diffsim/mesh/faces.py`
- Modify: `src/diffsim/assembly/femelm.py` (add `fe_d2N_s(d2Ntab, fe, a, i, j, d2scale)` accessor: `d2Ntab[fe.q, a, i, j] * d2scale`, `d2scale = (2/h)²` — used by the Neumann kernels which index flattened per-face tables, see Task 8)
- Test: `tests/test_face_tables.py`

**Construction:** face `f = 2*ax + side`; the fixed axis carries `ξ_ax = −1` (side 0) or `+1` (side 1); the remaining dim−1 axes carry the 1D Gauss lattice (x-fastest among the tangent axes, same reversed-itertools idiom as `basis_tables`). Volume basis functions evaluated at these face points: `N[f, q, a] = Π_d N1d(ξ_d)`, `dN[f, q, a, i]` replaces factor i with its 1D derivative, `d2N[f, q, a, i, j]` replaces factors i and j (i == j uses the 1D second derivative). `w[q] = Π (dim−1 Gauss weights)`. All reference-element; physical scaling documented in the header docblock.

**Steps:**

- [ ] **Step 1: Failing tests** (`tests/test_face_tables.py`, `pytestmark = pytest.mark.tier2`)

```python
import numpy as np
import pytest
from diffsim.mesh.faces import face_tables

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("p", [1, 2])
def test_face_tables_pou_and_quadrature(dim, p):
    ft = face_tables(p, dim)
    nf = 2 * dim
    assert ft.N.shape == (nf, ft.nqf, (p + 1) ** dim)
    assert np.allclose(ft.N.sum(axis=2), 1.0, atol=1e-14)          # PoU on every face
    assert np.allclose(ft.dN.sum(axis=2), 0.0, atol=1e-13)
    assert abs(ft.w.sum() - 2.0 ** (dim - 1)) < 1e-13              # reference face measure

@pytest.mark.parametrize("dim", [2, 3])
def test_face_normal_axis_values(dim):
    # On face f, the fixed axis coordinate is +-1: basis values must match the
    # 1D trace (nodes on the face carry all the weight for p1).
    ft = face_tables(1, dim)
    for ax in range(dim):
        for side, xi in ((0, -1.0), (1, 1.0)):
            f = 2 * ax + side
            # sum of |N| over nodes NOT on that face is zero for p1
            from diffsim.mesh.nodes import _local_offsets
            offs = _local_offsets(1, dim)
            off_face = offs[:, ax] != (0 if side == 0 else 1)
            assert np.abs(ft.N[f][:, off_face]).max() < 1e-14

def test_d2N_exact_for_quadratic():
    # p2, dim=2: interpolate u = xi^2 * eta + 3 eta^2; check Hessian at face GPs.
    ft = face_tables(2, 2)
    from diffsim.mesh.nodes import _local_offsets
    offs = _local_offsets(2, 2)
    xi_n = offs.astype(float) - 1.0                                 # nodes at -1, 0, 1
    ue = xi_n[:, 0] ** 2 * xi_n[:, 1] + 3.0 * xi_n[:, 1] ** 2
    for f in range(4):
        # reconstruct face GP coords to evaluate the exact Hessian
        # H = [[2 eta, 2 xi], [2 xi, 6]]
        xg = np.einsum("qa,ad->qd", ft.N[f], xi_n)
        H_exact = np.empty((ft.nqf, 2, 2))
        H_exact[:, 0, 0] = 2 * xg[:, 1]; H_exact[:, 0, 1] = 2 * xg[:, 0]
        H_exact[:, 1, 0] = 2 * xg[:, 0]; H_exact[:, 1, 1] = 6.0
        H_num = np.einsum("qaij,a->qij", ft.d2N[f], ue)
        assert np.allclose(H_num, H_exact, atol=1e-12)

def test_p1_d2N_zero():
    assert np.abs(face_tables(1, 3).d2N).max() == 0.0
```

- [ ] **Step 2: Verify failure, implement, run, full suite, commit**

```bash
git add -A && git commit -m "feat: face basis tables with second derivatives (N/dN/d2N at face GPs)"
```

---

### Task 5: λ-classification, surrogate extraction, epoch geometry cache + π/4 lock

**Files:**
- Create: `src/diffsim/sbm/__init__.py`, `src/diffsim/sbm/surrogate.py`
- Test: `tests/test_sbm_surrogate.py`

**Semantics (spec §4.2):**

- `classify_lambda(tree, oracle, lam, domain="inside", n1=3)`: sample ψ on the per-element `n1^dim` tensor Gauss lattice (reuse the `gauss_1d(2)` points mapped to each element); `frac_in = (#on-domain-side)/n1^dim` where the domain side is ψ<0 (`"inside"`) or ψ>0 (`"outside"`). Retain elements with `frac_in ≥ lam` (Interior have frac 1 and are always retained; frac 0 dropped). `lam = 1.0` → only fully-interior elements (flux-accurate surrogate strictly inside Ω); `lam = 0.5` → optimal surrogate. Returns the retained `Octree` + `frac_in` for diagnostics.
- `extract_surrogate(tree_retained)`: a face of a retained element is surrogate iff **no retained element lies across it AND it is not on the unit-cube boundary**. Probe just outside each face center via `LeafLookup.find`; validate with the `2^(dim−1)` sub-face probes — if the sub-probes disagree (partially exposed face), raise `ValueError("partially exposed surrogate face; refine the narrowband uniformly (M1a invariant)")`. Returns `SurrogateFaces(elem, face)` sorted by (elem, face) for determinism.
- `face_gauss_points(tree, sf, ftab)`: physical coords of the `nqf` face Gauss points per surrogate face (fixed axis at the face plane, tangent axes at Gauss ξ mapped by `lo + (ξ+1)h/2`), flat `[Nf*nqf, dim]` in (face, q) order — kernels index `fi*nqf + q`.
- `GeometryData.evaluate(oracle, tree, sf, ftab, domain="inside")`: one batched oracle call (spec §4.2 step 4): `d, n_grad, ok = oracle.distance_vector(xq)`; `n = s·n_grad` with `s = +1/−1` per the domain flag (n always out of Ω); `corr[i] = ñ_face(i) · n[i]`; assert `ok.all()` (raise with the admissibility report otherwise); everything FP64, immutable per epoch.

Narrowband helper for tests: `narrowband_refine(oracle, max_level, dim, pad=1.0)` — `build_adaptive` with predicate `|ψ(center)| ≤ pad·√dim·h` (h arrives as `[N,1]`, broadcast-ready — the documented M0 signature), then `balance2to1`, then `classify_lambda`. This produces uniform-level narrowbands, satisfying the M1a invariant by construction.

**Steps:**

- [ ] **Step 1: Failing tests** (`tests/test_sbm_surrogate.py`, `pytestmark = pytest.mark.tier4`)

```python
import numpy as np
import pytest
from diffsim.octree.build import build_uniform
from diffsim.geometry.csg import Sphere
from diffsim.mesh.faces import face_tables
from diffsim.sbm.surrogate import (classify_lambda, extract_surrogate,
                                   face_gauss_points, GeometryData)

def _circle_case(level, lam):
    tree = build_uniform(level, dim=2)
    oracle = Sphere((0.5, 0.5), 0.3)              # interior problem: domain = disk
    ret, frac = classify_lambda(tree, oracle, lam)
    sf = extract_surrogate(ret)
    return oracle, ret, sf

def test_lambda_retention_ordering():
    tree = build_uniform(5, dim=2)
    oracle = Sphere((0.5, 0.5), 0.3)
    n_all = [len(classify_lambda(tree, oracle, lam)[0]) for lam in (0.25, 0.5, 1.0)]
    assert n_all[0] >= n_all[1] >= n_all[2] > 0   # retention shrinks with lambda
    ret1, frac1 = classify_lambda(tree, oracle, 1.0)
    assert (frac1 == 1.0).all()                    # lam=1: fully-interior only

def test_surrogate_faces_enclose_domain():
    oracle, ret, sf = _circle_case(5, 0.5)
    assert len(sf.elem) > 0
    ftab = face_tables(1, 2)
    xq = face_gauss_points(ret, sf, ftab)
    # face GPs sit near the true boundary: |psi| < 2h
    assert np.abs(oracle.classify(xq)).max() < 2 * ret.h().max()

def test_geometry_data_circle():
    oracle, ret, sf = _circle_case(6, 1.0)
    ftab = face_tables(1, 2)
    geo = GeometryData.evaluate(oracle, ret, sf, ftab)
    # d points from surrogate GP to the circle: x + d lies on it
    r = np.linalg.norm(geo.xq + geo.d - 0.5, axis=1)
    assert np.abs(r - 0.3).max() < 1e-11
    assert (geo.corr > 0.0).all()                  # outward alignment
    assert np.allclose(np.linalg.norm(geo.n, axis=1), 1.0, atol=1e-12)

@pytest.mark.parametrize("level", [6, 7])
def test_pi_over_4_area_correction_lock(level):
    """LOCKED (spec S4.2 step 5): staircase perimeter of a circle -> 8r; the
    (n_bar . n) corrected perimeter -> 2 pi r. Ratio -> pi/4. Without the
    correction, any surrogate-boundary flux integral inherits the 4/pi error."""
    oracle, ret, sf = _circle_case(level, 1.0)
    ftab = face_tables(1, 2)
    geo = GeometryData.evaluate(oracle, ret, sf, ftab)
    h = ret.h()[sf.elem]
    dS = np.repeat(h, ftab.nqf) * np.tile(ftab.w, len(sf.elem)) / 2.0  # (h/2)^(dim-1) * w
    P_stair = dS.sum()
    P_corr = (geo.corr * dS).sum()
    assert abs(P_corr - 2 * np.pi * 0.3) < 0.15 * 2 * np.pi * 0.3 / 2 ** (level - 5)
    assert abs(P_corr / P_stair - np.pi / 4) < 0.03    # the pi/4 signature

def test_partially_exposed_face_raises():
    # Handcraft: refine ONE boundary-band element so its dropped neighbor is
    # coarser -> its sibling face is partially exposed. Expect ValueError.
    from diffsim.octree.build import refine_elements
    tree = build_uniform(4, dim=2)
    oracle = Sphere((0.5, 0.5), 0.3)
    ret, _ = classify_lambda(tree, oracle, 1.0)
    # refine exactly one retained element that touches the surrogate boundary
    sf = extract_surrogate(ret)
    mask = np.zeros(len(ret), bool); mask[sf.elem[0]] = True
    ret2 = refine_elements(ret, mask)
    with pytest.raises(ValueError, match="partially exposed"):
        extract_surrogate(ret2)

def test_exterior_configuration():
    # domain = square minus disk (the M1b configuration), via the domain flag
    tree = build_uniform(5, dim=2)
    oracle = Sphere((0.5, 0.5), 0.25)
    ret, _ = classify_lambda(tree, oracle, 1.0, domain="outside")
    sf = extract_surrogate(ret)
    ftab = face_tables(1, 2)
    geo = GeometryData.evaluate(oracle, ret, sf, ftab, domain="outside")
    # normals point INTO the disk (outward from the domain)
    to_center = (0.5 - geo.xq) / np.linalg.norm(0.5 - geo.xq, axis=1, keepdims=True)
    assert (np.einsum("id,id->i", geo.n, to_center) > 0.9).all()
```

- [ ] **Step 2: Verify failure, implement `surrogate.py`.** Notes: probes just outside face f of element e: anchor-grid point `center(e) + off_f * (size_e/2 + 1)` (the `face_neighbors` idiom); sub-probes shift the tangent axes by ±size_e/4. Outer-boundary detection: probe coordinate out of `[0, 2^lmax)` on a non-periodic axis. Reuse `LeafLookup`, `face_offsets`. Determinism: pure integer/array ops, single-threaded host code.

- [ ] **Step 3: Run + full suite + commit**

```bash
git add -A && git commit -m "feat: lambda classification, surrogate extraction, epoch geometry cache + pi/4 lock"
```

---

### Task 6: SBM Dirichlet Poisson (assembled path) + P4 keystone patch test

**Files:**
- Modify: `src/diffsim/assembly/operators.py` (add `CSROperator`)
- Create: `src/diffsim/sbm/poisson.py`
- Test: `tests/test_sbm_poisson.py`

**Weak form (Main & Scovazzi 2018; κ-scaled; Su = u + ∇u·d, Sw = w + ∇w·d, ḡ = g(x̃+d)):**

```
∫_Ω̃ κ∇w·∇u dV  − ∫_Γ̃D κ w (∇u·ñ) dS  − ∫_Γ̃D κ (∇w·ñ)(Su) dS  + ∫_Γ̃D (ακ/h)(Sw)(Su) dS
   = ∫_Ω̃ w f dV  − ∫_Γ̃D κ (∇w·ñ) ḡ dS  + ∫_Γ̃D (ακ/h)(Sw) ḡ dS
```

Sanity limits (assert in tests): d = 0 recovers classical Nitsche; exact linear u makes Su − ḡ ≡ 0 and the consistency term equal the true flux ⇒ **linear patch is machine-exact for any α, λ, and geometry rotation** — the P4 keystone. The −κ(∇w·ñ)(∇u·d) piece is unpaired ⇒ nonsymmetric ⇒ bicgstab.

**Face element-matrix kernel** (factory `make_sbm_dirichlet_Ae(nbf, nqf, dim)`, key `("sbm_dir_Ae", nbf, nqf, dim)`):

```python
@wp.kernel
def sbm_dir_Ae(felem: wp.array(dtype=wp.int32),   # [Nf] bin-local element row
               fface: wp.array(dtype=wp.int32),   # [Nf] face id
               h: wp.array(dtype=wp.float64),     # per-element (bin) h
               Nf: wp.array3d(dtype=wp.float64),  # [2*dim, nqf, nbf]
               dNf: wp.array4d(dtype=wp.float64), # [2*dim, nqf, nbf, dim] (reference)
               wf: wp.array(dtype=wp.float64),    # [nqf]
               dvec: wp.array2d(dtype=wp.float64),# [Nf*nqf, dim]
               alpha: wp.float64, kappa: wp.float64,
               Ae: wp.array3d(dtype=wp.float64)): # [Nf, nbf, nbf]
    fi = wp.tid()
    e = felem[fi]; f = fface[fi]
    he = h[e]; half = he * wp.float64(0.5)
    jacS = wp.float64(1.0)
    for _ in range(dim - 1):
        jacS = jacS * half                         # dS = w * (h/2)^(dim-1)
    dscale = wp.float64(2.0) / he
    ax = f / 2                                     # integer div
    sgn = wp.float64(1.0)
    if f % 2 == 0:
        sgn = wp.float64(-1.0)                     # n_tilde = sgn * e_ax
    for q in range(nqf):
        dS = wf[q] * jacS
        for a in range(nbf):
            Na = Nf[f, q, a]
            gna = sgn * dNf[f, q, a, ax] * dscale              # grad(N_a) . n_tilde
            Sa = Na
            for dd in range(dim):
                Sa += dNf[f, q, a, dd] * dscale * dvec[fi * nqf + q, dd]
            for b in range(nbf):
                Nb = Nf[f, q, b]
                gnb = sgn * dNf[f, q, b, ax] * dscale
                Sb = Nb
                for dd in range(dim):
                    Sb += dNf[f, q, b, dd] * dscale * dvec[fi * nqf + q, dd]
                Ae[fi, a, b] += kappa * (-Na * gnb - gna * Sb
                                         + alpha / he * Sa * Sb) * dS
```

RHS kernel `make_sbm_dirichlet_be(...)` (key `"sbm_dir_be"`): `be_full[conn[e,a]] += κ(−gna + (α/h)·Sa)·ḡ_q·dS` (atomic).

**`SBMPoisson` (Dirichlet part; Neumann arrives Task 8):**

- Constructor takes `(dm, geo, sf, g_fn=None, kappa, alpha)` where `g_fn(x np [M,dim]) -> [M]` evaluates Dirichlet data (used at the **mapped** points `geo.xq + geo.d`). Mixed-p: split `sf` by `mesh.p_elem[sf.elem]`, map global elem → bin-local row (invert `mesh.bins[pv]`), one `face_tables(pv, dim)` + one kernel launch per bin.
- `.assemble()`: volume CSR at κ=1 via existing `assemble_csr(dm)` (unconstrained K then T-congruence is what it already does), face COO from `Ae` per bin at κ=1 scattered with `conn_of[pv]`, RHS from load kernel (f) + face `be` at κ=1; returns κ-scaled `A = κ(K_vol + K_faceD)`, `b = b_f + κ b_gD` **plus the κ=1 pieces in `meta`** (`meta.A1`, `meta.bg1` — the κ-gradient in Task 10 needs `∂R/∂κ = A1·u − bg1`). Outer strong Dirichlet (exterior configs): row-replacement on the constrained CSR (`A[i,:] = e_i`, `b[i] = g_outer(x_i)`) for free nodes with `mesh.boundary_nodes` — nonsymmetric solver anyway.
- `.solve(tol=1e-12)`: `CSROperator` on device + `bicgstab` with Jacobi `diag=A.diagonal()`; returns `u_all = T @ u_free`. Assert converged.
- `CSROperator` (in `operators.py`): wraps a scipy CSR on device (`_csr_to_device`), `.matvec` via `csr_spmv`, `.matvec_numpy`, `.n_free`, `.device` — the operator protocol shape (M0 deferred item).

**Steps:**

- [ ] **Step 1: Failing tests** (`tests/test_sbm_poisson.py`, `pytestmark = pytest.mark.tier5`)

```python
import numpy as np
import pytest
import torch
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.geometry.csg import Sphere, Box
from diffsim.sbm.surrogate import classify_lambda, extract_surrogate, GeometryData
from diffsim.sbm.poisson import SBMPoisson


def sbm_setup(oracle, level, p, lam, dim, device, domain="inside"):
    tree = build_uniform(level, dim=dim)
    ret, _ = classify_lambda(tree, oracle, lam, domain=domain)
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=dim), device)
    geo = GeometryData.evaluate(oracle, ret, sf, face_tables(p, dim), domain=domain)
    return dm, geo, sf


LIN = {2: (0.7, np.array([1.3, -0.4])), 3: (0.7, np.array([1.3, -0.4, 0.9]))}


def _patch(oracle, level, p, lam, dim, device, atol=1e-11, domain="inside"):
    c0, cv = LIN[dim]
    u_lin = lambda x: c0 + x @ cv
    dm, geo, sf = sbm_setup(oracle, level, p, lam, dim, device, domain=domain)
    prob = SBMPoisson(dm, geo, sf, g_fn=u_lin, kappa=1.0, alpha=10.0)
    u = prob.solve(f_fn=lambda x: np.zeros(len(x)))
    err = np.abs(u - u_lin(dm.mesh.node_coords)).max()
    assert err < atol, err


@pytest.mark.parametrize("lam", [0.5, 1.0])
@pytest.mark.parametrize("p", [1, 2])
def test_P4_patch_circle_2d(lam, p, device):
    _patch(Sphere((0.5, 0.5), 0.3), 5, p, lam, 2, device)


@pytest.mark.parametrize("lam", [0.5, 1.0])
def test_P4_patch_rotated_box_2d(lam, device):
    b = Box((0.5, 0.5), (0.23, 0.17), rotation=torch.tensor(0.4, dtype=torch.float64))
    _patch(b, 5, 1, lam, 2, device)


@pytest.mark.parametrize("lam", [0.5, 1.0])
def test_P4_patch_sphere_3d(lam, device):
    _patch(Sphere((0.5, 0.5, 0.5), 0.32), 4, 1, lam, 3, device)


def test_P4_patch_rotated_box_3d(device):
    b = Box((0.5, 0.5, 0.5), (0.25, 0.18, 0.22),
            rotation=torch.tensor([0.3, 0.5, 0.2], dtype=torch.float64))
    _patch(b, 4, 1, 1.0, 3, device)


def test_P4_patch_exterior_with_outer_dirichlet(device):
    # square minus disk (the M1b configuration): SBM on the disk + strong outer
    # Dirichlet, domain = {psi > 0} via the flag
    _patch(Sphere((0.5, 0.5), 0.25), 5, 1, 1.0, 2, device, domain="outside")


def test_alpha_insensitivity_of_patch(device):
    # keystone property: exactness does not depend on the penalty
    c0, cv = LIN[2]
    u_lin = lambda x: c0 + x @ cv
    for alpha in (2.0, 10.0, 100.0):
        dm, geo, sf = sbm_setup(Sphere((0.5, 0.5), 0.3), 5, 1, 0.5, 2, device)
        prob = SBMPoisson(dm, geo, sf, g_fn=u_lin, alpha=alpha)
        u = prob.solve(f_fn=lambda x: np.zeros(len(x)))
        assert np.abs(u - u_lin(dm.mesh.node_coords)).max() < 1e-10


def test_operator_nonsymmetric_documented(device):
    # the adjoint-consistency shift term breaks symmetry: assert it, so a future
    # accidental "cg" swap fails loudly
    dm, geo, sf = sbm_setup(Sphere((0.5, 0.5), 0.3), 4, 1, 0.5, 2, device)
    A, b, meta = SBMPoisson(dm, geo, sf, g_fn=lambda x: x[:, 0]).assemble(
        f_fn=lambda x: np.zeros(len(x)))
    asym = abs(A - A.T).max()
    assert asym > 1e-12
```

- [ ] **Step 2: Verify failure, implement.** Care points: (i) the Dirichlet data is evaluated at the **mapped** points `geo.xq + geo.d` — `g_fn` receives those, not `geo.xq`; (ii) face `Ae` scatter uses the **unconstrained** node ids then the same T-congruence as the volume part (single COO concat before `tocsr`, matching `assemble_csr`'s pattern — factor its bin loop so `SBMPoisson` can reuse the volume triplets without re-assembling); (iii) `bicgstab` tol 1e-14 for patch tests (tests pass tighter tol per spec §9.2 class 2); (iv) if the level-5 circle at λ=0.5 exposes a partially-exposed face (possible at coarse levels), bump the test level — the invariant error message tells you.

- [ ] **Step 3: Run + full suite + commit**

```bash
git add -A && git commit -m "feat: SBM Dirichlet Poisson (assembled path); P4 rotated-geometry patch keystone"
```

---

### Task 7: SBM Dirichlet MMS convergence orders

**Files:**
- Modify: `src/diffsim/physics/poisson.py` (add `l2_error_masked(dm, u_all, exact_fn, mask_fn)`: same kernel pattern with a per-GP FP64 mask array — key `("l2m", nbf, nqp, dim)`; mask = 1 where `oracle.classify(xq) < 0` — **error measured on Ω, not Ω̃** (spec §13.3.3 warning))
- Test: append to `tests/test_sbm_poisson.py`

**Configurations:**

1. **Interior disk (2D):** Ω = disk(c=(0.5,0.5), r=0.3), u* = sin(πx)sin(πy), f = 2π²u*, all boundary via SBM Dirichlet. Levels 4–7, p1 → order 2; levels 4–6, p2 → order 3. λ = 1.0 and one λ = 0.5 spot check (same order).
2. **Exterior square-minus-disk (2D, `domain="outside"` — the M1b configuration):** strong outer Dirichlet + SBM disk, same u*. Levels 4–7, p1.
3. **Interior sphere (3D):** u* = sin(πx)sin(πy)sin(πz), f = 3π²u*, levels 3–5, p1 → order 2 (one 3D confirmation; 2D carries the ladder per spec §11.1 "k=2 is the cheap verification dimension").
4. κ ≠ 1 spot check: config 1 at κ = 2.5, f = 2.5·2π²u* — identical errors to κ=1 within rtol 1e-12 (pure scaling).

Order asserted via least-squares slope of log(err) vs log(h) over the last 3 levels, within ±0.10 of target (2 or 3). Store the level-ladder errors in a dict for Task 11's baseline lock.

**Steps:**

- [ ] **Step 1: Failing tests** — the MMS ladder helper:

```python
def _mms_ladder(oracle, levels, p, lam, dim, device, u_fn, f_fn, kappa=1.0,
                domain="inside"):
    sgn = -1.0 if domain == "inside" else 1.0
    errs = []
    for lv in levels:
        dm, geo, sf = sbm_setup(oracle, lv, p, lam, dim, device, domain=domain)
        prob = SBMPoisson(dm, geo, sf, g_fn=u_fn, kappa=kappa)
        u = prob.solve(f_fn=f_fn, g_outer_fn=u_fn)
        from diffsim.physics.poisson import l2_error_masked
        errs.append(l2_error_masked(dm, u, u_fn,
                                    lambda x: sgn * oracle.classify(x) > 0))
    return np.array(errs)

def _slope(errs, levels):
    h = 2.0 ** -np.asarray(levels, float)
    k = min(3, len(errs))
    return np.polyfit(np.log(h[-k:]), np.log(errs[-k:]), 1)[0]

def test_mms_disk_p1_order2(device):
    u = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
    f = lambda x: 2 * np.pi ** 2 * u(x)
    errs = _mms_ladder(Sphere((0.5, 0.5), 0.3), [4, 5, 6, 7], 1, 1.0, 2, device, u, f)
    assert abs(_slope(errs, [4, 5, 6, 7]) - 2.0) < 0.10, errs

def test_mms_disk_p2_order3(device):
    u = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
    f = lambda x: 2 * np.pi ** 2 * u(x)
    errs = _mms_ladder(Sphere((0.5, 0.5), 0.3), [4, 5, 6], 2, 1.0, 2, device, u, f)
    assert abs(_slope(errs, [4, 5, 6]) - 3.0) < 0.10, errs
```

plus the exterior-2D, 3D-sphere, λ=0.5, and κ=2.5 variants per the configuration list.

- [ ] **Step 2: Verify failure (l2_error_masked missing), implement, run.** The 3D level-5 case runs on `cuda:0`; keep it as the largest problem in the suite (~1.5e5 elements retained — fine).

- [ ] **Step 3: Full suite + commit**

```bash
git add -A && git commit -m "test: SBM Dirichlet MMS orders (interior/exterior, p1/p2, 2D/3D, error on Omega)"
```

---

### Task 8: SBM Neumann — area correction + Hessian shift + §13.3.3 p2-band acceptance

**Files:**
- Modify: `src/diffsim/sbm/poisson.py` (Neumann face kernels + `surrogate_flux`)
- Test: `tests/test_sbm_neumann.py`

**Weak form (Atallah–Scovazzi 2020; S∇u = ∇u + H(u)·d is the shifted gradient, q̄_N = q_N(x̃+d) the prescribed true flux κ∇u·n∘M):** the true-boundary term −∫_Γ w q_N dS becomes, on the surrogate,

```
LHS += − ∫_Γ̃N κ w [ (S∇u)·ñ − (n·ñ)((S∇u)·n) ] dS        (u-dependent tangential correction)
RHS += + ∫_Γ̃N w (n·ñ) q̄_N dS                              (area-corrected mapped flux data)
```

Sanity limits (assert in tests): Γ̃ → Γ (n = ñ, d = 0) kills the LHS correction and recovers the classical Neumann term; **dropping (n·ñ) from the RHS inflates any flux-driven solution by the staircase ratio (the π/4 pathology — locked)**. For p1, H ≡ 0 truncates S∇u — the §13.1 representability failure; the p2 band restores it. The **hard rule** (spec §13.1): every shifted-Neumann quadrature point lies inside p2 cells — asserted in the band test.

Kernels: `make_sbm_neumann_Ae(nbf, nqf, dim)` (key `"sbm_neu_Ae"`) — inputs add `nvec [Nf*nqf, dim]`, `corr [Nf*nqf]`, `d2Nf [2*dim, nqf, nbf, dim, dim]`; per (q, b): `S∇u_b[i] = dN_b[i]·dscale + Σ_j d2N_b[i,j]·d2scale·d_j` with `d2scale = dscale²`, then `Ae[fi,a,b] −= κ·Na·(S∇u_b·ñ − corr·(S∇u_b·n))·dS`. RHS `make_sbm_neumann_be` (key `"sbm_neu_be"`): `be[conn[a]] += Na·corr·q̄_q·dS`.

`SBMPoisson` gains `neumann=(sf_N, geo_N, q_fn)` — a problem may carry Dirichlet faces, Neumann faces, or both (disjoint `SurrogateFaces` sets; the test splits a circle's faces is NOT needed — M1a configs use all-Dirichlet or all-Neumann-on-the-disk + strong outer Dirichlet for well-posedness).

`surrogate_flux(u_all) -> float`: ∫_Γ̃ κ(S∇u·n)(n·ñ) dS̃ — the area-corrected shifted estimate of the true-boundary flux ∫_Γ κ∇u·n dΓ (the "Nusselt" observable; kernel key `"sbm_flux"`).

**Acceptance test (§13.3.3):** exterior square-minus-disk (`domain="outside"`), u* = sin(πx)cos(πy) (nonzero flux on the disk), strong outer Dirichlet from u*, SBM **Neumann** on the disk with q_N = κ∇u*·n evaluated at mapped points. Measure `|surrogate_flux(u) − F*|` where F* = ∫_Γ κ∇u*·n dΓ (semi-analytic: 1D trapezoid quadrature over the circle at 4096 points, computed in the test).

1. **π/4 lock (solve level):** with `corr` forced to 1 (test-only flag `_area_correction=False`), the flux error stalls O(1); with correction, it converges.
2. **p1 vs p2 band:** uniform-level mesh, `p_elem = 2` for elements within `n_layers·h` of the disk (|ψ(center)| test), 1 elsewhere; one-knob is trivially satisfied (single level). p1-everywhere flux order degraded (≈1); band n_layers=2 restores order ≈2 (±0.10 both).
3. **Layer sweep:** n_layers ∈ {1, 2, 3} — restored rate insensitive (pairwise order difference < 0.10) once every shifted-Neumann GP is in p2 cells (assert `(mesh.p_elem[sf_N.elem] == 2).all()`).

**Steps:**

- [ ] **Step 1: Failing tests** (`tests/test_sbm_neumann.py`, `tier5`; helpers shared from `test_sbm_poisson` via a small `tests/helpers/sbm_cases.py` if imports get awkward)
- [ ] **Step 2: Verify failure; implement kernels + `surrogate_flux`; wire `neumann=` into assemble.** Care points: (i) mixed-p face binning must route each face to its element's p-bin tables (p2 band ⇒ all Neumann faces in the p2 bin — assert); (ii) `d2N` physical scaling is `(2/h)²`; (iii) Neumann-only-on-disk + outer strong Dirichlet keeps the system nonsingular.
- [ ] **Step 3: Run (expect the three acceptance properties), full suite, commit**

```bash
git add -A && git commit -m "feat: SBM Neumann (area correction + Hessian shift); S13.3.3 p2-band acceptance"
```

---

### Task 9: TriMesh (STL) + GridSDF backends + cross-backend oracle suite

**Files:**
- Create: `src/diffsim/geometry/gridsdf.py`, `src/diffsim/geometry/trimesh.py`
- Test: append to `tests/test_geometry_oracles.py` (tier4 unit) and `tests/test_sbm_poisson.py` (tier5 cross-backend)

(NeuralSDF is deferred to M1c with the hero demo; the oracle contract and the IFT projection it rides on are fully exercised here by the two non-analytic backends.)

**GridSDF:** voxel corner values `V [n+1]^dim` (torch FP64, the differentiable parameter = level-set shape optimization, spec §4.1) on the unit cube; `psi(x)` = multilinear interpolation written with explicit gather + lerp (autograd-clean, exact FP64 — not `grid_sample`). `from_oracle(oracle, n)` sampler. `near_eikonal=False` (interpolation of an SDF is only approximately eikonal) → Newton projection. Note: ψ is C⁰ across cell faces — Newton on a multilinear field converges within cells; cap+mask handles edge cases, and the admissibility report quantifies ε̂∞ ~ Δx².

**TriMeshOracle:** vertices `verts` (torch FP64 [Nv,3], the differentiable parameter), triangles `tris` (int32 [Nt,3]). Forward queries via `wp.Mesh` (fp32) for **candidate location only**: `wp.mesh_query_point_sign_winding_number` → (sign, nearest triangle index). FP64 refinement in torch: exact closest-point-on-triangle for the found triangle (clamped barycentric regions; region selected from the FP64 computation, then the smooth per-region formula keeps autograd exactness — piecewise-smooth per spec §1.5). `psi = sign·‖x − y‖`; `distance_vector` overridden (no Newton): `d = y − x`, `n = sign-corrected triangle/edge/vertex normal via normalize(x−y)·(−sign)` — document: n from the closest-point direction, exact for points off the surface. `icosphere(n_sub, center, radius)` generator (subdivided icosahedron, no file I/O) + `load_stl(path)` (binary STL via `struct`, no new dependency). `params = [verts]`.
2D note: TriMesh is 3D-only (document); the cross-backend suite runs it on the 3D sphere case.

**Cross-backend verification suite (the oracle suite; three of M1's four backends here — NeuralSDF joins the same parametrized test in M1c):** one parametrized tier5 test running the interior-sphere SBM Dirichlet solve per backend and comparing to the AnalyticCSG baseline:

```python
@pytest.mark.parametrize("backend", ["csg", "trimesh", "grid"])
def test_cross_backend_sphere_mms(backend, device):
    dim, level, r = 3, 4, 0.32
    target = Sphere((0.5,) * 3, r)
    oracle = {
        "csg": lambda: target,
        "trimesh": lambda: icosphere(3, (0.5,) * 3, r),
        "grid": lambda: GridSDF.from_oracle(target, n=128),
    }[backend]()
    u = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]) * np.sin(np.pi * x[:, 2])
    f = lambda x: 3 * np.pi ** 2 * u(x)
    dm, geo, sf = sbm_setup(oracle, level, 1, 1.0, dim, device)
    diag = admissibility(oracle, geo.xq)
    prob = SBMPoisson(dm, geo, sf, g_fn=u)
    uh = prob.solve(f_fn=f)
    err = l2_error_masked(dm, uh, u, lambda x: target.classify(x) < 0)
    baseline = ERR_CSG_L4          # captured from the csg run / Task 11 baseline file
    # discretization error dominates once the geometry error eps_inf << h^2:
    assert err < 2.0 * baseline + 10.0 * diag["eps_inf"]
```

plus tier4 unit tests per backend: projection/closest-point accuracy vs the analytic sphere (`grid`: < 5·Δx²; `trimesh`: < faceting sagitta bound `r(1−cos θ)` computed from the subdivision level), sign correctness inside/outside, admissibility report sanity, and the **P4-with-approximate-geometry check**: linear patch error < 50·ε̂∞ (validates that the diagnostics bound what the patch test sees — analytic backends stay at 1e-11).

**Steps:**

- [ ] **Step 1: Failing unit tests per backend** (as specified above)
- [ ] **Step 2: Implement the two backends** (each with a header docblock citing spec §4.1's backend table and the differentiable-parameter column)
- [ ] **Step 3: Cross-backend suite; run; full suite; commit.**

```bash
git add -A && git commit -m "feat: TriMesh(STL) + GridSDF backends + cross-backend oracle suite"
```

---

### Task 10: Adjoint gradients — Tier-2 VJP #1, ∂R/∂θ tape sweep, torch twin, three-way gate

**Files:**
- Create: `src/diffsim/sbm/adjoint.py`, `src/diffsim/sbm/reference.py`, `src/diffsim/mesh/pointeval.py`
- Modify: `src/diffsim/sbm/poisson.py` (face **residual** kernels for the tape sweep)
- Test: `tests/test_ad_gradients.py`

**The gradient chain (spec §4.3, §5.2):** within an epoch, R(u; θ, κ) = A(θ, κ)u − b(θ, κ) and geometry enters only through the cached face data. For QoI J(u):

```
dJ/dθ = −λᵀ (∂R/∂θ),   Aᵀ λ = (dJ/du)ᵀ            (Tier-2 VJP #1: adjoint solve on A.T CSR)
∂R/∂θ:  Warp tape over pure face-RESIDUAL kernels (inputs u_full, dvec, gbar [, nvec, corr, qbar])
        seeded with λ_full = T λ_free  →  cotangents (d̄, ḡ̄ [, n̄, corr̄, q̄])
        →  torch backward through {d, n} = distance_torch(oracle, xq),  corr = ñ·n,
           ḡ = g(x̃+d), q̄ = q(x̃+d)  →  θ̄ on oracle.params.
dJ/dκ = −λᵀ (A₁u − b_g1)                            (κ-linearity; meta from Task 6)
```

Face residual kernels (keys `"sbm_dir_res"`, `"sbm_neu_res"`): compute the face part of Au − b directly — Dirichlet per (fi, q, a): `r[conn[a]] += κ(−Na(∇u·ñ) − gna(Su − ḡ) + (α/h)Sa(Su − ḡ))dS` with `Su = Σ_b N_b u_b + Σ_b (∇N_b·d) u_b` — pure, all data via arrays, `dvec`/`gbar` marked `requires_grad=True` under `wp.Tape`; `tape.backward(grads={r_full: λ_full_wp})` yields the cotangents. Row-replaced outer-Dirichlet rows: zero λ_full there before seeding (their residual rows are θ-independent). Dirichlet-data direct dependence: ḡ = g(x̃+d) is part of the torch chain, so g's θ-dependence via d is captured; for the *shape-inverse* QoI g is θ-independent (given data).

**Torch dense twin (`sbm/reference.py`, spec §16.2 "clear path"):** `solve_dense_torch(mesh, cons, oracle, g_fn_torch, f_fn_torch, kappa, alpha, lam_faces) -> (J, u)` — same retained mesh, same face list, basis tables as torch constants; volume + face element matrices by `einsum`; T as a dense torch matrix; `torch.linalg.solve`; QoI in torch. End-to-end autograd through `distance_torch` (IFT). This is simultaneously (a) the "unrolled tape" leg of the three-way check and (b) the readable reference documentation of the SBM weak form. Tiny meshes only (≤ ~1500 DOFs); docstring says so.

**QoIs:** `volume_qoi` (J = ∫_Ω̃ u dV; dJ/du_free = Tᵀ(M·1) via the load kernel with f≡1); `probe_qoi(mesh, T, pts, targets)` (J = Σ(u(xᵢ)−u*ᵢ)²; weights from `point_eval_weights` — host CSR of Lagrange evaluation weights per probe, built with the `FieldEvaluator` logic moved into `src/diffsim/mesh/pointeval.py`; the tests/helpers version stays as-is).

**Steps:**

- [ ] **Step 1: Failing tests** (`tests/test_ad_gradients.py`, `pytestmark = pytest.mark.ad`)

```python
def _fd_check(J_of_theta, theta0, g_analytic, rel=1e-6, eps=1e-6):
    for i, (th, g) in enumerate(zip(np.atleast_1d(theta0), np.atleast_1d(g_analytic))):
        tp = theta0.copy(); tp[i] += eps
        tm = theta0.copy(); tm[i] -= eps
        fd = (J_of_theta(tp) - J_of_theta(tm)) / (2 * eps)
        assert abs(fd - g) <= rel * max(abs(fd), abs(g), 1e-12), (i, fd, g)

def test_dot_product_identity(device):
    # <A v, w> == <v, A^T w> for the assembled SBM operator (exact for CSR)
    ...  # rel < 1e-13

def test_shape_gradient_csg_three_way(device):
    """Circle radius+center gradient of J = sum (u(x_i) - u*_i)^2:
    adjoint  vs  torch dense twin  vs  central FD."""
    # 2D disk, level 4, p1, lam=1.0 (~<1.5k dofs)
    # adjoint: solve, adjoint solve, tape sweep, torch chain -> g_adj [3]
    # twin:    solve_dense_torch(...).backward()             -> g_twin [3]
    # FD:      re-run the full pipeline at theta +- eps (classification must not
    #          change: assert retained-tree keys identical across all FD evals)
    # asserts: |g_adj - g_twin| / |g| < 1e-9 ;  each vs FD rel < 1e-6
    ...

def test_kappa_gradient(device):
    # J = volume_qoi; dJ/dkappa adjoint vs central FD, rel < 1e-8
    ...

def test_neumann_shape_gradient(device):
    # same three-way structure on the exterior Neumann config (exercises
    # n_bar, corr, qbar cotangents through the projection IFT)
    ...

def test_frozen_classification_trust_region(device):
    # spec S1.5: for perturbations below an element crossing the active set is
    # constant and gradients are exact — assert classification identical at
    # theta +- eps AND adjoint-vs-FD agreement at 1e-6 (already covered above);
    # here additionally assert the retained key set is bit-identical.
    ...

def test_gradient_grid_stl(device):
    # stl:  dJ/dverts for 3 random vertices (9 comps) vs FD (rel 1e-5 —
    #       fp32 BVH candidate selection is forward-only; FP64 refinement keeps
    #       the gradient itself exact, but candidate flips near Voronoi edges
    #       justify the slightly looser bar; assert no flip via triangle-id equality)
    # grid: dJ/dV for 5 random voxels vs FD (rel 1e-6)
    # (neural weight-direction check arrives with the NeuralSDF backend in M1c)
    ...
```

- [ ] **Step 2: Verify failure; implement `adjoint.py` (+ residual kernels + `pointeval.py` + `reference.py`).** Care points: (i) λ_full = `T @ lam_free` (host scipy) then to device — the tape seed; (ii) cotangent sign: tape gives λᵀ∂R/∂•, gradient is its negation; (iii) torch backward call: `torch.autograd.backward([d_t, gbar_t, ...], [torch.from_numpy(dbar), ...])` accumulates into `oracle.params[i].grad`; zero grads first; (iv) the FD legs re-run carve→classify→solve — assert the retained Morton key arrays are identical between +ε and −ε evaluations (else shrink ε; generic positions make this robust); (v) `ConstrainedOperator` scratch-buffer non-reentrancy (M0 finding) is irrelevant here (CSR path) — note it in `adjoint.py`'s docblock for M1b.

- [ ] **Step 3: Run + full suite + commit**

```bash
git add -A && git commit -m "feat: adjoint shape/kappa gradients (Tier-2 VJP 1 + tape sweep + torch twin); three-way AD gate"
```

---

### Task 11: Capstone shape inverse + baselines + docs

**Files:**
- Test: append to `tests/test_ad_gradients.py`; create `tests/baselines/m1a_baselines.json`
- Docs: update `docs/superpowers/m0-deferred-findings.md` (mark addressed items); create `docs/superpowers/m1a-deferred-findings.md`

**Capstone (`test_shape_inverse_recovers_circle`, marker `ad`):** 2D, ground truth circle (c*, r*) = ((0.52, 0.47), 0.31); synthesize u* at ~30 probe points from a forward solve on the truth geometry (level 5). Optimize θ = (c, r) from (0.5, 0.5, 0.25) with torch Adam (lr 0.02, ~40 iterations), each iteration re-running the epoch pipeline (carve → classify → mesh → constraints → assemble → solve → J, adjoint → θ̄) — the spec §4.3 shape-optimization loop with reclassification between (never inside) differentiated segments. Assert `|r − r*| < 5e-3`, `‖c − c*‖ < 5e-3`, and J decreased monotonically after the first 5 iterations (allow line-search-free Adam noise: assert `J_final < 1e-3 * J_init`). Runtime target < 90 s on `cuda:0`.

**Baselines (`m1a_baselines.json`, locked at rtol 1e-6 like m0/m05):** MMS error ladders (Task 7 configs), Neumann flux-error ladders p1/p2-band (Task 8), corrected-perimeter value (Task 5), cross-backend solve errors (Task 9), and the three-way gradient values for the CSG case (Task 10). Regression test `test_m1a_baselines_locked` compares with rtol 1e-6 and a formatted mismatch message (M0.5 informational finding: format the diff, not a raw tuple).

**Docs:** mark the addressed items in `m0-deferred-findings.md` (dead code, ValueError, docstrings, bicgstab guards, operator protocol, integrand-tag keys); open `m1a-deferred-findings.md` recording at minimum: matrix-free SBM matvec + transposed matvec (needed by M1b NS), nonconforming surrogate facets (M4), `ConstrainedOperator` scratch-buffer reentrancy (M1b, GPU streams), TriMesh 2D support, GridSDF tricubic option, NeuralSDF backend + weight-direction gradient check (M1c, with the hero demo), surfacing the `domain` flag through the eventual `InputData` config (spec §3.1), and anything discovered during execution.

**Steps:**

- [ ] **Step 1: Write the capstone test (failing: baseline file missing), implement the loop, tune iterations for runtime**
- [ ] **Step 2: Generate + lock baselines; write the regression test**
- [ ] **Step 3: Docs updates**
- [ ] **Step 4: Full suite (all markers), commit**

```bash
git add -A && git commit -m "feat: shape-inverse capstone; lock m1a baselines; M1a deferred findings"
```

---

## Execution notes

- **Order is strict** (each task consumes the previous task's interfaces). Tasks 2–4 are independent of each other after Task 1 and may be parallelized by separate workers if desired; everything from Task 5 on is sequential.
- **Runtime discipline:** the full suite must stay under ~5 minutes on the workstation GPU (CI gate, spec §9.3). The expensive items are the 3D MMS level-5 solve and the capstone. If any test exceeds its budget, shrink the level, never the tolerance.
- **When something fails to converge or a formula looks sign-suspect:** the Γ̃→Γ limit tests (d=0 recovers classical Nitsche/Neumann) and the α-insensitivity patch test isolate weak-form bugs from geometry bugs. Trust the patch test: if P4 fails at 1e-11 the weak form or the geometry cache is wrong — do not loosen the tolerance (spec §9.2: pass criteria are the contract).
- **M1b inputs to carry forward** (record in m1a-deferred-findings as they materialize): measured bicgstab iteration counts vs α and λ (feeds the MF3 conditioning narrative), GridSDF Δx vs solve-error data (the geometry-limited-refinement plateau, spec §4.1), the `domain="outside"` test coverage M1b builds on, and the adjoint/forward wall-clock ratio (spec §9.3 gate: ≤ 2.5×).
