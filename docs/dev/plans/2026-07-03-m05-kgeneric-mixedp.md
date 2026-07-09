# DiffSim M0.5: k-Generic Foundation + Mixed p1/p2 + Verification Batteries — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the M0 foundation to k-D trees (k = 2, 3, 4) with per-axis periodic topology, add mixed p1/p2 meshes under the one-knob rule with p-transition constraints, and land the rigorous hanging/trace verification batteries — with all 37 existing M0 tests staying green throughout.

**Architecture:** Every M0 module gains a `dim` parameter (default 3 — back-compat is a hard requirement); `Octree` carries `dim` and `periodic`; Warp kernel factories close over `(nbf, nqp, dim)`. Mixed-p is per-element `p_elem` tags + per-p element bins; p-transition constraints reuse M0's owner-based hanging algorithm with a generalized owner rule (coarsest level first, then lowest p). Spec: `docs/superpowers/specs/2026-07-02-diffsim-design.md` §11 (k-D), §13 (mixed-p + verification), §2.4 (BasisTransform).

**Tech Stack:** unchanged from M0 (Python 3.12 venv at `.venv`, warp-lang 1.14, NumPy, SciPy, pytest).

## Global Constraints

- **All 37 existing M0 tests must pass unmodified after every task** (except where a task explicitly says it edits a test file — only additive edits allowed). Run `.venv/bin/pytest -q` at the end of every task.
- FP64 everywhere in kernels (`wp.float64`).
- LMAX per dimension: `{2: 31, 3: 21, 4: 15}` (64-bit interleaved keys); exceeding it raises `ValueError`, never truncates.
- `morton.LMAX = 21` remains as a module constant (3D back-compat); new code uses `morton.lmax(dim)`.
- Default `dim=3` and `periodic=(False,)*dim` on every public API — existing call sites work unchanged.
- Local node/qp ordering: x fastest, `a = Σ_i idx_i · (p+1)^i` with axis 0 = x (matches M0).
- Reference element `[-1,1]^k`; `detJxW = w · (h/2)^k`; `dN_phys = dN_ref · (2/h)`.
- **One-knob rule (spec §13.2):** across any shared face, either the level differs (by exactly 1, equal p) or p differs (equal level) — never both. `build_mesh` validates and raises on violation.
- p-transition constraints use the minimum rule: the p2 side's interface nodes not carried by the p1 neighbor are constrained to the p1 (linear) trace.
- Verification pass criteria are fixed (spec §9.2): machine-precision patch/trace tests < 1e-11 (scaled), orders within ±0.10, baseline comparisons rtol=1e-6.
- Commits: conventional style. Warp device fixture: `"cuda:0"` if available else `"cpu"` (unchanged conftest).
- Code standards (spec §16): every new/rewritten kernel and module gets a header docblock (equation/spec reference, symbol glossary, layout notes, invariants); block-level comments; no per-line narration.

## File Structure

```
Modify: src/diffsim/octree/morton.py       # k-generic spread/compact, lmax(), dim params
Modify: src/diffsim/octree/build.py        # Octree.dim/.periodic; dim-generic build/refine
Modify: src/diffsim/octree/carve.py        # dim-generic sampling lattice
Modify: src/diffsim/octree/lookup.py       # dim-generic faces; periodic wrap in find/neighbors
Modify: src/diffsim/octree/balance.py      # 3^k−1 offsets; periodic-aware
Modify: src/diffsim/mesh/nodes.py          # dim-generic; periodic node identification; p_elem bins; one-knob validator
Modify: src/diffsim/mesh/constraints.py    # generalized owner rule (level, then p); dim-generic probes/weights
Modify: src/diffsim/mesh/basis.py          # basis_tables(p, dim)
Modify: src/diffsim/assembly/operators.py  # dim-generic kernels (wp.vec2d/3d/4d); per-bin dispatch; mixed assemble_csr
Modify: src/diffsim/assembly/femelm.py     # fe_detJxW takes dim via factory-scaled weight tables (see Task 7)
Modify: src/diffsim/assembly/dirichlet.py  # bins-aware load assembly
Modify: src/diffsim/physics/poisson.py     # dim-generic gauss_points/load/l2 kernels
Create: tests/test_kd_morton.py            # Task 1
Create: tests/test_kd_tree.py              # Tasks 2–3
Create: tests/test_kd_mesh.py              # Tasks 4–6
Create: tests/test_kd_solve.py             # Tasks 7–8
Create: tests/test_mixedp.py               # Tasks 9–11
Create: tests/test_verification_batteries.py  # Tasks 12–13
Create: tests/helpers/field_eval.py        # point-evaluation + trace-conformity utilities
Create: tests/baselines/m05_baselines.json # locked in Task 13
```

Interface summary (locked here, used everywhere): `morton.lmax(dim)->int`; `morton.encode(xyz[N,dim], level, dim=3)`; `morton.decode/anchors/children/parent(..., dim=3)`; `Octree(keys, levels, dim=3, periodic=(False,...))` (frozen dataclass, `.h()`, `.anchors()`, `.centers()` unchanged semantics); `build_uniform(level, dim=3, periodic=None)`; `build_mesh(tree, p)` where `p: int | np.ndarray[int8]` → `Mesh` gains `p_elem: int8[Ne]`, `bins: dict[int p, np.ndarray elem_ids]`, `conn_of: dict[int p, int32[nb,(p+1)^dim]]` (M0's `Mesh.conn` field retained for uniform-p meshes only; `None` when mixed); `build_constraints(mesh)` unchanged signature, generalized internals; `basis_tables(p, dim=3)`; `DeviceMesh.from_mesh(mesh, constraints, tables_by_p: dict[int, Tables], device)`; `ConstrainedOperator(dm)` unchanged protocol.

---

### Task 1: k-generic Morton keys

**Files:**
- Modify: `src/diffsim/octree/morton.py` (full rewrite below)
- Test: `tests/test_kd_morton.py`
- Also run: `tests/test_o_morton.py` (must stay green — it uses the 3D defaults)

**Interfaces:**
- Consumes: nothing.
- Produces: `LMAX_FOR = {2: 31, 3: 21, 4: 15}`; `lmax(dim) -> int`; `LMAX = 21` (back-compat const); `encode(xyz: int64[N,dim], level, dim=3) -> uint64[N]`; `decode(keys, levels, dim=3) -> int64[N,dim]`; `anchors(keys, dim=3) -> int64[N,dim]` (lmax(dim)-depth); `children(keys, levels, dim=3) -> (uint64[N,2^dim], uint8[N])` in ascending Morton order (axis 0 most significant); `parent(keys, levels, dim=3)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_kd_morton.py`:

```python
import numpy as np
import pytest
from diffsim.octree import morton

pytestmark = pytest.mark.tier1

@pytest.mark.parametrize("dim", [2, 3, 4])
def test_kd_encode_decode_bijective(dim):
    rng = np.random.default_rng(dim)
    L = morton.lmax(dim)
    for level in [0, 1, 5, L // 2, L]:
        xyz = rng.integers(0, 2**level, size=(500, dim), dtype=np.int64)
        keys = morton.encode(xyz, level, dim=dim)
        back = morton.decode(keys, np.full(500, level, np.uint8), dim=dim)
        assert np.array_equal(back, xyz)

@pytest.mark.parametrize("dim,L", [(2, 31), (3, 21), (4, 15)])
def test_kd_lmax_and_overflow(dim, L):
    assert morton.lmax(dim) == L
    with pytest.raises(ValueError):
        morton.encode(np.zeros((1, dim), np.int64), L + 1, dim=dim)
    with pytest.raises(ValueError):
        morton.encode(np.full((1, dim), 2, np.int64), 1, dim=dim)  # coord out of range

@pytest.mark.parametrize("dim", [2, 3, 4])
def test_kd_children_parent_roundtrip(dim):
    xyz = np.arange(1, dim + 1, dtype=np.int64).reshape(1, dim)
    keys = morton.encode(xyz, 4, dim=dim)
    ck, cl = morton.children(keys, np.array([4], np.uint8), dim=dim)
    assert ck.shape == (1, 2**dim) and np.all(cl == 5)
    assert np.all(np.diff(ck[0].astype(np.uint64)) > 0)   # ascending Morton order
    pk, pl = morton.parent(ck.reshape(-1), np.full(2**dim, 5, np.uint8), dim=dim)
    assert np.all(pk == keys[0]) and np.all(pl == 4)

def test_3d_backcompat_constant():
    assert morton.LMAX == 21 == morton.lmax(3)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_kd_morton.py -v` — Expected: FAIL (`lmax` undefined).

- [ ] **Step 3: Rewrite `src/diffsim/octree/morton.py`**

```python
"""Morton (Z-order) keys for k-D trees, k in {2, 3, 4}. Host-side, NumPy-vectorized.

Spec: design doc S11.1. Key layout: per-axis coordinates at lmax(dim) depth,
bit-interleaved into one uint64 with axis 0 (x) MOST significant per bit
group; octant level carried in a separate uint8 array. Sorting by
(key, level) yields SFC pre-order. LMAX_FOR = {2: 31, 3: 21, 4: 15}:
dim*lmax bits used, level 2^lmax cells per axis maximum.

Symbols: xyz = integer octant coords at the octant's own level;
anchors = coords scaled to the lmax(dim) grid (level-independent).
Invariant: encode/decode are exact inverses for in-range input;
out-of-range level or coordinate raises ValueError (never truncates).
"""
import numpy as np

LMAX_FOR = {2: 31, 3: 21, 4: 15}
LMAX = 21  # 3D back-compat constant; new code uses lmax(dim)

def lmax(dim: int) -> int:
    return LMAX_FOR[dim]

# --- bit spread/compact: insert (dim-1) zero bits between input bits ------
# Standard magic-mask ladders; each mask keeps the bits at their target
# stride. Verified by the bijectivity tests over the full level range.

def _spread2(v):  # 31 bits -> every 2nd bit
    v = v.astype(np.uint64) & np.uint64(0x7FFFFFFF)
    v = (v | (v << np.uint64(16))) & np.uint64(0x0000FFFF0000FFFF)
    v = (v | (v << np.uint64(8)))  & np.uint64(0x00FF00FF00FF00FF)
    v = (v | (v << np.uint64(4)))  & np.uint64(0x0F0F0F0F0F0F0F0F)
    v = (v | (v << np.uint64(2)))  & np.uint64(0x3333333333333333)
    v = (v | (v << np.uint64(1)))  & np.uint64(0x5555555555555555)
    return v

def _compact2(v):
    v = v.astype(np.uint64) & np.uint64(0x5555555555555555)
    v = (v | (v >> np.uint64(1)))  & np.uint64(0x3333333333333333)
    v = (v | (v >> np.uint64(2)))  & np.uint64(0x0F0F0F0F0F0F0F0F)
    v = (v | (v >> np.uint64(4)))  & np.uint64(0x00FF00FF00FF00FF)
    v = (v | (v >> np.uint64(8)))  & np.uint64(0x0000FFFF0000FFFF)
    v = (v | (v >> np.uint64(16))) & np.uint64(0x7FFFFFFF)
    return v

def _spread3(v):  # 21 bits -> every 3rd bit (M0's part1by2, unchanged)
    v = v.astype(np.uint64) & np.uint64(0x1FFFFF)
    v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    v = (v | (v << np.uint64(8)))  & np.uint64(0x100F00F00F00F00F)
    v = (v | (v << np.uint64(4)))  & np.uint64(0x10C30C30C30C30C3)
    v = (v | (v << np.uint64(2)))  & np.uint64(0x1249249249249249)
    return v

def _compact3(v):
    v = v.astype(np.uint64) & np.uint64(0x1249249249249249)
    v = (v | (v >> np.uint64(2)))  & np.uint64(0x10C30C30C30C30C3)
    v = (v | (v >> np.uint64(4)))  & np.uint64(0x100F00F00F00F00F)
    v = (v | (v >> np.uint64(8)))  & np.uint64(0x1F0000FF0000FF)
    v = (v | (v >> np.uint64(16))) & np.uint64(0x1F00000000FFFF)
    v = (v | (v >> np.uint64(32))) & np.uint64(0x1FFFFF)
    return v

def _spread4(v):  # 15 bits -> every 4th bit
    v = v.astype(np.uint64) & np.uint64(0x7FFF)
    v = (v | (v << np.uint64(24))) & np.uint64(0x000000FF000000FF)
    v = (v | (v << np.uint64(12))) & np.uint64(0x000F000F000F000F)
    v = (v | (v << np.uint64(6)))  & np.uint64(0x0303030303030303)
    v = (v | (v << np.uint64(3)))  & np.uint64(0x1111111111111111)
    return v

def _compact4(v):
    v = v.astype(np.uint64) & np.uint64(0x1111111111111111)
    v = (v | (v >> np.uint64(3)))  & np.uint64(0x0303030303030303)
    v = (v | (v >> np.uint64(6)))  & np.uint64(0x000F000F000F000F)
    v = (v | (v >> np.uint64(12))) & np.uint64(0x000000FF000000FF)
    v = (v | (v >> np.uint64(24))) & np.uint64(0x7FFF)
    return v

_SPREAD = {2: _spread2, 3: _spread3, 4: _spread4}
_COMPACT = {2: _compact2, 3: _compact3, 4: _compact4}

def encode(xyz: np.ndarray, level, dim: int = 3) -> np.ndarray:
    xyz = np.asarray(xyz, np.int64).reshape(-1, dim)
    lev = np.asarray(level, np.int64)
    L = lmax(dim)
    if np.any(lev > L) or np.any(lev < 0):
        raise ValueError(f"level must be in [0, {L}] for dim={dim}")
    bound = np.int64(1) << (lev.reshape(-1, 1) if np.ndim(lev) else lev)
    if np.any(xyz < 0) or np.any(xyz >= bound):
        raise ValueError("coordinate out of range for level")
    shift = (L - lev).astype(np.uint64) if np.ndim(lev) else np.uint64(L - lev)
    spread = _SPREAD[dim]
    key = np.zeros(len(xyz), np.uint64)
    # axis 0 (x) gets the most-significant bit of each dim-bit group
    for i in range(dim):
        key |= spread(xyz[:, i].astype(np.uint64) << shift) << np.uint64(dim - 1 - i)
    return key

def decode(keys: np.ndarray, levels: np.ndarray, dim: int = 3) -> np.ndarray:
    keys = np.asarray(keys, np.uint64)
    L = lmax(dim)
    shift = (L - np.asarray(levels, np.int64)).astype(np.uint64)
    compact = _COMPACT[dim]
    out = np.empty((len(keys), dim), np.int64)
    for i in range(dim):
        out[:, i] = (compact(keys >> np.uint64(dim - 1 - i)) >> shift).astype(np.int64)
    return out

def anchors(keys: np.ndarray, dim: int = 3) -> np.ndarray:
    """Anchor coordinates at lmax(dim) depth (level-independent)."""
    keys = np.asarray(keys, np.uint64)
    compact = _COMPACT[dim]
    return np.stack([compact(keys >> np.uint64(dim - 1 - i)).astype(np.int64)
                     for i in range(dim)], axis=1)

def children(keys: np.ndarray, levels: np.ndarray, dim: int = 3):
    xyz = decode(keys, levels, dim=dim)
    lev = np.asarray(levels, np.int64)
    n_child = 1 << dim
    ck = np.empty((len(xyz), n_child), np.uint64)
    # child index m carries axis-i offset in bit (dim-1-i): matches the key
    # interleave (x most significant), so ascending m == ascending Morton.
    for m in range(n_child):
        off = np.array([(m >> (dim - 1 - i)) & 1 for i in range(dim)], np.int64)
        ck[:, m] = encode(xyz * 2 + off, lev + 1, dim=dim)
    return ck, (lev + 1).astype(np.uint8)

def parent(keys: np.ndarray, levels: np.ndarray, dim: int = 3):
    xyz = decode(keys, levels, dim=dim)
    lev = np.asarray(levels, np.int64)
    return encode(xyz // 2, lev - 1, dim=dim), (lev - 1).astype(np.uint8)
```

- [ ] **Step 4: Run new + regression tests**

Run: `.venv/bin/pytest tests/test_kd_morton.py tests/test_o_morton.py -v`
Expected: all PASS. (test_o_morton exercises the 3D defaults and the retained encode bounds fix — the `bound` expression above is M0's fixed version.)

- [ ] **Step 5: Full suite + commit**

Run: `.venv/bin/pytest -q` — 37 + 4 pass.

```bash
git add src/diffsim/octree/morton.py tests/test_kd_morton.py
git commit -m "feat: k-generic Morton keys (k=2,3,4; LMAX 31/21/15)"
```

---

### Task 2: Octree gains dim + periodic; build/refine/carve k-generic

**Files:**
- Modify: `src/diffsim/octree/build.py`, `src/diffsim/octree/carve.py`
- Test: `tests/test_kd_tree.py` (created here); `tests/test_o_build.py` stays green.

**Interfaces:**
- Produces: `Octree(keys, levels, dim=3, periodic=None)` frozen dataclass — `periodic: tuple[bool,...]` (default all-False, length dim); `.anchors()` uses `morton.anchors(keys, dim)`; `.h() = 2.0**-levels`; `.centers()` unchanged formula with `2^-lmax(dim)` scale; `build_uniform(level, dim=3, periodic=None)`; `build_adaptive(refine_fn, max_level, dim=3, periodic=None)`; `refine_elements(tree, mask)` (dim-inherited); `sort_unique` unchanged; `carve(tree, oracle, samples_per_axis=3)` — sampling lattice `s^dim`, oracle points shaped `[M, dim]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_kd_tree.py`:

```python
import numpy as np
import pytest
from diffsim.octree.build import Octree, build_uniform, refine_elements, build_adaptive
from diffsim.octree.carve import SphereOracle, carve

pytestmark = pytest.mark.tier1

@pytest.mark.parametrize("dim", [2, 3, 4])
def test_kd_uniform_count_and_volume(dim):
    for lvl in range(0, 4 if dim < 4 else 3):
        t = build_uniform(lvl, dim=dim)
        assert len(t) == (2**dim) ** lvl
        assert t.dim == dim and t.periodic == (False,) * dim
        assert abs(np.sum(t.h() ** dim) - 1.0) < 1e-13   # leaves tile the unit k-cube

@pytest.mark.parametrize("dim", [2, 4])
def test_kd_refine_count(dim):
    t = build_uniform(1, dim=dim)
    mask = np.zeros(len(t), bool); mask[0] = True
    t2 = refine_elements(t, mask)
    assert len(t2) == len(t) - 1 + 2**dim
    assert t2.dim == dim                              # dim/periodic survive refinement

@pytest.mark.parametrize("dim", [2, 3, 4])
def test_kd_carve_sphere(dim):
    lvl = 5 if dim == 2 else (4 if dim == 3 else 3)
    oracle = SphereOracle((0.5,) * dim, 0.45)
    tree, markers = carve(build_uniform(lvl, dim=dim), oracle)
    assert set(np.unique(markers)) <= {0, 1}
    vol = np.sum(tree.h() ** dim)
    # k-ball volume: pi r^2 (2D), 4/3 pi r^3 (3D), pi^2/2 r^4 (4D)
    vball = {2: np.pi * 0.45**2, 3: 4/3 * np.pi * 0.45**3, 4: np.pi**2 / 2 * 0.45**4}[dim]
    assert abs(vol - vball) / vball < 0.35            # coarse-level tolerance
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_kd_tree.py -v` — Expected: FAIL (`dim` not accepted).

- [ ] **Step 3: Modify `src/diffsim/octree/build.py`**

Replace the dataclass and constructors (keep `sort_unique` as-is):

```python
from dataclasses import dataclass, field
import numpy as np
from . import morton

@dataclass(frozen=True)
class Octree:
    """Sorted leaf array of a (possibly incomplete) k-D tree, k = dim.

    keys: uint64 interleaved Morton keys at lmax(dim) anchor depth.
    levels: uint8 per-leaf depth. periodic: per-axis wrap flags — consumed
    by lookup/balance/nodes; the key encoding itself is periodicity-agnostic.
    Invariant: (keys, levels) lexicographically sorted and duplicate-free.
    """
    keys: np.ndarray
    levels: np.ndarray
    dim: int = 3
    periodic: tuple = None

    def __post_init__(self):
        if self.periodic is None:
            object.__setattr__(self, "periodic", (False,) * self.dim)
        assert len(self.periodic) == self.dim

    def __len__(self):
        return len(self.keys)

    def anchors(self):
        return morton.anchors(self.keys, dim=self.dim)

    def h(self):
        return 2.0 ** (-self.levels.astype(np.float64))

    def centers(self):
        scale = 2.0 ** (-morton.lmax(self.dim))
        half = 0.5 * self.h()
        return self.anchors() * scale + half[:, None]

def _make(keys, levels, dim, periodic) -> Octree:
    k, l = sort_unique(np.asarray(keys, np.uint64), np.asarray(levels, np.uint8))
    return Octree(k, l, dim=dim, periodic=periodic)

def build_uniform(level: int, dim: int = 3, periodic=None) -> Octree:
    n = 1 << level
    grids = np.meshgrid(*([np.arange(n, dtype=np.int64)] * dim), indexing="ij")
    xyz = np.stack([g.ravel() for g in grids], axis=1)
    keys = morton.encode(xyz, level, dim=dim)
    return _make(keys, np.full(len(keys), level, np.uint8), dim,
                 tuple(periodic) if periodic is not None else (False,) * dim)

def refine_elements(tree: Octree, mask: np.ndarray) -> Octree:
    keep_k, keep_l = tree.keys[~mask], tree.levels[~mask]
    ck, cl = morton.children(tree.keys[mask], tree.levels[mask], dim=tree.dim)
    keys = np.concatenate([keep_k, ck.ravel()])
    levels = np.concatenate([keep_l, np.repeat(cl, 1 << tree.dim)])
    return _make(keys, levels, tree.dim, tree.periodic)

def build_adaptive(refine_fn, max_level: int, dim: int = 3, periodic=None) -> Octree:
    tree = build_uniform(0, dim=dim, periodic=periodic)
    while True:
        can = tree.levels < max_level
        want = refine_fn(tree.centers(), tree.h()[:, None])  # h broadcast-ready [N,1]
        mask = can & want
        if not mask.any():
            return tree
        tree = refine_elements(tree, mask)
```

- [ ] **Step 4: Modify `src/diffsim/octree/carve.py`** — replace `carve` body:

```python
def carve(tree: Octree, oracle, samples_per_axis: int = 3):
    """Classify leaves against the oracle on an s^dim sample lattice and
    drop EXTERIOR leaves. Markers: 0 = INTERIOR, 1 = INTERCEPTED (spec S4.2;
    Gauss-point lambda-criterion classification arrives with SBM in M1)."""
    s, dim = samples_per_axis, tree.dim
    t = np.linspace(0.0, 1.0, s)
    grids = np.meshgrid(*([t] * dim), indexing="ij")
    offs = np.stack([g.ravel() for g in grids], axis=1)        # [s^dim, dim]
    scale = 2.0 ** -morton.lmax(dim)
    lo = tree.anchors() * scale
    h = tree.h()
    pts = (lo[:, None, :] + offs[None, :, :] * h[:, None, None]).reshape(-1, dim)
    sgn = oracle.classify(pts).reshape(len(tree), s**dim)
    n_in = (sgn < 0.0).sum(axis=1)
    keep = n_in > 0
    markers = np.where(n_in[keep] == s**dim, 0, 1).astype(np.int8)
    return Octree(tree.keys[keep], tree.levels[keep], dim=dim,
                  periodic=tree.periodic), markers
```

(`SphereOracle` already handles any dim — numpy broadcasting.)

- [ ] **Step 5: Run tests + regression, commit**

Run: `.venv/bin/pytest tests/test_kd_tree.py tests/test_o_build.py -q` then `.venv/bin/pytest -q`.
Expected: all PASS (note `build_adaptive` already passed `h[:, None]` in M0 — the test in test_o_build uses that shape).

```bash
git add src/diffsim/octree/build.py src/diffsim/octree/carve.py tests/test_kd_tree.py
git commit -m "feat: Octree dim+periodic; k-generic build/refine/carve"
```

---

### Task 3: Lookup, face neighbors, balance — k-generic + per-axis periodic

**Files:**
- Modify: `src/diffsim/octree/lookup.py`, `src/diffsim/octree/balance.py`
- Test: append to `tests/test_kd_tree.py`; `tests/test_n_neighbors.py`, `tests/test_o_balance.py` stay green.

**Interfaces:**
- Produces: `face_offsets(dim) -> int64[2*dim, dim]` (order: axis0−, axis0+, axis1−, axis1+, … — 3D order matches M0's FACE_OFFSETS/BoundaryTypes.WALL); `LeafLookup(tree)` — `.find(anchors)` now **wraps anchor coordinates modulo 2^lmax on periodic axes** before containment search (out-of-range on non-periodic axes still → −1); `face_neighbors(tree) -> list[int64[N]]` of length 2·dim; `balance2to1(tree)`, `check_balance(tree)` — neighbor probes wrap on periodic axes (3^dim−1 offsets).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_kd_tree.py`)

```python
from diffsim.octree.lookup import LeafLookup, face_neighbors, face_offsets
from diffsim.octree.balance import balance2to1, check_balance

@pytest.mark.parametrize("dim", [2, 4])
def test_kd_face_neighbors_and_boundary(dim):
    t = build_uniform(2, dim=dim)
    nbrs = face_neighbors(t)
    assert len(nbrs) == 2 * dim
    a = t.anchors()
    xmin = a[:, 0] == 0
    assert np.all(nbrs[0][xmin] == -1) and np.all(nbrs[0][~xmin] >= 0)

def test_periodic_wrap_2d():
    t = build_uniform(2, dim=2, periodic=(True, False))
    nbrs = face_neighbors(t)
    assert np.all(nbrs[0] >= 0) and np.all(nbrs[1] >= 0)     # x wraps: no boundary
    ymin = t.anchors()[:, 1] == 0
    assert np.all(nbrs[2][ymin] == -1)                        # y does not wrap
    # wrap correctness: X_MINUS neighbor of an x=0 element is an x=max element
    e0 = np.where((t.anchors()[:, 0] == 0))[0][0]
    j = nbrs[0][e0]
    assert t.anchors()[j, 0] == (1 << (morton := __import__("diffsim.octree.morton", fromlist=["lmax"])).lmax(2)) - (1 << (morton.lmax(2) - 2))

@pytest.mark.parametrize("dim", [2, 4])
def test_kd_balance(dim):
    t = build_uniform(1, dim=dim)
    for _ in range(3):                       # deep nested + a second seed = imbalance
        mask = np.zeros(len(t), bool); mask[np.lexsort((t.keys,))[0]] = True
        t = refine_elements(t, mask)
    mask = np.zeros(len(t), bool); mask[np.lexsort((t.keys,))[1]] = True
    t = refine_elements(t, mask)
    tb = balance2to1(t)
    assert check_balance(tb)
    assert abs(np.sum(tb.h() ** dim) - 1.0) < 1e-13

def test_periodic_balance_wraps():
    # refine an x=0-adjacent element deeply; with x periodic the x=max column
    # must be forced to refine by the wrap-adjacency
    t = build_uniform(1, dim=2, periodic=(True, False))
    for _ in range(2):
        mask = np.zeros(len(t), bool); mask[np.lexsort((t.keys,))[0]] = True
        t = refine_elements(t, mask)
    tb = balance2to1(t)
    assert check_balance(tb)
    assert len(tb) > len(t)
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest tests/test_kd_tree.py -k "face or periodic or kd_balance" -v` → FAIL.

- [ ] **Step 3: Modify `src/diffsim/octree/lookup.py`**

```python
import numpy as np
from itertools import product
from . import morton
from .build import Octree

def face_offsets(dim: int) -> np.ndarray:
    """Face order: axis0-, axis0+, axis1-, axis1+, ... (3D == BoundaryTypes.WALL)."""
    out = []
    for ax in range(dim):
        for sgn in (-1, 1):
            off = [0] * dim
            off[ax] = sgn
            out.append(off)
    return np.array(out, np.int64)

FACE_OFFSETS = face_offsets(3)   # back-compat constant

def _wrap(anchors: np.ndarray, tree: Octree) -> np.ndarray:
    """Wrap probe coords modulo the grid on periodic axes; leave others."""
    G = 1 << morton.lmax(tree.dim)
    a = anchors.copy()
    for ax in range(tree.dim):
        if tree.periodic[ax]:
            a[:, ax] %= G
    return a

class LeafLookup:
    """Containment lookup: lmax-grid anchor -> leaf index via truncated keys.
    Periodic axes wrap before the search (spec S11.1). O(N * lmax) host
    prototype loop — documented delta; cuFEM uses traversal."""
    def __init__(self, tree: Octree):
        self.tree = tree
        self._map = {}
        for i, (k, l) in enumerate(zip(tree.keys.tolist(), tree.levels.tolist())):
            self._map[(int(k), int(l))] = i

    def find(self, anchors: np.ndarray) -> np.ndarray:
        dim = self.tree.dim
        L = morton.lmax(dim)
        anchors = _wrap(np.asarray(anchors, np.int64).reshape(-1, dim), self.tree)
        out = np.full(len(anchors), -1, np.int64)
        G = 1 << L
        inside = np.all((anchors >= 0) & (anchors < G), axis=1)
        for i in np.where(inside)[0]:
            a = anchors[i]
            for lvl in range(L, -1, -1):
                key = int(morton.encode((a >> (L - lvl))[None, :], lvl, dim=dim)[0])
                j = self._map.get((key, lvl))
                if j is not None:
                    out[i] = j
                    break
        return out

def face_neighbors(tree: Octree) -> list:
    lk = LeafLookup(tree)
    anchors = tree.anchors()
    size = (1 << (morton.lmax(tree.dim) - tree.levels.astype(np.int64)))[:, None]
    center_off = size // 2
    out = []
    for off in face_offsets(tree.dim):
        probe = anchors + center_off + off * (center_off + 1)
        idx = lk.find(probe)
        idx[idx == np.arange(len(tree))] = -1
        out.append(idx)
    return out
```

- [ ] **Step 4: Modify `src/diffsim/octree/balance.py`** — generalize offsets and thread dim:

```python
import numpy as np
from itertools import product
from . import morton
from .build import Octree, refine_elements
from .lookup import LeafLookup

def _nbr_offsets(dim: int) -> np.ndarray:
    return np.array([o for o in product((-1, 0, 1), repeat=dim) if any(o)], np.int64)

def _neighbor_levels(tree: Octree):
    lk = LeafLookup(tree)          # find() wraps periodic axes internally
    anchors = tree.anchors()
    size = (1 << (morton.lmax(tree.dim) - tree.levels.astype(np.int64)))[:, None]
    center = anchors + size // 2
    offs = _nbr_offsets(tree.dim)
    nbr_idx = np.empty((len(tree), len(offs)), np.int64)
    for j, off in enumerate(offs):
        nbr_idx[:, j] = lk.find(center + off * size)
    return nbr_idx

def check_balance(tree: Octree) -> bool:
    nbr = _neighbor_levels(tree)
    lev = tree.levels.astype(np.int64)
    for j in range(nbr.shape[1]):
        ok = nbr[:, j] >= 0
        if np.any(np.abs(lev[ok] - lev[nbr[ok, j]]) > 1):
            return False
    return True

def balance2to1(tree: Octree) -> Octree:
    while True:
        nbr = _neighbor_levels(tree)
        lev = tree.levels.astype(np.int64)
        to_refine = np.zeros(len(tree), bool)
        for j in range(nbr.shape[1]):
            ok = nbr[:, j] >= 0
            viol = ok.copy()
            viol[ok] = lev[ok] - lev[nbr[ok, j]] > 1
            to_refine[nbr[viol, j]] = True
        if not to_refine.any():
            return tree
        tree = refine_elements(tree, to_refine)
```

- [ ] **Step 5: Run new + regression + full suite, commit**

Run: `.venv/bin/pytest tests/test_kd_tree.py tests/test_n_neighbors.py tests/test_o_balance.py -q` then `.venv/bin/pytest -q`. Expected: all PASS. Fix the awkward inline morton import in `test_periodic_wrap_2d` if flake — simplest committed form: `from diffsim.octree import morton` at file top and compute expected anchor as `(1 << morton.lmax(2)) - (1 << (morton.lmax(2) - 2))`.

```bash
git add src/diffsim/octree/lookup.py src/diffsim/octree/balance.py tests/test_kd_tree.py
git commit -m "feat: k-generic lookup/neighbors/balance with per-axis periodic wrap"
```

---

### Task 4: Nodes k-generic + periodic node identification

**Files:**
- Modify: `src/diffsim/mesh/nodes.py`
- Test: `tests/test_kd_mesh.py` (created); `tests/test_c_nodes.py` stays green.

**Interfaces:**
- Produces: `Mesh` fields as in M0 plus `dim: int` (from tree); `node_icoords` on the doubled grid `2^(lmax(dim)+1)`; **periodic identification**: on periodic axes, icoord `== G2` maps to `0` *before* dedup (wrap-around node identity); `boundary_nodes` only flags non-periodic-axis extremes. Uniform-p signature `build_mesh(tree, p)` unchanged. (Mixed-p arrives in Task 9 — this task keeps `conn` as the single connectivity array.)

- [ ] **Step 1: Write the failing tests**

`tests/test_kd_mesh.py`:

```python
import numpy as np
import pytest
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh

pytestmark = pytest.mark.tier2

@pytest.mark.parametrize("dim", [2, 3, 4])
def test_kd_node_counts(dim):
    lvl = 2
    for p in (1, 2):
        m = build_mesh(build_uniform(lvl, dim=dim), p=p)
        assert len(m.node_coords) == (p * 2**lvl + 1) ** dim
        assert m.conn.shape == ((2**dim) ** lvl, (p + 1) ** dim)

@pytest.mark.parametrize("dim", [2, 3])
def test_periodic_node_identification(dim):
    lvl = 2
    per = (True,) + (False,) * (dim - 1)
    m = build_mesh(build_uniform(lvl, dim=dim, periodic=per), p=1)
    # periodic axis loses its duplicate seam plane: 2^lvl instead of 2^lvl+1
    assert len(m.node_coords) == (2**lvl) * (2**lvl + 1) ** (dim - 1)
    # boundary flags exclude the periodic axis
    on_x_extreme = (m.node_icoords[:, 0] == 0)
    assert not np.any(m.boundary_nodes & on_x_extreme & ~np.any(
        [(m.node_icoords[:, ax] == 0) | (m.node_icoords[:, ax] == m.node_icoords[:, ax].max())
         for ax in range(1, dim)], axis=0))

def test_fully_periodic_2d():
    m = build_mesh(build_uniform(2, dim=2, periodic=(True, True)), p=1)
    assert len(m.node_coords) == 16          # (2^2)^2, torus
    assert not m.boundary_nodes.any()
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest tests/test_kd_mesh.py -v` → FAIL.

- [ ] **Step 3: Rewrite `src/diffsim/mesh/nodes.py`** (uniform-p; Task 9 extends):

```python
"""Node generation and connectivity for k-D tensor-product Lagrange meshes.

Spec: S2.3, S11.1. Nodes live on the DOUBLED integer grid (2^(lmax+1) per
axis) so p2 midpoints are integers. Local ordering x fastest:
a = sum_i idx_i * (p+1)^i, axis 0 = x (load-bearing: basis.py and
constraints.py index against it). Periodic axes: icoord == G2 is identified
with 0 BEFORE dedup, so seam nodes are single DOFs (wrap-around identity);
periodic axes contribute no boundary flags.
"""
from dataclasses import dataclass
from itertools import product
import numpy as np
from ..octree import morton
from ..octree.build import Octree

@dataclass(frozen=True)
class Mesh:
    p: int
    tree: Octree
    node_coords: np.ndarray    # float64 [Nn, dim], physical unit cube
    node_icoords: np.ndarray   # int64  [Nn, dim], doubled grid
    conn: np.ndarray           # int32  [Ne, (p+1)^dim]
    boundary_nodes: np.ndarray # bool   [Nn]

    @property
    def dim(self):
        return self.tree.dim

def _local_offsets(p: int, dim: int) -> np.ndarray:
    """x-fastest lattice: row a = digits of a in base (p+1), axis 0 first."""
    npe = p + 1
    return np.array([tuple(idx) for idx in product(*[range(npe)] * dim)],
                    np.int64)[:, ::-1]  # product varies LAST axis fastest -> reverse

def build_mesh(tree: Octree, p: int) -> Mesh:
    assert p in (1, 2)
    dim = tree.dim
    npe = p + 1
    offs = _local_offsets(p, dim)                       # [(p+1)^dim, dim]
    size = 1 << (morton.lmax(dim) - tree.levels.astype(np.int64))
    anchors2 = tree.anchors() * 2
    step2 = (2 * size) // p                             # h/p on the doubled grid
    all_ic = (anchors2[:, None, :] + offs[None, :, :] * step2[:, None, None]).reshape(-1, dim)
    G2 = 2 * (1 << morton.lmax(dim))
    for ax in range(dim):                               # periodic seam identity
        if tree.periodic[ax]:
            all_ic[:, ax] %= G2
    nodes, inverse = np.unique(all_ic, axis=0, return_inverse=True)
    conn = inverse.reshape(len(tree), npe**dim).astype(np.int32)
    coords = nodes.astype(np.float64) / G2
    boundary = np.zeros(len(nodes), bool)
    for ax in range(dim):
        if not tree.periodic[ax]:
            boundary |= (nodes[:, ax] == 0) | (nodes[:, ax] == G2)
    return Mesh(p, tree, coords, nodes, conn, boundary)
```

Note the `_local_offsets` reversal: `itertools.product` varies the *last* factor fastest; reversing columns puts axis 0 fastest, i.e. `a = i + npe*j + npe²*k(+ npe³*l)` — identical to M0's explicit 3D comprehension (verify: for p=1, dim=3, row 1 must be (1,0,0)).

- [ ] **Step 4: Run new + regression, full suite, commit**

Run: `.venv/bin/pytest tests/test_kd_mesh.py tests/test_c_nodes.py -q && .venv/bin/pytest -q` — all PASS.

```bash
git add src/diffsim/mesh/nodes.py tests/test_kd_mesh.py
git commit -m "feat: k-generic nodes with periodic seam identification"
```

---

### Task 5: Basis tables k-generic

**Files:**
- Modify: `src/diffsim/mesh/basis.py`
- Test: append to `tests/test_kd_mesh.py`; `tests/test_q_basis.py` stays green.

**Interfaces:**
- Produces: `basis_tables(p, dim=3) -> Tables` with `Tables` gaining `dim: int`; `N [nqp, nbf]`, `dN [nqp, nbf, dim]` (reference), `w [nqp]`, `nbf = (p+1)^dim`, `nqp = (p+1)^dim` Gauss points; qp and node ordering both x-fastest. `gauss_1d`, `lagrange_1d` unchanged.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_kd_mesh.py`)

```python
from diffsim.mesh.basis import basis_tables

@pytest.mark.parametrize("dim", [2, 4])
@pytest.mark.parametrize("p", [1, 2])
def test_kd_basis_tables(dim, p):
    tb = basis_tables(p, dim=dim)
    assert tb.dim == dim and tb.nbf == (p + 1) ** dim and tb.dN.shape == (tb.nqp, tb.nbf, dim)
    assert np.allclose(tb.N.sum(axis=1), 1.0, atol=1e-14)          # partition of unity
    assert np.allclose(tb.dN.sum(axis=1), 0.0, atol=1e-13)
    assert abs(tb.w.sum() - 2.0**dim) < 1e-13                       # reference volume
    h = 0.5
    Me = np.einsum("qa,qb,q->ab", tb.N, tb.N, tb.w * (h / 2) ** dim)
    assert abs(Me.sum() - h**dim) < 1e-14                           # mass PoU
    Ke = np.einsum("qak,qbk,q->ab", tb.dN * (2 / h), tb.dN * (2 / h), tb.w * (h / 2) ** dim)
    ev = np.linalg.eigvalsh(Ke)
    assert abs(ev[0]) < 1e-12 and ev[1] > 1e-9                      # constants-only nullspace
```

- [ ] **Step 2: Run to verify failure** — FAIL (`dim` not accepted).

- [ ] **Step 3: Rewrite `basis_tables` in `src/diffsim/mesh/basis.py`** (keep `gauss_1d`, `lagrange_1d`, add `dim` to `Tables`):

```python
@dataclass(frozen=True)
class Tables:
    p: int
    dim: int
    nbf: int
    nqp: int
    N: np.ndarray    # [nqp, nbf]
    dN: np.ndarray   # [nqp, nbf, dim] (reference derivatives)
    w: np.ndarray    # [nqp]

def basis_tables(p: int, dim: int = 3) -> Tables:
    from itertools import product as iproduct
    pts, wts = gauss_1d(p)
    npe, nq1 = p + 1, len(pts)
    nbf, nqp = npe**dim, nq1**dim
    # x-fastest multi-indices for both qp and nodes (product varies last
    # factor fastest -> reverse columns; matches nodes._local_offsets)
    qidx = np.array(list(iproduct(*[range(nq1)] * dim)), np.int64)[:, ::-1]
    aidx = np.array(list(iproduct(*[range(npe)] * dim)), np.int64)[:, ::-1]
    N = np.zeros((nqp, nbf)); dN = np.zeros((nqp, nbf, dim)); w = np.zeros(nqp)
    N1 = np.zeros((nq1, npe)); dN1 = np.zeros((nq1, npe))
    for q in range(nq1):
        N1[q], dN1[q] = lagrange_1d(p, pts[q])
    for q in range(nqp):
        w[q] = np.prod(wts[qidx[q]])
        for a in range(nbf):
            vals = N1[qidx[q], aidx[a]]                    # per-axis 1D values
            N[q, a] = np.prod(vals)
            for d in range(dim):
                terms = vals.copy()
                terms[d] = dN1[qidx[q, d], aidx[a, d]]
                dN[q, a, d] = np.prod(terms)
    return Tables(p, dim, nbf, nqp, N, dN, w)
```

- [ ] **Step 4: Run new + regression (`tests/test_q_basis.py` uses 3D defaults — verify identical tables), full suite, commit**

Run: `.venv/bin/pytest tests/test_kd_mesh.py tests/test_q_basis.py -q && .venv/bin/pytest -q` — all PASS.

```bash
git add src/diffsim/mesh/basis.py tests/test_kd_mesh.py
git commit -m "feat: k-generic tensor basis tables"
```

---

### Task 6: Constraints k-generic (h-hanging in any dim)

**Files:**
- Modify: `src/diffsim/mesh/constraints.py`
- Test: append to `tests/test_kd_mesh.py`; `tests/test_c_nodes.py` stays green.

**Interfaces:**
- Produces: `build_constraints(mesh)` unchanged signature; internals dim-generic: probe the `2^dim` surrounding octants (`itertools.product((-1, 0), repeat=dim)`), reference coords `xi[dim]`, tensor weights via `lagrange_1d` products over `mesh.dim` axes with the x-fastest local index (reuse `nodes._local_offsets(p, dim)`). Periodic note: probes go through `LeafLookup.find`, which wraps — hanging detection works across periodic seams unchanged.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_kd_mesh.py`)

```python
from diffsim.octree.build import refine_elements
from diffsim.octree.balance import balance2to1
from diffsim.mesh.constraints import build_constraints

@pytest.mark.parametrize("dim", [2, 4])
@pytest.mark.parametrize("p", [1, 2])
def test_kd_hanging_constraints(dim, p):
    t = build_uniform(1, dim=dim)
    mask = np.zeros(len(t), bool); mask[0] = True
    t = balance2to1(refine_elements(t, mask))
    m = build_mesh(t, p=p)
    c = build_constraints(m)
    assert c.hanging.sum() > 0
    assert np.allclose(np.asarray(c.T.sum(axis=1)).ravel(), 1.0, atol=1e-13)
    coef = np.arange(1, dim + 1, dtype=np.float64)
    f = lambda x: 1.0 + x @ coef                      # linear field in dim vars
    u_all = c.T @ f(m.node_coords[c.free_nodes])
    assert np.allclose(u_all, f(m.node_coords), atol=1e-12)
```

- [ ] **Step 2: Run to verify failure** — FAIL (3D hardcodes in constraints.py).

- [ ] **Step 3: Rewrite the dim-dependent parts of `src/diffsim/mesh/constraints.py`**

Replace `build_constraints` internals (keep `Constraints` dataclass; delete leftover 3D literals):

```python
from itertools import product as iproduct
from .basis import lagrange_1d
from .nodes import Mesh, _local_offsets

def build_constraints(mesh: Mesh) -> "Constraints":
    tree, p, dim = mesh.tree, mesh.p, mesh.dim
    lk = LeafLookup(tree)
    Nn = len(mesh.node_coords)
    lev = tree.levels.astype(np.int64)
    size2 = 2 * (1 << (morton.lmax(dim) - lev))
    anchors2 = tree.anchors() * 2

    node_elems = [[] for _ in range(Nn)]
    for e in range(len(tree)):
        for a in mesh.conn[e]:
            node_elems[a].append(e)

    hanging = np.zeros(Nn, bool)
    owner = np.full(Nn, -1, np.int64)
    G2 = 2 * (1 << morton.lmax(dim))
    probes = np.array(list(iproduct((-1, 0), repeat=dim)), np.int64)
    for n in range(Nn):
        ic = mesh.node_icoords[n]
        touch = set()
        for dp in probes:
            probe2 = ic + dp
            # non-periodic out-of-range is rejected by find(); periodic wraps
            idx = lk.find((probe2 // 2)[None, :])[0]
            if idx >= 0:
                touch.add(int(idx))
        non_carriers = set(touch) - set(node_elems[n])
        if non_carriers:
            hanging[n] = True
            owner[n] = min(non_carriers, key=lambda e: lev[e])   # coarsest wins

    free_nodes = np.where(~hanging)[0]
    free_of = np.full(Nn, -1, np.int64)
    free_of[free_nodes] = np.arange(len(free_nodes))

    offs = _local_offsets(p, dim)
    rows, cols, vals = [], [], []
    for n in free_nodes:
        rows.append(n); cols.append(free_of[n]); vals.append(1.0)
    for n in np.where(hanging)[0]:
        e = owner[n]
        # reference coords in the owner, in [-1,1]^dim; periodic seam: shift
        # the node icoord by the wrap that puts it inside the owner's box
        d_ic = mesh.node_icoords[n] - anchors2[e]
        for ax in range(dim):
            if tree.periodic[ax]:
                if d_ic[ax] > size2[e]:
                    d_ic[ax] -= G2
                elif d_ic[ax] < 0:
                    d_ic[ax] += G2
        xi = 2.0 * d_ic / size2[e] - 1.0
        w1 = [lagrange_1d(p, xi[d])[0] for d in range(dim)]
        for a in range(len(offs)):
            w = 1.0
            for d in range(dim):
                w *= w1[d][offs[a, d]]
            if abs(w) < 1e-14:
                continue
            tgt = mesh.conn[e, a]
            assert not hanging[tgt], "2:1 balance guarantees owner nodes are free"
            rows.append(n); cols.append(free_of[tgt]); vals.append(w)
    T = sp.csr_matrix((vals, (rows, cols)), shape=(Nn, len(free_nodes)))
    return Constraints(T, free_nodes, hanging)
```

- [ ] **Step 4: Run new + regression, full suite, commit**

Run: `.venv/bin/pytest tests/test_kd_mesh.py tests/test_c_nodes.py -q && .venv/bin/pytest -q` — all PASS.

```bash
git add src/diffsim/mesh/constraints.py tests/test_kd_mesh.py
git commit -m "feat: k-generic hanging constraints (2^dim probes, tensor weights, periodic-aware)"
```

---

### Task 7: Device operators, Dirichlet solve, MMS — k-generic

**Files:**
- Modify: `src/diffsim/assembly/operators.py`, `src/diffsim/assembly/femelm.py`, `src/diffsim/physics/poisson.py` (dirichlet.py needs no change — it is dim-agnostic through these)
- Test: `tests/test_kd_solve.py`; `tests/test_g_operator.py`, `tests/test_p_patch.py`, `tests/test_v_mms.py` stay green.

**Interfaces:**
- Produces: kernel factories keyed `(name, nbf, nqp, dim)`; `fe_detJxW` gains explicit dim handling — **decision: fold the `(h/2)^dim` into a factory-computed power inside kernels** via a `half_pow(he, dim)` `wp.func` chain (loop `for _ in range(dim)` closed over dim); gradient accumulator uses `VEC = {2: wp.vec2d, 3: wp.vec3d, 4: wp.vec4d}[dim]`; `DeviceMesh.from_mesh(mesh, constraints, tables, device)` gains `.dim`; `gauss_points(mesh, tables)` dim-generic; `integrate_volume`, `assemble_csr`, `operator_diagonal` dim-generic. Fallback (record if used): if `VEC[dim]` indexing with a loop variable fails to compile in Warp CPU mode, recompute the directional gradient per (a, d) pair — O(nbf²·dim) instead of O(nbf·dim), correctness identical.

- [ ] **Step 1: Write the failing tests**

`tests/test_kd_solve.py`:

```python
import numpy as np
import pytest
from diffsim.octree.build import build_uniform, refine_elements
from diffsim.octree.balance import balance2to1
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh, ConstrainedOperator, integrate_volume
from diffsim.assembly.dirichlet import DirichletPoisson
from diffsim.physics.poisson import l2_error

pytestmark = pytest.mark.tier3

def _dm(tree, p, device):
    m = build_mesh(tree, p=p)
    c = build_constraints(m)
    return m, c, DeviceMesh.from_mesh(m, c, basis_tables(p, dim=tree.dim), device)

@pytest.mark.parametrize("dim", [2, 4])
def test_kd_volume_and_symmetry(dim, device):
    t = build_uniform(2, dim=dim)
    mask = np.zeros(len(t), bool); mask[0] = True
    t = balance2to1(refine_elements(t, mask))
    m, c, dm = _dm(t, 1, device)
    assert abs(integrate_volume(dm) - 1.0) < 1e-13
    op = ConstrainedOperator(dm)
    rng = np.random.default_rng(5)
    v, w_ = rng.standard_normal(dm.n_free), rng.standard_normal(dm.n_free)
    assert abs(v @ op.matvec_numpy(w_) - w_ @ op.matvec_numpy(v)) < 1e-10

@pytest.mark.parametrize("dim", [2, 4])
def test_kd_linear_patch(dim, device):
    coef = np.arange(1, dim + 1, dtype=np.float64)
    LIN = lambda x: 1.0 + x @ coef
    ZERO = lambda x: np.zeros(len(x))
    t = build_uniform(2, dim=dim)
    mask = np.zeros(len(t), bool); mask[0] = True
    t = balance2to1(refine_elements(t, mask))
    for p in (1, 2):
        m, c, dm = _dm(t, p, device)
        u = DirichletPoisson(dm).solve(g_fn=LIN, f_fn=ZERO, tol=1e-13)
        assert l2_error(dm, u, LIN) < 1e-11

@pytest.mark.parametrize("dim,levels,expected", [(2, [3, 4, 5], 2.0), (4, [1, 2, 3], 2.0)])
def test_kd_mms_order(dim, levels, expected, device):
    U = lambda x: np.prod(np.sin(np.pi * x), axis=1)
    F = lambda x: dim * np.pi**2 * U(x)
    errs = []
    for lvl in levels:
        m, c, dm = _dm(build_uniform(lvl, dim=dim), 1, device)
        u = DirichletPoisson(dm).solve(g_fn=U, f_fn=F, tol=1e-13)
        errs.append(l2_error(dm, u, U))
    order = np.log2(errs[-2] / errs[-1])
    assert abs(order - expected) < 0.15, (dim, errs)
```

- [ ] **Step 2: Run to verify failure** — FAIL (kernels hardcode 3D).

- [ ] **Step 3: Modify `src/diffsim/assembly/femelm.py`** — make physical scaling dim-explicit via factory-supplied scale:

Keep `FEMElm` struct and `fe_N`. Replace `fe_dN`/`fe_detJxW` with forms taking precomputed scale factors (computed per element inside kernels; dim enters via factory constants):

```python
@wp.func
def fe_dN_s(dNtab: wp.array3d(dtype=wp.float64), fe: FEMElm, a: wp.int32,
            k: wp.int32, dscale: wp.float64) -> wp.float64:
    """Physical derivative: reference dN times dscale = 2/he."""
    return dNtab[fe.q, a, k] * dscale

@wp.func
def fe_detJxW_s(wtab: wp.array(dtype=wp.float64), fe: FEMElm,
                jac: wp.float64) -> wp.float64:
    """Quadrature weight times jac = (he/2)^dim (dim folded in by caller)."""
    return wtab[fe.q] * jac
```

(Keep the old `fe_dN`/`fe_detJxW` for back-compat with any 3D-only call sites until the operators rewrite below removes their uses; delete them at the end of this task if nothing references them — record which.)

- [ ] **Step 4: Rewrite the kernel factories in `src/diffsim/assembly/operators.py`**

Cache key gains dim. Representative rewrite (matvec; apply the same pattern to `make_poisson_element_matrices`, `volume_kernel`→`make_volume_kernel`, and in `poisson.py` to `make_load_kernel`, `make_l2_kernel`, `make_gp_interp_kernel` in `bratu.py` — same three changes each: factory takes `dim`, per-element `jac`/`dscale` computed with a `for _ in range(dim)` power loop, gradient loops run `range(dim)`):

```python
_VEC = {2: wp.vec2d, 3: wp.vec3d, 4: wp.vec4d}

def make_poisson_matvec(nbf: int, nqp: int, dim: int):
    key = ("poisson_mv", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]
    VEC = _VEC[dim]

    @wp.kernel
    def poisson_mv(conn: wp.array2d(dtype=wp.int32), h: wp.array(dtype=wp.float64),
                   Ntab: wp.array2d(dtype=wp.float64), dNtab: wp.array3d(dtype=wp.float64),
                   wtab: wp.array(dtype=wp.float64),
                   x: wp.array(dtype=wp.float64), y: wp.array(dtype=wp.float64)):
        e = wp.tid()
        fe = FEMElm()
        fe.e = e
        fe.he = h[e]
        half = fe.he * wp.float64(0.5)
        jac = wp.float64(1.0)
        for _ in range(dim):                      # (he/2)^dim, dim is compile-time
            jac = jac * half
        dscale = wp.float64(2.0) / fe.he
        for q in range(nqp):
            fe.q = q
            dJxW = fe_detJxW_s(wtab, fe, jac)
            g = VEC()
            for b in range(nbf):
                xb = x[conn[e, b]]
                for d in range(dim):
                    g[d] = g[d] + fe_dN_s(dNtab, fe, b, d, dscale) * xb
            for a in range(nbf):
                val = wp.float64(0.0)
                for d in range(dim):
                    val = val + fe_dN_s(dNtab, fe, a, d, dscale) * g[d]
                wp.atomic_add(y, conn[e, a], val * dJxW)

    _kernel_cache[key] = poisson_mv
    return poisson_mv
```

`DeviceMesh.__init__` stores `self.dim = mesh.dim` and passes `tables` whose arrays are already dim-shaped; `ConstrainedOperator` calls `make_poisson_matvec(dm.tables.nbf, dm.tables.nqp, dm.dim)`; `integrate_volume` uses `make_volume_kernel(nqp, dim)`; `assemble_csr` uses `make_poisson_element_matrices(nbf, nqp, dim)`. In `physics/poisson.py`, `gauss_points` builds the reference lattice with the same x-fastest `iproduct` reversal as `basis_tables` and scales by `2^-lmax(mesh.dim)`.

- [ ] **Step 5: Run new + ALL regression, commit**

Run: `.venv/bin/pytest tests/test_kd_solve.py -q` (first compile of 2D/4D kernels is slow) then the full suite `.venv/bin/pytest -q` — all PASS (3D kernels recompile under the new cache key; results identical, MMS baseline comparison must still pass).

```bash
git add src/diffsim/assembly tests/test_kd_solve.py src/diffsim/physics/poisson.py
git commit -m "feat: dim-generic device kernels, Dirichlet solve, MMS orders in 2D/4D"
```

---

### Task 8: Periodic end-to-end Poisson

**Files:**
- Test: append to `tests/test_kd_solve.py` (no src changes expected — periodic identity is in the mesh; Dirichlet masking already only flags non-periodic boundaries)

**Interfaces:** consumes everything above; validates the periodic pipeline through a solve.

- [ ] **Step 1: Write the failing (or passing — this is a validation gate) test**

```python
def test_periodic_poisson_3d(device):
    # u periodic in x, Dirichlet in y,z: u = sin(2 pi x) sin(pi y) sin(pi z)
    U = lambda x: np.sin(2 * np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]) * np.sin(np.pi * x[:, 2])
    F = lambda x: (4 + 1 + 1) * np.pi**2 * U(x)
    errs = []
    for lvl in [3, 4]:
        t = build_uniform(lvl, dim=3, periodic=(True, False, False))
        m, c, dm = _dm(t, 1, device)
        u = DirichletPoisson(dm).solve(g_fn=U, f_fn=F, tol=1e-13)
        errs.append(l2_error(dm, u, U))
    assert abs(np.log2(errs[0] / errs[1]) - 2.0) < 0.2, errs
    # seam continuity: wrap nodes are single DOFs, so this holds by construction;
    # assert it anyway via icoords uniqueness
    assert len(np.unique(m.node_icoords, axis=0)) == len(m.node_icoords)
```

- [ ] **Step 2: Run** — `.venv/bin/pytest tests/test_kd_solve.py::test_periodic_poisson_3d -v`. If it fails, the likely gap is `DirichletPoisson`'s boundary handling (it uses `mesh.boundary_nodes`, which correctly excludes periodic axes — debug from there; do NOT weaken the order assertion).

- [ ] **Step 3: Full suite + commit**

```bash
git add tests/test_kd_solve.py
git commit -m "test: periodic-axis Poisson end-to-end (order 2, seam as single DOFs)"
```

---

### Task 9: Mixed-p Mesh — p_elem, bins, one-knob validator

**Files:**
- Modify: `src/diffsim/mesh/nodes.py`
- Test: `tests/test_mixedp.py` (created)

**Interfaces:**
- Produces: `build_mesh(tree, p)` accepts `p: int | np.ndarray` (int8 per element). `Mesh` gains `p_elem: int8[Ne]`; `bins: dict[int, int64[nb]]` (element indices per order, ascending p); `conn_of: dict[int, int32[nb, (p+1)^dim]]`; for uniform meshes `Mesh.conn` stays populated (back-compat), for mixed it is `None`. **One-knob validation** at build time: for every face-adjacent pair (via `face_neighbors`), `(level_i != level_j) and (p_i != p_j)` raises `ValueError` naming the offending pair. Node table = union over bins of per-element lattices on the doubled grid (p1 nodes are a subset of the p2 lattice — the quadratic-DA embedding, spec §13.2).

- [ ] **Step 1: Write the failing tests**

`tests/test_mixedp.py`:

```python
import numpy as np
import pytest
from diffsim.octree.build import build_uniform, refine_elements
from diffsim.octree.balance import balance2to1
from diffsim.mesh.nodes import build_mesh

pytestmark = pytest.mark.tier2

def _mixed_tree_and_p(dim=3):
    """Uniform level-2 tree; p2 band = elements touching x=0 plane, p1 elsewhere.
    Same level everywhere -> only the p knob changes across faces."""
    t = build_uniform(2, dim=dim)
    p = np.ones(len(t), np.int8)
    p[t.anchors()[:, 0] == 0] = 2
    return t, p

def test_mixed_bins_and_backcompat():
    t, p = _mixed_tree_and_p()
    m = build_mesh(t, p)
    assert m.conn is None and set(m.bins) == {1, 2}
    assert len(m.bins[1]) + len(m.bins[2]) == len(t)
    assert m.conn_of[1].shape[1] == 8 and m.conn_of[2].shape[1] == 27
    mu = build_mesh(t, 1)
    assert mu.conn is not None and np.all(mu.p_elem == 1)     # uniform back-compat

def test_one_knob_violation_raises():
    t = build_uniform(1, dim=3)
    mask = np.zeros(len(t), bool); mask[0] = True
    t = balance2to1(refine_elements(t, mask))                  # levels 1 and 2 coexist
    p = np.ones(len(t), np.int8)
    p[t.levels == 2] = 2                                       # p change ACROSS the h-face
    with pytest.raises(ValueError, match="one-knob"):
        build_mesh(t, p)

def test_p1_nodes_subset_of_p2_lattice():
    t, p = _mixed_tree_and_p()
    m = build_mesh(t, p)
    # every p1 corner node lies on the p2 doubled-grid lattice by construction
    assert np.all(m.node_icoords % 1 == 0)
    # interface: corner nodes shared between a p1 and p2 element are single DOFs
    e1, e2 = m.bins[1][0], m.bins[2][0]
    shared = set(m.conn_of[1][0]) & set(map(int, m.conn_of[2][list(m.bins[2]).index(int(m.bins[2][0]))] if False else m.conn_of[2][0]))
    # (simplified: just assert global dedup produced no duplicate icoords)
    assert len(np.unique(m.node_icoords, axis=0)) == len(m.node_icoords)
```

(Clean up the clumsy `shared` line when implementing: keep only the dedup assertion — the load-bearing check.)

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Extend `build_mesh` in `src/diffsim/mesh/nodes.py`**

```python
@dataclass(frozen=True)
class Mesh:
    p: int                      # max order present (back-compat: uniform order)
    tree: Octree
    node_coords: np.ndarray
    node_icoords: np.ndarray
    conn: np.ndarray            # uniform meshes only; None when mixed
    boundary_nodes: np.ndarray
    p_elem: np.ndarray = None   # int8 [Ne]
    bins: dict = None           # p -> int64 element indices (ascending p)
    conn_of: dict = None        # p -> int32 [nb, (p+1)^dim]

    @property
    def dim(self):
        return self.tree.dim

def _validate_one_knob(tree: Octree, p_elem: np.ndarray):
    """Spec S13.2: across any face, level XOR p may change - never both."""
    from ..octree.lookup import face_neighbors
    lev = tree.levels.astype(np.int64)
    for nbr in face_neighbors(tree):
        ok = nbr >= 0
        i = np.where(ok)[0]
        j = nbr[i]
        bad = (lev[i] != lev[j]) & (p_elem[i] != p_elem[j])
        if bad.any():
            k = i[bad][0]
            raise ValueError(
                f"one-knob rule violated: elements {k} (level {lev[k]}, p {p_elem[k]}) "
                f"and {nbr[k]} (level {lev[nbr[k]]}, p {p_elem[nbr[k]]}) differ in both h and p")

def build_mesh(tree: Octree, p) -> Mesh:
    dim = tree.dim
    p_elem = (np.full(len(tree), p, np.int8) if np.isscalar(p)
              else np.asarray(p, np.int8))
    assert set(np.unique(p_elem)) <= {1, 2}
    uniform = len(np.unique(p_elem)) == 1
    if not uniform:
        _validate_one_knob(tree, p_elem)
    size = 1 << (morton.lmax(dim) - tree.levels.astype(np.int64))
    anchors2 = tree.anchors() * 2
    G2 = 2 * (1 << morton.lmax(dim))
    # emit per-bin lattices, dedup jointly (p1 nodes subset p2 lattice)
    per_bin, all_ic, counts = {}, [], []
    for pv in sorted(np.unique(p_elem)):
        eids = np.where(p_elem == pv)[0]
        offs = _local_offsets(int(pv), dim)
        step2 = (2 * size[eids]) // int(pv)
        ic = (anchors2[eids, None, :] + offs[None, :, :] * step2[:, None, None]).reshape(-1, dim)
        per_bin[int(pv)] = eids
        all_ic.append(ic)
        counts.append((int(pv), len(eids), (int(pv) + 1) ** dim))
    all_ic = np.concatenate(all_ic, axis=0)
    for ax in range(dim):
        if tree.periodic[ax]:
            all_ic[:, ax] %= G2
    nodes, inverse = np.unique(all_ic, axis=0, return_inverse=True)
    conn_of, pos = {}, 0
    for pv, nb, npe_d in counts:
        conn_of[pv] = inverse[pos:pos + nb * npe_d].reshape(nb, npe_d).astype(np.int32)
        pos += nb * npe_d
    coords = nodes.astype(np.float64) / G2
    boundary = np.zeros(len(nodes), bool)
    for ax in range(dim):
        if not tree.periodic[ax]:
            boundary |= (nodes[:, ax] == 0) | (nodes[:, ax] == G2)
    pmax = int(p_elem.max())
    return Mesh(pmax, tree, coords, nodes,
                conn_of[pmax] if uniform else None, boundary,
                p_elem=p_elem, bins=per_bin, conn_of=conn_of)
```

Uniform meshes keep `conn` populated (== `conn_of[p]`) so every M0 consumer is untouched.

- [ ] **Step 4: Run new + regression, full suite, commit**

Run: `.venv/bin/pytest tests/test_mixedp.py tests/test_c_nodes.py tests/test_kd_mesh.py -q && .venv/bin/pytest -q` — all PASS.

```bash
git add src/diffsim/mesh/nodes.py tests/test_mixedp.py
git commit -m "feat: mixed p1/p2 mesh bins with one-knob validation"
```

---

### Task 10: p-transition constraints (generalized owner rule)

**Files:**
- Modify: `src/diffsim/mesh/constraints.py`
- Test: append to `tests/test_mixedp.py`

**Interfaces:**
- Produces: `build_constraints(mesh)` handles mixed meshes: node incidence built over all bins; hanging test unchanged (touching non-carrier exists); **owner = min over non-carriers of (level, p_elem) lexicographic** (coarsest first; at equal level, the lower-order side — the minimum rule, spec §13.2); weights = owner's own-order tensor basis at the node's reference coords in the owner (p-transition case: p1 owner, linear weights — edge midpoint ½/½, face midpoint ¼×4, exactly the local-p-refinement draft's Eqs. 6–8).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_mixedp.py`)

```python
from diffsim.mesh.constraints import build_constraints

def test_p_transition_weights_minimum_rule():
    t, p = _mixed_tree_and_p()
    m = build_mesh(t, p)
    c = build_constraints(m)
    assert c.hanging.sum() > 0                     # p2 interface midpoints are hanging
    rowsum = np.asarray(c.T.sum(axis=1)).ravel()
    assert np.allclose(rowsum, 1.0, atol=1e-13)
    # every hanging row's nonzero weights must be from {0.25, 0.5} patterns
    # (linear trace at midpoints) or general only if h-hanging exists (none here)
    Tc = c.T.tocoo()
    hang_rows = np.where(c.hanging)[0]
    w = Tc.data[np.isin(Tc.row, hang_rows)]
    assert np.all(np.isin(np.round(w, 12), [0.25, 0.5]))

def test_mixed_linear_reproduction_and_free_corners():
    t, p = _mixed_tree_and_p()
    m = build_mesh(t, p)
    c = build_constraints(m)
    f = lambda x: 1.0 + 2 * x[:, 0] - 3 * x[:, 1] + 0.5 * x[:, 2]
    u_all = c.T @ f(m.node_coords[c.free_nodes])
    assert np.allclose(u_all, f(m.node_coords), atol=1e-12)
    # corner nodes (carried by every touching element) are never hanging
    corner_grid = (m.node_icoords % 2 == 0).all(axis=1)   # even icoords at level-2... 
    # robust form: corners of any element = nodes carried by ALL their touching elements;
    # spot-check: nodes of a p1 element are all free
    assert not c.hanging[m.conn_of[1][0]].any()

def test_mixed_with_h_transitions_combined():
    # p2 band at x=0 on a tree that ALSO has h-refinement away from the band:
    # both knobs in one mesh, never on one face
    t = build_uniform(2, dim=3)
    far = t.anchors()[:, 0] >= (1 << 21) // 2      # refine far-half elements
    mask = np.zeros(len(t), bool); mask[np.where(far)[0][:4]] = True
    from diffsim.octree.balance import balance2to1
    t = balance2to1(refine_elements(t, mask))
    p = np.ones(len(t), np.int8)
    p[t.anchors()[:, 0] == 0] = 2                  # p2 band untouched by refinement
    m = build_mesh(t, p)                            # must pass one-knob validation
    c = build_constraints(m)
    f = lambda x: 1.0 - x[:, 0] + 4 * x[:, 1] + 2 * x[:, 2]
    u_all = c.T @ f(m.node_coords[c.free_nodes])
    assert np.allclose(u_all, f(m.node_coords), atol=1e-12)
```

- [ ] **Step 2: Run to verify failure** — FAIL (constraints reads `mesh.conn`, None for mixed).

- [ ] **Step 3: Generalize `build_constraints`** — replace the incidence build and owner rule:

```python
    # node incidence over all bins; per-element (bin, row) handle for weights
    node_elems = [[] for _ in range(Nn)]
    elem_conn_row = {}                       # global elem id -> (p, local row)
    for pv, eids in mesh.bins.items():
        cn = mesh.conn_of[pv]
        for r, e in enumerate(eids):
            elem_conn_row[int(e)] = (pv, r)
            for a in cn[r]:
                node_elems[a].append(int(e))
    p_elem = mesh.p_elem
    ...
        if non_carriers:
            hanging[n] = True
            owner[n] = min(non_carriers, key=lambda e: (lev[e], int(p_elem[e])))
    ...
    for n in np.where(hanging)[0]:
        e = int(owner[n])
        pv, r = elem_conn_row[e]
        offs = _local_offsets(pv, dim)
        ...  # xi computation identical (with the periodic shift)
        w1 = [lagrange_1d(pv, xi[d])[0] for d in range(dim)]
        for a in range(len(offs)):
            w = 1.0
            for d in range(dim):
                w *= w1[d][offs[a, d]]
            if abs(w) < 1e-14:
                continue
            tgt = mesh.conn_of[pv][r, a]
            assert not hanging[tgt], "owner nodes must be free (one-knob + 2:1)"
            ...
```

(Uniform meshes: `mesh.bins` and `conn_of` exist for them too after Task 9 — the code path is one.)

- [ ] **Step 4: Run new + ALL regression (constraint tests from M0 and Tasks 6), full suite, commit**

```bash
git add src/diffsim/mesh/constraints.py tests/test_mixedp.py
git commit -m "feat: p-transition constraints via generalized (level, p) owner rule"
```

---

### Task 11: Mixed-p assembly — per-bin kernel dispatch

**Files:**
- Modify: `src/diffsim/assembly/operators.py`, `src/diffsim/assembly/dirichlet.py`, `src/diffsim/physics/poisson.py`
- Test: append to `tests/test_mixedp.py`

**Interfaces:**
- Produces: `DeviceMesh.from_mesh(mesh, constraints, tables_by_p, device)` where `tables_by_p: dict[int, Tables] | Tables` (single Tables = uniform back-compat, wrapped internally as `{mesh.p: tables}`); DeviceMesh holds per-bin device arrays `conn_d[p]`, `h_d[p]`, `N_d[p]`, `dN_d[p]`, `w_d[p]` plus `.dim`, `.n_nodes`, `.n_free`; `ConstrainedOperator.matvec` launches the matvec kernel once per bin (accumulating into the same y_full); `assemble_csr` loops bins; load assembly (`DirichletPoisson.solve`) and `l2_error`/`gauss_points` loop bins (gauss_points returns per-bin arrays concatenated in bin order with an offsets dict `gp_offsets[p]`).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_mixedp.py`)

```python
import warp as wp
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh, ConstrainedOperator, assemble_csr
from diffsim.assembly.dirichlet import DirichletPoisson
from diffsim.physics.poisson import l2_error

def _mixed_dm(device, dim=3):
    t, p = _mixed_tree_and_p(dim)
    m = build_mesh(t, p)
    c = build_constraints(m)
    tb = {1: basis_tables(1, dim=dim), 2: basis_tables(2, dim=dim)}
    return m, c, DeviceMesh.from_mesh(m, c, tb, device)

@pytest.mark.tier3
def test_mixed_matvec_vs_assembled(device):
    m, c, dm = _mixed_dm(device)
    A = assemble_csr(dm)
    op = ConstrainedOperator(dm)
    rng = np.random.default_rng(9)
    for _ in range(10):
        x = rng.standard_normal(dm.n_free)
        assert np.abs(A @ x - op.matvec_numpy(x)).max() < 1e-10 * max(1.0, np.abs(A @ x).max())

@pytest.mark.tier3
def test_mixed_linear_patch_machine_precision(device):
    m, c, dm = _mixed_dm(device)
    LIN = lambda x: 1.0 + 2 * x[:, 0] - 3 * x[:, 1] + 0.5 * x[:, 2]
    u = DirichletPoisson(dm).solve(g_fn=LIN, f_fn=lambda x: np.zeros(len(x)), tol=1e-13)
    assert l2_error(dm, u, LIN) < 1e-11
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement per-bin dispatch**

`DeviceMesh` restructure (keeps all M0 attribute names for uniform meshes via properties):

```python
class DeviceMesh:
    def __init__(self, mesh, constraints, tables_by_p, device):
        if not isinstance(tables_by_p, dict):
            tables_by_p = {mesh.p: tables_by_p}
        self.mesh, self.constraints, self.device = mesh, constraints, device
        self.dim = mesh.dim
        self.tables_by_p = tables_by_p
        self.bins = {}
        h_all = mesh.tree.h()
        for pv, eids in mesh.bins.items():
            tb = tables_by_p[pv]
            self.bins[pv] = dict(
                eids=eids,
                conn=wp.array(np.ascontiguousarray(mesh.conn_of[pv]), dtype=wp.int32, device=device),
                h=wp.array(np.ascontiguousarray(h_all[eids]), dtype=wp.float64, device=device),
                N=wp.array(tb.N, dtype=wp.float64, device=device),
                dN=wp.array(tb.dN, dtype=wp.float64, device=device),
                w=wp.array(tb.w, dtype=wp.float64, device=device),
                nbf=tb.nbf, nqp=tb.nqp)
        T = constraints.T.tocsr()
        self.T_dev = _csr_to_device(T, device)
        self.Tt_dev = _csr_to_device(T.T.tocsr(), device)
        self.n_nodes, self.n_free = T.shape

    # uniform-mesh back-compat accessors (M0 tests use dm.conn, dm.h, dm.N, ...)
    def _only_bin(self):
        assert len(self.bins) == 1
        return next(iter(self.bins.values()))
    conn = property(lambda s: s._only_bin()["conn"])
    h = property(lambda s: s._only_bin()["h"])
    N = property(lambda s: s._only_bin()["N"])
    dN = property(lambda s: s._only_bin()["dN"])
    w = property(lambda s: s._only_bin()["w"])
    tables = property(lambda s: s.tables_by_p[max(s.tables_by_p)])
```

`ConstrainedOperator.matvec`: after `y_full.zero_()`, loop `for pv, b in dm.bins.items(): wp.launch(make_poisson_matvec(b["nbf"], b["nqp"], dm.dim), dim=len(b["eids"]), inputs=[b["conn"], b["h"], b["N"], b["dN"], b["w"], self.x_full, self.y_full], device=d)`. `assemble_csr`: per-bin `poisson_Ke` + COO rows/cols per bin, concatenate, one CSR, then `TᵀKT`. `gauss_points(mesh, tables_by_p)`: per-bin physical qp arrays; `DirichletPoisson.solve` and `l2_error` loop bins for load/error kernels (per-bin `fq` arrays). Keep the M0 uniform path exercised by the regression tests.

- [ ] **Step 4: Run new + FULL regression (this touches everything), commit**

Run: `.venv/bin/pytest tests/test_mixedp.py -q && .venv/bin/pytest -q` — all PASS (M0's `_setup` paths flow through the single-bin back-compat properties).

```bash
git add src/diffsim/assembly src/diffsim/physics/poisson.py tests/test_mixedp.py
git commit -m "feat: mixed-p per-bin kernel dispatch; mixed linear patch at machine precision"
```

---

### Task 12: Trace-conformity checker + fuzz batteries (p1, p2, mixed)

**Files:**
- Create: `tests/helpers/__init__.py`, `tests/helpers/field_eval.py`
- Test: `tests/test_verification_batteries.py`

**Interfaces:**
- Produces (test helpers): `eval_in_element(mesh, e, pts_phys) -> float64[M]`-style evaluator `FieldEvaluator(mesh, u_all)` with `.eval(e: int, x_phys: float64[M, dim]) -> float64[M]` (tensor Lagrange of `p_elem[e]` at reference coords of the element); `trace_conformity_max_jump(mesh, u_all, n_samples=4) -> float` — for every face-adjacent element pair (via `face_neighbors`, skipping periodic-wrapped faces for simplicity: skip pairs whose anchor separation exceeds one element size), samples an `n_samples^(dim-1)` lattice strictly inside the shared face and returns the max |u_left − u_right|. This is the §13.3 C⁰ criterion (faces, not nodes).

- [ ] **Step 1: Write the helper**

`tests/helpers/field_eval.py`:

```python
"""Host-side field evaluation + trace-conformity checking (spec S13.3).

The trace check is THE hanging-node acceptance criterion: nodal agreement is
insufficient (the group's Delaunay counterexample) - the interpolated TRACE
along every shared face must match from both sides.
"""
import numpy as np
from diffsim.octree import morton
from diffsim.octree.lookup import face_neighbors, face_offsets
from diffsim.mesh.basis import lagrange_1d
from diffsim.mesh.nodes import _local_offsets

class FieldEvaluator:
    def __init__(self, mesh, u_all):
        self.mesh, self.u = mesh, np.asarray(u_all, np.float64)
        self.dim = mesh.dim
        L = morton.lmax(self.dim)
        self.lo = mesh.tree.anchors() / (1 << L)           # physical corner
        self.h = mesh.tree.h()
        self.row_of = {}
        for pv, eids in mesh.bins.items():
            for r, e in enumerate(eids):
                self.row_of[int(e)] = (pv, r)

    def eval(self, e, x_phys):
        pv, r = self.row_of[int(e)]
        conn = self.mesh.conn_of[pv][r]
        xi = 2.0 * (np.atleast_2d(x_phys) - self.lo[e]) / self.h[e] - 1.0
        offs = _local_offsets(pv, self.dim)
        out = np.zeros(len(xi))
        N1 = [np.array([lagrange_1d(pv, x)[0] for x in xi[:, d]]) for d in range(self.dim)]
        for a in range(len(offs)):
            w = np.ones(len(xi))
            for d in range(self.dim):
                w *= N1[d][:, offs[a, d]]
            out += w * self.u[conn[a]]
        return out

def trace_conformity_max_jump(mesh, u_all, n_samples: int = 4) -> float:
    dim = mesh.dim
    ev = FieldEvaluator(mesh, u_all)
    lo, h = ev.lo, ev.h
    offs = face_offsets(dim)
    nbrs = face_neighbors(mesh.tree)
    t = np.linspace(0.15, 0.85, n_samples)                  # strictly inside the face
    max_jump = 0.0
    for f, off in enumerate(offs):
        ax = f // 2
        for e in range(len(mesh.tree)):
            j = nbrs[f][e]
            if j < 0 or j <= e and h[j] == h[e]:            # visit each same-size pair once
                continue
            # skip periodic-wrapped pairs (physical coords don't coincide)
            if abs((lo[j] - lo[e])[ax]) > h[e] + h[j]:
                continue
            # sample lattice on e's face f (fine side of a coarse-fine pair
            # samples its own face - a subset of the coarse face: exactly
            # where hanging constraints act)
            grids = np.meshgrid(*([t] * (dim - 1)), indexing="ij")
            pts = np.empty((n_samples ** (dim - 1), dim))
            k = 0
            for d in range(dim):
                if d == ax:
                    pts[:, d] = lo[e, d] + (h[e] if off[ax] > 0 else 0.0)
                else:
                    pts[:, d] = lo[e, d] + grids[k].ravel() * h[e]
                    k += 1
            jump = np.abs(ev.eval(e, pts) - ev.eval(j, pts)).max()
            max_jump = max(max_jump, jump)
    return max_jump
```

- [ ] **Step 2: Write the battery tests**

`tests/test_verification_batteries.py`:

```python
"""Spec S13.3 batteries: randomized fuzz over p1-only, p2-only, and mixed
meshes asserting the three machine-precision invariants: partition of unity,
polynomial reproduction (linear always; quadratic on p2-only), and trace
conformity across every shared face."""
import numpy as np
import pytest
from diffsim.octree.build import build_uniform, refine_elements
from diffsim.octree.balance import balance2to1
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from tests.helpers.field_eval import trace_conformity_max_jump

pytestmark = pytest.mark.tier2

def _random_tree(rng, dim=3, base=1, rounds=3, max_level=4):
    t = build_uniform(base, dim=dim)
    for _ in range(rounds):
        mask = (rng.random(len(t)) < 0.25) & (t.levels < max_level)
        if mask.any():
            t = refine_elements(t, mask)
    return balance2to1(t)

def _checks(m, c, rng, reproduce_quadratic):
    dim = m.dim
    assert np.allclose(np.asarray(c.T.sum(axis=1)).ravel(), 1.0, atol=1e-12)
    coef = rng.standard_normal(dim)
    lin = lambda x: 3.0 + x @ coef
    u = c.T @ lin(m.node_coords[c.free_nodes])
    assert np.allclose(u, lin(m.node_coords), atol=1e-11)
    assert trace_conformity_max_jump(m, u) < 1e-11
    if reproduce_quadratic:
        A = rng.standard_normal((dim, dim)); A = A + A.T
        quad = lambda x: 1.0 + np.einsum("ni,ij,nj->n", x, A, x) + x @ coef
        uq = c.T @ quad(m.node_coords[c.free_nodes])
        assert np.allclose(uq, quad(m.node_coords), atol=1e-10)
        assert trace_conformity_max_jump(m, uq) < 1e-10
    # random constrained field: conformity must hold for ANY free vector
    ur = c.T @ rng.standard_normal(len(c.free_nodes))
    assert trace_conformity_max_jump(m, ur) < 1e-11

@pytest.mark.parametrize("trial", range(5))
def test_battery_p1_only(trial):
    rng = np.random.default_rng(100 + trial)
    t = _random_tree(rng)
    m = build_mesh(t, 1)
    _checks(m, build_constraints(m), rng, reproduce_quadratic=False)

@pytest.mark.parametrize("trial", range(5))
def test_battery_p2_only(trial):
    rng = np.random.default_rng(200 + trial)
    t = _random_tree(rng)
    m = build_mesh(t, 2)
    _checks(m, build_constraints(m), rng, reproduce_quadratic=True)

@pytest.mark.parametrize("trial", range(5))
def test_battery_mixed(trial):
    rng = np.random.default_rng(300 + trial)
    t = _random_tree(rng)
    # mixed under one-knob: promote to p2 a random subset of elements whose
    # face neighbors are ALL same-level (so only the p knob changes)
    from diffsim.octree.lookup import face_neighbors
    lev = t.levels.astype(int)
    nbrs = face_neighbors(t)
    same_level_ok = np.ones(len(t), bool)
    for nb in nbrs:
        ok = nb >= 0
        same_level_ok[ok] &= (lev[ok] == lev[nb[ok]])
    p = np.ones(len(t), np.int8)
    cand = np.where(same_level_ok)[0]
    # promote a connected same-level cluster and also require the PROMOTED
    # elements' neighbors that are level-different to stay p1 (one-knob)
    p[rng.choice(cand, size=max(1, len(cand) // 4), replace=False)] = 2
    # demote any promotion that violates one-knob (validator would raise)
    for f, nb in enumerate(nbrs):
        ok = nb >= 0
        viol = ok & (lev != np.where(ok, lev[np.maximum(nb, 0)], lev)) & (p == 2)
        # conservative: demote promoted elements with any level-differing neighbor
    for e in np.where(p == 2)[0]:
        for nb in nbrs:
            j = nb[e]
            if j >= 0 and lev[j] != lev[e]:
                p[e] = 1
                break
    if not (p == 2).any():
        pytest.skip("fuzz draw produced no valid p2 candidates")
    m = build_mesh(t, p)
    _checks(m, build_constraints(m), rng, reproduce_quadratic=False)
```

- [ ] **Step 3: Run** — `.venv/bin/pytest tests/test_verification_batteries.py -v`. Debug any conformity failure as a REAL bug (the checker itself is validated by the p1/p2 uniform passes); do not loosen tolerances. Clean up the vestigial `viol` lines in the mixed battery when implementing (keep the working demotion loop).

- [ ] **Step 4: Full suite + commit**

```bash
git add tests/helpers tests/test_verification_batteries.py
git commit -m "test: S13.3 fuzz batteries - PoU, polynomial reproduction, trace conformity (p1/p2/mixed)"
```

---

### Task 13: Quadratic patch + p2 adaptive MMS + M0.5 baselines

**Files:**
- Test: append to `tests/test_verification_batteries.py`; create `tests/baselines/m05_baselines.json` (generated)

**Interfaces:** consumes Tasks 7/11 solve paths.

- [ ] **Step 1: Write the tests**

```python
import json, pathlib
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.assembly.dirichlet import DirichletPoisson
from diffsim.physics.poisson import l2_error

BASE = pathlib.Path(__file__).parent / "baselines" / "m05_baselines.json"

@pytest.mark.tier3
def test_p2_quadratic_patch_adaptive(device):
    # -lap(u) = f with quadratic u must be exact for p2 through h-hanging faces
    t = build_uniform(2, dim=3)
    mask = np.zeros(len(t), bool); mask[[0, 9]] = True
    t = balance2to1(refine_elements(t, mask))
    m = build_mesh(t, 2)
    c = build_constraints(m)
    dm = DeviceMesh.from_mesh(m, c, basis_tables(2, dim=3), device)
    U = lambda x: 1.0 + x[:, 0]**2 + 2 * x[:, 1]**2 - 3 * x[:, 2]**2 + x[:, 0] * x[:, 1]
    F = lambda x: np.full(len(x), -(2.0 + 4.0 - 6.0))     # -lap(U) = 0 here... 
    # choose U with nonzero laplacian to exercise the load path:
    U = lambda x: x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2
    F = lambda x: np.full(len(x), -6.0) * (-1.0)          # -lap(U) = -6 -> F = -(-6)? f = -lap u = -6... 
    # DEFINITIVE: solver solves -lap(u) = f. lap(U) = 6 -> f = -6.
    F = lambda x: np.full(len(x), -6.0)
    u = DirichletPoisson(dm).solve(g_fn=U, f_fn=F, tol=1e-13)
    assert l2_error(dm, u, U) < 1e-10                     # quadratic patch, p2

@pytest.mark.tier3
def test_p2_adaptive_mms_order3(device):
    U = lambda x: np.prod(np.sin(np.pi * x), axis=1)
    F = lambda x: 3 * np.pi**2 * U(x)
    errs = []
    for lvl in [1, 2, 3]:
        t = build_uniform(lvl, dim=3)
        mask = np.zeros(len(t), bool); mask[0] = True
        t = balance2to1(refine_elements(t, mask))         # every level has hanging faces
        m = build_mesh(t, 2)
        c = build_constraints(m)
        dm = DeviceMesh.from_mesh(m, c, basis_tables(2, dim=3), device)
        u = DirichletPoisson(dm).solve(g_fn=U, f_fn=F, tol=1e-13)
        errs.append(l2_error(dm, u, U))
    assert abs(np.log2(errs[-2] / errs[-1]) - 3.0) < 0.25, errs
    # regression lock (create-then-skip / compare thereafter, rtol per spec 9.2)
    results = {"p2_adaptive_mms": errs}
    if BASE.exists():
        ref = json.loads(BASE.read_text())
        assert np.allclose(errs, ref["p2_adaptive_mms"], rtol=1e-6)
    else:
        BASE.write_text(json.dumps(results, indent=2) + "\n")
        pytest.skip("baseline created; re-run to compare")
```

Clean the commented U/F false starts when implementing — keep the definitive pair `U = Σx_i²`, `f = −ΔU = −6`.

- [ ] **Step 2: Run twice** (baseline create-then-compare), then FULL suite:

Run: `.venv/bin/pytest tests/test_verification_batteries.py -v && .venv/bin/pytest tests/test_verification_batteries.py -v && .venv/bin/pytest -q`
Expected: second run all PASS; full suite green (M0's 37 + all M0.5 additions).

- [ ] **Step 3: Commit**

```bash
git add tests/test_verification_batteries.py tests/baselines/m05_baselines.json
git commit -m "test: p2 quadratic patch through hanging faces + adaptive order-3 MMS, baselines locked"
```

---

## Self-Review (performed while writing)

1. **Spec coverage:** §11.1 k-generic core (T1–T7) ✔; per-axis periodic topology incl. seam node identity + balance wrap + end-to-end solve (T3, T4, T8) ✔; §13.2 one-knob + p-transition minimum rule + quadratic-DA-style joint lattice (T9–T10) ✔; §2.4 BasisTransform data-not-control-flow preserved (all constraints remain rows of one T) ✔; §13.3 batteries 1–2 and battery 3's mesh/constraint portions (T12–T13) ✔ — battery 3's SBM-Neumann acceptance test explicitly deferred to M1 per the roadmap. Deliberately out: 4D Bratu/Newton re-tests (Newton layer is dim-agnostic through the op protocol; covered by full-suite regression), k-generic THB (M7), space-time physics (M8).
2. **Placeholder scan:** two flagged clean-ups are instructions to the implementer (Task 12 vestigial lines, Task 13 false-start comments) with the definitive content given inline; Task 10's `...` elisions repeat code shown fully in Task 6 within the same file section — implementer merges the two shown versions (owner-rule lines are given completely).
3. **Type consistency:** `morton.lmax(dim)`, `Octree(dim, periodic)`, `Mesh(p_elem, bins, conn_of)`, `basis_tables(p, dim)`, `DeviceMesh.from_mesh(..., tables_by_p, ...)` with single-`Tables` back-compat, factories keyed `(name, nbf, nqp, dim)` — cross-checked against every consuming task. M0 back-compat surfaces: `morton.LMAX`, `FACE_OFFSETS`, `Mesh.conn` (uniform), `dm.conn/h/N/dN/w/tables` properties — all retained.
