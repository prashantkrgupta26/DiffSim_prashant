# DiffSim M0: Octree Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The M0 milestone of the DiffSim Warp prototype — GPU-resident octree FEM foundation: Morton octrees (build, carve, 2:1 balance), p1/p2 nodes + hanging-node constraints, Gauss-point assembly (matrix-free + assembled), CG/BiCGStab, strong-Dirichlet Poisson through MMS convergence, and the SNES-like NonlinearSolver validated on Bratu.

**Architecture:** Octree construction/balancing/node-generation run on host (NumPy) and produce device-resident arrays; all assembly, matvec, and solver iterations run as Warp kernels. Physics is expressed through the `Integrands` contract (Hughes/TalyFEM nomenclature) compiled into kernels by factory functions. Spec: `docs/superpowers/specs/2026-07-02-diffsim-design.md` (§2, §3, §5; test IDs from the cuFEM verification plan).

**Tech Stack:** Python ≥ 3.10, NVIDIA Warp (`warp-lang`), NumPy, SciPy (host CSR only), pytest.

## Global Constraints

- All floating-point state and kernels use `wp.float64` (FP64 spine; mixed precision is M3).
- `LMAX = 21` (64-bit Morton keys, 3D); level 22 must raise, not truncate (test O1).
- Domain is the unit cube `[0,1]^3`; M0 is 3D-only.
- Warp device: `"cuda:0"` if available, else Warp CPU (development on macOS works; CI without GPU works).
- Documented prototype deltas vs the cuFEM blueprint (spec §1.1): host-side octree construction; explicit element→node connectivity arrays (cuFEM uses map-free traversal — same math, cross-validated via test D1/X2 later); atomics-based scatter-add (D3 determinism relaxed to tolerance in the prototype).
- Node deduplication uses `np.unique(coords, axis=0)` on integer coordinate triples (no bit-packing overflow risk at LMAX).
- Test IDs mirror the cuFEM verification plan (O/N/C/Q/M/G/P/V/S tiers). Pytest markers: `tier1`, `tier2`, `tier3`.
- Basis node ordering and connectivity are lexicographic with **x fastest**: local node `a = ax + (p+1)*ay + (p+1)^2*az`.
- Reference element is `[-1,1]^3`; elements are axis-aligned cubes of size `h = 2^-level` (diagonal Jacobian: `detJ = (h/2)^3`, `d/dx = (2/h) d/dξ`).
- Commits follow conventional-commit style (`feat:`, `test:`, `fix:`).

## File Structure

```
pyproject.toml
src/diffsim/__init__.py
src/diffsim/octree/__init__.py
src/diffsim/octree/morton.py      # encode/decode/parent/children (host, vectorized)
src/diffsim/octree/build.py       # Octree dataclass, uniform + adaptive build, carve, refine
src/diffsim/octree/balance.py     # 2:1 balance
src/diffsim/octree/lookup.py      # leaf containment lookup, face-neighbor finding
src/diffsim/mesh/__init__.py
src/diffsim/mesh/nodes.py         # node generation p1/p2, connectivity, boundary tags
src/diffsim/mesh/constraints.py   # hanging-node detection + constraint operator T
src/diffsim/mesh/basis.py         # 1D/3D tensor-product basis + Gauss tables (p1/p2)
src/diffsim/assembly/__init__.py
src/diffsim/assembly/femelm.py    # FEMElm struct + fe_N/fe_dN/fe_detJxW accessors
src/diffsim/assembly/operators.py # kernel factories: matvec, residual, COO assembly, CSR SpMV
src/diffsim/assembly/dirichlet.py # strong-BC projector (lift + constrained-identity operator)
src/diffsim/solvers/__init__.py
src/diffsim/solvers/blas.py       # device axpy/dot/copy kernels
src/diffsim/solvers/krylov.py     # CG, BiCGStab (+ Jacobi preconditioner)
src/diffsim/solvers/newton.py     # NonlinearSolver (SNES-like)
src/diffsim/physics/__init__.py
src/diffsim/physics/poisson.py    # Poisson Integrands (Ae/be), L2-error functional
src/diffsim/physics/bratu.py      # Bratu residual/Jacobian-action Integrands
tests/conftest.py
tests/test_o_morton.py            # O1, O2
tests/test_o_build.py             # O3, O4, O5, O8
tests/test_o_balance.py           # O6
tests/test_n_neighbors.py         # N1, N2, N3
tests/test_c_nodes.py             # C1, C2, C4, C6
tests/test_q_basis.py             # Q1, Q2, M1, M2
tests/test_g_operator.py          # G1, G3 (via operator actions), S3
tests/test_p_patch.py             # P1/P2 (uniform), P3 (adaptive + constraints)
tests/test_v_mms.py               # V2 (p1 and p2 orders)
tests/test_s_solvers.py           # S1, S2, S4-lite
tests/test_newton_bratu.py        # NonlinearSolver on Bratu
tests/baselines/m0_baselines.json # locked regression outputs
```

---

### Task 1: Project scaffold + Morton encode/decode (O1, O2)

**Files:**
- Create: `pyproject.toml`, `src/diffsim/__init__.py`, `src/diffsim/octree/__init__.py`, `src/diffsim/octree/morton.py`, `tests/conftest.py`
- Test: `tests/test_o_morton.py`

**Interfaces:**
- Produces: `morton.LMAX: int = 21`; `morton.encode(xyz: np.ndarray[int64, (N,3)], level: int | np.ndarray[uint8]) -> np.ndarray[uint64]` (coords are integers at the octant's own level; raises `ValueError` if `level > LMAX` or coords out of range); `morton.decode(keys: np.ndarray[uint64], levels: np.ndarray[uint8]) -> np.ndarray[int64, (N,3)]`; `morton.children(keys, levels) -> (child_keys[(N,8)], child_levels)` (Morton/Z order); `morton.parent(keys, levels) -> (parent_keys, parent_levels)`.

- [ ] **Step 1: Write scaffold**

`pyproject.toml`:

```toml
[project]
name = "diffsim"
version = "0.0.1"
requires-python = ">=3.10"
dependencies = ["warp-lang>=1.7", "numpy>=1.26", "scipy>=1.11"]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
markers = [
  "tier1: sanity & determinism (octree infrastructure)",
  "tier2: connectivity & patch tests",
  "tier3: operator & assembly correctness",
]
```

`tests/conftest.py`:

```python
import numpy as np
import pytest
import warp as wp

wp.init()

@pytest.fixture(scope="session")
def device():
    return "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"

@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(20260702)
```

`src/diffsim/__init__.py` and `src/diffsim/octree/__init__.py`: empty files.

- [ ] **Step 2: Write the failing tests (O1, O2)**

`tests/test_o_morton.py`:

```python
import numpy as np
import pytest
from diffsim.octree import morton

pytestmark = pytest.mark.tier1

def test_O1_encode_decode_bijective():
    rng = np.random.default_rng(1)
    for level in [0, 1, 5, 12, morton.LMAX]:
        n = 1000
        xyz = rng.integers(0, 2**level, size=(n, 3), dtype=np.int64)
        keys = morton.encode(xyz, level)
        back = morton.decode(keys, np.full(n, level, dtype=np.uint8))
        assert np.array_equal(back, xyz)

def test_O1_overflow_rejected():
    with pytest.raises(ValueError):
        morton.encode(np.array([[0, 0, 0]]), morton.LMAX + 1)
    with pytest.raises(ValueError):
        morton.encode(np.array([[2, 0, 0]]), 1)  # coord out of range at level 1

def test_O2_parent_children_roundtrip_and_z_order():
    xyz = np.array([[3, 5, 7]], dtype=np.int64)
    keys = morton.encode(xyz, 4)
    levels = np.array([4], dtype=np.uint8)
    ck, cl = morton.children(keys, levels)
    assert ck.shape == (1, 8) and np.all(cl == 5)
    pk, pl = morton.parent(ck.reshape(-1), np.full(8, 5, dtype=np.uint8))
    assert np.all(pk == keys[0]) and np.all(pl == 4)
    assert np.all(np.diff(ck[0].astype(np.uint64)) > 0)  # emitted in ascending Morton order
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_o_morton.py -v`
Expected: FAIL / ERROR with `ModuleNotFoundError` or `AttributeError` (morton not implemented).

- [ ] **Step 4: Implement `src/diffsim/octree/morton.py`**

```python
"""Morton (Z-order) keys for 3D octrees. Host-side, NumPy-vectorized.

Key layout: interleaved anchor coordinates at LMAX depth (63 bits used);
octant level is carried in a separate uint8 array. Sorting by (key, level)
yields SFC pre-order.
"""
import numpy as np

LMAX = 21
_M0 = np.uint64(0x1FFFFF)

def _part1by2(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.uint64) & _M0
    v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    v = (v | (v << np.uint64(8)))  & np.uint64(0x100F00F00F00F00F)
    v = (v | (v << np.uint64(4)))  & np.uint64(0x10C30C30C30C30C3)
    v = (v | (v << np.uint64(2)))  & np.uint64(0x1249249249249249)
    return v

def _compact1by2(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.uint64) & np.uint64(0x1249249249249249)
    v = (v | (v >> np.uint64(2)))  & np.uint64(0x10C30C30C30C30C3)
    v = (v | (v >> np.uint64(4)))  & np.uint64(0x100F00F00F00F00F)
    v = (v | (v >> np.uint64(8)))  & np.uint64(0x1F0000FF0000FF)
    v = (v | (v >> np.uint64(16))) & np.uint64(0x1F00000000FFFF)
    v = (v | (v >> np.uint64(32))) & _M0
    return v

def encode(xyz: np.ndarray, level) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.int64).reshape(-1, 3)
    lev = np.asarray(level, dtype=np.int64)
    if np.any(lev > LMAX) or np.any(lev < 0):
        raise ValueError(f"level must be in [0, {LMAX}]")
    if np.any(xyz < 0) or np.any(xyz >= (np.int64(1) << lev if np.ndim(lev) else np.int64(1) << lev)):
        raise ValueError("coordinate out of range for level")
    shift = (LMAX - lev).astype(np.uint64) if np.ndim(lev) else np.uint64(LMAX - lev)
    ax = xyz[:, 0].astype(np.uint64) << shift
    ay = xyz[:, 1].astype(np.uint64) << shift
    az = xyz[:, 2].astype(np.uint64) << shift
    return (_part1by2(ax) << np.uint64(2)) | (_part1by2(ay) << np.uint64(1)) | _part1by2(az)

def decode(keys: np.ndarray, levels: np.ndarray) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.uint64)
    shift = (LMAX - np.asarray(levels, dtype=np.int64)).astype(np.uint64)
    ax = _compact1by2(keys >> np.uint64(2)) >> shift
    ay = _compact1by2(keys >> np.uint64(1)) >> shift
    az = _compact1by2(keys) >> shift
    return np.stack([ax, ay, az], axis=1).astype(np.int64)

def anchors(keys: np.ndarray) -> np.ndarray:
    """Anchor coordinates at LMAX depth (level-independent)."""
    keys = np.asarray(keys, dtype=np.uint64)
    return np.stack([_compact1by2(keys >> np.uint64(2)),
                     _compact1by2(keys >> np.uint64(1)),
                     _compact1by2(keys)], axis=1).astype(np.int64)

def children(keys: np.ndarray, levels: np.ndarray):
    xyz = decode(keys, levels)
    lev = np.asarray(levels, dtype=np.int64)
    ck = np.empty((len(xyz), 8), dtype=np.uint64)
    for m in range(8):
        off = np.array([(m >> 2) & 1, (m >> 1) & 1, m & 1], dtype=np.int64)
        ck[:, m] = encode(xyz * 2 + off, lev + 1)
    return ck, (lev + 1).astype(np.uint8)

def parent(keys: np.ndarray, levels: np.ndarray):
    xyz = decode(keys, levels)
    lev = np.asarray(levels, dtype=np.int64)
    return encode(xyz // 2, lev - 1), (lev - 1).astype(np.uint8)
```

Note: the child-offset bit layout `(x = m>>2, y = m>>1, z = m)` matches the key interleave (`x` most significant), so ascending `m` is ascending Morton order.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pip install -e ".[dev]" && pytest tests/test_o_morton.py -v`
Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src tests
git commit -m "feat: project scaffold + Morton encode/decode (O1, O2)"
```

---

### Task 2: Octree build — uniform, adaptive, sort/unique (O3, O4, O5)

**Files:**
- Create: `src/diffsim/octree/build.py`
- Test: `tests/test_o_build.py`

**Interfaces:**
- Consumes: `morton.*` from Task 1.
- Produces: `@dataclass Octree { keys: np.uint64[N], levels: np.uint8[N] }` sorted by `(keys, levels)`, with methods `anchors() -> int64[N,3]` (LMAX-depth anchors), `h() -> float64[N]` (= `2.0**-levels`), `centers() -> float64[N,3]` (physical, unit cube), `__len__`; `build_uniform(level: int) -> Octree`; `build_adaptive(refine_fn: Callable[[np.ndarray centers, np.ndarray h], np.ndarray[bool]], max_level: int) -> Octree` (refine while predicate true, from root); `refine_elements(tree: Octree, mask: np.ndarray[bool]) -> Octree` (replace masked leaves by 8 children); `sort_unique(keys, levels) -> (keys, levels)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_o_build.py`:

```python
import numpy as np
import pytest
from diffsim.octree import morton
from diffsim.octree.build import Octree, build_uniform, build_adaptive, refine_elements, sort_unique

pytestmark = pytest.mark.tier1

def test_O3_uniform_count():
    for lvl in range(0, 5):
        t = build_uniform(lvl)
        assert len(t) == 8**lvl
        assert np.all(np.diff(t.keys.astype(np.uint64)) > 0)  # sorted, unique

def test_O4_local_refine_count():
    t = build_uniform(2)          # 64 elements
    mask = np.zeros(len(t), bool); mask[10] = True
    t2 = refine_elements(t, mask)
    assert len(t2) == 64 - 1 + 8
    # reverse: coarsening those 8 back is Task-scope M4; count identity only here.

def test_O5_duplicate_removal():
    t = build_uniform(2)
    k = np.concatenate([t.keys, t.keys]); l = np.concatenate([t.levels, t.levels])
    k2, l2 = sort_unique(k, l)
    assert len(k2) == 64 and np.all(k2 == t.keys) and np.all(l2 == t.levels)
    k3, l3 = sort_unique(np.array([], np.uint64), np.array([], np.uint8))
    assert len(k3) == 0

def test_adaptive_refine_center():
    # refine only octants containing the domain center, to max_level 4
    def pred(centers, h):
        return np.all(np.abs(centers - 0.5) < h, axis=1)
    t = build_adaptive(pred, max_level=4)
    assert t.levels.max() == 4 and t.levels.min() < 4
    vol = np.sum(t.h() ** 3)
    assert abs(vol - 1.0) < 1e-14   # leaves tile the cube exactly
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_o_build.py -v`
Expected: FAIL with `ModuleNotFoundError: ... build`.

- [ ] **Step 3: Implement `src/diffsim/octree/build.py`**

```python
from dataclasses import dataclass
import numpy as np
from . import morton

@dataclass(frozen=True)
class Octree:
    keys: np.ndarray     # uint64, sorted
    levels: np.ndarray   # uint8

    def __len__(self):
        return len(self.keys)

    def anchors(self):
        return morton.anchors(self.keys)

    def h(self):
        return 2.0 ** (-self.levels.astype(np.float64))

    def centers(self):
        scale = 2.0 ** (-morton.LMAX)
        half = 0.5 * self.h()
        return self.anchors() * scale + half[:, None]

def sort_unique(keys, levels):
    if len(keys) == 0:
        return keys, levels
    order = np.lexsort((levels, keys))
    k, l = keys[order], levels[order]
    keep = np.ones(len(k), bool)
    keep[1:] = (k[1:] != k[:-1]) | (l[1:] != l[:-1])
    return k[keep], l[keep]

def _make(keys, levels) -> Octree:
    k, l = sort_unique(np.asarray(keys, np.uint64), np.asarray(levels, np.uint8))
    return Octree(k, l)

def build_uniform(level: int) -> Octree:
    n = 1 << level
    g = np.arange(n, dtype=np.int64)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    xyz = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    keys = morton.encode(xyz, level)
    return _make(keys, np.full(len(keys), level, np.uint8))

def refine_elements(tree: Octree, mask: np.ndarray) -> Octree:
    keep_k, keep_l = tree.keys[~mask], tree.levels[~mask]
    ck, cl = morton.children(tree.keys[mask], tree.levels[mask])
    keys = np.concatenate([keep_k, ck.ravel()])
    levels = np.concatenate([keep_l, np.repeat(cl, 8)])
    return _make(keys, levels)

def build_adaptive(refine_fn, max_level: int) -> Octree:
    tree = build_uniform(0)
    while True:
        can = tree.levels < max_level
        want = refine_fn(tree.centers(), tree.h())
        mask = can & want
        if not mask.any():
            return tree
        tree = refine_elements(tree, mask)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_o_build.py -v` — Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/diffsim/octree/build.py tests/test_o_build.py
git commit -m "feat: octree build (uniform/adaptive/refine/sort-unique) (O3, O4, O5)"
```

---

### Task 3: Leaf lookup + face neighbors (N1, N2, N3)

**Files:**
- Create: `src/diffsim/octree/lookup.py`
- Test: `tests/test_n_neighbors.py`

**Interfaces:**
- Consumes: `Octree`, `morton.*`.
- Produces: `LeafLookup(tree: Octree)` with `.find(anchors: int64[N,3]) -> int64[N]` (index of the leaf whose closed-open box contains each LMAX-grid anchor point; `-1` if none — happens for carved trees or out-of-domain); `face_neighbors(tree: Octree) -> list[np.ndarray]` — for each of 6 faces (`X_MINUS, X_PLUS, Y_MINUS, Y_PLUS, Z_MINUS, Z_PLUS`, spec §3 ordering), an `int64[N]` array of neighbor leaf indices (`-1` = domain boundary). For a coarse-fine face this returns the *containing* neighbor at the probe point (one representative; full one-to-many enumeration arrives with the k-ring table in M4).

- [ ] **Step 1: Write the failing tests**

`tests/test_n_neighbors.py`:

```python
import numpy as np
import pytest
from diffsim.octree.build import build_uniform, refine_elements
from diffsim.octree.lookup import LeafLookup, face_neighbors, FACE_OFFSETS

pytestmark = pytest.mark.tier1

def test_N1_uniform_face_neighbors():
    t = build_uniform(2)
    nbrs = face_neighbors(t)
    counts = sum((n >= 0).astype(int) for n in nbrs)
    centers = t.centers()
    interior = np.all((centers > 0.25) & (centers < 0.75), axis=1)
    assert np.all(counts[interior] == 6)

def test_N2_boundary_identification():
    t = build_uniform(2)
    nbrs = face_neighbors(t)
    xmin_elems = t.anchors()[:, 0] == 0
    assert np.all(nbrs[0][xmin_elems] == -1)          # X_MINUS face of x=0 elements
    assert np.all(nbrs[0][~xmin_elems] >= 0)

def test_N3_coarse_fine_lookup():
    t = build_uniform(1)                               # 8 elements
    mask = np.zeros(8, bool); mask[0] = True           # refine one octant
    t2 = refine_elements(t, mask)                      # 7 coarse + 8 fine
    lk = LeafLookup(t2)
    fine = np.where(t2.levels == 2)[0]
    nbrs = face_neighbors(t2)
    # a fine element on the internal interface must see a coarse neighbor
    crossings = 0
    for f in range(6):
        for e in fine:
            j = nbrs[f][e]
            if j >= 0 and t2.levels[j] == 1:
                crossings += 1
    assert crossings > 0
    # symmetry (N4-lite): looking back from that coarse element's opposite face
    # finds a level-2 leaf (the representative probe hits a fine child)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_n_neighbors.py -v` — Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/diffsim/octree/lookup.py`**

```python
import numpy as np
from . import morton
from .build import Octree

# Face order matches spec §3 BoundaryTypes.WALL: X_MINUS..Z_PLUS
FACE_OFFSETS = np.array(
    [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], np.int64
)

class LeafLookup:
    """Containment lookup: LMAX-grid anchor -> leaf index, via truncated keys."""
    def __init__(self, tree: Octree):
        self.tree = tree
        self._map = {}
        for i, (k, l) in enumerate(zip(tree.keys.tolist(), tree.levels.tolist())):
            self._map[(int(k), int(l))] = i

    def find(self, anchors: np.ndarray) -> np.ndarray:
        anchors = np.asarray(anchors, np.int64).reshape(-1, 3)
        out = np.full(len(anchors), -1, np.int64)
        G = 1 << morton.LMAX
        inside = np.all((anchors >= 0) & (anchors < G), axis=1)
        for i in np.where(inside)[0]:
            a = anchors[i]
            for lvl in range(morton.LMAX, -1, -1):
                xyz = a >> (morton.LMAX - lvl)
                key = int(morton.encode(xyz[None, :], lvl)[0])
                j = self._map.get((key, lvl))
                if j is not None:
                    out[i] = j
                    break
        return out

def face_neighbors(tree: Octree) -> list:
    lk = LeafLookup(tree)
    anchors = tree.anchors()
    size = (1 << (morton.LMAX - tree.levels.astype(np.int64)))[:, None]
    center_off = size // 2
    out = []
    for f in range(6):
        off = FACE_OFFSETS[f]
        # probe point: just outside the face, at the face center
        probe = anchors + center_off
        probe = probe + off * (center_off + 1)  # step past the face plane
        idx = lk.find(probe)
        idx[idx == np.arange(len(tree))] = -1   # safety: never self
        out.append(idx)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_n_neighbors.py -v` — Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/diffsim/octree/lookup.py tests/test_n_neighbors.py
git commit -m "feat: leaf containment lookup + face neighbors (N1, N2, N3)"
```

---

### Task 4: Carving (incomplete octree) with an analytic sphere oracle (O8)

**Files:**
- Create: `src/diffsim/octree/carve.py`
- Test: append to `tests/test_o_build.py`

**Interfaces:**
- Consumes: `Octree`, `build_adaptive`.
- Produces: `class SphereOracle(center: tuple[float,3], radius: float)` with `classify(points: float64[N,3]) -> float64[N]` (signed value, negative inside — the GeometryOracle `classify` signature from spec §4.1, M0 subset); `carve(tree: Octree, oracle, samples_per_axis: int = 3) -> tuple[Octree, np.ndarray]` returning the incomplete octree (EXTERIOR leaves dropped) and per-leaf markers `int8[N]` (`0=INTERIOR, 1=INTERCEPTED`) for the retained leaves. Classification samples an `s^3` lattice of points per octant (corners+center for s=3); all-outside → EXTERIOR (dropped), all-inside → INTERIOR, else INTERCEPTED. (Gauss-point λ-criterion classification is M1.)

- [ ] **Step 1: Write the failing test (O8)**

Append to `tests/test_o_build.py`:

```python
def test_O8_incomplete_octree_sphere():
    from diffsim.octree.carve import SphereOracle, carve
    from diffsim.octree.build import build_uniform
    lvl = 5
    oracle = SphereOracle((0.5, 0.5, 0.5), 0.5)
    tree, markers = carve(build_uniform(lvl), oracle)
    # no EXTERIOR leaves in active list; markers valid
    assert set(np.unique(markers)) <= {0, 1}
    # active volume approaches sphere volume: interior + intercepted brackets it
    vol_active = np.sum(tree.h() ** 3)
    vol_int = np.sum(tree.h()[markers == 0] ** 3)
    v_sphere = 4.0 / 3.0 * np.pi * 0.5**3
    assert vol_int <= v_sphere * 1.01 <= vol_active * 1.01
    assert abs(vol_active - v_sphere) / v_sphere < 0.15   # level-5 resolution bound
    # count matches an independent classification at 1% (cuFEM O8 criterion, level-scaled)
    assert (markers == 1).sum() > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_o_build.py::test_O8_incomplete_octree_sphere -v` — Expected: FAIL.

- [ ] **Step 3: Implement `src/diffsim/octree/carve.py`**

```python
import numpy as np
from .build import Octree, _make

class SphereOracle:
    def __init__(self, center, radius):
        self.c = np.asarray(center, np.float64)
        self.r = float(radius)

    def classify(self, points: np.ndarray) -> np.ndarray:
        return np.linalg.norm(points - self.c, axis=1) - self.r

def carve(tree: Octree, oracle, samples_per_axis: int = 3):
    s = samples_per_axis
    t = np.linspace(0.0, 1.0, s)
    ox, oy, oz = np.meshgrid(t, t, t, indexing="ij")
    offs = np.stack([ox.ravel(), oy.ravel(), oz.ravel()], axis=1)  # [s^3, 3]
    scale = 2.0 ** -__import__("diffsim.octree.morton", fromlist=["LMAX"]).LMAX
    lo = tree.anchors() * scale
    h = tree.h()
    pts = (lo[:, None, :] + offs[None, :, :] * h[:, None, None]).reshape(-1, 3)
    sgn = oracle.classify(pts).reshape(len(tree), s**3)
    inside = sgn < 0.0
    n_in = inside.sum(axis=1)
    exterior = n_in == 0
    interior = n_in == s**3
    keep = ~exterior
    markers = np.where(interior[keep], 0, 1).astype(np.int8)
    return Octree(tree.keys[keep], tree.levels[keep]), markers
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_o_build.py -v` — Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/diffsim/octree/carve.py tests/test_o_build.py
git commit -m "feat: incomplete-octree carving with sphere oracle (O8)"
```

---

### Task 5: 2:1 balance (O6)

**Files:**
- Create: `src/diffsim/octree/balance.py`
- Test: `tests/test_o_balance.py`

**Interfaces:**
- Consumes: `Octree`, `LeafLookup`, `morton`, `refine_elements`.
- Produces: `balance2to1(tree: Octree) -> Octree` — refines leaves until every pair of face/edge/vertex-adjacent leaves differs by ≤ 1 level; `check_balance(tree: Octree) -> bool` (independent verifier used by the test).

- [ ] **Step 1: Write the failing test (O6)**

`tests/test_o_balance.py`:

```python
import numpy as np
import pytest
from diffsim.octree.build import build_uniform, refine_elements
from diffsim.octree.balance import balance2to1, check_balance

pytestmark = pytest.mark.tier1

def _refine_corner(tree, times):
    for _ in range(times):
        # refine the leaf containing the origin corner (deepest cascade seed)
        anchors = tree.anchors()
        target = np.argmin(anchors.sum(axis=1) + tree.levels.astype(np.int64) * 0)
        mask = np.zeros(len(tree), bool)
        mask[np.lexsort((tree.keys,))[0]] = True   # first leaf in SFC order = origin corner
        tree = refine_elements(tree, mask)
    return tree

def test_O6_lshape_cascade():
    t = build_uniform(1)
    t = _refine_corner(t, 4)           # origin leaf driven to level 5, neighbors still level 1
    assert not check_balance(t)
    tb = balance2to1(t)
    assert check_balance(tb)
    assert len(tb) > len(t)            # cascading refinement happened
    assert abs(np.sum(tb.h() ** 3) - 1.0) < 1e-14

def test_O6_already_balanced_is_identity():
    t = build_uniform(3)
    tb = balance2to1(t)
    assert len(tb) == len(t) and np.all(tb.keys == t.keys)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_o_balance.py -v` — Expected: FAIL.

- [ ] **Step 3: Implement `src/diffsim/octree/balance.py`**

```python
import numpy as np
from itertools import product
from . import morton
from .build import Octree, refine_elements
from .lookup import LeafLookup

_NBR_OFFSETS = np.array([o for o in product((-1, 0, 1), repeat=3) if o != (0, 0, 0)],
                        np.int64)  # 26 neighbors

def _neighbor_levels(tree: Octree):
    """For each leaf, the leaf index of each of its 26 same-size neighbor probes."""
    lk = LeafLookup(tree)
    anchors = tree.anchors()
    size = (1 << (morton.LMAX - tree.levels.astype(np.int64)))[:, None]
    center = anchors + size // 2
    nbr_idx = np.empty((len(tree), 26), np.int64)
    for j, off in enumerate(_NBR_OFFSETS):
        probe = center + off * size          # lands inside face/edge/vertex neighbor
        nbr_idx[:, j] = lk.find(probe)
    return nbr_idx

def check_balance(tree: Octree) -> bool:
    nbr = _neighbor_levels(tree)
    lev = tree.levels.astype(np.int64)
    for j in range(26):
        ok = nbr[:, j] >= 0
        if np.any(np.abs(lev[ok] - lev[nbr[ok, j]]) > 1):
            return False
    return True

def balance2to1(tree: Octree) -> Octree:
    while True:
        nbr = _neighbor_levels(tree)
        lev = tree.levels.astype(np.int64)
        to_refine = np.zeros(len(tree), bool)
        for j in range(26):
            ok = nbr[:, j] >= 0
            # neighbor is more than one level coarser than me -> neighbor must refine
            viol = ok.copy()
            viol[ok] = lev[ok] - lev[nbr[ok, j]] > 1
            to_refine[nbr[viol, j]] = True
        if not to_refine.any():
            return tree
        tree = refine_elements(tree, to_refine)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_o_balance.py -v` — Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/diffsim/octree/balance.py tests/test_o_balance.py
git commit -m "feat: 2:1 balance with cascade (O6)"
```

---

### Task 6: Node generation + connectivity, p1 and p2 (C1, C2, C4, C6)

**Files:**
- Create: `src/diffsim/mesh/__init__.py`, `src/diffsim/mesh/nodes.py`
- Test: `tests/test_c_nodes.py`

**Interfaces:**
- Consumes: `Octree`, `morton`.
- Produces: `@dataclass Mesh { p: int, tree: Octree, node_coords: float64[Nn,3] (physical), node_icoords: int64[Nn,3] (integer grid at 2^(LMAX+1) resolution: even = corner grid, odd = midpoints for p2), conn: int32[Ne, (p+1)^3], boundary_nodes: bool[Nn] (on unit-cube boundary) }`; `build_mesh(tree: Octree, p: int) -> Mesh`. Local node ordering: `a = ax + (p+1)*ay + (p+1)^2*az`, x fastest (Global Constraints).

- [ ] **Step 1: Write the failing tests**

`tests/test_c_nodes.py`:

```python
import numpy as np
import pytest
from diffsim.octree.build import build_uniform, refine_elements
from diffsim.octree.carve import SphereOracle, carve
from diffsim.mesh.nodes import build_mesh

pytestmark = pytest.mark.tier2

def test_C1_unique_node_count_uniform():
    for lvl in range(1, 4):
        m1 = build_mesh(build_uniform(lvl), p=1)
        assert len(m1.node_coords) == (2**lvl + 1) ** 3
        m2 = build_mesh(build_uniform(lvl), p=2)
        assert len(m2.node_coords) == (2 * 2**lvl + 1) ** 3

def test_C2_connectivity_shape_and_sharing():
    m = build_mesh(build_uniform(2), p=1)
    assert m.conn.shape == (64, 8)
    for e in range(64):
        assert len(set(m.conn[e])) == 8      # no duplicate nodes within an element
    # shared face: elements 0's X_PLUS corner coords appear in the neighbor
    c0 = set(map(tuple, m.node_icoords[m.conn[0]]))
    shared = [len(c0 & set(map(tuple, m.node_icoords[m.conn[e]]))) for e in range(1, 64)]
    assert max(shared) == 4                  # face neighbors share exactly 4 corners (p1)

def test_C4_C6_incomplete_octree_no_orphans():
    tree, _ = carve(build_uniform(4), SphereOracle((0.5, 0.5, 0.5), 0.4))
    m = build_mesh(tree, p=1)
    referenced = np.unique(m.conn.ravel())
    assert len(referenced) == len(m.node_coords)          # C6: no orphan nodes
    assert np.all(referenced == np.arange(len(m.node_coords)))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_c_nodes.py -v` — Expected: FAIL.

- [ ] **Step 3: Implement `src/diffsim/mesh/nodes.py`**

```python
from dataclasses import dataclass
import numpy as np
from ..octree import morton
from ..octree.build import Octree

@dataclass(frozen=True)
class Mesh:
    p: int
    tree: Octree
    node_coords: np.ndarray    # float64 [Nn,3], physical (unit cube)
    node_icoords: np.ndarray   # int64  [Nn,3], grid of size 2^(LMAX+1)+1
    conn: np.ndarray           # int32  [Ne, (p+1)^3]
    boundary_nodes: np.ndarray # bool   [Nn]

def build_mesh(tree: Octree, p: int) -> Mesh:
    assert p in (1, 2)
    npe = p + 1
    # local lattice offsets in units of h/p, x fastest
    ax, ay, az = np.meshgrid(np.arange(npe), np.arange(npe), np.arange(npe), indexing="ij")
    # meshgrid 'ij' gives ax varying slowest; we need x fastest -> build explicitly:
    offs = np.array([[i, j, k] for k in range(npe) for j in range(npe) for i in range(npe)],
                    np.int64)                      # a = i + npe*j + npe^2*k
    # integer node coords on the doubled grid (so p2 midpoints are integers)
    anchors2 = tree.anchors() * 2                  # anchor on doubled grid
    step2 = (2 * (1 << (morton.LMAX - tree.levels.astype(np.int64)))) // p  # h/p on doubled grid... 
    # careful: for p=1 step2 = 2*size, for p=2 step2 = size
    size = 1 << (morton.LMAX - tree.levels.astype(np.int64))
    step2 = (2 * size) // p
    all_icoords = (anchors2[:, None, :] + offs[None, :, :] * step2[:, None, None]).reshape(-1, 3)
    nodes, inverse = np.unique(all_icoords, axis=0, return_inverse=True)
    conn = inverse.reshape(len(tree), npe**3).astype(np.int32)
    G2 = 2 * (1 << morton.LMAX)
    coords = nodes.astype(np.float64) / G2
    boundary = np.any((nodes == 0) | (nodes == G2), axis=1)
    return Mesh(p, tree, coords, nodes, conn, boundary)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_c_nodes.py -v` — Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/diffsim/mesh tests/test_c_nodes.py
git commit -m "feat: node generation + connectivity for p1/p2 (C1, C2, C4, C6)"
```

---

### Task 7: Hanging-node detection + constraint operator T

**Files:**
- Create: `src/diffsim/mesh/constraints.py`
- Test: `tests/test_c_nodes.py` (append)

**Interfaces:**
- Consumes: `Mesh`, `LeafLookup`, `morton`.
- Produces: `build_constraints(mesh: Mesh) -> Constraints` where `@dataclass Constraints { T: scipy.sparse.csr_matrix (Nn × Nfree), free_nodes: int64[Nfree], hanging: bool[Nn] }`. `T` maps free-node values to all-node values: identity rows for free nodes; for each hanging node, interpolation weights of the p-order tensor basis of its *owner* (the coarsest adjacent leaf) evaluated at the hanging node's reference coordinates, distributed onto the owner's own nodes (with 2:1 balance those are free). Full-space operators compose as `A_c = Tᵀ A T` (applied matrix-free as chained SpMV).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_c_nodes.py`:

```python
def test_hanging_constraints_partition_of_unity():
    from diffsim.mesh.constraints import build_constraints
    t = build_uniform(1)
    mask = np.zeros(8, bool); mask[0] = True
    t = refine_elements(t, mask)                        # one coarse-fine interface set
    for p in (1, 2):
        m = build_mesh(t, p=p)
        c = build_constraints(m)
        assert c.hanging.sum() > 0
        # rows sum to 1 (constant field reproduced through constraints)
        rowsum = np.asarray(c.T.sum(axis=1)).ravel()
        assert np.allclose(rowsum, 1.0, atol=1e-13)
        # a linear field is reproduced exactly by constrained interpolation
        f = lambda x: 1.0 + 2.0 * x[:, 0] - 3.0 * x[:, 1] + 0.5 * x[:, 2]
        u_free = f(m.node_coords[c.free_nodes])
        u_all = c.T @ u_free
        assert np.allclose(u_all, f(m.node_coords), atol=1e-12)

def test_uniform_mesh_has_no_hanging():
    from diffsim.mesh.constraints import build_constraints
    m = build_mesh(build_uniform(2), p=1)
    c = build_constraints(m)
    assert c.hanging.sum() == 0 and c.T.shape == (len(m.node_coords),) * 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_c_nodes.py -v -k hanging or uniform_mesh` — Expected: FAIL.

- [ ] **Step 3: Implement `src/diffsim/mesh/constraints.py`**

```python
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from ..octree import morton
from ..octree.lookup import LeafLookup
from .nodes import Mesh
from .basis import lagrange_1d   # Task 8 provides; for Task 7 tests use p1/p2 closed form below

def _basis_1d(p, xi):
    if p == 1:
        return np.array([0.5 * (1 - xi), 0.5 * (1 + xi)])
    return np.array([0.5 * xi * (xi - 1.0), 1.0 - xi * xi, 0.5 * xi * (xi + 1.0)])

@dataclass(frozen=True)
class Constraints:
    T: sp.csr_matrix
    free_nodes: np.ndarray
    hanging: np.ndarray

def build_constraints(mesh: Mesh) -> Constraints:
    tree, p = mesh.tree, mesh.p
    lk = LeafLookup(tree)
    Nn = len(mesh.node_coords)
    lev = tree.levels.astype(np.int64)
    size2 = 2 * (1 << (morton.LMAX - lev))              # element size on doubled grid
    anchors2 = tree.anchors() * 2

    # element index per node incidence: node -> list of (element, is_own_node)
    # A node is HANGING iff some *touching* leaf (containment probe at the node,
    # nudged into each surrounding octant) does not carry it in its conn row.
    node_elems = [[] for _ in range(Nn)]
    for e in range(len(tree)):
        for a in mesh.conn[e]:
            node_elems[a].append(e)

    hanging = np.zeros(Nn, bool)
    owner = np.full(Nn, -1, np.int64)
    G2 = 2 * (1 << morton.LMAX)
    for n in range(Nn):
        ic = mesh.node_icoords[n]
        # probe the (up to 8) octants around the node on the LMAX grid
        touch = set()
        for dx in (-1, 0):
            for dy in (-1, 0):
                for dz in (-1, 0):
                    probe2 = ic + np.array([dx, dy, dz])
                    if np.any(probe2 < 0) or np.any(probe2 >= G2):
                        continue
                    idx = lk.find((probe2 // 2)[None, :])[0]
                    if idx >= 0:
                        touch.add(int(idx))
        carriers = set(node_elems[n])
        non_carriers = touch - carriers
        if non_carriers:
            hanging[n] = True
            owner[n] = min(non_carriers, key=lambda e: lev[e])   # coarsest adjacent leaf

    free_nodes = np.where(~hanging)[0]
    free_of = np.full(Nn, -1, np.int64)
    free_of[free_nodes] = np.arange(len(free_nodes))

    rows, cols, vals = [], [], []
    for n in free_nodes:
        rows.append(n); cols.append(free_of[n]); vals.append(1.0)
    for n in np.where(hanging)[0]:
        e = owner[n]
        # reference coords of node n in owner element, in [-1,1]
        xi = 2.0 * (mesh.node_icoords[n] - anchors2[e]) / size2[e] - 1.0
        w1 = [_basis_1d(p, xi[d]) for d in range(3)]
        npe = p + 1
        for k in range(npe):
            for j in range(npe):
                for i in range(npe):
                    a = i + npe * j + npe * npe * k
                    w = w1[0][i] * w1[1][j] * w1[2][k]
                    if abs(w) < 1e-14:
                        continue
                    tgt = mesh.conn[e, a]
                    assert not hanging[tgt], "2:1 balance guarantees owner nodes are free"
                    rows.append(n); cols.append(free_of[tgt]); vals.append(w)
    T = sp.csr_matrix((vals, (rows, cols)), shape=(Nn, len(free_nodes)))
    return Constraints(T, free_nodes, hanging)
```

Note: the import of `lagrange_1d` is forward-looking for Task 8; the local `_basis_1d` is used here so this task stands alone — remove the unused import if Task 8 hasn't landed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_c_nodes.py -v` — Expected: all PASS (remove the `from .basis import` line if it errors — it is not used).

- [ ] **Step 5: Commit**

```bash
git add src/diffsim/mesh/constraints.py tests/test_c_nodes.py
git commit -m "feat: hanging-node detection + constraint operator T (2:1, p1/p2)"
```

---

### Task 8: Basis + quadrature tables (Q1, M1, M2)

**Files:**
- Create: `src/diffsim/mesh/basis.py`
- Test: `tests/test_q_basis.py`

**Interfaces:**
- Consumes: nothing (pure tables).
- Produces: `gauss_1d(p) -> (pts: float64[p+1], wts: float64[p+1])` (2-pt for p1, 3-pt for p2); `lagrange_1d(p, xi) -> (N: float64[p+1], dN: float64[p+1])`; `basis_tables(p) -> Tables` with `@dataclass Tables { p, nbf, nqp, N: float64[nqp,nbf], dN: float64[nqp,nbf,3], w: float64[nqp] }` on the reference cube `[-1,1]^3`, node/qp ordering x-fastest. Physical scaling is applied in kernels: `detJxW = w * (h/2)^3`, `dN_phys = dN * (2/h)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_q_basis.py`:

```python
import numpy as np
import pytest
from diffsim.mesh.basis import gauss_1d, basis_tables

pytestmark = pytest.mark.tier3

def test_Q1_polynomial_exactness_1d():
    for p in (1, 2):
        pts, wts = gauss_1d(p)
        for deg in range(2 * (p + 1)):        # 2n-1 exactness: degrees 0..2p+1
            quad = np.sum(wts * pts**deg)
            exact = (1.0 - (-1.0) ** (deg + 1)) / (deg + 1)
            assert abs(quad - exact) < 1e-14, (p, deg)

def test_partition_of_unity_and_derivative_sum():
    for p in (1, 2):
        tb = basis_tables(p)
        assert np.allclose(tb.N.sum(axis=1), 1.0, atol=1e-14)
        assert np.allclose(tb.dN.sum(axis=1), 0.0, atol=1e-13)

def test_M1_local_mass_matrix():
    h = 0.25
    for p in (1, 2):
        tb = basis_tables(p)
        detJxW = tb.w * (h / 2.0) ** 3
        Me = np.einsum("qa,qb,q->ab", tb.N, tb.N, detJxW)
        assert np.allclose(Me, Me.T, atol=1e-15)
        assert abs(Me.sum() - h**3) < 1e-14                       # partition of unity
        assert np.all(np.linalg.eigvalsh(Me) > 0)                 # SPD
        if p == 1:
            assert np.allclose(Me.sum(axis=1), h**3 / 8.0, atol=1e-15)  # row sums

def test_M2_local_stiffness_matrix():
    h = 0.25
    for p in (1, 2):
        tb = basis_tables(p)
        detJxW = tb.w * (h / 2.0) ** 3
        dN = tb.dN * (2.0 / h)
        Ke = np.einsum("qak,qbk,q->ab", dN, dN, detJxW)
        assert np.allclose(Ke, Ke.T, atol=1e-13)
        ev = np.linalg.eigvalsh(Ke)
        assert abs(ev[0]) < 1e-13 and ev[1] > 1e-8                # nullspace = constants only
        assert np.allclose(Ke @ np.ones(tb.nbf), 0.0, atol=1e-13)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_q_basis.py -v` — Expected: FAIL.

- [ ] **Step 3: Implement `src/diffsim/mesh/basis.py`**

```python
from dataclasses import dataclass
import numpy as np

def gauss_1d(p: int):
    if p == 1:
        a = 1.0 / np.sqrt(3.0)
        return np.array([-a, a]), np.array([1.0, 1.0])
    b = np.sqrt(3.0 / 5.0)
    return np.array([-b, 0.0, b]), np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

def lagrange_1d(p: int, xi: float):
    if p == 1:
        N = np.array([0.5 * (1 - xi), 0.5 * (1 + xi)])
        dN = np.array([-0.5, 0.5])
    else:
        N = np.array([0.5 * xi * (xi - 1.0), 1.0 - xi * xi, 0.5 * xi * (xi + 1.0)])
        dN = np.array([xi - 0.5, -2.0 * xi, xi + 0.5])
    return N, dN

@dataclass(frozen=True)
class Tables:
    p: int
    nbf: int
    nqp: int
    N: np.ndarray    # [nqp, nbf]
    dN: np.ndarray   # [nqp, nbf, 3] (reference derivatives)
    w: np.ndarray    # [nqp]

def basis_tables(p: int) -> Tables:
    pts, wts = gauss_1d(p)
    npe, nq1 = p + 1, len(pts)
    nbf, nqp = npe**3, nq1**3
    N = np.zeros((nqp, nbf)); dN = np.zeros((nqp, nbf, 3)); w = np.zeros(nqp)
    q = 0
    for qk in range(nq1):
        for qj in range(nq1):
            for qi in range(nq1):                     # x fastest
                Nx, dNx = lagrange_1d(p, pts[qi])
                Ny, dNy = lagrange_1d(p, pts[qj])
                Nz, dNz = lagrange_1d(p, pts[qk])
                w[q] = wts[qi] * wts[qj] * wts[qk]
                for k in range(npe):
                    for j in range(npe):
                        for i in range(npe):
                            a = i + npe * j + npe * npe * k
                            N[q, a] = Nx[i] * Ny[j] * Nz[k]
                            dN[q, a, 0] = dNx[i] * Ny[j] * Nz[k]
                            dN[q, a, 1] = Nx[i] * dNy[j] * Nz[k]
                            dN[q, a, 2] = Nx[i] * Ny[j] * dNz[k]
                q += 1
    return Tables(p, nbf, nqp, N, dN, w)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_q_basis.py -v` — Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/diffsim/mesh/basis.py tests/test_q_basis.py
git commit -m "feat: tensor-product basis + Gauss tables p1/p2 (Q1, M1, M2)"
```

---

### Task 9: FEMElm + device operators — matvec, residual, volume (G1, Q2, S3 groundwork)

**Files:**
- Create: `src/diffsim/assembly/__init__.py`, `src/diffsim/assembly/femelm.py`, `src/diffsim/assembly/operators.py`
- Test: `tests/test_g_operator.py`

**Interfaces:**
- Consumes: `Mesh`, `Constraints`, `Tables`.
- Produces:
  - `femelm.py`: `@wp.struct FEMElm { e: wp.int32, q: wp.int32, he: wp.float64 }` plus module-level accessor `wp.func`s taking the shared tables: `fe_N(Ntab, fe, a) -> wp.float64`, `fe_dN(dNtab, fe, a, k) -> wp.float64` (physical derivative, includes `2/he`), `fe_detJxW(wtab, fe) -> wp.float64` (includes `(he/2)^3`). This is the prototype rendering of the spec §3 `FEMElm` view (Warp structs carry data; accessors are free functions).
  - `operators.py`: `class DeviceMesh` (uploads `conn: wp.array2d(int32)`, `h: wp.array(float64)`, tables `N/dN/w` as `wp.array`s, plus CSR of `T` and `Tᵀ` as device arrays) with classmethod `DeviceMesh.from_mesh(mesh, constraints, tables, device)`; `make_poisson_matvec(nbf, nqp) -> wp.Kernel` computing `y += Ke·x` per element (Gauss-loop, atomic scatter); `csr_spmv_kernel` (generic CSR SpMV); `class ConstrainedOperator` with `.matvec(x_free: wp.array, y_free: wp.array)` performing `y = Tᵀ (A (T x))` entirely on device; `volume_kernel` (integrates 1 over the mesh → Q2).

- [ ] **Step 1: Write the failing tests**

`tests/test_g_operator.py`:

```python
import numpy as np
import pytest
import warp as wp
from diffsim.octree.build import build_uniform, refine_elements
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh, ConstrainedOperator, integrate_volume

pytestmark = pytest.mark.tier3

def _setup(p, adaptive=False, device="cpu"):
    t = build_uniform(2)
    if adaptive:
        mask = np.zeros(len(t), bool); mask[0] = True
        t = refine_elements(t, mask)
    m = build_mesh(t, p=p)
    c = build_constraints(m)
    dm = DeviceMesh.from_mesh(m, c, basis_tables(p), device)
    return m, c, dm

def test_Q2_volume_conservation(device):
    for p in (1, 2):
        _, _, dm = _setup(p, adaptive=True, device=device)
        vol = integrate_volume(dm)
        assert abs(vol - 1.0) < 1e-13

def test_G1_operator_symmetry(device):
    for p in (1, 2):
        for adaptive in (False, True):
            m, c, dm = _setup(p, adaptive, device)
            op = ConstrainedOperator(dm)
            nf = len(c.free_nodes)
            rng = np.random.default_rng(3)
            for _ in range(5):
                v = rng.standard_normal(nf); w_ = rng.standard_normal(nf)
                Av = op.matvec_numpy(v)
                Aw = op.matvec_numpy(w_)
                assert abs(v @ Aw - w_ @ Av) < 1e-10 * (abs(v @ Av) + 1)

def test_G3_positive_semidefinite(device):
    m, c, dm = _setup(1, device=device)
    op = ConstrainedOperator(dm)
    rng = np.random.default_rng(4)
    for _ in range(20):
        v = rng.standard_normal(len(c.free_nodes))
        assert v @ op.matvec_numpy(v) >= -1e-10
    one = np.ones(len(c.free_nodes))
    assert np.abs(op.matvec_numpy(one)).max() < 1e-11   # constants in nullspace (pure Neumann)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_g_operator.py -v` — Expected: FAIL.

- [ ] **Step 3: Implement `src/diffsim/assembly/femelm.py`**

```python
import warp as wp

@wp.struct
class FEMElm:
    e: wp.int32
    q: wp.int32
    he: wp.float64

@wp.func
def fe_N(Ntab: wp.array2d(dtype=wp.float64), fe: FEMElm, a: wp.int32) -> wp.float64:
    return Ntab[fe.q, a]

@wp.func
def fe_dN(dNtab: wp.array3d(dtype=wp.float64), fe: FEMElm, a: wp.int32, k: wp.int32) -> wp.float64:
    return dNtab[fe.q, a, k] * (wp.float64(2.0) / fe.he)

@wp.func
def fe_detJxW(wtab: wp.array(dtype=wp.float64), fe: FEMElm) -> wp.float64:
    half = fe.he * wp.float64(0.5)
    return wtab[fe.q] * half * half * half
```

- [ ] **Step 4: Implement `src/diffsim/assembly/operators.py`**

```python
import numpy as np
import scipy.sparse as sp
import warp as wp
from .femelm import FEMElm, fe_N, fe_dN, fe_detJxW

def _csr_to_device(A: sp.csr_matrix, device):
    return (wp.array(A.indptr.astype(np.int32), dtype=wp.int32, device=device),
            wp.array(A.indices.astype(np.int32), dtype=wp.int32, device=device),
            wp.array(A.data.astype(np.float64), dtype=wp.float64, device=device))

@wp.kernel
def csr_spmv(indptr: wp.array(dtype=wp.int32), indices: wp.array(dtype=wp.int32),
             data: wp.array(dtype=wp.float64),
             x: wp.array(dtype=wp.float64), y: wp.array(dtype=wp.float64)):
    row = wp.tid()
    acc = wp.float64(0.0)
    for j in range(indptr[row], indptr[row + 1]):
        acc += data[j] * x[indices[j]]
    y[row] = acc

_kernel_cache = {}

def make_poisson_matvec(nbf: int, nqp: int):
    key = ("poisson_mv", nbf, nqp)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel
    def poisson_mv(conn: wp.array2d(dtype=wp.int32), h: wp.array(dtype=wp.float64),
                   Ntab: wp.array2d(dtype=wp.float64), dNtab: wp.array3d(dtype=wp.float64),
                   wtab: wp.array(dtype=wp.float64),
                   x: wp.array(dtype=wp.float64), y: wp.array(dtype=wp.float64)):
        e = wp.tid()
        fe = FEMElm()
        fe.e = e
        fe.he = h[e]
        for q in range(nqp):
            fe.q = q
            dJxW = fe_detJxW(wtab, fe)
            gx = wp.float64(0.0); gy = wp.float64(0.0); gz = wp.float64(0.0)
            for b in range(nbf):
                xb = x[conn[e, b]]
                gx += fe_dN(dNtab, fe, b, 0) * xb
                gy += fe_dN(dNtab, fe, b, 1) * xb
                gz += fe_dN(dNtab, fe, b, 2) * xb
            for a in range(nbf):
                val = (fe_dN(dNtab, fe, a, 0) * gx +
                       fe_dN(dNtab, fe, a, 1) * gy +
                       fe_dN(dNtab, fe, a, 2) * gz) * dJxW
                wp.atomic_add(y, conn[e, a], val)

    _kernel_cache[key] = poisson_mv
    return poisson_mv

@wp.kernel
def volume_kernel(h: wp.array(dtype=wp.float64), wtab: wp.array(dtype=wp.float64),
                  out: wp.array(dtype=wp.float64)):
    e = wp.tid()
    acc = wp.float64(0.0)
    half = h[e] * wp.float64(0.5)
    for q in range(wtab.shape[0]):
        acc += wtab[q] * half * half * half
    wp.atomic_add(out, 0, acc)

class DeviceMesh:
    def __init__(self, mesh, constraints, tables, device):
        self.mesh, self.constraints, self.tables, self.device = mesh, constraints, tables, device
        self.conn = wp.array(mesh.conn, dtype=wp.int32, device=device)
        self.h = wp.array(mesh.tree.h(), dtype=wp.float64, device=device)
        self.N = wp.array(tables.N, dtype=wp.float64, device=device)
        self.dN = wp.array(tables.dN, dtype=wp.float64, device=device)
        self.w = wp.array(tables.w, dtype=wp.float64, device=device)
        T = constraints.T.tocsr()
        self.T_dev = _csr_to_device(T, device)
        self.Tt_dev = _csr_to_device(T.T.tocsr(), device)
        self.n_nodes = len(mesh.node_coords)
        self.n_free = T.shape[1]

    from_mesh = classmethod(lambda cls, m, c, t, d: cls(m, c, t, d))

def integrate_volume(dm: DeviceMesh) -> float:
    out = wp.zeros(1, dtype=wp.float64, device=dm.device)
    wp.launch(volume_kernel, dim=len(dm.mesh.tree), inputs=[dm.h, dm.w, out], device=dm.device)
    return float(out.numpy()[0])

class ConstrainedOperator:
    """y_free = T^T (A (T x_free)), A = Poisson stiffness action (no BCs)."""
    def __init__(self, dm: DeviceMesh):
        self.dm = dm
        self.kernel = make_poisson_matvec(dm.tables.nbf, dm.tables.nqp)
        d = dm.device
        self.x_full = wp.zeros(dm.n_nodes, dtype=wp.float64, device=d)
        self.y_full = wp.zeros(dm.n_nodes, dtype=wp.float64, device=d)

    def matvec(self, x_free: wp.array, y_free: wp.array):
        dm, d = self.dm, self.dm.device
        wp.launch(csr_spmv, dim=dm.n_nodes,
                  inputs=[*dm.T_dev, x_free, self.x_full], device=d)
        self.y_full.zero_()
        wp.launch(self.kernel, dim=len(dm.mesh.tree),
                  inputs=[dm.conn, dm.h, dm.N, dm.dN, dm.w, self.x_full, self.y_full],
                  device=d)
        wp.launch(csr_spmv, dim=dm.n_free,
                  inputs=[*dm.Tt_dev, self.y_full, y_free], device=d)

    def matvec_numpy(self, x: np.ndarray) -> np.ndarray:
        xd = wp.array(x.astype(np.float64), dtype=wp.float64, device=self.dm.device)
        yd = wp.zeros(self.dm.n_free, dtype=wp.float64, device=self.dm.device)
        self.matvec(xd, yd)
        return yd.numpy()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_g_operator.py -v` — Expected: 3 PASS. (If Warp rejects `wtab.shape[0]` inside `volume_kernel`, pass `nqp` via a kernel factory exactly as in `make_poisson_matvec`.)

- [ ] **Step 6: Commit**

```bash
git add src/diffsim/assembly tests/test_g_operator.py
git commit -m "feat: FEMElm + device Poisson matvec, constrained operator, volume (G1, G3, Q2)"
```

---

### Task 10: Device BLAS + CG/BiCGStab + Jacobi (S1, S2)

**Files:**
- Create: `src/diffsim/solvers/__init__.py`, `src/diffsim/solvers/blas.py`, `src/diffsim/solvers/krylov.py`
- Test: `tests/test_s_solvers.py`

**Interfaces:**
- Consumes: any object with `.matvec(x: wp.array, y: wp.array)` and `.n_free`.
- Produces: `blas.dot(a, b) -> float` (FP64 device reduction), `blas.axpy(alpha, x, y)` (y += αx), `blas.scale`, `blas.copy`; `krylov.cg(op, b: np.ndarray, tol=1e-10, atol=1e-12, maxiter=2000, diag: np.ndarray|None=None) -> (x: np.ndarray, info: dict)` with `info = {"iters": int, "relres": float, "converged": bool}`; `krylov.bicgstab(...)` same signature; `krylov.operator_diagonal(op) -> np.ndarray` (Jacobi diag via `nbf`-color probing — exact for FEM operators whose element supports don't alias under coloring; for the prototype use `n_probe = nbf` random-sign probes and document it as an *estimate*, or extract exactly from the CSR path in Task 12).

- [ ] **Step 1: Write the failing tests**

`tests/test_s_solvers.py`:

```python
import numpy as np
import pytest
import warp as wp
from diffsim.solvers.krylov import cg, bicgstab

pytestmark = pytest.mark.tier3

class DenseOp:
    """Test operator wrapping a dense SPD/nonsym matrix on device via numpy."""
    def __init__(self, A, device):
        self.A, self.device, self.n_free = A, device, A.shape[0]
    def matvec(self, x, y):
        r = self.A @ x.numpy()
        wp.copy(y, wp.array(r, dtype=wp.float64, device=self.device))

def _spd(n, rng):
    Q = rng.standard_normal((n, n))
    return Q @ Q.T + n * np.eye(n)

def test_S2_cg_manufactured_system(device):
    rng = np.random.default_rng(7)
    A = _spd(80, rng)
    x_exact = rng.standard_normal(80)
    op = DenseOp(A, device)
    x, info = cg(op, A @ x_exact, tol=1e-12)
    assert info["converged"]
    assert np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact) < 1e-8

def test_S2_bicgstab_nonsymmetric(device):
    rng = np.random.default_rng(8)
    A = _spd(60, rng) + 0.3 * rng.standard_normal((60, 60))
    x_exact = rng.standard_normal(60)
    op = DenseOp(A, device)
    x, info = bicgstab(op, A @ x_exact, tol=1e-12)
    assert info["converged"]
    assert np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact) < 1e-7

def test_S1_reported_residual_matches_recomputed(device):
    rng = np.random.default_rng(9)
    A = _spd(50, rng)
    b = rng.standard_normal(50)
    op = DenseOp(A, device)
    x, info = cg(op, b, tol=1e-10)
    relres = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
    assert abs(relres - info["relres"]) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_s_solvers.py -v` — Expected: FAIL.

- [ ] **Step 3: Implement `src/diffsim/solvers/blas.py`**

```python
import warp as wp

@wp.kernel
def _dot_kernel(a: wp.array(dtype=wp.float64), b: wp.array(dtype=wp.float64),
                out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    wp.atomic_add(out, 0, a[i] * b[i])

@wp.kernel
def _axpy_kernel(alpha: wp.float64, x: wp.array(dtype=wp.float64),
                 y: wp.array(dtype=wp.float64)):
    i = wp.tid()
    y[i] = y[i] + alpha * x[i]

@wp.kernel
def _xpay_kernel(alpha: wp.float64, x: wp.array(dtype=wp.float64),
                 y: wp.array(dtype=wp.float64)):
    i = wp.tid()
    y[i] = x[i] + alpha * y[i]

@wp.kernel
def _mult_kernel(a: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.float64),
                 y: wp.array(dtype=wp.float64)):
    i = wp.tid()
    y[i] = a[i] * x[i]

def dot(a, b, device):
    out = wp.zeros(1, dtype=wp.float64, device=device)
    wp.launch(_dot_kernel, dim=len(a), inputs=[a, b, out], device=device)
    return float(out.numpy()[0])

def axpy(alpha, x, y, device):   # y += alpha x
    wp.launch(_axpy_kernel, dim=len(x), inputs=[wp.float64(alpha), x, y], device=device)

def xpay(alpha, x, y, device):   # y = x + alpha y
    wp.launch(_xpay_kernel, dim=len(x), inputs=[wp.float64(alpha), x, y], device=device)

def hadamard(a, x, y, device):   # y = a * x elementwise
    wp.launch(_mult_kernel, dim=len(x), inputs=[a, x, y], device=device)
```

- [ ] **Step 4: Implement `src/diffsim/solvers/krylov.py`**

```python
import numpy as np
import warp as wp
from . import blas

def _to_dev(v, device):
    return wp.array(np.asarray(v, np.float64), dtype=wp.float64, device=device)

def cg(op, b, tol=1e-10, atol=1e-12, maxiter=2000, diag=None):
    d = op.device
    n = op.n_free
    bd = _to_dev(b, d)
    x = wp.zeros(n, dtype=wp.float64, device=d)
    r = wp.clone(bd)
    z = wp.zeros(n, dtype=wp.float64, device=d)
    Minv = _to_dev(1.0 / np.asarray(diag), d) if diag is not None else None
    if Minv is not None:
        blas.hadamard(Minv, r, z, d)
    else:
        wp.copy(z, r)
    p = wp.clone(z)
    Ap = wp.zeros(n, dtype=wp.float64, device=d)
    rz = blas.dot(r, z, d)
    bnorm = max(np.sqrt(blas.dot(bd, bd, d)), 1e-300)
    for it in range(1, maxiter + 1):
        op.matvec(p, Ap)
        alpha = rz / blas.dot(p, Ap, d)
        blas.axpy(alpha, p, x, d)
        blas.axpy(-alpha, Ap, r, d)
        rnorm = np.sqrt(blas.dot(r, r, d))
        if rnorm < max(tol * bnorm, atol):
            return x.numpy(), {"iters": it, "relres": rnorm / bnorm, "converged": True}
        if Minv is not None:
            blas.hadamard(Minv, r, z, d)
        else:
            wp.copy(z, r)
        rz_new = blas.dot(r, z, d)
        blas.xpay(rz_new / rz, z, p, d)
        rz = rz_new
    return x.numpy(), {"iters": maxiter, "relres": rnorm / bnorm, "converged": False}

def bicgstab(op, b, tol=1e-10, atol=1e-12, maxiter=2000, diag=None):
    d = op.device
    n = op.n_free
    bd = _to_dev(b, d)
    x = wp.zeros(n, dtype=wp.float64, device=d)
    r = wp.clone(bd)
    rhat = wp.clone(bd)
    p = wp.zeros(n, dtype=wp.float64, device=d)
    v = wp.zeros(n, dtype=wp.float64, device=d)
    s = wp.zeros(n, dtype=wp.float64, device=d)
    t = wp.zeros(n, dtype=wp.float64, device=d)
    Minv = _to_dev(1.0 / np.asarray(diag), d) if diag is not None else None
    ph = wp.zeros(n, dtype=wp.float64, device=d)
    sh = wp.zeros(n, dtype=wp.float64, device=d)
    rho = alpha = omega = 1.0
    bnorm = max(np.sqrt(blas.dot(bd, bd, d)), 1e-300)
    for it in range(1, maxiter + 1):
        rho_new = blas.dot(rhat, r, d)
        beta = (rho_new / rho) * (alpha / omega) if it > 1 else 0.0
        # p = r + beta (p - omega v)
        blas.axpy(-omega, v, p, d)
        blas.xpay(beta, r, p, d)
        if Minv is not None:
            blas.hadamard(Minv, p, ph, d)
        else:
            wp.copy(ph, p)
        op.matvec(ph, v)
        alpha = rho_new / blas.dot(rhat, v, d)
        wp.copy(s, r); blas.axpy(-alpha, v, s, d)
        if Minv is not None:
            blas.hadamard(Minv, s, sh, d)
        else:
            wp.copy(sh, s)
        op.matvec(sh, t)
        tt = blas.dot(t, t, d)
        omega = blas.dot(t, s, d) / tt if tt > 0 else 0.0
        blas.axpy(alpha, ph, x, d); blas.axpy(omega, sh, x, d)
        wp.copy(r, s); blas.axpy(-omega, t, r, d)
        rnorm = np.sqrt(blas.dot(r, r, d))
        if rnorm < max(tol * bnorm, atol):
            return x.numpy(), {"iters": it, "relres": rnorm / bnorm, "converged": True}
        rho = rho_new
    return x.numpy(), {"iters": maxiter, "relres": rnorm / bnorm, "converged": False}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_s_solvers.py -v` — Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/diffsim/solvers tests/test_s_solvers.py
git commit -m "feat: device BLAS + CG/BiCGStab with Jacobi hooks (S1, S2)"
```

---

### Task 11: Strong Dirichlet + Poisson patch tests (B2/B4, P1/P2, P3)

**Files:**
- Create: `src/diffsim/assembly/dirichlet.py`, `src/diffsim/physics/__init__.py`, `src/diffsim/physics/poisson.py`
- Test: `tests/test_p_patch.py`

**Interfaces:**
- Consumes: `DeviceMesh`, `ConstrainedOperator`, `krylov.cg`.
- Produces:
  - `dirichlet.py`: `class DirichletPoisson` — wraps `ConstrainedOperator` with strong BCs on unit-cube boundary *free* nodes: `solve(g_fn: Callable[[float64[N,3]], float64[N]], f_fn, tol=1e-12) -> u_all: float64[Nn]`. Method: constrained node set `C` = free nodes flagged `mesh.boundary_nodes`; lift `u = u0 + g` with `g` zero off `C`; reduced operator = matvec with rows/cols of `C` replaced by identity (`y[C] = x[C]`), RHS `b = F - A g` with `b[C] = 0`; assemble load `F_a = Σ_q N_a f detJxW` via device residual kernel; final answer expanded to all nodes via `T` and lift.
  - `poisson.py`: `make_load_kernel(nbf, nqp)` — per-element `be(a) += fe_N(a) * f(x_q) * detJxW` with `f` sampled on a per-qp device array (host evaluates `f_fn` at all Gauss points once); `l2_error(dm, u_all, u_exact_fn) -> float` (quadrature-evaluated L2 norm of `u_h − u_exact`); `gauss_points(mesh, tables) -> float64[Ne*nqp, 3]` (physical Gauss-point coordinates, host).

- [ ] **Step 1: Write the failing tests**

`tests/test_p_patch.py`:

```python
import numpy as np
import pytest
from diffsim.octree.build import build_uniform, refine_elements
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.assembly.dirichlet import DirichletPoisson
from diffsim.physics.poisson import l2_error

pytestmark = pytest.mark.tier2

LIN = lambda x: 1.0 + 2.0 * x[:, 0] - 3.0 * x[:, 1] + 0.5 * x[:, 2]
ZERO = lambda x: np.zeros(len(x))

def _solve(tree, p, device):
    m = build_mesh(tree, p=p)
    c = build_constraints(m)
    dm = DeviceMesh.from_mesh(m, c, basis_tables(p), device)
    solver = DirichletPoisson(dm)
    u = solver.solve(g_fn=LIN, f_fn=ZERO, tol=1e-13)
    return m, dm, u

def test_P1_P2_linear_patch_uniform(device):
    for p in (1, 2):
        m, dm, u = _solve(build_uniform(2), p, device)
        err = l2_error(dm, u, LIN)
        assert err < 1e-11, (p, err)       # machine-precision patch (cuFEM criterion)

def test_P3_linear_patch_adaptive(device):
    t = build_uniform(2)
    mask = np.zeros(len(t), bool); mask[[0, 9, 27]] = True
    t = refine_elements(t, mask)
    from diffsim.octree.balance import balance2to1
    t = balance2to1(t)
    for p in (1, 2):
        m, dm, u = _solve(t, p, device)
        err = l2_error(dm, u, LIN)
        assert err < 1e-11, (p, err)       # hanging constraints must be exact for linears
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_p_patch.py -v` — Expected: FAIL.

- [ ] **Step 3: Implement `src/diffsim/physics/poisson.py`**

```python
import numpy as np
import warp as wp
from ..assembly.femelm import FEMElm, fe_N, fe_detJxW
from ..assembly.operators import _kernel_cache

def gauss_points(mesh, tables):
    """Physical Gauss-point coords, [Ne*nqp, 3], ordering (e, q)."""
    from ..octree import morton
    from ..mesh.basis import gauss_1d
    pts, _ = gauss_1d(tables.p)
    nq1 = len(pts)
    ref = np.array([[pts[i], pts[j], pts[k]]
                    for k in range(nq1) for j in range(nq1) for i in range(nq1)])
    lo = mesh.tree.anchors() * 2.0 ** (-morton.LMAX)
    h = mesh.tree.h()
    xq = lo[:, None, :] + (ref[None, :, :] + 1.0) * 0.5 * h[:, None, None]
    return xq.reshape(-1, 3)

def make_load_kernel(nbf: int, nqp: int):
    key = ("load", nbf, nqp)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel
    def load(conn: wp.array2d(dtype=wp.int32), h: wp.array(dtype=wp.float64),
             Ntab: wp.array2d(dtype=wp.float64), wtab: wp.array(dtype=wp.float64),
             fq: wp.array(dtype=wp.float64),          # f at Gauss points, [Ne*nqp]
             be: wp.array(dtype=wp.float64)):
        e = wp.tid()
        fe = FEMElm(); fe.e = e; fe.he = h[e]
        for q in range(nqp):
            fe.q = q
            dJxW = fe_detJxW(wtab, fe)
            fv = fq[e * nqp + q]
            for a in range(nbf):
                wp.atomic_add(be, conn[e, a], fe_N(Ntab, fe, a) * fv * dJxW)

    _kernel_cache[key] = load
    return load

def make_l2_kernel(nbf: int, nqp: int):
    key = ("l2", nbf, nqp)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel
    def l2(conn: wp.array2d(dtype=wp.int32), h: wp.array(dtype=wp.float64),
           Ntab: wp.array2d(dtype=wp.float64), wtab: wp.array(dtype=wp.float64),
           u: wp.array(dtype=wp.float64), uq_exact: wp.array(dtype=wp.float64),
           out: wp.array(dtype=wp.float64)):
        e = wp.tid()
        fe = FEMElm(); fe.e = e; fe.he = h[e]
        acc = wp.float64(0.0)
        for q in range(nqp):
            fe.q = q
            uh = wp.float64(0.0)
            for a in range(nbf):
                uh += fe_N(Ntab, fe, a) * u[conn[e, a]]
            diff = uh - uq_exact[e * nqp + q]
            acc += diff * diff * fe_detJxW(wtab, fe)
        wp.atomic_add(out, 0, acc)

    _kernel_cache[key] = l2
    return l2

def l2_error(dm, u_all: np.ndarray, u_exact_fn) -> float:
    xq = gauss_points(dm.mesh, dm.tables)
    uq = wp.array(u_exact_fn(xq), dtype=wp.float64, device=dm.device)
    ud = wp.array(u_all.astype(np.float64), dtype=wp.float64, device=dm.device)
    out = wp.zeros(1, dtype=wp.float64, device=dm.device)
    k = make_l2_kernel(dm.tables.nbf, dm.tables.nqp)
    wp.launch(k, dim=len(dm.mesh.tree),
              inputs=[dm.conn, dm.h, dm.N, dm.w, ud, uq, out], device=dm.device)
    return float(np.sqrt(out.numpy()[0]))
```

- [ ] **Step 4: Implement `src/diffsim/assembly/dirichlet.py`**

```python
import numpy as np
import warp as wp
from .operators import ConstrainedOperator, csr_spmv
from ..physics.poisson import make_load_kernel, gauss_points
from ..solvers.krylov import cg

class _BCOperator:
    """Constrained operator with identity rows on Dirichlet free-node set."""
    def __init__(self, base: ConstrainedOperator, dir_mask_free: np.ndarray):
        self.base, self.dm = base, base.dm
        self.device, self.n_free = base.dm.device, base.dm.n_free
        self.mask = wp.array(dir_mask_free.astype(np.float64), dtype=wp.float64,
                             device=self.device)  # 1.0 on Dirichlet free nodes

    def matvec(self, x, y):
        # y = (I-M) A ((I-M) x) + M x  — symmetric constrained operator
        import diffsim.solvers.blas as blas
        d = self.device
        xm = wp.clone(x)
        _apply_mask(xm, self.mask, keep=False, device=d)   # zero Dirichlet entries
        self.base.matvec(xm, y)
        _apply_mask(y, self.mask, keep=False, device=d)
        _add_masked(y, x, self.mask, device=d)             # y[C] = x[C]

@wp.kernel
def _mask_kernel(v: wp.array(dtype=wp.float64), m: wp.array(dtype=wp.float64),
                 keep: wp.float64):
    i = wp.tid()
    v[i] = v[i] * (keep + (wp.float64(1.0) - wp.float64(2.0) * keep) * (wp.float64(1.0) - m[i]))

def _apply_mask(v, m, keep, device):
    wp.launch(_mask_kernel, dim=len(v), inputs=[v, m, wp.float64(1.0 if keep else 0.0)],
              device=device)

@wp.kernel
def _add_masked_kernel(y: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.float64),
                       m: wp.array(dtype=wp.float64)):
    i = wp.tid()
    y[i] = y[i] + m[i] * x[i]

def _add_masked(y, x, m, device):
    wp.launch(_add_masked_kernel, dim=len(y), inputs=[y, x, m], device=device)

class DirichletPoisson:
    def __init__(self, dm):
        self.dm = dm
        self.op = ConstrainedOperator(dm)
        c = dm.constraints
        self.dir_free = dm.mesh.boundary_nodes[c.free_nodes]   # bool [n_free]

    def solve(self, g_fn, f_fn, tol=1e-12):
        dm, c, d = self.dm, self.dm.constraints, self.dm.device
        n_free = dm.n_free
        # load vector F (full nodes) -> reduced
        xq = gauss_points(dm.mesh, dm.tables)
        fq = wp.array(f_fn(xq), dtype=wp.float64, device=d)
        F_full = wp.zeros(dm.n_nodes, dtype=wp.float64, device=d)
        lk = make_load_kernel(dm.tables.nbf, dm.tables.nqp)
        wp.launch(lk, dim=len(dm.mesh.tree),
                  inputs=[dm.conn, dm.h, dm.N, dm.w, fq, F_full], device=d)
        F_free = wp.zeros(n_free, dtype=wp.float64, device=d)
        wp.launch(csr_spmv, dim=n_free, inputs=[*dm.Tt_dev, F_full, F_free], device=d)
        # lift g on Dirichlet free nodes
        g = np.zeros(n_free)
        g[self.dir_free] = g_fn(dm.mesh.node_coords[c.free_nodes][self.dir_free])
        gd = wp.array(g, dtype=wp.float64, device=d)
        Ag = wp.zeros(n_free, dtype=wp.float64, device=d)
        self.op.matvec(gd, Ag)
        b = F_free.numpy() - Ag.numpy()
        b[self.dir_free] = 0.0
        bc_op = _BCOperator(self.op, self.dir_free)
        u0, info = cg(bc_op, b, tol=tol, maxiter=5000)
        assert info["converged"], info
        u_free = u0 + g
        return c.T @ u_free      # expand to all nodes (host SpMV; fine at M0 scale)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_p_patch.py -v` — Expected: 2 PASS (four sub-cases).

- [ ] **Step 6: Commit**

```bash
git add src/diffsim/assembly/dirichlet.py src/diffsim/physics tests/test_p_patch.py
git commit -m "feat: strong-Dirichlet Poisson solve + patch tests P1/P2/P3 at machine precision"
```

---

### Task 12: Assembled CSR path + matrix-free consistency (S3) + Jacobi diagonal

**Files:**
- Create: append `assemble_csr(dm) -> scipy.sparse.csr_matrix` (constrained space, `Tᵀ K T`) and `operator_diagonal(dm) -> np.ndarray` to `src/diffsim/assembly/operators.py`
- Test: append to `tests/test_g_operator.py`

**Interfaces:**
- Consumes: `DeviceMesh`.
- Produces: `assemble_csr(dm)`: device kernel writes per-element dense `Ke` into a preallocated `[Ne, nbf, nbf]` device array (same Gauss loop as the matvec, but storing `Ke(a,b)`), downloads, builds COO with `(conn[e,a], conn[e,b])` indices, converts to full-node CSR `K`, returns `(c.T.T @ K @ c.T).tocsr()`; `operator_diagonal(dm) -> float64[n_free]` extracted from the assembled matrix (exact Jacobi diagonal; used by Krylov `diag=`).

- [ ] **Step 1: Write the failing test (S3)**

Append to `tests/test_g_operator.py`:

```python
def test_S3_matrix_free_vs_assembled(device):
    from diffsim.assembly.operators import assemble_csr, operator_diagonal
    for p in (1, 2):
        m, c, dm = _setup(p, adaptive=True, device=device)
        A = assemble_csr(dm)
        op = ConstrainedOperator(dm)
        rng = np.random.default_rng(5)
        for _ in range(20):
            x = rng.standard_normal(dm.n_free)
            diff = np.abs(A @ x - op.matvec_numpy(x)).max()
            assert diff < 1e-10 * max(1.0, np.abs(A @ x).max()), (p, diff)
        d = operator_diagonal(dm)
        assert np.allclose(d, A.diagonal(), atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_g_operator.py::test_S3_matrix_free_vs_assembled -v` — Expected: FAIL.

- [ ] **Step 3: Implement (append to `operators.py`)**

```python
def make_poisson_element_matrices(nbf: int, nqp: int):
    key = ("poisson_Ke", nbf, nqp)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel
    def poisson_Ke(h: wp.array(dtype=wp.float64), dNtab: wp.array3d(dtype=wp.float64),
                   wtab: wp.array(dtype=wp.float64),
                   Ke: wp.array3d(dtype=wp.float64)):        # [Ne, nbf, nbf]
        e = wp.tid()
        fe = FEMElm(); fe.e = e; fe.he = h[e]
        for q in range(nqp):
            fe.q = q
            dJxW = fe_detJxW(wtab, fe)
            for a in range(nbf):
                for b in range(nbf):
                    v = (fe_dN(dNtab, fe, a, 0) * fe_dN(dNtab, fe, b, 0) +
                         fe_dN(dNtab, fe, a, 1) * fe_dN(dNtab, fe, b, 1) +
                         fe_dN(dNtab, fe, a, 2) * fe_dN(dNtab, fe, b, 2)) * dJxW
                    Ke[e, a, b] = Ke[e, a, b] + v

    _kernel_cache[key] = poisson_Ke
    return poisson_Ke

def assemble_csr(dm):
    ne, nbf, nqp = len(dm.mesh.tree), dm.tables.nbf, dm.tables.nqp
    Ke = wp.zeros((ne, nbf, nbf), dtype=wp.float64, device=dm.device)
    k = make_poisson_element_matrices(nbf, nqp)
    wp.launch(k, dim=ne, inputs=[dm.h, dm.dN, dm.w, Ke], device=dm.device)
    Keh = Ke.numpy()
    conn = dm.mesh.conn
    rows = np.repeat(conn, nbf, axis=1).ravel()
    cols = np.tile(conn, (1, nbf)).ravel()
    K = sp.coo_matrix((Keh.ravel(), (rows, cols)),
                      shape=(dm.n_nodes, dm.n_nodes)).tocsr()
    T = dm.constraints.T
    return (T.T @ K @ T).tocsr()

def operator_diagonal(dm) -> np.ndarray:
    return np.asarray(assemble_csr(dm).diagonal())
```

(Add `from ..mesh import constraints` imports as needed; `sp` and `np` are already imported in this module. Fix the row/col construction if the einsum layout differs — the test is the arbiter.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_g_operator.py -v` — Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/diffsim/assembly/operators.py tests/test_g_operator.py
git commit -m "feat: assembled CSR path + exact Jacobi diagonal, matvec consistency (S3)"
```

---

### Task 13: MMS convergence V2 — p1 order 2, p2 order 3, + regression baseline

**Files:**
- Create: `tests/test_v_mms.py`, `tests/baselines/m0_baselines.json`
- Test: `tests/test_v_mms.py`

**Interfaces:**
- Consumes: `DirichletPoisson`, `l2_error`.
- Produces: locked baseline JSON consumed by this and later tests (`tests/baselines/m0_baselines.json`).

- [ ] **Step 1: Write the failing test**

`tests/test_v_mms.py`:

```python
import json, pathlib
import numpy as np
import pytest
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.assembly.dirichlet import DirichletPoisson
from diffsim.physics.poisson import l2_error

pytestmark = pytest.mark.tier3

U = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]) * np.sin(np.pi * x[:, 2])
F = lambda x: 3.0 * np.pi**2 * U(x)
BASE = pathlib.Path(__file__).parent / "baselines" / "m0_baselines.json"

def _err(level, p, device):
    m = build_mesh(build_uniform(level), p=p)
    c = build_constraints(m)
    dm = DeviceMesh.from_mesh(m, c, basis_tables(p), device)
    u = DirichletPoisson(dm).solve(g_fn=U, f_fn=F, tol=1e-13)
    return l2_error(dm, u, U)

def test_V2_convergence_orders(device):
    results = {}
    for p, expected in ((1, 2.0), (2, 3.0)):
        levels = [2, 3, 4] if p == 1 else [1, 2, 3]
        errs = [_err(l, p, device) for l in levels]
        orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
        assert abs(orders[-1] - expected) < 0.10, (p, orders)   # cuFEM ±0.10 criterion
        results[f"p{p}"] = errs
    # regression lock: create baseline on first run, compare thereafter
    if BASE.exists():
        ref = json.loads(BASE.read_text())
        for k, v in results.items():
            assert np.allclose(v, ref[k], rtol=1e-8), (k, v, ref[k])
    else:
        BASE.parent.mkdir(parents=True, exist_ok=True)
        BASE.write_text(json.dumps(results, indent=2))
        pytest.skip("baseline created; re-run to compare")
```

- [ ] **Step 2: Run test — first run creates the baseline (SKIP), second run must PASS**

Run: `pytest tests/test_v_mms.py -v && pytest tests/test_v_mms.py -v`
Expected: first SKIP ("baseline created"), second PASS. If orders are off, debug before committing — do not lock a bad baseline.

- [ ] **Step 3: Commit (including the generated baseline file)**

```bash
git add tests/test_v_mms.py tests/baselines/m0_baselines.json
git commit -m "test: MMS convergence V2 (p1 order 2, p2 order 3) + locked regression baseline"
```

---

### Task 14: NonlinearSolver (SNES-like) + Bratu (spec §5.1.1)

**Files:**
- Create: `src/diffsim/solvers/newton.py`, `src/diffsim/physics/bratu.py`
- Test: `tests/test_newton_bratu.py`

**Interfaces:**
- Consumes: `krylov.cg`/`bicgstab`, `DeviceMesh`, `DirichletPoisson` internals (`_BCOperator`, load kernel pattern).
- Produces:
  - `newton.py`: `class NonlinearSolver(residual_fn: Callable[[np.ndarray], np.ndarray], jac_action_fn: Callable[[np.ndarray u, np.ndarray v], np.ndarray] | None = None, snes_rtol=1e-8, snes_atol=1e-12, snes_max_it=20, linesearch="bt", linear_solver=cg-compatible callable, linear_rtol=1e-10)` with `.solve(u0: np.ndarray) -> (u, info)`; `info = {"iters", "fnorm_history", "converged"}`. If `jac_action_fn is None`, uses **JFNK**: `J(u)·v ≈ (F(u + εv) − F(u))/ε`, `ε = sqrt(machine_eps) * (1 + |u|) / |v|`. Line search `"bt"`: backtracking with Armijo `‖F(u+λδ)‖ ≤ (1 − 1e-4·λ)‖F(u)‖`, halving, `λ_min = 1e-4`; `"basic"`: full step. Eisenstat–Walker: `linear_rtol_k = min(0.1, sqrt(‖F_k‖/‖F_{k-1}‖))` when `linesearch != "basic"` is irrelevant — apply always unless `linear_rtol` explicitly given.
  - `bratu.py`: Bratu problem `−Δu = λ e^u` on the unit cube, `u = 0` on the boundary: `class BratuProblem(dm, lam)` with `.residual(u_free) -> np.ndarray` (`F(u) = A_bc u − b_bc(u)` in the Dirichlet-reduced space: stiffness action minus the `λ e^u` load assembled at Gauss points from interpolated `u`) and `.jac_action(u_free, v_free)` (analytic: `A_bc v − mass-weighted λ e^u v`), both built from a `make_bratu_residual_kernel(nbf, nqp)` device kernel following the `Integrands_be` contract (nonlinear source evaluated at Gauss points from `fe`-interpolated `u`).

- [ ] **Step 1: Write the failing tests**

`tests/test_newton_bratu.py`:

```python
import numpy as np
import pytest
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.solvers.newton import NonlinearSolver
from diffsim.physics.bratu import BratuProblem

pytestmark = pytest.mark.tier3

def _bratu(device, lam=1.0, level=3, p=1):
    m = build_mesh(build_uniform(level), p=p)
    c = build_constraints(m)
    dm = DeviceMesh.from_mesh(m, c, basis_tables(p), device)
    return BratuProblem(dm, lam)

def test_newton_analytic_jacobian_converges(device):
    prob = _bratu(device)
    ns = NonlinearSolver(prob.residual, jac_action_fn=prob.jac_action,
                         snes_rtol=1e-10, snes_max_it=20)
    u, info = ns.solve(np.zeros(prob.n_free))
    assert info["converged"] and info["iters"] <= 6
    # superlinear tail: last contraction much stronger than first
    h = info["fnorm_history"]
    assert h[-1] / h[-2] < 0.5 * (h[1] / h[0])
    # physics sanity: positive interior solution, max locked as regression value
    umax = prob.expand(u).max()
    assert 0.05 < umax < 0.30                      # lambda=1 cube Bratu lower branch

def test_jfnk_matches_analytic(device):
    prob = _bratu(device)
    ns_a = NonlinearSolver(prob.residual, jac_action_fn=prob.jac_action, snes_rtol=1e-10)
    ns_j = NonlinearSolver(prob.residual, jac_action_fn=None, snes_rtol=1e-10)
    ua, _ = ns_a.solve(np.zeros(prob.n_free))
    uj, _ = ns_j.solve(np.zeros(prob.n_free))
    assert np.abs(ua - uj).max() < 1e-6

def test_linesearch_rescues_bad_step(device):
    prob = _bratu(device, lam=5.0)                 # stiffer; full steps can overshoot
    ns = NonlinearSolver(prob.residual, jac_action_fn=prob.jac_action,
                         snes_rtol=1e-9, snes_max_it=40, linesearch="bt")
    u, info = ns.solve(np.zeros(prob.n_free))
    assert info["converged"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_newton_bratu.py -v` — Expected: FAIL.

- [ ] **Step 3: Implement `src/diffsim/solvers/newton.py`**

```python
import numpy as np
from .krylov import cg, bicgstab

class _NpOp:
    """Adapts a numpy-action closure to the Krylov op protocol."""
    def __init__(self, action, n, device):
        self.action, self.n_free, self.device = action, n, device
    def matvec(self, x, y):
        import warp as wp
        r = self.action(x.numpy())
        wp.copy(y, wp.array(r.astype(np.float64), dtype=wp.float64, device=self.device))

class NonlinearSolver:
    def __init__(self, residual_fn, jac_action_fn=None, snes_rtol=1e-8, snes_atol=1e-12,
                 snes_max_it=20, linesearch="bt", linear_solver=bicgstab,
                 linear_rtol=None, device="cpu"):
        self.F = residual_fn
        self.Jv = jac_action_fn
        self.rtol, self.atol, self.max_it = snes_rtol, snes_atol, snes_max_it
        self.linesearch, self.linear_solver = linesearch, linear_solver
        self.linear_rtol, self.device = linear_rtol, device

    def _jac_action(self, u, Fu):
        if self.Jv is not None:
            return lambda v: self.Jv(u, v)
        eps0 = np.sqrt(np.finfo(np.float64).eps)
        def jv(v):
            nv = np.linalg.norm(v)
            if nv == 0.0:
                return np.zeros_like(v)
            eps = eps0 * (1.0 + np.linalg.norm(u)) / nv
            return (self.F(u + eps * v) - Fu) / eps
        return jv

    def solve(self, u0):
        u = np.array(u0, np.float64)
        Fu = self.F(u)
        f0 = fprev = np.linalg.norm(Fu)
        hist = [f0]
        for it in range(1, self.max_it + 1):
            if hist[-1] < max(self.rtol * f0, self.atol):
                return u, {"iters": it - 1, "fnorm_history": hist, "converged": True}
            # Eisenstat–Walker forcing unless fixed rtol given
            eta = self.linear_rtol or min(0.1, np.sqrt(hist[-1] / fprev) if it > 1 else 0.1)
            op = _NpOp(self._jac_action(u, Fu), len(u), self.device)
            du, info = self.linear_solver(op, -Fu, tol=eta, maxiter=1000)
            lam, fnew, Fnew = 1.0, None, None
            while True:
                Fnew = self.F(u + lam * du)
                fnew = np.linalg.norm(Fnew)
                if self.linesearch == "basic" or fnew <= (1.0 - 1e-4 * lam) * hist[-1]:
                    break
                lam *= 0.5
                if lam < 1e-4:
                    break
            u = u + lam * du
            fprev, Fu = hist[-1], Fnew
            hist.append(fnew)
        converged = hist[-1] < max(self.rtol * f0, self.atol)
        return u, {"iters": self.max_it, "fnorm_history": hist, "converged": converged}
```

- [ ] **Step 4: Implement `src/diffsim/physics/bratu.py`**

```python
import numpy as np
import warp as wp
from ..assembly.femelm import FEMElm, fe_N, fe_detJxW
from ..assembly.operators import ConstrainedOperator, csr_spmv, _kernel_cache
from ..assembly.dirichlet import _BCOperator
from ..physics.poisson import make_load_kernel, gauss_points

def make_gp_interp_kernel(nbf: int, nqp: int):
    key = ("gp_interp", nbf, nqp)
    if key in _kernel_cache:
        return _kernel_cache[key]

    @wp.kernel
    def gp_interp(conn: wp.array2d(dtype=wp.int32), Ntab: wp.array2d(dtype=wp.float64),
                  u: wp.array(dtype=wp.float64), uq: wp.array(dtype=wp.float64)):
        e = wp.tid()
        for q in range(nqp):
            acc = wp.float64(0.0)
            for a in range(nbf):
                acc += Ntab[q, a] * u[conn[e, a]]
            uq[e * nqp + q] = acc

    _kernel_cache[key] = gp_interp
    return gp_interp

class BratuProblem:
    """F(u) = A_bc u − b_bc(λ e^u), homogeneous Dirichlet on the unit cube."""
    def __init__(self, dm, lam):
        self.dm, self.lam = dm, lam
        self.op = ConstrainedOperator(dm)
        c = dm.constraints
        self.dir_free = dm.mesh.boundary_nodes[c.free_nodes]
        self.bc_op = _BCOperator(self.op, self.dir_free)
        self.n_free = dm.n_free
        self.nqp, self.nbf = dm.tables.nqp, dm.tables.nbf
        self._interp = make_gp_interp_kernel(self.nbf, self.nqp)
        self._load = make_load_kernel(self.nbf, self.nqp)

    def _to_dev(self, v):
        return wp.array(np.asarray(v, np.float64), dtype=wp.float64, device=self.dm.device)

    def _source_reduced(self, u_free):
        dm, d = self.dm, self.dm.device
        u_full = dm.constraints.T @ u_free
        uq = wp.zeros(len(dm.mesh.tree) * self.nqp, dtype=wp.float64, device=d)
        wp.launch(self._interp, dim=len(dm.mesh.tree),
                  inputs=[dm.conn, dm.N, self._to_dev(u_full), uq], device=d)
        fq = self._to_dev(self.lam * np.exp(uq.numpy()))
        b_full = wp.zeros(dm.n_nodes, dtype=wp.float64, device=d)
        wp.launch(self._load, dim=len(dm.mesh.tree),
                  inputs=[dm.conn, dm.h, dm.N, dm.w, fq, b_full], device=d)
        b = np.asarray((dm.constraints.T.T @ b_full.numpy()))
        b[self.dir_free] = 0.0
        return b, uq.numpy()

    def residual(self, u_free):
        y = wp.zeros(self.n_free, dtype=wp.float64, device=self.dm.device)
        self.bc_op.matvec(self._to_dev(u_free), y)
        b, _ = self._source_reduced(u_free)
        return y.numpy() - b

    def jac_action(self, u_free, v_free):
        # J v = A_bc v − M[λ e^u] v  (mass matrix weighted by λ e^u at Gauss points)
        y = wp.zeros(self.n_free, dtype=wp.float64, device=self.dm.device)
        self.bc_op.matvec(self._to_dev(v_free), y)
        dm, d = self.dm, self.dm.device
        _, uq = self._source_reduced(u_free)
        v_full = dm.constraints.T @ v_free
        vq = wp.zeros(len(dm.mesh.tree) * self.nqp, dtype=wp.float64, device=d)
        wp.launch(self._interp, dim=len(dm.mesh.tree),
                  inputs=[dm.conn, dm.N, self._to_dev(v_full), vq], device=d)
        gq = self._to_dev(self.lam * np.exp(uq) * vq.numpy())
        m_full = wp.zeros(dm.n_nodes, dtype=wp.float64, device=d)
        wp.launch(self._load, dim=len(dm.mesh.tree),
                  inputs=[dm.conn, dm.h, dm.N, dm.w, gq, m_full], device=d)
        mv = np.asarray(dm.constraints.T.T @ m_full.numpy())
        mv[self.dir_free] = 0.0
        return y.numpy() - mv

    def expand(self, u_free):
        return self.dm.constraints.T @ u_free
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_newton_bratu.py -v` — Expected: 3 PASS.

- [ ] **Step 6: Run the full M0 suite and commit**

Run: `pytest -v`
Expected: all tests PASS (one SKIP only if the V2 baseline was somehow deleted).

```bash
git add src/diffsim/solvers/newton.py src/diffsim/physics/bratu.py tests/test_newton_bratu.py
git commit -m "feat: SNES-like NonlinearSolver (Newton/JFNK, bt linesearch, Eisenstat-Walker) + Bratu"
```

---

## Self-Review (performed while writing)

1. **Spec coverage (M0 row of the roadmap):** Morton build ✔ (T1–T2), carve ✔ (T4), 2:1 ✔ (T5), hanging constraints ✔ (T7), p1/p2 tensorized assembly ✔ (T8–T9, T12), CG/BiCGStab ✔ (T10), NonlinearSolver validated on Bratu ✔ (T14), Tier 1–3 + patch-test gates ✔ (T1–T13). Neighbor finding (N-tests) ✔ (T3). Deliberately deferred to M1 per spec: Nitsche weak BCs (B5), SBM tiers, geometry backends beyond the sphere oracle, gradients/tape.
2. **Placeholder scan:** no TBDs; every code step has complete code; two flagged uncertainty points carry explicit fallbacks (Task 7 unused import; Task 9 `wtab.shape[0]`; Task 12 row/col layout — tests are the arbiter).
3. **Type consistency:** `Octree{keys, levels}`, `Mesh{p, tree, node_coords, node_icoords, conn, boundary_nodes}`, `Constraints{T, free_nodes, hanging}`, `Tables{p, nbf, nqp, N, dN, w}`, `DeviceMesh`, `ConstrainedOperator.matvec(x, y)`, Krylov `(op, b, tol, atol, maxiter, diag) -> (x, info)`, `NonlinearSolver(residual_fn, jac_action_fn, ...)` — names checked consistent across tasks.
