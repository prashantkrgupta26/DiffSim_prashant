"""Projection Validation Ladder — shared mesh fixtures (Task 1).

Every rung driver (Tasks 2-7) consumes these fixtures; the signatures here are
the ladder's interfaces. This module builds NO flow solve — only the carved
octree meshes, the SBM geometry cache, and the boundary node masks the drivers
need. The same-mesh monolithic oracle is the correctness bar at the rung level.

The enabler (verified 2026-07-23): the octree mesher produces an EXACT
body-fitted carve. For a `Box(center, half)` obstacle with a CELL-ALIGNED
half-width `half = k / 2^level`, `classify_lambda(..., lam=0.0, domain="outside")`
yields NO cut cells => `extract_surrogate` faces coincide with the true box
boundary => `GeometryData.evaluate` gives `d = 0`, `corr = 1.0` => the SBM
Nitsche form reduces to standard Nitsche exactly (`S N_a = N_a` when `d = 0`).
A NON-aligned half-width (an `offset` sub-cell shift of the center) breaks the
alignment => `0 < |d|_max < h` => a GENUINE SBM shift on the SAME code path
(rung C). The fixture delta from the sphere test (tests/test_sphere.py,
tests/p2r0_task10_sphere_derisk.py) is `Sphere -> Box` + `lam=0.0`.

Domain: the octree unit cube `[0, 1]^dim` is the channel. Inflow is a strong
uniform `U_IN` Dirichlet at `x = 0`; the outflow at `x = 1` is FREE (the Taly
physical-pressure BC node set, `outflow_nodes`, in the stepper FREE-node space).
Lateral faces are walls per the benchmark. `Re` is set via
`nu = U_IN * D / Re` with `D = 2*half` (the obstacle side length). The cavity
(rung 0) is the full unit square with a moving lid at `y = 1`, no obstacle.
"""
import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh, _local_offsets
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.geometry.csg import Box
from diffsim.sbm.surrogate import (classify_lambda, extract_surrogate,
                                   GeometryData)

U_IN = 1.0                       # uniform inflow speed (nondimensional)


# --------------------------------------------------------------------------
# obstacle boundary node identification
# --------------------------------------------------------------------------
def _surrogate_face_node_ids(mesh, sf):
    """MESH node ids that are p1 corners of a surrogate (carved-obstacle) face.

    A surrogate face (elem `e`, face `f = 2*ax + side`) is a cell face of a
    retained (fluid) element with no retained neighbour across it => it hugs
    the carved obstacle. Its p1 corner nodes are the fluid nodes ON the
    obstacle boundary. Works for aligned (faces == true box faces) and offset
    (faces are the grid-aligned carved boundary) carves alike."""
    dim = mesh.dim
    p_elem = np.asarray(mesh.p_elem)
    # per-bin: global elem id -> (bin p, bin-local row) so this works for the
    # UNIFORM path (mesh.conn) AND the mixed p1/p2 path (mesh.conn is None ->
    # index conn_of[pv]). The "p1 corners of a face" are the order-p lattice
    # nodes whose per-order offset equals the face `side` on axis `ax` AND whose
    # tangential offsets are corners (0 or p) — the geometric box corners of the
    # face, shared with the P1 corners under the joint dedup.
    row_of = np.full(len(p_elem), -1, np.int64)
    for pv, eids in mesh.bins.items():
        row_of[eids] = np.arange(len(eids))
    offs_by_p = {pv: _local_offsets(int(pv), dim) for pv in mesh.bins}
    ids = set()
    for e, f in zip(sf.elem, sf.face):
        ax, side = int(f) // 2, int(f) % 2
        pv = int(p_elem[e])
        offs = offs_by_p[pv]
        corner = (offs[:, ax] == (side * pv))            # side face on axis ax
        for d in range(dim):
            if d != ax:
                corner &= np.isin(offs[:, d], (0, pv))   # tangential corners
        conn = mesh.conn_of[pv][row_of[e]]
        for l in np.where(corner)[0]:
            ids.add(int(conn[l]))
    return np.array(sorted(ids), dtype=np.int64)


def obstacle_boundary_nodes(mesh, oracle):
    """MESH node ids on the carved obstacle faces (the immersed `Box` surface).

    Returns the p1 corner nodes of the surrogate faces of `mesh.tree` (the
    retained/fluid carve). `oracle` documents the immersed geometry these
    faces hug; for the aligned carve the nodes lie exactly on the `Box` faces.
    Rung drivers add these to the strong no-slip mask (rung A) or impose weak
    Nitsche there (rungs B/C)."""
    sf = extract_surrogate(mesh.tree)
    return _surrogate_face_node_ids(mesh, sf)


# --------------------------------------------------------------------------
# shared mesh-build chain (mirrors build_sphere_3d)
# --------------------------------------------------------------------------
def _p2_band_array(ret, sf, band):
    """Per-element order array (int8): P2 (order 2) for every retained element
    within `band` cell-layers of a surrogate (obstacle) face, P1 (order 1)
    elsewhere. `band=1` marks exactly the elements that own a surrogate face.

    Grows the P2 set by BFS over face-neighbours `band` times (2:1-balanced
    octree face adjacency via `face_neighbors`). Because the mesh is uniform-
    level here, the one-knob rule (level XOR p across a face) is automatically
    satisfied for ANY P1/P2 partition (level is constant, so only p changes)."""
    from diffsim.octree.lookup import face_neighbors
    ne = len(ret)
    is_p2 = np.zeros(ne, dtype=bool)
    is_p2[np.unique(np.asarray(sf.elem))] = True      # band=1 seed: face owners
    # face_neighbors -> list of 2*dim arrays [ne]; entry k is the neighbour
    # across face-direction k (-1 = domain boundary / carved-away).
    nbrs = face_neighbors(ret)
    for _ in range(int(band) - 1):
        grow = is_p2.copy()
        for j in nbrs:                                 # j: [ne] neighbour ids
            ok = j >= 0
            src = np.where(ok)[0]                       # elems with a neighbour
            grow[src[is_p2[j[ok]]]] = True             # neighbour is P2 -> grow
        is_p2 = grow
    p_elem = np.where(is_p2, 2, 1).astype(np.int8)
    return p_elem


def _build_channel(level, Re, half, offset, device, dim, center, p=1,
                   p2_band=0, y_offset=0.0):
    """Carve a `Box` obstacle out of the unit-cube channel and build the mesh
    chain (mirrors tests/p2r0_task10_sphere_derisk.py::build_sphere_3d, with
    Sphere -> Box and lam=0.0 for the exact body-fitted carve).

    `p` is the (equal-order) element order for BOTH velocity and pressure
    (ndof = dim+1 collocated nodes). p=1 is the default (bit-for-bit unchanged);
    p=2 builds the P2 mesh/basis/face tables for the FN3 basis-order stress test.
    The whole chain (build_mesh/basis_tables/face_tables) is p-generic; the SBM
    GeometryData.evaluate reads the p2 face tables so `dmax==0` still holds for
    the aligned carve.

    `p2_band > 0` overrides `p` with a VARIABLE-order mesh (FN3 axis 3): a band
    of `p2_band` cell-layers of P2 elements around the obstacle, P1 in the far
    field — EQUAL ORDER within every element (P2v+P2p in the band, P1v+P1p
    outside), so PSPG stays on. Mixed-degree is first-class in build_mesh
    (per-element `p_elem`, joint node dedup) + build_constraints (the P2 mid-edge
    node on a P1/P2 interface is a p-hanging DOF with an interpolation row). The
    DeviceMesh then carries a {1: P1-tables, 2: P2-tables} dict and the surrogate
    faces are P2 (the band always contains the face owners)."""
    ndof = dim + 1
    D = 2 * half                               # obstacle side length
    nu = U_IN * D / Re
    center = np.asarray(center, dtype=float).copy()
    center[0] += offset                        # sub-cell shift breaks alignment
    # y_offset: a PERMANENT transverse shift of the obstacle center. A sub-cell
    # y_offset breaks the mesh's y-symmetry (dmax>0, a genuine SBM shift on the
    # rung-C code path) and seeds vortex shedding that the symmetry-preserving
    # scheme cannot erase. y_offset=0 (default) is bit-for-bit unchanged.
    if dim >= 2:
        center[1] += y_offset
    oracle = Box(tuple(center), tuple([half] * dim))
    tree = build_uniform(level, dim=dim)
    n_full = len(tree)
    # lam=0.0 keeps only fully-interior cells => exact body-fitted carve.
    ret, _ = classify_lambda(tree, oracle, lam=0.0, domain="outside")
    sf = extract_surrogate(ret)
    if p2_band and p2_band > 0:
        p_elem = _p2_band_array(ret, sf, p2_band)
        mesh = build_mesh(ret, p=p_elem)
        tabs = {1: basis_tables(1, dim=dim), 2: basis_tables(2, dim=dim)}
        # surrogate faces are P2 (band contains face owners) -> P2 face tables.
        geo_ftab = face_tables(2, dim)
    else:
        mesh = build_mesh(ret, p=p)
        tabs = basis_tables(p, dim=dim)
        geo_ftab = face_tables(p, dim)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, tabs, device)
    geo = GeometryData.evaluate(oracle, ret, sf, geo_ftab,
                                domain="outside")
    dmax = float(np.abs(geo.d).max())

    coords = mesh.node_coords[cons.free_nodes]
    on = lambda v, c: np.abs(coords[:, c] - v) < 1e-12
    # strong inflow (x=0) + lateral walls (y faces in 2-D; y,z faces in 3-D);
    # outflow (x=1) is free.
    strong_sel = on(0.0, 0) | on(0.0, 1) | on(1.0, 1)
    if dim == 3:
        strong_sel |= on(0.0, 2) | on(1.0, 2)
    strong_mask = strong_sel.copy()

    # obstacle boundary nodes, in FREE-node space (the drivers' index space).
    obstacle_mesh_ids = _surrogate_face_node_ids(mesh, sf)
    free_of = -np.ones(len(mesh.node_coords), dtype=np.int64)
    free_of[cons.free_nodes] = np.arange(len(cons.free_nodes))
    obstacle_free = free_of[obstacle_mesh_ids]
    obstacle_free = obstacle_free[obstacle_free >= 0]     # drop hanging (none @ p1)
    obstacle_node_mask = np.zeros(len(coords), dtype=bool)
    obstacle_node_mask[obstacle_free] = True

    inflow_mask = on(0.0, 0)
    u_inf = np.zeros((len(coords), dim))
    u_inf[inflow_mask, 0] = U_IN
    outflow_nodes = np.where(on(1.0, 0))[0]

    return dict(
        oracle=oracle, dm=dm, cons=cons, mesh=mesh, sf=sf, geo=geo,
        coords=coords, strong_mask=strong_mask,
        obstacle_node_mask=obstacle_node_mask, inflow_mask=inflow_mask,
        outflow_nodes=outflow_nodes, u_inf=u_inf, nu=nu, ndof=ndof, dim=dim,
        dmax=dmax, half=half, center=tuple(center), U_IN=U_IN,
        n_fluid_cells=len(ret), n_full_cells=n_full)


def build_square_channel_2d(level, Re, half, offset, device, y_offset=0.0):
    """2-D square-in-channel fixture (rungs A/B/C).

    Returns a dict with (device mesh `dm`, constraints `cons`, `oracle`,
    `obstacle_node_mask`, `inflow_mask`, `outflow_nodes`, `geo`, `dmax`) plus
    the mesh objects and BC data the drivers need. `half = k/2^level` +
    `offset=0` => exact body-fitted (`dmax==0`, `geo.corr==1`); a sub-cell
    `offset` => genuine SBM shift (`0 < dmax < h`). `Re` via `nu = U_IN*D/Re`,
    `D = 2*half`. `y_offset` (default 0) shifts the obstacle transversely to
    seed shedding (a permanent asymmetry the symmetry-preserving scheme cannot
    restore); a sub-cell `y_offset` yields `dmax>0` (SBM shift path)."""
    return _build_channel(level, Re, half, offset, device, dim=2,
                          center=(0.5, 0.5), y_offset=y_offset)


def build_cube_channel_3d(level, Re, half, offset, device):
    """3-D cube-in-channel fixture (rungs A'/C') — the 2-D analogue with
    `dim=3`, `Box((cx,cy,cz),(half,half,half))`, lateral walls on y and z."""
    return _build_channel(level, Re, half, offset, device, dim=3,
                          center=(0.5, 0.5, 0.5))


def build_cavity_2d(level, Re, device):
    """2-D lid-driven cavity fixture (rung 0): the full unit square, no
    obstacle, all four walls strong Dirichlet, moving lid `U_IN` at `y = 1`.

    Returns (device mesh `dm`, constraints `cons`, `oracle` (None — no
    immersed geometry), `lid_mask`, plus the strong-wall mask, `u_inf`, and
    `coords`). `Re` via `nu = U_IN * L / Re` with `L = 1` (the cavity side)."""
    dim, ndof = 2, 3
    L = 1.0
    nu = U_IN * L / Re
    tree = build_uniform(level, dim=dim)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), device)

    coords = mesh.node_coords[cons.free_nodes]
    on = lambda v, c: np.abs(coords[:, c] - v) < 1e-12
    lid_mask = on(1.0, 1)                       # top row (moving lid) y = 1
    walls = on(0.0, 0) | on(1.0, 0) | on(0.0, 1)   # left/right/bottom, no-slip
    strong_mask = lid_mask | walls
    u_inf = np.zeros((len(coords), dim))
    u_inf[lid_mask, 0] = U_IN                   # lid drives +x

    return dict(
        oracle=None, dm=dm, cons=cons, mesh=mesh, coords=coords,
        lid_mask=lid_mask, strong_mask=strong_mask, u_inf=u_inf,
        nu=nu, ndof=ndof, dim=dim, U_IN=U_IN, n_fluid_cells=len(tree))
