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
    offs = _local_offsets(1, dim)              # [(p+1)^dim, dim] corner offsets
    conn = mesh.conn                           # [Ne, 2^dim] p1 corner node ids
    ids = set()
    for e, f in zip(sf.elem, sf.face):
        ax, side = int(f) // 2, int(f) % 2
        local = np.where(offs[:, ax] == side)[0]
        for l in local:
            ids.add(int(conn[e, l]))
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
def _build_channel(level, Re, half, offset, device, dim, center):
    """Carve a `Box` obstacle out of the unit-cube channel and build the mesh
    chain (mirrors tests/p2r0_task10_sphere_derisk.py::build_sphere_3d, with
    Sphere -> Box and lam=0.0 for the exact body-fitted carve)."""
    ndof = dim + 1
    D = 2 * half                               # obstacle side length
    nu = U_IN * D / Re
    center = np.asarray(center, dtype=float).copy()
    center[0] += offset                        # sub-cell shift breaks alignment
    oracle = Box(tuple(center), tuple([half] * dim))
    tree = build_uniform(level, dim=dim)
    n_full = len(tree)
    # lam=0.0 keeps only fully-interior cells => exact body-fitted carve.
    ret, _ = classify_lambda(tree, oracle, lam=0.0, domain="outside")
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), device)
    geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, dim),
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


def build_square_channel_2d(level, Re, half, offset, device):
    """2-D square-in-channel fixture (rungs A/B/C).

    Returns a dict with (device mesh `dm`, constraints `cons`, `oracle`,
    `obstacle_node_mask`, `inflow_mask`, `outflow_nodes`, `geo`, `dmax`) plus
    the mesh objects and BC data the drivers need. `half = k/2^level` +
    `offset=0` => exact body-fitted (`dmax==0`, `geo.corr==1`); a sub-cell
    `offset` => genuine SBM shift (`0 < dmax < h`). `Re` via `nu = U_IN*D/Re`,
    `D = 2*half`."""
    return _build_channel(level, Re, half, offset, device, dim=2,
                          center=(0.5, 0.5))


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
