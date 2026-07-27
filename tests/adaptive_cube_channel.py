"""Adaptive band-refined cube-in-channel fixture (Task 8b scope increment).

Mirrors `build_adaptive_plate_mesh` (tests/p2r1c_thin_plate_flow_3d.py) —
uniform base tree -> mark cells within a band of the geometry surface ->
`refine_elements` -> `balance2to1` -> classify -> build_mesh/build_constraints —
but the band criterion is distance to the CUBE surface (|Box.psi| <
band_cells * local h), and the carve/classify is the VOLUMETRIC one-sided
surrogate extraction the bluff-body ladder uses
(`classify_lambda(lam=0.0, domain="outside")` + `extract_surrogate` +
`GeometryData.evaluate`), copied from `ladder_fixtures._build_channel`.

The returned dict has the exact shape `march_monolithic_3d`
(tests/ladder_rung3d_cube.py) consumes: dm/mesh/cons + (sf, geo) + strong-BC
masks + n_excluded/n_hanging/n_nodes.  New file — no existing file modified.
"""
import time

import numpy as np
import torch

from diffsim.octree.build import build_uniform, refine_elements
from diffsim.octree.balance import balance2to1
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.geometry.csg import Box
from diffsim.sbm.surrogate import (classify_lambda, extract_surrogate,
                                   GeometryData)

from ladder_fixtures import U_IN, _surrogate_face_node_ids


def build_adaptive_cube_channel_3d(base_level, refine_to, Re=40, half=0.125,
                                   offset=0.05, band_cells=2, device="cpu",
                                   center=(0.5, 0.5, 0.5)):
    """3-D cube-in-channel on an ADAPTIVE band-refined octree.

    Parameters
    ----------
    base_level : int
        Uniform starting octree level (e.g. 5 = 32^3 cells).
    refine_to : int
        Target refinement level near the cube surface (>= base_level).
    Re : float
        Reynolds number via nu = U_IN * D / Re, D = 2*half.
    half : float
        Cube half-side.
    offset : float
        Sub-cell x-shift of the cube center (genuine SBM shift when the
        resulting half-width is not cell-aligned at the finest level).
    band_cells : int
        Half-width of the refinement band in local cell sizes (default 2,
        same convention as build_adaptive_plate_mesh).
    device : str
        DeviceMesh device ("cpu" | "cuda:0").
    center : tuple
        Cube center BEFORE the x-offset.

    Returns
    -------
    dict with the `march_monolithic_3d` fixture contract (oracle, dm, cons,
    mesh, sf, geo, coords, strong_mask, obstacle_node_mask, inflow_mask,
    outflow_nodes, u_inf, nu, ndof, dim, dmax, half, center, U_IN,
    n_fluid_cells, n_full_cells) PLUS the adaptive-mesh reporting keys
    (n_nodes, n_hanging, n_excluded, build_time).
    """
    dim = 3
    ndof = dim + 1
    D = 2.0 * half
    nu = U_IN * D / Re
    t0 = time.time()

    ctr = np.asarray(center, dtype=float).copy()
    ctr[0] += offset
    oracle = Box(tuple(ctr), tuple([half] * dim))

    # ---- band refinement (mirrors build_adaptive_plate_mesh) --------------
    tree = build_uniform(base_level, dim=dim)
    for _ in range(base_level + 1, refine_to + 1):
        centers = tree.centers()                       # [N, 3] in [0,1]^3
        pts_t = torch.tensor(centers, dtype=torch.float64)
        # Box.psi is SIGNED (negative inside); band = distance to the SURFACE
        psi_vals = np.abs(oracle.psi(pts_t).detach().numpy())
        h_local = tree.h()                             # [N] local cell sizes
        mask = psi_vals < band_cells * h_local
        if not mask.any():
            break
        tree = refine_elements(tree, mask)
        tree = balance2to1(tree)
    n_full = len(tree)

    # ---- volumetric carve + surrogate (copies _build_channel) -------------
    # lam=0.0 keeps only fully-exterior cells => surrogate hugs the cube.
    ret, _ = classify_lambda(tree, oracle, lam=0.0, domain="outside")
    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), device)
    geo = GeometryData.evaluate(oracle, ret, sf, face_tables(1, dim),
                                domain="outside")
    dmax = float(np.abs(geo.d).max())

    n_nodes = len(mesh.node_coords)
    n_hanging = int(cons.hanging.sum())
    n_excluded = n_full - len(ret)
    build_time = time.time() - t0

    # ---- strong-BC masks (copies _build_channel; box = [0,1]^3) -----------
    coords = mesh.node_coords[cons.free_nodes]
    on = lambda v, c: np.abs(coords[:, c] - v) < 1e-12
    strong_sel = (on(0.0, 0) | on(0.0, 1) | on(1.0, 1)
                  | on(0.0, 2) | on(1.0, 2))
    strong_mask = strong_sel.copy()

    obstacle_mesh_ids = _surrogate_face_node_ids(mesh, sf)
    free_of = -np.ones(len(mesh.node_coords), dtype=np.int64)
    free_of[cons.free_nodes] = np.arange(len(cons.free_nodes))
    obstacle_free = free_of[obstacle_mesh_ids]
    obstacle_free = obstacle_free[obstacle_free >= 0]   # drop hanging nodes
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
        dmax=dmax, half=half, center=tuple(ctr), U_IN=U_IN,
        n_fluid_cells=len(ret), n_full_cells=n_full,
        n_nodes=n_nodes, n_hanging=n_hanging, n_excluded=n_excluded,
        build_time=build_time,
    )
