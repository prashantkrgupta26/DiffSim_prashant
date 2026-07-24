"""Projection Validation Ladder — Task 1 fixture tests.

Assert the shared mesh fixtures that every rung driver consumes:
- exact body-fitted carve (aligned Box half-width => d=0, corr=1.0),
- genuine SBM shift (offset Box => 0 < |d|_max < h),
- obstacle boundary node mask (non-empty, on the carved box faces),
- fluid node count consistent with the carved cell count,
- 3-D cube analogues,
- lid-driven cavity lid mask = the top boundary row.

These are geometry assertions (no flow solve), so they run fast on a CPU/Warp
Mac at small octree levels. The same-mesh monolithic oracle is the correctness
bar at the RUNG level (Tasks 2-7); here we only pin the mesh interfaces.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from diffsim.octree.build import build_uniform

from ladder_fixtures import (
    build_cavity_2d,
    build_square_channel_2d,
    build_cube_channel_3d,
    obstacle_boundary_nodes,
)

pytestmark = pytest.mark.tier4

# aligned half-width at level 4: k / 2^level (0.25 = 4/16), center on the grid.
LEVEL_2D = 4
HALF_ALIGNED = 0.25


def _h(level):
    return 1.0 / 2 ** level


# --------------------------------------------------------------------------
# 2-D square-in-channel
# --------------------------------------------------------------------------
def test_square_aligned_is_body_fitted(device):
    """(a) aligned Box (offset=0, half=k/2^level) => d==0 and corr≈1.0 exactly:
    the SBM Nitsche form reduces to standard Nitsche (S N_a = N_a when d=0)."""
    fx = build_square_channel_2d(LEVEL_2D, Re=40.0, half=HALF_ALIGNED,
                                 offset=0.0, device=device)
    assert fx["dmax"] == 0.0
    assert np.allclose(fx["geo"].corr, 1.0)


def test_square_offset_is_genuine_shift(device):
    """(b) offset Box (sub-cell shift) => 0 < |d|_max < h: a genuine SBM shift
    on the SAME code path (anti-vacuity guard for rung C)."""
    fx = build_square_channel_2d(LEVEL_2D, Re=40.0, half=HALF_ALIGNED,
                                 offset=0.03, device=device)
    h = _h(LEVEL_2D)
    assert 0.0 < fx["dmax"] < h


def test_square_obstacle_mask_on_box_faces(device):
    """(c) obstacle mask is non-empty and its nodes lie on the box faces."""
    fx = build_square_channel_2d(LEVEL_2D, Re=40.0, half=HALF_ALIGNED,
                                 offset=0.0, device=device)
    mask = fx["obstacle_node_mask"]
    assert mask.any()
    coords = fx["coords"][mask]
    cx, cy = fx["center"]
    half = fx["half"]
    on_face = (
        (np.abs(coords[:, 0] - (cx - half)) < 1e-9)
        | (np.abs(coords[:, 0] - (cx + half)) < 1e-9)
        | (np.abs(coords[:, 1] - (cy - half)) < 1e-9)
        | (np.abs(coords[:, 1] - (cy + half)) < 1e-9)
    )
    assert on_face.all()


def test_square_fluid_nodes_match_carved_cells(device):
    """(d) fluid cell count = full uniform cells minus the carved box cells."""
    fx = build_square_channel_2d(LEVEL_2D, Re=40.0, half=HALF_ALIGNED,
                                 offset=0.0, device=device)
    full = build_uniform(LEVEL_2D, dim=2)
    box_cells = round((2 * HALF_ALIGNED / _h(LEVEL_2D)) ** 2)
    assert fx["n_fluid_cells"] == len(full) - box_cells


def test_square_re_sets_nu(device):
    """Re is set via nu = U_IN * D / Re with D = obstacle side (2*half)."""
    fx = build_square_channel_2d(LEVEL_2D, Re=40.0, half=HALF_ALIGNED,
                                 offset=0.0, device=device)
    D = 2 * HALF_ALIGNED
    assert np.isclose(fx["nu"], fx["U_IN"] * D / 40.0)


def test_square_inflow_and_outflow(device):
    """Inflow mask sits on x=0 with U_IN; outflow-free nodes on x=Lx=1."""
    fx = build_square_channel_2d(LEVEL_2D, Re=40.0, half=HALF_ALIGNED,
                                 offset=0.0, device=device)
    coords = fx["coords"]
    assert fx["inflow_mask"].any()
    assert (np.abs(coords[fx["inflow_mask"], 0]) < 1e-12).all()
    outf = fx["outflow_nodes"]
    assert len(outf) > 0
    assert (np.abs(coords[outf, 0] - 1.0) < 1e-12).all()


# --------------------------------------------------------------------------
# 3-D cube-in-channel analogue
# --------------------------------------------------------------------------
def test_cube_aligned_is_body_fitted(device):
    fx = build_cube_channel_3d(3, Re=40.0, half=0.25, offset=0.0, device=device)
    assert fx["dmax"] == 0.0
    assert np.allclose(fx["geo"].corr, 1.0)


def test_cube_offset_is_genuine_shift(device):
    fx = build_cube_channel_3d(3, Re=40.0, half=0.25, offset=0.05,
                               device=device)
    h = _h(3)
    assert 0.0 < fx["dmax"] < h


def test_cube_obstacle_mask_on_box_faces(device):
    fx = build_cube_channel_3d(3, Re=40.0, half=0.25, offset=0.0, device=device)
    mask = fx["obstacle_node_mask"]
    assert mask.any()
    coords = fx["coords"][mask]
    cx, cy, cz = fx["center"]
    half = fx["half"]
    on_face = (
        (np.abs(coords[:, 0] - (cx - half)) < 1e-9)
        | (np.abs(coords[:, 0] - (cx + half)) < 1e-9)
        | (np.abs(coords[:, 1] - (cy - half)) < 1e-9)
        | (np.abs(coords[:, 1] - (cy + half)) < 1e-9)
        | (np.abs(coords[:, 2] - (cz - half)) < 1e-9)
        | (np.abs(coords[:, 2] - (cz + half)) < 1e-9)
    )
    assert on_face.all()


def test_cube_fluid_cells_match_carved(device):
    fx = build_cube_channel_3d(3, Re=40.0, half=0.25, offset=0.0, device=device)
    full = build_uniform(3, dim=3)
    box_cells = round((2 * 0.25 / _h(3)) ** 3)
    assert fx["n_fluid_cells"] == len(full) - box_cells


# --------------------------------------------------------------------------
# Rung 0 — lid-driven cavity
# --------------------------------------------------------------------------
def test_cavity_lid_mask_is_top_row(device):
    """Cavity lid mask = the top boundary row (y = 1); no obstacle carved."""
    level = 4
    fx = build_cavity_2d(level, Re=100.0, device=device)
    coords = fx["coords"]
    lid = fx["lid_mask"]
    assert lid.any()
    assert (np.abs(coords[lid, 1] - 1.0) < 1e-12).all()
    # every top-row node is in the lid mask (2^level + 1 of them).
    top = np.abs(coords[:, 1] - 1.0) < 1e-12
    assert lid.sum() == top.sum() == 2 ** level + 1


def test_cavity_has_no_obstacle(device):
    """The cavity is the full unit square (Rung 0 base projection soundness)."""
    level = 4
    fx = build_cavity_2d(level, Re=100.0, device=device)
    full = build_uniform(level, dim=2)
    assert fx["n_fluid_cells"] == len(full)


# --------------------------------------------------------------------------
# obstacle_boundary_nodes standalone
# --------------------------------------------------------------------------
def test_obstacle_boundary_nodes_helper(device):
    """obstacle_boundary_nodes(mesh, oracle) returns node ids on the carved
    obstacle faces; matches the fixture's obstacle_node_mask."""
    fx = build_square_channel_2d(LEVEL_2D, Re=40.0, half=HALF_ALIGNED,
                                 offset=0.0, device=device)
    ids = obstacle_boundary_nodes(fx["mesh"], fx["oracle"])
    assert len(ids) > 0
    mask_ids = np.where(fx["obstacle_node_mask"])[0]
    # obstacle_node_mask indexes FREE nodes; the helper returns MESH node ids.
    free = fx["cons"].free_nodes
    assert set(ids.tolist()) == set(free[mask_ids].tolist())
