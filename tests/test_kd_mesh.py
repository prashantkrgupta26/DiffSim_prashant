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
    # boundary flags exclude the periodic axis: nodes on x_extreme are not flagged as boundary
    # if they don't also touch a non-periodic axis extreme
    seam = m.node_icoords[:, 0] == 0
    interior_otherwise = np.all(
        (m.node_icoords[:, 1:] != 0) & (m.node_icoords[:, 1:] != m.node_icoords[:, 1:].max(axis=0)),
        axis=1
    )
    assert not np.any(m.boundary_nodes & seam & interior_otherwise)

def test_fully_periodic_2d():
    m = build_mesh(build_uniform(2, dim=2, periodic=(True, True)), p=1)
    assert len(m.node_coords) == 16          # (2^2)^2, torus
    assert not m.boundary_nodes.any()
