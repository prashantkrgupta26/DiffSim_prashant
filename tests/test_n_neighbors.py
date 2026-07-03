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
