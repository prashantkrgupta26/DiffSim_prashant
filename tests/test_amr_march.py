"""A2 (C3 Phase-3 dynamic AMR): the driven solution-adaptive cycle.

A short Cahn--Hilliard march where refinement tracks a diffuse interface and
coarsening reclaims the bulk, remeshing every few steps. Gates: mass conserved
across every remesh to solver tolerance; the free-energy jump across a remesh
is only the (small) transfer projection error; refinement tracks the interface
(finest cells sit on the interface band); the march stays stable (no blow-up).
Needs the CH assembly kernel -> GPU (tier2).
"""
import numpy as np
import pytest

from diffsim.octree.build import build_uniform
from diffsim.adaptivity.amr_march import amr_march

pytestmark = pytest.mark.tier2

EPS = 0.03


def _c0(coords):
    r = np.sqrt(((coords - 0.5) ** 2).sum(1))
    return np.tanh((0.3 - r) / (np.sqrt(2.0) * EPS))


def test_amr_driven_cycle_conserves_and_tracks(device):
    out = amr_march(build_uniform(4, dim=2), _c0, M=1.0, kappa=EPS ** 2,
                    dt=2e-4, t_end=6e-3, device=device, order=1,
                    remesh_every=5, refine_frac=0.15, coarse_frac=0.03,
                    max_level=6, min_level=3)
    rec = out["rec"]
    dm = np.abs(np.array(rec["remesh_mass_after"])
                - np.array(rec["remesh_mass_before"]))
    dE = np.abs(np.array(rec["remesh_E_after"])
                - np.array(rec["remesh_E_before"]))
    assert len(dm) >= 3                       # several remeshes happened
    # mass conserved across every remesh to solver tolerance
    assert dm.max() < 1e-11
    # energy jump across remesh is small (transfer error, not a spurious jump);
    # the largest is the initial coarse->fine resolution gain
    assert dE.max() < 1e-3
    # refinement tracks the interface: finest cells sit on the interface band
    assert min(rec["overlap"]) > 0.8
    # stability: field stays bounded, finite, and the mesh doesn't explode
    c = out["final_c"]
    assert np.isfinite(c).all()
    assert c.min() > -1.2 and c.max() < 1.2
    assert max(rec["dofs"]) < 3000
