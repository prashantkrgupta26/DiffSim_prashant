"""M1b Task 2 gate — the lego test (spec S3/S6.1): the M1a Poisson
re-expressed as a CEquation brick reproduces the framework assembly exactly
and the locked m1a baseline through the SBM solve path."""
import json
import os

import numpy as np
import pytest
import warp as wp
from diffsim.api.equation import (CEquation, assemble_brick_csr,
                                  brick_load_vector)
from diffsim.assembly.femelm import FEMElm, fe_N, fe_dN_s
from diffsim.assembly.operators import assemble_csr

pytestmark = pytest.mark.tier3


class PoissonBrick(CEquation):
    """-div(grad u) = f — the Hughes-form brick (spec S3.1 example)."""
    ndof = 1

    @staticmethod
    @wp.func
    def Integrands_Ae(fe: FEMElm,
                      Ntab: wp.array2d(dtype=wp.float64),
                      dNtab: wp.array3d(dtype=wp.float64),
                      detJxW: wp.float64, dscale: wp.float64,
                      nbf: wp.int32, dim: wp.int32, ndof: wp.int32,
                      Ae: wp.array3d(dtype=wp.float64), e: wp.int32):
        for a in range(nbf):
            for b in range(nbf):
                K = wp.float64(0.0)
                for k in range(dim):
                    K += fe_dN_s(dNtab, fe, a, k, dscale) \
                         * fe_dN_s(dNtab, fe, b, k, dscale)
                Ae[e, ndof * a, ndof * b] += K * detJxW

    @staticmethod
    @wp.func
    def Integrands_be(fe: FEMElm,
                      Ntab: wp.array2d(dtype=wp.float64),
                      dNtab: wp.array3d(dtype=wp.float64),
                      detJxW: wp.float64, dscale: wp.float64,
                      nbf: wp.int32, dim: wp.int32, ndof: wp.int32,
                      fq: wp.array(dtype=wp.float64), nqp: wp.int32,
                      be: wp.array2d(dtype=wp.float64), e: wp.int32):
        fv = fq[e * nqp + fe.q]
        for a in range(nbf):
            be[e, ndof * a] += fe_N(Ntab, fe, a) * fv * detJxW


def _dm(dim, p, device, mixed=False):
    from diffsim.octree.build import build_uniform, refine_elements
    from diffsim.octree.balance import balance2to1
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh
    t = build_uniform(3 if dim == 2 else 2, dim=dim)
    mask = np.zeros(len(t), bool)
    mask[0] = True
    t = balance2to1(refine_elements(t, mask))
    if mixed:
        # uniform level => any p split honors the one-knob rule
        t = build_uniform(3, dim=dim)
        p_elem = np.ones(len(t), np.int8)
        p_elem[t.anchors()[:, 0] == 0] = 2
        m = build_mesh(t, p=p_elem)
        tables = {pv: basis_tables(pv, dim=dim) for pv in m.bins}
    else:
        m = build_mesh(t, p=p)
        tables = basis_tables(p, dim=dim)
    c = build_constraints(m)
    return DeviceMesh.from_mesh(m, c, tables, device)


@pytest.mark.parametrize("dim,p", [(2, 1), (2, 2), (3, 1)])
def test_brick_matches_framework_assembly(dim, p, device):
    dm = _dm(dim, p, device)
    A_brick = assemble_brick_csr(dm, PoissonBrick)
    A_fw = assemble_csr(dm)
    diff = abs(A_brick - A_fw)
    assert diff.max() < 1e-13 * max(abs(A_fw).max(), 1.0)


def test_brick_load_matches_framework(device):
    dm = _dm(2, 1, device)
    f = lambda x: np.sin(np.pi * x[:, 0]) * x[:, 1]
    b_brick = brick_load_vector(dm, PoissonBrick, f)
    # framework path
    from diffsim.physics.poisson import gauss_points, make_load_kernel
    F_full = wp.zeros(dm.n_nodes, dtype=wp.float64, device=dm.device)
    xq = gauss_points(dm.mesh, dm.tables_by_p)
    for pv, b in dm.bins.items():
        fq = wp.array(f(xq[pv]), dtype=wp.float64, device=dm.device)
        lk = make_load_kernel(b["nbf"], b["nqp"], dm.dim)
        wp.launch(lk, dim=len(b["eids"]),
                  inputs=[b["conn"], b["h"], b["N"], b["w"], fq, F_full],
                  device=dm.device)
    b_fw = np.asarray(dm.constraints.T.T @ F_full.numpy())
    assert np.abs(b_brick - b_fw).max() < 1e-13 * max(np.abs(b_fw).max(), 1.0)


def test_brick_reproduces_m1a_baseline(device):
    """THE lego gate: the brick-assembled volume operator drives the M1a SBM
    solve and reproduces the locked disk_p1_l5 baseline at rtol 1e-6."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_sbm_poisson import sbm_setup, U2, F2
    from diffsim.geometry.csg import Sphere
    from diffsim.sbm.poisson import SBMPoisson
    from diffsim.physics.poisson import l2_error_masked
    from scipy.sparse.linalg import splu
    import scipy.sparse as sp

    oracle = Sphere((0.5, 0.5), 0.3)
    dm, geo, sf = sbm_setup(oracle, 5, 1, 0.0, 2, device)
    prob = SBMPoisson(dm, geo, sf, g_fn=U2, kappa=1.0)
    A_fw, rhs_fw, meta = prob.assemble(F2)
    # swap the volume part for the BRICK-assembled operator
    A_vol_brick = assemble_brick_csr(dm, PoissonBrick)
    A_vol_fw = assemble_csr(dm)
    A_swapped = (A_fw - A_vol_fw + A_vol_brick).tocsr()
    u = np.asarray(dm.constraints.T @ splu(A_swapped.tocsc()).solve(rhs_fw))
    err = l2_error_masked(dm, u, U2, lambda x: oracle.classify(x) < 0)
    with open(os.path.join(os.path.dirname(__file__), "baselines",
                           "m1a_baselines.json")) as fh:
        ref = json.load(fh)["disk_p1_l5"]
    assert abs(err - ref) < 1e-6 * ref, (err, ref)


def test_brick_mixed_p_bins(device):
    # per-bin dispatch through the brick factory (the M0.5 machinery)
    dm = _dm(2, None, device, mixed=True)
    A_brick = assemble_brick_csr(dm, PoissonBrick)
    A_fw = assemble_csr(dm)
    assert abs(A_brick - A_fw).max() < 1e-13
